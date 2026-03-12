"""
Test: Verify SDPA decode dual-source KV reading (L1 + DRAM).

This test creates:
  - Full KV cache in DRAM (baseline)
  - Same full KV cache in DRAM + a partial L1 copy of the recent window
Then calls scaled_dot_product_attention_decode with the new l1_k_tensor/l1_v_tensor
kwargs and compares outputs against the DRAM-only baseline.

Usage:
    python tests/test_dual_source_kv.py
"""

import torch
import ttnn
import math


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    return 1 << math.ceil(math.log2(x))


def get_chunk_size(s):
    i = 1
    for i in range(1, s):
        if s % (2 ** (i + 1)) != 0:
            break
    return min(512, 2**i)


def main():
    # --- Config ---
    b = 1  # batch size
    nh = 8  # number of Q heads
    nkv = 1  # number of KV heads (GQA ratio = 8)
    s = 2048  # full sequence length
    d = 128  # head dimension
    l1_window = 256  # number of recent tokens to cache in L1
    grid_size = (8, 4)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    cur_pos = s // 2  # current decode position (halfway through cache)
    start_indices = [cur_pos] * b
    k_chunk_size = get_chunk_size(s)

    # C++ dataflow computes L1 tile bounds strictly rounded to 32
    seq_tiles = (cur_pos + 1 + 31) // 32
    l1_window_size_tiles = l1_window // 32
    cur_l1_window_start_tile = seq_tiles - l1_window_size_tiles if seq_tiles > l1_window_size_tiles else 0
    l1_start = cur_l1_window_start_tile * 32
    l1_end = l1_start + (l1_window_size_tiles * 32)

    print(f"Config: batch={b}, nh={nh}, nkv={nkv}, d={d}, s={s}")
    print(f"cur_pos={cur_pos}, seq_tiles={seq_tiles}, L1 window=[{l1_start}, {l1_end})")
    print(f"padded_nh={padded_num_heads}, k_chunk_size={k_chunk_size}, grid={grid_size}")
    print(f"L1 window size: {l1_window} tokens = {l1_window_size_tiles} tile rows")
    # --- Generate test data ---
    torch.manual_seed(42)
    Q = torch.randn(1, b, padded_num_heads, d)  # [1, B, padded_nh, D]
    K = torch.randn(b, nkv, s, d)  # [B, nkv, S, D]
    V = torch.randn(b, nkv, s, d)  # [B, nkv, S, D]

    # Extract the L1 window slice
    K_l1 = K[:, :, l1_start:l1_end, :].contiguous()
    V_l1 = V[:, :, l1_start:l1_end, :].contiguous()

    # Poison the L1 cache
    # K_l1 = torch.zeros_like(K[:, :, l1_start:l1_end, :]).contiguous()
    # V_l1 = torch.zeros_like(V[:, :, l1_start:l1_end, :]).contiguous()

    print(f"K_l1 shape: {K_l1.shape}, V_l1 shape: {V_l1.shape}")

    # --- Open device ---
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    print(f"Device grid: {device.compute_with_storage_grid_size()}")
    ttnn.device.DisablePersistentKernelCache()

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG
    l1_memcfg = ttnn.L1_MEMORY_CONFIG

    # ======= Run 1: Baseline (all DRAM, no L1 KV) =======
    print("\n--- BASELINE (all DRAM) ---")
    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)

    output_baseline = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos=start_indices,
        scale=d**-0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram_memcfg,
    )
    ttnn.synchronize_device(device)
    result_baseline = ttnn.to_torch(output_baseline)
    print(f"  Output shape: {result_baseline.shape}")
    print(f"  Output mean: {result_baseline.float().mean():.6f}")

    ttnn.deallocate(tt_Q)
    ttnn.deallocate(tt_K)
    ttnn.deallocate(tt_V)
    ttnn.deallocate(output_baseline)

    # ======= Run 2: Dual-source (DRAM + L1 window) =======
    print("\n--- DUAL-SOURCE (DRAM + L1 window) ---")
    tt_Q2 = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)
    tt_K2 = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)
    tt_V2 = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)

    # L1 window tensors
    tt_K_l1 = ttnn.from_torch(
        K_l1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_memcfg
    )
    tt_V_l1 = ttnn.from_torch(
        V_l1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_memcfg
    )

    print(f"  K_l1 buffer_type: {tt_K_l1.memory_config().buffer_type}")
    print(f"  V_l1 buffer_type: {tt_V_l1.memory_config().buffer_type}")

    output_dual = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_Q2,
        tt_K2,
        tt_V2,
        cur_pos=start_indices,
        scale=d**-0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram_memcfg,
        l1_k_tensor=tt_K_l1,
        l1_v_tensor=tt_V_l1,
    )
    ttnn.synchronize_device(device)
    result_dual = ttnn.to_torch(output_dual)
    print(f"  Output shape: {result_dual.shape}")
    print(f"  Output mean: {result_dual.float().mean():.6f}")

    ttnn.deallocate(tt_Q2)
    ttnn.deallocate(tt_K2)
    ttnn.deallocate(tt_V2)
    ttnn.deallocate(tt_K_l1)
    ttnn.deallocate(tt_V_l1)
    ttnn.deallocate(output_dual)

    # ======= Compare =======
    out_base_f = result_baseline.float()[:, :, :nh, :]
    out_dual_f = result_dual.float()[:, :, :nh, :]

    d_flat = out_base_f.flatten()
    l_flat = out_dual_f.flatten()

    mask = (d_flat != 0) | (l_flat != 0)
    if mask.sum() > 0:
        pcc = torch.corrcoef(torch.stack([d_flat[mask], l_flat[mask]]))[0, 1].item()
    else:
        pcc = 1.0

    max_diff = (out_base_f - out_dual_f).abs().max().item()
    mean_diff = (out_base_f - out_dual_f).abs().mean().item()

    print(f"\n{'='*60}")
    print(f"DUAL-SOURCE COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"  PCC:            {pcc:.8f}")
    print(f"  Max Abs Diff:   {max_diff:.8f}")
    print(f"  Mean Abs Diff:  {mean_diff:.8f}")

    if pcc > 0.999:
        print(f"\n  ✅ PASS: Dual-source output matches baseline (PCC={pcc:.6f})")
    elif pcc > 0.99:
        print(f"\n  ⚠️  MARGINAL: PCC={pcc:.6f}")
    else:
        print(f"\n  ❌ FAIL: PCC={pcc:.6f}")

    ttnn.close_mesh_device(device)
    print("\nDone!")


if __name__ == "__main__":
    main()
