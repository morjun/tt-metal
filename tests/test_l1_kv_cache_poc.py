"""
PoC: Verify SDPA decode works with L1-resident KV cache.

This test creates a small KV cache in both DRAM (baseline) and L1,
runs scaled_dot_product_attention_decode on each, and compares outputs.
If the kernel works correctly with L1 KV, the outputs should match closely.

Usage:
    python tests/test_l1_kv_cache_poc.py
"""

import torch
import ttnn
import math


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    power = math.ceil(math.log2(x))
    return 1 << power


def get_chunk_size(max_start_pos, s):
    if max_start_pos <= 32:
        chunk_size = 32
    elif max_start_pos <= 64:
        chunk_size = 32
    elif max_start_pos <= 128:
        chunk_size = 32
    elif max_start_pos <= 1024:
        chunk_size = 128
    else:
        chunk_size = 512
    for i in range(1, s):
        if s % (2 ** (i + 1)) != 0:
            break
    chunk_size = min(chunk_size, 2**i)
    return chunk_size


def run_sdpa_decode(device, Q, K, V, start_indices, memory_config_kv, label, grid_size, padded_num_heads, k_chunk_size):
    """Run SDPA decode with given KV memory config and return torch result."""
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

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

    # Q stays in DRAM always
    tt_Q = ttnn.from_torch(
        Q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memcfg,
    )

    # K and V go to the specified memory config (DRAM or L1)
    tt_K = ttnn.from_torch(
        K,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config_kv,
    )

    tt_V = ttnn.from_torch(
        V,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config_kv,
    )

    print(f"\n--- {label} ---")
    print(f"  Q shape: {tt_Q.shape}, mem: {tt_Q.memory_config()}")
    print(f"  K shape: {tt_K.shape}, buffer_type: {tt_K.memory_config().buffer_type}")
    print(f"  V shape: {tt_V.shape}, buffer_type: {tt_V.memory_config().buffer_type}")

    # Run SDPA decode
    output = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos=start_indices,
        scale=Q.shape[-1] ** -0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram_memcfg,
    )

    ttnn.synchronize_device(device)
    result = ttnn.to_torch(output)

    ttnn.deallocate(tt_Q)
    ttnn.deallocate(tt_K)
    ttnn.deallocate(tt_V)
    ttnn.deallocate(output)

    print(f"  Output shape: {result.shape}")
    print(f"  Output mean: {result.float().mean():.6f}, std: {result.float().std():.6f}")

    return result


def main():
    # Config matching existing SDPA decode test patterns
    b = 1  # batch size
    nh = 8  # number of Q heads
    nkv = 1  # number of KV heads (GQA ratio = 8)
    s = 8192  # sequence length
    d = 128  # head dimension
    grid_size = (8, 4)  # compute grid (must fit in device; smaller = less CB pressure)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    max_start_idx = s // 2  # current position at halfway point
    start_indices = [max_start_idx] * b
    k_chunk_size = get_chunk_size(max_start_idx + 1, s)

    # Size estimation
    kv_bytes = b * nkv * s * d * 2  # bfloat16 = 2 bytes
    print(f"\nKV cache size per tensor: {kv_bytes / 1024:.1f} KB")
    print(f"Total KV (K+V): {2 * kv_bytes / 1024:.1f} KB")
    print(f"Padded num heads: {padded_num_heads}")
    print(f"k_chunk_size: {k_chunk_size}")

    # Generate test data
    torch.manual_seed(42)
    Q = torch.randn(1, b, padded_num_heads, d)  # [1, B, padded_nh, D]
    K = torch.randn(b, nkv, s, d)  # [B, nkv, S, D]
    V = torch.randn(b, nkv, s, d)  # [B, nkv, S, D]

    print(
        f"\nConfig: batch={b}, n_heads={nh}, n_kv_heads={nkv}, "
        f"head_dim={d}, seq_len={s}, padded_nh={padded_num_heads}"
    )
    print(f"start_indices: {start_indices}")
    print(f"grid_size: {grid_size}")

    # Open device
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    print(f"Device grid: {device.compute_with_storage_grid_size()}")
    ttnn.device.DisablePersistentKernelCache()

    # --- Run baseline: KV in DRAM ---
    output_dram = run_sdpa_decode(
        device,
        Q,
        K,
        V,
        start_indices=start_indices,
        memory_config_kv=ttnn.DRAM_MEMORY_CONFIG,
        label="BASELINE (KV in DRAM)",
        grid_size=grid_size,
        padded_num_heads=padded_num_heads,
        k_chunk_size=k_chunk_size,
    )

    # --- Run test: KV in L1 ---
    output_l1 = run_sdpa_decode(
        device,
        Q,
        K,
        V,
        start_indices=start_indices,
        memory_config_kv=ttnn.L1_MEMORY_CONFIG,
        label="TEST (KV in L1)",
        grid_size=grid_size,
        padded_num_heads=padded_num_heads,
        k_chunk_size=k_chunk_size,
    )

    # --- Compare outputs ---
    out_dram_f = output_dram.float()
    out_l1_f = output_l1.float()

    # Only compare the actual heads (not padded)
    out_dram_f = out_dram_f[:, :, :nh, :]
    out_l1_f = out_l1_f[:, :, :nh, :]

    # Compute PCC (Pearson Correlation Coefficient)
    d_flat = out_dram_f.flatten()
    l_flat = out_l1_f.flatten()

    # Filter out zeros for meaningful comparison
    mask = (d_flat != 0) | (l_flat != 0)
    if mask.sum() > 0:
        d_masked = d_flat[mask]
        l_masked = l_flat[mask]
        pcc = torch.corrcoef(torch.stack([d_masked, l_masked]))[0, 1].item()
    else:
        pcc = 1.0  # Both all-zeros

    max_abs_diff = (out_dram_f - out_l1_f).abs().max().item()
    mean_abs_diff = (out_dram_f - out_l1_f).abs().mean().item()

    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"  PCC:            {pcc:.8f}")
    print(f"  Max Abs Diff:   {max_abs_diff:.8f}")
    print(f"  Mean Abs Diff:  {mean_abs_diff:.8f}")

    if pcc > 0.999:
        print(f"\n  ✅ PASS: L1 KV cache outputs match DRAM baseline (PCC={pcc:.6f})")
    elif pcc > 0.99:
        print(f"\n  ⚠️  MARGINAL: PCC={pcc:.6f} — outputs similar but some divergence")
    else:
        print(f"\n  ❌ FAIL: PCC={pcc:.6f} — significant divergence!")

    # Cleanup
    ttnn.close_mesh_device(device)

    print("\nDone!")


if __name__ == "__main__":
    main()
