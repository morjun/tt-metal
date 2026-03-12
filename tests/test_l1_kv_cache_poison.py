"""
Poison Test: Definitively prove that SDPA decode reads KV tiles from L1.

Strategy:
  Run 1 (BASELINE): DRAM KV cache with values A → output_A
  Run 2 (POISON):   DRAM KV cache with values A, but L1 KV has DIFFERENT values B
                     for the hot region → output_P

If L1 is truly read:
  - output_P != output_A  (because the kernel read B from L1 instead of A from DRAM)

If L1 is NOT read (kernel ignores L1):
  - output_P == output_A  (because both runs read A from DRAM)

We also do Run 3 with a DRAM cache that has B in the hot region (no L1) to confirm
that the poison output matches what we'd get by reading B directly from DRAM.

Usage:
    source python_env/bin/activate
    python tests/test_l1_kv_cache_poison.py
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


def run_sdpa(device, Q, K, V, start_indices, program_config, compute_kernel_config, K_l1=None, V_l1=None):
    """Run SDPA decode and return result as a torch tensor."""
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG
    l1_memcfg = ttnn.L1_MEMORY_CONFIG

    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_memcfg)

    kwargs = dict(
        cur_pos=start_indices,
        scale=K.shape[-1] ** -0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram_memcfg,
    )

    if K_l1 is not None and V_l1 is not None:
        tt_K_l1 = ttnn.from_torch(
            K_l1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_memcfg
        )
        tt_V_l1 = ttnn.from_torch(
            V_l1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_memcfg
        )
        kwargs["l1_k_tensor"] = tt_K_l1
        kwargs["l1_v_tensor"] = tt_V_l1

    output = ttnn.transformer.scaled_dot_product_attention_decode(tt_Q, tt_K, tt_V, **kwargs)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(output)

    # Cleanup
    ttnn.deallocate(tt_Q)
    ttnn.deallocate(tt_K)
    ttnn.deallocate(tt_V)
    ttnn.deallocate(output)
    if K_l1 is not None:
        ttnn.deallocate(tt_K_l1)
        ttnn.deallocate(tt_V_l1)

    return result


def pcc(a, b):
    """Pearson correlation coefficient between two flat tensors."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    mask = (a_f != 0) | (b_f != 0)
    if mask.sum() == 0:
        return 1.0
    return torch.corrcoef(torch.stack([a_f[mask], b_f[mask]]))[0, 1].item()


def main():
    # --- Config ---
    b = 1  # batch size
    nh = 8  # Q heads
    nkv = 1  # KV heads (GQA)
    s = 2048  # full sequence length
    d = 128  # head dimension
    l1_window = 256  # L1 window size in tokens
    grid_size = (8, 4)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    cur_pos = s // 2  # position 1024 — well past the L1 window
    start_indices = [cur_pos] * b
    k_chunk_size = get_chunk_size(s)

    # Compute L1 window bounds (matching kernel logic)
    seq_tiles = (cur_pos + 1 + 31) // 32
    l1_window_size_tiles = l1_window // 32
    l1_start_tile = seq_tiles - l1_window_size_tiles if seq_tiles > l1_window_size_tiles else 0
    l1_start = l1_start_tile * 32
    l1_end = l1_start + l1_window

    print("=" * 70)
    print("L1 KV CACHE POISON TEST")
    print("=" * 70)
    print(f"Config: batch={b}, Q_heads={nh}, KV_heads={nkv}, dim={d}, seq_len={s}")
    print(f"cur_pos={cur_pos}, L1 window: tokens [{l1_start}, {l1_end})")
    print(f"L1 window: {l1_window} tokens = {l1_window_size_tiles} tile rows")
    print()

    # --- Generate test data ---
    torch.manual_seed(42)
    Q = torch.randn(1, b, padded_num_heads, d)
    K_original = torch.randn(b, nkv, s, d)
    V_original = torch.randn(b, nkv, s, d)

    # Create POISONED data: use completely different random values
    torch.manual_seed(999)  # Different seed!
    K_poison_seq = torch.randn(b, nkv, l1_window, d)
    V_poison_seq = torch.randn(b, nkv, l1_window, d)

    # For RUN 3 (Ground truth): paste sequentially into DRAM hot region
    K_with_poison = K_original.clone()
    V_with_poison = V_original.clone()
    K_with_poison[:, :, l1_start:l1_end, :] = K_poison_seq
    V_with_poison[:, :, l1_start:l1_end, :] = V_poison_seq

    # For RUN 2 (L1 Poison): pack the sequential poison data into a RING BUFFER layout!
    # This accurately simulates `attention.py` using `paged_update_cache` with modulo math.
    K_l1_ring = torch.zeros(b, nkv, l1_window, d)
    V_l1_ring = torch.zeros(b, nkv, l1_window, d)

    for global_tile in range(l1_start_tile, l1_start_tile + l1_window_size_tiles):
        l1_row = global_tile % l1_window_size_tiles

        seq_start = (global_tile - l1_start_tile) * 32
        seq_end = seq_start + 32

        l1_start_idx = l1_row * 32
        l1_end_idx = l1_start_idx + 32

        K_l1_ring[:, :, l1_start_idx:l1_end_idx, :] = K_poison_seq[:, :, seq_start:seq_end, :]
        V_l1_ring[:, :, l1_start_idx:l1_end_idx, :] = V_poison_seq[:, :, seq_start:seq_end, :]

    print(f"Original K hot region mean:  {K_original[:, :, l1_start:l1_end, :].mean():.4f}")
    print(f"Poisoned L1 K mean:          {K_poison_seq.mean():.4f}")
    print(f"These are DIFFERENT values — if L1 is read, output will differ from baseline.")
    print()

    # --- Open device ---
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
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
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # ===== RUN 1: BASELINE (DRAM only, original data) =====
    print("--- RUN 1: BASELINE (DRAM only, original data) ---")
    result_baseline = run_sdpa(device, Q, K_original, V_original, start_indices, program_config, compute_kernel_config)
    out_baseline = result_baseline.float()[:, :, :nh, :]
    print(f"  Output mean: {out_baseline.mean():.6f}")

    # ===== RUN 2: POISON (DRAM has original, L1 has DIFFERENT values) =====
    print("\n--- RUN 2: POISON (DRAM=original, L1=poisoned) ---")
    result_poison = run_sdpa(
        device,
        Q,
        K_original,
        V_original,
        start_indices,
        program_config,
        compute_kernel_config,
        K_l1=K_l1_ring,
        V_l1=V_l1_ring,
    )
    out_poison = result_poison.float()[:, :, :nh, :]
    print(f"  Output mean: {out_poison.mean():.6f}")

    # ===== RUN 3: GROUND TRUTH (DRAM has poison values baked into hot region, no L1) =====
    print("\n--- RUN 3: GROUND TRUTH (DRAM with poison values in hot region, no L1) ---")
    result_ground_truth = run_sdpa(
        device, Q, K_with_poison, V_with_poison, start_indices, program_config, compute_kernel_config
    )
    out_ground_truth = result_ground_truth.float()[:, :, :nh, :]
    print(f"  Output mean: {out_ground_truth.mean():.6f}")

    # ===== ANALYSIS =====
    pcc_baseline_vs_poison = pcc(out_baseline, out_poison)
    pcc_poison_vs_groundtruth = pcc(out_poison, out_ground_truth)
    pcc_baseline_vs_groundtruth = pcc(out_baseline, out_ground_truth)

    max_diff_baseline_poison = (out_baseline - out_poison).abs().max().item()
    max_diff_poison_gt = (out_poison - out_ground_truth).abs().max().item()

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("PCC(baseline, poison):        {:.8f}  ← Should be LOW if L1 is read".format(pcc_baseline_vs_poison))
    print("PCC(poison, ground_truth):    {:.8f}  ← Should be HIGH if L1 is read".format(pcc_poison_vs_groundtruth))
    print("PCC(baseline, ground_truth):  {:.8f}  ← Should be LOW (different data)".format(pcc_baseline_vs_groundtruth))
    print()
    print("Max diff (baseline vs poison):      {:.6f}".format(max_diff_baseline_poison))
    print("Max diff (poison vs ground_truth):  {:.6f}".format(max_diff_poison_gt))
    print()

    # Verdict
    l1_is_used = (pcc_baseline_vs_poison < 0.99) and (pcc_poison_vs_groundtruth > 0.99)
    l1_is_ignored = pcc_baseline_vs_poison > 0.99

    if l1_is_used:
        print("✅ PROOF: L1 KV cache IS being read!")
        print("   The poison output differs from baseline (DRAM-only),")
        print("   and matches the ground truth (DRAM with poison values in hot region).")
        print("   This means the kernel read from L1 for the hot tiles.")
    elif l1_is_ignored:
        print("❌ DISPROOF: L1 KV cache is NOT being read!")
        print("   The poison output matches baseline exactly,")
        print("   meaning the kernel ignored L1 and read everything from DRAM.")
    else:
        print("⚠️  INCONCLUSIVE: Results are ambiguous.")
        print("   Consider checking kernel compile flags and compute config.")

    print()
    ttnn.close_mesh_device(device)
    print("Done!")


if __name__ == "__main__":
    main()
