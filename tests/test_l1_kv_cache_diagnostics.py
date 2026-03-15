"""
Diagnostics for dual-source KV cache correctness.

This script separates the L1 SRAM path into three pieces:
1. SDPA reader path: direct ring-packed L1 tensors are read correctly.
2. Prefill-style writer path: fill_cache can populate an L1 KV tensor correctly.
3. Decode-style writer path: paged_update_cache into an L1 KV tensor is compared
   against the same update into a DRAM KV tensor.

Usage:
    source python_env/bin/activate
    python tests/test_l1_kv_cache_diagnostics.py
"""

import math
import sys

import torch
import ttnn


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


def corrcoef(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    mask = (a_f != 0) | (b_f != 0)
    if mask.sum() == 0:
        return 1.0
    return torch.corrcoef(torch.stack([a_f[mask], b_f[mask]]))[0, 1].item()


def run_reader_path_test():
    print("=" * 80)
    print("1. SDPA READER PATH TEST")
    print("=" * 80)

    b = 1
    nh = 8
    nkv = 1
    s = 2048
    d = 128
    l1_window = 256
    grid_size = (8, 4)

    padded_num_heads = nearest_pow_2(nearest_n(nh, 32))
    cur_pos = s // 2
    start_indices = [cur_pos] * b
    k_chunk_size = get_chunk_size(s)

    seq_tiles = (cur_pos + 1 + 31) // 32
    l1_window_size_tiles = l1_window // 32
    l1_start_tile = seq_tiles - l1_window_size_tiles if seq_tiles > l1_window_size_tiles else 0
    l1_start = l1_start_tile * 32

    torch.manual_seed(0)
    Q = torch.randn(1, b, padded_num_heads, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)

    K_l1_ring = torch.zeros(b, nkv, l1_window, d)
    V_l1_ring = torch.zeros(b, nkv, l1_window, d)

    for global_tile in range(l1_start_tile, l1_start_tile + l1_window_size_tiles):
        l1_row = global_tile % l1_window_size_tiles
        seq_start = (global_tile - l1_start_tile) * 32
        seq_end = seq_start + 32
        l1_start_idx = l1_row * 32
        l1_end_idx = l1_start_idx + 32
        K_l1_ring[:, :, l1_start_idx:l1_end_idx, :] = K[:, :, l1_start + seq_start : l1_start + seq_end, :]
        V_l1_ring[:, :, l1_start_idx:l1_end_idx, :] = V[:, :, l1_start + seq_start : l1_start + seq_end, :]

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
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

        tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out_dram = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            cur_pos=start_indices,
            scale=d**-0.5,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(device)
        out_dram_t = ttnn.to_torch(out_dram).float()[:, :, :nh, :]

        tt_Q2 = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_K2 = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tt_V2 = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tt_K_l1 = ttnn.from_torch(
            K_l1_ring, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tt_V_l1 = ttnn.from_torch(
            V_l1_ring, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out_dual = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_Q2,
            tt_K2,
            tt_V2,
            cur_pos=start_indices,
            scale=d**-0.5,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            l1_k_tensor=tt_K_l1,
            l1_v_tensor=tt_V_l1,
        )
        ttnn.synchronize_device(device)
        out_dual_t = ttnn.to_torch(out_dual).float()[:, :, :nh, :]
    finally:
        ttnn.close_mesh_device(device)

    pcc = corrcoef(out_dram_t, out_dual_t)
    max_diff = (out_dram_t - out_dual_t).abs().max().item()
    print(f"PCC(dram, dual_source) = {pcc:.8f}")
    print(f"max_abs_diff           = {max_diff:.8f}")

    passed = pcc > 0.99999 and max_diff == 0.0
    print("PASS" if passed else "FAIL")
    print()
    return passed


def run_fill_cache_test(dtype):
    print("=" * 80)
    print(f"2. FILL_CACHE -> L1 TEST ({dtype})")
    print("=" * 80)

    device = ttnn.open_device(device_id=0)
    try:
        seq_len = 64
        head_dim = 128
        max_seq_len = 256
        num_users = 1
        num_heads = 1

        cache = torch.zeros([num_users, num_heads, max_seq_len, head_dim]).bfloat16().float()
        cache_tt = ttnn.from_torch(
            cache, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        x = torch.randn([1, num_heads, seq_len, head_dim]).bfloat16().float()
        x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        cache_tt = ttnn.fill_cache(cache_tt, x_tt, 0)
        got = ttnn.to_torch(cache_tt)
    finally:
        ttnn.close_device(device)

    ref = cache.clone()
    ref[0:1, :, :seq_len, :] = x
    pcc = corrcoef(ref, got)
    max_diff = (ref - got.float()).abs().max().item()
    print(f"PCC(ref, l1_fill_cache) = {pcc:.8f}")
    print(f"max_abs_diff            = {max_diff:.8f}")

    passed = pcc > (0.999 if dtype == ttnn.bfloat16 else 0.9999)
    print("PASS" if passed else "FAIL")
    print()
    return passed


def run_single_user_ring_update_test(cache_dtype):
    print("=" * 80)
    print(f"3. SINGLE-USER PAGED_UPDATE_CACHE RING TEST ({cache_dtype})")
    print("=" * 80)

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        batch = 1
        n_local_heads = 8
        n_local_kv_heads = 1
        head_dim = 128
        total_heads = n_local_heads + 2 * n_local_kv_heads
        window = 256
        max_seq = 512

        dram_cache = ttnn.from_torch(
            torch.zeros(1, 1, max_seq, head_dim),
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        l1_cache = ttnn.from_torch(
            torch.zeros(1, 1, window, head_dim),
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Drive the exact model-shaped decode writer path across the wrap boundary.
        for pos in range(0, 320):
            proj_output = torch.zeros(1, 1, batch, head_dim * total_heads)
            k_start = head_dim * n_local_heads
            proj_output[0, 0, 0, k_start : k_start + head_dim] = float(pos + 1)

            proj_output_tt = ttnn.from_torch(proj_output, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
            proj_output_tt = proj_output_tt.to(device=device, mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
            _, k_heads, _ = ttnn.experimental.nlp_create_qkv_heads_decode(
                proj_output_tt,
                num_heads=n_local_heads,
                num_kv_heads=n_local_kv_heads,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )

            current_pos = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32),
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            orig_shape = current_pos.shape
            l1_pos = ttnn.to_layout(current_pos, ttnn.TILE_LAYOUT)
            l1_pos = ttnn.typecast(l1_pos, ttnn.float32)
            l1_pos = ttnn.remainder(l1_pos, float(window))
            l1_pos = ttnn.typecast(l1_pos, ttnn.int32)
            l1_pos = ttnn.to_layout(l1_pos, ttnn.ROW_MAJOR_LAYOUT)

            slice_starts = [0] * len(l1_pos.shape)
            slice_ends = list(l1_pos.shape)
            for i in range(len(orig_shape)):
                slice_ends[-(i + 1)] = orig_shape[-(i + 1)]
            l1_pos = ttnn.slice(l1_pos, slice_starts, slice_ends)

            k_heads_l1 = ttnn.mul(k_heads, 1.0)
            ttnn.experimental.paged_update_cache(dram_cache, k_heads, update_idxs_tensor=current_pos)
            ttnn.experimental.paged_update_cache(l1_cache, k_heads_l1, update_idxs_tensor=l1_pos)

        ttnn.synchronize_device(device)
        dram_t = ttnn.to_torch(dram_cache)
        l1_t = ttnn.to_torch(l1_cache)
    finally:
        ttnn.close_mesh_device(device)

    worst_diff = 0.0
    passed = True
    for pos in range(64, 320):
        ring = pos % window
        diff = (dram_t[0, 0, pos, :].float() - l1_t[0, 0, ring, :].float()).abs().max().item()
        worst_diff = max(worst_diff, diff)
        if diff > 0:
            passed = False
            print(f"Mismatch at pos={pos}, ring={ring}, diff={diff}")
            print(f"DRAM first8 = {dram_t[0, 0, pos, :8]}")
            print(f"L1   first8 = {l1_t[0, 0, ring, :8]}")
            break

    print(f"worst_ring_diff = {worst_diff:.8f}")
    print("PASS" if passed else "FAIL")
    print()
    return passed


def main():
    ok_reader = run_reader_path_test()
    ok_fill_bf16 = run_fill_cache_test(ttnn.bfloat16)
    ok_fill_bf8 = run_fill_cache_test(ttnn.bfloat8_b)
    ok_update_bf16 = run_single_user_ring_update_test(ttnn.bfloat16)
    ok_update_bf8 = run_single_user_ring_update_test(ttnn.bfloat8_b)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Reader path (SDPA consumes L1 ring correctly): {'PASS' if ok_reader else 'FAIL'}")
    print(f"Prefill-style writer (fill_cache -> L1, BF16): {'PASS' if ok_fill_bf16 else 'FAIL'}")
    print(f"Prefill-style writer (fill_cache -> L1, BF8):  {'PASS' if ok_fill_bf8 else 'FAIL'}")
    print(f"Decode-style writer (paged_update_cache -> L1, BF16): {'PASS' if ok_update_bf16 else 'FAIL'}")
    print(f"Decode-style writer (paged_update_cache -> L1, BF8):  {'PASS' if ok_update_bf8 else 'FAIL'}")

    sys.exit(0 if (ok_reader and ok_fill_bf16 and ok_fill_bf8 and ok_update_bf16 and ok_update_bf8) else 1)


if __name__ == "__main__":
    main()
