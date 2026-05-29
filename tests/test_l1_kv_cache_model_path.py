"""
Verify the live model path populates the L1 KV ring buffer correctly.

This script builds a real TT Transformer model, runs a short prefill plus a few
decode steps, then compares the hot window in `layer_past` (DRAM) against the
ring-buffered `l1_kv_cache` for the first attention layer.

Usage:
    source python_env/bin/activate
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct python tests/test_l1_kv_cache_model_path.py
"""

import argparse
import math

import torch
import ttnn

from models.tt_transformers.tt.common import create_tt_model
from models.tt_transformers.tt.model_config import DecodersPrecision


def corrcoef(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    mask = (a_f != 0) | (b_f != 0)
    if mask.sum() == 0:
        return 1.0
    return torch.corrcoef(torch.stack([a_f[mask], b_f[mask]]))[0, 1].item()


def compare_ring_window(dram_cache, l1_cache, end_pos, window_size):
    start = max(0, end_pos + 1 - window_size)
    expected = []
    got = []
    worst_diff = 0.0

    for global_pos in range(start, end_pos + 1):
        ring_pos = global_pos % window_size
        dram_slice = dram_cache[:, :, global_pos : global_pos + 1, :]
        l1_slice = l1_cache[:, :, ring_pos : ring_pos + 1, :]
        expected.append(dram_slice)
        got.append(l1_slice)
        worst_diff = max(worst_diff, (dram_slice.float() - l1_slice.float()).abs().max().item())

    expected = torch.cat(expected, dim=2)
    got = torch.cat(got, dim=2)
    return corrcoef(expected, got), worst_diff


def check_attention_layer(attn, prompt_length, final_pos, window_size, layer_idx):
    dram_k = ttnn.to_torch(attn.layer_past[0])
    dram_v = ttnn.to_torch(attn.layer_past[1])
    l1_k = ttnn.to_torch(attn.l1_kv_cache[0])
    l1_v = ttnn.to_torch(attn.l1_kv_cache[1])

    if prompt_length <= window_size:
        prefill_k_pcc = corrcoef(dram_k[:, :, :prompt_length, :], l1_k[:, :, :prompt_length, :])
        prefill_v_pcc = corrcoef(dram_v[:, :, :prompt_length, :], l1_v[:, :, :prompt_length, :])
        print(f"Layer {layer_idx} prefill K PCC = {prefill_k_pcc:.8f}")
        print(f"Layer {layer_idx} prefill V PCC = {prefill_v_pcc:.8f}")

    k_pcc, k_diff = compare_ring_window(dram_k, l1_k, final_pos, window_size)
    v_pcc, v_diff = compare_ring_window(dram_v, l1_v, final_pos, window_size)
    print(f"Layer {layer_idx} decode K ring PCC = {k_pcc:.8f}, worst_diff = {k_diff:.8f}")
    print(f"Layer {layer_idx} decode V ring PCC = {v_pcc:.8f}, worst_diff = {v_diff:.8f}")
    return k_pcc > 0.999 and v_pcc > 0.999 and k_diff < 1e-3 and v_diff < 1e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1-kv-window-size", type=int, default=256)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--check-all-layers", action="store_true")
    args = parser.parse_args()

    assert args.prompt_length % 128 == 0, "prompt-length must be divisible by 128 for prefill"

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        ttnn.device.DisablePersistentKernelCache()

        model_args, model, _, _ = create_tt_model(
            mesh_device=mesh_device,
            instruct=args.instruct,
            max_batch_size=1,
            optimizations=lambda cfg: DecodersPrecision.accuracy(cfg.n_layers, cfg.model_name),
            max_seq_len=max(args.prompt_length + args.decode_steps + 32, 512),
            paged_attention_config=None,
            dtype=ttnn.bfloat8_b,
            state_dict=None,
            num_layers=args.num_layers,
            use_adaptive_l1_kv_cache=True,
            l1_kv_window_size=args.l1_kv_window_size,
        )

        attn_layers = [layer.attention for layer in model.layers]
        if any(attn.l1_kv_cache is None for attn in attn_layers):
            raise RuntimeError("L1 KV cache was not allocated for one or more layers")

        tokenizer = model_args.tokenizer
        token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1

        prefill_tokens = torch.full((1, args.prompt_length), token_id, dtype=torch.int64)
        prefill_inputs = model.prepare_inputs_prefill(prefill_tokens, start_pos=0)
        tt_out = model.ttnn_prefill_forward(
            prefill_inputs[0],
            rot_mats_global=prefill_inputs[1],
            rot_mats_local=prefill_inputs[2],
            user_id=0,
            page_table=prefill_inputs[3],
            chunk_page_table=prefill_inputs[4],
        )
        ttnn.deallocate(tt_out)

        decode_token = torch.tensor([token_id], dtype=torch.int64)
        for step in range(args.decode_steps):
            current_pos = torch.tensor([args.prompt_length + step], dtype=torch.int64)
            decode_inputs = model.prepare_inputs_decode(decode_token, current_pos)
            tt_out = model.ttnn_decode_forward(*decode_inputs, kv_cache=None, sampling_on_device=False)
            ttnn.deallocate(tt_out)

        ttnn.synchronize_device(mesh_device)

        final_pos = args.prompt_length + args.decode_steps - 1
        layer_indices = range(len(attn_layers)) if args.check_all_layers else [0]
        passed = True
        for layer_idx in layer_indices:
            passed &= check_attention_layer(
                attn_layers[layer_idx], args.prompt_length, final_pos, args.l1_kv_window_size, layer_idx
            )
        print("PASS" if passed else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
