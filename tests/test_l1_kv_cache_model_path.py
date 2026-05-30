"""
Verify the live model path populates the interleaved L1 KV cache correctly.

Builds a real TT Transformer model in `l1_kv_mode="interleaved"`, runs prefill +
a few decode steps, and compares the decode-written positions in the L1 KV tier
(`attn.l1_kv_tiers[0]`) against the DRAM cache (`attn.layer_past`).

The adaptive/interleaved L1 KV cache is allocated POST-COMPILE (after CB regions
freeze), so this script mimics the generator: it runs one warmup decode step to
compile the decode programs, then allocates + seeds the tiers, then runs the
remaining decode steps (which write the L1 ring). The window is sized >= the
sequence so the cache is linear (no ring wrap): L1[pos] == DRAM[pos].

This is a standalone diagnostic. The canonical end-to-end validation is
models/tt_transformers/demo/simple_text_demo.py with --l1_kv_mode.

Usage:
    source python_env/bin/activate
    HF_MODEL=meta-llama/Llama-3.1-8B-Instruct python tests/test_l1_kv_cache_model_path.py
"""

import argparse

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


def compare_range(dram_cache, l1_cache, lo, hi):
    """Compare L1[pos] vs DRAM[pos] for pos in [lo, hi) (interleaved tier is linear)."""
    dram_slice = dram_cache[:, :, lo:hi, :]
    l1_slice = l1_cache[:, :, lo:hi, :]
    worst_diff = (dram_slice.float() - l1_slice.float()).abs().max().item()
    return corrcoef(dram_slice, l1_slice), worst_diff


def check_attention_layer(attn, decode_lo, decode_hi, layer_idx):
    dram_k = ttnn.to_torch(attn.layer_past[0])
    dram_v = ttnn.to_torch(attn.layer_past[1])
    # Interleaved mode stores the whole cache in a single tier; tier tuple is
    # (k_tensor, v_tensor, token_start, tok_count).
    l1_k = ttnn.to_torch(attn.l1_kv_tiers[0][0])
    l1_v = ttnn.to_torch(attn.l1_kv_tiers[0][1])

    k_pcc, k_diff = compare_range(dram_k, l1_k, decode_lo, decode_hi)
    v_pcc, v_diff = compare_range(dram_v, l1_v, decode_lo, decode_hi)
    print(f"Layer {layer_idx} decode K PCC = {k_pcc:.8f}, worst_diff = {k_diff:.8f}")
    print(f"Layer {layer_idx} decode V PCC = {v_pcc:.8f}, worst_diff = {v_diff:.8f}")
    return k_pcc > 0.999 and v_pcc > 0.999 and k_diff < 1e-3 and v_diff < 1e-3


def _allocate_and_seed_tiers(attn_layers):
    for attn in attn_layers:
        # Interleaved mode ignores the headroom map (sizes from window+sink), so {} is fine.
        attn.allocate_l1_kv_cache({})
        attn.seed_adaptive_l1_sinks(attn.layer_past[0], attn.layer_past[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1-kv-window-size", type=int, default=768)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--check-all-layers", action="store_true")
    args = parser.parse_args()

    assert args.prompt_length % 128 == 0, "prompt-length must be divisible by 128 for prefill"
    assert args.decode_steps >= 2, "need >=2 decode steps (1 warmup + >=1 measured)"

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
            l1_kv_mode="interleaved",
            l1_kv_window_size=args.l1_kv_window_size,
        )

        attn_layers = [layer.attention for layer in model.layers]

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
            # Mimic the generator: allocate + seed the L1 tiers after the first decode
            # step compiles the decode programs (CB regions now frozen).
            if step == 0:
                _allocate_and_seed_tiers(attn_layers)
                if any(not attn.l1_kv_tiers for attn in attn_layers):
                    raise RuntimeError("L1 KV tiers were not allocated for one or more layers")

        ttnn.synchronize_device(mesh_device)

        # Decode positions written to the L1 ring after allocation: [prompt+1, final+1).
        decode_lo = args.prompt_length + 1
        decode_hi = args.prompt_length + args.decode_steps
        layer_indices = range(len(attn_layers)) if args.check_all_layers else [0]
        passed = True
        for layer_idx in layer_indices:
            passed &= check_attention_layer(attn_layers[layer_idx], decode_lo, decode_hi, layer_idx)
        print("PASS" if passed else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
