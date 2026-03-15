"""
Compare dual-source SDPA vs DRAM-only SDPA on the real model path.

This wraps `ttnn.transformer.scaled_dot_product_attention_decode` so each decode
layer computes both:
1. normal dual-source output using `l1_k_tensor` / `l1_v_tensor`
2. reference DRAM-only output with those arguments removed

The script prints PCC/max-diff per layer for a single decode step after prefill.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--l1-kv-window-size", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=36)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--instruct", action="store_true")
    args = parser.parse_args()

    assert args.prefill_seq_len % 128 == 0
    assert args.prompt_tokens <= args.prefill_seq_len

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    original_sdpa = ttnn.transformer.scaled_dot_product_attention_decode
    results = []
    call_idx = {"value": 0}

    def wrapped_sdpa(q_heads, keys, values, **kwargs):
        layer_idx = call_idx["value"]
        call_idx["value"] += 1

        out_l1 = original_sdpa(q_heads, keys, values, **kwargs)
        if "l1_k_tensor" not in kwargs or "l1_v_tensor" not in kwargs:
            return out_l1

        dram_kwargs = {k: v for k, v in kwargs.items() if k not in {"l1_k_tensor", "l1_v_tensor"}}
        out_dram = original_sdpa(q_heads, keys, values, **dram_kwargs)

        torch_l1 = ttnn.to_torch(out_l1)
        torch_dram = ttnn.to_torch(out_dram)
        pcc = corrcoef(torch_l1, torch_dram)
        max_diff = (torch_l1.float() - torch_dram.float()).abs().max().item()
        results.append((layer_idx, pcc, max_diff))
        print(f"Layer {layer_idx} SDPA dual-source vs DRAM: PCC={pcc:.8f}, max_diff={max_diff:.8f}")
        ttnn.deallocate(out_dram)
        return out_l1

    try:
        ttnn.device.DisablePersistentKernelCache()
        ttnn.transformer.scaled_dot_product_attention_decode = wrapped_sdpa

        model_args, model, _, _ = create_tt_model(
            mesh_device=mesh_device,
            instruct=args.instruct,
            max_batch_size=1,
            optimizations=lambda cfg: DecodersPrecision.accuracy(cfg.n_layers, cfg.model_name),
            max_seq_len=max(args.prefill_seq_len + 32, 512),
            paged_attention_config=None,
            dtype=ttnn.bfloat8_b,
            state_dict=None,
            num_layers=args.num_layers,
            use_l1_weight_sharding=False,
            l1_kv_window_size=args.l1_kv_window_size,
        )

        tokenizer = model_args.tokenizer
        tok = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tok

        prefill_tokens = torch.full((1, args.prefill_seq_len), pad_tok, dtype=torch.int64)
        prefill_tokens[:, : args.prompt_tokens] = tok

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

        current_pos = torch.tensor([args.prompt_tokens - 1], dtype=torch.int64)
        decode_token = torch.tensor([tok], dtype=torch.int64)
        decode_inputs = model.prepare_inputs_decode(decode_token, current_pos)
        tt_out = model.ttnn_decode_forward(*decode_inputs, kv_cache=None, sampling_on_device=False)
        ttnn.deallocate(tt_out)
        ttnn.synchronize_device(mesh_device)

        print(f"Compared {len(results)} decode-layer SDPA calls")
    finally:
        ttnn.transformer.scaled_dot_product_attention_decode = original_sdpa
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
