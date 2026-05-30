# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import parse_optimizations


# These inputs override the default inputs used by simple_text_demo.py. Check the main demo to see the default values.
def pytest_addoption(parser):
    parser.addoption("--input_prompts", action="store", help="input prompts json file")
    parser.addoption("--instruct", action="store", type=int, help="Use instruct weights")
    parser.addoption("--repeat_batches", action="store", type=int, help="Number of consecutive batches of users to run")
    parser.addoption("--max_seq_len", action="store", type=int, help="Maximum context length supported by the model")
    parser.addoption("--batch_size", action="store", type=int, help="Number of users in a batch ")
    parser.addoption(
        "--max_generated_tokens", action="store", type=int, help="Maximum number of tokens to generate for each user"
    )
    parser.addoption("--data_parallel", action="store", type=int, help="Number of data parallel workers")
    parser.addoption(
        "--paged_attention",
        action="store",
        type=int,
        default=None,
        help="Whether to use paged attention (1) or default attention (0)",
    )
    parser.addoption("--page_params", action="store", type=dict, help="Page parameters for paged attention")
    parser.addoption("--sampling_params", action="store", type=dict, help="Sampling parameters for decoding")
    parser.addoption(
        "--stop_at_eos", action="store", type=int, help="Whether to stop decoding when the model generates an EoS token"
    )
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )
    parser.addoption(
        "--decoder_config_file",
        action="store",
        default=None,
        type=str,
        help="Provide a JSON file defining per-decoder precision and fidelity settings",
    )
    parser.addoption(
        "--token_accuracy",
        action="store",
        default=False,
        type=bool,
        help="Whether to compute top1 and top5 exact token matching accuracy",
    )
    parser.addoption(
        "--stress_test",
        action="store",
        default=False,
        type=bool,
        help="Run stress test (same decode iteration over a large number of iterations",
    )
    parser.addoption(
        "--enable_trace",
        action="store",
        default=None,
        type=bool,
        help="Whether to enable tracing",
    )
    parser.addoption(
        "--num_layers",
        action="store",
        default=None,
        type=int,
        help="Number of layers to use",
    )
    parser.addoption(
        "--mode",
        action="store",
        default="full",
        type=str,
        help="Mode to use for full model demo tests (values can be 'prefill','decode','full')",
    )
    parser.addoption(
        "--l1_kv_window_size",
        action="store",
        default=0,
        type=int,
        help="L1 KV cache window size (0 = disabled, e.g. 256 for ring-buffer cache)",
    )
    parser.addoption(
        "--l1_kv_sink_size",
        action="store",
        default=0,
        type=int,
        help="Pinned L1 KV sink size in tokens (0 = disabled)",
    )
    parser.addoption(
        "--l1_kv_min_expected_hit_ratio",
        action="store",
        default=0.0,
        type=float,
        help="Disable dual-source L1 reads for decode steps whose expected hit ratio is below this threshold",
    )
    parser.addoption(
        "--l1_kv_safety_margin",
        action="store",
        default=64 * 1024,
        type=int,
        help="Bytes reserved between CB top and KV bottom when sizing the adaptive L1 KV cache (default: 65536)",
    )
    parser.addoption(
        "--l1_kv_min_viable_tokens",
        action="store",
        default=64,
        type=int,
        help="Skip cores that cannot fit at least this many KV tokens after applying the safety margin (default: 64)",
    )
    parser.addoption(
        "--l1_memory_view_path",
        action="store",
        default=None,
        type=str,
        help="Optional JSON output path for summarized L1 memory-view snapshots from simple_text_demo",
    )
    parser.addoption(
        "--l1_kv_mode",
        action="store",
        default="dram",
        choices=["dram", "interleaved", "sharded", "hybrid"],
        help=(
            "L1 KV cache layout mode (default: dram = no L1 KV cache):\n"
            "  dram        - KV cache in DRAM only (baseline; no L1 KV).\n"
            "  interleaved - single interleaved L1 buffer across all banks, sized by "
            "--l1_kv_window_size (+ sink). Highest capacity, parallel reads.\n"
            "  sharded     - N HEIGHT_SHARDED tiers auto-sized from --l1_kv_headroom_json.\n"
            "  hybrid      - sharded tiers + an interleaved tier (not yet implemented).\n"
            "Combine with --l1_kv_only_mode for StreamingLLM L1-only decode."
        ),
    )
    parser.addoption(
        "--l1_kv_only_mode",
        action="store_true",
        default=False,
        help=(
            "StreamingLLM-style L1-only inference: SDPA decode attends only to L1-cached positions, "
            "skipping DRAM reads/writes. Applies to interleaved/sharded/hybrid modes. With capacity >= "
            "sequence the full prompt is seeded into L1 so output matches the DRAM baseline."
        ),
    )
    parser.addoption(
        "--l1_kv_headroom_json",
        action="store",
        default=None,
        type=str,
        help=(
            "Path to an offline headroom-map JSON (produced by TT_METAL_LOG_L1_CB_MAP profiling). "
            "When provided, gap_bytes_free_headroom per core is used directly instead of the live scan. "
            "This is the most accurate source as it captures both CB and mid-step transient top-down buffers."
        ),
    )
