# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Gemma4 it-assistant drafter: DRAM weights vs L1/SRAM weights.

The drafter is normally the *draft* half of speculative decoding and cannot run
on its own — ``Gemma4AssistantModel`` borrows three things from the target
(``tt/assistant/model.py``):

    self.target                    -> only for embed_tokens()   (model.py:197)
    target_model.rope_caches_2d    -> per-layer-type RoPE        (model.py:107)
    target.get_shared_kv_caches()  -> the KV it cross-attends to (spec_decode.py)

:class:`_TargetStub` supplies exactly those three and nothing else, so the
drafter runs alone on the mesh with no 10 GB target resident. The KV it attends
to is fabricated with ``init_kv_cache`` + ``paged_fill_cache``, the same way
``test_spec_decode.py::test_assistant_step_pcc_vs_hf`` fabricates it.

What the A/B measures
---------------------
``GEMMA4_WEIGHTS_IN_L1`` / the ``placement`` param flips every drafter weight
from DRAM to L1 subject to a per-device byte budget
(``models/demos/gemma4/tt/weight_placement.py``). At E2B/tp=2 that pins ~14 MB
per device — the 4 decoder layers, both projections and the CME centroid/ordering
tables. The 128 MiB replicated ``embed_table`` does not fit and is reported as a
DRAM fallback rather than silently excluded.

Expect a small effect. ~14 MB per device per step at ~440 GB/s is ~32 us against
a step measured in milliseconds: the drafter is dispatch/CCL-bound, not
weight-bandwidth-bound. The point of the per-op numbers (run this under tracy)
is to show *whether the matmuls themselves moved* even when the step does not.

Embedding table
---------------
By default the stub's [V, backbone] token-embedding table is deterministic
random rather than the target checkpoint's, so this harness needs only the
~158 MB assistant checkpoint and no target download. Embedding-lookup cost is
shape-dependent, not value-dependent, and both arms get the identical table, so
neither the perf A/B nor the DRAM-vs-L1 parity check is affected. Set
``GEMMA4_STANDALONE_REAL_EMBED=1`` to load the real table from ``HF_MODEL``.

Usage:
    export GEMMA4_ASSISTANT_MODEL=google/gemma-4-E2B-it-assistant
    pytest models/demos/gemma4/tests/unit/test_assistant_standalone_l1.py -s -k 1x2
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.assistant.masked_embedding import CmeLogits, Gemma4TTMaskedEmbedder, _same_buffer
from models.demos.gemma4.tt.assistant.model import Gemma4AssistantModel
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allgather
from models.demos.gemma4.tt.matmul_tuning import DecodeMatmulTuner
from models.demos.gemma4.tt.model import create_rope_caches
from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs
from models.demos.gemma4.tt.weight_placement import WeightPlacement
from models.tt_transformers.tt.common import PagedAttentionConfig

from ...tests.test_factory import parametrize_mesh_with_fabric

# The KV-to-mesh layout helper is shared with the spec-decode tests rather than
# duplicated: it encodes the production K_proj sharding (GQA-replicated when
# num_kv_heads < tp, which is exactly the E2B drafter's 1-kv-head case).
from .test_spec_decode import _kv_to_tt

ASSISTANT_PATH = os.getenv("GEMMA4_ASSISTANT_MODEL")
_needs_assistant = pytest.mark.skipif(not ASSISTANT_PATH, reason="set GEMMA4_ASSISTANT_MODEL to run")

BLOCK_SIZE = 64
DEFAULT_CONTEXT = 512
TRACE_REPS = 50


class _TargetStub:
    """The minimal surface ``Gemma4AssistantModel`` needs from a target model.

    Deliberately not a Gemma4Model: constructing the real 35-layer E2B target
    would put 10 GB in DRAM and make the drafter-only measurement depend on the
    target's allocator state.
    """

    def __init__(self, mesh_device, mesh_config, ccl_manager, text_args, backbone_hidden_size, max_seq_len, seed=0):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = backbone_hidden_size
        # create_assistant_model refuses a target with bounded sliding caches;
        # the drafter reads absolute cache positions.
        self.bounded_sliding_kv_cache = False
        self.embed_scale = backbone_hidden_size**0.5

        # Same RoPE the real target builds: head_dim and theta are identical
        # between an assistant and its target (verified for E2B: 256 sliding /
        # 512 global), which is why the drafter shares the target's caches.
        # create_rope_caches needs the real HF text config, which
        # Gemma4AssistantArgs.from_hf_config stashes on text_args for this.
        _, self.rope_caches_2d = create_rope_caches(mesh_device, text_args._hf_text_config, max_seq_len)

        tp = mesh_config.tp if mesh_config else 1
        vocab = text_args.vocab_size
        table = self._embedding_table(vocab, backbone_hidden_size, seed)
        # Column-parallel on hidden, all-gathered after lookup — mirrors
        # Gemma4Model.__init__ (tt/model.py:330-348).
        mapper = mesh_config.column_parallel(mesh_device) if tp > 1 else ttnn.ReplicateTensorToMesh(mesh_device)
        self.embedding_weight = ttnn.as_tensor(
            table.unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _embedding_table(vocab, hidden, seed):
        if os.getenv("GEMMA4_STANDALONE_REAL_EMBED", "0") == "1":
            table = _load_target_embedding(vocab, hidden)
            if table is not None:
                logger.info(f"[stub] loaded REAL target embedding {tuple(table.shape)}")
                return table
            logger.warning("[stub] GEMMA4_STANDALONE_REAL_EMBED=1 but the table could not be loaded; using random")
        g = torch.Generator().manual_seed(seed)
        logger.info(f"[stub] synthetic token embedding [{vocab}, {hidden}] (seed={seed})")
        return (torch.randn(vocab, hidden, generator=g) * 0.02).to(torch.bfloat16)

    def embed_tokens(self, tokens):
        """Scaled token embedding — mirrors Gemma4Model.embed_tokens (model.py:1224)."""
        embeds = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        embeds = ttnn.mul(embeds, self.embed_scale)
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            embeds = ttnn.unsqueeze_to_4D(embeds)
            embeds = ccl_allgather(embeds, self.mesh_config, self.ccl_manager)
        return embeds


def _load_target_embedding(vocab, hidden):
    """Read ONLY model.embed_tokens.weight from the target checkpoint."""
    model_path = os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH")
    if not model_path:
        return None
    try:
        import glob

        from safetensors import safe_open

        root = model_path
        if not os.path.isdir(root):
            hits = glob.glob(
                os.path.join(
                    os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                    "hub",
                    "models--" + model_path.replace("/", "--"),
                    "snapshots",
                    "*",
                )
            )
            if not hits:
                return None
            root = hits[0]
        for path in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
            with safe_open(path, framework="pt") as f:
                for key in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"):
                    if key in f.keys():
                        return f.get_tensor(key).to(torch.bfloat16)
    except Exception as e:  # noqa: BLE001 - falls back to the synthetic table
        logger.warning(f"[stub] real embedding load failed: {e}")
    return None


def _dev0(t, mesh_device):
    """Read a (possibly mesh-replicated) TT tensor from device 0 to torch."""
    n = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]) if n > 1 else ttnn.to_torch(t)


def _pcc(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _mesh_bits(mesh_device):
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    tp = mesh_device.shape[1] if num_devices > 1 else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    return num_devices, tp, mesh_config


def _fabricate_shared_kv(mesh_device, text_args, tp, num_devices, mapper, context_len, seed=0):
    """Fresh per-layer-type KV caches filled with known random K/V.

    Mirrors test_spec_decode.py::test_assistant_step_pcc_vs_hf — the drafter
    only reads these, so their contents just need to be identical across arms.
    """
    pac = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=math.ceil(context_len / BLOCK_SIZE))
    page_table = torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(1, pac.max_num_blocks)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=mapper
    )

    type_to_idx = {}
    for i, lt in enumerate(text_args.layer_types):
        type_to_idx.setdefault(lt, i)

    torch.manual_seed(seed)
    shared_kv = {}
    for lt, idx in type_to_idx.items():
        cfg = Gemma4AttentionConfig(text_args, idx)
        kvc = init_kv_cache(mesh_device=mesh_device, config=cfg, paged_attention_config=pac, cache_dtype=ttnn.bfloat16)
        k_ref = torch.randn(1, cfg.num_key_value_heads, context_len, cfg.head_dim).bfloat16().float()
        v_ref = torch.randn(1, cfg.num_key_value_heads, context_len, cfg.head_dim).bfloat16().float()
        k_fill = _kv_to_tt(k_ref, mesh_device, cfg.num_key_value_heads, cfg.num_attention_heads, tp, num_devices)
        v_fill = _kv_to_tt(v_ref, mesh_device, cfg.num_key_value_heads, cfg.num_attention_heads, tp, num_devices)
        ttnn.experimental.paged_fill_cache(kvc[0], k_fill, page_table_tt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(kvc[1], v_fill, page_table_tt, batch_idx=0)
        shared_kv[lt] = kvc
        logger.info(f"[kv] {lt}: layer_idx={idx} kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")
    return shared_kv, page_table_tt


def _build_standalone(
    mesh_device, placement_mode, context_len=DEFAULT_CONTEXT, max_seq_len=1024, budget_mb=None, tune_matmuls=None
):
    """Build the drafter alone, on a stub target, with the given weight placement.

    Returns a dict of everything a step needs plus the placement policy object.
    """
    if not ASSISTANT_PATH:
        pytest.skip("set GEMMA4_ASSISTANT_MODEL to run")

    num_devices, tp, mesh_config = _mesh_bits(mesh_device)
    ccl_manager = CCLManager(mesh_device) if tp > 1 else None
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 else None

    hf_config = Gemma4AssistantArgs.load_hf_config(ASSISTANT_PATH)
    assistant_args = Gemma4AssistantArgs.from_hf_config(hf_config)
    assistant_args.model_cache_path = assistant_args.resolve_model_cache_path(ASSISTANT_PATH)
    text_args = assistant_args.text_args
    text_args.max_seq_len = max_seq_len

    stub = _TargetStub(
        mesh_device=mesh_device,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        text_args=text_args,
        backbone_hidden_size=assistant_args.backbone_hidden_size,
        max_seq_len=max_seq_len,
    )

    kwargs = {} if budget_mb is None else {"budget_bytes": int(budget_mb * (1 << 20))}
    placement = WeightPlacement(mode=placement_mode, label=f"assistant-{placement_mode}", **kwargs)
    # l1_sharded needs the matching program config or the matmul validator FATALs.
    if placement.sharded:
        tune_matmuls = True

    if tune_matmuls is None:
        tune_matmuls = os.getenv("GEMMA4_TUNE_MATMULS", "0") == "1"
    tuner = DecodeMatmulTuner(mesh_device, enabled=tune_matmuls)

    state_dict = Gemma4AssistantArgs.load_state_dict(ASSISTANT_PATH, dummy_weights=False)
    assistant = Gemma4AssistantModel(
        mesh_device=mesh_device,
        assistant_args=assistant_args,
        target_model=stub,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=str(assistant_args.weight_cache_path(ttnn.bfloat16)),
        mesh_config=mesh_config,
        weight_placement=placement,
        matmul_tuner=tuner,
    )

    shared_kv, page_table_tt = _fabricate_shared_kv(mesh_device, text_args, tp, num_devices, mapper, context_len)
    page_tables = {lt: page_table_tt for lt in shared_kv}

    # Fixed drafter inputs: one token id, one recurrent hidden, one position.
    pos = context_len - 1
    token_tt = ttnn.from_torch(
        torch.tensor([[12345]], dtype=torch.int64),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mapper,
    )
    pu = torch.zeros((1, 32), dtype=torch.int64)
    pu[0, 0] = pos
    pos_uint32 = ttnn.from_torch(
        pu, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=mapper
    )
    pos_int32 = ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=mapper,
    )
    g = torch.Generator().manual_seed(7)
    hidden_host = (torch.randn(1, 1, 1, assistant_args.backbone_hidden_size, generator=g) * 5.0).bfloat16()
    hidden = ttnn.from_torch(
        hidden_host, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
    )

    return {
        "assistant": assistant,
        "stub": stub,
        "placement": placement,
        "shared_kv": shared_kv,
        "page_tables": page_tables,
        "token": token_tt,
        "hidden": hidden,
        "pos_uint32": pos_uint32,
        "pos_int32": pos_int32,
        "tp": tp,
        "context_len": context_len,
    }


def _step_args(rig):
    return (rig["token"], rig["hidden"], rig["shared_kv"], rig["page_tables"], rig["pos_uint32"], rig["pos_int32"])


def _time_traced_step(mesh_device, rig, return_logits, reps=TRACE_REPS):
    """Compile -> capture -> replay N times. Same shape as test_draft_step_breakdown."""
    assistant = rig["assistant"]
    lg, hn = assistant.step(*_step_args(rig), return_logits=return_logits)
    ttnn.synchronize_device(mesh_device)
    if lg is not None:
        lg.deallocate(True)
    hn.deallocate(True)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    lg, hn = assistant.step(*_step_args(rig), return_logits=return_logits)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    for _ in range(3):  # warm the replay path before timing
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) / reps * 1e3

    ttnn.release_trace(mesh_device, tid)
    if lg is not None:
        lg.deallocate(True)
    hn.deallocate(True)
    return ms


def _argmax_token(assistant, logits, rows=1):
    """Next-token id as a DEVICE [1,1,rows] uint32 tensor. Mirrors SpeculativeDecoder._argmax_last.

    Under CME the head does its own argmax over the ~4096 candidates and maps the
    winner back to a vocab id on device. The dense fallback needs the pad-to-32
    dance because ``ttnn.argmax``'s multicore path is row-parallel and returns
    garbage unless the row dim is exactly one tile.
    """
    if isinstance(logits, CmeLogits):
        return assistant.masked_embedding.argmax_token_id(logits, rows)
    R32 = 32
    src, padded = logits, None
    if rows < R32:
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, R32 - rows), (0, 0)], value=0.0)
        src = padded
    u = ttnn.untilize(src, use_multicore=True)
    if padded is not None and not _same_buffer(padded, logits):
        padded.deallocate(True)
    idx = ttnn.argmax(u, dim=-1, keepdim=False)
    u.deallocate(True)
    if rows < R32:
        sliced = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])
        idx.deallocate(True)
        idx = sliced
    return idx


def _time_fused_k_steps(mesh_device, rig, k, reps=TRACE_REPS, mode="full"):
    """K drafter steps CHAINED inside ONE trace. Zero host work between replays.

    This is the standalone twin of ``spec_decode._fused_body_batched``'s draft
    section: argmax on device, ``ttnn.reshape`` re-feeds the id as the next token,
    and the hidden recurrence is a plain Python rebind chained in-graph. Nothing
    crosses to host.

    It is a valid steady-state benchmark because a drafter iteration is
    side-effect-free on device state: the drafter never writes KV
    (``is_kv_shared=True``, so ``attention/decode.py`` skips K/V proj, the K
    rotation and ``paged_update_cache``), and all K steps query ONE fixed position
    (HF SinglePositionMTP). So replaying the identical trace N times repeats
    identical work, and no buffer needs resetting between replays.

    ``mode`` decomposes the step, all three chaining the hidden identically so
    only the head differs:
      ``full``      — CME head + argmax_token_id, the real drafter step
      ``no_argmax`` — CME head runs, argmax_token_id does not (token stays fixed)
      ``backbone``  — ``return_logits=False``: no head at all
    full - no_argmax isolates ``argmax_token_id`` (its argmax over ~4096
    candidates plus the second ``ttnn.gather`` over ``pack.ids``), which every
    previous standalone number omitted by stopping at logits.

    Returns (ms per K-step iteration, capture seconds).
    """
    assistant = rig["assistant"]
    shared_kv, page_tables = rig["shared_kv"], rig["page_tables"]
    pu, pi = rig["pos_uint32"], rig["pos_int32"]
    want_logits = mode != "backbone"

    def body():
        tok, h = rig["token"], rig["hidden"]
        last = None
        for _ in range(k):
            logits, h = assistant.step(tok, h, shared_kv, page_tables, pu, pi, return_logits=want_logits)
            if mode == "full":
                last = _argmax_token(assistant, logits, rows=1)
                logits.deallocate(True)
                # ROW_MAJOR reshape is a free VIEW of `last`. Never deallocate
                # either one: freeing the view hands its storage back and a later
                # replay reads zeros (the verify_x bug in spec_decode.py:1641).
                tok = ttnn.reshape(last, (1, 1))
            elif logits is not None:
                logits.deallocate(True)
        return last, h

    idx, h = body()  # compile
    ttnn.synchronize_device(mesh_device)
    if idx is not None:
        idx.deallocate(True)
    h.deallocate(True)

    t_cap = time.perf_counter()
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    idx, h = body()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    capture_s = time.perf_counter() - t_cap

    for _ in range(3):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) / reps * 1e3

    ttnn.release_trace(mesh_device, tid)
    if idx is not None:
        idx.deallocate(True)
    h.deallocate(True)
    return ms, capture_s


def _time_per_step_replay(mesh_device, rig, k, reps=max(4, TRACE_REPS // 8)):
    """CONTROL: one step traced, replayed K times, paying host dispatch per step.

    Reproduces ``spec_decode._draft_traced`` standalone — the design the fused
    path replaced. Per draft step it pays, outside the trace: a device->device
    ``ttnn.copy`` for the hidden recurrence (an in-trace copy fails capture with
    "Writes not supported during trace capture"), an eagerly dispatched argmax, a
    ``to_torch`` readback of one uint32, and a host->device token write.

    The delta against ``_time_fused_k_steps`` IS the host dispatch overhead.
    Fewer reps by default: each one is K host round-trips, so it is slow.
    """
    assistant = rig["assistant"]
    shared_kv, page_tables = rig["shared_kv"], rig["page_tables"]
    pu, pi = rig["pos_uint32"], rig["pos_int32"]
    tp = mesh_device.get_num_devices()
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if tp > 1 else None

    tok_in = ttnn.clone(rig["token"])
    h_in = ttnn.clone(rig["hidden"])

    logits, h_next = assistant.step(tok_in, h_in, shared_kv, page_tables, pu, pi)
    ttnn.synchronize_device(mesh_device)
    logits.deallocate(True)
    h_next.deallocate(True)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    logits, h_next = assistant.step(tok_in, h_in, shared_kv, page_tables, pu, pi)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    def one_iteration():
        ttnn.copy(rig["hidden"], h_in)  # reset the recurrence to this iter's seed
        for step in range(k):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            if step < k - 1:
                ttnn.copy(h_next, h_in)  # the trace cannot update its own input
            idx = _argmax_token(assistant, logits, rows=1)  # eager, not traced
            t = ttnn.to_torch(ttnn.get_device_tensors(idx)[0] if tp > 1 else idx)
            tok = int(t.reshape(-1)[0])
            idx.deallocate(True)
            if step < k - 1:
                host_tok = ttnn.from_torch(
                    torch.tensor([[tok]], dtype=torch.int64),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    mesh_mapper=mapper,
                )
                ttnn.copy_host_to_device_tensor(host_tok, tok_in)

    for _ in range(2):
        one_iteration()
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter()
    for _ in range(reps):
        one_iteration()
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) / reps * 1e3

    ttnn.release_trace(mesh_device, tid)
    for t in (logits, h_next, tok_in, h_in):
        t.deallocate(True)
    return ms


def _buffer_types(assistant):
    """Buffer type of a representative weight from each module, for assertions."""
    layer0 = assistant.layers[0]
    out = {
        "mlp.gate_proj": layer0.shared_mlp.gate_proj.memory_config().buffer_type,
        "mlp.down_proj": layer0.shared_mlp.down_proj.memory_config().buffer_type,
        "attn.wqkv": layer0.self_attn.weights.wqkv.memory_config().buffer_type,
        "attn.o_proj": layer0.self_attn.weights.o_proj.memory_config().buffer_type,
        "pre_projection": assistant.pre_projection.memory_config().buffer_type,
        "post_projection": assistant.post_projection.memory_config().buffer_type,
    }
    if assistant.masked_embedding is not None:
        out["cme.centroids"] = assistant.masked_embedding.centroids.memory_config().buffer_type
        out["cme.embed_table"] = assistant.masked_embedding.embed_table.memory_config().buffer_type
    return out


# ── tests ────────────────────────────────────────────────────────────────────


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_placement_accounting(mesh_device, reset_seeds):
    """L1 mode actually moves the weights, and the big table is reported, not hidden."""
    rig = _build_standalone(mesh_device, "l1")
    placement = rig["placement"]
    logger.info("\n" + placement.report())

    types = _buffer_types(rig["assistant"])
    for name, bt in types.items():
        logger.info(f"[buffer] {name:<20} {bt}")

    pinned = [
        "mlp.gate_proj",
        "mlp.down_proj",
        "attn.wqkv",
        "attn.o_proj",
        "pre_projection",
        "post_projection",
    ]
    for name in pinned:
        assert types[name] == ttnn.BufferType.L1, f"{name} should be in L1, got {types[name]}"

    summary = placement.summary()
    assert summary["l1_bytes"] > 8 * (1 << 20), f"expected >8 MB pinned, got {summary['l1_bytes']}"
    if "cme.embed_table" in types:
        # 128 MiB replicated: must NOT have been pinned under the default budget.
        assert types["cme.embed_table"] == ttnn.BufferType.DRAM
        assert summary["dram_bytes"] > 100 * (1 << 20)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_linear_l1_weight_matches_fp32(mesh_device, reset_seeds):
    """The real correctness claim: ttnn.linear is as accurate with an L1 weight as with a DRAM one.

    Checked against an fp32 torch reference at the drafter's actual weight
    shapes, across the 2x2 of {in0, in1} x {DRAM, L1}.

    MEASURED: all four combinations land within ~7e-4 PCC of fp32. But the
    all-DRAM combination is not bit-identical to the other three whenever
    K > 256 and N == 256 (the o_proj / down_proj / pre_projection shapes) —
    putting *either* operand in L1 selects a different matmul program, whose
    K-block accumulation order rounds differently in bf16. That is a few ULPs,
    not a defect, but it is why the model-level check below has to be
    perturbation-calibrated rather than exact.
    """
    shapes = [
        (32, 256, 1024),  # gate_proj / up_proj / wqkv
        (32, 256, 1536),  # post_projection
        (32, 512, 256),  # o_proj
        (32, 1024, 256),  # down_proj
        (32, 3072, 256),  # pre_projection
        (32, 1024, 1024),
    ]
    mcs = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
    torch.manual_seed(0)
    for M, K, N in shapes:
        x = torch.randn(1, 1, M, K).bfloat16()
        w = torch.randn(1, 1, K, N).bfloat16()
        ref = x.float() @ w.float()
        pccs = {}
        for a, b in (("dram", "dram"), ("dram", "l1"), ("l1", "dram"), ("l1", "l1")):
            xt = ttnn.from_torch(
                x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mcs[a]
            )
            wt = ttnn.from_torch(
                w, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mcs[b]
            )
            y = ttnn.linear(xt, wt)
            pccs[f"in0={a},in1={b}"] = _pcc(ref, ttnn.to_torch(y))
            for t in (xt, wt, y):
                t.deallocate(True)
        logger.info(f"[linear] M={M} K={K} N={N} " + " ".join(f"{k}:{v:.6f}" for k, v in pccs.items()))
        for key, p in pccs.items():
            assert p >= 0.999, f"linear({M},{K},{N}) {key} lost accuracy vs fp32: pcc={p}"
        spread = max(pccs.values()) - min(pccs.values())
        assert spread < 1e-3, f"linear({M},{K},{N}) placement changed accuracy materially: spread={spread}"


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_standalone_parity_dram_vs_l1(mesh_device, reset_seeds):
    """L1 weights must perturb the drafter no more than a 1-ULP input change does.

    An exact-match assertion would be wrong here. The op-level test above shows
    L1 placement can pick a different matmul program for the K>256,N=256 shapes,
    differing in the last bf16 ULPs; and this harness drives the drafter with a
    synthetic hidden and a synthetic embedding table, which is out of
    distribution and numerically ill-conditioned — 4 layers plus a top-k over
    2048 centroids amplify any ULP into a different candidate set.

    So the test calibrates against that sensitivity directly: re-run the DRAM
    arm with every element of the input hidden nudged by one bf16 ULP, and
    require the DRAM-vs-L1 distance to be no larger than the DRAM-vs-perturbed
    distance. MEASURED: perturbing the input diverges MORE (hidden pcc ~0.64)
    than switching to L1 does (~0.81), i.e. L1 placement is within the model's
    own noise floor on this input.

    A DRAM-vs-DRAM control is included: it is bit-identical, which is what makes
    the comparison above meaningful rather than a measurement of run-to-run jitter.
    """

    def _outputs(rig):
        logits, next_hidden = rig["assistant"].step(*_step_args(rig))
        ttnn.synchronize_device(mesh_device)
        return (
            _dev0(logits.values, mesh_device).float(),
            _dev0(logits.ids, mesh_device).to(torch.int64),
            _dev0(next_hidden, mesh_device).float(),
        )

    rig_dram = _build_standalone(mesh_device, "dram")
    v0, i0, h0 = _outputs(rig_dram)

    # Control: the harness itself is deterministic.
    v_ctl, i_ctl, h_ctl = _outputs(_build_standalone(mesh_device, "dram"))
    assert torch.equal(h0, h_ctl) and torch.equal(i0, i_ctl), "DRAM-vs-DRAM is not deterministic; harness is unsound"
    logger.info("[parity] control DRAM vs DRAM: bit-identical")

    v1, i1, h1 = _outputs(_build_standalone(mesh_device, "l1"))

    # Yardstick: one bf16 ULP on the input hidden, DRAM weights throughout.
    hb = _dev0(rig_dram["hidden"], mesh_device)
    perturbed = ((hb.view(torch.int16).int() + 1).to(torch.int16)).view(torch.bfloat16)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    rig_dram["hidden"] = ttnn.from_torch(
        perturbed, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
    )
    v2, i2, h2 = _outputs(rig_dram)

    d_l1 = _pcc(h0, h1)
    d_ulp = _pcc(h0, h2)
    logger.info(f"[parity] DRAM vs L1        : hidden_pcc={d_l1:.6f} logits_pcc={_pcc(v0, v1):.6f}")
    logger.info(f"[parity] DRAM vs +1ULP-in  : hidden_pcc={d_ulp:.6f} logits_pcc={_pcc(v0, v2):.6f}")
    assert d_l1 >= d_ulp, (
        f"L1 placement perturbs the drafter MORE than a 1-ULP input change "
        f"(pcc {d_l1:.6f} < {d_ulp:.6f}) — that is beyond rounding, investigate"
    )


@_needs_assistant
@pytest.mark.parametrize("context_len", [128, 512, 1024], ids=["ctx128", "ctx512", "ctx1024"])
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_standalone_perf_dram_vs_l1(mesh_device, context_len, reset_seeds):
    """Traced drafter step: DRAM weights vs L1 weights, at one context length.

    Reports ms/step and tok/s/u for each arm, with and without the output head,
    so the head cost is separable from the 4-layer backbone. Context is swept by
    the parametrization because KV read scales with context while weight read
    does not — if L1 weights help, the relative gain should shrink as context
    grows.
    """
    # GEMMA4_L1_ARM restricts the run to a single arm. Needed when profiling:
    # tracy attributes ops per process, so the two arms must not share one.
    arm = os.getenv("GEMMA4_L1_ARM", "").strip().lower()
    modes = (arm,) if arm in ("dram", "l1") else ("dram", "l1")

    results = {}
    for mode in modes:
        rig = _build_standalone(mesh_device, mode, context_len=context_len)
        if rig["placement"].enabled:
            logger.info("\n" + rig["placement"].report())
        full = _time_traced_step(mesh_device, rig, return_logits=True)
        nolm = _time_traced_step(mesh_device, rig, return_logits=False)
        results[mode] = (full, nolm, rig["placement"].summary()["l1_bytes"])
        logger.info(
            f"[perf ctx={context_len}] {mode.upper():<4} full={full:.3f} ms/step "
            f"({1e3/full:.2f} tok/s/u)  backbone-only={nolm:.3f} ms  head={full-nolm:.3f} ms"
        )

    if len(modes) == 1:
        return  # single-arm profiling run; nothing to compare

    (fd, nd, _), (fl, nl, l1b) = results["dram"], results["l1"]
    logger.info(
        f"[perf ctx={context_len}] ===== L1 vs DRAM: full {fd:.3f} -> {fl:.3f} ms "
        f"({(fd-fl)/fd*100:+.2f}%), backbone {nd:.3f} -> {nl:.3f} ms ({(nd-nl)/nd*100:+.2f}%), "
        f"pinned {l1b/(1<<20):.1f} MB/device ====="
    )
    # No perf assertion: the measurement IS the result. Guard only against a
    # broken L1 arm that silently fell back to DRAM.
    assert l1b > 8 * (1 << 20), "L1 arm pinned almost nothing — placement did not take effect"


@_needs_assistant
@parametrize_mesh_with_fabric(
    mesh_shapes=[(1, 2)],
    device_params_extra={"trace_region_size": int(os.getenv("GEMMA4_TRACE_REGION_SIZE", 400_000_000))},
)
def test_fused_draft_k_steps(mesh_device, reset_seeds):
    """THE instrument: K chained drafter steps in one trace, vs per-step replay.

    Every previously published standalone drafter number came from
    ``_time_traced_step``, which captures exactly ONE step and re-reads the same
    hidden and token on every replay (``h_next`` is discarded). The K-step cost
    was only ever EXTRAPOLATED — ``test_spec_decode.py::test_draft_step_breakdown``
    literally prints ``full K={K} steps ~= full_ms*K``. This measures it.

    Two arms:
      fused   — K steps chained in ONE trace, zero host work between replays
      control — one step traced and replayed K times, paying ttnn.copy +
                eager argmax + uint32 readback + host->device token write per step

    The control is the design the fused end-to-end path replaced; the fused
    write-up measured it at 10.2 ms/step against ~4-5 ms of device work. The
    fused/control ratio here is that same overhead, isolated to the drafter.

    K is swept because the packed-verify constraint ``(H_local*P) % 32 == 0``
    is verify-side only, so a drafter-only benchmark can use any K.
    """
    ks = [int(s) for s in os.getenv("GEMMA4_DRAFT_K_SWEEP", "1,2,3,4,8").split(",") if s.strip()]
    rig = _build_standalone(mesh_device, "dram")

    logger.info(
        f"{'K':>3} {'fused ms/iter':>14} {'ms/step':>9} {'tok/s/u':>9} "
        f"{'control ms/iter':>16} {'ms/step':>9} {'speedup':>8} {'capture s':>10}"
    )
    rows = []
    for k in ks:
        fused_ms, cap_s = _time_fused_k_steps(mesh_device, rig, k)
        ctl_ms = _time_per_step_replay(mesh_device, rig, k)
        rows.append((k, fused_ms, ctl_ms))
        logger.info(
            f"{k:>3} {fused_ms:>14.3f} {fused_ms/k:>9.3f} {k*1e3/fused_ms:>9.2f} "
            f"{ctl_ms:>16.3f} {ctl_ms/k:>9.3f} {ctl_ms/fused_ms:>7.2f}x {cap_s:>10.2f}"
        )

    # Where the fused step actually goes. Every previous standalone number
    # stopped at logits, so argmax_token_id was never in any published figure.
    kb = ks[-1]
    full_ms, _ = _time_fused_k_steps(mesh_device, rig, kb, mode="full")
    noam_ms, _ = _time_fused_k_steps(mesh_device, rig, kb, mode="no_argmax")
    back_ms, _ = _time_fused_k_steps(mesh_device, rig, kb, mode="backbone")
    logger.info(f"[breakdown K={kb}] per step, ms:")
    logger.info(f"    backbone (4 layers, no head)   {back_ms/kb:8.3f}   {back_ms/full_ms*100:5.1f}%")
    logger.info(
        f"    CME head forward               {(noam_ms-back_ms)/kb:8.3f}   {(noam_ms-back_ms)/full_ms*100:5.1f}%"
    )
    logger.info(
        f"    argmax_token_id (+2nd gather)  {(full_ms-noam_ms)/kb:8.3f}   {(full_ms-noam_ms)/full_ms*100:5.1f}%"
    )
    logger.info(f"    TOTAL                          {full_ms/kb:8.3f}")

    # The fused chain must beat the per-step-replay control at every K>1. At K=1
    # they are the same graph plus one eager argmax + readback, so the control is
    # only slightly worse there; the gap is what grows with K.
    for k, fused_ms, ctl_ms in rows:
        if k > 1:
            assert ctl_ms > fused_ms, f"K={k}: fused ({fused_ms:.3f} ms) did not beat per-step replay ({ctl_ms:.3f} ms)"
    # Scaling check: a chained trace should be close to linear in K, since every
    # step is the same graph and nothing is amortized across steps. A large
    # sublinearity would mean the K=1 number was dominated by per-replay overhead.
    if len(rows) > 1:
        k1 = next((f for k, f, _ in rows if k == 1), None)
        if k1:
            for k, fused_ms, _ in rows:
                logger.info(f"[scale] K={k}: {fused_ms/(k1*k):.3f}x of linear extrapolation from K=1")


def _op_timed(mesh_device, fn, inner=4, replays=8, protect=()):
    """Per-call us for one op graph, by repeating it `inner` times in one trace.

    `fn` must not deallocate anything it did not create. `protect` lists tensors
    the caller owns: a returned tensor that ALIASES one of them is not freed.
    Required because a shape-only op can hand back a view of its input — e.g.
    ``ttnn.pad`` to 32 rows on a [1,1,rows<32,N] TILE tensor is a logical no-op
    that ttnn satisfies with an alias, so freeing the "output" frees the pack
    (``masked_embedding._argmax_rows:404-413``).
    """

    def _free(o):
        try:
            if any(_same_buffer(o, p) for p in protect):
                return
            o.deallocate(True)
        except Exception:  # noqa: BLE001
            pass

    outs = [fn() for _ in range(inner)]
    ttnn.synchronize_device(mesh_device)
    for o in outs:
        _free(o)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [fn() for _ in range(inner)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    for _ in range(3):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / replays / inner * 1e6
    ttnn.release_trace(mesh_device, tid)
    for o in outs:
        _free(o)
    return us


@_needs_assistant
@parametrize_mesh_with_fabric(
    mesh_shapes=[(1, 2)],
    device_params_extra={"trace_region_size": int(os.getenv("GEMMA4_TRACE_REGION_SIZE", 400_000_000))},
)
def test_argmax_token_id_breakdown(mesh_device, reset_seeds):
    """Split ``argmax_token_id``, measured at 66% of the fused drafter step.

    ``test_fused_draft_k_steps`` attributes 2.26 of 3.43 ms/step to this call,
    which no previous standalone measurement included (they all stopped at
    logits). ``CME_DIGIT_GATHER.md`` §7 flagged the second ``ttnn.gather`` here as
    unmeasured; this decides whether that gather, the argmax, the untilize or the
    layout conversions owns the time — the first CME gather had exactly this
    shape of problem (one core, cost tracking the INPUT not the work).

    Cumulative prefixes, so each row's cost is the delta from the row above.
    """
    rig = _build_standalone(mesh_device, "dram")
    logits, h = rig["assistant"].step(*_step_args(rig), return_logits=True)
    ttnn.synchronize_device(mesh_device)
    h.deallocate(True)
    assert isinstance(logits, CmeLogits), "this breakdown is CME-specific"
    vals, ids = logits.values, logits.ids
    rows = 1
    N = int(vals.shape[-1])
    logger.info(f"[argmax-bd] values{tuple(vals.shape)} {vals.dtype}  ids{tuple(ids.shape)} {ids.dtype}  N={N}")

    def p_pad():
        return ttnn.pad(vals, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], value=0.0)

    def p_untilize():
        return ttnn.untilize(p_pad(), use_multicore=True)

    def p_argmax():
        u = p_untilize()
        idx = ttnn.argmax(u, dim=-1, keepdim=False)
        u.deallocate(True)
        return idx

    def p_argmax_rows():
        return Gemma4TTMaskedEmbedder._argmax_rows(vals, rows)

    def p_col():
        # ROW_MAJOR reshape = free view of the argmax output; do not free either.
        return ttnn.reshape(p_argmax_rows(), (1, 1, rows, 1))

    def p_gather():
        return ttnn.gather(ids, dim=-1, index=p_col())

    def p_full():
        return rig["assistant"].masked_embedding.argmax_token_id(logits, rows)

    stages = [
        ("pad (alias when rows<32)", p_pad),
        ("+ untilize(multicore)", p_untilize),
        ("+ argmax", p_argmax),
        ("+ slice = _argmax_rows", p_argmax_rows),
        ("+ reshape to column (RM view)", p_col),
        ("+ ttnn.gather(ids) ROW_MAJOR", p_gather),
        ("+ reshape = FULL argmax_token_id", p_full),
    ]
    assert ids.layout == ttnn.ROW_MAJOR_LAYOUT, (
        "pack.ids must be ROW_MAJOR: gathering from a TILE [1,1,rows,N] costs 2069 us "
        f"against 10.5 ROW_MAJOR, got {ids.layout}"
    )
    prev = 0.0
    logger.info(f"{'stage':<34}{'cumulative us':>15}{'delta us':>11}")
    for name, fn in stages:
        us = _op_timed(mesh_device, fn, protect=(vals, ids))
        logger.info(f"{name:<34}{us:>15.1f}{us-prev:>11.1f}")
        prev = us
    logger.info(f"[argmax-bd] full call {prev:.1f} us/step; drafter step measured at ~1335 us")
    logits.deallocate(True)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_second_gather_alternatives(mesh_device, reset_seeds):
    """Replace ``ttnn.gather(pack.ids, dim=-1, index=argmax_col)`` — 2045 us/step.

    Measured at 60% of the fused drafter step. Same failure mode as the first CME
    gather: cost tracks the INPUT, and a [1,1,1,4096] TILE tensor physically
    occupies 32x4096 = 131072 elements, so the generic gather walks 32x more than
    the row holds.

    The operation is ``out[s] = ids[s, local[s]]`` — one element per row. Arms:
      a  current, TILE ids + TILE index
      b  same over 32 real rows (does cost track the padded volume?)
      c  ROW_MAJOR ids and index (does dropping tile padding fix it?)
      d  mask-and-reduce: eq(arange, local) * ids, summed. Exact in fp32
         (max id 262143 < 2^24) and every op is multicore elementwise/reduction.
    """
    torch.manual_seed(0)
    N, V = 4096, 262144
    ar = torch.arange(N, dtype=torch.float32).reshape(1, 1, 1, N)
    ids_t = torch.randint(0, V, (1, 1, 1, N), dtype=torch.int32)
    sel = 1234
    idx_t = torch.tensor([[[[sel]]]], dtype=torch.int32)
    gold = int(ids_t[0, 0, 0, sel])

    def dev(t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=mesh_device, layout=layout, dtype=dtype)

    ids_tile = dev(ids_t, ttnn.uint32)
    idx_tile = dev(idx_t, ttnn.uint32)
    arange_tile = dev(ar, ttnn.float32)
    ids_f32 = dev(ids_t.float(), ttnn.float32)

    ids_rm = dev(ids_t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    idx_rm = dev(idx_t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    ids32_tile = dev(ids_t.repeat(1, 1, 32, 1), ttnn.uint32)
    idx32_tile = dev(idx_t.repeat(1, 1, 32, 1), ttnn.uint32)

    def a():
        return ttnn.gather(ids_tile, dim=-1, index=idx_tile)

    def b():
        return ttnn.gather(ids32_tile, dim=-1, index=idx32_tile)

    def c():
        return ttnn.gather(ids_rm, dim=-1, index=idx_rm)

    def d():
        # local index -> fp32 column, broadcast-compare against 0..N-1
        col = ttnn.typecast(idx_tile, ttnn.float32)
        mask = ttnn.eq(arange_tile, col)  # [1,1,1,N] fp32, one-hot
        col.deallocate(True)
        prod = ttnn.mul(ids_f32, mask)
        mask.deallocate(True)
        out = ttnn.sum(prod, dim=-1)
        prod.deallocate(True)
        return out

    protect = (ids_tile, idx_tile, arange_tile, ids_f32, ids_rm, idx_rm, ids32_tile, idx32_tile)
    for name, fn, note in (
        ("a TILE ids, 1 row (CURRENT)", a, ""),
        ("b TILE ids, 32 real rows", b, "cost vs padded volume"),
        ("c ROW_MAJOR ids + index", c, ""),
        ("d mask-and-reduce (fp32)", d, ""),
    ):
        try:
            out = fn()
            ttnn.synchronize_device(mesh_device)
            got = ttnn.to_torch(out).reshape(-1)[0]
            ok = int(got) == gold
            out.deallocate(True)
            us = _op_timed(mesh_device, fn, protect=protect)
            logger.info(f"[gather-alt] {name:<30} {us:>9.1f} us  correct={ok} (got {int(got)} want {gold}) {note}")
        except Exception as ex:  # noqa: BLE001
            logger.info(f"[gather-alt] {name:<30} FAILED: {str(ex).split('backtrace')[0].strip()[:110]}")


@_needs_assistant
@parametrize_mesh_with_fabric(
    mesh_shapes=[(1, 2)],
    device_params_extra={"trace_region_size": int(os.getenv("GEMMA4_TRACE_REGION_SIZE", 400_000_000))},
)
def test_fused_k_step_dram_vs_l1(mesh_device, reset_seeds):
    """The L1 weight A/B, re-measured on the K-chained harness.

    Supersedes ``test_standalone_perf_dram_vs_l1`` as the number to quote: that
    one measures a single step replayed with a frozen input, which is not what a
    drafter does. Also dumps ``WeightPlacement.report()`` for each arm, which is
    the ground truth for WHICH tensors got pinned — admission is whole-tensor
    greedy first-fit in construction order, with a refund for anything that turns
    out not to be a shardable matmul weight, so the pinned set is not a layer
    prefix and cannot be inferred from the total.
    """
    k = int(os.getenv("GEMMA4_SPEC_DRAFT_LEN", "3"))
    budget = float(os.getenv("GEMMA4_L1_WEIGHT_BUDGET_MB", "8"))

    # EVERY arm gets the tuner. ``_build_standalone`` force-enables it for
    # l1_sharded (the matmul validator FATALs without the matching program
    # config), so leaving the DRAM arm on ttnn's automatic choice would credit
    # the tuner's 1.1-2.4x to L1 placement. That confound was measured once
    # already; the untuned DRAM arm is kept as a separate row, not as the baseline.
    arms = [
        ("dram (auto cfg)", "dram", False, None),
        ("dram (tuned)", "dram", True, None),
        ("l1 interleaved", "l1", True, budget),
        ("l1 WIDTH_SHARDED", "l1_sharded", True, budget),
    ]

    results = {}
    for label, mode, tuned, mb in arms:
        rig = _build_standalone(mesh_device, mode, budget_mb=mb, tune_matmuls=tuned)
        if rig["placement"].enabled:
            logger.info("\n" + rig["placement"].report())
        ms, _ = _time_fused_k_steps(mesh_device, rig, k)
        pinned = rig["placement"].summary()["l1_bytes"]
        tn, ttl = rig["assistant"].mm.stats()
        results[label] = (ms, pinned)
        logger.info(
            f"[k-perf K={k}] {label:<18} {ms:>8.3f} ms/iter  {ms/k:>7.3f} ms/step  "
            f"{k*1e3/ms:>7.2f} tok/s/u  pinned {pinned/(1<<20):>6.2f} MB/device  tuned {tn}/{ttl} shapes"
        )

    base = results["dram (tuned)"][0]
    logger.info(f"[k-perf K={k}] ===== all vs 'dram (tuned)' =====")
    for label, (ms, pinned) in results.items():
        logger.info(f"    {label:<18} {ms:>8.3f} ms/iter  {(base-ms)/base*100:>+6.2f}%  pinned {pinned/(1<<20):.2f} MB")


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_perf_by_weight_class(mesh_device, reset_seeds):
    """Attribute the L1-vs-DRAM step-time delta to individual weight classes.

    Pins one class at a time via ``GEMMA4_L1_ONLY`` and times the backbone
    (``return_logits=False``, so the CME head — which dominates wall-clock and is
    unaffected — is excluded). Answers "which weights actually cost time when
    they move to L1", without needing the profiler.
    """
    classes = [
        "gate_proj",
        "up_proj",
        "down_proj",
        "wqkv",
        "o_proj",
        "pre_projection",
        "post_projection",
        "layernorm",
    ]
    prev = os.environ.get("GEMMA4_L1_ONLY")
    try:
        os.environ.pop("GEMMA4_L1_ONLY", None)
        base = _time_traced_step(mesh_device, _build_standalone(mesh_device, "dram"), return_logits=False)
        logger.info(f"[byclass] DRAM baseline backbone = {base:.4f} ms")
        for cls in classes:
            os.environ["GEMMA4_L1_ONLY"] = cls
            rig = _build_standalone(mesh_device, "l1")
            ms = _time_traced_step(mesh_device, rig, return_logits=False)
            mb = rig["placement"].summary()["l1_bytes"] / (1 << 20)
            logger.info(
                f"[byclass] L1 only={cls:<16} pinned={mb:5.2f} MB  backbone={ms:.4f} ms  "
                f"delta={ms-base:+.4f} ms ({(ms-base)/base*100:+.2f}%)"
            )
    finally:
        os.environ.pop("GEMMA4_L1_ONLY", None)
        if prev is not None:
            os.environ["GEMMA4_L1_ONLY"] = prev


@_needs_assistant
@pytest.mark.parametrize("budget_mb", [160, 256], ids=["budget160", "budget256"])
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_l1_capacity_ceiling(mesh_device, budget_mb, reset_seeds):
    """Raise the budget until the 128 MiB embed_table is admitted; record what breaks.

    This is the hard capacity number: whether a replicated full-vocab output
    embedding can be L1-resident at all on a P150, or whether it collides with
    the statically-allocated circular buffers.
    """
    try:
        rig = _build_standalone(mesh_device, "l1", budget_mb=budget_mb)
    except Exception as e:  # noqa: BLE001 - the failure mode IS the measurement
        logger.info(f"[capacity] budget={budget_mb} MB FAILED to allocate: {type(e).__name__}: {e}")
        pytest.skip(f"L1 budget {budget_mb} MB does not fit: {e}")

    placement = rig["placement"]
    logger.info("\n" + placement.report())
    types = _buffer_types(rig["assistant"])
    logger.info(f"[capacity] embed_table landed in {types.get('cme.embed_table')}")
    logger.info(f"[capacity] pinned {placement.summary()['l1_bytes']/(1<<20):.1f} MB/device")

    try:
        ms = _time_traced_step(mesh_device, rig, return_logits=True, reps=10)
        logger.info(f"[capacity] step still runs at budget={budget_mb} MB: {ms:.3f} ms/step")
    except Exception as e:  # noqa: BLE001
        logger.info(f"[capacity] allocation succeeded but the step FAILED: {type(e).__name__}: {e}")


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_tuned_matmuls_in_model(mesh_device, reset_seeds):
    """Explicit decode matmul program configs, measured in the real drafter.

    The isolated numbers (`test_matmul_weight_placement.py`) say the tuned config
    is 1.1-2.4x per matmul with the weights left in DRAM. This checks what that is
    worth once it is inside the model, where the linears are only part of the
    step — and that the output does not change.
    """
    out = {}
    for tag, tune in (("auto", False), ("tuned", True)):
        rig = _build_standalone(mesh_device, "dram", tune_matmuls=tune)
        logits, next_hidden = rig["assistant"].step(*_step_args(rig))
        ttnn.synchronize_device(mesh_device)
        ids = _dev0(logits.ids, mesh_device).to(torch.int64)
        vals = _dev0(logits.values, mesh_device).float()
        hid = _dev0(next_hidden, mesh_device).float()
        logits.deallocate(True)
        next_hidden.deallocate(True)
        full = _time_traced_step(mesh_device, rig, return_logits=True)
        nolm = _time_traced_step(mesh_device, rig, return_logits=False)
        out[tag] = (full, nolm, ids, vals, hid)
        n_tuned, n_shapes = rig["assistant"].mm.stats()
        logger.info(
            f"[tuned] {tag:<6} full={full:.3f} ms/step ({1e3/full:6.2f} tok/s/u)  "
            f"backbone={nolm:.3f} ms  head={full-nolm:.3f} ms  "
            f"[{n_tuned}/{n_shapes} shapes tuned]"
        )
        if tune:
            assert n_tuned > 0, "tuner enabled but no shape qualified — the config never fired"

    (fa, na, ida, va, ha), (ft, nt, idt, vt, ht) = out["auto"], out["tuned"]
    logger.info(
        f"[tuned] ===== backbone {na:.3f} -> {nt:.3f} ms ({(na-nt)/na*100:+.2f}%), "
        f"step {fa:.3f} -> {ft:.3f} ms ({(fa-ft)/fa*100:+.2f}%, "
        f"{1e3/fa:.1f} -> {1e3/ft:.1f} tok/s/u) ====="
    )

    # This config is NOT bit-neutral: it reblocks the K accumulation, so it rounds
    # differently (measured per-op: 0.99988 vs 0.99997 PCC against fp32 — both
    # correct, the tuned one marginally less so). The drafter amplifies any ULP,
    # so calibrate against a 1-bf16-ULP input perturbation exactly as
    # test_standalone_parity_dram_vs_l1 does, rather than demanding equality.
    rig = _build_standalone(mesh_device, "dram", tune_matmuls=False)
    hb = _dev0(rig["hidden"], mesh_device)
    perturbed = ((hb.view(torch.int16).int() + 1).to(torch.int16)).view(torch.bfloat16)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    rig["hidden"] = ttnn.from_torch(
        perturbed, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
    )
    lg_u, nh_u = rig["assistant"].step(*_step_args(rig))
    ttnn.synchronize_device(mesh_device)
    hu = _dev0(nh_u, mesh_device).float()

    d_tuned, d_ulp = _pcc(ha, ht), _pcc(ha, hu)
    logger.info(
        f"[tuned] auto vs tuned    : hidden_pcc={d_tuned:.6f} logits_pcc={_pcc(va, vt):.6f} ids_equal={bool(torch.equal(ida, idt))}"
    )
    logger.info(f"[tuned] auto vs +1ULP-in : hidden_pcc={d_ulp:.6f}")
    assert d_tuned >= d_ulp, (
        f"tuned config perturbs the drafter MORE than a 1-ULP input change "
        f"(pcc {d_tuned:.6f} < {d_ulp:.6f}) — that is beyond reblocking, investigate"
    )
    assert nt < na, f"tuned backbone ({nt:.3f} ms) not faster than auto ({na:.3f} ms)"


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_cme_digit_gather_vs_legacy(mesh_device, reset_seeds):
    """Base-64 digit embeddings vs the legacy ttnn.gather in the CME head.

    The digit path must produce **bit-identical** token ids — it is exact integer
    arithmetic (every digit < 64 is exact in bf16, the recombine is fp32, and the
    result needs 18 bits) — while replacing a single 5.13 ms one-core gather with
    3 embeddings plus an fp32 recombine.
    """
    prev = os.environ.get("GEMMA4_CME_GATHER")
    try:
        results = {}
        for tag, flag in (("legacy-gather", "1"), ("digit-embed", "0")):
            os.environ["GEMMA4_CME_GATHER"] = flag
            rig = _build_standalone(mesh_device, "dram")
            logits, next_hidden = rig["assistant"].step(*_step_args(rig))
            ttnn.synchronize_device(mesh_device)
            ids = _dev0(logits.ids, mesh_device).to(torch.int64)
            vals = _dev0(logits.values, mesh_device).float()
            logits.deallocate(True)
            next_hidden.deallocate(True)
            full = _time_traced_step(mesh_device, rig, return_logits=True)
            nolm = _time_traced_step(mesh_device, rig, return_logits=False)
            results[tag] = (ids, vals, full, nolm)
            logger.info(
                f"[cme] {tag:<14} full={full:.3f} ms/step ({1e3/full:6.2f} tok/s/u)  "
                f"backbone={nolm:.3f} ms  head={full-nolm:.3f} ms"
            )

        (id_g, v_g, f_g, n_g), (id_d, v_d, f_d, n_d) = results["legacy-gather"], results["digit-embed"]
        logger.info(
            f"[cme] ===== head {f_g-n_g:.3f} -> {f_d-n_d:.3f} ms ({(f_g-n_g)/(f_d-n_d):.1f}x), "
            f"step {f_g:.3f} -> {f_d:.3f} ms ({f_g/f_d:.2f}x, "
            f"{1e3/f_g:.1f} -> {1e3/f_d:.1f} tok/s/u) ====="
        )
        assert torch.equal(
            id_g, id_d
        ), f"digit path changed the candidate ids: {int((id_g != id_d).sum())} of {id_g.numel()} differ"
        assert torch.equal(v_g, v_d), "digit path changed the candidate logits"
    finally:
        os.environ.pop("GEMMA4_CME_GATHER", None)
        if prev is not None:
            os.environ["GEMMA4_CME_GATHER"] = prev


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_profile_eager_step(mesh_device, reset_seeds):
    """Eager (untraced) drafter steps, for per-op device times under the profiler.

    Run once per arm, selected with GEMMA4_L1_ARM, e.g.

        GEMMA4_L1_ARM=dram python -m tracy -r -p -v -m pytest \\
            models/.../test_assistant_standalone_l1.py::test_profile_eager_step

    then diff the two ops_perf_results CSVs. This exists separately from the
    perf test because tracy's post-processing cannot attribute ops replayed
    from a captured Metal trace ("Op N not present in cpp_device_perf_report.csv"),
    so the profiled run has to be eager. Device kernel durations are unaffected
    by tracing; only host dispatch is.
    """
    arm = os.getenv("GEMMA4_L1_ARM", "dram").strip().lower()
    rig = _build_standalone(mesh_device, "l1" if arm == "l1" else "dram")
    logger.info(f"[profile] arm={arm} pinned={rig['placement'].summary()['l1_bytes']/(1<<20):.2f} MB/device")
    for _ in range(5):
        lg, hn = rig["assistant"].step(*_step_args(rig))
        ttnn.synchronize_device(mesh_device)
        if lg is not None:
            lg.deallocate(True)
        hn.deallocate(True)


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_l1_headroom_probe(mesh_device, reset_seeds):
    """How much L1 can weights occupy before the drafter's own CBs stop fitting?

    The budget knob alone cannot answer this: the drafter's tensors are discrete
    (14 MB of transformer weights, then a single 128 MiB embed_table), so there
    is nothing to test in between. Instead keep the normal 14 MB pinned and add
    increasing filler as L1-interleaved tensors until the step fails, which it
    does with "Statically allocated circular buffers ... clash with L1 buffers".

    Reports the largest filler that still runs — i.e. the real weight budget on
    this build, model and grid, which is the number to cite rather than the raw
    L1 capacity.
    """
    rig = _build_standalone(mesh_device, "l1")
    base = rig["placement"].summary()["l1_bytes"] / (1 << 20)
    _time_traced_step(mesh_device, rig, return_logits=True, reps=5)
    logger.info(f"[headroom] baseline OK with {base:.1f} MB/device pinned")

    filler, last_ok = [], base
    for extra_mb in (16, 32, 48, 64, 80, 96, 112, 128):
        want = extra_mb - sum(t.volume() * 2 for t in filler) / (1 << 20)
        if want > 0:
            rows = max(32, int(want * (1 << 20) / 2 / 1024 / 32) * 32)
            try:
                filler.append(
                    ttnn.from_torch(
                        torch.zeros(1, 1, rows, 1024, dtype=torch.bfloat16),
                        device=mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
                        if mesh_device.get_num_devices() > 1
                        else None,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.info(f"[headroom] +{extra_mb} MB filler FAILED TO ALLOCATE: {str(e)[:160]}")
                break
        total = base + extra_mb
        try:
            ms = _time_traced_step(mesh_device, rig, return_logits=True, reps=5)
            last_ok = total
            logger.info(f"[headroom] {total:6.1f} MB/device pinned: step OK ({ms:.3f} ms)")
        except Exception as e:  # noqa: BLE001
            first = str(e).split("backtrace")[0].strip().replace("\n", " ")
            logger.info(f"[headroom] {total:6.1f} MB/device pinned: step FAILED -> {first[:220]}")
            break
    logger.info(f"[headroom] ===== largest L1 weight footprint that still runs: {last_ok:.1f} MB/device =====")
    for t in filler:
        t.deallocate(True)


# ── why L1 weights cost time: it is kernel selection, not the L1 read path ────

_MM_MC = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
_MM_INNER, _MM_REPLAYS = 20, 30


def _mm_timed(md, x_t, w_t, a, b, pc=None):
    x = ttnn.from_torch(x_t, device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=_MM_MC[a])
    w = ttnn.from_torch(w_t, device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=_MM_MC[b])
    kw = {"program_config": pc} if pc is not None else {}
    outs = [ttnn.linear(x, w, **kw) for _ in range(_MM_INNER)]
    ttnn.synchronize_device(md)
    ref = ttnn.to_torch(outs[0]).float()
    for o in outs:
        o.deallocate(True)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    outs = [ttnn.linear(x, w, **kw) for _ in range(_MM_INNER)]
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(_MM_REPLAYS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    us = (time.perf_counter() - t0) / _MM_REPLAYS / _MM_INNER * 1e6
    ttnn.release_trace(md, tid)
    for o in outs:
        o.deallocate(True)
    x.deallocate(True)
    w.deallocate(True)
    return us, ref


#: The drafter's skinny-N shapes (o_proj, down_proj, pre_projection) plus a larger control.
_MM_SHAPES = [(32, 512, 256), (32, 1024, 256), (32, 3072, 256), (32, 6144, 256)]


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_l1_penalty_is_kernel_selection(mesh_device, reset_seeds):
    """The L1-weight penalty is ttnn's matmul heuristic, NOT the L1 read path.

    Times ttnn.linear with the weight in DRAM vs L1, first letting ttnn choose
    the program config and then forcing one explicit config on both.

    MEASURED: with the auto config an L1 weight costs +72% to +93%, growing with
    K. With one fixed config the penalty VANISHES (-0.1% to -4.3%, i.e. L1 is
    equal or marginally faster) and the two outputs become bit-identical — so
    the earlier "different accumulation order" numerical difference was the same
    artifact. The fixed config is also 1.4-1.7x faster than ttnn's choice for
    the DRAM baseline, so the heuristic is leaving throughput on the table at
    these shapes regardless of where the weight lives.
    """
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    torch.manual_seed(0)
    logger.info(
        f"{'K':>6} | {'auto dram':>10}{'auto l1':>9}{'pen':>8} | {'fixed dram':>11}{'fixed l1':>10}{'pen':>8} |"
        f" {'speedup':>8} | {'pcc vs fp32':>22} | bitwise d==l1"
    )
    for M, K, N in _MM_SHAPES:
        x_t = torch.randn(1, 1, M, K).bfloat16()
        w_t = torch.randn(1, 1, K, N).bfloat16()
        gold = x_t.float() @ w_t.float()
        ad, rad = _mm_timed(mesh_device, x_t, w_t, "dram", "dram")
        al, ral = _mm_timed(mesh_device, x_t, w_t, "dram", "l1")
        fd, rfd = _mm_timed(mesh_device, x_t, w_t, "dram", "dram", pc)
        fl, rfl = _mm_timed(mesh_device, x_t, w_t, "dram", "l1", pc)
        logger.info(
            f"{K:>6} | {ad:>10.1f}{al:>9.1f}{(al-ad)/ad*100:>+7.1f}% | {fd:>11.1f}{fl:>10.1f}{(fl-fd)/fd*100:>+7.1f}% |"
            f" {ad/fd:>7.2f}x | auto {_pcc(gold,rad):.5f} fixed {_pcc(gold,rfd):.5f} |"
            f" auto={bool(torch.equal(rad,ral))} fixed={bool(torch.equal(rfd,rfl))}"
        )
        assert _pcc(gold, rfd) > 0.999 and _pcc(gold, rfl) > 0.999, "forced config lost accuracy"


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_l1_headroom_with_real_target(mesh_device, reset_seeds):
    """Usable L1 alongside the 35-layer TARGET, not just the 4-layer drafter.

    The 46 MB figure quoted elsewhere was measured with the drafter's circular
    buffers. The target's CBs are far larger, and TP changes them again (narrower
    per-device matmuls => smaller CBs), so the real weight-pinning budget for the
    target is a different number. This measures it: allocate L1-interleaved filler
    until a real target decode forward stops fitting, and report the last size
    that ran.
    """
    import math as _math

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    max_seq_len, block_size = 1024, 64
    pac = PagedAttentionConfig(block_size=block_size, max_num_blocks=_math.ceil(max_seq_len / block_size))
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=pac,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, pac)
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        ["Explain how a CPU pipeline works."], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token, anchor_pos = int(encoded[0][prefill_lens[0] - 1]), prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=3,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    def _target_step():
        lg, hid = spec._verify([anchor_token], [anchor_pos])
        ttnn.synchronize_device(mesh_device)
        hid.deallocate(True)

    _target_step()
    tp = mesh_device.shape[1] if mesh_device.get_num_devices() > 1 else 1
    logger.info(f"[tgt-headroom] tp={tp}: baseline target decode OK, adding L1 filler")

    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    filler, last_ok = [], 0
    for total_mb in (16, 32, 48, 64, 80, 96, 112, 128):
        have = sum(t.volume() * 2 for t in filler) / (1 << 20)
        want_mb = total_mb - have
        if want_mb > 0:
            rows = max(32, int(want_mb * (1 << 20) / 2 / 1024 / 32) * 32)
            try:
                filler.append(
                    ttnn.from_torch(
                        torch.zeros(1, 1, rows, 1024, dtype=torch.bfloat16),
                        device=mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        mesh_mapper=mapper,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.info(f"[tgt-headroom] {total_mb} MB: FILLER ALLOC FAILED: {str(e)[:120]}")
                break
        try:
            _target_step()
            last_ok = total_mb
            logger.info(f"[tgt-headroom] {total_mb:4d} MB pinned: target decode OK")
        except Exception as e:  # noqa: BLE001
            first = str(e).split("backtrace")[0].strip().replace("\n", " ")
            logger.info(f"[tgt-headroom] {total_mb:4d} MB pinned: FAILED -> {first[:190]}")
            break
    logger.info(f"[tgt-headroom] ===== tp={tp}: usable L1 alongside the 35-layer target = {last_ok} MB/device =====")
    for t in filler:
        t.deallocate(True)


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 2)], device_params_extra={"trace_region_size": 200_000_000})
def test_sharded_l1_weights_in_model(mesh_device, reset_seeds):
    """L1 WIDTH_SHARDED weights, in the real drafter.

    The drafter is the one model half that can take load-time sharding: it has no
    prefill path, so its weights are only ever consumed at Mt == 1, where the
    tuned program config applies. (The target reuses the same weights for prefill,
    where the tuner deliberately returns None, so a decode-shaped shard makes
    prefill's auto-selected config FATAL — see the report.)

    Its whole weight set is ~14.5 MB against a ~48 MB budget, so this is the
    fully-resident case: 100% of the drafter's pinnable weights in L1.
    """
    out = {}
    for mode in ("dram", "l1_sharded"):
        # BOTH arms get the tuned program config. l1_sharded forces it on (it
        # cannot run without it), so an untuned DRAM baseline would credit the
        # tuner's ~9% to the sharding.
        rig = _build_standalone(mesh_device, mode, tune_matmuls=True)
        a = rig["assistant"]
        types = {
            "mlp.gate_proj": a.layers[0].shared_mlp.gate_proj.memory_config(),
            "mlp.down_proj": a.layers[0].shared_mlp.down_proj.memory_config(),
            "attn.wqkv": a.layers[0].self_attn.weights.wqkv.memory_config(),
            "pre_projection": a.pre_projection.memory_config(),
        }
        for n, mc in types.items():
            logger.info(f"[sharded] {mode:<11} {n:<16} {mc.buffer_type} {mc.memory_layout}")
        full = _time_traced_step(mesh_device, rig, return_logits=True)
        nolm = _time_traced_step(mesh_device, rig, return_logits=False)
        pinned = rig["placement"].summary()["l1_bytes"] / (1 << 20)
        out[mode] = (full, nolm, types, pinned)
        logger.info(f"[sharded] {mode:<11} full={full:.3f} ms  backbone={nolm:.3f} ms  pinned={pinned:.2f} MB")

    (fd, nd, _, _), (fs, ns, ts, ps) = out["dram"], out["l1_sharded"]
    logger.info(
        f"[sharded] ===== backbone {nd:.3f} -> {ns:.3f} ms ({(nd-ns)/nd*100:+.2f}%), "
        f"step {fd:.3f} -> {fs:.3f} ms ({(fd-fs)/fd*100:+.2f}%), pinned {ps:.1f} MB ====="
    )
    n_sharded = sum(1 for mc in ts.values() if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED)
    assert n_sharded >= 3, f"expected the matmul weights WIDTH_SHARDED in L1, got {ts}"
