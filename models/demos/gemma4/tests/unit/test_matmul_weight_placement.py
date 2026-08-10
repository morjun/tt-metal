"""Where a decode matmul's weight should live: DRAM vs L1, interleaved vs sharded.

Isolated ``ttnn.linear`` at the Gemma4 drafter's and target's real decode shapes
(M = 1 tile row, batch 1). Each arm gets its own best ``in0_block_w`` (swept), so
the DRAM baseline is not handicapped; unsupported combinations report their
TT_FATAL rather than killing the run.

MEASURED (Blackhole P150, bf16) — two findings:

1. **ttnn's automatic program config is 1.4-2.4x off** on every shape here, with
   the weights in DRAM where they already are (arm a vs arm b). This has no
   capacity limit, so it applies to the 35-layer target too.

2. **An L1 weight really is ~2x faster than DRAM — but only WIDTH_SHARDED onto
   the compute grid with ``gather_in0``** (arm e), so each core holds exactly the
   N-slice it multiplies. Arm g runs the *identical* kernel and dataflow with the
   weight in DRAM and is 1.3-2.4x slower at **bit-identical PCC**: that is the
   clean isolation of locality from kernel choice. Plain ``ttnn.L1_MEMORY_CONFIG``
   (arm c, interleaved) buys nothing, because interleaved L1 is striped across all
   110 cores — remote SRAM, not a local cache.

Caveat for wiring arm e into a model: it also requires the ACTIVATION
width-sharded on the same grid. These numbers exclude the cost of getting it
there; a real integration pays a reshard this microbenchmark does not.

Arm h (DRAM WIDTH_SHARDED + gather_in0) returned pcc ~0.12-0.26 and was dropped —
most likely a malformed shard spec on our side rather than a ttnn defect, but it
was not chased down.
"""

import time

import torch
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric

INNER, REPLAYS = 20, 30
SHAPES = [
    # drafter, tp=2 per-device shapes
    ("gate/up/wqkv 256x1024", 32, 256, 1024),
    ("post_proj 256x1536", 32, 256, 1536),
    ("o_proj", 32, 512, 256),
    ("down_proj", 32, 1024, 256),
    ("pre_proj", 32, 3072, 256),
    # target-scale
    ("tgt 1536x1536", 32, 1536, 1536),
    ("tgt 1536x3072", 32, 1536, 3072),
]


def _pcc(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _divisors_upto(n, cap=8):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0] or [1]


def _pick_grid(Nt, Kt=None, max_x=8, max_y=8):
    """Largest core rectangle whose core count divides Nt (and Kt, if given).

    gather_in0 width-shards the ACTIVATION on K too, so the core count must also
    divide Kt — otherwise the skinny-K shapes (gate/up/wqkv have Kt=8) get a grid
    they cannot satisfy and the sharded arms are silently skipped.
    """
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            c = gx * gy
            if Nt % c or (Kt is not None and Kt % c):
                continue
            if c > best[0] * best[1]:
                best = (gx, gy)
    return best


def _grid_set(gx, gy):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def _sharded(md, t, buffer_type, grid, shard_hw):
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type,
        ttnn.ShardSpec(grid, shard_hw, ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(t, device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=mc)


def _time(md, fn):
    outs = [fn() for _ in range(INNER)]
    ttnn.synchronize_device(md)
    ref = ttnn.to_torch(outs[0]).float()
    for o in outs:
        o.deallocate(True)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    outs = [fn() for _ in range(INNER)]
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    us = (time.perf_counter() - t0) / REPLAYS / INNER * 1e6
    ttnn.release_trace(md, tid)
    for o in outs:
        o.deallocate(True)
    return us, ref


def _try(md, build, gold):
    """Run one arm; return (us, pcc) or (None, error)."""
    tensors = []
    try:
        tensors, fn = build()
        us, ref = _time(md, fn)
        return us, _pcc(gold, ref)
    except Exception as ex:  # noqa: BLE001
        msg = str(ex).split("backtrace")[0].strip().replace("\n", " ")
        return None, msg[:130]
    finally:
        for t in tensors:
            try:
                t.deallocate(True)
            except Exception:  # noqa: BLE001
                pass


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_sharded_l1_weight(mesh_device, reset_seeds):
    torch.manual_seed(0)
    T = ttnn.TILE_SIZE
    for name, M, K, N in SHAPES:
        Mt, Kt, Nt = M // T, K // T, N // T
        gx, gy = _pick_grid(Nt, Kt)
        cores = gx * gy
        grid = _grid_set(gx, gy)
        gsize = ttnn.CoreCoord(gx, gy)
        per_core_N = Nt // cores
        Kt_per_core = Kt // cores if Kt % cores == 0 else None

        x_t = torch.randn(1, 1, M, K).bfloat16()
        w_t = torch.randn(1, 1, K, N).bfloat16()
        gold = x_t.float() @ w_t.float()

        def pc1d(blk, mcast=True, gather=False):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=gsize,
                in0_block_w=blk,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=Mt,
                per_core_N=per_core_N,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=mcast,
                gather_in0=gather,
            )

        out_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [M, N // cores], ttnn.ShardOrientation.ROW_MAJOR),
        )

        def dram(t):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def l1i(t):
            return ttnn.from_torch(
                t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        logger.info(f"[shard] === {name}: M={M} K={K} N={N} | grid {gx}x{gy}={cores} per_core_N={per_core_N}")

        # a: today's baseline, auto config, everything DRAM interleaved
        us_a, pcc_a = _try(
            mesh_device, lambda: ((lambda x, w: ([x, w], lambda: ttnn.linear(x, w)))(dram(x_t), dram(w_t))), gold
        )
        logger.info(f"[shard]    a dram/auto            {us_a if us_a is None else f'{us_a:8.2f} us'}  pcc={pcc_a}")

        # b: DRAM interleaved, fixed 1D config, in0_block_w swept
        best_b, best_blk = None, None
        for blk in _divisors_upto(Kt):
            us, pc = _try(
                mesh_device,
                (
                    lambda b: lambda: (
                        (lambda x, w: ([x, w], lambda: ttnn.linear(x, w, program_config=pc1d(b))))(dram(x_t), dram(w_t))
                    )
                )(blk),
                gold,
            )
            if us is not None and (best_b is None or us < best_b):
                best_b, best_blk, best_pcc = us, blk, pc
        logger.info(
            f"[shard]    b dram/fixed           {best_b:8.2f} us  pcc={best_pcc:.5f}  (best in0_block_w={best_blk})"
        )

        # c: L1 interleaved weight, same fixed config
        us_c, pcc_c = _try(
            mesh_device,
            lambda: (
                (lambda x, w: ([x, w], lambda: ttnn.linear(x, w, program_config=pc1d(best_blk))))(dram(x_t), l1i(w_t))
            ),
            gold,
        )
        _rel = lambda u: f"{(u-best_b)/best_b*100:+6.1f}% vs b" if u is not None else ""
        logger.info(
            f"[shard]    c L1 interleaved       {us_c if us_c is None else f'{us_c:8.2f} us'}  pcc={pcc_c}  {_rel(us_c)}"
        )

        # d: L1 WIDTH_SHARDED in1 + in0, mcast_in0. in0_block_w must divide Kt_per_core.
        if Kt_per_core:
            best_d, best_d_blk, best_d_pcc = None, None, None
            for blk in _divisors_upto(Kt_per_core):

                def build(b=blk):
                    x = _sharded(mesh_device, x_t, ttnn.BufferType.L1, grid, [M, K // cores])
                    w = _sharded(mesh_device, w_t, ttnn.BufferType.L1, grid, [K, N // cores])
                    return [x, w], (lambda: ttnn.linear(x, w, program_config=pc1d(b), memory_config=out_mc))

                us, pc = _try(mesh_device, build, gold)
                if us is not None and (best_d is None or us < best_d):
                    best_d, best_d_blk, best_d_pcc = us, blk, pc
            if best_d is None:
                logger.info(f"[shard]    d L1 shard/mcast       FAILED: {pc}")
            else:
                logger.info(
                    f"[shard]    d L1 shard/mcast       {best_d:8.2f} us  pcc={best_d_pcc:.5f}  {_rel(best_d)}  (blk={best_d_blk})"
                )

            # e: same tensors, gather_in0 (in0_block_w = Kt per core)
            def build_e():
                x = _sharded(mesh_device, x_t, ttnn.BufferType.L1, grid, [M, K // cores])
                w = _sharded(mesh_device, w_t, ttnn.BufferType.L1, grid, [K, N // cores])
                return [x, w], (
                    lambda: ttnn.linear(
                        x, w, program_config=pc1d(Kt_per_core, mcast=False, gather=True), memory_config=out_mc
                    )
                )

            us_e, pcc_e = _try(mesh_device, build_e, gold)
            logger.info(
                f"[shard]    e L1 shard/gather      {us_e if us_e is None else f'{us_e:8.2f} us'}  pcc={pcc_e}  {_rel(us_e)}"
            )

            # f: DRAM WIDTH_SHARDED weight
            def build_f():
                dc = mesh_device.dram_grid_size().x
                n_per = ((Nt + dc - 1) // dc) * T
                dgrid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dc - 1, 0))})
                x = _sharded(mesh_device, x_t, ttnn.BufferType.L1, grid, [M, K // cores])
                w = _sharded(mesh_device, w_t, ttnn.BufferType.DRAM, dgrid, [K, n_per])
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=Kt_per_core, per_core_M=Mt, per_core_N=(Nt + dc - 1) // dc, fused_activation=None
                )
                return [x, w], (lambda: ttnn.linear(x, w, program_config=pc, memory_config=out_mc))

            us_f, pcc_f = _try(mesh_device, build_f, gold)
            logger.info(
                f"[shard]    f DRAM shard           {us_f if us_f is None else f'{us_f:8.2f} us'}  pcc={pcc_f}  {_rel(us_f)}"
            )

            # g/h: SAME gather_in0 dataflow as e, but the weight in DRAM. This is
            # the clean isolation of "L1 vs DRAM" at identical kernel + dataflow:
            # if e beats these, the win is genuine SRAM locality, not the ring.
            def build_g():
                x = _sharded(mesh_device, x_t, ttnn.BufferType.L1, grid, [M, K // cores])
                w = dram(w_t)
                return [x, w], (
                    lambda: ttnn.linear(
                        x, w, program_config=pc1d(Kt_per_core, mcast=False, gather=True), memory_config=out_mc
                    )
                )

            us_g, pcc_g = _try(mesh_device, build_g, gold)
            logger.info(
                f"[shard]    g gather, in1 DRAM int {us_g if us_g is None else f'{us_g:8.2f} us'}  pcc={pcc_g}  {_rel(us_g)}"
            )

            if us_e is not None and us_g is not None:
                logger.info(
                    f"[shard]    >>> SRAM locality effect at fixed dataflow: e/g = {us_e/us_g:.3f} "
                    f"({(us_e-us_g)/us_g*100:+.1f}%)"
                )
                # e and g differ ONLY in where the weight lives, so their outputs
                # must be identical; anything else means the arms are not comparable.
                assert (
                    abs(pcc_e - pcc_g) < 1e-9
                ), f"{name}: e and g should compute identically, got pcc {pcc_e} vs {pcc_g}"
                assert us_e < us_g, f"{name}: L1-sharded weight ({us_e:.2f} us) not faster than DRAM ({us_g:.2f} us)"
            assert best_b < us_a, f"{name}: fixed program config ({best_b:.2f} us) not faster than auto ({us_a:.2f} us)"
            for tag, p in (("b", best_pcc), ("e", pcc_e), ("g", pcc_g)):
                if isinstance(p, float):
                    assert p > 0.999, f"{name}: arm {tag} lost accuracy vs fp32 (pcc={p})"
        else:
            logger.info(f"[shard]    d/e/f SKIP (Kt={Kt} not divisible by {cores} cores)")


# ── does sharding ONLY the weight work, with no activation reshard? ───────────

_NR_SHAPES = [
    ("o_proj", 32, 512, 256),
    ("down_proj", 32, 1024, 256),
    ("pre_proj", 32, 3072, 256),
    ("tgt 1536x1536", 32, 1536, 1536),
]
_NR_INNER, _NR_REPLAYS = INNER, REPLAYS


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)], device_params_extra={"trace_region_size": 200_000_000})
def test_weight_sharded_without_reshard(mesh_device, reset_seeds):
    """Shard the WEIGHT into L1 and leave the activation interleaved — no reshard.

    The `gather_in0` arm of the test above needs the activation width-sharded on
    the same grid, which a model would have to produce. This asks whether the
    cheaper thing — change only the weight's memory config at load time, keep
    `mcast_in0` and an interleaved activation — is worth anything.

    MEASURED: at the drafter's skinny-N shapes, essentially nothing (0-3%); at
    target-scale N=1536, **-38%** (and -45% if the output is sharded too). So the
    weight-only change is free to implement and pays off exactly where N is wide
    enough that each core reads a meaningful slice of the weight. The larger 2x at
    skinny N needs gather_in0, i.e. a sharded activation.
    """
    torch.manual_seed(0)
    T = ttnn.TILE_SIZE
    for name, M, K, N in _NR_SHAPES:
        Mt, Kt, Nt = M // T, K // T, N // T
        cores = 8 if Nt % 8 == 0 else Nt
        gx, gy = (cores, 1) if cores <= 8 else (8, cores // 8)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        x_t = torch.randn(1, 1, M, K).bfloat16()
        w_t = torch.randn(1, 1, K, N).bfloat16()
        gold = x_t.float() @ w_t.float()
        blk = max(d for d in range(1, 9) if Kt % d == 0)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=blk,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=Mt,
            per_core_N=Nt // cores,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        def dram(t):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def wsh(t, buf):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    buf,
                    ttnn.ShardSpec(grid, [K, N // cores], ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )

        logger.info(f"[nr] === {name} K={K} N={N} grid {gx}x{gy}")
        for tag, build in (
            (
                "base: in0 DRAM int, in1 DRAM int",
                lambda: (lambda x, w: ([x, w], lambda: ttnn.linear(x, w, program_config=pc)))(dram(x_t), dram(w_t)),
            ),
            (
                "NO-RESHARD: in0 DRAM int, in1 L1 WIDTH_SHARDED",
                lambda: (lambda x, w: ([x, w], lambda: ttnn.linear(x, w, program_config=pc)))(
                    dram(x_t), wsh(w_t, ttnn.BufferType.L1)
                ),
            ),
            (
                "NO-RESHARD + sharded out",
                lambda: (
                    lambda x, w: (
                        [x, w],
                        lambda: ttnn.linear(
                            x,
                            w,
                            program_config=pc,
                            memory_config=ttnn.MemoryConfig(
                                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                ttnn.BufferType.L1,
                                ttnn.ShardSpec(grid, [M, N // cores], ttnn.ShardOrientation.ROW_MAJOR),
                            ),
                        ),
                    )
                )(dram(x_t), wsh(w_t, ttnn.BufferType.L1)),
            ),
        ):
            us, extra = _try(mesh_device, build, gold)
            logger.info(f"[nr]    {tag:<48} " + (f"FAILED: {extra}" if us is None else f"{us:8.2f} us pcc={extra:.5f}"))
