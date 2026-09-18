# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Two knobs for speculative decoding: WHICH PLI implementation, and WHICH route.

They used to be three knobs that were really one-and-a-half, and the confusion was
structural rather than cosmetic:

    GEMMA4_DECODE_PLI_DEV      host/device PLI in plain decode
    GEMMA4_SPEC_PLI_DEV        host/device PLI in the host-loop packed verify
    GEMMA4_SPEC_FUSED_PLI_DEV  ... and in the fused trace, EXCEPT that the fused trace
                               cannot do host PLI at all, so setting it to 0 does not
                               change an implementation -- it changes the ROUTE.

Two consequences, both of which this module removes.

1. The first two are the SAME decision applied at two call sites, and they must agree:
   device and host PLI differ by ~1 bf16 ULP (PCC 0.9999947), so any spec-vs-plain number
   taken with them disagreeing also measures that gap and is not comparable to the record.
   ``GEMMA4_PLI`` is one value for every route, so the invariant holds by construction
   instead of by discipline.

2. The third was named for a mechanism and observable only as a policy. Route selection is
   now ``GEMMA4_SPEC_ROUTE``, which already existed for a narrower job (choosing between
   the two fused bodies on a non-PLI target) and is extended here rather than duplicated.

Every old name still works and is mapped below, because `MEASUREMENT_RECORD.md` §2.2
requires the knob state of a recorded number to stay executable. They warn once.

    GEMMA4_PLI          = device (default) | host
    GEMMA4_SPEC_ROUTE   = auto (default) | host-loop | fused-packed | fused-batch-dim
"""

import os

from loguru import logger

PLI_ENV = "GEMMA4_PLI"
ROUTE_ENV = "GEMMA4_SPEC_ROUTE"

#: Old per-route PLI names -> the route they governed, for the deprecation message.
LEGACY_PLI_ENVS = {
    "GEMMA4_DECODE_PLI_DEV": "plain decode",
    "GEMMA4_SPEC_PLI_DEV": "the host-loop packed verify",
}
#: Old route selector. ``=0`` meant "do not take the fused route", i.e. the host loop.
LEGACY_ROUTE_ENV = "GEMMA4_SPEC_FUSED_PLI_DEV"

ROUTES = ("auto", "host-loop", "fused-packed", "fused-batch-dim")
#: The value this knob shipped with before it covered PLI targets.
ROUTE_ALIASES = {"fused-batched": "fused-packed"}

#: Routes whose PLI is necessarily on device: the drafter's candidate ids are argmaxed and
#: re-embedded in-graph and never reach the host, so there is nothing to build host PLI from.
DEVICE_PLI_ROUTES = ("fused-packed", "fused-batch-dim")

_warned = set()


def _warn_once(old, new):
    if old in _warned:
        return
    _warned.add(old)
    logger.warning(f"{old} is DEPRECATED and maps to {new}. It still works; prefer the new name.")


def pli_on_device(legacy_env=None):
    """True if PLI should be computed on device.

    ``legacy_env`` is the deprecated per-route name this call site used to read. An
    explicitly set legacy name WINS, so any recorded configuration reproduces exactly --
    including a mixed one, which ``SpeculativeDecoder._assert_consistent_pli`` refuses
    unless ``GEMMA4_PLI_ALLOW_MIXED=1``.
    """
    if legacy_env is not None:
        raw = os.environ.get(legacy_env)
        if raw is not None:
            _warn_once(legacy_env, f"{PLI_ENV}={'device' if raw == '1' else 'host'}")
            return raw == "1"
    value = os.environ.get(PLI_ENV, "device").strip().lower()
    if value not in ("device", "host"):
        raise ValueError(f"{PLI_ENV}={value!r} is not valid. Use 'device' (default) or 'host'.")
    return value == "device"


def spec_route():
    """The speculative iteration structure to run: one of ``ROUTES``.

    ``auto`` defers to the target: a per-layer-input target (E2B/E4B) takes the packed
    fused trace, anything else keeps upstream's batch-dim fused body. That is the shipping
    behaviour and is what every current default number was taken under.
    """
    raw = os.environ.get(ROUTE_ENV)
    legacy = os.environ.get(LEGACY_ROUTE_ENV)

    if raw is None:
        if legacy is None:
            return "auto"
        # =1 was "the fused route is allowed", which is what auto already does.
        route = "auto" if legacy == "1" else "host-loop"
        _warn_once(LEGACY_ROUTE_ENV, f"{ROUTE_ENV}={route}")
        return route

    route = ROUTE_ALIASES.get(raw.strip().lower(), raw.strip().lower())
    if route not in ROUTES:
        raise ValueError(f"{ROUTE_ENV}={raw!r} is not valid. Use one of: {', '.join(ROUTES)}.")
    if legacy is not None:
        implied = "auto" if legacy == "1" else "host-loop"
        if implied != route and not (legacy == "1" and route != "host-loop"):
            raise ValueError(
                f"{ROUTE_ENV}={route!r} contradicts {LEGACY_ROUTE_ENV}={legacy!r} "
                f"(which means {implied!r}). Set only {ROUTE_ENV}."
            )
    return route


def resolve_route(route, target_has_pli, pli_device):
    """``auto`` -> a concrete route, and refuse a route that cannot honour the PLI choice.

    Host PLI and a fused route are mutually exclusive by construction (see
    ``DEVICE_PLI_ROUTES``). Under ``auto`` that is resolved silently in favour of the host
    loop; asked for explicitly it raises, because silently ignoring the route someone named
    is how a number gets filed against the wrong configuration.
    """
    if route == "auto":
        if not pli_device and target_has_pli:
            return "host-loop"
        return "fused-packed" if target_has_pli else "fused-batch-dim"
    if route in DEVICE_PLI_ROUTES and not pli_device and target_has_pli:
        raise ValueError(
            f"{ROUTE_ENV}={route} needs PLI on device -- the fused trace's candidate ids "
            f"never reach the host, so it has no host-PLI option -- but {PLI_ENV}=host was "
            f"requested. Use {ROUTE_ENV}=host-loop, or drop {PLI_ENV}=host."
        )
    return route
