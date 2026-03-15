import atexit
import json
import os
import threading
import time
from contextlib import contextmanager

_LOCK = threading.Lock()
_REGISTERED = False
_STATS = {}


def enabled():
    return os.getenv("TT_L1_KV_PERF", "0") == "1"


def _bucket(name):
    bucket = _STATS.get(name)
    if bucket is None:
        bucket = {"count": 0, "sum": 0.0, "min": None, "max": None}
        _STATS[name] = bucket
    return bucket


def add_sample(name, value):
    if not enabled():
        return
    with _LOCK:
        bucket = _bucket(name)
        bucket["count"] += 1
        bucket["sum"] += float(value)
        bucket["min"] = float(value) if bucket["min"] is None else min(bucket["min"], float(value))
        bucket["max"] = float(value) if bucket["max"] is None else max(bucket["max"], float(value))


def add_duration(name, duration_s):
    add_sample(name, duration_s)


def increment(name, value=1):
    add_sample(name, value)


@contextmanager
def timed(name):
    if not enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        add_duration(name, time.perf_counter() - start)


def reset():
    with _LOCK:
        _STATS.clear()


def snapshot():
    with _LOCK:
        report = {}
        for name, bucket in sorted(_STATS.items()):
            count = bucket["count"]
            avg = bucket["sum"] / count if count else 0.0
            report[name] = {
                "count": count,
                "sum": bucket["sum"],
                "avg": avg,
                "min": bucket["min"],
                "max": bucket["max"],
                "sum_ms": bucket["sum"] * 1000.0,
                "avg_ms": avg * 1000.0,
                "min_ms": None if bucket["min"] is None else bucket["min"] * 1000.0,
                "max_ms": None if bucket["max"] is None else bucket["max"] * 1000.0,
            }
        return report


def dump(path=None):
    report_path = path or os.getenv("TT_L1_KV_PERF_REPORT")
    if not report_path:
        return None
    report = {"generated_at_epoch_s": time.time(), "stats": snapshot()}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report_path


def _dump_on_exit():
    if enabled():
        dump()


def register_atexit_dump():
    global _REGISTERED
    if _REGISTERED:
        return
    atexit.register(_dump_on_exit)
    _REGISTERED = True


register_atexit_dump()
