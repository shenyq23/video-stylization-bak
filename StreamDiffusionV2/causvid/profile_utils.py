"""Lightweight per-submodule timer for MotionFlow profiling.

Usage from gpu.py:
    pipeline.generator.profile_timings = timings_dict
    pipeline.vae.model.profile_timings = timings_dict
    pipeline.inference_stream(...)
    # timings_dict now has entries like {"DiT/Self Attn": 12.3, ...}

The model's forward() must call `set_active_timings(self.profile_timings)` at
entry and `clear_active_timings()` on exit. Sub-modules (attention, conv,
warp etc.) call `time_block("DiT/Self Attn")` as a context manager.

Each `time_block` records CUDA events and forces an event synchronize before
recording the elapsed ms, so timings are clean even though kernels are
otherwise async. This adds ~10-50us overhead per block — only active when
profiling is enabled.
"""
import contextlib

import torch

_ACTIVE_TIMINGS = None


def set_active_timings(timings):
    """Install the dict that subsequent `time_block` calls will accumulate into.
    Pass None (or call clear_active_timings) to disable timing."""
    global _ACTIVE_TIMINGS
    _ACTIVE_TIMINGS = timings


def clear_active_timings():
    global _ACTIVE_TIMINGS
    _ACTIVE_TIMINGS = None


def get_active_timings():
    return _ACTIVE_TIMINGS


def begin_segment(key):
    """Imperative counterpart to `time_block` for code where wrapping in a
    context manager would require large indentation changes. Returns a handle
    to pass to `end_segment`. Both calls are no-ops if profiling is off.
    """
    timings = _ACTIVE_TIMINGS
    if timings is None:
        return None
    start_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    return (key, start_evt, timings)


def end_segment(handle):
    if handle is None:
        return
    key, start_evt, timings = handle
    end_evt = torch.cuda.Event(enable_timing=True)
    end_evt.record()
    end_evt.synchronize()
    timings[key] = timings.get(key, 0.0) + start_evt.elapsed_time(end_evt)


@contextlib.contextmanager
def time_block(key):
    """Accumulate the wrapped block's GPU time into the active timings dict.

    No-op if no timings dict is active.
    Safe to nest, but nested blocks measure the same wall time; prefer flat
    instrumentation when you want sub-sums to match the parent.
    """
    timings = _ACTIVE_TIMINGS
    if timings is None:
        yield
        return
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    try:
        yield
    finally:
        end_evt.record()
        end_evt.synchronize()
        timings[key] = timings.get(key, 0.0) + start_evt.elapsed_time(end_evt)
