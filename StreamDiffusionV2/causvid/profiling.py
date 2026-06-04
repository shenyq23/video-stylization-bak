"""Lightweight, opt-in CUDA-event profiler for per-submodule timing.

Design goals:
  * Zero behavioral / perf impact when disabled (``PROFILER.enabled is False``):
    ``record()`` returns a shared no-op context manager, so the only cost on the
    normal inference path is one attribute check + one cheap object return per call.
  * Low distortion when enabled: we only *record* CUDA events around each region
    (no synchronization between regions). The single ``torch.cuda.synchronize()``
    happens once per iteration in ``collect_iter()``, after which the elapsed
    times are read back and accumulated by region name.

Usage (see streamv2v/profile_motionflow.py):

    from causvid.profiling import PROFILER
    PROFILER.enabled = True
    ...
    with PROFILER.record("Self Attention"):
        y = self.self_attn(...)
    ...
    per_region_ms = PROFILER.collect_iter()   # dict: name -> ms for this iter

Multiple ``record(name)`` regions sharing a name are summed within an iteration,
which is exactly what we want for "all self-attentions across all blocks/steps".
"""

from collections import defaultdict

import torch


class _NullRegion:
    """A do-nothing context manager used when profiling is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_REGION = _NullRegion()


class _Region:
    """Records a CUDA start/end event pair around a code region."""

    __slots__ = ("prof", "name", "start")

    def __init__(self, prof, name):
        self.prof = prof
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.prof._events.append((self.name, self.start, end))
        return False


class Profiler:
    def __init__(self):
        self.enabled = False
        # Pending (name, start_event, end_event) tuples for the current iteration.
        self._events = []
        # name -> list of per-iteration millisecond totals.
        self.records = defaultdict(list)

    def record(self, name):
        """Return a context manager timing the wrapped region (no-op if disabled)."""
        if not self.enabled:
            return _NULL_REGION
        return _Region(self, name)

    def collect_iter(self):
        """Synchronize once, aggregate this iteration's regions by name, store them.

        Returns the per-name millisecond dict for the iteration just finished.
        """
        if not self._events:
            return {}
        torch.cuda.synchronize()
        agg = defaultdict(float)
        for name, start, end in self._events:
            agg[name] += start.elapsed_time(end)
        self._events.clear()
        for name, ms in agg.items():
            self.records[name].append(ms)
        return dict(agg)

    def reset(self):
        self._events.clear()
        self.records.clear()

    def summary(self):
        """Return {name: mean_ms} averaged over all collected iterations."""
        return {
            name: (sum(vals) / len(vals) if vals else 0.0)
            for name, vals in self.records.items()
        }


# Process-wide singleton shared by the instrumented model code.
PROFILER = Profiler()
