"""
Background power sampler for AccelMark.

Resolves the active vendor platform plug-in once, then polls its
``sample_power_watts()`` function on a daemon thread between
``start()`` and ``stop()``. All failures degrade gracefully to
``(None, None)`` so power measurement is purely additive — the
benchmark always runs to completion unchanged.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

# Repo root for sys.path resolution
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent

# Default sampling interval in seconds
DEFAULT_SAMPLE_INTERVAL_S = 0.1

# Module-level cache for the resolved sample function — the active vendor
# does not change during a benchmark run, so we resolve it once and reuse.
_cached_sample_fn: Optional[callable] = None
_cached_sample_fn_resolved: bool = False


@dataclass
class PowerStats:
    """Aggregated power samples from a measurement window."""

    avg: Optional[float] = None
    """Mean power in watts over the sampled window, or None if no samples."""

    peak: Optional[float] = None
    """Maximum power in watts observed, or None if no samples."""


class PowerSampler:
    """
    Background power sampler for a benchmark scenario's timed window.

    Resolves the active vendor plug-in's ``sample_power_watts()`` once at
    construction time, then polls it on a daemon thread between ``start()``
    and ``stop()``. If no vendor plug-in exports a usable power function,
    the sampler is a no-op and ``stop()`` returns ``PowerStats(None, None)``.

    Usage::

        sampler = PowerSampler()
        sampler.start()
        # ... timed benchmark window ...
        stats: PowerStats = sampler.stop()
        print(stats.avg, stats.peak)
    """

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S):
        self._interval_s = interval_s
        self._sample_fn = _get_sample_fn()
        self._thread: Optional[threading.Thread] = None
        self._samples: list[float] = []
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Begin sampling power in the background. No-op if no source is available."""
        if self._sample_fn is None:
            return
        if self._running:
            return
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> PowerStats:
        """Stop sampling, join the background thread, and return aggregated stats."""
        if not self._running:
            return PowerStats()

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            samples = list(self._samples)
            self._samples = []

        if not samples:
            return PowerStats()

        avg = round(sum(samples) / len(samples), 2)
        peak = round(max(samples), 2)
        return PowerStats(avg=avg, peak=peak)

    def _run(self) -> None:
        """Daemon thread loop — polls the power source at the configured interval."""
        while self._running:
            try:
                val = self._sample_fn()
                if val is not None and val > 0:
                    with self._lock:
                        self._samples.append(float(val))
            except Exception:
                pass
            time.sleep(self._interval_s)


# ── Module-level resolution (cached — called once per process) ──────────────

def _get_sample_fn():
    """Return the active plug-in's ``sample_power_watts`` callable,
    or None if no vendor plug-in provides one. Result is cached at
    module level — vendor detection subprocesses run at most once."""
    global _cached_sample_fn, _cached_sample_fn_resolved
    if _cached_sample_fn_resolved:
        return _cached_sample_fn

    _cached_sample_fn_resolved = True

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    try:
        from runners.platforms import get_active_plugin

        plugin = get_active_plugin()
        if plugin is None:
            _cached_sample_fn = None
            return None
        fn = getattr(plugin, "sample_power_watts", None)
        if fn is None:
            _cached_sample_fn = None
            return None
        _cached_sample_fn = fn
        return fn
    except Exception:
        _cached_sample_fn = None
        return None
