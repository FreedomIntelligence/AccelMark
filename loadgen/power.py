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
        self._sample_fn = None  # callable returning float | None
        self._thread: Optional[threading.Thread] = None
        self._samples: list[float] = []
        self._running = False
        self._lock = threading.Lock()

        # Resolve the active vendor's sample_power_watts() once.
        self._sample_fn = self._resolve_sample_fn()

    @staticmethod
    def _resolve_sample_fn():
        """Return the active plug-in's ``sample_power_watts`` callable,
        or None if no vendor plug-in provides one."""
        # Ensure the runners package is importable
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))

        try:
            from runners.platforms import get_active_plugin

            plugin = get_active_plugin()
            if plugin is None:
                return None
            fn = getattr(plugin, "sample_power_watts", None)
            if fn is None:
                return None
            # Verify the function actually returns something on a test call
            try:
                test_val = fn()
                if test_val is None:
                    # Not an error — the function exists but reports no power
                    pass
            except Exception:
                # Function exists but fails at runtime — still use it;
                # the sampling thread handles per-call exceptions
                pass
            return fn
        except Exception:
            return None

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
                # Per-sample failures are silently dropped — a single bad
                # reading must not perturb the aggregate
                pass
            time.sleep(self._interval_s)
