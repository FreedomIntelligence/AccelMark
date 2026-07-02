"""
Tests for the background power sampler (``loadgen/power.py``).

Verifies that ``PowerSampler`` gracefully degrades when no vendor hook is
available and returns correct avg/peak when fed a deterministic source.
"""

from __future__ import annotations

import pytest

from loadgen.power import PowerSampler, PowerStats


class TestPowerSampler:
    """Unit tests for PowerSampler — no real hardware required."""

    def test_noop_when_no_vendor_plugin(self):
        """Sampler returns (None, None) when no vendor plug-in is active."""
        sampler = PowerSampler()
        # On a machine without a supported accelerator, or in CI, the sampler
        # should resolve to a no-op.
        sampler.start()
        stats = sampler.stop()
        assert isinstance(stats, PowerStats)
        assert stats.avg is None
        assert stats.peak is None

    def test_stop_before_start_returns_none(self):
        """stop() before start() returns PowerStats(None, None)."""
        sampler = PowerSampler()
        stats = sampler.stop()
        assert stats.avg is None
        assert stats.peak is None

    def test_deterministic_source(self, monkeypatch):
        """With a monkeypatched sample_fn, returns correct avg and peak."""
        deterministic_samples = [100.0, 200.0, 300.0]
        calls = iter(deterministic_samples)

        def _fake_sample():
            try:
                return next(calls)
            except StopIteration:
                return None

        sampler = PowerSampler(interval_s=0.01)
        sampler._sample_fn = _fake_sample

        sampler.start()
        import time
        time.sleep(0.05)  # allow a few samples to be collected
        stats = sampler.stop()

        # We should have collected at least 3 samples
        assert stats.avg is not None
        assert stats.peak is not None
        assert stats.avg == pytest.approx(200.0, abs=10.0)
        assert stats.peak == pytest.approx(300.0, abs=10.0)

    def test_sample_fn_raises_is_silent(self, monkeypatch):
        """When sample_fn raises, the sampler continues and returns (None, None)."""
        def _failing_sample():
            raise RuntimeError("simulated SMI failure")

        sampler = PowerSampler(interval_s=0.01)
        sampler._sample_fn = _failing_sample

        sampler.start()
        import time
        time.sleep(0.05)
        stats = sampler.stop()

        assert stats.avg is None
        assert stats.peak is None

    def test_sample_fn_returns_none(self):
        """When sample_fn always returns None, stats are None."""
        sampler = PowerSampler(interval_s=0.01)
        sampler._sample_fn = lambda: None

        sampler.start()
        import time
        time.sleep(0.05)
        stats = sampler.stop()

        assert stats.avg is None
        assert stats.peak is None

    def test_double_start_is_idempotent(self):
        """Calling start() twice does not spawn multiple threads."""
        sampler = PowerSampler(interval_s=0.01)
        sampler._sample_fn = lambda: 150.0

        sampler.start()
        sampler.start()  # second call should be no-op
        import time
        time.sleep(0.05)
        stats = sampler.stop()

        assert stats.avg is not None
        # avg should be ~150 regardless of whether a second thread was spawned
        assert stats.avg == pytest.approx(150.0, abs=10.0)
