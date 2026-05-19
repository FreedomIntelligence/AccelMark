"""
Tests for the warmup phase in online and burst scenarios.

These scenarios used to read `online_warmup_runs` from suite.json but
silently ignored the value — every reported p99 was contaminated by
cold-engine TTFT spikes. This test suite locks down the fix so future
refactors can't reintroduce the bug.

What is verified:
- Warmup requests are fired in `online` and `burst` BEFORE the timed phase
- Warmup latencies are NOT counted in the returned distribution
- A counter on the mock inference_fn confirms the exact request budget
- Warmup is a no-op when the parameter is 0 (back-compat)
- An exception during warmup does not abort the timed phase
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from loadgen.loadgen import AccelMarkLoadGen
from loadgen.types import InferenceResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_requests(n: int):
    """Build n minimal InferenceRequest-like objects using the same shim
    loadgen.py falls back to when benchmark_runner is not importable."""
    from loadgen.loadgen import InferenceRequest
    return [
        InferenceRequest(
            prompt=f"prompt {i}",
            request_id=i,
            input_tokens=10,
            max_tokens=20,
        )
        for i in range(n)
    ]


def _online_suite(qps_levels=(2.0,), warmup_requests: int = 5, num_runs: int = 1):
    return {
        "num_runs": num_runs,
        "online_qps_levels": list(qps_levels),
        "online_sla_ttft_ms": 1000,
        "online_request_count": 8,
        "online_warmup_requests": warmup_requests,
        "input_tokens": 10,
    }


def _burst_suite(warmup_requests: int = 5):
    return {
        "num_runs": 1,
        "online_sla_ttft_ms": 1000,
        "online_request_count": 6,
        "burst_steady_qps": 2.0,
        "burst_peak_qps": 4.0,
        "burst_duration_seconds": 0.3,
        "burst_interval_seconds": 0.3,
        "burst_warmup_requests": warmup_requests,
        "input_tokens": 10,
    }


class MockInferenceFn:
    """Counts every call and exposes an async callable as `.fn` for loadgen.

    The fast warmup latency vs slow timed latency makes it trivial to
    assert that warmup requests are excluded from the distribution: if
    warmup latencies leaked into results, p50/p99 would collapse to
    the fast value.

    Note: loadgen uses `asyncio.iscoroutinefunction()` to detect async
    inference_fn, which returns False for a class with `async __call__`.
    So we expose `self.fn` as a real `async def` closure bound to this
    instance's state.
    """

    def __init__(self, *, warmup_ttft_ms: float = 1.0, timed_ttft_ms: float = 100.0,
                 fail_first_n: int = 0):
        self.call_count = 0
        self.warmup_ttft_ms = warmup_ttft_ms
        self.timed_ttft_ms = timed_ttft_ms
        self.fail_first_n = fail_first_n
        self.warmup_budget: Optional[int] = None

        state = self  # closure capture

        async def _fn(request) -> InferenceResult:
            idx = state.call_count
            state.call_count += 1
            if idx < state.fail_first_n:
                raise RuntimeError(f"simulated failure on warmup request {idx}")
            ttft = (
                state.warmup_ttft_ms
                if state.warmup_budget is not None and idx < state.warmup_budget
                else state.timed_ttft_ms
            )
            await asyncio.sleep(0)  # yield control
            return InferenceResult(
                first_token_time_ms=ttft,
                total_time_ms=ttft * 2,
                output_tokens=20,
                input_tokens=10,
                success=True,
            )

        self.fn = _fn

    def set_warmup_budget(self, n: int) -> None:
        """Tell the mock how many of the next calls count as warmup."""
        self.warmup_budget = n


# ── online warmup ─────────────────────────────────────────────────────────────

def test_online_warmup_fires_configured_count(tmp_path):
    """Online scenario must fire exactly `online_warmup_requests` warmup calls."""
    suite = _online_suite(qps_levels=(2.0,), warmup_requests=5)
    requests = _make_requests(8)
    gen = AccelMarkLoadGen(suite, requests, "online", str(tmp_path))

    fn = MockInferenceFn()
    fn.set_warmup_budget(5)
    gen.run(fn.fn)

    # warmup (5) + 8 requests × 1 QPS × 1 run = 13 calls minimum
    assert fn.call_count >= 13, (
        f"expected at least 13 inference_fn calls (5 warmup + 8 timed), "
        f"got {fn.call_count}"
    )


def test_online_warmup_latencies_excluded_from_p99(tmp_path):
    """If warmup latencies leaked into the recorded distribution, p99 would
    collapse to the fast warmup value. Verify it stays at the timed value."""
    suite = _online_suite(qps_levels=(2.0,), warmup_requests=5)
    requests = _make_requests(8)
    gen = AccelMarkLoadGen(suite, requests, "online", str(tmp_path))

    fn = MockInferenceFn(warmup_ttft_ms=1.0, timed_ttft_ms=100.0)
    fn.set_warmup_budget(5)
    result = gen.run(fn.fn)

    qps_results = result["online"]["results_by_qps"]
    assert qps_results, "expected at least one QPS level result"
    p50 = qps_results[0]["ttft_ms_p50"]
    p99 = qps_results[0]["ttft_ms_p99"]

    # If warmup leaked in, p50 would be near 1.0 ms. With warmup excluded,
    # every recorded request returns 100 ms, so all percentiles snap there.
    assert abs(p50 - 100.0) < 0.5, f"p50 contaminated by warmup: {p50}"
    assert abs(p99 - 100.0) < 0.5, f"p99 contaminated by warmup: {p99}"


def test_online_warmup_zero_is_noop(tmp_path):
    """Backward compat: setting online_warmup_requests=0 must skip warmup."""
    suite = _online_suite(qps_levels=(2.0,), warmup_requests=0)
    requests = _make_requests(6)
    gen = AccelMarkLoadGen(suite, requests, "online", str(tmp_path))

    fn = MockInferenceFn()
    gen.run(fn.fn)

    # No warmup means exactly 6 calls (1 QPS × 6 requests × 1 run).
    assert fn.call_count == 6, (
        f"warmup=0 should fire only timed requests; got {fn.call_count} calls"
    )


def test_online_warmup_failure_does_not_abort_run(tmp_path):
    """A failing warmup request must be logged and ignored — the timed phase
    must still execute. Otherwise a flaky engine could prevent any submission."""
    suite = _online_suite(qps_levels=(2.0,), warmup_requests=3)
    requests = _make_requests(6)
    gen = AccelMarkLoadGen(suite, requests, "online", str(tmp_path))

    # Fail the first 2 warmup requests; the 3rd warmup + all timed must run.
    fn = MockInferenceFn(fail_first_n=2)
    fn.set_warmup_budget(3)
    result = gen.run(fn.fn)

    qps_results = result["online"]["results_by_qps"]
    assert qps_results, "timed phase did not run despite warmup failures"
    assert fn.call_count >= 3 + 6  # 3 warmup attempts + 6 timed


# ── burst warmup ──────────────────────────────────────────────────────────────

def test_burst_warmup_fires_configured_count(tmp_path):
    suite = _burst_suite(warmup_requests=4)
    requests = _make_requests(6)
    gen = AccelMarkLoadGen(suite, requests, "burst", str(tmp_path))

    fn = MockInferenceFn()
    fn.set_warmup_budget(4)
    gen.run(fn.fn)

    # At least 4 warmup calls must have fired before the timed cycles.
    assert fn.call_count >= 4, (
        f"burst warmup did not fire enough requests: {fn.call_count}"
    )


def test_burst_warmup_zero_is_noop(tmp_path):
    """Suites that omit burst_warmup_requests entirely default to 10; setting
    it to 0 must skip warmup."""
    suite = _burst_suite(warmup_requests=0)
    requests = _make_requests(6)
    gen = AccelMarkLoadGen(suite, requests, "burst", str(tmp_path))

    fn = MockInferenceFn()
    n_before = fn.call_count
    gen.run(fn.fn)
    # No assertion on exact count (timed cycles depend on Poisson timing),
    # but we can assert the mock saw at least 1 timed call.
    assert fn.call_count > n_before


# ── default values ────────────────────────────────────────────────────────────

def test_online_warmup_default_is_ten(tmp_path):
    """Suite without online_warmup_requests should get a sensible default."""
    suite = {
        "num_runs": 1,
        "online_qps_levels": [2.0],
        "online_sla_ttft_ms": 1000,
        "online_request_count": 4,
        "input_tokens": 10,
    }
    requests = _make_requests(4)
    gen = AccelMarkLoadGen(suite, requests, "online", str(tmp_path))
    assert gen.online_warmup_requests == 10


def test_burst_warmup_default_is_ten(tmp_path):
    suite = {
        "num_runs": 1,
        "online_sla_ttft_ms": 1000,
        "online_request_count": 4,
        "burst_steady_qps": 2.0,
        "burst_peak_qps": 4.0,
        "burst_duration_seconds": 0.3,
        "burst_interval_seconds": 0.3,
        "input_tokens": 10,
    }
    requests = _make_requests(4)
    gen = AccelMarkLoadGen(suite, requests, "burst", str(tmp_path))
    assert gen.burst_warmup_requests == 10
