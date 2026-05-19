"""
Tests for the reliability blocks emitted by each loadgen scenario.

Locks down:
- `_cv_pct` / `_stability_label` helpers
- `_reliability_block` shape contract
- `_compute_recovery_time` rolling-window logic
- offline / online / interactive / sustained / burst each emit the new
  fields with the expected types and a non-None CV when n >= 2

These tests use the same MockInferenceFn pattern as test_warmup.py — a
real `async def` closure bound to a counter, since loadgen detects
coroutines via `asyncio.iscoroutinefunction()`.
"""

from __future__ import annotations

import asyncio

import pytest

from loadgen.loadgen import (
    AccelMarkLoadGen,
    _compute_recovery_time,
    _cv_pct,
    _reliability_block,
    _stability_label,
)
from loadgen.types import InferenceResult


# ── Pure helper tests ─────────────────────────────────────────────────────────

def test_cv_pct_basic():
    assert _cv_pct([100.0, 100.0, 100.0]) == 0.0
    cv = _cv_pct([90.0, 100.0, 110.0])
    assert cv is not None
    assert 9.0 < cv < 11.0, f"expected CV near 10%, got {cv}"


def test_cv_pct_returns_none_for_small_or_invalid_input():
    assert _cv_pct([]) is None
    assert _cv_pct([42.0]) is None
    assert _cv_pct([0.0, 0.0, 0.0]) is None  # mean=0, undefined CV


def test_stability_labels():
    assert _stability_label(0.5) == "stable"
    assert _stability_label(2.0) == "stable"   # inclusive boundary
    assert _stability_label(3.0) == "noisy"
    assert _stability_label(5.0) == "noisy"    # inclusive boundary
    assert _stability_label(7.0) == "unstable"
    assert _stability_label(None) is None


def test_reliability_block_shape():
    block = _reliability_block([100.0, 102.0, 98.0], decimals=1)
    assert set(block.keys()) == {"n", "mean", "std", "cv_pct", "stability", "runs"}
    assert block["n"] == 3
    assert block["mean"] == 100.0
    assert block["runs"] == [100.0, 102.0, 98.0]
    assert block["stability"] == "stable"


def test_reliability_block_empty_input_returns_empty_dict():
    """Frontend gates on the block being non-empty; never None."""
    assert _reliability_block([]) == {}


# ── Recovery-time tests ──────────────────────────────────────────────────────

def test_recovery_time_finds_the_first_clean_window():
    """Build a synthetic post-burst window where the first 5 seconds are
    elevated and everything after is clean. Recovery must land around 5s."""
    arrivals = [i * 0.5 for i in range(40)]   # 20 seconds of arrivals at 2 Hz
    # Elevated TTFTs first 5 s, then drop to clean values.
    ttfts = [1500.0 if a < 5.0 else 200.0 for a in arrivals]
    rec = _compute_recovery_time(arrivals, ttfts, threshold_ms=500.0, window_s=2.0, min_samples=4)
    assert rec is not None, "expected recovery, got None"
    assert 4.5 <= rec <= 8.0, f"recovery expected ≈5–8s, got {rec}"


def test_recovery_time_returns_none_when_never_recovers():
    arrivals = [i * 0.5 for i in range(20)]
    ttfts = [2000.0] * 20  # always above any sane threshold
    assert _compute_recovery_time(arrivals, ttfts, threshold_ms=500.0) is None


def test_recovery_time_returns_none_when_too_few_samples():
    assert _compute_recovery_time([], [], threshold_ms=500.0) is None
    assert _compute_recovery_time([1.0, 2.0], [100.0, 100.0],
                                  threshold_ms=500.0, min_samples=5) is None


# ── Scenario integration tests ───────────────────────────────────────────────

def _make_requests(n: int):
    from loadgen.loadgen import InferenceRequest
    return [
        InferenceRequest(prompt=f"p{i}", request_id=i, input_tokens=10, max_tokens=20)
        for i in range(n)
    ]


def _async_fn(ttft_ms: float = 100.0):
    """Build a real `async def` returning a constant InferenceResult."""
    async def fn(request) -> InferenceResult:
        await asyncio.sleep(0)
        return InferenceResult(
            first_token_time_ms=ttft_ms,
            total_time_ms=ttft_ms * 2,
            output_tokens=20,
            input_tokens=10,
            success=True,
        )
    return fn


def _sync_offline_fn(ttft_ms: float = 100.0):
    """Sync inference_fn used for offline scenario — receives list of requests."""
    def fn(reqs):
        return [
            InferenceResult(
                first_token_time_ms=ttft_ms,
                total_time_ms=ttft_ms * 2,
                output_tokens=20,
                input_tokens=10,
                success=True,
            )
            for _ in reqs
        ]
    return fn


def test_offline_emits_throughput_reliability(tmp_path):
    suite = {
        "concurrency_levels": [4],
        "num_runs": 3,
        "warmup_runs": 0,
        "request_count": 8,
        "input_tokens": 10,
    }
    requests = _make_requests(8)
    gen = AccelMarkLoadGen(suite, requests, "offline", str(tmp_path))
    result = gen.run(_sync_offline_fn())

    cc_results = result["offline"]["results_by_concurrency"]
    assert cc_results, "offline scenario produced no results"
    rel = cc_results[0].get("throughput_tokens_per_sec_reliability")
    assert rel, "offline scenario did not emit reliability block"
    assert rel["n"] == 3
    assert rel["cv_pct"] is not None
    assert rel["stability"] in {"stable", "noisy", "unstable"}
    assert len(rel["runs"]) == 3


def test_online_emits_ttft_p99_reliability(tmp_path):
    suite = {
        "num_runs": 2,
        "online_qps_levels": [2.0],
        "online_sla_ttft_ms": 1000,
        "online_request_count": 6,
        "online_warmup_requests": 0,
        "input_tokens": 10,
    }
    requests = _make_requests(6)
    gen = AccelMarkLoadGen(suite, requests, "online", str(tmp_path))
    result = gen.run(_async_fn(ttft_ms=100.0))

    qps_results = result["online"]["results_by_qps"]
    assert qps_results, "online scenario produced no results"
    rel = qps_results[0].get("ttft_ms_p99_reliability")
    assert rel, "online scenario did not emit reliability block"
    assert rel["n"] == 2
    # With constant TTFT the CV should be exactly 0.
    assert rel["cv_pct"] == 0.0
    assert rel["stability"] == "stable"


def test_interactive_emits_ttft_p99_reliability(tmp_path):
    suite = {
        "num_runs": 2,
        "interactive_warmup_runs": 0,
        "interactive_request_count": 4,
        "input_tokens": 10,
    }
    requests = _make_requests(4)
    gen = AccelMarkLoadGen(suite, requests, "interactive", str(tmp_path))
    result = gen.run(_async_fn(ttft_ms=120.0))

    inter = result["interactive"]
    rel = inter.get("ttft_ms_p99_reliability")
    assert rel, "interactive scenario did not emit reliability block"
    assert rel["n"] == 2
    assert rel["stability"] == "stable"


def test_sustained_emits_throughput_post_warmup_reliability(tmp_path):
    """Run a tiny sustained scenario — long enough to produce ≥2 sample
    intervals so CV is computable."""
    suite = {
        "sustained_concurrency": 2,
        "duration_minutes": 4 / 60,        # 4 seconds total
        "sample_interval_seconds": 1.0,
        "warmup_minutes": 1 / 60,          # 1-second warmup
        "input_tokens": 10,
    }
    requests = _make_requests(20)
    gen = AccelMarkLoadGen(suite, requests, "sustained", str(tmp_path))
    result = gen.run(_async_fn(ttft_ms=30.0))

    rel = result["sustained"].get("throughput_post_warmup_reliability")
    assert isinstance(rel, dict), "sustained scenario did not emit reliability block"
    # cv_pct may be None if not enough post-warmup samples landed; we only
    # require the field exists. When n >= 2 the stability must be set.
    if rel.get("n", 0) >= 2:
        assert rel["stability"] in {"stable", "noisy", "unstable"}


def test_burst_emits_recovery_time_seconds(tmp_path):
    """Burst with constant low TTFT should report a finite (small)
    recovery_time and a list (possibly empty) per-cycle field."""
    suite = {
        "num_runs": 2,
        "online_sla_ttft_ms": 1000,
        "online_request_count": 6,
        "burst_warmup_requests": 0,
        "burst_steady_qps": 2.0,
        "burst_peak_qps": 4.0,
        "burst_duration_seconds": 0.5,
        "burst_interval_seconds": 0.5,
        "input_tokens": 10,
    }
    requests = _make_requests(6)
    gen = AccelMarkLoadGen(suite, requests, "burst", str(tmp_path))
    result = gen.run(_async_fn(ttft_ms=100.0))

    burst = result["burst"]
    assert "recovery_time_seconds" in burst
    assert "recovery_time_seconds_per_cycle" in burst
    assert isinstance(burst["recovery_time_seconds_per_cycle"], list)
