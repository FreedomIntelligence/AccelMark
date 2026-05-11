"""
Unit tests for serve/capacity.py.

Tests capacity estimate parsing from fixture result.json files.
No GPU, no network, no real results directory required.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from serve.capacity import (
    CapacityEstimate,
    load_capacity_estimate,
    format_capacity_log,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_result(tmpdir: Path, tier: str, name: str, data: dict) -> Path:
    """Write a result.json to a temporary results directory."""
    d = tmpdir / tier / name
    d.mkdir(parents=True)
    p = d / "result.json"
    p.write_text(json.dumps(data))
    return p


def _base_result(impl_id: str = "nvidia_vllm_47f5d58e") -> dict:
    return {
        "suite_id": "suite_A",
        "implementation_id": impl_id,
        "chip": {"name": "NVIDIA A100-SXM4-80GB"},
        "meta": {"date": "2026-03-22"},
        "metrics": {
            "offline": {
                "results_by_concurrency": [
                    {"concurrency": 8,   "throughput_tokens_per_sec": 4800.0, "oom": False},
                    {"concurrency": 32,  "throughput_tokens_per_sec": 5321.0, "oom": False},
                    {"concurrency": 128, "throughput_tokens_per_sec": 5100.0, "oom": False},
                ]
            },
            "online": {"max_valid_qps": 5.0},
            "interactive": {"ttft_ms_p99": 68.4},
        },
    }


# ── load_capacity_estimate ────────────────────────────────────────────────────

def test_returns_none_for_empty_results_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("serve.capacity._RESULTS_DIR", Path(tmpdir)):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")
    assert est is None


def test_returns_none_when_no_matching_impl_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _write_result(tmppath, "community", "sub1", _base_result("other_impl_abc12345"))
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")
    assert est is None


def test_returns_none_for_empty_string_impl_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("serve.capacity._RESULTS_DIR", Path(tmpdir)):
            est = load_capacity_estimate("")
    assert est is None


def test_returns_none_for_none_impl_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("serve.capacity._RESULTS_DIR", Path(tmpdir)):
            est = load_capacity_estimate(None)
    assert est is None


def test_finds_matching_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _write_result(tmppath, "community", "sub1", _base_result("nvidia_vllm_47f5d58e"))
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est is not None
    assert isinstance(est, CapacityEstimate)
    assert est.implementation_id == "nvidia_vllm_47f5d58e"
    assert est.suite_id == "suite_A"
    assert est.chip == "NVIDIA A100-SXM4-80GB"
    assert est.date == "2026-03-22"


def test_extracts_best_offline_throughput():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _write_result(tmppath, "community", "sub1", _base_result())
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    # Best is 5321.0 from concurrency=32
    assert est.offline_throughput_tokens_per_sec == 5321.0


def test_extracts_online_max_qps():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _write_result(tmppath, "community", "sub1", _base_result())
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est.online_max_qps == 5.0


def test_extracts_interactive_ttft():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _write_result(tmppath, "community", "sub1", _base_result())
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est.interactive_ttft_p99_ms == pytest.approx(68.4)


def test_picks_most_recent_when_multiple_results():
    """When multiple results match, the most recent date wins."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        old = _base_result()
        old["meta"]["date"] = "2026-01-01"
        old["metrics"]["online"]["max_valid_qps"] = 3.0

        new = _base_result()
        new["meta"]["date"] = "2026-03-22"
        new["metrics"]["online"]["max_valid_qps"] = 5.0

        _write_result(tmppath, "community", "old_sub", old)
        _write_result(tmppath, "verified", "new_sub", new)

        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est.online_max_qps == 5.0
    assert est.date == "2026-03-22"


def test_skips_oom_rows():
    """OOM rows should not contribute to throughput calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        data = _base_result()
        data["metrics"]["offline"]["results_by_concurrency"] = [
            {"concurrency": 8,   "throughput_tokens_per_sec": 5000.0, "oom": False},
            {"concurrency": 128, "throughput_tokens_per_sec": 9999.0, "oom": True},
        ]
        _write_result(tmppath, "community", "sub1", data)
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est.offline_throughput_tokens_per_sec == 5000.0


def test_falls_back_to_legacy_batch_size_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        data = _base_result()
        data["metrics"]["offline"] = {
            "results_by_batch_size": [
                {"batch_size": 32, "throughput_tokens_per_sec": 4200.0, "oom": False},
            ]
        }
        _write_result(tmppath, "community", "sub1", data)
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est.offline_throughput_tokens_per_sec == 4200.0


def test_suite_e_scaling_fallback():
    """Suite E results without offline section fall back to scaling base throughput."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        data = {
            "suite_id": "suite_E",
            "implementation_id": "nvidia_vllm_47f5d58e",
            "chip": {"name": "NVIDIA A100-SXM4-80GB"},
            "meta": {"date": "2026-03-22"},
            "metrics": {
                "scaling": {
                    "base_throughput_tokens_per_sec": 6018.79,
                    "results_by_chip_count": [],
                },
            },
        }
        _write_result(tmppath, "verified", "sub1", data)
        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est.offline_throughput_tokens_per_sec == pytest.approx(6018.79)


def test_tolerates_corrupt_result_json():
    """Corrupted result.json files should be silently skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        bad_dir = tmppath / "community" / "corrupt_sub"
        bad_dir.mkdir(parents=True)
        (bad_dir / "result.json").write_text("NOT VALID JSON {{{")

        # Also add a valid result
        _write_result(tmppath, "community", "good_sub", _base_result())

        with patch("serve.capacity._RESULTS_DIR", tmppath):
            est = load_capacity_estimate("nvidia_vllm_47f5d58e")

    assert est is not None
    assert est.offline_throughput_tokens_per_sec == 5321.0


# ── format_capacity_log ───────────────────────────────────────────────────────

def test_format_capacity_log_full():
    est = CapacityEstimate(
        implementation_id="nvidia_vllm_47f5d58e",
        suite_id="suite_A",
        chip="NVIDIA A100-SXM4-80GB",
        date="2026-03-22",
        offline_throughput_tokens_per_sec=5321.0,
        online_max_qps=5.0,
        online_sla_ttft_ms=None,
        interactive_ttft_p99_ms=68.4,
        result_path="/results/community/sub/result.json",
    )
    lines = format_capacity_log(est)
    full = "\n".join(lines)

    assert "suite_A" in full
    assert "2026-03-22" in full
    assert "5,321" in full      # formatted throughput
    assert "5.0" in full        # max QPS
    assert "68" in full         # TTFT p99
    assert "result.json" in full


def test_format_capacity_log_minimal():
    """Should not crash when most fields are None."""
    est = CapacityEstimate(
        implementation_id="nvidia_vllm_47f5d58e",
        suite_id="suite_A",
        chip="NVIDIA A100",
        date="2026-01-01",
        offline_throughput_tokens_per_sec=None,
        online_max_qps=None,
        online_sla_ttft_ms=None,
        interactive_ttft_p99_ms=None,
        result_path="/some/path",
    )
    lines = format_capacity_log(est)
    # Should return at least the header line
    assert len(lines) >= 1
    assert "suite_A" in lines[0]
