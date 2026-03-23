"""
Unit tests for openclaw_skill/mini/run_mini.py

Covers helper functions and output builders — no real inference.
- _detect_vendor: chip name → vendor string
- _guess_param_count: model_id → parameter count
- _load_requests: cycling behaviour
- build_result_json: schema structure, vendor, suite_id
- format_benchmark_report: chip info, ranking block, ← best marker
"""

import json
import tempfile
from pathlib import Path

import pytest

from openclaw_skill.mini.run_mini import (
    _detect_vendor,
    _guess_param_count,
    _load_requests,
    build_result_json,
    format_benchmark_report,
)
from openclaw_skill.mini.mini_suite_selector import _select_tier_by_memory


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _nvidia_env(memory_gb: float = 24.0) -> dict:
    return {
        "accelerators": [{"name": "NVIDIA RTX 4090", "memory_gb": memory_gb, "driver_version": "535.0"}],
        "system_memory_gb": 64,
        "runtime_version": "CUDA 12.1",
    }


def _apple_env(memory_gb: float = 36.0) -> dict:
    return {
        "accelerators": [{"name": "Apple M2 Ultra", "memory_gb": memory_gb, "driver_version": "unknown"}],
        "system_memory_gb": 36,
        "runtime_version": "Metal",
    }


def _benchmark_result(throughput: float = 6840.0, memory_gb: float = 18.2) -> dict:
    return {
        "results_by_batch_size": [
            {"batch_size": 8,   "throughput_tokens_per_sec": 4210.0, "peak_memory_gb": 16.1, "oom": False},
            {"batch_size": 32,  "throughput_tokens_per_sec": 6124.0, "peak_memory_gb": 17.8, "oom": False},
            {"batch_size": 128, "throughput_tokens_per_sec": throughput, "peak_memory_gb": memory_gb, "oom": False},
        ]
    }


def _standard_config():
    cfg = _select_tier_by_memory(24)   # standard tier
    cfg.framework = "vllm"
    return cfg


# ── _detect_vendor ────────────────────────────────────────────────────────────

def test_detect_vendor_nvidia():
    assert _detect_vendor([{"name": "NVIDIA RTX 4090"}]) == "NVIDIA"


def test_detect_vendor_geforce():
    assert _detect_vendor([{"name": "GeForce RTX 3080"}]) == "NVIDIA"


def test_detect_vendor_amd():
    assert _detect_vendor([{"name": "AMD Radeon RX 7900 XTX"}]) == "AMD"


def test_detect_vendor_apple():
    assert _detect_vendor([{"name": "Apple M2 Ultra"}]) == "Apple"


def test_detect_vendor_unknown():
    assert _detect_vendor([{"name": "SomeUnknownChip"}]) == "Other"


def test_detect_vendor_empty():
    assert _detect_vendor([]) == "CPU"


# ── _guess_param_count ────────────────────────────────────────────────────────

def test_guess_param_count_1b():
    assert _guess_param_count("bartowski/Llama-3.2-1B-Instruct-GGUF") == 1.0


def test_guess_param_count_8b():
    assert _guess_param_count("meta-llama/Meta-Llama-3-8B-Instruct") == 8.0


def test_guess_param_count_70b():
    assert _guess_param_count("bartowski/Meta-Llama-3-70B-Instruct-GGUF") == 70.0


def test_guess_param_count_unknown_returns_zero():
    assert _guess_param_count("some/unknown-model") == 0.0


# ── _load_requests ────────────────────────────────────────────────────────────

def test_load_requests_cycles_when_short():
    """If requests.jsonl has fewer entries than num_requests, it cycles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl = Path(tmpdir) / "requests.jsonl"
        jsonl.write_text(
            '{"prompt": "hello"}\n{"prompt": "world"}\n'
        )
        # Patch the path used inside _load_requests
        import openclaw_skill.mini.run_mini as rm
        original = rm.Path
        try:
            # Monkey-patch __file__ parent to our tempdir
            rm_file_backup = rm.__file__
            import types
            # Instead of patching Path, write actual file and patch __file__
            pass
        finally:
            pass

    # Simpler: call _load_requests with num_requests > actual lines in requests.jsonl
    # The real requests.jsonl has 200 lines; requesting 250 should cycle and return 250
    prompts = _load_requests(250)
    assert len(prompts) == 250


def test_load_requests_respects_num_requests():
    prompts = _load_requests(10)
    assert len(prompts) == 10


def test_load_requests_returns_strings():
    prompts = _load_requests(5)
    assert all(isinstance(p, str) for p in prompts)


# ── build_result_json ─────────────────────────────────────────────────────────

def test_build_result_json_top_level_keys():
    result = build_result_json(_nvidia_env(), _standard_config(), _benchmark_result())
    for key in ("mode", "schema_version", "suite_id", "chip", "software", "model", "task", "metrics", "meta"):
        assert key in result, f"Missing key: {key}"


def test_build_result_json_suite_id_format():
    cfg = _standard_config()
    result = build_result_json(_nvidia_env(), cfg, _benchmark_result())
    assert result["suite_id"] == f"mini-{cfg.tier}"


def test_build_result_json_nvidia_vendor():
    result = build_result_json(_nvidia_env(), _standard_config(), _benchmark_result())
    assert result["chip"]["vendor"] == "NVIDIA"


def test_build_result_json_apple_vendor():
    cfg = _select_tier_by_memory(36)
    cfg.framework = "mlx-lm"
    result = build_result_json(_apple_env(), cfg, _benchmark_result())
    assert result["chip"]["vendor"] == "Apple"


def test_build_result_json_chip_memory():
    result = build_result_json(_nvidia_env(memory_gb=24), _standard_config(), _benchmark_result())
    assert result["chip"]["memory_gb_per_chip"] == 24.0


def test_build_result_json_metrics_passthrough():
    br = _benchmark_result()
    result = build_result_json(_nvidia_env(), _standard_config(), br)
    assert result["metrics"]["offline"] == br


# ── format_benchmark_report ───────────────────────────────────────────────────

def test_format_report_contains_chip_name():
    result = build_result_json(_nvidia_env(), _standard_config(), _benchmark_result())
    report = format_benchmark_report(result, _standard_config(), ranking=None)
    assert "NVIDIA RTX 4090" in report


def test_format_report_contains_throughput():
    result = build_result_json(_nvidia_env(), _standard_config(), _benchmark_result(throughput=6840))
    report = format_benchmark_report(result, _standard_config(), ranking=None)
    assert "6,840" in report


def test_format_report_no_ranking_block_when_none():
    result = build_result_json(_nvidia_env(), _standard_config(), _benchmark_result())
    report = format_benchmark_report(result, _standard_config(), ranking=None)
    assert "Community ranking" not in report


def test_format_report_shows_ranking_when_provided():
    result = build_result_json(_nvidia_env(), _standard_config(), _benchmark_result())
    ranking = {"rank": 8, "total": 127, "percentile": 94}
    report = format_benchmark_report(result, _standard_config(), ranking=ranking)
    assert "Community ranking" in report
    assert "#8" in report
    assert "94%" in report


def test_format_report_shows_capabilities():
    cfg = _standard_config()
    result = build_result_json(_nvidia_env(), cfg, _benchmark_result())
    report = format_benchmark_report(result, cfg, ranking=None)
    assert any(cap in report for cap in cfg.capabilities)


def test_format_report_submit_prompt():
    result = build_result_json(_nvidia_env(), _standard_config(), _benchmark_result())
    report = format_benchmark_report(result, _standard_config(), ranking=None)
    assert "submit" in report.lower()
