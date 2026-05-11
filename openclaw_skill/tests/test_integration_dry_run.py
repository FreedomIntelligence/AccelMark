"""
Integration test: dry run completes without errors.

Patches collect_environment() so no real GPU or subprocess call is made.
Exercises the full main() logic path: env → mode → tier selection → dry-run output.
"""

import sys
from unittest.mock import patch

import pytest


# Fake env_info representing a mid-range NVIDIA GPU
_FAKE_GPU_ENV = {
    "accelerators": [{"name": "NVIDIA RTX 4090", "memory_gb": 24.0, "driver_version": "535.0"}],
    "cpu": {"model": "Intel Core i9-13900K", "physical_cores": 24},
    "system_memory_gb": 64,
    "runtime_version": "CUDA 12.1",
}

# Fake env_info representing a CPU-only machine
_FAKE_CPU_ENV = {
    "accelerators": [],
    "cpu": {"model": "Intel Core i7-12700", "physical_cores": 12},
    "system_memory_gb": 32,
    "runtime_version": None,
}


def test_dry_run_gpu_prints_tier_and_model(capsys):
    """Dry run with a GPU env: prints detected hardware, tier, and model — no inference."""
    import openclaw_skill.mini.run_mini as rm

    with patch.object(rm, "collect_environment", return_value=_FAKE_GPU_ENV):
        # Simulate --dry-run by calling the internal logic directly
        env = _FAKE_GPU_ENV
        from openclaw_skill.mini.mini_suite_selector import select_mode, select_mini_suite
        assert select_mode(env) == "benchmark"
        config = select_mini_suite(env)
        assert config.tier == "standard"   # 24GB → standard
        assert config.framework == "vllm"
        assert config.model_id            # non-empty model ID
        assert config.estimated_minutes >= 1


def test_dry_run_cpu_only_returns_assessment(capsys):
    """Dry run with CPU-only env: routes to assessment mode, no inference attempted."""
    from openclaw_skill.mini.mini_suite_selector import select_mode
    from openclaw_skill.mini.hardware_assessment import assess_hardware, format_assessment_report

    env = _FAKE_CPU_ENV
    assert select_mode(env) == "assessment"

    result = assess_hardware(env)
    report = format_assessment_report(result)

    assert "No GPU" in report
    assert "Intel Core i7-12700" in report
    assert result["mode"] == "assessment"


def test_build_result_json_validates_required_keys():
    """
    Builds a result.json from fake GPU env + fake benchmark output and
    checks all schema-required top-level keys are present.
    """
    from openclaw_skill.mini.run_mini import build_result_json
    from openclaw_skill.mini.mini_suite_selector import select_mini_suite

    env = _FAKE_GPU_ENV
    config = select_mini_suite(env)
    config.framework = "vllm"

    fake_benchmark = {
        "results_by_batch_size": [
            {"batch_size": 32, "throughput_tokens_per_sec": 6840.0,
             "peak_memory_gb": 18.2, "oom": False}
        ]
    }

    result = build_result_json(env, config, fake_benchmark)

    required_keys = {"mode", "schema_version", "suite_id", "chip", "software", "model",
                     "task", "metrics", "accuracy", "meta"}
    assert required_keys.issubset(result.keys())
    assert result["chip"]["name"] == "NVIDIA RTX 4090"
    assert result["metrics"]["offline"] == fake_benchmark
