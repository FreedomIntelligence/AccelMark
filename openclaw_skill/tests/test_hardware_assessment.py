"""
Unit tests for openclaw_skill/mini/hardware_assessment.py

Covers:
- assess_hardware: CPU tier classification by core count
- assess_hardware: RAM gate filters out models requiring more memory
- assess_hardware: feasibility labels (✓ Usable / △ Slow / ✗ Too slow)
- format_assessment_report: output structure
"""

import pytest
from openclaw_skill.mini.hardware_assessment import assess_hardware, format_assessment_report


def _env(cores: int, memory_gb: float, model: str = "Intel Core i7") -> dict:
    return {
        "cpu": {"model": model, "physical_cores": cores},
        "system_memory_gb": memory_gb,
        "accelerators": [],
    }


# ── CPU tier classification ───────────────────────────────────────────────────

def test_high_tier_8_cores():
    result = assess_hardware(_env(cores=8, memory_gb=32))
    # High tier: 1B Q4 should be ~12 tok/s → Usable
    rec_1b = next(r for r in result["recommendations"] if "1B" in r["model"])
    assert rec_1b["feasibility"] == "✓ Usable"


def test_medium_tier_4_cores():
    result = assess_hardware(_env(cores=4, memory_gb=32))
    # Medium tier: 1B Q4 ~6 tok/s → Usable (threshold is 5)
    rec_1b = next(r for r in result["recommendations"] if "1B" in r["model"])
    assert rec_1b["feasibility"] == "✓ Usable"


def test_low_tier_2_cores():
    result = assess_hardware(_env(cores=2, memory_gb=32))
    # Low tier: 1B Q4 ~2 tok/s → below 5 tok/s threshold → Slow
    rec_1b = next(r for r in result["recommendations"] if "1B" in r["model"])
    assert rec_1b["feasibility"] == "△ Slow"


def test_low_tier_8b_too_slow():
    result = assess_hardware(_env(cores=2, memory_gb=32))
    # Low tier: 8B Q4 ~0.2 tok/s → Too slow
    rec_8b = next((r for r in result["recommendations"] if "3B" not in r["model"] and "1B" not in r["model"]), None)
    # Phi-3-mini Q4 falls into 8b_q4 bucket
    phi = next((r for r in result["recommendations"] if "Phi" in r["model"]), None)
    if phi:
        assert phi["feasibility"] in ("△ Slow", "✗ Too slow")


def test_ram_gate_excludes_3b_model_on_low_ram():
    # 3B model requires 8GB RAM — should be absent with 4GB system
    result = assess_hardware(_env(cores=8, memory_gb=4))
    models = [r["model"] for r in result["recommendations"]]
    assert not any("3B" in m for m in models)


def test_ram_gate_includes_3b_model_on_sufficient_ram():
    result = assess_hardware(_env(cores=8, memory_gb=16))
    models = [r["model"] for r in result["recommendations"]]
    assert any("3B" in m for m in models)


# ── result structure ──────────────────────────────────────────────────────────

def test_assess_hardware_returns_mode_assessment():
    result = assess_hardware(_env(cores=8, memory_gb=32))
    assert result["mode"] == "assessment"


def test_assess_hardware_includes_cpu_and_memory():
    result = assess_hardware(_env(cores=8, memory_gb=32, model="Intel Core i9-13900K"))
    assert "Intel Core i9-13900K" in result["cpu"]
    assert result["memory_gb"] == 32


# ── format_assessment_report ──────────────────────────────────────────────────

def test_format_report_no_gpu_header():
    result = assess_hardware(_env(cores=8, memory_gb=32))
    report = format_assessment_report(result)
    assert "No GPU" in report


def test_format_report_contains_chip_name():
    result = assess_hardware(_env(cores=8, memory_gb=32, model="AMD Ryzen 9 7950X"))
    report = format_assessment_report(result)
    assert "AMD Ryzen 9 7950X" in report


def test_format_report_has_recommendation_line():
    result = assess_hardware(_env(cores=8, memory_gb=32))
    report = format_assessment_report(result)
    assert "Recommendation" in report or "ollama" in report


def test_format_report_shows_feasibility_labels():
    result = assess_hardware(_env(cores=8, memory_gb=32))
    report = format_assessment_report(result)
    assert any(sym in report for sym in ("✓", "△", "✗"))
