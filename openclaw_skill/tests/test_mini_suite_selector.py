"""
Unit tests for openclaw_skill/mini/mini_suite_selector.py

Covers:
- select_mode: benchmark vs assessment path
- _select_tier_by_memory: all 6 tiers at correct thresholds
- select_mini_suite: Apple Silicon uses mlx-lm, others use vllm
"""

import pytest
from openclaw_skill.mini.mini_suite_selector import (
    select_mode,
    select_mini_suite,
    _select_tier_by_memory,
    MiniSuiteConfig,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_env(memory_gb: float, name: str = "NVIDIA RTX 4090") -> dict:
    return {"accelerators": [{"name": name, "memory_gb": memory_gb}]}


def _apple_env(memory_gb: float) -> dict:
    return {"accelerators": [{"name": "Apple M2 Ultra", "memory_gb": memory_gb}]}


def _cpu_env(memory_gb: float = 32) -> dict:
    return {"accelerators": [], "system_memory_gb": memory_gb}


# ── select_mode ───────────────────────────────────────────────────────────────

def test_select_mode_no_accelerators():
    assert select_mode(_cpu_env()) == "assessment"


def test_select_mode_accelerator_below_4gb():
    assert select_mode(_gpu_env(memory_gb=2)) == "assessment"


def test_select_mode_accelerator_exactly_4gb():
    assert select_mode(_gpu_env(memory_gb=4)) == "benchmark"


def test_select_mode_with_gpu():
    assert select_mode(_gpu_env(memory_gb=24)) == "benchmark"


def test_select_mode_apple_silicon():
    assert select_mode(_apple_env(memory_gb=36)) == "benchmark"


def test_select_mode_empty_accelerator_list():
    assert select_mode({"accelerators": []}) == "assessment"


# ── tier selection by memory ──────────────────────────────────────────────────

def test_tier_nano_below_6gb():
    cfg = _select_tier_by_memory(4)
    assert cfg.tier == "nano"
    assert cfg.estimated_minutes == 2


def test_tier_mini_small_6_to_12gb():
    cfg = _select_tier_by_memory(8)
    assert cfg.tier == "mini-small"


def test_tier_mini_small_at_boundary_6gb():
    cfg = _select_tier_by_memory(6)
    assert cfg.tier == "mini-small"


def test_tier_mini_12_to_20gb():
    cfg = _select_tier_by_memory(16)
    assert cfg.tier == "mini"


def test_tier_mini_at_boundary_12gb():
    cfg = _select_tier_by_memory(12)
    assert cfg.tier == "mini"


def test_tier_standard_20_to_48gb():
    cfg = _select_tier_by_memory(24)
    assert cfg.tier == "standard"


def test_tier_pro_48_to_90gb():
    cfg = _select_tier_by_memory(80)
    assert cfg.tier == "pro"


def test_tier_pro_large_90gb_plus():
    cfg = _select_tier_by_memory(192)
    assert cfg.tier == "pro-large"


# ── framework selection ───────────────────────────────────────────────────────

def test_nvidia_gpu_uses_vllm():
    cfg = select_mini_suite(_gpu_env(memory_gb=24, name="NVIDIA RTX 4090"))
    assert cfg.framework == "vllm"


def test_apple_silicon_uses_mlx():
    cfg = select_mini_suite(_apple_env(memory_gb=36))
    assert cfg.framework == "mlx-lm"


# ── MiniSuiteConfig fields ────────────────────────────────────────────────────

def test_config_has_required_fields():
    cfg = _select_tier_by_memory(16)
    assert isinstance(cfg, MiniSuiteConfig)
    assert cfg.model_id
    assert cfg.batch_sizes
    assert cfg.num_requests > 0
    assert cfg.capabilities
