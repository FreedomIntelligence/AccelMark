"""Unit tests for platform detector isolation, protocol integrity checks, and runner tiers.

Runs purely on CPU without any GPU or vendor SDKs (mocks torch.cuda / torch.musa).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _make_fake_torch(device_names: list[str], *, has_musa: bool = False):
    """Build a minimal fake torch module exposing CUDA or MUSA devices."""
    torch_mod = types.ModuleType("torch")

    class _Props:
        def __init__(self, name: str):
            self.name = name
            self.total_memory = 32 * (1024 ** 3)

    class _Backend:
        @staticmethod
        def device_count():
            return len(device_names)

        @staticmethod
        def get_device_properties(idx: int):
            return _Props(device_names[idx])

        @staticmethod
        def is_bf16_supported():
            return True

    if has_musa:
        torch_mod.musa = _Backend()
        torch_mod.cuda = types.SimpleNamespace(device_count=lambda: 0)
    else:
        torch_mod.cuda = _Backend()

    torch_mod.version = types.SimpleNamespace(cuda="12.1")
    return torch_mod


def test_platform_detectors_do_not_hijack_torch_cuda_shims():
    """Moore Threads and Iluvatar detectors must not claim each other's or NVIDIA GPUs."""
    from runners.platforms import iluvatar, moorethreads

    orig_torch = sys.modules.get("torch")
    try:
        # 1. Iluvatar MR-V100 exposed via torch.cuda shim
        sys.modules["torch"] = _make_fake_torch(["Iluvatar MR-V100"], has_musa=False)
        iluv_devs = iluvatar.collect()
        mtt_devs = moorethreads._collect_via_torch()
        assert len(iluv_devs) == 1
        assert iluv_devs[0]["vendor"] == "Iluvatar"
        assert mtt_devs == [], f"moorethreads hijacked Iluvatar GPU: {mtt_devs}"

        # 2. NVIDIA A100 exposed via torch.cuda
        sys.modules["torch"] = _make_fake_torch(["NVIDIA A100-SXM4-80GB"], has_musa=False)
        assert iluvatar.collect() == []
        assert moorethreads._collect_via_torch() == []

        # 3. Moore Threads MTT S4000 exposed via torch.musa
        sys.modules["torch"] = _make_fake_torch(["MTT S4000"], has_musa=True)
        mtt_devs = moorethreads._collect_via_torch()
        assert len(mtt_devs) == 1
        assert mtt_devs[0]["vendor"] == "Moore Threads"
        assert iluvatar.collect() == []
    finally:
        if orig_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = orig_torch


def test_protocol_integrity_catches_swapped_throughput_and_bypass():
    """check_protocol_integrity catches swapped output/total TPS, low-score bypass, and Suite D truncation."""
    # Import directly without requiring jsonschema at module top if absent;
    # in CI jsonschema is installed so validate_submission imports cleanly.
    from runners.validate_submission import check_protocol_integrity

    # Swapped output-only > total throughput -> hard error
    bad_tps = {
        "suite_id": "suite_A",
        "metrics": {
            "offline": {
                "results_by_concurrency": [
                    {
                        "concurrency": 16,
                        "oom": False,
                        "throughput_tokens_per_sec": 5000.0,
                        "throughput_tokens_per_sec_total": 1000.0,
                    }
                ]
            }
        },
    }
    errs, warns = check_protocol_integrity(bad_tps)
    assert any("exceeds throughput_tokens_per_sec_total" in e for e in errs)

    # Low accuracy (<0.25) with baseline_delta=None -> warning
    bypass_acc = {
        "suite_id": "suite_E",
        "accuracy": {"subset_score": 0.07, "baseline_delta": None, "valid": True},
        "metrics": {},
    }
    errs, warns = check_protocol_integrity(bypass_acc)
    assert errs == []
    assert any("random-guess floor" in w for w in warns)

    # Suite D truncated prompts -> warning
    trunc_d = {
        "suite_id": "suite_D",
        "metrics": {
            "offline": {
                "results_by_concurrency": [
                    {
                        "concurrency": 4,
                        "oom": False,
                        "total_input_tokens": 10000,
                        "num_requests": 20,
                        "throughput_tokens_per_sec": 100.0,
                        "throughput_tokens_per_sec_total": 500.0,
                    }
                ]
            }
        },
    }
    errs, warns = check_protocol_integrity(trunc_d)
    assert errs == []
    assert any("silent prompt truncation" in w for w in warns)


def test_platforms_matrix_separates_reference_and_optimized():
    from tools.generate_platforms_matrix import _build_table

    table = _build_table()
    assert "Optimized &amp; experimental runners" in table
    # Reference runner appears before the <details> block
    details_idx = table.index("<details>")
    assert table.index("nvidia_vllm_47f5d58e") < details_idx
    assert table.index("nvidia_sglang_inference_opt_4fd1e59e") > details_idx
