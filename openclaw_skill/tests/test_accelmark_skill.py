"""
Unit tests for openclaw_skill/accelmark_skill.py

Covers the skill entry point functions — no real GPU, no real network.
All external calls (urllib, subprocess) are mocked.
- main(): follow-up flows for "submit" and "details"
- handle_submit(): GitHub Issue posting, success and failure paths
- query_ranking(): graceful None on any network error
- format_details(): table rendering, ← best marker
- _summarize_chips(): GPU env and CPU-only env
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from openclaw_skill.accelmark_skill import (
    _summarize_chips,
    format_details,
    handle_submit,
    main,
    query_ranking,
    query_submission_rank,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_result(chip_name: str = "NVIDIA RTX 4090", throughputs=(4210, 6124, 6840)) -> dict:
    return {
        "mode": "benchmark",
        "schema_version": "1.0",
        "suite_id": "mini-standard",
        "chip": {"name": chip_name, "vendor": "NVIDIA", "count": 1, "memory_gb_per_chip": 24.0},
        "software": {"framework": "vLLM", "framework_version": "0.6.6"},
        "model": {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "precision": "BF16"},
        "metrics": {
            "offline": {
                "results_by_batch_size": [
                    {"batch_size": 8,   "throughput_tokens_per_sec": throughputs[0], "peak_memory_gb": 16.1, "oom": False},
                    {"batch_size": 32,  "throughput_tokens_per_sec": throughputs[1], "peak_memory_gb": 17.8, "oom": False},
                    {"batch_size": 128, "throughput_tokens_per_sec": throughputs[2], "peak_memory_gb": 18.2, "oom": False},
                ]
            }
        },
        "meta": {"submitted_by": "", "date": "2026-03-21", "notes": ""},
    }


# ── _summarize_chips ──────────────────────────────────────────────────────────

def test_summarize_chips_gpu():
    env = {"accelerators": [{"name": "NVIDIA RTX 4090", "memory_gb": 24}]}
    assert "NVIDIA RTX 4090" in _summarize_chips(env)
    assert "24" in _summarize_chips(env)


def test_summarize_chips_cpu_only():
    env = {"accelerators": [], "cpu": {"model": "Intel Core i7-12700"}, "system_memory_gb": 32}
    summary = _summarize_chips(env)
    assert "Intel Core i7-12700" in summary
    assert "32" in summary


# ── query_ranking ─────────────────────────────────────────────────────────────

def test_query_ranking_returns_none_on_network_error():
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = query_ranking("NVIDIA RTX 4090")
    assert result is None


def test_query_ranking_returns_chip_data_on_success():
    fake_index = json.dumps({"NVIDIA RTX 4090": {"rank": 8, "total": 127, "percentile": 94}}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_index
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = query_ranking("NVIDIA RTX 4090")

    assert result == {"rank": 8, "total": 127, "percentile": 94}


def test_query_ranking_returns_none_for_unknown_chip():
    fake_index = json.dumps({}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_index
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = query_ranking("Unknown Chip XYZ")

    assert result is None


# ── handle_submit ─────────────────────────────────────────────────────────────

def test_handle_submit_success():
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        msg = handle_submit(_make_result(), openclaw_username="testuser")

    assert "✅" in msg
    assert "leaderboard" in msg.lower()


def test_handle_submit_sets_submitted_by():
    result = _make_result()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        handle_submit(result, openclaw_username="alice")

    assert result["meta"]["submitted_by"] == "alice"


def test_handle_submit_failure_returns_error_message():
    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        msg = handle_submit(_make_result(), openclaw_username="bob")

    assert "❌" in msg
    assert "manually" in msg.lower() or "github" in msg.lower()


# ── format_details ────────────────────────────────────────────────────────────

def test_format_details_renders_table():
    report = format_details(_make_result())
    assert "cc=8" in report or "bs=8" in report or "4,210" in report


def test_format_details_marks_best_row():
    report = format_details(_make_result())
    assert "← best" in report


def test_format_details_shows_framework_and_model():
    report = format_details(_make_result())
    assert "vLLM" in report
    assert "BF16" in report


# ── main: follow-up paths ─────────────────────────────────────────────────────

def test_main_submit_without_pending_result():
    msg = main("submit", {})
    assert "benchmark" in msg.lower() or "first" in msg.lower()


def test_main_details_without_pending_result():
    msg = main("details", {})
    assert "benchmark" in msg.lower() or "first" in msg.lower()


def test_main_submit_with_pending_result():
    ctx = {"accelmark_pending_result": _make_result(), "user": {"username": "carol"}}
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        msg = main("submit", ctx)

    assert "✅" in msg


def test_main_details_with_pending_result():
    ctx = {"accelmark_pending_result": _make_result()}
    msg = main("details", ctx)
    assert "benchmark" in msg.lower() or "tok/s" in msg.lower()
