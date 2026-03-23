"""
AccelMark Serve — Capacity estimation from prior benchmark results.

Reads results/ at server startup to surface capacity hints in the startup log.
Best-effort only — never blocks startup if no matching result is found.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"


@dataclass
class CapacityEstimate:
    implementation_id: str
    suite_id: str
    chip: str
    date: str
    # Offline
    offline_throughput_tokens_per_sec: Optional[float]
    # Online
    online_max_qps: Optional[float]
    online_sla_ttft_ms: Optional[float]
    # Interactive
    interactive_ttft_p99_ms: Optional[float]
    # Source
    result_path: str


def load_capacity_estimate(implementation_id: str) -> Optional[CapacityEstimate]:
    """
    Find the most recent result.json that matches the given implementation_id
    and return a CapacityEstimate. Returns None if no match found.

    Only matches on implementation_id — chip name and suite are informational.
    Skips results where implementation_id is absent (old results without the field).
    """
    if not implementation_id:
        return None

    candidates: list[tuple[str, Path]] = []  # (date, path)

    for tier in ("verified", "community"):
        tier_dir = _RESULTS_DIR / tier
        if not tier_dir.exists():
            continue
        for submission_dir in tier_dir.iterdir():
            if not submission_dir.is_dir():
                continue
            result_path = submission_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                with open(result_path) as f:
                    data = json.load(f)
                if data.get("implementation_id") != implementation_id:
                    continue
                date_str = data.get("meta", {}).get("date", "")
                candidates.append((date_str, result_path, data))
            except Exception:
                continue

    if not candidates:
        return None

    # Most recent first
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, result_path, data = candidates[0]

    metrics     = data.get("metrics", {})
    offline     = metrics.get("offline", {})
    online      = metrics.get("online", {})
    interactive = metrics.get("interactive", {})

    # Best offline throughput
    offline_throughput = None
    rows = (offline.get("results_by_concurrency")
            or offline.get("results_by_batch_size", []))
    valid = [r for r in rows if not r.get("oom") and r.get("throughput_tokens_per_sec")]
    if valid:
        offline_throughput = max(r["throughput_tokens_per_sec"] for r in valid)

    # Suite E fallback
    if offline_throughput is None:
        scaling = metrics.get("scaling", {})
        offline_throughput = (
            scaling.get("base_throughput_tokens_per_sec")
            or scaling.get("base_throughput_1x")
        )

    return CapacityEstimate(
        implementation_id=implementation_id,
        suite_id=data.get("suite_id", "unknown"),
        chip=data.get("chip", {}).get("name", "unknown"),
        date=data.get("meta", {}).get("date", "unknown"),
        offline_throughput_tokens_per_sec=offline_throughput,
        online_max_qps=online.get("max_valid_qps") if online else None,
        online_sla_ttft_ms=None,  # not stored in result.json directly
        interactive_ttft_p99_ms=(
            interactive.get("ttft_ms_p99") if interactive else None
        ),
        result_path=str(result_path),
    )


def format_capacity_log(est: CapacityEstimate) -> list[str]:
    """Return log lines summarising the capacity estimate."""
    lines = [
        f"Capacity estimate from {est.suite_id} result ({est.date}, {est.chip}):"
    ]
    if est.offline_throughput_tokens_per_sec:
        lines.append(
            f"  Offline throughput : "
            f"{est.offline_throughput_tokens_per_sec:,.0f} tokens/sec"
        )
    if est.online_max_qps is not None:
        lines.append(
            f"  Online max QPS     : {est.online_max_qps} "
            f"(within 500ms TTFT SLA)"
        )
    if est.interactive_ttft_p99_ms is not None:
        lines.append(
            f"  Interactive TTFT   : {est.interactive_ttft_p99_ms:.0f}ms p99"
        )
    lines.append(
        f"  Source: {est.result_path}"
    )
    return lines
