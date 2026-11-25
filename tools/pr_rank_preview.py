#!/usr/bin/env python3
"""Generate a Markdown leaderboard rank preview for PR submission comments.

Usage:
    python tools/pr_rank_preview.py --dirs-file /tmp/changed_dirs.txt --output /tmp/pr_preview.md
    python tools/pr_rank_preview.py results/community/<run_name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def extract_primary_metric(result: dict) -> tuple[float | None, str]:
    """Return (primary_metric_value, unit_label) matching leaderboard/generate.py."""
    metrics = result.get("metrics", {}) or {}

    offline = metrics.get("offline")
    if isinstance(offline, dict):
        rows = offline.get("results_by_concurrency") or offline.get("results_by_batch_size") or []
        valid = [
            row["throughput_tokens_per_sec"]
            for row in rows
            if isinstance(row, dict)
            and not row.get("oom")
            and row.get("throughput_tokens_per_sec")
        ]
        if valid:
            return float(max(valid)), "tok/s"

    scaling = metrics.get("scaling")
    if isinstance(scaling, dict):
        base = scaling.get("base_throughput_tokens_per_sec")
        if base:
            return float(base), "tok/s (1x)"
        for entry in scaling.get("results_by_chip_count", []):
            if isinstance(entry, dict) and entry.get("chip_count") == 1:
                val = entry.get("best_throughput_tokens_per_sec")
                if val:
                    return float(val), "tok/s (1x)"

    quant = metrics.get("quantization")
    if isinstance(quant, dict):
        entries = quant.get("results_by_precision", []) or []
        qes = [e.get("quality_efficiency") for e in entries if isinstance(e, dict) and e.get("quality_efficiency")]
        if qes:
            return float(max(qes)), "QE"
        thrs = [
            e.get("best_throughput_tokens_per_sec")
            for e in entries
            if isinstance(e, dict) and e.get("best_throughput_tokens_per_sec")
        ]
        if thrs:
            return float(max(thrs)), "tok/s (quant)"

    return None, ""


def load_all_results() -> list[dict]:
    records = []
    for tier in ("verified", "community"):
        tier_dir = RESULTS_DIR / tier
        if not tier_dir.exists():
            continue
        for sub_dir in sorted(tier_dir.iterdir()):
            rj = sub_dir / "result.json"
            if not sub_dir.is_dir() or not rj.exists():
                continue
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                continue
            val, unit = extract_primary_metric(data)
            records.append(
                {
                    "path": sub_dir.resolve(),
                    "name": sub_dir.name,
                    "tier": tier,
                    "suite_id": data.get("suite_id", "unknown"),
                    "chip_name": (data.get("chip") or {}).get("name", "Unknown"),
                    "chip_count": (data.get("chip") or {}).get("count", 1),
                    "metric": val,
                    "unit": unit,
                    "accuracy_valid": (data.get("accuracy") or {}).get("valid"),
                }
            )
    return records


def format_preview(target_dirs: list[Path]) -> str:
    all_records = load_all_results()
    by_path = {r["path"]: r for r in all_records}

    rows_md: list[str] = []
    for d in target_dirs:
        resolved = d.resolve()
        rec = by_path.get(resolved)
        if rec is None:
            rj = resolved / "result.json"
            if not rj.exists():
                continue
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                continue
            val, unit = extract_primary_metric(data)
            rec = {
                "path": resolved,
                "name": resolved.name,
                "tier": "community",
                "suite_id": data.get("suite_id", "unknown"),
                "chip_name": (data.get("chip") or {}).get("name", "Unknown"),
                "chip_count": (data.get("chip") or {}).get("count", 1),
                "metric": val,
                "unit": unit,
                "accuracy_valid": (data.get("accuracy") or {}).get("valid"),
            }
            all_records.append(rec)

        suite = rec["suite_id"]
        chip = rec["chip_name"]
        metric = rec["metric"]
        unit = rec["unit"]

        if metric is None:
            metric_str = "—"
            overall_str = "—"
            same_chip_str = "—"
        else:
            metric_str = f"{metric:,.1f} {unit}"
            suite_peers = [
                r for r in all_records
                if r["suite_id"] == suite and r["metric"] is not None
            ]
            suite_peers.sort(key=lambda r: r["metric"], reverse=True)
            overall_rank = next(
                (i + 1 for i, r in enumerate(suite_peers) if r["path"] == rec["path"]),
                len(suite_peers),
            )
            leader = suite_peers[0] if suite_peers else None
            if leader and leader["path"] != rec["path"] and leader["metric"]:
                pct = (metric / leader["metric"]) * 100.0
                overall_str = f"#{overall_rank}/{len(suite_peers)} ({pct:.0f}% of #1)"
            else:
                overall_str = f"#{overall_rank}/{len(suite_peers)} (New #1)"

            chip_peers = [r for r in suite_peers if r["chip_name"] == chip]
            chip_rank = next(
                (i + 1 for i, r in enumerate(chip_peers) if r["path"] == rec["path"]),
                len(chip_peers),
            )
            same_chip_str = f"#{chip_rank}/{len(chip_peers)}"

        acc = rec["accuracy_valid"]
        acc_str = "PASS" if acc is True else ("FAIL" if acc is False else "N/A")
        chip_label = f"{chip} ×{rec['chip_count']}" if rec["chip_count"] and rec["chip_count"] > 1 else chip
        rows_md.append(
            f"| `{rec['name']}` | {chip_label} | `{suite}` | {metric_str} | {overall_str} | {same_chip_str} | {acc_str} |"
        )

    if not rows_md:
        return ""

    header = (
        "### Projected Leaderboard Preview\n\n"
        "| Submission | Hardware | Suite | Primary Metric | Suite Rank | Same-Chip Rank | Accuracy Gate |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    footer = (
        "\n\n> Rankings are computed against all current `verified` + `community` results in the same suite. "
        "Once merged, the live leaderboard updates automatically via GitHub Pages."
    )
    return header + "\n".join(rows_md) + footer


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview projected leaderboard rankings for PR submissions.")
    parser.add_argument("dirs", nargs="*", help="Submission directories to preview")
    parser.add_argument("--dirs-file", type=Path, help="File with one submission directory per line")
    parser.add_argument("--output", type=Path, help="Optional output Markdown file path")
    args = parser.parse_args()

    target_dirs: list[Path] = [Path(d) for d in args.dirs]
    if args.dirs_file and args.dirs_file.exists():
        for line in args.dirs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                target_dirs.append(Path(line))

    md = format_preview(target_dirs)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
