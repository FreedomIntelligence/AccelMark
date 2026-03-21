"""
AccelMark Leaderboard Generator
Reads all result.json files from results/ and generates leaderboard/site/leaderboard.js.

Usage:
    python leaderboard/generate.py
"""

import json
from pathlib import Path

# Build ranking lookup per chip model per suite
import statistics
from collections import defaultdict
import numpy as np


RESULTS_DIR = Path("results")
SITE_DIR = Path("leaderboard/site")


def load_results() -> list[dict]:
    results = []
    for tier in ["verified", "community"]:
        tier_dir = RESULTS_DIR / tier
        if not tier_dir.exists():
            continue
        for submission_dir in sorted(tier_dir.iterdir()):
            result_path = submission_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                with open(result_path) as f:
                    data = json.load(f)
                data["_tier"] = tier
                data["_submission_name"] = submission_dir.name
                results.append(data)
            except Exception as e:
                print(f"Warning: could not load {result_path}: {e}")
    return results


def extract_row(result: dict) -> dict:
    chip = result.get("chip", {})
    software = result.get("software", {})
    model = result.get("model", {})
    task = result.get("task", {})
    metrics = result.get("metrics", {})
    accuracy = result.get("accuracy", {})
    meta = result.get("meta", {})
    derived = metrics.get("derived", {})

    # Primary metric depends on scenario
    scenario = task.get("scenario")
    primary_metric = None
    primary_metric_label = None

    if scenario == "offline":
        offline = metrics.get("offline", {})
        rows = offline.get("results_by_batch_size", []) if offline else []
        valid = [r for r in rows if not r.get("oom")]
        if valid:
            primary_metric = max(r.get("throughput_tokens_per_sec", 0) for r in valid)
            primary_metric_label = "tokens/sec (offline)"
    elif scenario == "online":
        online = metrics.get("online", {})
        if online:
            primary_metric = online.get("max_valid_qps")
            primary_metric_label = "max valid QPS"
    elif scenario == "training":
        training = metrics.get("training", {})
        if training:
            primary_metric = training.get("tokens_per_sec")
            primary_metric_label = "tokens/sec (training)"

    return {
        "submission": result.get("_submission_name"),
        "tier": result.get("_tier"),
        "chip": chip.get("name"),
        "vendor": chip.get("vendor"),
        "chip_count": chip.get("count"),
        "framework": software.get("framework"),
        "model": model.get("model_id", "").split("/")[-1],
        "precision": model.get("precision"),
        "suite": result.get("suite_id"),
        "scenario": scenario,
        "primary_metric": primary_metric,
        "primary_metric_label": primary_metric_label,
        "accuracy_valid": accuracy.get("valid"),
        "accuracy_score": accuracy.get("subset_score"),
        "tokens_per_watt": derived.get("tokens_per_sec_per_watt"),
        "date": meta.get("date"),
        "submitted_by": meta.get("submitted_by"),
    }


def generate_api(results: list[dict], output_dir: Path) -> None:
    """
    Generate static JSON files that the OpenClaw Skill queries for rankings.
    Written to leaderboard/site/api/ directory.

    Files generated:
      api/rank.json   — per-submission ranking: submission_name → {rank, total, percentile}
      api/chips.json  — list of all chips with stats
      api/index.json  — lightweight summary of all submissions (for Skill quick lookup)
    """
    api_dir = output_dir / "api"
    api_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # Group best throughput per submission, keyed by chip model
    # ----------------------------------------------------------------
    # Structure: by_chip[chip_name] = list of (submission_name, throughput)
    by_chip: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for r in results:
        chip_name = r.get("chip", {}).get("name", "Unknown")
        submission_name = r.get("_submission_name", "unknown")
        offline = r.get("metrics", {}).get("offline")
        if not offline:
            continue
        rows = offline.get("results_by_batch_size", [])
        valid = [
            row for row in rows
            if not row.get("oom") and row.get("throughput_tokens_per_sec")
        ]
        if not valid:
            continue
        best_thr = max(row["throughput_tokens_per_sec"] for row in valid)
        by_chip[chip_name].append((submission_name, best_thr))

    # ----------------------------------------------------------------
    # Build per-submission rank lookup
    # ----------------------------------------------------------------
    # rank_data[submission_name] = {rank, total, percentile, chip_name, best_throughput}
    rank_data: dict[str, dict] = {}

    for chip_name, entries in by_chip.items():
        # Sort descending by throughput
        sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
        total = len(sorted_entries)

        for rank_idx, (submission_name, thr) in enumerate(sorted_entries):
            rank = rank_idx + 1  # 1-based
            # percentile: what fraction of same-chip submissions this beats
            percentile = round((total - rank) / total * 100, 1) if total > 1 else 100.0
            rank_data[submission_name] = {
                "chip_name": chip_name,
                "rank": rank,
                "total": total,
                "percentile": percentile,
                "best_throughput_tokens_per_sec": thr,
            }

    with open(api_dir / "rank.json", "w") as f:
        json.dump(rank_data, f, indent=2)

    # ----------------------------------------------------------------
    # Build chips summary index
    # ----------------------------------------------------------------
    chips = []
    for chip_name, entries in by_chip.items():
        throughputs = [thr for _, thr in entries]
        chips.append({
            "name": chip_name,
            "submission_count": len(entries),
            "best_throughput_tokens_per_sec": max(throughputs),
            "median_throughput_tokens_per_sec": round(statistics.median(throughputs), 1),
        })
    chips.sort(key=lambda x: x["best_throughput_tokens_per_sec"], reverse=True)

    with open(api_dir / "chips.json", "w") as f:
        json.dump(chips, f, indent=2)

    # ----------------------------------------------------------------
    # Build lightweight index for Skill lookup (chip_name → best stats)
    # ----------------------------------------------------------------
    # OpenClaw Skill queries this to find "what's the best result for my chip"
    chip_index: dict[str, dict] = {}
    for chip_name, entries in by_chip.items():
        throughputs = [thr for _, thr in entries]
        chip_index[chip_name] = {
            "submission_count": len(entries),
            "best_throughput_tokens_per_sec": max(throughputs),
            "median_throughput_tokens_per_sec": round(statistics.median(throughputs), 1),
            "p25_throughput_tokens_per_sec": round(
                sorted(throughputs)[len(throughputs) // 4], 1
            ) if len(throughputs) >= 4 else None,
        }

    with open(api_dir / "index.json", "w") as f:
        json.dump(chip_index, f, indent=2)

    print(f"API files written to {api_dir}/")
    print(f"  rank.json:  {len(rank_data)} submissions indexed")
    print(f"  chips.json: {len(chips)} chips listed")
    print(f"  index.json: {len(chip_index)} chips in lookup table")

def main():
    results = load_results()
    print(f"Loaded {len(results)} results.")

    rows = [extract_row(r) for r in results]

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SITE_DIR / "leaderboard.js"
    with open(out_path, "w") as f:
        f.write("// Auto-generated by leaderboard/generate.py. Do not edit manually.\n")
        f.write(f"const LEADERBOARD_DATA = {json.dumps(rows, indent=2)};\n")

    print(f"Leaderboard data written to {out_path} ({len(rows)} rows).")

    generate_api(results, SITE_DIR)


if __name__ == "__main__":
    main()
