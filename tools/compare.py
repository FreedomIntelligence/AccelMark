"""
AccelMark Result Comparator
Compare two result.json files and report metric deltas.
Used during verification to confirm reproduction.

Usage:
    python tools/compare.py results/community/<submission>/result.json /tmp/verification/result.json
"""

import argparse
import json
import sys
from pathlib import Path


TOLERANCE_PRIMARY = 0.05   # 5% for primary metrics
TOLERANCE_POWER   = 0.10   # 10% for power


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def pct_delta(a, b):
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return abs(a - b) / abs(b)


def compare_offline(a_metrics: dict, b_metrics: dict) -> list[dict]:
    rows = []
    a_rows = {r["batch_size"]: r for r in a_metrics.get("results_by_batch_size", [])}
    b_rows = {r["batch_size"]: r for r in b_metrics.get("results_by_batch_size", [])}

    for bs in sorted(set(a_rows) | set(b_rows)):
        a = a_rows.get(bs, {})
        b = b_rows.get(bs, {})

        thr_a = a.get("throughput_tokens_per_sec")
        thr_b = b.get("throughput_tokens_per_sec")
        delta = pct_delta(thr_a, thr_b)

        rows.append({
            "metric": f"offline.bs={bs}.throughput_tokens_per_sec",
            "submitted": thr_a,
            "reproduced": thr_b,
            "delta_pct": delta,
            "within_tolerance": delta is None or delta <= TOLERANCE_PRIMARY,
        })

    return rows


def compare_online(a_metrics: dict, b_metrics: dict) -> list[dict]:
    rows = []

    qps_a = a_metrics.get("max_valid_qps")
    qps_b = b_metrics.get("max_valid_qps")
    delta = pct_delta(qps_a, qps_b)
    rows.append({
        "metric": "online.max_valid_qps",
        "submitted": qps_a,
        "reproduced": qps_b,
        "delta_pct": delta,
        "within_tolerance": delta is None or delta <= TOLERANCE_PRIMARY,
    })

    return rows


def compare_interactive(a_metrics: dict, b_metrics: dict) -> list[dict]:
    rows = []
    for key in ["ttft_ms_p50", "ttft_ms_p90", "ttft_ms_p99", "tpot_ms_p50", "tpot_ms_p90", "tpot_ms_p99"]:
        a_val = a_metrics.get(key)
        b_val = b_metrics.get(key)
        delta = pct_delta(a_val, b_val)
        rows.append({
            "metric": f"interactive.{key}",
            "submitted": a_val,
            "reproduced": b_val,
            "delta_pct": delta,
            "within_tolerance": delta is None or delta <= TOLERANCE_PRIMARY,
        })
    return rows


def compare_training(a_metrics: dict, b_metrics: dict) -> list[dict]:
    rows = []
    for key in ["tokens_per_sec", "tokens_per_sec_per_chip"]:
        a_val = a_metrics.get(key)
        b_val = b_metrics.get(key)
        delta = pct_delta(a_val, b_val)
        rows.append({
            "metric": f"training.{key}",
            "submitted": a_val,
            "reproduced": b_val,
            "delta_pct": delta,
            "within_tolerance": delta is None or delta <= TOLERANCE_PRIMARY,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submitted", help="Path to submitted result.json")
    parser.add_argument("reproduced", help="Path to reproduced result.json")
    args = parser.parse_args()

    a = load(args.submitted)
    b = load(args.reproduced)

    a_metrics = a.get("metrics", {})
    b_metrics = b.get("metrics", {})

    comparison_rows = []

    if a_metrics.get("offline") and b_metrics.get("offline"):
        comparison_rows.extend(compare_offline(a_metrics["offline"], b_metrics["offline"]))
    if a_metrics.get("online") and b_metrics.get("online"):
        comparison_rows.extend(compare_online(a_metrics["online"], b_metrics["online"]))
    if a_metrics.get("interactive") and b_metrics.get("interactive"):
        comparison_rows.extend(compare_interactive(a_metrics["interactive"], b_metrics["interactive"]))
    if a_metrics.get("training") and b_metrics.get("training"):
        comparison_rows.extend(compare_training(a_metrics["training"], b_metrics["training"]))

    if not comparison_rows:
        print("WARNING: No comparable metrics found.")
        sys.exit(1)

    print(f"{'Metric':<50} {'Submitted':>12} {'Reproduced':>12} {'Delta%':>8} {'OK':>4}")
    print("-" * 92)

    all_ok = True
    for row in comparison_rows:
        delta_str = f"{row['delta_pct'] * 100:.1f}%" if row["delta_pct"] is not None else "N/A"
        ok_str = "✓" if row["within_tolerance"] else "✗"
        if not row["within_tolerance"]:
            all_ok = False
        sub_str = f"{row['submitted']:.2f}" if isinstance(row["submitted"], float) else str(row["submitted"])
        rep_str = f"{row['reproduced']:.2f}" if isinstance(row["reproduced"], float) else str(row["reproduced"])
        print(f"{row['metric']:<50} {sub_str:>12} {rep_str:>12} {delta_str:>8} {ok_str:>4}")

    print()
    if all_ok:
        print("✓ All metrics within tolerance. Result can be verified.")
        sys.exit(0)
    else:
        print("✗ Some metrics exceed tolerance. Do not promote to verified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
