#!/usr/bin/env python3
"""Zero-GPU AccelMark leaderboard advisor & gap explorer.

Reads the generated static JSON data in leaderboard/site/api/ (or regenerates
them if needed) so users and coding agents can compare chips, engines, and
cloud cost-efficiency directly from the terminal.

Examples:
    python tools/advisor.py --suite suite_A
    python tools/advisor.py --compare "RTX 4090" "RTX 3090" "A100"
    python tools/advisor.py --wanted
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "leaderboard" / "site" / "api"
PRICING_PATH = ROOT / "schema" / "cloud_pricing.json"


def _ensure_api_data() -> None:
    """Auto-generate leaderboard/site/api/*.json on a fresh clone if missing."""
    required = (API_DIR / "rank.json", API_DIR / "chips.json")
    if all(p.exists() for p in required):
        return
    gen_script = ROOT / "leaderboard" / "generate.py"
    if not gen_script.exists():
        return
    print("Building local leaderboard index (first run after clone)...", file=sys.stderr)
    subprocess.run(
        [sys.executable, str(gen_script)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        check=True,
    )


def _load_json(path: Path):
    if path.parent == API_DIR and not path.exists():
        _ensure_api_data()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 'python leaderboard/generate.py' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_suite(suite_id: str, limit: int) -> int:
    rank = _load_json(API_DIR / "rank.json")
    pricing = _load_json(PRICING_PATH) if PRICING_PATH.exists() else {}

    rows = [
        (sub_id, info)
        for sub_id, info in rank.items()
        if info.get("suite_id") == suite_id
    ]
    if not rows:
        print(f"No submissions found for {suite_id}.")
        return 1

    rows.sort(key=lambda x: x[1].get("primary_metric") or 0, reverse=True)
    print(f"\n=== AccelMark {suite_id} Top {min(limit, len(rows))} (of {len(rows)}) ===")
    print(f"{'Rank':<5} {'Chip':<30} {'Tier':<10} {'Metric':>12} {'$/hr':>7} {'Tok/$':>12}")
    print("-" * 82)
    for i, (sub_id, info) in enumerate(rows[:limit], 1):
        chip = info.get("chip_name", "Unknown")
        tier = info.get("tier", "community")
        metric = info.get("primary_metric") or 0.0
        price = (pricing.get(chip) or {}).get("reference_price_usd_per_hr")
        price_str = f"${price:.2f}" if price else "—"
        tok_per_dollar = f"{(metric * 3600 / price):,.0f}" if price and metric else "—"
        print(f"#{i:<4} {chip:<30} {tier:<10} {metric:>12,.1f} {price_str:>7} {tok_per_dollar:>12}")
    return 0


def cmd_compare(queries: list[str]) -> int:
    rank = _load_json(API_DIR / "rank.json")
    pricing = _load_json(PRICING_PATH) if PRICING_PATH.exists() else {}

    matched: dict[str, dict[str, tuple[float, str]]] = {}
    for _, info in rank.items():
        chip = info.get("chip_name", "")
        if not any(q.lower() in chip.lower() for q in queries):
            continue
        suite = info.get("suite_id", "unknown")
        metric = info.get("primary_metric") or 0.0
        tier = info.get("tier", "community")
        prev = matched.setdefault(chip, {}).get(suite)
        if prev is None or metric > prev[0]:
            matched[chip][suite] = (metric, tier)

    if not matched:
        print(f"No chips matched: {', '.join(queries)}")
        return 1

    suites = sorted({s for m in matched.values() for s in m})
    print("\n=== Chip Comparison (best primary metric per suite) ===")
    header = f"{'Chip':<32} {'$/hr':>6} " + " ".join(f"{s:>10}" for s in suites)
    print(header)
    print("-" * len(header))
    for chip in sorted(matched):
        price = (pricing.get(chip) or {}).get("reference_price_usd_per_hr")
        price_str = f"${price:.2f}" if price else "—"
        cols = []
        for s in suites:
            val = matched[chip].get(s)
            cols.append(f"{val[0]:>10,.1f}" if val else f"{'—':>10}")
        print(f"{chip:<32} {price_str:>6} " + " ".join(cols))
    return 0


def cmd_wanted() -> int:
    rank = _load_json(API_DIR / "rank.json")
    chips = _load_json(API_DIR / "chips.json")
    covered = {c["name"] for c in chips}
    pricing_chips = [k for k in _load_json(PRICING_PATH) if not k.startswith("_")] if PRICING_PATH.exists() else []
    missing_from_pricing = [c for c in pricing_chips if c not in covered]

    print(f"\nCovered chips on leaderboard: {len(covered)}")
    print(f"Total indexed submissions: {len(rank)}")
    if missing_from_pricing:
        print("\nPriced accelerators missing submissions:")
        for c in missing_from_pricing:
            print(f"  - {c}")
    print("\nSee full interactive gaps & one-click run commands at:")
    print("  https://freedomintelligence.github.io/AccelMark/#/wanted")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Query AccelMark leaderboard data without a GPU.")
    parser.add_argument("--suite", help="Show top submissions for a suite (e.g. suite_A)")
    parser.add_argument("--limit", type=int, default=15, help="Max rows to display (default: 15)")
    parser.add_argument("--compare", nargs="+", help="Compare chips matching substrings (e.g. '4090' '3090')")
    parser.add_argument("--wanted", action="store_true", help="Show wanted hardware summary")
    args = parser.parse_args()

    if args.suite:
        return cmd_suite(args.suite, args.limit)
    if args.compare:
        return cmd_compare(args.compare)
    if args.wanted:
        return cmd_wanted()

    chips = _load_json(API_DIR / "chips.json")
    print(f"AccelMark static API: {len(chips)} chips indexed.")
    print("Try:")
    print("  python tools/advisor.py --suite suite_A")
    print("  python tools/advisor.py --compare '4090' '3090' 'A100'")
    print("  python tools/advisor.py --wanted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
