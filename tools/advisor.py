#!/usr/bin/env python3
"""Zero-GPU AccelMark leaderboard advisor, engine comparator, and gap explorer.

Reads the generated static JSON data in leaderboard/site/api/ (auto-generating
it on a fresh clone if missing) so users and coding agents can compare chips,
engines, and cloud cost-efficiency directly from the terminal.

Examples:
    python tools/advisor.py --suite suite_A
    python tools/advisor.py --suite suite_A --budget 2.0 --tier reference
    python tools/advisor.py --compare "RTX 4090" "RTX 3090" "A100"
    python tools/advisor.py --engine-compare "A100"
    python tools/advisor.py --wanted --format markdown
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def _load_json(path: Path) -> Any:
    if path.parent == API_DIR and not path.exists():
        _ensure_api_data()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 'python leaderboard/generate.py' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _match_tier(info: dict, runner_tier: str) -> bool:
    if runner_tier == "all":
        return True
    return (info.get("runner_tier") or "reference").lower() == runner_tier


def _emit_markdown_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def cmd_suite(
    suite_id: str,
    limit: int,
    budget: float | None,
    runner_tier: str,
    fmt: str,
) -> int:
    rank = _load_json(API_DIR / "rank.json")
    pricing = _load_json(PRICING_PATH) if PRICING_PATH.exists() else {}

    records = []
    for sub_id, info in rank.items():
        if info.get("suite_id") != suite_id:
            continue
        if not _match_tier(info, runner_tier):
            continue
        chip = info.get("chip_name", "Unknown")
        metric = float(info.get("primary_metric") or 0.0)
        price = (pricing.get(chip) or {}).get("reference_price_usd_per_hr")
        if budget is not None:
            if price is None or price > budget:
                continue
        tok_per_dollar = (metric * 3600.0 / price) if price and metric else None
        records.append(
            {
                "submission": sub_id,
                "chip": chip,
                "framework": info.get("framework") or "—",
                "runner_tier": info.get("runner_tier") or "reference",
                "tier": info.get("tier", "community"),
                "primary_metric": metric,
                "price_usd_per_hr": price,
                "tokens_per_dollar": round(tok_per_dollar, 1) if tok_per_dollar else None,
            }
        )

    if not records:
        msg = f"No submissions matched suite={suite_id}"
        if budget is not None:
            msg += f" with budget <= ${budget:.2f}/hr"
        if runner_tier != "all":
            msg += f" and runner_tier={runner_tier}"
        print(msg + ".")
        return 1

    if budget is not None:
        records.sort(key=lambda r: (r["tokens_per_dollar"] or 0.0, r["primary_metric"]), reverse=True)
    else:
        records.sort(key=lambda r: r["primary_metric"], reverse=True)

    shown = records[:limit]
    if fmt == "json":
        print(json.dumps({"suite_id": suite_id, "budget": budget, "runner_tier": runner_tier, "results": shown}, indent=2))
        return 0

    title = f"AccelMark {suite_id} Top {len(shown)} (of {len(records)})"
    if budget is not None:
        title += f" · Budget <= ${budget:.2f}/hr (sorted by Tok/$)"

    if fmt == "markdown":
        headers = ["Rank", "Chip", "Framework", "Tier", "Metric", "$/hr", "Tok/$"]
        md_rows = [
            [
                f"#{i}",
                r["chip"],
                r["framework"],
                r["tier"],
                f"{r['primary_metric']:,.1f}",
                f"${r['price_usd_per_hr']:.2f}" if r["price_usd_per_hr"] else "—",
                f"{r['tokens_per_dollar']:,.0f}" if r["tokens_per_dollar"] else "—",
            ]
            for i, r in enumerate(shown, 1)
        ]
        _emit_markdown_table(title, headers, md_rows)
        return 0

    print(f"\n=== {title} ===")
    print(f"{'Rank':<5} {'Chip':<28} {'Framework':<12} {'Tier':<10} {'Metric':>10} {'$/hr':>7} {'Tok/$':>12}")
    print("-" * 92)
    for i, r in enumerate(shown, 1):
        price_str = f"${r['price_usd_per_hr']:.2f}" if r["price_usd_per_hr"] else "—"
        tpd_str = f"{r['tokens_per_dollar']:,.0f}" if r["tokens_per_dollar"] else "—"
        print(
            f"#{i:<4} {r['chip']:<28} {r['framework']:<12} {r['tier']:<10} "
            f"{r['primary_metric']:>10,.1f} {price_str:>7} {tpd_str:>12}"
        )
    return 0


def cmd_compare(queries: list[str], runner_tier: str, fmt: str) -> int:
    rank = _load_json(API_DIR / "rank.json")
    pricing = _load_json(PRICING_PATH) if PRICING_PATH.exists() else {}

    matched: dict[str, dict[str, tuple[float, str]]] = {}
    for _, info in rank.items():
        if not _match_tier(info, runner_tier):
            continue
        chip = info.get("chip_name", "")
        if not any(q.lower() in chip.lower() for q in queries):
            continue
        suite = info.get("suite_id", "unknown")
        metric = float(info.get("primary_metric") or 0.0)
        tier = info.get("tier", "community")
        prev = matched.setdefault(chip, {}).get(suite)
        if prev is None or metric > prev[0]:
            matched[chip][suite] = (metric, tier)

    if not matched:
        print(f"No chips matched: {', '.join(queries)}")
        return 1

    suites = sorted({s for m in matched.values() for s in m})

    if fmt == "json":
        payload = {
            chip: {
                "price_usd_per_hr": (pricing.get(chip) or {}).get("reference_price_usd_per_hr"),
                "suites": {s: matched[chip][s][0] for s in suites if s in matched[chip]},
            }
            for chip in sorted(matched)
        }
        print(json.dumps(payload, indent=2))
        return 0

    if fmt == "markdown":
        headers = ["Chip", "$/hr", *suites]
        md_rows = []
        for chip in sorted(matched):
            price = (pricing.get(chip) or {}).get("reference_price_usd_per_hr")
            row = [chip, f"${price:.2f}" if price else "—"]
            for s in suites:
                val = matched[chip].get(s)
                row.append(f"{val[0]:,.1f}" if val else "—")
            md_rows.append(row)
        _emit_markdown_table("Chip Comparison (best primary metric per suite)", headers, md_rows)
        return 0

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


def cmd_engine_compare(chip_query: str, suite_filter: str | None, fmt: str) -> int:
    """Compare inference frameworks/runners head-to-head on matching chip(s)."""
    rank = _load_json(API_DIR / "rank.json")

    # (chip, suite) -> framework -> (best_metric, runner_tier)
    grouped: dict[tuple[str, str], dict[str, tuple[float, str]]] = {}
    for _, info in rank.items():
        chip = info.get("chip_name", "")
        if chip_query.lower() not in chip.lower():
            continue
        suite = info.get("suite_id", "unknown")
        if suite_filter and suite != suite_filter:
            continue
        fw = info.get("framework") or "Unknown"
        rtier = info.get("runner_tier") or "reference"
        metric = float(info.get("primary_metric") or 0.0)
        label = f"{fw} ({rtier})" if rtier != "reference" else fw
        bucket = grouped.setdefault((chip, suite), {})
        prev = bucket.get(label)
        if prev is None or metric > prev[0]:
            bucket[label] = (metric, rtier)

    if not grouped:
        print(f"No submissions matched chip query '{chip_query}'.")
        return 1

    if fmt == "json":
        payload = [
            {
                "chip": chip,
                "suite_id": suite,
                "engines": {
                    fw: metric for fw, (metric, _) in sorted(fws.items(), key=lambda kv: kv[1][0], reverse=True)
                },
            }
            for (chip, suite), fws in sorted(grouped.items())
        ]
        print(json.dumps(payload, indent=2))
        return 0

    rows_out: list[list[str]] = []
    for (chip, suite), fws in sorted(grouped.items()):
        ranked = sorted(fws.items(), key=lambda kv: kv[1][0], reverse=True)
        best_fw, (best_val, _) = ranked[0]
        breakdown = ", ".join(f"{fw}: {val:,.1f}" for fw, (val, _) in ranked)
        spread = (
            f"{best_val / ranked[-1][1][0]:.2f}x"
            if len(ranked) > 1 and ranked[-1][1][0] > 0
            else "1 engine"
        )
        rows_out.append([chip, suite, best_fw, spread, breakdown])

    title = f"Engine Comparison for '{chip_query}'"
    if fmt == "markdown":
        _emit_markdown_table(title, ["Chip", "Suite", "Best Engine", "Spread", "All Engines (tok/s)"], rows_out)
        return 0

    print(f"\n=== {title} ===")
    print(f"{'Chip':<26} {'Suite':<8} {'Best Engine':<20} {'Spread':<9} Engines")
    print("-" * 100)
    for chip, suite, best_fw, spread, breakdown in rows_out:
        print(f"{chip:<26} {suite:<8} {best_fw:<20} {spread:<9} {breakdown}")
    return 0


def cmd_wanted(fmt: str) -> int:
    rank = _load_json(API_DIR / "rank.json")
    chips = _load_json(API_DIR / "chips.json")
    covered = {c["name"] for c in chips}
    pricing_chips = (
        [k for k in _load_json(PRICING_PATH) if not k.startswith("_")]
        if PRICING_PATH.exists()
        else []
    )
    missing_from_pricing = [c for c in pricing_chips if c not in covered]

    if fmt == "json":
        print(
            json.dumps(
                {
                    "covered_chips": len(covered),
                    "total_submissions": len(rank),
                    "missing_priced_accelerators": missing_from_pricing,
                    "wanted_url": "https://freedomintelligence.github.io/AccelMark/#/wanted",
                },
                indent=2,
            )
        )
        return 0

    if fmt == "markdown":
        print(f"### AccelMark Coverage Gaps ({len(covered)} chips, {len(rank)} submissions)\n")
        for c in missing_from_pricing:
            print(f"- {c}")
        print("\nInteractive gaps & one-click commands: https://freedomintelligence.github.io/AccelMark/#/wanted")
        return 0

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
    parser.add_argument("--budget", type=float, default=None, help="Filter --suite by max $/hr and sort by Tok/$")
    parser.add_argument(
        "--tier",
        dest="runner_tier",
        default="all",
        choices=["all", "reference", "optimized", "experimental"],
        help="Filter by runner tier (default: all)",
    )
    parser.add_argument("--compare", nargs="+", help="Compare chips matching substrings (e.g. '4090' '3090')")
    parser.add_argument("--engine-compare", dest="engine_compare", help="Compare frameworks/engines on a chip (e.g. 'A100')")
    parser.add_argument("--wanted", action="store_true", help="Show wanted hardware summary")
    parser.add_argument(
        "--format",
        dest="fmt",
        default="table",
        choices=["table", "markdown", "json"],
        help="Output format: table (default), markdown, or json",
    )
    args = parser.parse_args()

    if args.engine_compare:
        return cmd_engine_compare(args.engine_compare, args.suite, args.fmt)
    if args.suite:
        return cmd_suite(args.suite, args.limit, args.budget, args.runner_tier, args.fmt)
    if args.compare:
        return cmd_compare(args.compare, args.runner_tier, args.fmt)
    if args.wanted:
        return cmd_wanted(args.fmt)

    chips = _load_json(API_DIR / "chips.json")
    print(f"AccelMark static API: {len(chips)} chips indexed.")
    print("Try:")
    print("  python tools/advisor.py --suite suite_A --budget 2.0 --tier reference")
    print("  python tools/advisor.py --compare '4090' '3090' 'A100'")
    print("  python tools/advisor.py --engine-compare 'A100'")
    print("  python tools/advisor.py --wanted --format markdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
