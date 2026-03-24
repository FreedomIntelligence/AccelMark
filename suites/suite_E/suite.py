"""
Suite E orchestration — Multi-Chip Scaling Benchmark.

Entry point: run(br, args, suite, env_info)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(br, args, suite: dict, env_info: dict) -> None:
    """
    Suite E entry point called by BenchmarkRunner.main().

    Dispatches based on scenario:
    - "default" / "all" → run full multi-chip scaling benchmark
    - anything else → single scenario (offline/accuracy), use generic path
    """
    if args.scenario in ("default", "all"):
        _run_suite_e(br, args, suite)
    else:
        br._setup_logging(args.output_dir)
        br._run_single_scenario(args, suite)


def _run_suite_e(br, args, suite: dict) -> None:
    """Run Suite E: accuracy gate first, then offline at multiple chip counts."""
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    all_counts      = suite.get("chip_counts_all", [1, 2, 4, 8])
    required_counts = suite.get("chip_counts_required", [1, 2])
    max_chips       = getattr(args, "max_chips", None) or 4
    chip_counts     = [c for c in all_counts if c <= max_chips]
    skip_gate       = getattr(args, "skip_accuracy_gate", False)
    platform_script = sys.argv[0]

    if not chip_counts:
        print(f"Error: --max-chips {max_chips} too low. Min: {min(required_counts)}")
        raise SystemExit(1)

    missing_required = [c for c in required_counts if c not in chip_counts]
    if missing_required:
        print(f"Warning: required counts {missing_required} excluded "
              f"by --max-chips={max_chips}.")

    results_summary = []
    total_start     = time.perf_counter()
    acc_result      = None

    print(f"\n{'='*60}")
    print(f"  Suite E — Scaling Efficiency Benchmark")
    print(f"  Chip counts: {chip_counts}")
    print(f"  Base output: {base_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Accuracy gate ─────────────────────────────────────────────
    _default_scenarios_e, _ = br._parse_scenarios_config(suite)
    if "accuracy" in _default_scenarios_e:
        acc_dir  = base_dir / "accuracy"
        acc_dir.mkdir(parents=True, exist_ok=True)
        acc_path = acc_dir / "accuracy.json"

        if acc_path.exists():
            try:
                with open(acc_path) as f:
                    acc_result = json.load(f)
                print(
                    f"\n  Accuracy already done — loading from {acc_path}\n"
                    f"  Score: {acc_result.get('subset_score')}, "
                    f"valid={acc_result.get('valid')}\n"
                )
            except Exception as e:
                print(f"  Warning: could not load existing accuracy.json ({e}) "
                      f"— re-running.")
                acc_result = None

        if acc_result is None:
            print(f"\n{'='*60}")
            print(f"  Step 1: Accuracy Gate (1× chip)")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, platform_script,
                "--suite",    args.suite,
                "--scenario", "accuracy",
                "--output-dir", str(acc_dir),
                "--tensor-parallel-size", "1",
            ]
            if getattr(args, "model_path", None):
                cmd += ["--model-path", args.model_path]

            try:
                subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
                if acc_path.exists():
                    with open(acc_path) as f:
                        acc_result = json.load(f)
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Accuracy subprocess failed (return code {e.returncode})")
                if not skip_gate:
                    print("  Aborting Suite E. Use --skip-accuracy-gate to override.")
                    return
                print("  --skip-accuracy-gate set — continuing anyway.\n")

        if acc_result and not acc_result.get("valid") and not skip_gate:
            delta     = acc_result.get("baseline_delta", "?")
            threshold = suite.get("accuracy_threshold_delta", 0.03)
            print(
                f"\n  ✗ ACCURACY GATE FAILED\n"
                f"  Score: {acc_result.get('subset_score')}\n"
                f"  Delta: {delta} (min allowed: -{threshold})\n"
                f"  Aborting Suite E. Use --skip-accuracy-gate to override.\n"
            )
            return

        print(f"\n  ✓ Accuracy passed: "
              f"{acc_result.get('subset_score') if acc_result else '?'}\n")

    # ── Step 2: Run chip counts ───────────────────────────────────────────
    for count in chip_counts:
        count_dir = base_dir / f"{count}x"
        count_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Running {count}× chips")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, platform_script,
            "--suite",    args.suite,
            "--scenario", "offline",
            "--output-dir", str(count_dir),
            "--tensor-parallel-size", str(count),
            "--skip-accuracy-gate",
        ]
        if getattr(args, "model_path", None):
            cmd += ["--model-path", args.model_path]
        if getattr(args, "enforce_eager", False):
            cmd += ["--enforce-eager"]

        print(f"  Command: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
            results_summary.append((count, "SUCCESS", str(count_dir)))
            print(f"\n✓ {count}× completed")
        except subprocess.CalledProcessError as e:
            results_summary.append(
                (count, f"FAILED: returncode={e.returncode}", str(count_dir))
            )
            print(f"\n✗ {count}× failed (return code {e.returncode})")

        print("  Waiting 10s before next chip count...")
        time.sleep(10)

    total_elapsed = round((time.perf_counter() - total_start) / 60, 1)

    print(f"\n{'='*60}")
    print(f"  Suite E complete ({total_elapsed} min total)")
    print(f"{'='*60}")
    for count, status, _ in results_summary:
        icon = "✓" if status == "SUCCESS" else "✗"
        print(f"  [{icon}] {count}× — {status}")
    print()

    successful = [c for c, status, _ in results_summary if status == "SUCCESS"]
    if not successful:
        print("All chip counts failed — no result.json merged.")
        return

    _merge_suite_e_results(br, base_dir, suite, successful, total_elapsed,
                           accuracy=acc_result)


def _merge_suite_e_results(
    br,
    base_dir: Path,
    suite: dict,
    successful_counts: list,
    total_elapsed_minutes: float,
    accuracy=None,
) -> None:
    """Merge per-chip-count results into one Suite E result with scaling efficiency."""
    count_results = {}
    for count in successful_counts:
        p = base_dir / f"{count}x" / "result.json"
        if p.exists():
            with open(p) as f:
                count_results[count] = json.load(f)

    if not count_results:
        print("No results to merge")
        return

    base_count  = min(count_results.keys())
    base_result = count_results[base_count]

    base_throughput = None
    rows_base = (
        base_result.get("metrics", {}).get("offline", {}).get("results_by_concurrency")
        or base_result.get("metrics", {}).get("offline", {}).get("results_by_batch_size", [])
    )
    valid_base = [r for r in rows_base
                  if not r.get("oom") and r.get("throughput_tokens_per_sec")]
    if valid_base:
        base_throughput = max(r["throughput_tokens_per_sec"] for r in valid_base)

    scaling_results = []
    for count in sorted(count_results.keys()):
        r    = count_results[count]
        rows = (
            r.get("metrics", {}).get("offline", {}).get("results_by_concurrency")
            or r.get("metrics", {}).get("offline", {}).get("results_by_batch_size", [])
        )
        valid = [row for row in rows
                 if not row.get("oom") and row.get("throughput_tokens_per_sec")]
        best  = max((row["throughput_tokens_per_sec"] for row in valid), default=None)

        efficiency = None
        if base_throughput and best and count > 0:
            efficiency = round(best / (base_throughput * count / base_count), 3)

        fixed_rows = []
        for row in rows:
            fixed_row = dict(row)
            thr = row.get("throughput_tokens_per_sec")
            if thr:
                fixed_row["throughput_tokens_per_sec_per_chip"] = round(thr / count, 2)
            fixed_rows.append(fixed_row)

        scaling_results.append({
            "chip_count":                       count,
            "best_throughput_tokens_per_sec":   round(best, 2) if best else None,
            "throughput_tokens_per_sec_per_chip": round(best / count, 2) if best else None,
            "scaling_efficiency":               efficiency,
            "results_by_concurrency":           fixed_rows,
            "result_dir":                       f"{count}x",
        })

    merged = {
        "schema_version":    "1.0",
        "suite_id":          "suite_E",
        "implementation_id": base_result.get("implementation_id"),
        "chip": {
            **base_result["chip"],
            "count": max(successful_counts),
            "_count_note": (
                "Maximum chip count used in this suite. "
                "See task.chip_counts_run for all counts tested."
            ),
        },
        "software": base_result["software"],
        "model":    base_result["model"],
        "task": {
            "scenarios_run":   ["offline"],
            "chip_counts_run": sorted(count_results.keys()),
            "parallelism_note": "Each chip_count uses tensor_parallel_size=N",
            "num_runs":        suite.get("num_runs", 3),
        },
        "metrics": {
            "scaling": {
                "base_chip_count": base_count,
                "base_throughput_tokens_per_sec": (
                    round(base_throughput, 2) if base_throughput else None
                ),
                "results_by_chip_count": scaling_results,
            },
            "derived": {},
        },
        "accuracy": accuracy or base_result.get("accuracy", {
            "subset_score":   None,
            "baseline_delta": None,
            "valid":          False,
            "notes":          "Run with --scenario accuracy to populate.",
        }),
        "meta": {
            **base_result["meta"],
            "benchmark_elapsed_minutes": total_elapsed_minutes,
            "benchmark_elapsed_minutes_note": "Total across all chip counts.",
            "chip_count_dirs": {
                str(c): f"{c}x" for c in sorted(count_results.keys())
            },
        },
    }

    out_path = base_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n  Scaling efficiency summary:")
    print(f"  {'Chips':>6}  {'Throughput':>14}  {'Per chip':>12}  {'Efficiency':>10}")
    print(f"  {'-'*48}")
    for r in scaling_results:
        eff = f"{r['scaling_efficiency']:.3f}" if r["scaling_efficiency"] else "—"
        thr = f"{r['best_throughput_tokens_per_sec']:,.0f}" \
              if r["best_throughput_tokens_per_sec"] else "—"
        per = f"{r['throughput_tokens_per_sec_per_chip']:,.0f}" \
              if r["throughput_tokens_per_sec_per_chip"] else "—"
        print(f"  {r['chip_count']:>6}x  {thr:>14}  {per:>12}  {eff:>10}")

    print(f"\nSuite E merged result written to {out_path}")
