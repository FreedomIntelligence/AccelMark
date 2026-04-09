"""
Suite C orchestration — Quantization Efficiency Benchmark.

This module is loaded dynamically by BenchmarkRunner.main() when
args.suite == "suite_C". It receives the runner instance as `br`
and calls its infrastructure methods directly.

Entry point: run(br, args, suite, env_info)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# _REPO_ROOT is needed for subprocess cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(br, args, suite: dict, env_info: dict) -> None:
    """
    Suite C entry point called by BenchmarkRunner.main().

    Dispatches based on scenario:
    - "default" / "all" with no --precision — run full multi-format benchmark
      (precision loop: BF16/FP8/W8A8/W8A16/W4A16, each as a subprocess)
    - explicit scenario (e.g. "offline", "online") with no --precision —
      run that single scenario across all precision formats, producing the
      same folder structure: bf16/offline/, fp8/offline/, etc.
    - any scenario with --precision set — per-precision subprocess called by
      the precision loop above; use generic multi-scenario or single-scenario path
    """
    if getattr(args, "precision", None) is None:
        # Top-level invocation — run precision loop for all formats
        _run_suite_c(br, args, suite, env_info)
    elif args.scenario in ("default", "all"):
        # Per-precision subprocess dispatched by the precision loop (default/all)
        br._run_all_scenarios(args, suite)
    else:
        # Per-precision subprocess dispatched by the precision loop (single scenario)
        br._setup_logging(args.output_dir)
        br._run_single_scenario(args, suite)


def _run_suite_c(br, args, suite: dict, env_info: dict) -> None:
    """
    Run Suite C: per-format accuracy + offline benchmark for each
    supported quantization format.

    Each format runs as a separate subprocess for clean GPU state.
    Accuracy is NOT a gate — it runs per format and records results
    without blocking. Missing baselines (placeholders) are allowed.

    Format selection:
    - Always includes BF16 (baseline)
    - Other formats: intersection of suite["precision_levels"] and
      runner.SUPPORTED_QUANTIZATIONS
    - Formats the runner doesn't declare are skipped with a warning
    """
    base_dir     = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Compute run_id/run_name once here — before any subprocess dispatch —
    # using the suite-level model_id (base model) and current settings.
    # This is the authoritative identity for the whole Suite C run.
    suite_run_id   = br._compute_run_id(args, suite, env_info)
    suite_run_name = br._compute_run_name(args, suite, env_info)

    precision_model_map  = suite.get("precision_model_map", {})
    all_precisions       = suite.get("precision_levels", ["BF16"])
    default_scenarios, _ = br._parse_scenarios_config(suite)
    platform_script      = sys.argv[0]
    total_start          = time.time()

    # ── Resolve which formats this runner supports ────────────────────────
    # Detect hardware-supported precisions to pick the right full-precision baseline.
    # On Ampere+ (A100, A800, H20) this is ["BF16","FP16","FP32"].
    # On V100/T4 (no BF16) this is ["FP16","FP32"].
    hw_precisions     = br._detect_supported_precisions(env_info)
    baseline_precision = "BF16" if "BF16" in hw_precisions else "FP16"

    # runner_backends   = [b.lower() for b in br.SUPPORTED_QUANTIZATION_BACKENDS]
    precisions_to_run = []
    skipped           = []   # runner doesn't declare support for this backend
    for p in all_precisions:
        if p == baseline_precision:
            precisions_to_run.append(p)
        elif p in ("BF16", "FP16"):
            # The other full-precision baseline — skip silently, not the hw baseline.
            skipped.append(p)
        else:
            # # Quantized format — only gate on whether the runner declares support
            # # for this backend. Hardware compatibility (e.g. FP8 on V100) is left
            # # to the inference engine: if the hardware can't run it, the subprocess
            # # fails with the engine's own error, which is recorded in the summary.
            # fmt_entry = precision_model_map.get(p, {})
            # backend   = (fmt_entry.get("engine_kwargs") or {}).get("quantization", "")
            # if not backend or backend.lower() not in runner_backends:
            #     skipped.append(p)
            # else:
            precisions_to_run.append(p)

    print(f"\n{'='*60}")
    print(f"  Suite C — Quantization Efficiency Benchmark")
    print(f"  Formats to run : {precisions_to_run}")
    if skipped:
        print(f"  Skipped        : {skipped} (backend not in SUPPORTED_QUANTIZATION_BACKENDS)")
    print(f"  Base output    : {base_dir}")
    print(f"{'='*60}\n")

    results_summary = []

    # ── Run each precision format as a subprocess ─────────────────────────
    for precision in precisions_to_run:
        fmt_info     = precision_model_map.get(precision)
        if fmt_info is None:
            if precision == baseline_precision:
                # Full-precision baseline: fall back to suite model_id — expected
                fmt_model_id = suite.get("model_id")
            else:
                # Quantized format with no model mapping — refuse to silently
                # run the base model under a quantized precision label
                print(f"  ✗ Skipping {precision}: no entry in precision_model_map "
                      f"and it is not the baseline precision ({baseline_precision}). "
                      f"Add a pre-quantized model_id for this format to suite.json precision_model_map.")
                results_summary.append(
                    (precision, "SKIPPED: missing precision_model_map entry", "")
                )
                continue
        else:
            fmt_model_id = fmt_info.get("model_id") or suite.get("model_id")

        precision_dir = base_dir / precision.lower()
        precision_dir.mkdir(parents=True, exist_ok=True)

        # Skip if this precision already has a completed result for the requested scenario.
        # - default/all: check precision_dir/result.json (the merged per-precision result)
        # - explicit scenario (e.g. offline): check precision_dir/offline/result.json
        if args.scenario in ("default", "all"):
            skip_check_path = precision_dir / "result.json"
        else:
            skip_check_path = precision_dir / args.scenario / "result.json"
        if skip_check_path.exists():
            try:
                with open(skip_check_path) as f:
                    json.load(f)  # validate parseable
                print(
                    f"\n  {precision}/{args.scenario} already done — skipping "
                    f"(found {skip_check_path})"
                )
                results_summary.append((precision, "SUCCESS", str(precision_dir)))
                continue
            except Exception as e:
                print(f"  Warning: existing {skip_check_path} unreadable ({e}) — re-running.")

        print(f"\n{'='*60}")
        print(f"  Precision: {precision}")
        print(f"  Model    : {fmt_model_id}")
        print(f"{'='*60}\n")

        # Resolve local path for this format's model_id
        fmt_model_path = br._resolve_model_path(
            fmt_model_id,
            getattr(args, "model_path", None) if precision == baseline_precision else None,
        )
        # Extract transparency fields set by _resolve_model_path() for this format
        fmt_model_note = getattr(br, "_model_note_override", None)
        fmt_model_name = getattr(br, "_model_name_override", None)
        # Clear so they don't bleed into the next precision iteration
        br.__dict__.pop("_model_note_override", None)
        br.__dict__.pop("_model_name_override", None)

        # Forward the user's --scenario to the per-precision subprocess.
        # - "all"     → "all":     extra scenarios (online, sustained) are included
        # - "default" → "default": only default scenarios (accuracy, offline)
        # - explicit  → forwarded as-is (e.g. "offline", "online", "sustained")
        #   The subprocess hits the else branch in run() and calls _run_single_scenario().
        subprocess_scenario = args.scenario
        cmd = [
            sys.executable, platform_script,
            "--suite",      args.suite,
            "--scenario",   subprocess_scenario,
            "--output-dir", str(precision_dir),
            "--precision",  precision,
            "--model-path", fmt_model_path,
            "--skip-accuracy-gate",
        ]
        if fmt_model_note:
            cmd += ["--model-note", fmt_model_note]
        if fmt_model_name:
            cmd += ["--model-name", fmt_model_name]
        # Forward all runner-specific flags (parallelism, enforce_eager, etc.).
        # Unlike Suite E, Suite C does not override tensor_parallel_size per-iteration
        # so get_extra_subprocess_args() output is used as-is without filtering.
        cmd += br.get_extra_subprocess_args(args)

        print(f"  Command: {' '.join(cmd)}\n")

        ok = br._run_subprocess(cmd, precision)
        if ok:
            results_summary.append((precision, "SUCCESS", str(precision_dir)))
            print(f"\n  {precision} completed")
        else:
            results_summary.append((precision, "FAILED: subprocess error", str(precision_dir)))


        print("  Waiting 10s before next precision level...")
        time.sleep(10)

    total_elapsed = round((time.time() - total_start) / 60, 1)

    print(f"\n{'='*60}")
    print(f"  Suite C complete ({total_elapsed} min total)")
    print(f"{'='*60}")
    for precision, status, _ in results_summary:
        icon = "✓" if status == "SUCCESS" else "✗"
        print(f"  [{icon}] {precision:6s} — {status}")
    if skipped:
        for p in skipped:
            print(f"  [—] {p:6s} — skipped (backend not in SUPPORTED_QUANTIZATION_BACKENDS)")
    print()

    successful = [p for p, status, _ in results_summary if status == "SUCCESS"]
    if successful:
        _merge_suite_c_results(br, base_dir, suite, successful, total_elapsed,
                               skipped=skipped, run_id=suite_run_id,
                               run_name=suite_run_name)


def _merge_suite_c_results(
    br,
    base_dir: Path,
    suite: dict,
    successful_precisions: list,
    total_elapsed_minutes: float,
    skipped: list = None,
    run_id: str = None,
    run_name: str = None,
) -> None:
    """
    Merge per-precision results into one Suite C result.json.

    Reads from:
      base_dir/{precision}/result.json            — throughput metrics
      base_dir/{precision}/accuracy/accuracy.json — per-format accuracy

    Computes quality_efficiency = throughput × accuracy_score per format.
    Primary metric: best quality_efficiency across all evaluated formats.
    """
    # ── Load per-precision result.json files ──────────────────────────────
    precision_results = {}
    for precision in successful_precisions:
        p = base_dir / precision.lower() / "result.json"
        if p.exists():
            with open(p) as f:
                precision_results[precision] = json.load(f)

    if not precision_results:
        print("No precision results to merge")
        return

    base_result = precision_results.get("BF16") or precision_results.get("FP16") or precision_results[
        next(iter(precision_results))
    ]

    precision_model_map = suite.get("precision_model_map", {})
    all_precisions      = suite.get("precision_levels",
                                    ["BF16", "FP8", "W8A8", "W8A16", "W4A16"])

    # Determine which precision was the full-precision baseline for this run.
    # Prefer BF16 if it was run; fall back to FP16 (V100 path).
    baseline_precision = (
        "BF16" if "BF16" in precision_results
        else "FP16" if "FP16" in precision_results
        else None
    )

    quant_results  = []
    baseline_best  = None

    for precision in all_precisions:
        if precision not in precision_results:
            continue

        r            = precision_results[precision]
        fmt_map_entry = precision_model_map.get(precision, {})
        fmt_model_id  = (fmt_map_entry.get("model_id")
                         or r.get("model", {}).get("model_id"))

        rows = (
            r.get("metrics", {}).get("offline", {}).get("results_by_concurrency")
            or r.get("metrics", {}).get("offline", {}).get("results_by_batch_size", [])
        )
        valid_rows = [
            row for row in rows
            if not row.get("oom") and row.get("throughput_tokens_per_sec")
        ]
        best = max(
            (row["throughput_tokens_per_sec"] for row in valid_rows),
            default=None
        )

        if precision == baseline_precision:
            baseline_best = best

        speedup = (
            round(best / baseline_best, 3)
            if baseline_best and best and precision != baseline_precision
            else (1.0 if precision == baseline_precision else None)
        )

        acc_score = None
        acc_delta = None
        acc_valid = None
        acc_file  = base_dir / precision.lower() / "accuracy" / "accuracy.json"
        if acc_file.exists():
            try:
                with open(acc_file) as f:
                    acc_data  = json.load(f)
                acc_score = acc_data.get("subset_score")
                acc_delta = acc_data.get("baseline_delta")
                acc_valid = acc_data.get("valid")
            except Exception as e:
                print(f"  Warning: could not load {acc_file}: {e}")

        quality_eff = round(best * acc_score, 1) if best and acc_score else None
        eff_dtype   = r.get("model", {}).get("effective_dtype")
        quant_meth  = r.get("model", {}).get("quantization_method")

        quant_results.append({
            "precision":                      precision,
            "model_id":                       fmt_model_id,
            "best_throughput_tokens_per_sec": round(best, 2) if best else None,
            "accuracy_score":                 acc_score,
            "accuracy_baseline_delta":        acc_delta,
            "accuracy_valid":                 acc_valid,
            "quality_efficiency":             quality_eff,
            "speedup_vs_bf16":                speedup,
            "results_by_concurrency":         rows,
            "result_dir":                     precision.lower(),
            "effective_dtype":                eff_dtype,
            "quantization_method":            quant_meth,
        })

    # Determine scenarios from suite config
    default_scenarios, extra_scenarios = br._parse_scenarios_config(suite)
    all_suite_scenarios = default_scenarios + [
        s for s in extra_scenarios if s not in default_scenarios
    ]
    # Detect which scenarios actually ran across the precision subfolders
    scenarios_run = []
    for s in all_suite_scenarios:
        if s == "accuracy":
            if any((base_dir / p.lower() / "accuracy" / "accuracy.json").exists()
                   for p in successful_precisions):
                scenarios_run.append(s)
        else:
            if any((base_dir / p.lower() / s).exists()
                   for p in successful_precisions):
                scenarios_run.append(s)

    # Build scenario_dirs for all successful precisions
    scenario_dirs = {}
    for precision in successful_precisions:
        for scenario in scenarios_run:
            if scenario == "accuracy":
                continue
            scenario_dir = base_dir / precision.lower() / scenario
            if scenario_dir.exists():
                scenario_dirs[f"{precision.lower()}/{scenario}"] = str(scenario_dir)

    # ── Collect per-format online metrics ─────────────────────────────────
    quantization_online = None
    if "online" in scenarios_run:
        online_by_precision = []
        for precision in successful_precisions:
            p_online = base_dir / precision.lower() / "online" / "result.json"
            if p_online.exists():
                try:
                    with open(p_online) as f:
                        p_result = json.load(f)
                    online_metrics = p_result.get("metrics", {}).get("online", {})
                    if online_metrics:
                        online_by_precision.append({
                            "precision":      precision,
                            "max_valid_qps":  online_metrics.get("max_valid_qps"),
                            "results_by_qps": online_metrics.get("results_by_qps", []),
                        })
                except Exception as e:
                    print(f"  Warning: could not load {p_online}: {e}")
        if online_by_precision:
            quantization_online = {"results_by_precision": online_by_precision}

    # ── Collect per-format sustained metrics ──────────────────────────────
    quantization_sustained = None
    if "sustained" in scenarios_run:
        sustained_by_precision = []
        for precision in successful_precisions:
            p_sus = base_dir / precision.lower() / "sustained" / "result.json"
            if p_sus.exists():
                try:
                    with open(p_sus) as f:
                        p_result = json.load(f)
                    sus_metrics = p_result.get("metrics", {}).get("sustained", {})
                    if sus_metrics:
                        sustained_by_precision.append({
                            "precision":                        precision,
                            "sustained_throughput_tokens_per_sec": sus_metrics.get("sustained_throughput_tokens_per_sec"),
                            "throttle_ratio":                   sus_metrics.get("throttle_ratio"),
                            "throttle_onset_minute":            sus_metrics.get("throttle_onset_minute"),
                            "ttft_p99_drift_ms":                sus_metrics.get("ttft_p99_drift_ms"),
                            "sustained_concurrency":            sus_metrics.get("sustained_concurrency"),
                            "duration_minutes":                 sus_metrics.get("duration_minutes"),
                            "samples":                          sus_metrics.get("samples", []),
                        })
                except Exception as e:
                    print(f"  Warning: could not load {p_sus}: {e}")
        if sustained_by_precision:
            quantization_sustained = {"results_by_precision": sustained_by_precision}

    # ── Build metrics block ────────────────────────────────────────────────
    metrics_block = {
        "quantization": {"results_by_precision": quant_results},
        "derived": {},
    }
    if quantization_online:
        metrics_block["quantization_online"] = quantization_online
    if quantization_sustained:
        metrics_block["quantization_sustained"] = quantization_sustained

    merged = {
        "schema_version":    "1.0",
        "suite_id":          "suite_C",
        "implementation_id": base_result.get("implementation_id"),
        "chip":              base_result["chip"],
        "software":          base_result["software"],
        "model": {
            **base_result["model"],
            "model_id": suite.get("model_id",
                                  base_result["model"].get("model_id", "")),
            "_note": ("suite model_id. Each precision level uses its own "
                      "quantized checkpoint."),
        },
        "task": {
            "scenarios_run":            scenarios_run,
            "precision_levels_run":     successful_precisions,
            "precision_levels_skipped": skipped or [],
            "parallelism":              base_result["task"]["parallelism"],
            "num_runs":                 suite.get("num_runs", 3),
            "extra_config":             base_result["task"].get("extra_config"),
        },
        "metrics": metrics_block,
        "accuracy": None,
        "meta": {
            **base_result["meta"],
            # Always use the suite-level run_id/run_name computed before
            # subprocess dispatch — not the BF16 subprocess's values.
            "run_id":   run_id   or base_result["meta"].get("run_id"),
            "run_name": run_name or base_result["meta"].get("run_name"),
            "benchmark_elapsed_minutes": round(
                sum(
                    (precision_results[p].get("meta", {}).get("benchmark_elapsed_minutes") or 0)
                    for p in successful_precisions if p in precision_results
                ), 1
            ),
            "benchmark_elapsed_minutes_note": "Sum of per-precision benchmark_elapsed_minutes (excludes sleep gaps and orchestrator overhead).",
            "precision_dirs":     {p: p.lower() for p in successful_precisions},
            "scenario_dirs":      scenario_dirs,
            "precision_model_map": {
                p: precision_model_map.get(p, {}) for p in successful_precisions
            },
        },
    }

    out_path = base_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n  Quantization efficiency summary:")
    print(f"  {'Precision':>8}  {'Throughput':>14}  {'Accuracy':>10}"
          f"  {'Delta':>8}  {'Speedup':>8}  {'Quality Eff':>12}")
    print(f"  {'-'*68}")
    for r in quant_results:
        thr  = f"{r['best_throughput_tokens_per_sec']:,.0f}" \
               if r["best_throughput_tokens_per_sec"] else "—"
        acc  = f"{r['accuracy_score']:.4f}" if r["accuracy_score"] else "—"
        delt = (f"{r['accuracy_baseline_delta']:+.4f}"
                if r["accuracy_baseline_delta"] is not None else "—")
        spd  = f"{r['speedup_vs_bf16']:.3f}×" if r["speedup_vs_bf16"] else "—"
        qe   = f"{r['quality_efficiency']:,.0f}" if r["quality_efficiency"] else "—"
        print(f"  {r['precision']:>8}  {thr:>14}  {acc:>10}"
              f"  {delt:>8}  {spd:>8}  {qe:>12}")

    best_qe = max(
        (r["quality_efficiency"] for r in quant_results if r["quality_efficiency"]),
        default=None
    )
    best_thr = max(
        (r["best_throughput_tokens_per_sec"]
         for r in quant_results if r["best_throughput_tokens_per_sec"]),
        default=None
    )
    if best_qe:
        print(f"\n  Best quality efficiency : {best_qe:,.0f}")
    if best_thr:
        print(f"  Best throughput         : {best_thr:,.0f} tok/s")
    print(f"\nSuite C merged result written to {out_path}")