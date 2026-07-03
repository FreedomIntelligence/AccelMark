"""
AccelMark Leaderboard Generator
Reads all result.json files from results/ and generates leaderboard/site/leaderboard.js.

Usage:
    python leaderboard/generate.py
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Load cloud pricing table once at module level
_pricing_cache: dict = {}
_pricing_path = Path("schema/cloud_pricing.json")
if _pricing_path.exists():
    with open(_pricing_path, encoding='utf-8') as _f:
        _pricing_cache = json.load(_f)

RESULTS_DIR = Path("results")
SITE_DIR    = Path("leaderboard/site")
RUNNERS_DIR = Path("runners")


def _precision_to_dtype(precision: str) -> str:
    """Map requested precision name to expected compute dtype."""
    _MAP = {
        "BF16":  "bfloat16",
        "FP16":  "float16",
        "FP32":  "float32",
        "FP8":   "float8_e4m3fn",
        "W8A8":  "int8",
        "W8A16": "float16",
        "W4A16": "float16",
    }
    return _MAP.get((precision or "").upper(), "")


def _get_suite_precision_required(suite_id: str) -> str:
    """Read precision_required from suite.json. Returns 'BF16' if not found."""
    path = Path("suites") / suite_id / "suite.json"
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f).get("precision_required", "BF16")
    except Exception:
        return "BF16"


def _collect_suite_specs() -> dict:
    """Collect UI-relevant per-suite spec from suites/suite_*/suite.json."""
    out: dict = {}
    suites_dir = Path("suites")
    if not suites_dir.exists():
        return out
    for sd in sorted(suites_dir.iterdir()):
        if not sd.is_dir():
            continue
        sf = sd / "suite.json"
        if not sf.exists():
            continue
        try:
            with open(sf, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        sid = data.get("suite_id") or sd.name
        rd = data.get("request_distribution") or {}
        scn = data.get("scenarios") or {}
        spec: dict = {}
        for k in (
            "model_id",
            "model_revision",
            "dataset",
            "precision_required",
            "allowed_precisions",
            "max_model_len",
            "concurrency_levels",
            "online_qps_levels",
            "online_sla_ttft_ms",
        ):
            if k in data and data[k] is not None:
                spec[k] = data[k]
        if rd.get("input_tokens_p50") is not None:
            spec["input_tokens_p50"] = rd["input_tokens_p50"]
        if rd.get("output_tokens_p50") is not None:
            spec["output_tokens_p50"] = rd["output_tokens_p50"]
        if scn.get("default"):
            spec["scenarios_default"] = list(scn["default"])
        if scn.get("extra"):
            spec["scenarios_extra"] = list(scn["extra"])
        out[sid] = spec
    return out


# ── Scenario metric extraction ────────────────────────────────────────────────

def _extract_scenario_metric(result: dict, scenario_name: str) -> dict:
    """Extract the best-throughput info for a single scenario from a result.

    Returns a dict with keys:
        throughput, metric_label, concurrency, peak_memory_gb, is_valid
    """
    metrics = result.get("metrics") or {}
    out = {
        "throughput": None,
        "metric_label": "",
        "concurrency": None,
        "peak_memory_gb": None,
        "is_valid": False,
    }

    if scenario_name == "offline":
        offline = metrics.get("offline")
        if offline:
            rows = offline.get("results_by_concurrency") or offline.get("results_by_batch_size") or []
            valid = [r for r in rows if not r.get("oom") and r.get("throughput_tokens_per_sec")]
            if valid:
                best = max(valid, key=lambda r: r["throughput_tokens_per_sec"])
                out["throughput"] = best["throughput_tokens_per_sec"]
                out["metric_label"] = "tokens/sec"
                out["concurrency"] = best.get("client_concurrency") or best.get("concurrency")
                out["peak_memory_gb"] = best.get("peak_memory_gb")
                out["is_valid"] = True

    elif scenario_name == "online":
        online = metrics.get("online")
        if online:
            qps = online.get("max_valid_qps")
            if qps is not None:
                out["throughput"] = qps
                out["metric_label"] = "max valid QPS"
                out["is_valid"] = True

    elif scenario_name == "interactive":
        # interactive uses the same inference path as offline — reuse offline metric
        offline = metrics.get("offline")
        if offline:
            rows = offline.get("results_by_concurrency") or offline.get("results_by_batch_size") or []
            valid = [r for r in rows if not r.get("oom") and r.get("throughput_tokens_per_sec")]
            if valid:
                best = max(valid, key=lambda r: r["throughput_tokens_per_sec"])
                out["throughput"] = best["throughput_tokens_per_sec"]
                out["metric_label"] = "tokens/sec"
                out["concurrency"] = best.get("client_concurrency") or best.get("concurrency")
                out["peak_memory_gb"] = best.get("peak_memory_gb")
                out["is_valid"] = True

    elif scenario_name == "sustained":
        sustained = metrics.get("sustained")
        if sustained:
            thr = sustained.get("sustained_throughput_tokens_per_sec")
            if thr is not None:
                out["throughput"] = thr
                out["metric_label"] = "tok/s (sustained mean)"
                out["concurrency"] = sustained.get("sustained_concurrency")
                out["is_valid"] = True

    elif scenario_name == "speculative":
        speculative = metrics.get("speculative")
        if speculative:
            rows = speculative.get("results_by_concurrency") or speculative.get("results_by_batch_size") or []
            valid = [r for r in rows if not r.get("oom") and r.get("throughput_tokens_per_sec")]
            if valid:
                best = max(valid, key=lambda r: r["throughput_tokens_per_sec"])
                out["throughput"] = best["throughput_tokens_per_sec"]
                out["metric_label"] = "tok/s (speculative)"
                out["concurrency"] = best.get("client_concurrency") or best.get("concurrency")
                out["peak_memory_gb"] = best.get("peak_memory_gb")
                out["is_valid"] = True

    elif scenario_name == "burst":
        burst = metrics.get("burst")
        if burst:
            ratio = burst.get("burst_degradation_ratio")
            if ratio is not None:
                # Invert: higher = better, same polarity as throughput
                out["throughput"] = round(1.0 - ratio, 4) if ratio <= 1.0 else 0.0
                out["metric_label"] = "1 − degradation_ratio"
                out["is_valid"] = True

    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    results = []
    for tier in ["verified", "community"]:
        tier_dir = RESULTS_DIR / tier
        if not tier_dir.exists():
            continue
        for submission_dir in sorted(tier_dir.iterdir()):
            if not submission_dir.is_dir():
                continue
            result_path = submission_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                with open(result_path, encoding='utf-8') as f:
                    data = json.load(f)
                data["_tier"]            = tier
                data["_submission_name"] = submission_dir.name
                data["_is_suite_level"]  = (
                    "scenarios_run"   in data.get("task", {}) or
                    "chip_counts_run" in data.get("task", {})
                )
                env_path = submission_dir / "env_info.json"
                if env_path.exists():
                    try:
                        with open(env_path, encoding='utf-8') as ef:
                            data["_env_info"] = json.load(ef)
                    except Exception as ee:
                        print(f"Warning: could not load {env_path}: {ee}")
                        data["_env_info"] = {}
                else:
                    data["_env_info"] = {}
                results.append(data)
            except Exception as e:
                print(f"Warning: could not load {result_path}: {e}")
    return results


# ── Detail extraction (modal details tab) ────────────────────────────────────

def extract_detail(result: dict) -> dict:
    """Full detail object for the modal panel, grouped by category."""
    chip        = result.get("chip") or {}
    software    = result.get("software") or {}
    model       = result.get("model") or {}
    task        = result.get("task") or {}
    accuracy    = result.get("accuracy") or {}
    meta        = result.get("meta") or {}
    parallelism = task.get("parallelism") or {}
    env         = result.get("_env_info") or {}

    cpu_info = env.get("cpu", {})
    cpu_str  = None
    if cpu_info.get("model"):
        cores   = cpu_info.get("physical_cores")
        cpu_str = cpu_info["model"] + (f", {cores} cores" if cores else "")

    nics    = env.get("network_interfaces", [])
    nic_str = None
    if nics:
        nic_types = list(dict.fromkeys(n.get("type", "") for n in nics if n.get("type")))
        nic_names = [n.get("name") for n in nics if n.get("name")]
        type_str  = nic_types[0] if nic_types else "unknown"
        names_str = ", ".join(nic_names) if nic_names else ""
        nic_str   = f"{len(nics)}x {type_str}" + (f" ({names_str})" if names_str else "")

    intra = chip.get("interconnect_intra_node")
    if not intra and env.get("accelerator_topology"):
        nv_matches = re.findall(r'NV(\d+)', env["accelerator_topology"])
        if nv_matches:
            intra = f"NVLink {max(int(x) for x in nv_matches)} (full mesh)"

    return {
        "hw_chip":               chip.get("name"),
        "hw_vendor":             chip.get("vendor"),
        "hw_count":              chip.get("count"),
        "hw_memory_gb":          chip.get("memory_gb"),
        "hw_interconnect_intra": intra,
        "hw_interconnect_inter": chip.get("interconnect_inter_node"),
        "hw_cpu":                cpu_str,
        "hw_system_memory_gb":   env.get("system_memory_gb"),
        "hw_pcie":               env.get("pcie_generation"),
        "hw_network":            nic_str,
        "sw_framework":         software.get("framework"),
        "sw_framework_version": software.get("framework_version"),
        "sw_driver":            software.get("driver_version"),
        "sw_runtime":           software.get("runtime_version"),
        "sw_os":                software.get("os"),
        "sw_python":            software.get("python_version"),
        "sw_pytorch":           env.get("pytorch_version"),
        "model_id":              model.get("model_id"),
        "model_revision":        model.get("model_revision"),
        "model_name":            model.get("model_name"),
        "model_note":            model.get("model_note"),
        "model_source":          model.get("model_source"),
        "model_arch":            model.get("architecture"),
        "model_params_b":        model.get("parameter_count_b"),
        "model_precision":       model.get("precision"),
        "model_effective_dtype": model.get("effective_dtype"),
        "model_quant_method":    model.get("quantization_method"),
        "model_format":          model.get("model_format"),
        "run_scenarios":   task.get("scenarios_run"),
        "run_chip_counts": task.get("chip_counts_run"),
        "run_num_runs":    task.get("num_runs"),
        "run_tp":          parallelism.get("tensor_parallel_size"),
        "run_pp":          parallelism.get("pipeline_parallel_size"),
        "run_dp":          parallelism.get("data_parallel_size"),
        "acc_score":          accuracy.get("subset_score"),
        "acc_baseline_delta": accuracy.get("baseline_delta"),
        "acc_valid":          accuracy.get("valid"),
        "acc_notes":          accuracy.get("notes"),
        "meta_submitted_by":     meta.get("submitted_by"),
        "meta_submission_type":  meta.get("submission_type"),
        "meta_date":             meta.get("date"),
        "meta_reproduce_script": meta.get("reproduce_script"),
        "meta_elapsed_min":      meta.get("benchmark_elapsed_minutes"),
        "meta_model_load_sec":   meta.get("model_load_seconds"),
        "meta_start_time":       meta.get("benchmark_start_time"),
        "meta_notes":            meta.get("notes"),
        "env_vendor_details":    env.get("vendor_details") or {},
    }


# ── Implementation extraction (modal impl tab) ───────────────────────────────

def extract_impl(result: dict) -> dict | None:
    impl_id = result.get("implementation_id")
    if not impl_id:
        return None

    meta_path = RUNNERS_DIR / impl_id / "meta.json"
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
    except Exception:
        return None

    return {
        "id":           meta.get("id"),
        "platform":     meta.get("platform"),
        "name":         meta.get("name"),
        "framework":    meta.get("framework"),
        "submitted_by": meta.get("submitted_by"),
        "description":  meta.get("description"),
        "notes":        meta.get("notes"),
        "created":      meta.get("created"),
        "supersedes_chain": meta.get("supersedes_chain"),
        "deprecated_by": meta.get("deprecated_by"),
        "github_url":   f"https://github.com/FreedomIntelligence/AccelMark/tree/main/runners/{impl_id}",
        "runner_url":   f"https://github.com/FreedomIntelligence/AccelMark/blob/main/runners/{impl_id}/runner.py",
    }


# ── Visualization data extraction ─────────────────────────────────────────────

def extract_viz(result: dict, metrics: dict) -> dict:
    """Chart-ready data for the per-suite visualization panel."""
    suite = result.get("suite_id", "")

    def _offline_rows():
        off = metrics.get("offline", {})
        return off.get("results_by_concurrency") or off.get("results_by_batch_size") or []

    def _concurrency_labels(rows):
        return [
            str(r.get("client_concurrency") or r.get("concurrency") or r.get("batch_size", ""))
            for r in rows
        ]

    def _online_block():
        online   = metrics.get("online", {})
        qps_rows = online.get("results_by_qps", [])
        return {
            "labels":        [str(r.get("target_qps", "")) for r in qps_rows],
            "ttft_p50":      [r.get("ttft_ms_p50") for r in qps_rows],
            "ttft_p90":      [r.get("ttft_ms_p90") for r in qps_rows],
            "tpot_p50":      [r.get("tpot_ms_p50") for r in qps_rows],
            "sla_met":       [r.get("sla_met")      for r in qps_rows],
            "max_valid_qps": online.get("max_valid_qps"),
            "ttft_ms_p99_reliability":
                [r.get("ttft_ms_p99_reliability") or {} for r in qps_rows],
        }

    def _interactive_block():
        iv = metrics.get("interactive", {})
        return {
            "ttft_p50": iv.get("ttft_ms_p50"),
            "ttft_p90": iv.get("ttft_ms_p90"),
            "ttft_p99": iv.get("ttft_ms_p99"),
            "tpot_p50": iv.get("tpot_ms_p50"),
            "tpot_p90": iv.get("tpot_ms_p90"),
            "tpot_p99": iv.get("tpot_ms_p99"),
            "ttft_ms_p99_reliability": iv.get("ttft_ms_p99_reliability") or {},
        }

    def _sustained_block():
        s = metrics.get("sustained")
        if not s:
            return None
        samples = s.get("samples", [])
        return {
            "minutes":               [x["minute"] for x in samples],
            "throughput":            [x["throughput_tokens_per_sec"] for x in samples],
            "ttft_p99":              [x.get("ttft_ms_p99") for x in samples],
            "is_warmup":             [x.get("is_warmup", False) for x in samples],
            "sustained_concurrency": s.get("sustained_concurrency"),
            "duration_minutes":      s.get("duration_minutes"),
            "warmup_minutes":        s.get("warmup_minutes"),
            "sustained_throughput":  s.get("sustained_throughput_tokens_per_sec"),
            "throttle_ratio":        s.get("throttle_ratio"),
            "throttle_onset_minute": s.get("throttle_onset_minute"),
            "ttft_p99_drift_ms":     s.get("ttft_p99_drift_ms"),
            "throughput_post_warmup_reliability":
                s.get("throughput_post_warmup_reliability") or {},
            "samples":               samples,
        }

    def _burst_block():
        b = metrics.get("burst")
        if not b:
            return None
        return {
            "burst_steady_qps":             b.get("burst_steady_qps"),
            "burst_peak_qps":               b.get("burst_peak_qps"),
            "steady_ttft_p50_ms":           b.get("steady_ttft_p50_ms"),
            "steady_ttft_p99_ms":           b.get("steady_ttft_p99_ms"),
            "burst_ttft_p50_ms":            b.get("burst_ttft_p50_ms"),
            "burst_ttft_p99_ms":            b.get("burst_ttft_p99_ms"),
            "steady_requests_total":        b.get("steady_requests_total"),
            "burst_requests_total":         b.get("burst_requests_total"),
            "sla_met_during_burst":         b.get("sla_met_during_burst"),
            "burst_degradation_ratio":      b.get("burst_degradation_ratio"),
            "recovery_time_seconds":        b.get("recovery_time_seconds"),
            "recovery_time_seconds_per_cycle":
                b.get("recovery_time_seconds_per_cycle") or [],
            "results_by_cycle":             b.get("results_by_cycle"),
        }

    def _speculative_block():
        spec = metrics.get("speculative")
        if not spec:
            return None
        task = result.get("task") or {}
        rm   = task.get("runtime_metrics") or {}
        rows = spec.get("results_by_concurrency") or spec.get("results_by_batch_size") or []
        valid = [r for r in rows if not r.get("oom") and r.get("throughput_tokens_per_sec")]
        tok_s = max((r["throughput_tokens_per_sec"] for r in valid), default=None) if valid else None
        return {
            "offline_tok_per_sec":  tok_s,
            "acceptance_rate":      rm.get("acceptance_rate"),
            "mean_accepted_tokens": rm.get("mean_accepted_tokens"),
        }

    def _offline_reliability(rows):
        return [r.get("throughput_tokens_per_sec_reliability") or {} for r in rows]

    if suite == "suite_A":
        rows = _offline_rows()
        return {
            "type": "suite_A",
            "offline": {
                "labels":     _concurrency_labels(rows),
                "throughput": [r.get("throughput_tokens_per_sec") for r in rows],
                "memory_gb":  [r.get("peak_memory_gb")            for r in rows],
                "throughput_reliability": _offline_reliability(rows),
            },
            "online":      _online_block(),
            "interactive": _interactive_block(),
            "sustained":   _sustained_block(),
            "speculative": _speculative_block(),
            "burst":       _burst_block(),
        }

    if suite == "suite_B":
        rows = _offline_rows()
        return {
            "type": "suite_B",
            "offline": {
                "labels":              _concurrency_labels(rows),
                "throughput":          [r.get("throughput_tokens_per_sec")          for r in rows],
                "throughput_per_chip": [r.get("throughput_tokens_per_sec_per_chip") for r in rows],
                "memory_gb":           [r.get("peak_memory_gb")                     for r in rows],
                "throughput_reliability": _offline_reliability(rows),
            },
            "online":    _online_block(),
            "sustained": _sustained_block(),
            "burst":     _burst_block(),
        }

    if suite == "suite_D":
        rows = _offline_rows()
        return {
            "type": "suite_D",
            "offline": {
                "labels":     _concurrency_labels(rows),
                "throughput": [r.get("throughput_tokens_per_sec") for r in rows],
                "memory_gb":  [r.get("peak_memory_gb")            for r in rows],
                "throughput_reliability": _offline_reliability(rows),
            },
            "interactive": _interactive_block(),
            "sustained":   _sustained_block(),
            "speculative": _speculative_block(),
        }

    if suite == "suite_C":
        quantization = metrics.get("quantization", {})
        entries      = quantization.get("results_by_precision", [])
        precisions, throughputs, speedups, quality_effs = [], [], [], []
        accuracies, acc_valid, acc_deltas = [], [], []
        effective_dtypes, quant_methods   = [], []
        for e in entries:
            precisions.append(e.get("precision", ""))
            throughputs.append(e.get("best_throughput_tokens_per_sec"))
            speedups.append(e.get("speedup_vs_bf16"))
            quality_effs.append(e.get("quality_efficiency"))
            accuracies.append(e.get("accuracy_score"))
            acc_valid.append(e.get("accuracy_valid"))
            acc_deltas.append(e.get("accuracy_baseline_delta"))
            effective_dtypes.append(e.get("effective_dtype"))
            quant_methods.append(e.get("quantization_method"))
        best_qe = max((q for q in quality_effs if q), default=None)
        bf16_thr = next(
            (throughputs[i] for i, p in enumerate(precisions) if p == "BF16"),
            None
        )

        online_by_precision = None
        q_online = metrics.get("quantization_online", {})
        if q_online:
            online_by_precision = []
            for e in q_online.get("results_by_precision", []):
                qps_rows = e.get("results_by_qps", [])
                online_by_precision.append({
                    "precision":     e.get("precision", ""),
                    "max_valid_qps": e.get("max_valid_qps"),
                    "qps_labels":    [str(r.get("target_qps", "")) for r in qps_rows],
                    "ttft_p50":      [r.get("ttft_ms_p50") for r in qps_rows],
                    "ttft_p99":      [r.get("ttft_ms_p99") for r in qps_rows],
                    "sla_met":       [r.get("sla_met") for r in qps_rows],
                })

        sustained_by_precision = None
        q_sus = metrics.get("quantization_sustained", {})
        if q_sus:
            sustained_by_precision = []
            for e in q_sus.get("results_by_precision", []):
                samples = e.get("samples", [])
                sustained_by_precision.append({
                    "precision":                          e.get("precision", ""),
                    "sustained_throughput_tokens_per_sec": e.get("sustained_throughput_tokens_per_sec"),
                    "throttle_ratio":                     e.get("throttle_ratio"),
                    "throttle_onset_minute":              e.get("throttle_onset_minute"),
                    "ttft_p99_drift_ms":                  e.get("ttft_p99_drift_ms"),
                    "duration_minutes":                   e.get("duration_minutes"),
                    "minutes":    [s["minute"] for s in samples],
                    "throughput": [s["throughput_tokens_per_sec"] for s in samples],
                    "is_warmup":  [s.get("is_warmup", False) for s in samples],
                })

        return {
            "type":               "suite_C",
            "precisions":         precisions,
            "throughput":         throughputs,
            "speedup":            speedups,
            "quality_efficiency": quality_effs,
            "accuracies":         accuracies,
            "acc_valid":          acc_valid,
            "acc_deltas":         acc_deltas,
            "effective_dtypes":   effective_dtypes,
            "quantization_methods": quant_methods,
            "best_quality_eff":   best_qe,
            "bf16_throughput":    bf16_thr,
            "online_by_precision":    online_by_precision,
            "sustained_by_precision": sustained_by_precision,
        }

    if suite == "suite_E":
        scaling = metrics.get("scaling", {})
        entries = scaling.get("results_by_chip_count", [])
        chip_counts, throughputs, efficiencies, per_chip = [], [], [], []
        for e in sorted(entries, key=lambda x: x.get("chip_count", 0)):
            chip_counts.append(e.get("chip_count"))
            throughputs.append(e.get("best_throughput_tokens_per_sec"))
            efficiencies.append(round((e.get("scaling_efficiency") or 0) * 100, 1))
            per_chip.append(e.get("throughput_tokens_per_sec_per_chip"))
        return {
            "type":                "suite_E",
            "chip_counts":         chip_counts,
            "throughput":          throughputs,
            "efficiency_pct":      efficiencies,
            "throughput_per_chip": per_chip,
        }

    if suite == "suite_F":
        rows = _offline_rows()
        return {
            "type": "suite_F",
            "offline": {
                "labels":     _concurrency_labels(rows),
                "throughput": [r.get("throughput_tokens_per_sec") for r in rows],
                "memory_gb":  [r.get("peak_memory_gb")            for r in rows],
                "throughput_reliability": _offline_reliability(rows),
            },
            "online":      _online_block(),
            "interactive": _interactive_block(),
            "sustained":   _sustained_block(),
        }

    if suite == "suite_G":
        task = result.get("task") or {}
        rm   = task.get("runtime_metrics") or {}
        rows = _offline_rows()
        return {
            "type":       "suite_G",
            "offline": {
                "labels":     _concurrency_labels(rows),
                "throughput": [r.get("throughput_tokens_per_sec") for r in rows],
                "memory_gb":  [r.get("peak_memory_gb")            for r in rows],
                "throughput_reliability": _offline_reliability(rows),
            },
            "online":      _online_block(),
            "interactive": _interactive_block(),
            "sustained":   _sustained_block(),
            "runtime_metrics": rm if rm else None,
        }

    if metrics.get("sustained"):
        sustained = metrics.get("sustained", {})
        samples   = sustained.get("samples", [])
        return {
            "type":                  "sustained",
            "minutes":               [s["minute"] for s in samples],
            "throughput":            [s["throughput_tokens_per_sec"] for s in samples],
            "ttft_p99":              [s.get("ttft_ms_p99") for s in samples],
            "is_warmup":             [s.get("is_warmup", False) for s in samples],
            "sustained_concurrency": sustained.get("sustained_concurrency"),
            "duration_minutes":      sustained.get("duration_minutes"),
            "warmup_minutes":        sustained.get("warmup_minutes"),
            "sustained_throughput":  sustained.get("sustained_throughput_tokens_per_sec"),
            "throttle_ratio":        sustained.get("throttle_ratio"),
            "throttle_onset_minute": sustained.get("throttle_onset_minute"),
            "ttft_p99_drift_ms":     sustained.get("ttft_p99_drift_ms"),
            "samples":               samples,
        }

    return {"type": "none"}


# ── Row extraction ────────────────────────────────────────────────────────────

def extract_row(result: dict) -> dict:
    chip     = result.get("chip", {})
    software = result.get("software", {})
    model    = result.get("model", {})
    task     = result.get("task") or {}
    metrics  = result.get("metrics") or {}
    accuracy = result.get("accuracy") or {}
    meta     = result.get("meta") or {}
    derived  = metrics.get("derived") or {}
    is_suite_level = result.get("_is_suite_level", False)
    suite_id       = result.get("suite_id", "")

    offline_throughput      = None
    tokens_per_sec_per_chip = None
    peak_memory_gb          = None

    offline = metrics.get("offline")
    if offline:
        rows  = offline.get("results_by_concurrency") or offline.get("results_by_batch_size") or []
        valid = [r for r in rows if not r.get("oom") and r.get("throughput_tokens_per_sec")]
        if valid:
            offline_throughput      = max(r["throughput_tokens_per_sec"] for r in valid)
            chip_count              = chip.get("count", 1) or 1
            tokens_per_sec_per_chip = round(offline_throughput / chip_count, 1)
        valid_mem = [r for r in rows if not r.get("oom") and r.get("peak_memory_gb")]
        if valid_mem:
            peak_memory_gb = max(
                valid_mem, key=lambda r: r.get("throughput_tokens_per_sec", 0)
            ).get("peak_memory_gb")

    online         = metrics.get("online")
    online_max_qps = online.get("max_valid_qps") if online else None

    interactive          = metrics.get("interactive")
    interactive_ttft_p99 = interactive.get("ttft_ms_p99") if interactive else None

    sustained_throughput   = None
    throttle_ratio         = None
    throttle_onset_minute  = None
    ttft_p99_drift_ms      = None

    sustained = metrics.get("sustained")
    sustained_concurrency  = None
    if sustained:
        sustained_throughput  = sustained.get("sustained_throughput_tokens_per_sec")
        throttle_ratio        = sustained.get("throttle_ratio")
        throttle_onset_minute = sustained.get("throttle_onset_minute")
        ttft_p99_drift_ms     = sustained.get("ttft_p99_drift_ms")
        sustained_concurrency = sustained.get("sustained_concurrency")

    speculative_throughput = None
    speculative_speedup    = None
    speculative_acceptance = None
    runtime_metrics        = task.get("runtime_metrics") or {}

    speculative = metrics.get("speculative")
    if speculative:
        rows  = speculative.get("results_by_concurrency") or speculative.get("results_by_batch_size") or []
        valid = [r for r in rows if not r.get("oom") and r.get("throughput_tokens_per_sec")]
        if valid:
            speculative_throughput = max(r["throughput_tokens_per_sec"] for r in valid)
        speculative_acceptance = runtime_metrics.get("acceptance_rate")
        if speculative_throughput and offline_throughput and offline_throughput > 0:
            speculative_speedup = round(speculative_throughput / offline_throughput, 3)

    burst_degradation      = None
    burst_steady_p99       = None
    burst_p99              = None
    burst_sla_met          = None

    burst = metrics.get("burst")
    if burst:
        burst_degradation = burst.get("burst_degradation_ratio")
        burst_steady_p99  = burst.get("steady_ttft_p99_ms")
        burst_p99         = burst.get("burst_ttft_p99_ms")
        burst_sla_met     = burst.get("sla_met_during_burst")

    scenario = task.get("scenario", "offline")
    if is_suite_level and suite_id not in ("suite_E", "suite_C", "suite_F"):
        primary_metric       = offline_throughput
        primary_metric_label = "tokens/sec (offline)"
    elif scenario == "offline":
        primary_metric       = offline_throughput
        primary_metric_label = "tokens/sec (offline)"
    elif scenario == "online":
        primary_metric       = online_max_qps
        primary_metric_label = "max valid QPS"
    elif scenario == "training":
        training             = metrics.get("training", {})
        primary_metric       = training.get("tokens_per_sec") if training else None
        primary_metric_label = "tokens/sec (training)"
    elif scenario == "sustained":
        primary_metric       = sustained_throughput
        primary_metric_label = "tok/s (sustained mean)"
    elif scenario == "speculative":
        primary_metric       = speculative_throughput
        primary_metric_label = "tok/s (speculative offline)"
    elif scenario == "burst":
        primary_metric       = burst_degradation
        primary_metric_label = "degradation ratio"
    else:
        primary_metric       = None
        primary_metric_label = None

    scaling_efficiency_2x  = None
    scaling_efficiency_4x  = None
    scaling_base_throughput = None

    scaling = metrics.get("scaling")
    if scaling:
        scaling_base_throughput = (
            scaling.get("base_throughput_tokens_per_sec") or
            scaling.get("base_throughput_1x")
        )
        for entry in scaling.get("results_by_chip_count", []):
            count = entry.get("chip_count")
            eff   = entry.get("scaling_efficiency")
            thr   = entry.get("best_throughput_tokens_per_sec")
            if count == 1 and not scaling_base_throughput and thr:
                scaling_base_throughput = thr
            elif count == 2:
                scaling_efficiency_2x = eff
            elif count == 4:
                scaling_efficiency_4x = eff
        if not offline_throughput and scaling_base_throughput:
            offline_throughput   = scaling_base_throughput
            primary_metric       = scaling_base_throughput
            primary_metric_label = "tokens/sec (1x baseline)"

    quant_bf16_throughput  = None
    quant_best_throughput  = None
    quant_best_precision   = None
    quant_int8_speedup     = None
    quant_int4_speedup     = None
    quant_quality_eff      = None

    quantization = metrics.get("quantization")
    if quantization:
        best_qe = None
        for entry in quantization.get("results_by_precision", []):
            p   = entry.get("precision", "")
            thr = entry.get("best_throughput_tokens_per_sec")
            spd = entry.get("speedup_vs_bf16")
            qe  = entry.get("quality_efficiency")

            if p == "BF16":
                quant_bf16_throughput = thr
            elif p in ("W8A8", "W8A16"):
                if quant_int8_speedup is None or p == "W8A16":
                    quant_int8_speedup = spd
            elif p == "W4A16":
                quant_int4_speedup = spd

            if thr and (quant_best_throughput is None or thr > quant_best_throughput):
                quant_best_throughput = thr
                quant_best_precision  = p

            if qe and (best_qe is None or qe > best_qe):
                best_qe           = qe
                quant_quality_eff = qe

        if quant_best_throughput:
            primary_metric       = quant_best_throughput
            primary_metric_label = f"tokens/sec ({quant_best_precision})"
        elif quant_bf16_throughput:
            primary_metric       = quant_bf16_throughput
            primary_metric_label = "tokens/sec (BF16 baseline)"

    memory_gb_per_chip     = chip.get("memory_gb", 0)
    memory_efficiency      = (
        round(offline_throughput / peak_memory_gb, 1)
        if offline_throughput and peak_memory_gb and peak_memory_gb > 0 else None
    )
    memory_utilization_pct = (
        round(peak_memory_gb / memory_gb_per_chip * 100, 1)
        if peak_memory_gb and memory_gb_per_chip else None
    )

    chip_full_name = chip.get("name", "")
    pricing        = _pricing_cache.get(chip_full_name, {})
    providers      = pricing.get("providers", [])
    min_price      = min((p["price_usd_per_hr"] for p in providers), default=None)
    cost_efficiency = (
        round(offline_throughput / min_price, 0)
        if offline_throughput and min_price and min_price > 0 else None
    )

    precision           = model.get("precision", "BF16")
    effective_dtype     = model.get("effective_dtype")
    quantization_method = model.get("quantization_method")
    suite_required      = _get_suite_precision_required(suite_id)
    precision_fallback  = (
        precision.upper() != suite_required.upper()
        if precision and suite_required else False
    )
    precision_emulated = (
        effective_dtype is not None
        and effective_dtype.replace("torch.", "") != _precision_to_dtype(precision)
    )

    return {
        "submission":         result.get("_submission_name"),
        "tier":               result.get("_tier"),
        "is_suite_level":     is_suite_level,
        "chip":               chip_full_name,
        "vendor":             chip.get("vendor"),
        "chip_count":         chip.get("count", 1),
        "memory_gb": memory_gb_per_chip,
        "framework":          software.get("framework"),
        "framework_version":  software.get("framework_version"),
        "model":              model.get("model_id", "").split("/")[-1],
        "precision":          precision,
        "precision_fallback": precision_fallback,
        "precision_emulated": precision_emulated,
        "effective_dtype":    effective_dtype,
        "quantization_method": quantization_method,
        "model_source":  model.get("model_source", "huggingface"),
        "model_name":    model.get("model_name"),
        "model_format":  model.get("model_format"),
        "architecture":  model.get("architecture"),
        "suite":              suite_id,
        "scenario":           "all" if is_suite_level else scenario,
        "primary_metric":          primary_metric,
        "primary_metric_label":    primary_metric_label,
        "tokens_per_sec_per_chip": tokens_per_sec_per_chip,
        "offline_throughput":   offline_throughput,
        "online_max_qps":       online_max_qps,
        "interactive_ttft_p99": interactive_ttft_p99,
        "peak_memory_gb":                     peak_memory_gb,
        "memory_utilization_pct":             memory_utilization_pct,
        "memory_efficiency_toks_per_gb":      memory_efficiency,
        "min_price_usd_per_hr":               min_price,
        "cost_efficiency_toks_per_dollar_hr": cost_efficiency,
        "tokens_per_watt":                    derived.get("tokens_per_sec_per_watt"),
        "accuracy_valid":   accuracy.get("valid"),
        "accuracy_score":   accuracy.get("subset_score"),
        "date":             meta.get("date"),
        "submitted_by":     meta.get("submitted_by"),
        "reproduce_script": meta.get("reproduce_script"),
        "notes":            meta.get("notes"),
        "run_id":           meta.get("run_id"),
        "run_name":         meta.get("run_name"),
        "flagged":          meta.get("flagged"),
        "scaling_efficiency_2x":   scaling_efficiency_2x,
        "scaling_efficiency_4x":   scaling_efficiency_4x,
        "scaling_base_throughput": scaling_base_throughput,
        "quant_bf16_throughput":  quant_bf16_throughput,
        "quant_best_throughput":  quant_best_throughput,
        "quant_best_precision":   quant_best_precision,
        "quant_int8_speedup":     quant_int8_speedup,
        "quant_int4_speedup":     quant_int4_speedup,
        "quant_quality_eff":      quant_quality_eff,
        "sustained_throughput":    sustained_throughput,
        "throttle_ratio":          throttle_ratio,
        "throttle_onset_minute":   throttle_onset_minute,
        "ttft_p99_drift_ms":       ttft_p99_drift_ms,
        "sustained_concurrency":   sustained_concurrency,
        "speculative_throughput":   speculative_throughput,
        "speculative_speedup":     speculative_speedup,
        "speculative_acceptance":  speculative_acceptance,
        "burst_degradation":       burst_degradation,
        "burst_steady_p99":        burst_steady_p99,
        "burst_p99":               burst_p99,
        "burst_sla_met":           burst_sla_met,
        "detail": extract_detail(result),
        "viz":    extract_viz(result, metrics),
        "impl":   extract_impl(result),
        "implementation_id": result.get("implementation_id"),
    }


# ── API generation ────────────────────────────────────────────────────────────

def generate_api(results: list[dict], output_dir: Path) -> None:
    """Generate static JSON API for external tooling (OpenClaw Skill etc.)."""
    api_dir = output_dir / "api"
    api_dir.mkdir(exist_ok=True)

    by_chip_suite: dict[tuple, list] = defaultdict(list)
    by_chip: dict[str, list] = defaultdict(list)

    for r in results:
        chip_name       = r.get("chip", {}).get("name", "Unknown")
        suite_id        = r.get("suite_id", "unknown")
        submission_name = r.get("_submission_name", "unknown")
        tier            = r.get("_tier", "community")

        offline = r.get("metrics", {}).get("offline")
        best_thr = None
        if offline:
            rows  = offline.get("results_by_concurrency") or \
                    offline.get("results_by_batch_size", [])
            valid = [row for row in rows
                     if not row.get("oom") and row.get("throughput_tokens_per_sec")]
            if valid:
                best_thr = max(row["throughput_tokens_per_sec"] for row in valid)

        if best_thr is None:
            scaling = r.get("metrics", {}).get("scaling", {})
            if scaling:
                best_thr = scaling.get("base_throughput_tokens_per_sec")
                if not best_thr:
                    for entry in scaling.get("results_by_chip_count", []):
                        if entry.get("chip_count") == 1:
                            best_thr = entry.get("best_throughput_tokens_per_sec")
                            break

        if best_thr is None:
            quant = r.get("metrics", {}).get("quantization", {})
            if quant:
                qes = [e.get("quality_efficiency")
                       for e in quant.get("results_by_precision", [])
                       if e.get("quality_efficiency")]
                if qes:
                    best_thr = max(qes)

        if not best_thr:
            continue

        by_chip_suite[(chip_name, suite_id)].append((submission_name, best_thr, tier))
        by_chip[chip_name].append((submission_name, best_thr, suite_id, tier))

    rank_data: dict[str, dict] = {}
    for (chip_name, suite_id), entries in by_chip_suite.items():
        sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
        total = len(sorted_entries)
        for rank_idx, (submission_name, metric, tier) in enumerate(sorted_entries):
            rank = rank_idx + 1
            rank_data[submission_name] = {
                "chip_name":    chip_name,
                "suite_id":     suite_id,
                "tier":         tier,
                "rank":         rank,
                "total":        total,
                "percentile":   round((total - rank) / total * 100, 1)
                                if total > 1 else 100.0,
                "primary_metric": metric,
            }
    with open(api_dir / "rank.json", "w") as f:
        json.dump(rank_data, f, indent=2)

    chips = []
    chip_bests: dict[str, float] = {}
    for chip_name, entries in by_chip.items():
        throughputs = [thr for _, thr, _, _ in entries]
        best        = max(throughputs)
        chip_bests[chip_name] = best
        chips.append({
            "name":                             chip_name,
            "submission_count":                 len(entries),
            "best_throughput_tokens_per_sec":   best,
            "median_throughput_tokens_per_sec": round(statistics.median(throughputs), 1),
        })
    chips.sort(key=lambda x: x["best_throughput_tokens_per_sec"], reverse=True)
    with open(api_dir / "chips.json", "w") as f:
        json.dump(chips, f, indent=2)

    chip_index: dict[str, dict] = {}
    for chip_name in by_chip:
        chip_index[chip_name] = {
            "best_throughput_tokens_per_sec": chip_bests[chip_name],
            "suites": {},
        }

    for r in results:
        chip_name = r.get("chip", {}).get("name", "Unknown")
        suite_id  = r.get("suite_id", "unknown")
        if chip_name not in chip_index:
            continue

        metrics   = r.get("metrics", {})
        online    = metrics.get("online")
        iv        = metrics.get("interactive")
        scaling   = metrics.get("scaling")
        sustained = metrics.get("sustained")

        suite_entry = chip_index[chip_name]["suites"].setdefault(suite_id, {})

        offline = metrics.get("offline")
        if offline:
            rows  = offline.get("results_by_concurrency") or \
                    offline.get("results_by_batch_size", [])
            valid = [row for row in rows
                     if not row.get("oom") and row.get("throughput_tokens_per_sec")]
            if valid:
                thr = max(row["throughput_tokens_per_sec"] for row in valid)
                cur = suite_entry.get("best_throughput_tokens_per_sec")
                if cur is None or thr > cur:
                    suite_entry["best_throughput_tokens_per_sec"] = round(thr, 1)

        if online:
            qps = online.get("max_valid_qps")
            if qps is not None:
                cur = suite_entry.get("best_online_max_qps")
                if cur is None or qps > cur:
                    suite_entry["best_online_max_qps"] = qps

        if iv:
            ttft = iv.get("ttft_ms_p99")
            if ttft is not None:
                cur = suite_entry.get("best_interactive_ttft_p99_ms")
                if cur is None or ttft < cur:
                    suite_entry["best_interactive_ttft_p99_ms"] = round(ttft, 1)

        if scaling:
            base_thr = (
                scaling.get("base_throughput_tokens_per_sec") or
                next(
                    (e.get("best_throughput_tokens_per_sec")
                     for e in scaling.get("results_by_chip_count", [])
                     if e.get("chip_count") == 1),
                    None
                )
            )
            if base_thr is not None:
                cur = suite_entry.get("best_throughput_tokens_per_sec")
                if cur is None or base_thr > cur:
                    suite_entry["best_throughput_tokens_per_sec"] = round(base_thr, 1)
            for entry in scaling.get("results_by_chip_count", []):
                count = entry.get("chip_count")
                eff   = entry.get("scaling_efficiency")
                if count == 2 and eff:
                    suite_entry["best_scaling_efficiency_2x"] = eff
                elif count == 4 and eff:
                    suite_entry["best_scaling_efficiency_4x"] = eff

        if sustained:
            s_thr    = sustained.get("sustained_throughput_tokens_per_sec")
            throttle = sustained.get("throttle_ratio")
            if s_thr is not None:
                cur = suite_entry.get("best_sustained_throughput_tokens_per_sec")
                if cur is None or s_thr > cur:
                    suite_entry["best_sustained_throughput_tokens_per_sec"] = round(s_thr, 1)
            if throttle is not None:
                suite_entry["throttle_ratio"] = throttle

        quant = metrics.get("quantization")
        if quant:
            qes = [(e.get("precision"), e.get("quality_efficiency"))
                   for e in quant.get("results_by_precision", [])
                   if e.get("quality_efficiency")]
            if qes:
                best_qe_entry = max(qes, key=lambda x: x[1])
                suite_entry["best_quality_efficiency"]        = best_qe_entry[1]
                suite_entry["best_quality_efficiency_format"] = best_qe_entry[0]

    with open(api_dir / "index.json", "w") as f:
        json.dump(chip_index, f, indent=2)

    suites_meta = {}
    for suite_dir in sorted(Path("suites").iterdir()):
        if not suite_dir.is_dir():
            continue
        suite_json = suite_dir / "suite.json"
        if not suite_json.exists():
            continue
        try:
            with open(suite_json, encoding='utf-8') as f:
                s = json.load(f)
            suite_id = s.get("suite_id", suite_dir.name)
            scenarios_cfg = s.get("scenarios", {})
            if isinstance(scenarios_cfg, list):
                default_scenarios = scenarios_cfg
                extra_scenarios   = []
            else:
                default_scenarios = scenarios_cfg.get("default", [])
                extra_scenarios   = scenarios_cfg.get("extra", [])
            suites_meta[suite_id] = {
                "suite_id":          suite_id,
                "description":       s.get("description", ""),
                "model_id":          s.get("model_id") or s.get("base_model_id"),
                "precision":         s.get("precision_required", "BF16"),
                "default_scenarios": default_scenarios,
                "extra_scenarios":   extra_scenarios,
                "dataset":           s.get("dataset"),
            }
        except Exception as e:
            print(f"Warning: could not read {suite_json}: {e}")

    with open(api_dir / "suites.json", "w") as f:
        json.dump(suites_meta, f, indent=2)

    print(f"API files written to {api_dir}/")
    print(f"  rank.json:   {len(rank_data)} submissions indexed")
    print(f"  chips.json:  {len(chips)} chips listed")
    print(f"  index.json:  {len(chip_index)} chips in lookup table")
    print(f"  suites.json: {len(suites_meta)} suites documented")


# ── Distribution data generation (新增) ───────────────────────────────────────

def generate_distribution_data(results: list[dict], output_dir: Path) -> None:
    """生成性能分布数据，用于分布图视图。

    为每个去重后的提交生成完整元信息（包含所有 scenario 的指标），
    支持前端按 suite / vendor / framework / model / scenario 筛选，
    并按 (chip, suite) 聚合生成分组统计数据。
    """

    # ── 1. 先去重（与 main() 相同的逻辑）─────────────────────────────────
    _seen: dict = {}
    for r in results:
        meta = r.get("meta") or {}
        rid = meta.get("run_id")
        if not rid:
            continue
        suite_id = r.get("suite_id", "")
        # 计算去重用的指标值
        if suite_id == "suite_C":
            quant = (r.get("metrics") or {}).get("quantization", {})
            qes = [e.get("quality_efficiency") for e in quant.get("results_by_precision", [])
                   if e.get("quality_efficiency")]
            metric = max(qes) if qes else 0
        elif suite_id == "suite_E":
            scaling = (r.get("metrics") or {}).get("scaling", {})
            metric = 0
            for e in scaling.get("results_by_chip_count", []):
                if e.get("chip_count") == 4:
                    metric = e.get("scaling_efficiency") or 0
            if not metric:
                for e in scaling.get("results_by_chip_count", []):
                    if e.get("chip_count") == 2:
                        metric = e.get("scaling_efficiency") or 0
            if not metric:
                metric = scaling.get("base_throughput_tokens_per_sec") or 0
        else:
            offline = (r.get("metrics") or {}).get("offline", {})
            rows = offline.get("results_by_concurrency") or offline.get("results_by_batch_size") or []
            valid_rows = [row for row in rows
                          if not row.get("oom") and row.get("throughput_tokens_per_sec")]
            metric = max((row["throughput_tokens_per_sec"] for row in valid_rows), default=0)
        if rid not in _seen or metric > _seen[rid]["metric"]:
            _seen[rid] = {"result": r, "metric": metric}
    deduped = [entry["result"] for entry in _seen.values()]
    print(f"  distribution: {len(results)} raw → {len(deduped)} deduplicated results")

    # ── 2. 构建每个提交的详细数据 ─────────────────────────────────────────
    all_submissions = []

    for r in deduped:
        chip_obj    = r.get("chip") or {}
        chip        = chip_obj.get("name", "Unknown")
        chip_vendor = chip_obj.get("vendor", "")
        chip_count  = chip_obj.get("count", 1)
        memory_gb   = chip_obj.get("memory_gb", 0)
        suite       = r.get("suite_id", "")

        model_obj       = r.get("model") or {}
        model_full      = model_obj.get("model_id", "")
        model_short     = model_full.split("/")[-1] if model_full else ""
        model_params_b  = model_obj.get("parameter_count_b")
        precision       = model_obj.get("precision", "BF16")
        effective_dtype = model_obj.get("effective_dtype")

        software          = r.get("software") or {}
        framework         = software.get("framework", "")
        framework_version = software.get("framework_version", "")

        meta             = r.get("meta") or {}
        submission_name  = r.get("_submission_name", "")
        tier             = r.get("_tier", "community")
        submitted_by     = meta.get("submitted_by", "")
        date             = meta.get("date", "")
        reproduce_script = meta.get("reproduce_script", "")
        run_id           = meta.get("run_id", "")
        impl_id          = r.get("implementation_id", "")

        # 收集该提交跑了哪些 scenario（从 task.scenarios_run 读取）
        task           = r.get("task") or {}
        scenarios_run  = task.get("scenarios_run") or []
        is_suite_level = "scenarios_run" in task or "chip_counts_run" in task

        # 为每个 scenario 提取指标
        scenarios = {}
        for sc_name in scenarios_run:
            scenarios[sc_name] = _extract_scenario_metric(r, sc_name)

        # 如果 scenarios_run 为空（旧格式），至少从 offline 提取
        if not scenarios_run:
            offline_metric = _extract_scenario_metric(r, "offline")
            if offline_metric["is_valid"]:
                scenarios["offline"] = offline_metric
            online_metric = _extract_scenario_metric(r, "online")
            if online_metric["is_valid"]:
                scenarios["online"] = online_metric

        # 处理 Suite E（scaling）：从 metrics.scaling 提取
        suite_primary_thr = None
        suite_primary_label = None
        suite_primary_scenario = "offline"

        if suite == "suite_E":
            scaling = (r.get("metrics") or {}).get("scaling", {})
            for entry in scaling.get("results_by_chip_count", []):
                if entry.get("chip_count") == 1:
                    suite_primary_thr = entry.get("best_throughput_tokens_per_sec")
                    suite_primary_label = "tokens/sec (1x baseline)"
                    suite_primary_scenario = "scaling"
                    break
            if not suite_primary_thr:
                suite_primary_thr = scaling.get("base_throughput_tokens_per_sec")
                suite_primary_label = "tokens/sec (1x baseline)"
                suite_primary_scenario = "scaling"

        # ── Suite C：按精度爆炸，每种精度一条独立提交 ─────────────────
        if suite == "suite_C":
            quant         = (r.get("metrics") or {}).get("quantization", {})
            quant_online  = (r.get("metrics") or {}).get("quantization_online", {})
            quant_sus     = (r.get("metrics") or {}).get("quantization_sustained", {})
            prec_online   = {e.get("precision",""): e.get("max_valid_qps")
                             for e in quant_online.get("results_by_precision", [])}
            prec_sustained = {e.get("precision",""): e.get("sustained_throughput_tokens_per_sec")
                              for e in quant_sus.get("results_by_precision", [])}
            tp = (task.get("parallelism") or {}).get("tensor_parallel_size")

            for entry in quant.get("results_by_precision", []):
                prec = entry.get("precision", "")
                thr  = entry.get("best_throughput_tokens_per_sec")
                if not thr:
                    continue
                prec_sc = {}
                prec_sc["offline"] = {"throughput": thr, "metric_label": f"tokens/sec ({prec})",
                                      "concurrency": None, "peak_memory_gb": None, "is_valid": True}
                qps = prec_online.get(prec)
                if qps is not None:
                    prec_sc["online"] = {"throughput": qps, "metric_label": "max valid QPS",
                                         "concurrency": None, "peak_memory_gb": None, "is_valid": True}
                sus = prec_sustained.get(prec)
                if sus is not None:
                    prec_sc["sustained"] = {"throughput": sus, "metric_label": "tok/s (sustained mean)",
                                            "concurrency": None, "peak_memory_gb": None, "is_valid": True}
                for sc_name in scenarios_run:
                    if sc_name not in prec_sc and sc_name != "accuracy":
                        m = _extract_scenario_metric(r, sc_name)
                        if m["is_valid"]:
                            prec_sc[sc_name] = m

                config = {"concurrency": None, "batch_size": None,
                          "tensor_parallel": tp, "peak_memory_gb": None}
                sub = {
                    "id": f"{run_id or submission_name}_{prec}",
                    "chip": chip, "chip_vendor": chip_vendor,
                    "chip_count": chip_count, "memory_gb": memory_gb,
                    "suite": suite, "model": model_short, "model_full": model_full,
                    "model_params_b": model_params_b,
                    "precision": prec, "effective_dtype": effective_dtype,
                    "framework": framework, "framework_version": framework_version,
                    "tier": tier, "submitted_by": submitted_by,
                    "date": date, "reproduce_script": reproduce_script,
                    "runner_id": impl_id,
                    "scenarios": prec_sc,
                    "primary_scenario": "quantization",
                    "primary_throughput": thr,
                    "primary_metric_label": f"tokens/sec ({prec})",
                    "config": config,
                }
                all_submissions.append(sub)

        else:
            # ── 非 Suite C：标准单条提交 ────────────────────────────

            # 确定 primary_throughput 和 primary_scenario
            primary_throughput = suite_primary_thr
            primary_scenario   = suite_primary_scenario
            primary_label      = suite_primary_label

            if primary_throughput is None:
                _SCENARIO_PRIORITY = ["offline", "online", "sustained", "speculative", "burst"]
                for sc in _SCENARIO_PRIORITY:
                    if sc in scenarios and scenarios[sc]["is_valid"]:
                        primary_throughput = scenarios[sc]["throughput"]
                        primary_scenario   = sc
                        primary_label      = scenarios[sc]["metric_label"]
                        break

            if primary_throughput is None:
                continue

            # 构建最佳配置信息
            best_sc = scenarios.get(primary_scenario) if primary_scenario in scenarios else None
            config = {}
            if best_sc:
                tp = (task.get("parallelism") or {}).get("tensor_parallel_size")
                config = {
                    "concurrency":    best_sc.get("concurrency"),
                    "batch_size":     None,
                    "tensor_parallel": tp,
                    "peak_memory_gb": best_sc.get("peak_memory_gb"),
                }
            else:
                offline = (r.get("metrics") or {}).get("offline", {})
                rows = offline.get("results_by_concurrency") or offline.get("results_by_batch_size") or []
                valid_rows = [row for row in rows
                              if not row.get("oom") and row.get("throughput_tokens_per_sec")]
                if valid_rows:
                    best_row = max(valid_rows, key=lambda row: row["throughput_tokens_per_sec"])
                    tp = (task.get("parallelism") or {}).get("tensor_parallel_size")
                    config = {
                        "concurrency":    best_row.get("client_concurrency") or best_row.get("concurrency"),
                        "batch_size":     best_row.get("batch_size"),
                        "tensor_parallel": tp,
                        "peak_memory_gb": best_row.get("peak_memory_gb"),
                    }

            sub = {
                "id":                  run_id or submission_name,
                "chip":                chip,
                "chip_vendor":         chip_vendor,
                "chip_count":          chip_count,
                "memory_gb":           memory_gb,
                "suite":               suite,
                "model":               model_short,
                "model_full":          model_full,
                "model_params_b":      model_params_b,
                "precision":           precision,
                "effective_dtype":     effective_dtype,
                "framework":           framework,
                "framework_version":   framework_version,
                "tier":                tier,
                "submitted_by":        submitted_by,
                "date":                date,
                "reproduce_script":    reproduce_script,
                "runner_id":           impl_id,
                "scenarios":           scenarios,
                "primary_scenario":    primary_scenario,
                "primary_throughput":  primary_throughput,
                "primary_metric_label": primary_label,
                "config":              config,
            }
            all_submissions.append(sub)

    # ── 3. 按 (chip, suite) 分组聚合 ──────────────────────────────────────
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "submissions": [],
        "throughputs": [],
    })
    for sub in all_submissions:
        key = (sub["chip"], sub["suite"])
        groups[key]["submissions"].append(sub)
        groups[key]["throughputs"].append(sub["primary_throughput"])

    group_list = []
    for (chip, suite), data in groups.items():
        thr_list = sorted(data["throughputs"])
        n = len(thr_list)
        median = thr_list[n // 2]
        best_sub = max(data["submissions"], key=lambda s: s["primary_throughput"])

        # 各 scenario 汇总
        scenario_summary = {}
        for sub in data["submissions"]:
            for sc_name, sc_info in sub["scenarios"].items():
                if sc_name not in scenario_summary:
                    scenario_summary[sc_name] = {
                        "count": 0,
                        "best_throughput": None,
                        "best_framework": "",
                    }
                sm = scenario_summary[sc_name]
                sm["count"] += 1
                if sc_info["is_valid"] and sc_info["throughput"]:
                    if sm["best_throughput"] is None or sc_info["throughput"] > sm["best_throughput"]:
                        sm["best_throughput"] = sc_info["throughput"]
                        sm["best_framework"] = sub["framework"]

        # 标准差
        stddev = None
        if n >= 2:
            stddev = round(statistics.stdev(thr_list), 2)

        group_list.append({
            "chip":               chip,
            "chip_vendor":        best_sub["chip_vendor"],
            "suite":              suite,
            "model":              best_sub["model"],
            "submission_count":   n,
            "best_throughput":    thr_list[-1],
            "median_throughput":  median,
            "min_throughput":     thr_list[0],
            "max_throughput":     thr_list[-1],
            "stddev_throughput":  stddev,
            "scenario_summary":   scenario_summary,
            "best_submission_id": best_sub["id"],
            "best_framework":     best_sub["framework"],
            "best_submitted_by":  best_sub["submitted_by"],
        })

    # 按厂商优先级排序，再按中位数吞吐量降序
    vendor_priority = {"NVIDIA": 1, "Nvidia": 1, "nvidia": 1,
                       "Huawei": 2, "华为": 2,
                       "AMD": 3, "amd": 3,
                       "Google": 4, "Apple": 5,
                       "Moore Threads": 6, "Iluvatar": 6, "Intel": 6}
    group_list.sort(key=lambda g: (
        vendor_priority.get(g["chip_vendor"], 99),
        -(g["median_throughput"] or 0)
    ))

    # ── 4. Suite 元数据 ───────────────────────────────────────────────────
    suite_meta = _collect_suite_specs()

    # ── 5. 写入 distribution.js ───────────────────────────────────────────
    out_path = output_dir / "distribution.js"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by leaderboard/generate.py. Do not edit manually.\n\n")
        f.write(f"const DISTRIBUTION_SUBMISSIONS = {json.dumps(all_submissions, indent=2, ensure_ascii=False)};\n")
        f.write("window.DISTRIBUTION_SUBMISSIONS = DISTRIBUTION_SUBMISSIONS;\n\n")
        f.write(f"const DISTRIBUTION_GROUPS = {json.dumps(group_list, indent=2, ensure_ascii=False)};\n")
        f.write("window.DISTRIBUTION_GROUPS = DISTRIBUTION_GROUPS;\n\n")
        f.write(f"const DISTRIBUTION_SUITE_META = {json.dumps(suite_meta, indent=2, ensure_ascii=False)};\n")
        f.write("window.DISTRIBUTION_SUITE_META = DISTRIBUTION_SUITE_META;\n")

    group_count      = len(group_list)
    submission_count = len(all_submissions)
    print(f"Distribution data written to {out_path} "
          f"({group_count} groups, {submission_count} submissions).")


def _bust_index_cache(data_path: Path, index_path: Path) -> None:
    """Rewrite <script src="leaderboard.js?v=<sha8>"> to match the short SHA-256."""
    if not index_path.exists():
        return
    sha8 = hashlib.sha256(data_path.read_bytes()).hexdigest()[:8]
    html = index_path.read_text(encoding='utf-8')
    pattern = re.compile(
        r'(<script\s+src="leaderboard\.js)(?:\?v=[0-9a-f]+)?(")',
        re.IGNORECASE,
    )
    new_html, n = pattern.subn(rf'\1?v={sha8}\2', html)
    if n and new_html != html:
        index_path.write_text(new_html, encoding='utf-8')
        print(f"  cache-busted leaderboard.js → ?v={sha8}")


def _bust_distribution_cache(data_path: Path, html_path: Path) -> None:
    """为 distribution.html 添加缓存破坏版本号"""
    if not html_path.exists() or not data_path.exists():
        return
    sha8 = hashlib.sha256(data_path.read_bytes()).hexdigest()[:8]
    html = html_path.read_text(encoding='utf-8')
    pattern = re.compile(
        r'(<script\s+src="distribution\.js)(?:\?v=[0-9a-f]+)?(")',
        re.IGNORECASE,
    )
    new_html, n = pattern.subn(rf'\1?v={sha8}\2', html)
    if n and new_html != html:
        html_path.write_text(new_html, encoding='utf-8')
        print(f"  cache-busted distribution.js → ?v={sha8}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    results = load_results()
    print(f"Loaded {len(results)} results.")

    rows = [extract_row(r) for r in results]

    # Deduplicate: for each run_id keep only the best result
    _seen: dict = {}
    _deduped: list = []

    for row in rows:
        rid = row.get("run_id")
        if not rid:
            _deduped.append(row)
            continue

        suite_id = row.get("suite", "")
        if suite_id == "suite_C":
            metric = row.get("quant_quality_eff") or 0
        elif suite_id == "suite_E":
            metric = row.get("scaling_efficiency_4x") or row.get("scaling_efficiency_2x") or 0
        elif suite_id == "suite_G":
            metric = row.get("offline_throughput") or 0
        elif suite_id == "suite_F":
            metric = row.get("offline_throughput") or 0
        else:
            metric = row.get("offline_throughput") or 0

        if rid not in _seen or metric > _seen[rid]["metric"]:
            _seen[rid] = {"row": row, "metric": metric}

    for entry in _seen.values():
        _deduped.append(entry["row"])

    rows = _deduped

    suite_specs = _collect_suite_specs()

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SITE_DIR / "leaderboard.js"
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("// Auto-generated by leaderboard/generate.py. Do not edit manually.\n")
        f.write(f"const LEADERBOARD_DATA = {json.dumps(rows, indent=2)};\n")
        f.write("window.LEADERBOARD_DATA = LEADERBOARD_DATA;\n")
        f.write(f"const SUITE_SPECS = {json.dumps(suite_specs, indent=2)};\n")
        f.write("window.SUITE_SPECS = SUITE_SPECS;\n")

    print(f"Leaderboard data written to {out_path} "
          f"({len(rows)} rows, {len(suite_specs)} suite specs).")

    _bust_index_cache(out_path, SITE_DIR / "index.html")

    generate_api(results, SITE_DIR)
    
    # 生成分布数据
    generate_distribution_data(results, SITE_DIR)
    
    # 为 distribution.html 做缓存破坏
    dist_js = SITE_DIR / "distribution.js"
    dist_html = SITE_DIR / "distribution.html"
    if dist_js.exists() and dist_html.exists():
        _bust_distribution_cache(dist_js, dist_html)
        print("  distribution.html cache-busted")
    else:
        print(f"  distribution.html or distribution.js not found, skip cache bust")


if __name__ == "__main__":
    main()