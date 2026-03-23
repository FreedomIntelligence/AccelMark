"""
AccelMark Leaderboard Generator
Reads all result.json files from results/ and generates leaderboard/site/leaderboard.js.

Usage:
    python leaderboard/generate.py
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# Load cloud pricing table once at module level
_pricing_cache: dict = {}
_pricing_path = Path("schema/cloud_pricing.json")
if _pricing_path.exists():
    with open(_pricing_path) as _f:
        _pricing_cache = json.load(_f)

RESULTS_DIR = Path("results")
SITE_DIR    = Path("leaderboard/site")
RUNNERS_DIR = Path("runners")


def _get_suite_precision_required(suite_id: str) -> str:
    """Read precision_required from suite.json. Returns 'BF16' if not found."""
    path = Path("suites") / suite_id / "suite.json"
    try:
        with open(path) as f:
            return json.load(f).get("precision_required", "BF16")
    except Exception:
        return "BF16"


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
                with open(result_path) as f:
                    data = json.load(f)
                data["_tier"]            = tier
                data["_submission_name"] = submission_dir.name
                data["_is_suite_level"]  = (
                    "scenarios_run"   in data.get("task", {}) or
                    "chip_counts_run" in data.get("task", {})
                )
                # Load env_info.json alongside result.json (optional, best-effort)
                env_path = submission_dir / "env_info.json"
                if env_path.exists():
                    try:
                        with open(env_path) as ef:
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
    chip        = result.get("chip", {})
    software    = result.get("software", {})
    model       = result.get("model", {})
    task        = result.get("task", {})
    accuracy    = result.get("accuracy", {})
    meta        = result.get("meta", {})
    parallelism = task.get("parallelism", {})
    env         = result.get("_env_info", {})

    # CPU string
    cpu_info = env.get("cpu", {})
    cpu_str  = None
    if cpu_info.get("model"):
        cores   = cpu_info.get("physical_cores")
        cpu_str = cpu_info["model"] + (f", {cores} cores" if cores else "")

    # NIC string
    nics    = env.get("network_interfaces", [])
    nic_str = None
    if nics:
        nic_types = list(dict.fromkeys(n.get("type", "") for n in nics if n.get("type")))
        nic_names = [n.get("name") for n in nics if n.get("name")]
        type_str  = nic_types[0] if nic_types else "unknown"
        names_str = ", ".join(nic_names) if nic_names else ""
        nic_str   = f"{len(nics)}x {type_str}" + (f" ({names_str})" if names_str else "")

    # Intra-node interconnect: prefer result.json, fall back to topology parse
    intra = chip.get("interconnect_intra_node")
    if not intra and env.get("accelerator_topology"):
        nv_matches = re.findall(r'NV(\d+)', env["accelerator_topology"])
        if nv_matches:
            intra = f"NVLink {max(int(x) for x in nv_matches)} (full mesh)"

    return {
        # Hardware
        "hw_chip":               chip.get("name"),
        "hw_vendor":             chip.get("vendor"),
        "hw_count":              chip.get("count"),
        "hw_memory_gb":          chip.get("memory_gb_per_chip"),
        "hw_interconnect_intra": intra,
        "hw_interconnect_inter": chip.get("interconnect_inter_node"),
        "hw_cpu":                cpu_str,
        "hw_system_memory_gb":   env.get("system_memory_gb"),
        "hw_pcie":               env.get("pcie_generation"),
        "hw_network":            nic_str,
        # Software
        "sw_framework":         software.get("framework"),
        "sw_framework_version": software.get("framework_version"),
        "sw_driver":            software.get("driver_version"),
        "sw_runtime":           software.get("runtime_version"),
        "sw_os":                software.get("os"),
        "sw_python":            software.get("python_version"),
        "sw_pytorch":           env.get("pytorch_version"),
        # Model
        "model_id":        model.get("model_id"),
        "model_revision":  model.get("model_revision"),
        "model_arch":      model.get("architecture"),
        "model_params_b":  model.get("parameter_count_b"),
        "model_precision": model.get("precision"),
        "model_format":    model.get("model_format"),
        # Run settings
        "run_scenarios":   task.get("scenarios_run"),
        "run_chip_counts": task.get("chip_counts_run"),
        "run_num_runs":    task.get("num_runs"),
        "run_tp":          parallelism.get("tensor_parallel_size"),
        "run_pp":          parallelism.get("pipeline_parallel_size"),
        "run_dp":          parallelism.get("data_parallel_size"),
        # Accuracy
        "acc_score":          accuracy.get("subset_score"),
        "acc_baseline_delta": accuracy.get("baseline_delta"),
        "acc_valid":          accuracy.get("valid"),
        "acc_notes":          accuracy.get("notes"),
        # Metadata
        "meta_submitted_by":     meta.get("submitted_by"),
        "meta_submission_type":  meta.get("submission_type"),
        "meta_date":             meta.get("date"),
        "meta_reproduce_script": meta.get("reproduce_script"),
        "meta_elapsed_min":      meta.get("benchmark_elapsed_minutes"),
        "meta_model_load_sec":   meta.get("model_load_seconds"),
        "meta_start_time":       meta.get("benchmark_start_time"),
        "meta_notes":            meta.get("notes"),
    }


# ── Implementation extraction (modal impl tab) ───────────────────────────────

def extract_impl(result: dict) -> dict | None:
    """
    Load runner meta.json for the implementation_id referenced in result.json.
    Returns None if implementation_id is absent or the runner folder is not found.
    Fields returned match meta.json schema plus a GitHub link.
    """
    impl_id = result.get("implementation_id")
    if not impl_id:
        return None

    meta_path = RUNNERS_DIR / impl_id / "meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
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
        "supersedes":   meta.get("supersedes"),
        "deprecated_by": meta.get("deprecated_by"),
        "github_url":   f"https://github.com/JuhaoLiang1997/AccelMark/tree/main/runners/{impl_id}",
        "runner_url":   f"https://github.com/JuhaoLiang1997/AccelMark/blob/main/runners/{impl_id}/runner.py",
    }


# ── Visualization data extraction (modal viz tab) ─────────────────────────────

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
            "samples":               samples,
        }

    if suite == "suite_A":
        rows = _offline_rows()
        return {
            "type": "suite_A",
            "offline": {
                "labels":     _concurrency_labels(rows),
                "throughput": [r.get("throughput_tokens_per_sec") for r in rows],
                "memory_gb":  [r.get("peak_memory_gb")            for r in rows],
            },
            "online":      _online_block(),
            "interactive": _interactive_block(),
            "sustained":   _sustained_block(),
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
            },
            "online":    _online_block(),
            "sustained": _sustained_block(),
        }

    if suite == "suite_D":
        rows = _offline_rows()
        return {
            "type": "suite_D",
            "offline": {
                "labels":     _concurrency_labels(rows),
                "throughput": [r.get("throughput_tokens_per_sec") for r in rows],
                "memory_gb":  [r.get("peak_memory_gb")            for r in rows],
            },
            "interactive": _interactive_block(),
            "sustained":   _sustained_block(),
        }

    if suite == "suite_C":
        quantization = metrics.get("quantization", {})
        entries      = quantization.get("results_by_precision", [])
        labels, throughputs, speedups, quality_effs = [], [], [], []
        for e in entries:
            labels.append(e.get("precision", ""))
            throughputs.append(e.get("best_throughput_tokens_per_sec"))
            speedups.append(e.get("speedup_vs_bf16"))
            quality_effs.append(e.get("quality_efficiency"))
        return {
            "type":              "suite_C",
            "labels":            labels,
            "throughput":        throughputs,
            "speedup":           speedups,
            "quality_efficiency": quality_effs,
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
    task     = result.get("task", {})
    metrics  = result.get("metrics", {})
    accuracy = result.get("accuracy", {})
    meta     = result.get("meta", {})
    derived  = metrics.get("derived", {})
    is_suite_level = result.get("_is_suite_level", False)
    suite_id       = result.get("suite_id", "")

    # ── Offline ───────────────────────────────────────────────────────────────
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

    # ── Online ────────────────────────────────────────────────────────────────
    online         = metrics.get("online")
    online_max_qps = online.get("max_valid_qps") if online else None

    # ── Interactive ───────────────────────────────────────────────────────────
    interactive          = metrics.get("interactive")
    interactive_ttft_p99 = interactive.get("ttft_ms_p99") if interactive else None

    # ── Sustained ─────────────────────────────────────────────────────────────
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

    # ── Primary metric ────────────────────────────────────────────────────────
    scenario = task.get("scenario", "offline")
    if is_suite_level and suite_id != "suite_E":
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
    else:
        primary_metric       = None
        primary_metric_label = None

    # ── Suite E scaling ───────────────────────────────────────────────────────
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

    # ── Suite C quantization ──────────────────────────────────────────────────
    quant_bf16_throughput  = None
    quant_int8_throughput  = None
    quant_int4_throughput  = None
    quant_int8_speedup     = None
    quant_int4_speedup     = None
    quant_int8_quality_eff = None
    quant_int4_quality_eff = None

    quantization = metrics.get("quantization")
    if quantization:
        for entry in quantization.get("results_by_precision", []):
            p   = entry.get("precision")
            thr = entry.get("best_throughput_tokens_per_sec")
            spd = entry.get("speedup_vs_bf16")
            qe  = entry.get("quality_efficiency")
            if p == "BF16":
                quant_bf16_throughput = thr
                primary_metric        = thr
                primary_metric_label  = "tokens/sec (BF16 baseline)"
            elif p == "INT8":
                quant_int8_throughput  = thr
                quant_int8_speedup     = spd
                quant_int8_quality_eff = qe
            elif p == "INT4":
                quant_int4_throughput  = thr
                quant_int4_speedup     = spd
                quant_int4_quality_eff = qe

    # ── Efficiency ────────────────────────────────────────────────────────────
    memory_gb_per_chip     = chip.get("memory_gb_per_chip", 0)
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

    # ── Precision fallback detection ──────────────────────────────────────────
    precision         = model.get("precision", "BF16")
    suite_required    = _get_suite_precision_required(suite_id)
    precision_fallback = precision.upper() != suite_required.upper()

    return {
        "submission":         result.get("_submission_name"),
        "tier":               result.get("_tier"),
        "is_suite_level":     is_suite_level,
        "chip":               chip_full_name,
        "vendor":             chip.get("vendor"),
        "chip_count":         chip.get("count", 1),
        "memory_gb_per_chip": memory_gb_per_chip,
        "framework":          software.get("framework"),
        "framework_version":  software.get("framework_version"),
        "model":              model.get("model_id", "").split("/")[-1],
        "precision":          precision,
        "precision_fallback": precision_fallback,
        "suite":              suite_id,
        "scenario":           "all" if is_suite_level else scenario,
        # Primary
        "primary_metric":          primary_metric,
        "primary_metric_label":    primary_metric_label,
        "tokens_per_sec_per_chip": tokens_per_sec_per_chip,
        # Scenario metrics
        "offline_throughput":   offline_throughput,
        "online_max_qps":       online_max_qps,
        "interactive_ttft_p99": interactive_ttft_p99,
        # Efficiency
        "peak_memory_gb":                     peak_memory_gb,
        "memory_utilization_pct":             memory_utilization_pct,
        "memory_efficiency_toks_per_gb":      memory_efficiency,
        "min_price_usd_per_hr":               min_price,
        "cost_efficiency_toks_per_dollar_hr": cost_efficiency,
        "tokens_per_watt":                    derived.get("tokens_per_sec_per_watt"),
        # Metadata
        "accuracy_valid":   accuracy.get("valid"),
        "accuracy_score":   accuracy.get("subset_score"),
        "date":             meta.get("date"),
        "submitted_by":     meta.get("submitted_by"),
        "reproduce_script": meta.get("reproduce_script"),
        "notes":            meta.get("notes"),
        # Suite E
        "scaling_efficiency_2x":   scaling_efficiency_2x,
        "scaling_efficiency_4x":   scaling_efficiency_4x,
        "scaling_base_throughput": scaling_base_throughput,
        # Suite C
        "quant_bf16_throughput":   quant_bf16_throughput,
        "quant_int8_throughput":   quant_int8_throughput,
        "quant_int4_throughput":   quant_int4_throughput,
        "quant_int8_speedup":      quant_int8_speedup,
        "quant_int4_speedup":      quant_int4_speedup,
        "quant_int8_quality_eff":  quant_int8_quality_eff,
        "quant_int4_quality_eff":  quant_int4_quality_eff,
        # Sustained
        "sustained_throughput":    sustained_throughput,
        "throttle_ratio":          throttle_ratio,
        "throttle_onset_minute":   throttle_onset_minute,
        "ttft_p99_drift_ms":       ttft_p99_drift_ms,
        "sustained_concurrency":   sustained_concurrency,
        # Panel data
        "detail": extract_detail(result),
        "viz":    extract_viz(result, metrics),
        "impl":   extract_impl(result),
        # Implementation ID (flat, for filtering/display without loading impl)
        "implementation_id": result.get("implementation_id"),
    }


# ── API generation ────────────────────────────────────────────────────────────

def generate_api(results: list[dict], output_dir: Path) -> None:
    """
    Generate static JSON for external tooling.
      api/rank.json   — per-submission ranking
      api/chips.json  — chip summary list
      api/index.json  — chip lookup index
    """
    api_dir = output_dir / "api"
    api_dir.mkdir(exist_ok=True)

    by_chip: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for r in results:
        chip_name       = r.get("chip", {}).get("name", "Unknown")
        submission_name = r.get("_submission_name", "unknown")
        best_thr        = None

        offline = r.get("metrics", {}).get("offline")
        if offline:
            rows  = offline.get("results_by_concurrency") or offline.get("results_by_batch_size", [])
            valid = [row for row in rows if not row.get("oom") and row.get("throughput_tokens_per_sec")]
            if valid:
                best_thr = max(row["throughput_tokens_per_sec"] for row in valid)

        # Suite E: fall back to scaling 1x entry
        if best_thr is None:
            scaling = r.get("metrics", {}).get("scaling", {})
            if scaling:
                best_thr = scaling.get("base_throughput_tokens_per_sec") or scaling.get("base_throughput_1x")
                if not best_thr:
                    for entry in scaling.get("results_by_chip_count", []):
                        if entry.get("chip_count") == 1:
                            best_thr = entry.get("best_throughput_tokens_per_sec")
                            break

        if not best_thr:
            continue
        by_chip[chip_name].append((submission_name, best_thr))

    # rank.json
    rank_data: dict[str, dict] = {}
    for chip_name, entries in by_chip.items():
        sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
        total = len(sorted_entries)
        for rank_idx, (submission_name, thr) in enumerate(sorted_entries):
            rank = rank_idx + 1
            rank_data[submission_name] = {
                "chip_name":  chip_name,
                "rank":       rank,
                "total":      total,
                "percentile": round((total - rank) / total * 100, 1) if total > 1 else 100.0,
                "best_throughput_tokens_per_sec": thr,
            }
    with open(api_dir / "rank.json", "w") as f:
        json.dump(rank_data, f, indent=2)

    # chips.json
    chips = []
    for chip_name, entries in by_chip.items():
        throughputs = [thr for _, thr in entries]
        chips.append({
            "name":                             chip_name,
            "submission_count":                 len(entries),
            "best_throughput_tokens_per_sec":   max(throughputs),
            "median_throughput_tokens_per_sec": round(statistics.median(throughputs), 1),
        })
    chips.sort(key=lambda x: x["best_throughput_tokens_per_sec"], reverse=True)
    with open(api_dir / "chips.json", "w") as f:
        json.dump(chips, f, indent=2)

    # index.json
    chip_index: dict[str, dict] = {}
    for chip_name, entries in by_chip.items():
        throughputs = [thr for _, thr in entries]
        chip_index[chip_name] = {
            "submission_count":                 len(entries),
            "best_throughput_tokens_per_sec":   max(throughputs),
            "median_throughput_tokens_per_sec": round(statistics.median(throughputs), 1),
            "p25_throughput_tokens_per_sec":    round(
                sorted(throughputs)[len(throughputs) // 4], 1
            ) if len(throughputs) >= 4 else None,
        }

    for r in results:
        chip_name   = r.get("chip", {}).get("name", "Unknown")
        if chip_name not in chip_index:
            continue
        online      = r.get("metrics", {}).get("online")
        interactive = r.get("metrics", {}).get("interactive")
        scaling     = r.get("metrics", {}).get("scaling")

        if online:
            qps = online.get("max_valid_qps")
            if qps is not None:
                cur = chip_index[chip_name].get("best_online_max_qps")
                if cur is None or qps > cur:
                    chip_index[chip_name]["best_online_max_qps"] = qps

        if interactive:
            ttft = interactive.get("ttft_ms_p99")
            if ttft is not None:
                cur = chip_index[chip_name].get("best_interactive_ttft_p99_ms")
                if cur is None or ttft < cur:
                    chip_index[chip_name]["best_interactive_ttft_p99_ms"] = round(ttft, 1)

        if scaling:
            for entry in scaling.get("results_by_chip_count", []):
                count = entry.get("chip_count")
                eff   = entry.get("scaling_efficiency")
                if count == 2 and eff:
                    chip_index[chip_name]["best_scaling_efficiency_2x"] = eff
                elif count == 4 and eff:
                    chip_index[chip_name]["best_scaling_efficiency_4x"] = eff

    for chip_name in chip_index:
        chip_index[chip_name].setdefault("best_online_max_qps", None)
        chip_index[chip_name].setdefault("best_interactive_ttft_p99_ms", None)

    with open(api_dir / "index.json", "w") as f:
        json.dump(chip_index, f, indent=2)

    print(f"API files written to {api_dir}/")
    print(f"  rank.json:  {len(rank_data)} submissions indexed")
    print(f"  chips.json: {len(chips)} chips listed")
    print(f"  index.json: {len(chip_index)} chips in lookup table")


# ── Entry point ───────────────────────────────────────────────────────────────

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