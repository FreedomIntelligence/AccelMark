"""
AccelMark — NVIDIA platform script using vLLM.
Supports: Suite A (offline, online, interactive), Suite B, Suite D.

Requirements: pip install -r scripts/nvidia/requirements.txt

Usage:
    # Run all scenarios (recommended) — output dir is auto-named:
    #   results/community/a100x1_llama3-8b_suite-A_YYYY-MM-DD
    python scripts/nvidia/run_vllm.py --suite suite_A --scenario all

    # Single scenario:
    python scripts/nvidia/run_vllm.py --suite suite_A --scenario offline

    # Multi-chip:
    python scripts/nvidia/run_vllm.py --suite suite_A --scenario all --tensor-parallel-size 2

    # Override output directory (e.g. for verified submissions):
    python scripts/nvidia/run_vllm.py \
        --suite suite_A \
        --scenario all \
        --output-dir ./results/verified/a100x1_llama3-8b_suite-A_2026-03-22
"""

import sys
import os
from pathlib import Path

# Auto-add repo root to Python path so loadgen can be imported
# regardless of where the script is run from
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import asyncio
import copy
import json
import logging
import subprocess
import tempfile
import time
from datetime import datetime, timezone

# Suppress vLLM verbose request-level logging unless ACCELMARK_VERBOSE is set.
# Hides per-request "Added request / Finished request" spam.
# vLLM metrics summaries (every 5s) are still shown.
if not os.environ.get("ACCELMARK_VERBOSE"):
    logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
    logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)

from transformers import AutoTokenizer

import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, LLM, SamplingParams
from vllm.utils import random_uuid

from loadgen.loadgen import AccelMarkLoadGen
from loadgen.types import InferenceResult


# Async engine — used for online and interactive scenarios (streaming, TTFT measurement)
engine: AsyncLLMEngine = None
# Sync engine — used for offline scenario (sends all requests at once, optimal scheduler use)
llm: LLM = None

sampling_params: SamplingParams = None
_loop: asyncio.AbstractEventLoop = None
tokenizer = None


def _generate_output_dir(args, suite: dict, env_info: dict) -> str:
    """
    Auto-generate a standardized submission directory name.

    Format: results/community/{chip}x{count}_{model}_{suite}[_{scenario}]_{date}

    Examples:
        --scenario all:         results/community/a100x1_llama3-8b_suite-A_2026-03-22
        --scenario offline:     results/community/a100x1_llama3-8b_suite-A_offline_2026-03-22
        --tensor-parallel 2:    results/community/a100x2_llama3-8b_suite-A_2026-03-22
    """
    import re
    from datetime import date

    # --- Chip short name ---
    chip_full = "unknown"
    try:
        accelerators = env_info.get("accelerators", [])
        if accelerators:
            chip_full = accelerators[0].get("name", "unknown")
    except Exception:
        pass

    chip_short = chip_full.lower()
    for prefix in ["nvidia ", "amd instinct ", "amd ", "apple ", "ascend "]:
        chip_short = chip_short.replace(prefix, "")
    # Take the first word/segment (split on spaces AND hyphens)
    # "a100-sxm4-80gb" → "a100", "m2 ultra" → "m2", "mi300x" → "mi300x"
    first_token = re.split(r"[\s\-]", chip_short)[0] if chip_short else chip_short
    chip_short = re.sub(r"[^a-z0-9]", "", first_token) or "gpu"

    # --- Chip count ---
    count = args.tensor_parallel_size * getattr(args, "pipeline_parallel_size", 1)

    # --- Model short name ---
    model_id = suite.get("model_id", "unknown")
    model_short = model_id.split("/")[-1].lower()
    model_short = re.sub(r"-(instruct|chat|hf|base|v\d[\d.]*)$", "", model_short)
    model_short = re.sub(r"^meta-llama-", "llama", model_short)
    model_short = re.sub(r"^llama-", "llama", model_short)
    model_short = model_short[:20].rstrip("-")

    # --- Suite short name ---
    suite_short = args.suite.replace("_", "-")  # "suite_A" -> "suite-A"

    # --- Scenario suffix (omit for --scenario all) ---
    scenario_suffix = f"_{args.scenario}" if args.scenario != "all" else ""

    # --- Date ---
    date_str = date.today().strftime("%Y-%m-%d")

    dir_name = f"{chip_short}x{count}_{model_short}_{suite_short}{scenario_suffix}_{date_str}"
    return str(Path("results") / "community" / dir_name)


def _release_gpu_memory() -> None:
    """
    Explicitly release GPU memory between scenarios when running --scenario all.
    Required because offline uses sync LLM and online uses AsyncLLMEngine —
    switching between them without releasing memory causes CUDA OOM.
    """
    global llm, engine, tokenizer, sampling_params

    print("\nReleasing GPU memory before next scenario...")

    # Destroy sync LLM
    if llm is not None:
        try:
            del llm
        except Exception:
            pass
        llm = None

    # Destroy async engine
    if engine is not None:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(engine.shutdown())
        except Exception:
            pass
        try:
            del engine
        except Exception:
            pass
        engine = None

    # Clear PyTorch CUDA cache
    import gc
    gc.collect()

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory released — free: {free/1024**3:.1f} GB / {total/1024**3:.1f} GB\n")
    except Exception as e:
        print(f"Warning: could not clear CUDA cache: {e}")


def setup_logging(output_dir: str) -> None:
    """
    Redirect all stdout and stderr to both console and run.log.
    Called at the start of run_single_scenario() after output_dir is created.
    """
    log_path = Path(output_dir) / "run.log"

    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers

        def write(self, text):
            for w in self.writers:
                w.write(text)
                w.flush()

        def flush(self):
            for w in self.writers:
                w.flush()

        def fileno(self):
            # Return fileno of first real file for subprocess compatibility
            for w in self.writers:
                try:
                    return w.fileno()
                except Exception:
                    continue
            raise OSError("no fileno available")

    log_file = open(log_path, "w", buffering=1)
    tee = TeeWriter(sys.__stdout__, log_file)
    sys.stdout = tee

    # Also capture stderr (vLLM logs go to stderr)
    tee_err = TeeWriter(sys.__stderr__, log_file)
    sys.stderr = tee_err

    print(f"Logging to {log_path}")


def resolve_model_path(model_id: str, cli_override: str | None = None) -> str:
    """
    Resolve the actual model path to use.

    Priority:
    1. --model-path CLI argument (highest priority)
    2. configs/models_local.yaml local_path
    3. configs/models.yaml local_path
    4. HuggingFace model_id (fallback)
    """
    if cli_override:
        return cli_override

    import yaml

    repo_root = Path(__file__).resolve().parent.parent.parent

    for config_file in ["configs/models_local.yaml", "configs/models.yaml"]:
        config_path = repo_root / config_file
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            models = config.get("models", {})
            entry = models.get(model_id, {})
            local_path = entry.get("local_path")
            if local_path and Path(local_path).exists():
                print(f"Using local model path: {local_path}")
                return local_path
        except Exception as e:
            print(f"Warning: could not read {config_file}: {e}")

    print(f"Using HuggingFace model: {model_id}")
    return model_id


def load_model(model_id: str, model_revision: str, max_model_len: int, tp_size: int, suite: dict,
               scenario: str, enforce_eager: bool = False) -> None:
    global engine, llm, sampling_params, _loop, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
    )
    sampling_params = SamplingParams(
        max_tokens=suite["output_tokens_max"],
        temperature=0.0,   # greedy for reproducibility
    )

    if scenario == "offline":
        # Sync LLM: sends all requests to vLLM at once.
        # vLLM's internal scheduler handles batching optimally.
        # max_num_seqs=512 ensures the scheduler is never client-side bottlenecked.
        llm = LLM(
            model=model_id,
            revision=model_revision,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            trust_remote_code=False,
            enforce_eager=enforce_eager,
            max_num_seqs=512,
            max_model_len=max_model_len
        )
    else:
        # Async engine with streaming: required for TTFT measurement (online/interactive).
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        engine_args = AsyncEngineArgs(
            model=model_id,
            revision=model_revision,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            trust_remote_code=False,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)


def _apply_chat_template(prompt: str) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def inference_fn_offline(prompts: list[str]) -> list[InferenceResult]:
    """
    Send ALL prompts to vLLM at once.
    vLLM's internal scheduler handles batching — do NOT chunk client-side.
    Throughput is measured as (input + output) tokens / elapsed to match vLLM's metric.
    """
    formatted = [_apply_chat_template(p) for p in prompts]

    t_start = time.perf_counter()
    outputs = llm.generate(formatted, sampling_params)
    t_end = time.perf_counter()
    total_time_ms = (t_end - t_start) * 1000

    results = []
    for output in outputs:
        results.append(InferenceResult(
            first_token_time_ms=None,  # not available from sync API
            total_time_ms=total_time_ms,
            output_tokens=len(output.outputs[0].token_ids),
            input_tokens=len(output.prompt_token_ids),
            success=True,
        ))
    return results


async def _run_one_streaming(prompt: str) -> InferenceResult:
    request_id = random_uuid()
    t_start = time.perf_counter()
    first_token_time_ms = None
    output_tokens = 0
    input_tokens = 0

    formatted = _apply_chat_template(prompt)

    async for output in engine.generate(formatted, sampling_params, request_id):
        if first_token_time_ms is None and len(output.outputs[0].token_ids) > 0:
            first_token_time_ms = (time.perf_counter() - t_start) * 1000
        output_tokens = len(output.outputs[0].token_ids)
        input_tokens = len(output.prompt_token_ids)

    total_time_ms = (time.perf_counter() - t_start) * 1000
    return InferenceResult(
        first_token_time_ms=first_token_time_ms,
        total_time_ms=total_time_ms,
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        success=True,
    )


def inference_fn_streaming(prompts: list[str]) -> list[InferenceResult]:
    async def run_all():
        tasks = [_run_one_streaming(p) for p in prompts]
        return await asyncio.gather(*tasks)
    return _loop.run_until_complete(run_all())


def get_peak_memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def build_result_json(args, suite: dict, metrics: dict, effective_model_path: str,
                      benchmark_start: datetime, benchmark_end: datetime,
                      model_load_seconds: float, driver_version: str) -> dict:
    import platform as pl
    from vllm import __version__ as vllm_version

    profile = load_submitter_profile()

    return {
        "schema_version": "1.0",
        "suite_id": args.suite,
        "chip": {
            "name": torch.cuda.get_device_name(0),
            "vendor": "NVIDIA",
            "count": args.tensor_parallel_size * args.pipeline_parallel_size,
            "memory_gb_per_chip": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
            "interconnect_intra_node": "NVLink",
            "interconnect_inter_node": None,
        },
        "software": {
            "framework": "vLLM",
            "framework_version": vllm_version,
            "driver_version": driver_version,
            "runtime_version": f"CUDA {torch.version.cuda}",
            "os": pl.platform(),
            "python_version": pl.python_version(),
        },
        "model": {
            "model_id": suite["model_id"],
            "model_revision": suite["model_revision"],
            "architecture": "dense",
            "parameter_count_b": 8.0,
            "precision": "BF16",
            "model_format": "HuggingFace original",
        },
        "task": {
            "scenario": args.scenario,
            "num_runs": suite["num_runs"],
            "warmup_runs": suite["warmup_runs"],
            "parallelism": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "data_parallel_size": 1,
                "expert_parallel_size": None,
            },
            "extra_config": None,
        },
        "metrics": {**metrics, "derived": {}},
        "accuracy": {
            "subset_score": None,
            "baseline_delta": None,
            "valid": False,
            "notes": "Run scripts/run_accuracy.py to populate this field.",
        },
        "meta": {
            "submitted_by": profile.get("submitted_by", ""),
            "submission_type": profile.get("submission_type", "individual"),
            "date": time.strftime("%Y-%m-%d"),
            "reproduce_script": "scripts/nvidia/run_vllm.py",
            "env_info_file": "env_info.json",
            "log_file": "run.log",
            "samples_file": "samples.jsonl",
            "notes": None,
            "benchmark_start_time": benchmark_start.isoformat(),
            "benchmark_end_time": benchmark_end.isoformat(),
            "benchmark_elapsed_minutes": round(
                (benchmark_end - benchmark_start).total_seconds() / 60, 1
            ),
            "model_load_seconds": model_load_seconds,
        },
    }


def load_submitter_profile() -> dict:
    """Load submitter info from configs/submitter.yaml if it exists."""
    profile_path = _REPO_ROOT / "configs" / "submitter.yaml"
    if not profile_path.exists():
        print(
            "Warning: configs/submitter.yaml not found. "
            "Copy configs/submitter.yaml.example to configs/submitter.yaml "
            "and fill in your details."
        )
        return {"submitted_by": "", "submission_type": "individual"}
    import yaml
    with open(profile_path) as f:
        return yaml.safe_load(f) or {}


def find_reusable_accuracy(model_id: str, revision: str, precision: str) -> dict | None:
    """
    Search for an existing valid accuracy result for this model + precision.

    Search order:
    1. results/accuracy/          — dedicated accuracy folder (new)
    2. results/verified/          — legacy: accuracy.json inside submission dirs
    3. results/community/         — legacy fallback
    """
    import re as _re

    # Search 1: dedicated accuracy folder
    accuracy_dir = _REPO_ROOT / "results" / "accuracy"
    if accuracy_dir.exists():
        model_short = model_id.split("/")[-1].lower()
        model_short = _re.sub(r"[^a-z0-9\-\.]", "-", model_short).strip("-")[:40]

        for acc_file in sorted(accuracy_dir.glob("*.json"), reverse=True):
            if model_short in acc_file.name.lower() and precision in acc_file.name:
                try:
                    with open(acc_file) as f:
                        acc = json.load(f)
                    if acc.get("valid"):
                        print(f"Reusing accuracy from: {acc_file.relative_to(_REPO_ROOT)}")
                        acc["notes"] = (
                            f"Reused from {acc_file.name} "
                            f"— same model and precision ({precision})"
                        )
                        return acc
                except Exception:
                    continue

    # Search 2 & 3: legacy — scan submission directories
    for tier in ["verified", "community"]:
        tier_dir = _REPO_ROOT / "results" / tier
        if not tier_dir.exists():
            continue
        for submission_dir in sorted(tier_dir.iterdir(), reverse=True):
            if not submission_dir.is_dir():
                continue
            acc_path = submission_dir / "accuracy.json"
            result_path = submission_dir / "result.json"
            if not acc_path.exists() or not result_path.exists():
                continue
            try:
                with open(result_path) as f:
                    r = json.load(f)
                m = r.get("model", {})
                if (
                    m.get("model_id") == model_id
                    and m.get("model_revision") == revision
                    and m.get("precision") == precision
                ):
                    with open(acc_path) as f:
                        acc = json.load(f)
                    if acc.get("valid"):
                        print(
                            f"Reusing accuracy from: "
                            f"results/{tier}/{submission_dir.name}/accuracy.json"
                        )
                        acc["notes"] = (
                            f"Reused from {submission_dir.name} "
                            f"— same model_id, revision, and precision"
                        )
                        return acc
            except Exception:
                continue

    return None


def _collect_env_info(output_dir: Path) -> str:
    """Collect env_info.json and return driver_version."""
    import subprocess
    env_info_path = output_dir / "env_info.json"
    if not env_info_path.exists():
        print("Collecting environment info...")
        subprocess.run(
            ["python", str(_REPO_ROOT / "scripts/collect_env.py"), "--output", str(env_info_path)],
            check=True
        )
        print(f"Environment info saved to {env_info_path}")

    driver_version = "unknown"
    if env_info_path.exists():
        with open(env_info_path) as f:
            env_info = json.load(f)
        accelerators = env_info.get("accelerators", [])
        if accelerators:
            driver_version = accelerators[0].get("driver_version", "unknown")
    return driver_version


def run_single_scenario(args, suite: dict) -> None:
    """Run one scenario. Called by main() or run_all_scenarios()."""
    global llm, engine, tokenizer, sampling_params

    # Reset engine state — previous scenario may have released it
    # This ensures load_model() always initializes a fresh engine
    llm = None
    engine = None
    tokenizer = None
    sampling_params = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    effective_model_path = resolve_model_path(suite["model_id"], args.model_path)

    # Load model only if not already loaded for this scenario type
    t_load_start = time.perf_counter()
    if args.scenario == "offline":
        if llm is None:
            print(f"Loading {suite['model_id']} (offline/sync)...")
            load_model(
                effective_model_path,
                suite["model_revision"] if not args.model_path else None,
                suite.get("max_model_len", None),
                args.tensor_parallel_size,
                suite,
                args.scenario,
                args.enforce_eager,
            )
    else:
        if engine is None:
            print(f"Loading {suite['model_id']} (online/async)...")
            load_model(
                effective_model_path,
                suite["model_revision"] if not args.model_path else None,
                suite.get("max_model_len", None),
                args.tensor_parallel_size,
                suite,
                args.scenario,
                args.enforce_eager,
            )
    model_load_seconds = round(time.perf_counter() - t_load_start, 1)

    # Load requests
    requests = []
    requests_path = _REPO_ROOT / f"suites/{args.suite}/requests.jsonl"
    with open(requests_path) as f:
        for line in f:
            requests.append(json.loads(line))

    driver_version = _collect_env_info(output_dir)

    loadgen = AccelMarkLoadGen(
        suite=suite,
        requests=requests,
        scenario=args.scenario,
        output_dir=args.output_dir,
    )

    if args.scenario == "offline":
        inference_fn = inference_fn_offline
    elif args.scenario == "online":
        inference_fn = _run_one_streaming
    else:
        # interactive
        inference_fn = _run_one_streaming

    torch.cuda.reset_peak_memory_stats()
    benchmark_start = datetime.now(timezone.utc)
    metrics = loadgen.run(inference_fn)
    benchmark_end = datetime.now(timezone.utc)

    # Inject peak memory into offline results
    if args.scenario == "offline":
        peak_mem = get_peak_memory_gb()
        for row in metrics["offline"]["results_by_batch_size"]:
            if not row["oom"]:
                row["peak_memory_gb"] = round(peak_mem, 2)

    result = build_result_json(
        args, suite, metrics, effective_model_path,
        benchmark_start, benchmark_end, model_load_seconds, driver_version,
    )

    out_path = output_dir / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {out_path}")

    # Try to reuse existing accuracy result
    model = suite.get("model_id", "")
    revision = suite.get("model_revision", "")
    precision = suite.get("precision_required", "BF16")

    reused_acc = find_reusable_accuracy(model, revision, precision)
    if reused_acc:
        # Write reused accuracy to this submission
        acc_path = output_dir / "accuracy.json"
        with open(acc_path, "w") as f:
            json.dump(reused_acc, f, indent=2)
        # Update result.json accuracy field
        result["accuracy"] = reused_acc
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Accuracy reused and written to {acc_path}")
    else:
        print(
            "\nNo reusable accuracy result found.\n"
            "Run accuracy check after benchmark completes:\n"
            f"  python scripts/run_accuracy.py \\\n"
            f"    --model-path {effective_model_path} \\\n"
            f"    --suite {args.suite} \\\n"
            f"    --output {output_dir}/accuracy.json"
        )


def run_all_scenarios(args, suite: dict) -> None:
    """
    Run all scenarios defined in suite.json sequentially.
    Each scenario gets its own output directory.
    Model is loaded once per engine type and reused across compatible scenarios.
    """
    from datetime import date

    scenarios = suite.get("scenarios", ["offline"])
    # Skip training for inference suites
    scenarios = [s for s in scenarios if s != "training"]

    base_dir = Path(args.output_dir)

    results_summary = []

    print(f"\n{'='*60}")
    print(f"  Running all scenarios: {scenarios}")
    print(f"  Base output dir: {base_dir}")
    print(f"{'='*60}\n")

    total_start = time.perf_counter()

    for scenario in scenarios:
        scenario_dir = base_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Starting scenario: {scenario}")
        print(f"  Output: {scenario_dir}")
        print(f"{'='*60}\n")

        scenario_args = copy.copy(args)
        scenario_args.scenario = scenario
        scenario_args.output_dir = str(scenario_dir)

        try:
            run_single_scenario(scenario_args, suite)
            results_summary.append((scenario, "SUCCESS", str(scenario_dir)))
            print(f"\n✓ {scenario} completed successfully")
        except Exception as e:
            results_summary.append((scenario, f"FAILED: {str(e)[:120]}", str(scenario_dir)))
            print(f"\n✗ {scenario} failed: {e}")
            print("Continuing with next scenario...")
        finally:
            # Always release GPU memory before starting the next scenario
            _release_gpu_memory()
            time.sleep(3)  # Give OS time to fully reclaim memory

    total_elapsed_minutes = round((time.perf_counter() - total_start) / 60, 1)

    print(f"\n{'='*60}")
    print(f"  All scenarios complete ({total_elapsed_minutes} min total)")
    print(f"{'='*60}")
    for scenario, status, path in results_summary:
        icon = "✓" if status == "SUCCESS" else "✗"
        print(f"  [{icon}] {scenario:12s} -- {status}")
        if status == "SUCCESS":
            print(f"             {path}")
    print()

    successful = [s for s, status, _ in results_summary if status == "SUCCESS"]
    failed = [s for s, status, _ in results_summary if not status.startswith("SUCCESS")]

    if successful:
        print(f"\nMerging {len(successful)} successful scenario(s) into suite-level result.json...")
        merged = merge_scenario_results(base_dir, suite, successful_scenarios=successful)

        if merged:
            result_path = base_dir / "result.json"
            try:
                with open(result_path) as f:
                    r = json.load(f)
                r["meta"]["benchmark_elapsed_minutes"] = total_elapsed_minutes
                r["meta"]["benchmark_elapsed_minutes_note"] = (
                    "Total wall-clock time for all scenarios including GPU memory release "
                    "between scenarios. Excludes model load time."
                )
                if failed:
                    r["meta"]["notes"] = (
                        f"Partial run: {successful} succeeded, {failed} failed."
                    )
                with open(result_path, "w") as f:
                    json.dump(r, f, indent=2)
                print(f"Total elapsed time ({total_elapsed_minutes} min) written to result.json")
            except Exception as e:
                print(f"Warning: could not update elapsed time in result.json: {e}")
    else:
        print("All scenarios failed — no result.json merged.")


def merge_scenario_results(
    base_dir: Path,
    suite: dict,
    successful_scenarios: list[str] | None = None,
) -> dict:
    """
    Merge individual scenario result.json files into one suite-level result.json.
    Only merges scenarios in successful_scenarios (defaults to all suite scenarios).
    """
    scenarios_to_merge = successful_scenarios or suite.get("scenarios", ["offline"])
    scenarios_to_merge = [s for s in scenarios_to_merge if s != "training"]

    # Use the first available scenario result as the base (has chip/software/model info)
    base_result = None
    for scenario in scenarios_to_merge:
        result_path = base_dir / scenario / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                base_result = json.load(f)
            break

    if not base_result:
        print("Warning: no scenario results found to merge")
        return {}

    # Build merged metrics
    merged_metrics = {"derived": {}}
    for scenario in scenarios_to_merge:
        result_path = base_dir / scenario / "result.json"
        if not result_path.exists():
            print(f"Warning: {scenario}/result.json not found, skipping")
            continue
        with open(result_path) as f:
            r = json.load(f)
        scenario_metrics = r.get("metrics", {})
        for key in ["offline", "online", "interactive", "training"]:
            if scenario_metrics.get(key):
                merged_metrics[key] = scenario_metrics[key]

    merged = {
        "schema_version": "1.0",
        "suite_id": base_result["suite_id"],
        "chip": base_result["chip"],
        "software": base_result["software"],
        "model": base_result["model"],
        "task": {
            "scenarios_run": scenarios_to_merge,
            "parallelism": base_result["task"]["parallelism"],
            "num_runs": suite.get("num_runs", 3),
        },
        "metrics": merged_metrics,
        "accuracy": base_result.get("accuracy", {}),
        "meta": {
            **base_result["meta"],
            "submission_type": base_result["meta"]["submission_type"],
            "scenario_dirs": {
                s: str((base_dir / s).relative_to(_REPO_ROOT))
                for s in scenarios_to_merge
                if (base_dir / s).exists()
            },
        },
    }

    merged_path = base_dir / "result.json"
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged suite result written to {merged_path}")
    return merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["offline", "online", "interactive", "training", "all"],
        help="Scenario to run. Use 'all' to run all scenarios defined in suite.json sequentially.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for results. "
            "If not specified, auto-generated as "
            "results/community/{chip}x{count}_{model}_{suite}_{date}."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model path. If not set, uses configs/models_local.yaml or HuggingFace.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable torch.compile (enforce eager execution). Use if compilation fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show verbose vLLM request-level logs (Added/Finished request). Default: hidden.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.verbose:
        logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
        logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)

    # Load suite
    suite_path = _REPO_ROOT / f"suites/{args.suite}/suite.json"
    with open(suite_path) as f:
        suite = json.load(f)

    # Collect env_info early — needed for chip name in auto-generated output dir
    _tmp_env = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    subprocess.run(
        ["python", str(_REPO_ROOT / "scripts/collect_env.py"), "--output", _tmp_env.name],
        check=False,
        capture_output=True,
    )
    try:
        with open(_tmp_env.name) as f:
            _env_info_preview = json.load(f)
    except Exception:
        _env_info_preview = {}

    # Auto-generate output dir if not specified
    if args.output_dir is None:
        args.output_dir = _generate_output_dir(args, suite, _env_info_preview)
        print(f"Output directory: {args.output_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.scenario == "all":
        run_all_scenarios(args, suite)
    else:
        run_single_scenario(args, suite)


if __name__ == "__main__":
    main()
