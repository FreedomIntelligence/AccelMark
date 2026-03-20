"""
AccelMark — Huawei Ascend platform script using MindIE.
Supports: Suite A (offline, online, interactive).

Requirements: pip install -r scripts/ascend/requirements.txt
CANN toolkit with MindIE required.

Usage:
    python scripts/ascend/run_mindie.py \
        --suite suite_A \
        --scenario offline \
        --output-dir ./my_submission/ \
        --npu-ids 0
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from loadgen.loadgen import AccelMarkLoadGen
from loadgen.types import InferenceResult

try:
    import mindieservice_sdk as mindie
    HAS_MINDIE = True
except ImportError:
    HAS_MINDIE = False


model_client = None
output_tokens_max = 128


def load_model(model_id: str, model_revision: str, tp_size: int, suite: dict, npu_ids: list[int]) -> None:
    global model_client, output_tokens_max

    if not HAS_MINDIE:
        raise ImportError(
            "MindIE SDK (mindieservice_sdk) is not installed. "
            "Install CANN toolkit with MindIE support from https://www.hiascend.com/software/mindie"
        )

    output_tokens_max = suite["output_tokens"]

    # Initialize MindIE service
    config = mindie.MindIEConfig()
    config.model_path = model_id  # HuggingFace model ID or local path
    config.device_ids = npu_ids
    config.max_new_tokens = output_tokens_max

    model_client = mindie.LLMClient(config)
    model_client.start()


def inference_fn(prompts: list[str]) -> list[InferenceResult]:
    results = []

    for prompt in prompts:
        t_start = time.perf_counter()
        first_token_time_ms = None
        output_tokens = 0

        # MindIE streaming callback
        def on_token(token_str: str, is_last: bool):
            nonlocal first_token_time_ms, output_tokens
            if first_token_time_ms is None:
                first_token_time_ms = (time.perf_counter() - t_start) * 1000
            output_tokens += 1

        try:
            model_client.generate(
                prompt=prompt,
                max_new_tokens=output_tokens_max,
                stream_callback=on_token,
            )
            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            results.append(InferenceResult(
                first_token_time_ms=first_token_time_ms,
                total_time_ms=total_time_ms,
                output_tokens=output_tokens,
                success=True,
            ))
        except Exception as e:
            t_end = time.perf_counter()
            results.append(InferenceResult(
                first_token_time_ms=None,
                total_time_ms=(t_end - t_start) * 1000,
                output_tokens=0,
                success=False,
                error=str(e),
            ))

    return results


def get_peak_memory_gb() -> float:
    try:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "usages", "-i", "0"],
            text=True
        )
        # Parse HBM usage from npu-smi output (format varies by CANN version)
        for line in out.splitlines():
            if "HBM" in line and "MB" in line.upper():
                parts = line.split()
                for p in parts:
                    try:
                        return float(p) / 1024
                    except ValueError:
                        continue
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--npu-ids", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    suite_path = Path(f"suites/{args.suite}/suite.json")
    with open(suite_path) as f:
        suite = json.load(f)

    requests_path = Path(f"suites/{args.suite}/requests.jsonl")
    requests = []
    with open(requests_path) as f:
        for line in f:
            requests.append(json.loads(line))

    print(f"Loading {suite['model_id']} on Ascend NPU(s) {args.npu_ids}...")
    load_model(suite["model_id"], suite["model_revision"], len(args.npu_ids), suite, args.npu_ids)

    loadgen = AccelMarkLoadGen(
        suite=suite,
        requests=requests,
        scenario=args.scenario,
        output_dir=args.output_dir,
    )

    metrics = loadgen.run(inference_fn)

    if args.scenario == "offline":
        peak_mem = get_peak_memory_gb()
        for row in metrics["offline"]["results_by_batch_size"]:
            if not row["oom"]:
                row["peak_memory_gb"] = round(peak_mem, 2)

    import platform as pl

    # Get Ascend NPU name
    npu_name = "Huawei Ascend NPU"
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True)
        for line in out.splitlines():
            if "NPU Name" in line or "Chip Name" in line:
                npu_name = line.split(":")[-1].strip()
                break
    except Exception:
        pass

    result = {
        "schema_version": "1.0",
        "suite_id": args.suite,
        "chip": {
            "name": npu_name,
            "vendor": "Huawei",
            "count": len(args.npu_ids),
            "memory_gb_per_chip": 0,  # fill in manually or parse from npu-smi
            "interconnect_intra_node": "HCCS",
            "interconnect_inter_node": None,
        },
        "software": {
            "framework": "MindIE",
            "framework_version": getattr(mindie, "__version__", "unknown") if HAS_MINDIE else "unknown",
            "driver_version": "see env_info.json",
            "runtime_version": "see env_info.json",
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
                "tensor_parallel_size": len(args.npu_ids),
                "pipeline_parallel_size": 1,
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
            "submitted_by": "",
            "submission_type": "individual",
            "date": time.strftime("%Y-%m-%d"),
            "reproduce_script": "scripts/ascend/run_mindie.py",
            "env_info_file": f"{args.output_dir}/env_info.json",
            "log_file": f"{args.output_dir}/run.log",
            "samples_file": f"{args.output_dir}/samples.jsonl",
            "notes": None,
        },
    }

    out_path = Path(args.output_dir) / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    main()
