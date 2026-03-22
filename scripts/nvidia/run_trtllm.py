"""
AccelMark — NVIDIA platform script using TensorRT-LLM.
Supports: Suite A (offline, online, interactive).

Requirements: pip install -r scripts/nvidia/requirements.txt
Also requires TensorRT-LLM installation. See: https://github.com/NVIDIA/TensorRT-LLM

The model must be compiled to a TRT-LLM engine before running.
See TensorRT-LLM documentation for model compilation steps.

Usage:
    python scripts/nvidia/run_trtllm.py \
        --suite suite_A \
        --scenario all \
        --output-dir ./results/community/a100x1_llama3-8b_suite-A_2026-03-22 \
        --engine-dir /path/to/compiled/engine/ \
        --tensor-parallel-size 1
"""

import argparse
import json
import time
from pathlib import Path

import torch

from loadgen.loadgen import AccelMarkLoadGen
from loadgen.types import InferenceResult

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, SamplingConfig
    HAS_TRTLLM = True
except ImportError:
    HAS_TRTLLM = False


runner = None
sampling_config = None


def load_model(model_id: str, model_revision: str, tp_size: int, engine_dir: str, suite: dict) -> None:
    global runner, sampling_config

    if not HAS_TRTLLM:
        raise ImportError(
            "tensorrt_llm is not installed. "
            "See https://github.com/NVIDIA/TensorRT-LLM for installation."
        )

    runner = ModelRunner.from_dir(
        engine_dir=engine_dir,
        rank=0,
    )
    sampling_config = SamplingConfig(
        end_id=runner.tokenizer.eos_token_id,
        pad_id=runner.tokenizer.pad_token_id,
        max_new_tokens=suite["output_tokens"],
        temperature=1e-6,  # near-greedy
    )


def inference_fn(prompts: list[str]) -> list[InferenceResult]:
    results = []
    for prompt in prompts:
        t_start = time.perf_counter()
        first_token_time_ms = None

        input_ids = runner.tokenizer(prompt, return_tensors="pt").input_ids

        # TRT-LLM does not natively support streaming in the Python API
        # TTFT is approximated as None (not measurable without streaming)
        output = runner.generate(
            batch_input_ids=[input_ids[0]],
            sampling_config=sampling_config,
        )

        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000
        output_tokens = output.shape[-1] - input_ids.shape[-1]

        results.append(InferenceResult(
            first_token_time_ms=None,  # TRT-LLM non-streaming: TTFT not available
            total_time_ms=total_time_ms,
            output_tokens=int(output_tokens),
            success=True,
        ))

    return results


def get_peak_memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine-dir", required=True, help="Path to compiled TRT-LLM engine directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    suite_path = Path(f"suites/{args.suite}/suite.json")
    with open(suite_path) as f:
        suite = json.load(f)

    requests_path = Path(f"suites/{args.suite}/requests.jsonl")
    requests = []
    with open(requests_path) as f:
        for line in f:
            requests.append(json.loads(line))

    print(f"Loading TRT-LLM engine from {args.engine_dir}...")
    load_model(suite["model_id"], suite["model_revision"], args.tensor_parallel_size, args.engine_dir, suite)

    loadgen = AccelMarkLoadGen(
        suite=suite,
        requests=requests,
        scenario=args.scenario,
        output_dir=args.output_dir,
    )

    torch.cuda.reset_peak_memory_stats()
    metrics = loadgen.run(inference_fn)

    if args.scenario == "offline":
        peak_mem = get_peak_memory_gb()
        for row in metrics["offline"]["results_by_batch_size"]:
            if not row["oom"]:
                row["peak_memory_gb"] = round(peak_mem, 2)

    import platform as pl

    result = {
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
            "framework": "TensorRT-LLM",
            "framework_version": tensorrt_llm.__version__ if HAS_TRTLLM else "unknown",
            "driver_version": "see env_info.json",
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
            "model_format": "TensorRT-LLM engine",
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
            "extra_config": {"engine_dir": args.engine_dir},
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
            "reproduce_script": "scripts/nvidia/run_trtllm.py",
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
