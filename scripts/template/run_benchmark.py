"""
AccelMark Platform Script Template
===================================
Copy this file to scripts/{your_platform}/run_benchmark.py
Implement the sections marked with TODO.

This template covers the offline and interactive scenarios.
For online scenario, the same inference_fn is used — LoadGen handles the timing.
For training scenario, see the training section at the bottom.

Usage:
    python scripts/your_platform/run_benchmark.py \
        --suite suite_A \
        --scenario all \
        --output-dir ./results/community/chipx1_llama3-8b_suite-A_2026-03-22 \
        --tensor-parallel-size 1
"""

import argparse
import json
import time
from pathlib import Path

# LoadGen is always imported from the shared location.
# Do not copy loadgen.py into your platform directory.
from loadgen.loadgen import AccelMarkLoadGen
from loadgen.types import InferenceResult


# ===========================================================================
# TODO 1: Import your platform's inference library
# ===========================================================================
# Examples:
#   from vllm import LLM, SamplingParams          # NVIDIA/AMD vLLM
#   import mindspore                              # Ascend MindIE
#   import torch                                  # Generic PyTorch


# ===========================================================================
# TODO 2: Global model variable
# Initialize once, reuse across all inference_fn calls.
# ===========================================================================
model = None


def load_model(model_id: str, model_revision: str, tp_size: int) -> None:
    """
    Load the model into memory.
    Called once before benchmarking starts.
    """
    global model

    # TODO: Initialize your platform's model here.
    # Example for vLLM:
    #   from vllm import LLM
    #   model = LLM(
    #       model=model_id,
    #       revision=model_revision,
    #       tensor_parallel_size=tp_size,
    #       dtype="bfloat16",
    #   )
    raise NotImplementedError("Implement load_model for your platform")


def inference_fn(prompts: list[str]) -> list[InferenceResult]:
    """
    Run inference on a batch of prompts.

    This function is called by LoadGen. Do not add timing logic here —
    LoadGen handles all timing. Just run the inference and return results.

    IMPORTANT: To measure TTFT accurately, you must use streaming output.
    Record the time when the first token is received.

    Args:
        prompts: List of input strings

    Returns:
        List of InferenceResult, one per prompt, in the same order.
    """
    results = []

    for prompt in prompts:
        t_start = time.perf_counter()
        first_token_time_ms = None
        output_tokens = 0

        # TODO: Run inference with streaming enabled.
        # Example for vLLM async streaming:
        #
        #   import asyncio
        #   async def run_one(p):
        #       nonlocal first_token_time_ms, output_tokens
        #       async for output in model.agenerate(p, sampling_params):
        #           if first_token_time_ms is None:
        #               first_token_time_ms = (time.perf_counter() - t_start) * 1000
        #           output_tokens = len(output.outputs[0].token_ids)
        #   asyncio.run(run_one(prompt))

        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000

        results.append(InferenceResult(
            first_token_time_ms=first_token_time_ms,
            total_time_ms=total_time_ms,
            output_tokens=output_tokens,
            success=True,
        ))

    return results


def get_peak_memory_gb() -> float:
    """
    Return current peak memory usage in GB.
    Called after each run to record memory usage.
    """
    # TODO: Return peak memory for your platform.
    # Examples:
    #   import torch
    #   return torch.cuda.max_memory_allocated() / (1024**3)  # NVIDIA/AMD
    #
    #   import npu_bridge
    #   return npu_bridge.get_peak_memory() / (1024**3)       # Ascend
    raise NotImplementedError("Implement get_peak_memory_gb for your platform")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=["suite_A", "suite_B", "suite_C", "suite_D"])
    parser.add_argument("--scenario", required=True, choices=["offline", "online", "interactive", "training"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    # Load suite definition
    suite_path = Path(f"suites/{args.suite}/suite.json")
    with open(suite_path) as f:
        suite = json.load(f)

    # Load fixed request set
    requests = []
    if args.scenario != "training":
        requests_path = Path(f"suites/{args.suite}/requests.jsonl")
        with open(requests_path) as f:
            for line in f:
                requests.append(json.loads(line))

    # Load model (once)
    print(f"Loading model {suite['model_id']}...")
    load_model(suite["model_id"], suite["model_revision"], args.tensor_parallel_size)
    print("Model loaded.")

    # Run benchmark via LoadGen
    loadgen = AccelMarkLoadGen(
        suite=suite,
        requests=requests,
        scenario=args.scenario,
        output_dir=args.output_dir,
    )
    metrics = loadgen.run(inference_fn)

    # Inject peak memory (platform-specific)
    if args.scenario == "offline" and "offline" in metrics:
        peak_mem = get_peak_memory_gb()
        for row in (metrics["offline"].get("results_by_concurrency") or metrics["offline"].get("results_by_batch_size", [])):
            if not row["oom"]:
                row["peak_memory_gb"] = peak_mem

    # Build result.json
    import subprocess, platform as pl
    result = {
        "schema_version": "1.0",
        "suite_id": args.suite,
        "chip": {
            # TODO: Fill in your chip details
            "name": "YOUR CHIP NAME",
            "vendor": "OTHER",
            "count": args.tensor_parallel_size * args.pipeline_parallel_size,
            "memory_gb_per_chip": 0,
            "interconnect_intra_node": "unknown",
            "interconnect_inter_node": None,
        },
        "software": {
            "framework": "YOUR FRAMEWORK",
            "framework_version": "unknown",
            "driver_version": "unknown",
            "runtime_version": "unknown",
            "os": pl.platform(),
            "python_version": pl.python_version(),
        },
        "model": {
            "model_id": suite["model_id"],
            "model_revision": suite["model_revision"],
            "architecture": "dense",
            "parameter_count_b": 0,
            "precision": suite["precision_required"],
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
            "notes": "Run --scenario accuracy to check model accuracy.",
        },
        "meta": {
            "submitted_by": "YOUR_GITHUB_USERNAME",
            "submission_type": "individual",
            "date": time.strftime("%Y-%m-%d"),
            "reproduce_script": f"scripts/YOUR_PLATFORM/run_benchmark.py",
            "env_info_file": f"{args.output_dir}/env_info.json",
            "log_file": f"{args.output_dir}/run.log",
            "samples_file": f"{args.output_dir}/samples.jsonl",
            "notes": None,
        },
    }

    output_path = Path(args.output_dir) / "result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {output_path}")
    print("Next step: python scripts/validate_submission.py --dir", args.output_dir)


if __name__ == "__main__":
    main()
