"""
AccelMark — NVIDIA platform script using vLLM.
Supports: Suite A (offline, online, interactive), Suite B, Suite D.

Requirements: pip install -r scripts/nvidia/requirements.txt

Usage:
    python scripts/nvidia/run_vllm.py \
        --suite suite_A \
        --scenario offline \
        --output-dir ./my_submission/ \
        --tensor-parallel-size 1
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
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


def load_model(model_id: str, model_revision: str, tp_size: int, suite: dict,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model directory. If provided, overrides the model_id "
             "from suite.json for loading purposes. The result.json will still "
             "record the standard suite model_id for leaderboard comparability."
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable torch.compile (enforce eager execution). Use if compilation fails.",
    )
    args = parser.parse_args()

    suite_path = Path(f"suites/{args.suite}/suite.json")
    with open(suite_path) as f:
        suite = json.load(f)

    requests_path = Path(f"suites/{args.suite}/requests.jsonl")
    requests = []
    with open(requests_path) as f:
        for line in f:
            requests.append(json.loads(line))

    print(f"Loading {suite['model_id']}...")
    effective_model_path = args.model_path if args.model_path else suite["model_id"]
    load_model(effective_model_path, suite["model_revision"] if not args.model_path else None,
               args.tensor_parallel_size, suite, args.scenario, args.enforce_eager)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-collect environment info
    import subprocess
    env_info_path = Path(args.output_dir) / "env_info.json"
    if not env_info_path.exists():
        print("Collecting environment info...")
        subprocess.run(
            ["python", "scripts/collect_env.py", "--output", str(env_info_path)],
            check=True
        )
        print(f"Environment info saved to {env_info_path}")

    loadgen = AccelMarkLoadGen(
        suite=suite,
        requests=requests,
        scenario=args.scenario,
        output_dir=args.output_dir,
    )

    inference_fn = inference_fn_offline if args.scenario == "offline" else inference_fn_streaming

    torch.cuda.reset_peak_memory_stats()
    metrics = loadgen.run(inference_fn)

    # Inject peak memory into offline results
    if args.scenario == "offline":
        peak_mem = get_peak_memory_gb()
        for row in metrics["offline"]["results_by_batch_size"]:
            if not row["oom"]:
                row["peak_memory_gb"] = round(peak_mem, 2)

    # Write result.json
    import platform as pl
    from vllm import __version__ as vllm_version

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
            "framework": "vLLM",
            "framework_version": vllm_version,
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
            "submitted_by": "",
            "submission_type": "individual",
            "date": time.strftime("%Y-%m-%d"),
            "reproduce_script": "scripts/nvidia/run_vllm.py",
            "env_info_file": "env_info.json",
            "log_file": "run.log",
            "samples_file": "samples.jsonl",
            "notes": None,
        },
    }

    out_path = Path(args.output_dir) / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    main()
