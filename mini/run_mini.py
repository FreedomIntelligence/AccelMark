"""
Lightweight benchmark runner for the Consumer Layer.
Designed to run in under 5 minutes on any hardware.
Used by the OpenClaw Skill.

Unlike the full suite scripts, this:
- Auto-selects the model based on hardware
- Downloads the model if needed
- Produces both a result.json (for leaderboard) and a human-readable summary
- Does NOT require the user to configure anything

Usage:
    python mini/run_mini.py --output-dir ./my_result/
    python mini/run_mini.py --output-dir ./my_result/ --dry-run
"""

import argparse
import json
import time
from pathlib import Path

from mini.mini_suite_selector import select_mode, select_mini_suite, MiniSuiteConfig


def collect_environment() -> dict:
    """Run scripts/collect_env.py and return parsed env_info."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        env_path = f.name

    subprocess.run(
        ["python", "scripts/collect_env.py", "--output", env_path],
        check=True, capture_output=True
    )

    with open(env_path) as f:
        return json.load(f)


def _load_requests(num_requests: int) -> list[str]:
    """Load prompts from requests.jsonl, falling back to synthetic prompts."""
    requests_path = Path(__file__).parent / "requests.jsonl"
    prompts = []
    if requests_path.exists():
        with open(requests_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    prompts.append(obj.get("prompt", ""))
    if not prompts:
        prompts = [
            f"Write a short paragraph about topic number {i}."
            for i in range(num_requests)
        ]
    # Cycle if needed
    while len(prompts) < num_requests:
        prompts.extend(prompts)
    return prompts[:num_requests]


def run_benchmark_vllm(config: MiniSuiteConfig, env: dict) -> dict:
    """Run offline inference using vLLM (GPU backend)."""
    from vllm import LLM, SamplingParams
    import numpy as np

    llm = LLM(
        model=config.model_id,
        revision=config.model_revision if config.model_revision != "PLACEHOLDER" else None,
        dtype="bfloat16" if config.quantization is None else "auto",
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(
        max_tokens=config.output_tokens,
        temperature=0.0,
    )

    prompts = _load_requests(config.num_requests)
    results_by_batch = []

    for batch_size in config.batch_sizes:
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        throughputs = []
        peak_memories = []

        for run in range(config.num_runs + 1):  # +1 for warmup
            is_warmup = run == 0

            try:
                import torch
                if hasattr(torch, 'cuda'):
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

            t_start = time.perf_counter()
            total_tokens = 0
            for batch in batches:
                outputs = llm.generate(batch, sampling_params)
                total_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
            t_end = time.perf_counter()

            if is_warmup:
                continue

            elapsed = t_end - t_start
            throughputs.append(total_tokens / elapsed if elapsed > 0 else 0)

            try:
                import torch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    peak_memories.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
            except ImportError:
                pass

        results_by_batch.append({
            "batch_size": batch_size,
            "throughput_tokens_per_sec": round(float(np.median(throughputs)), 2),
            "peak_memory_gb": round(float(np.median(peak_memories)), 2) if peak_memories else None,
            "power_watts_avg": None,
            "power_watts_peak": None,
            "oom": False,
        })

    return {"results_by_batch_size": results_by_batch}


def run_benchmark_mlx(config: MiniSuiteConfig, env: dict) -> dict:
    """Run inference using mlx-lm (Apple Silicon backend)."""
    from mlx_lm import load, generate
    import numpy as np

    model, tokenizer = load(config.model_id)
    prompts = _load_requests(config.num_requests)

    throughputs = []

    for run in range(config.num_runs + 1):  # +1 for warmup
        is_warmup = run == 0
        total_tokens = 0

        t_start = time.perf_counter()
        for prompt in prompts:
            response = generate(model, tokenizer, prompt=prompt,
                                max_tokens=config.output_tokens, verbose=False)
            total_tokens += len(response.split())
        t_end = time.perf_counter()

        if is_warmup:
            continue

        elapsed = t_end - t_start
        throughputs.append(total_tokens / elapsed if elapsed > 0 else 0)

    # mlx-lm does not support batching natively — report batch_size=1 only
    return {
        "results_by_batch_size": [{
            "batch_size": 1,
            "throughput_tokens_per_sec": round(float(np.median(throughputs)), 2),
            "peak_memory_gb": None,
            "power_watts_avg": None,
            "power_watts_peak": None,
            "oom": False,
        }]
    }


def run_benchmark(config: MiniSuiteConfig, env: dict) -> dict:
    """Dispatch to the appropriate backend based on config.framework."""
    if config.framework == "mlx-lm":
        return run_benchmark_mlx(config, env)
    else:
        return run_benchmark_vllm(config, env)


def build_result_json(env: dict, config: MiniSuiteConfig, benchmark_result: dict) -> dict:
    """Build a result.json compatible with the AccelMark schema."""
    import platform as pl

    accelerators = env.get("accelerators", [])
    chip_name = accelerators[0]["name"] if accelerators else "Unknown"
    memory_gb = accelerators[0].get("memory_gb", 0) if accelerators else 0
    vendor = _detect_vendor(accelerators)

    framework_display = "mlx-lm" if config.framework == "mlx-lm" else "vLLM"

    return {
        "mode": "benchmark",
        "schema_version": "1.0",
        "suite_id": f"mini-{config.tier}",
        "chip": {
            "name": chip_name,
            "vendor": vendor,
            "count": len(accelerators) if accelerators else 1,
            "memory_gb_per_chip": memory_gb,
            "interconnect_intra_node": "unknown",
            "interconnect_inter_node": None,
        },
        "software": {
            "framework": framework_display,
            "framework_version": _get_framework_version(config.framework),
            "driver_version": accelerators[0].get("driver_version", "unknown") if accelerators else "unknown",
            "runtime_version": env.get("runtime_version", "unknown"),
            "os": pl.platform(),
            "python_version": pl.python_version(),
        },
        "model": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "architecture": "dense",
            "parameter_count_b": _guess_param_count(config.model_id),
            "precision": "BF16" if config.quantization is None else config.quantization,
            "model_format": "GGUF" if config.quantization and "K" in config.quantization else "HuggingFace original",
        },
        "task": {
            "scenario": "offline",
            "num_runs": config.num_runs,
            "warmup_runs": 1,
            "parallelism": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "expert_parallel_size": None,
            },
            "extra_config": {"mini_suite": True, "auto_selected_tier": config.tier},
        },
        "metrics": {
            "offline": benchmark_result,
            "online": None,
            "interactive": None,
            "training": None,
            "derived": {},
        },
        "accuracy": {
            "subset_score": None,
            "baseline_delta": None,
            "valid": True,
            "notes": "Mini suite: accuracy check skipped. Use full Suite A for accuracy validation.",
        },
        "meta": {
            "submitted_by": "",
            "submission_type": "individual",
            "date": time.strftime("%Y-%m-%d"),
            "reproduce_script": "mini/run_mini.py",
            "env_info_file": "env_info.json",
            "log_file": "run.log",
            "samples_file": None,
            "notes": f"Auto-generated by OpenClaw AccelMark Skill. Tier: {config.tier}",
        },
    }


def format_benchmark_report(result: dict, config: MiniSuiteConfig, ranking: dict | None) -> str:
    """Format a human-readable benchmark report for chat display."""
    chip_name = result["chip"]["name"]
    memory_gb = result["chip"]["memory_gb_per_chip"]
    model_display = config.model_id.split("/")[-1]
    precision = "BF16" if config.quantization is None else config.quantization
    framework_display = result["software"]["framework"]

    rows = result["metrics"]["offline"]["results_by_batch_size"]
    valid_rows = [r for r in rows if not r.get("oom") and r["throughput_tokens_per_sec"]]
    best = max(valid_rows, key=lambda r: r["throughput_tokens_per_sec"]) if valid_rows else None

    lines = [
        "✅ Benchmark complete!\n",
        f"🖥️  {chip_name} ({memory_gb:.0f}GB)",
        f"🤖  {model_display} · {precision} · {framework_display}\n",
    ]

    if best:
        lines.append(f"⚡ {best['throughput_tokens_per_sec']:,.0f} tokens/sec")
        if best.get("peak_memory_gb"):
            pct = best["peak_memory_gb"] / memory_gb * 100 if memory_gb else 0
            lines.append(
                f"💾 Memory: {best['peak_memory_gb']:.1f}GB / {memory_gb:.0f}GB ({pct:.0f}%)"
            )

    if ranking:
        rank = ranking.get("rank")
        total = ranking.get("total")
        percentile = ranking.get("percentile", 0)
        lines.append("\n📊 Community ranking:")
        if rank and total:
            lines.append(f"   #{rank} of {total} {chip_name} submissions")
        lines.append(f"   Better than {percentile:.0f}% of same chip")

    lines.append("\n✓ Your hardware is good for:")
    for cap in config.capabilities:
        lines.append(f"  • {cap}")

    if config.limitations:
        lines.append("\n⚠️  Limitations:")
        for lim in config.limitations:
            lines.append(f"  • {lim}")

    lines.append("\nReply 'submit' to add to leaderboard")
    lines.append("Reply 'details' for full benchmark data")

    return "\n".join(lines)


def _detect_vendor(accelerators: list) -> str:
    if not accelerators:
        return "CPU"
    name = accelerators[0].get("name", "").lower()
    if "nvidia" in name or "geforce" in name or "tesla" in name or "quadro" in name:
        return "NVIDIA"
    if "amd" in name or "radeon" in name or "instinct" in name:
        return "AMD"
    if "apple" in name or "m1" in name or "m2" in name or "m3" in name or "m4" in name:
        return "Apple"
    if "ascend" in name or "npu" in name:
        return "Huawei"
    return "Other"


def _get_framework_version(framework: str) -> str:
    try:
        if framework == "mlx-lm":
            import mlx_lm
            return getattr(mlx_lm, "__version__", "unknown")
        else:
            import vllm
            return vllm.__version__
    except Exception:
        return "unknown"


def _guess_param_count(model_id: str) -> float:
    model_id_lower = model_id.lower()
    if "1b" in model_id_lower or "1.5b" in model_id_lower:
        return 1.0
    if "3b" in model_id_lower:
        return 3.0
    if "7b" in model_id_lower or "8b" in model_id_lower:
        return 8.0
    if "13b" in model_id_lower:
        return 13.0
    if "70b" in model_id_lower:
        return 70.0
    return 0.0


def _summarize_chips(env: dict) -> str:
    accelerators = env.get("accelerators", [])
    if not accelerators:
        cpu = env.get("cpu", {})
        return f"{cpu.get('model', 'Unknown CPU')} · {env.get('system_memory_gb', 0):.0f}GB RAM"
    chip = accelerators[0]
    mem = chip.get("memory_gb", 0)
    return f"{chip.get('name', 'Unknown')} ({mem:.0f}GB)"


def main():
    parser = argparse.ArgumentParser(description="AccelMark Mini Benchmark")
    parser.add_argument("--output-dir", default="./mini_result")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect hardware and show what would run, without actually running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Detecting hardware...")
    env = collect_environment()

    with open(output_dir / "env_info.json", "w") as f:
        json.dump(env, f, indent=2)

    chip_summary = _summarize_chips(env)
    print(f"   Detected: {chip_summary}")

    mode = select_mode(env)

    if mode == "assessment":
        from mini.hardware_assessment import assess_hardware, format_assessment_report
        result = assess_hardware(env)
        print("\n" + format_assessment_report(result))
        return

    config = select_mini_suite(env)
    print(f"   Selected tier: {config.display_name}")
    print(f"   Framework: {config.framework}")
    print(f"   Estimated time: ~{config.estimated_minutes} minutes")

    if args.dry_run:
        print("\n[Dry run] Would run:")
        print(f"  Model: {config.model_id}")
        print(f"  Batch sizes: {config.batch_sizes}")
        print(f"  Requests: {config.num_requests}")
        return

    print(f"\n⏳ Loading model (may download on first run)...")
    benchmark_result = run_benchmark(config, env)

    result = build_result_json(env, config, benchmark_result)

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    report = format_benchmark_report(result, config, ranking=None)
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
