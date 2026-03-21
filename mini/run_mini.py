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

from mini.mini_suite_selector import select_mini_suite, MiniSuiteConfig


def detect_hardware() -> dict:
    """Run collect_env.py and return parsed env_info."""
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


def get_total_memory(env: dict) -> float:
    """Sum memory across all detected accelerators."""
    accelerators = env.get("accelerators", [])
    if not accelerators:
        return 0.0
    return sum(a.get("memory_gb", 0) for a in accelerators)


def get_vendor(env: dict) -> str:
    """Detect primary vendor from environment."""
    accelerators = env.get("accelerators", [])
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


def load_model(config: MiniSuiteConfig):
    """
    Load model into memory. Auto-downloads if needed.
    Returns the loaded model object (framework-dependent).
    """
    vendor = config.model_id  # detect framework from config

    # For now, use vLLM as the default backend for NVIDIA/AMD
    # Apple Silicon would use MLX (future work)
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=config.model_id,
            revision=config.model_revision if config.model_revision != "PLACEHOLDER" else None,
            dtype="bfloat16" if config.quantization is None else "auto",
            tensor_parallel_size=1,
            trust_remote_code=False,
        )
        return llm, "vllm"
    except ImportError:
        raise RuntimeError(
            "vLLM not installed. Run: pip install vllm\n"
            "For Apple Silicon: pip install mlx-lm (coming soon)"
        )


def run_offline_inference(llm, config: MiniSuiteConfig, framework: str) -> dict:
    """Run the offline scenario and return raw metrics."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=config.output_tokens,
        temperature=0.0,
    )

    # Generate prompts (simple synthetic prompts for mini suite)
    # In future: load from mini/requests.jsonl
    prompts = [
        f"Write a short paragraph about topic number {i}."
        for i in range(config.num_requests)
    ]

    results_by_batch = []

    for batch_size in config.batch_sizes:
        batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

        throughputs = []
        peak_memories = []

        for run in range(config.num_runs + 1):  # +1 for warmup
            is_warmup = run == 0

            import torch
            if hasattr(torch, 'cuda'):
                torch.cuda.reset_peak_memory_stats()

            t_start = time.perf_counter()
            total_tokens = 0
            for batch in batches:
                outputs = llm.generate(batch, sampling_params)
                total_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
            t_end = time.perf_counter()

            if is_warmup:
                continue

            elapsed = t_end - t_start
            throughput = total_tokens / elapsed if elapsed > 0 else 0
            throughputs.append(throughput)

            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                peak_memories.append(peak_mem)

        import numpy as np
        results_by_batch.append({
            "batch_size": batch_size,
            "throughput_tokens_per_sec": round(float(np.median(throughputs)), 2),
            "peak_memory_gb": round(float(np.median(peak_memories)), 2) if peak_memories else None,
            "power_watts_avg": None,
            "power_watts_peak": None,
            "oom": False,
        })

    return results_by_batch


def build_result_json(
    env: dict,
    config: MiniSuiteConfig,
    results_by_batch: list,
    vendor: str,
) -> dict:
    """Build a result.json compatible with the AccelMark schema."""
    import platform as pl

    accelerators = env.get("accelerators", [])
    chip_name = accelerators[0]["name"] if accelerators else "Unknown"
    memory_gb = accelerators[0]["memory_gb"] if accelerators else 0

    return {
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
            "framework": "vLLM",
            "framework_version": _get_vllm_version(),
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
            "offline": {"results_by_batch_size": results_by_batch},
            "online": None,
            "interactive": None,
            "training": None,
            "derived": {},
        },
        "accuracy": {
            "subset_score": None,
            "baseline_delta": None,
            "valid": True,  # Mini suite skips accuracy check
            "notes": "Mini suite: accuracy check skipped. Use full Suite A for accuracy validation.",
        },
        "meta": {
            "submitted_by": "",  # Filled by Skill from OpenClaw user profile
            "submission_type": "individual",
            "date": time.strftime("%Y-%m-%d"),
            "reproduce_script": "mini/run_mini.py",
            "env_info_file": "env_info.json",
            "log_file": "run.log",
            "samples_file": None,
            "notes": f"Auto-generated by OpenClaw AccelMark Skill. Tier: {config.tier}",
        },
    }


def format_user_report(result: dict, config: MiniSuiteConfig, ranking: dict | None) -> str:
    """
    Format a human-readable report for the OpenClaw user.
    This is what gets sent back via Telegram/WhatsApp.
    """
    chip_name = result["chip"]["name"]
    memory_gb = result["chip"]["memory_gb_per_chip"]
    model_display = config.model_id.split("/")[-1]

    # Best throughput (highest batch size that didn't OOM)
    rows = result["metrics"]["offline"]["results_by_batch_size"]
    valid_rows = [r for r in rows if not r.get("oom") and r["throughput_tokens_per_sec"]]
    best = max(valid_rows, key=lambda r: r["throughput_tokens_per_sec"]) if valid_rows else None

    lines = [
        f"✅ Benchmark complete!\n",
        f"🖥️  {chip_name} ({memory_gb:.0f}GB)",
        f"🤖 Model: {model_display} [{config.display_name}]\n",
    ]

    if best:
        lines.append(f"⚡ Performance:")
        lines.append(f"   Speed: {best['throughput_tokens_per_sec']:,.0f} tokens/sec")
        if best.get("peak_memory_gb"):
            lines.append(f"   Memory used: {best['peak_memory_gb']:.1f}GB / {memory_gb:.0f}GB")
        lines.append("")

    # Ranking (if leaderboard data available)
    if ranking:
        pct = ranking.get("percentile", 0)
        rank = ranking.get("rank")
        total = ranking.get("total")
        lines.append(f"📊 Community ranking:")
        if rank and total:
            lines.append(f"   #{rank} of {total} {chip_name} submissions")
        lines.append(f"   Better than {pct:.0f}% of same chip\n")

    # Capabilities
    lines.append("✓ Your hardware is good for:")
    for cap in config.capabilities:
        lines.append(f"  • {cap}")

    if config.limitations:
        lines.append("\n⚠️  Limitations:")
        for lim in config.limitations:
            lines.append(f"  • {lim}")

    lines.append("\nSubmit to AccelMark leaderboard? Reply 'yes' to share your results.")

    return "\n".join(lines)


def _get_vllm_version() -> str:
    try:
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


def main():
    parser = argparse.ArgumentParser(description="AccelMark Mini Benchmark")
    parser.add_argument("--output-dir", default="./mini_result")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect hardware and show what would run, without actually running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Detecting hardware...")
    env = detect_hardware()

    # Save env_info
    with open(output_dir / "env_info.json", "w") as f:
        json.dump(env, f, indent=2)

    total_memory = get_total_memory(env)
    vendor = get_vendor(env)
    chip_name = env.get("accelerators", [{}])[0].get("name", "Unknown")

    print(f"   Detected: {chip_name} ({total_memory:.0f}GB)")

    config = select_mini_suite(total_memory, vendor, chip_name)
    print(f"   Selected tier: {config.display_name}")
    print(f"   Estimated time: ~{config.estimated_minutes} minutes")

    if args.dry_run:
        print("\n[Dry run] Would run:")
        print(f"  Model: {config.model_id}")
        print(f"  Batch sizes: {config.batch_sizes}")
        print(f"  Requests: {config.num_requests}")
        return

    print(f"\n⏳ Loading model (may download on first run)...")
    llm, framework = load_model(config)

    print(f"🚀 Running benchmark...")
    results_by_batch = run_offline_inference(llm, config, framework)

    result = build_result_json(env, config, results_by_batch, vendor)

    # Save result.json
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Format and print user report
    report = format_user_report(result, config, ranking=None)
    print("\n" + "="*50)
    print(report)
    print("="*50)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
