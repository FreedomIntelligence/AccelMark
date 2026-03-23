"""
Hardware Assessment for CPU-only systems.

Used when no accelerator is detected or VRAM < 4GB.
Does NOT run any inference. Returns recommendations based on CPU specs.
"""

CPU_PERFORMANCE_TABLE = {
    # Approximate tokens/sec for Llama-3.2-1B Q4 on CPU
    # Based on llama.cpp benchmarks
    "high":   {"1b_q4": 12, "3b_q4": 4,   "8b_q4": 1},    # 8+ cores, modern
    "medium": {"1b_q4": 6,  "3b_q4": 2,   "8b_q4": 0.5},  # 4-8 cores
    "low":    {"1b_q4": 2,  "3b_q4": 0.5, "8b_q4": 0.2},  # < 4 cores
}

MODEL_RECOMMENDATIONS = [
    {
        "model": "Llama-3.2-1B Q4",
        "min_ram_gb": 4,
        "min_toks_per_sec": 5,
        "use_case": "Simple Q&A, basic tasks",
        "tool": "ollama",
    },
    {
        "model": "Phi-3-mini Q4",
        "min_ram_gb": 4,
        "min_toks_per_sec": 3,
        "use_case": "Code generation, reasoning",
        "tool": "ollama",
    },
    {
        "model": "Llama-3.2-3B Q4",
        "min_ram_gb": 8,
        "min_toks_per_sec": 2,
        "use_case": "Better quality responses",
        "tool": "ollama",
    },
]


def assess_hardware(env: dict) -> dict:
    """
    Analyze CPU hardware and return recommendations.
    No inference is run.
    """
    cpu = env.get("cpu", {})
    memory_gb = env.get("system_memory_gb", 0)
    cores = cpu.get("physical_cores", 1)

    tier = "high" if cores >= 8 else "medium" if cores >= 4 else "low"
    perf = CPU_PERFORMANCE_TABLE[tier]

    recommendations = []
    for model in MODEL_RECOMMENDATIONS:
        if memory_gb >= model["min_ram_gb"]:
            key = (
                "1b_q4" if "1B" in model["model"] else
                "3b_q4" if "3B" in model["model"] else
                "8b_q4"
            )
            toks = perf.get(key)
            feasible = (
                "✓ Usable" if toks and toks >= model["min_toks_per_sec"]
                else "△ Slow" if toks and toks > 0.5
                else "✗ Too slow"
            )
            recommendations.append({
                "model": model["model"],
                "estimated_toks_per_sec": toks,
                "feasibility": feasible,
                "tool": model["tool"],
                "use_case": model["use_case"],
            })

    return {
        "mode": "assessment",
        "cpu": cpu.get("model", "Unknown CPU"),
        "memory_gb": memory_gb,
        "cores": cores,
        "recommendations": recommendations,
    }


def format_assessment_report(result: dict) -> str:
    """Format CPU assessment result for chat display."""
    cpu = result["cpu"]
    memory_gb = result["memory_gb"]
    recs = result["recommendations"]

    lines = [
        f"No GPU detected on this machine.\n",
        f"🖥️  {cpu} · {memory_gb:.0f}GB RAM\n",
        "📊 Hardware Assessment:",
        "Based on your CPU specs, estimated performance",
        "for local LLM inference:\n",
        f"{'Model':<20} {'Speed':<14} {'Feasibility'}",
    ]

    for rec in recs:
        toks = rec["estimated_toks_per_sec"]
        speed = f"~{toks} tok/s" if toks else "N/A"
        lines.append(f"{rec['model']:<20} {speed:<14} {rec['feasibility']}")

    # Find best usable tool
    usable = [r for r in recs if "Usable" in r["feasibility"]]
    if usable:
        tools = list(dict.fromkeys(r["tool"] for r in usable))
        models_str = " or ".join(r["model"] for r in usable)
        lines.append(
            f"\n💡 Recommendation:\n"
            f"For local inference on CPU, use {tools[0]} with\n"
            f"{models_str}.\n"
            f"For better performance, consider a machine\n"
            f"with a dedicated GPU."
        )
    else:
        lines.append(
            "\n💡 Recommendation:\n"
            "Your CPU is below the threshold for comfortable local LLM inference.\n"
            "Consider a machine with a dedicated GPU."
        )

    return "\n".join(lines)
