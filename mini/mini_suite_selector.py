"""
Auto-selects the appropriate mini suite configuration based on detected hardware.
Called by the OpenClaw Skill before running a benchmark.

The mini suite is NOT the same as Suite A/B/C/D.
It is a shorter, hardware-adaptive test designed to complete in under 5 minutes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MiniSuiteConfig:
    """The selected test configuration for this hardware."""

    # Identity
    tier: str                    # "nano" | "mini" | "standard" | "pro"
    display_name: str            # Human-readable, shown to user

    # Model
    model_id: str                # HuggingFace model ID
    model_revision: str          # Locked revision
    quantization: Optional[str]  # None | "Q4_K_M" | "INT8" | "INT4"

    # Test parameters
    input_tokens: int
    output_tokens: int
    batch_sizes: list[int]       # Smaller than full suites
    num_requests: int            # Much smaller than full suites (50-200)
    num_runs: int                # 2 (faster than full suite's 3)

    # Estimated runtime
    estimated_minutes: int

    # What this tier can and cannot do (shown to user)
    capabilities: list[str]      # e.g. ["local 8B models", "code generation"]
    limitations: list[str]       # e.g. ["70B models require quantization"]


# Locked model revisions — update at launch
MODELS = {
    "llama3-1b-q4":  ("bartowski/Llama-3.2-1B-Instruct-GGUF",    "PLACEHOLDER", "Q4_K_M"),
    "llama3-8b-q4":  ("bartowski/Meta-Llama-3-8B-Instruct-GGUF",  "PLACEHOLDER", "Q4_K_M"),
    "llama3-8b-bf16":("meta-llama/Meta-Llama-3-8B-Instruct",      "PLACEHOLDER", None),
    "llama3-70b-q4": ("bartowski/Meta-Llama-3-70B-Instruct-GGUF", "PLACEHOLDER", "Q4_K_M"),
    "llama3-70b-bf16":("meta-llama/Meta-Llama-3-70B-Instruct",    "PLACEHOLDER", None),
}


def select_mini_suite(
    memory_gb: float,
    vendor: str,
    chip_name: str,
) -> MiniSuiteConfig:
    """
    Select the appropriate mini suite based on available memory.

    Args:
        memory_gb:  Total accelerator memory in GB (sum if multi-chip)
        vendor:     Chip vendor string from env_info.json
        chip_name:  Full chip name string from env_info.json

    Returns:
        MiniSuiteConfig with all parameters needed to run the test
    """

    # Apple Silicon: use unified memory, different model format needed
    if vendor == "Apple" or "Apple" in chip_name:
        return _select_apple(memory_gb)

    # CPU-only fallback (no GPU detected)
    if vendor == "CPU" or memory_gb == 0:
        return _tier_nano()

    # GPU tiers based on VRAM
    if memory_gb < 6:
        return _tier_nano()
    elif memory_gb < 12:
        return _tier_mini_small()
    elif memory_gb < 20:
        return _tier_mini()
    elif memory_gb < 48:
        return _tier_standard()
    elif memory_gb < 90:
        return _tier_pro()
    else:
        return _tier_pro_large()


def _tier_nano() -> MiniSuiteConfig:
    model_id, revision, quant = MODELS["llama3-1b-q4"]
    return MiniSuiteConfig(
        tier="nano",
        display_name="Nano (1B quantized)",
        model_id=model_id,
        model_revision=revision,
        quantization=quant,
        input_tokens=256,
        output_tokens=64,
        batch_sizes=[1, 4],
        num_requests=50,
        num_runs=2,
        estimated_minutes=2,
        capabilities=["simple Q&A", "short text generation"],
        limitations=["not suitable for complex tasks", "8B+ models won't fit"],
    )


def _tier_mini_small() -> MiniSuiteConfig:
    model_id, revision, quant = MODELS["llama3-8b-q4"]
    return MiniSuiteConfig(
        tier="mini-small",
        display_name="Mini (8B Q4 quantized)",
        model_id=model_id,
        model_revision=revision,
        quantization=quant,
        input_tokens=512,
        output_tokens=128,
        batch_sizes=[1, 8],
        num_requests=100,
        num_runs=2,
        estimated_minutes=3,
        capabilities=["local 8B models (quantized)", "most everyday tasks"],
        limitations=["reduced quality vs full precision", "70B models won't fit"],
    )


def _tier_mini() -> MiniSuiteConfig:
    model_id, revision, quant = MODELS["llama3-8b-bf16"]
    return MiniSuiteConfig(
        tier="mini",
        display_name="Mini (8B full precision)",
        model_id=model_id,
        model_revision=revision,
        quantization=None,
        input_tokens=512,
        output_tokens=128,
        batch_sizes=[1, 8, 32],
        num_requests=100,
        num_runs=2,
        estimated_minutes=3,
        capabilities=["local 8B models (full quality)", "code generation", "reasoning"],
        limitations=["70B models require quantization"],
    )


def _tier_standard() -> MiniSuiteConfig:
    model_id, revision, quant = MODELS["llama3-70b-q4"]
    return MiniSuiteConfig(
        tier="standard",
        display_name="Standard (70B Q4 quantized)",
        model_id=model_id,
        model_revision=revision,
        quantization=quant,
        input_tokens=512,
        output_tokens=128,
        batch_sizes=[1, 8, 32],
        num_requests=100,
        num_runs=2,
        estimated_minutes=4,
        capabilities=["local 70B models (quantized)", "complex reasoning", "long context"],
        limitations=["full precision 70B needs 140GB+"],
    )


def _tier_pro() -> MiniSuiteConfig:
    model_id, revision, quant = MODELS["llama3-70b-bf16"]
    return MiniSuiteConfig(
        tier="pro",
        display_name="Pro (70B full precision)",
        model_id=model_id,
        model_revision=revision,
        quantization=None,
        input_tokens=512,
        output_tokens=128,
        batch_sizes=[1, 8, 32, 128],
        num_requests=200,
        num_runs=2,
        estimated_minutes=5,
        capabilities=["local 70B full precision", "production-grade inference", "multi-user serving"],
        limitations=["405B models need multiple chips"],
    )


def _tier_pro_large() -> MiniSuiteConfig:
    # Same as pro but with larger batch sizes for high-memory chips
    config = _tier_pro()
    config.tier = "pro-large"
    config.display_name = "Pro Large (70B full precision, large batches)"
    config.batch_sizes = [1, 8, 32, 128, 256]
    config.num_requests = 200
    return config


def _select_apple(memory_gb: float) -> MiniSuiteConfig:
    # Apple Silicon uses MLX framework, different model format
    # For now, map to similar tiers but note the difference
    if memory_gb < 16:
        config = _tier_mini_small()
    elif memory_gb < 32:
        config = _tier_mini()
    else:
        config = _tier_standard()
    config.display_name = f"[Apple MLX] {config.display_name}"
    return config
