"""
AccelMark Empirical Arithmetic Intensity Profiler.

Provides platform-specific hardware profiling backends that measure
FLOPs and DRAM traffic via vendor performance counters, enabling
quantitative roofline classification of LLM inference workloads.

Quick start::

    from profiling import IntensityProfiler

    profiler = IntensityProfiler(
        suite="suite_A",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        batch=32,
        prompt_len=280,
        dtype_str="bfloat16",
    )
    result = profiler.run()
    print(result.classification)
    # {'prefill': 'compute-bound', 'decode': 'bandwidth-bound'}

CLI usage::

    python tools/profile_intensity.py --suite suite_A \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --out results/profiling/A800_suite_A_intensity.json
"""

from profiling.intensity import IntensityProfiler
from profiling.backends import (
    get_active_backend,
    get_backend_by_id,
    discover_backends,
)
from profiling.chip_specs import (
    lookup,
    list_known_chips,
    register_chip,
    i_star_table,
    print_i_star_table,
)

__all__ = [
    "IntensityProfiler",
    "get_active_backend",
    "get_backend_by_id",
    "discover_backends",
    "lookup",
    "list_known_chips",
    "register_chip",
    "i_star_table",
    "print_i_star_table",
]
