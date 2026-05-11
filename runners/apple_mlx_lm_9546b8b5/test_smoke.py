#!/usr/bin/env python3
"""
Local smoke test for the Apple mlx-lm runner (requires Apple Silicon + mlx-lm).

Usage (from AccelMark repo root):

    python runners/apple_mlx_lm_9546b8b5/smoke_test.py
    python runners/apple_mlx_lm_9546b8b5/smoke_test.py Qwen2.5-0.5B-Instruct-bf16

Same model loading pattern as mlx_example.py at repo root.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_RUNNER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RUNNER_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_RUNNER_DIR))

from runner import AppleMLXLMRunner
from runners.benchmark_runner import InferenceRequest


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-0.5B-Instruct-bf16"

    r = AppleMLXLMRunner()
    r._effective_precision = "BF16"
    r._parallelism = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "data_parallel_size": 1,
    }
    r._chip_count = 1

    print(f"Loading {model_id} ...")
    r.load_model(
        model_id,
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "data_parallel_size": 1,
            "max_tokens": 64,
            "max_model_len": None,
            "use_async": False,
        },
    )

    prompt = "Say hello in one short sentence."
    print("Offline batch (1 request):")
    out = r.inference_fn_offline(
        [InferenceRequest(prompt=prompt, request_id=0)]
    )
    print(" ", out[0].output_text)
    print("  success=", out[0].success, "total_time_ms=", round(out[0].total_time_ms, 2))

    async def _stream():
        print("Streaming:")
        res = await r.inference_fn_streaming(
            InferenceRequest(prompt=prompt, request_id=1)
        )
        print(" ", res.output_text)
        print("  ttft_ms=", res.first_token_time_ms)

    asyncio.run(_stream())
    r.release_resources()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
