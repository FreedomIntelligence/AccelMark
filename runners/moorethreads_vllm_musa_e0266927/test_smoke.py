#!/usr/bin/env python3
"""
Standalone vllm-musa smoke test (does not use the AccelMark runner).

Usage (from repo root):

    python runners/moorethreads_vllm_musa_f2f6f965/test_smoke.py
    python runners/moorethreads_vllm_musa_f2f6f965/test_smoke.py /path/to/model

    MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \\
    python runners/moorethreads_vllm_musa_f2f6f965/test_smoke.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

import torch  # noqa: F401 — before transformers/vllm (libstdc++ load order)

from vllm import LLM, SamplingParams

_DEFAULT_MODEL = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")

PROMPTS = [
    "The capital of France is",
    "Say hello in one short sentence.",
]


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_MODEL

    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    print(f"Loading {model_path} ...")
    t_load = time.perf_counter()
    llm = LLM(
        model=model_path,
        device="musa",
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        trust_remote_code=False,
    )
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s\n")

    t_infer = time.perf_counter()
    outputs = llm.generate(PROMPTS, sampling_params)
    print(f"Inference done in {time.perf_counter() - t_infer:.1f}s\n")

    for prompt, output in zip(PROMPTS, outputs):
        text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)
        print(f"Prompt:  {prompt!r}")
        print(f"Output:  {text!r}")
        print(f"Tokens:  {n_tokens}\n")

    del llm
    gc.collect()
    try:
        if hasattr(torch, "musa"):
            torch.musa.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
