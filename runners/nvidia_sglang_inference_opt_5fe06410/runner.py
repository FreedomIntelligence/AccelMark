"""SGLang runner with an optimized offline inference fast path.

The reference runner is intentionally retained for model loading, streaming,
resource cleanup, precision handling, and CLI compatibility.  This subclass
changes the part that Suite F actually measures:

* apply the chat template to the complete request batch in one tokenizer call;
* batch-tokenize generated strings instead of calling ``encode`` 200 times,
  and use SGLang's ``completion_tokens`` fast path only after a warmup batch
  proves it is identical to the reference runner's decoded-text token count;
* select offline-only tokenizer/scheduler/speculative policies without
  imposing their latency trade-offs on online and interactive scenarios.

No benchmark inputs, output limits, accuracy logic, or timing boundaries are
changed.  The fallback paths deliberately preserve reference-runner semantics
for older SGLang/tokenizer versions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# The runner is launched as a file, so Python initially exposes only this
# runner directory.  Add the AccelMark workspace root before shared imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from loadgen.types import InferenceResult
from runners.benchmark_runner import InferenceRequest
from runners.nvidia_sglang_c43a8309.runner import SGLangRunner


class SGLangInferenceOptimizedRunner(SGLangRunner):
    """Reference SGLang runner with a scenario-aware offline fast path."""

    _BATCHED_SCENARIOS = frozenset({"accuracy", "offline", "speculative"})
    _DEFAULT_CONTINUOUS_DECODE_STEPS = 4

    def __init__(self) -> None:
        super().__init__()
        self._metadata_counts_verified: bool | None = None

    def load_model(self, model_path: str, parallelism: dict[str, Any]) -> None:
        """Apply throughput policies only to synchronous batch scenarios."""
        config = dict(getattr(self, "_runner_config", {}) or {})
        engine_kwargs = dict(config.get("engine_kwargs") or {})
        scenario = getattr(self, "_current_scenario", None)

        if scenario in self._BATCHED_SCENARIOS:
            # These throughput-oriented defaults are part of the content-hashed
            # runner so a submitted result is reproducible without a local,
            # gitignored runner_config YAML. Explicit config still overrides
            # every default for follow-up experiments.
            engine_kwargs.setdefault("attention_backend", "triton")
            if config.get("enable_tokenizer_batch_encode", True):
                engine_kwargs.setdefault("enable_tokenizer_batch_encode", True)

            continuous_steps = int(
                config.get(
                    "continuous_decode_steps",
                    self._DEFAULT_CONTINUOUS_DECODE_STEPS,
                )
            )
            if continuous_steps < 1:
                raise ValueError("continuous_decode_steps must be at least 1")
            engine_kwargs.setdefault(
                "num_continuous_decode_steps", continuous_steps
            )

            speculative_mode = config.get("speculative_mode")
            if speculative_mode not in (None, "none", "ngram"):
                raise ValueError(
                    "speculative_mode must be one of: none, ngram"
                )
            if speculative_mode == "ngram":
                window = int(config.get("ngram_window", 4))
                breadth = int(config.get("ngram_breadth", 1))
                if window < 2 or breadth < 1:
                    raise ValueError(
                        "ngram_window must be >= 2 and ngram_breadth must be >= 1"
                    )
                engine_kwargs.setdefault("speculative_algorithm", "NGRAM")
                engine_kwargs.setdefault(
                    "speculative_ngram_min_match_window_size", 1
                )
                engine_kwargs.setdefault(
                    "speculative_ngram_max_match_window_size", window
                )
                engine_kwargs.setdefault(
                    "speculative_ngram_min_bfs_breadth", 1
                )
                engine_kwargs.setdefault(
                    "speculative_ngram_max_bfs_breadth", breadth
                )
                # SGLang's C++ NGRAM cache requires branch_length to be
                # strictly greater than max_match_window_size.
                engine_kwargs.setdefault(
                    "speculative_ngram_branch_length", window + 1
                )
                engine_kwargs.setdefault("speculative_num_draft_tokens", window)

        config["engine_kwargs"] = engine_kwargs
        self._runner_config = config
        super().load_model(model_path, parallelism)

    def _format_prompts_batch(
        self, requests: list[InferenceRequest]
    ) -> list[str]:
        """Format all prompts at once when the tokenizer supports batching."""
        tokenizer = self.tokenizer
        if tokenizer and getattr(tokenizer, "chat_template", None):
            conversations = [
                [{"role": "user", "content": request.prompt}]
                for request in requests
            ]
            try:
                formatted = tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if (
                    isinstance(formatted, list)
                    and len(formatted) == len(requests)
                    and all(isinstance(prompt, str) for prompt in formatted)
                ):
                    return formatted
            except (TypeError, ValueError):
                # Older tokenizers may accept only one conversation at a time.
                pass
        return [self.format_prompt(request.prompt) for request in requests]

    @staticmethod
    def _metadata_token_counts(outputs: list[Any]) -> list[int] | None:
        counts: list[int] = []
        for output in outputs:
            if not isinstance(output, dict):
                return None
            meta_info = output.get("meta_info")
            count = (
                meta_info.get("completion_tokens")
                if isinstance(meta_info, dict)
                else None
            )
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                return None
            counts.append(count)
        return counts

    def _batch_encoded_token_counts(self, texts: list[str]) -> list[int]:
        """Match ``tokenizer.encode`` semantics with one vectorized call."""
        try:
            encoded = self.tokenizer(texts, add_special_tokens=True)
            input_ids = encoded["input_ids"]
            if len(input_ids) == len(texts):
                return [len(token_ids) if text else 0 for token_ids, text in zip(input_ids, texts)]
        except (KeyError, TypeError, ValueError):
            pass
        return [len(self.tokenizer.encode(text)) if text else 0 for text in texts]

    def _completion_token_counts(
        self, outputs: list[Any], texts: list[str]
    ) -> list[int]:
        """Preserve reference counting while avoiding redundant serial work."""
        metadata_counts = self._metadata_token_counts(outputs)
        if self._metadata_counts_verified is True and metadata_counts is not None:
            return metadata_counts

        encoded_counts = self._batch_encoded_token_counts(texts)
        if self._metadata_counts_verified is None:
            self._metadata_counts_verified = metadata_counts == encoded_counts
            mode = "metadata" if self._metadata_counts_verified else "batch-encode"
            print(f"  output token-count fast path = {mode}")
        return metadata_counts if self._metadata_counts_verified else encoded_counts

    def inference_fn_offline(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """Run one synchronous SGLang batch with low-overhead postprocessing."""
        if (getattr(self, "_runner_config", {}) or {}).get(
            "use_reference_offline", False
        ):
            return super().inference_fn_offline(requests)

        formatted = self._format_prompts_batch(requests)
        start = time.perf_counter()
        outputs = self.engine.generate(
            prompt=formatted,
            sampling_params=self._sampling_params,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        texts = [
            (
                output.get("text", "")
                if isinstance(output, dict)
                else str(output)
            )
            for output in outputs
        ]
        token_counts = self._completion_token_counts(outputs, texts)

        results: list[InferenceResult] = []
        for request, text, output_tokens in zip(requests, texts, token_counts):
            results.append(
                InferenceResult(
                    first_token_time_ms=None,
                    total_time_ms=elapsed_ms,
                    output_tokens=output_tokens,
                    input_tokens=request.input_tokens or 0,
                    success=True,
                    output_text=text,
                )
            )
        return results


if __name__ == "__main__":
    SGLangInferenceOptimizedRunner().main()
