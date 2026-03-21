"""
AccelMark LoadGen — Shared request sending and measurement component.

All platforms use this. Do not modify per-platform.
Platform scripts implement inference_fn and pass it to AccelMarkLoadGen.run().

Scenarios:
  offline     — all requests sent as a single batch, measures throughput
  online      — Poisson arrival at target QPS, measures latency under load
  interactive — one request at a time, measures single-request latency
  training    — step-based measurement, measures training throughput
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .types import InferenceResult, SampleRecord

SAMPLE_SEED = 42
MAX_SAMPLES_PER_CONFIG = 200


class AccelMarkLoadGen:

    def __init__(
        self,
        suite: dict,
        requests: list[dict],
        scenario: str,
        output_dir: str,
    ):
        """
        Args:
            suite:      Parsed contents of suite.json
            requests:   Parsed contents of requests.jsonl (list of {"prompt": str} dicts)
            scenario:   One of: offline, online, interactive, training
            output_dir: Directory where samples.jsonl will be written
        """
        self.suite = suite
        self.requests = requests
        self.scenario = scenario
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(SAMPLE_SEED)

    def run(self, inference_fn: Callable) -> dict:
        """
        Run the benchmark for the configured scenario.

        inference_fn signature:
            def inference_fn(prompts: list[str]) -> list[InferenceResult]

        Returns:
            Aggregated metrics dict suitable for embedding in result.json["metrics"]
        """
        if self.scenario == "offline":
            return self._run_offline(inference_fn)
        elif self.scenario == "online":
            return self._run_online(inference_fn)
        elif self.scenario == "interactive":
            return self._run_interactive(inference_fn)
        elif self.scenario == "training":
            return self._run_training(inference_fn)
        elif self.scenario == "multiturn":
            return self._run_multiturn(inference_fn)
        elif self.scenario == "sustained":
            return self._run_sustained(inference_fn)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    # ------------------------------------------------------------------
    # Offline scenario
    # ------------------------------------------------------------------

    def _run_offline(self, inference_fn: Callable) -> dict:
        """
        Send all requests as a single batch for each configured batch_size.
        Measures throughput (tokens/sec) and peak memory.
        """
        results_by_batch_size = []
        all_samples: list[SampleRecord] = []
        prompts = [r["prompt"] for r in self.requests]

        for batch_size in self.suite["batch_sizes"]:
            print(f"[offline] batch_size={batch_size}")
            run_throughputs = []
            run_peak_memories = []
            run_samples: list[SampleRecord] = []

            for run_idx in range(self.suite["num_runs"] + self.suite["warmup_runs"]):
                is_warmup = run_idx < self.suite["warmup_runs"]

                batches = [
                    prompts[i:i + batch_size]
                    for i in range(0, len(prompts), batch_size)
                ]

                t_start = time.perf_counter()
                all_results: list[InferenceResult] = []
                for batch in batches:
                    batch_results = inference_fn(batch)
                    all_results.extend(batch_results)
                t_end = time.perf_counter()

                if is_warmup:
                    continue

                total_output_tokens = sum(
                    r.output_tokens for r in all_results if r.success
                )
                elapsed = t_end - t_start
                throughput = total_output_tokens / elapsed if elapsed > 0 else 0
                run_throughputs.append(throughput)

                # Collect samples (subset only)
                sampled = self._sample_results(all_results, batch_size, "offline")
                run_samples.extend(sampled)

            all_samples.extend(run_samples)

            median_throughput = float(np.median(run_throughputs))
            results_by_batch_size.append({
                "batch_size": batch_size,
                "throughput_tokens_per_sec": round(median_throughput, 2),
                "peak_memory_gb": None,  # platform script should inject this
                "power_watts_avg": None,
                "power_watts_peak": None,
                "oom": False,
            })

        self._write_samples(all_samples)
        return {"offline": {"results_by_batch_size": results_by_batch_size}}

    # ------------------------------------------------------------------
    # Online scenario
    # ------------------------------------------------------------------

    def _run_online(self, inference_fn: Callable) -> dict:
        """
        Poisson arrival at each target QPS level.
        Identifies max QPS where p99 TTFT < suite SLA.
        """
        sla_ms = self.suite["online_sla_ttft_ms"]
        results_by_qps = []
        all_samples: list[SampleRecord] = []
        prompts = [r["prompt"] for r in self.requests]
        max_valid_qps = 0.0

        for target_qps in self.suite["online_qps_levels"]:
            print(f"[online] target_qps={target_qps}")
            run_ttfts: list[list[float]] = []
            run_tpots: list[list[float]] = []

            for run_idx in range(self.suite["num_runs"] + self.suite["warmup_runs"]):
                is_warmup = run_idx < self.suite["warmup_runs"]

                # Generate Poisson inter-arrival times
                n = len(prompts)
                inter_arrivals = [
                    self._rng.expovariate(target_qps) for _ in range(n)
                ]

                ttfts: list[float] = []
                tpots: list[float] = []
                t_next = time.perf_counter()

                for i, prompt in enumerate(prompts):
                    now = time.perf_counter()
                    sleep_time = t_next - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    t_next += inter_arrivals[i]

                    results = inference_fn([prompt])
                    r = results[0]
                    if r.success:
                        if r.first_token_time_ms is not None:
                            ttfts.append(r.first_token_time_ms)
                        tpot = (r.total_time_ms - (r.first_token_time_ms or 0)) / max(r.output_tokens - 1, 1)
                        tpots.append(tpot)

                if not is_warmup:
                    run_ttfts.append(ttfts)
                    run_tpots.append(tpots)

            all_ttfts = [v for run in run_ttfts for v in run]
            all_tpots = [v for run in run_tpots for v in run]
            achieved_qps = len(all_ttfts) / (self.suite["num_runs"] * len(prompts) / target_qps) if target_qps > 0 else 0

            ttft_p50 = float(np.percentile(all_ttfts, 50)) if all_ttfts else 0
            ttft_p90 = float(np.percentile(all_ttfts, 90)) if all_ttfts else 0
            ttft_p99 = float(np.percentile(all_ttfts, 99)) if all_ttfts else 0
            tpot_p50 = float(np.percentile(all_tpots, 50)) if all_tpots else 0
            tpot_p90 = float(np.percentile(all_tpots, 90)) if all_tpots else 0
            tpot_p99 = float(np.percentile(all_tpots, 99)) if all_tpots else 0

            sla_met = ttft_p99 < sla_ms
            if sla_met:
                max_valid_qps = target_qps

            results_by_qps.append({
                "target_qps": target_qps,
                "achieved_qps": round(achieved_qps, 2),
                "ttft_ms_p50": round(ttft_p50, 2),
                "ttft_ms_p90": round(ttft_p90, 2),
                "ttft_ms_p99": round(ttft_p99, 2),
                "tpot_ms_p50": round(tpot_p50, 2),
                "tpot_ms_p90": round(tpot_p90, 2),
                "tpot_ms_p99": round(tpot_p99, 2),
                "sla_met": sla_met,
            })

        self._write_samples(all_samples)
        return {"online": {"max_valid_qps": max_valid_qps, "results_by_qps": results_by_qps}}

    # ------------------------------------------------------------------
    # Interactive scenario
    # ------------------------------------------------------------------

    def _run_interactive(self, inference_fn: Callable) -> dict:
        """
        One request at a time. Measures single-request latency distribution.
        """
        prompts = [r["prompt"] for r in self.requests]
        all_ttfts: list[float] = []
        all_tpots: list[float] = []
        all_samples: list[SampleRecord] = []

        for run_idx in range(self.suite["num_runs"] + self.suite["warmup_runs"]):
            is_warmup = run_idx < self.suite["warmup_runs"]
            run_ttfts: list[float] = []
            run_tpots: list[float] = []

            for i, prompt in enumerate(prompts):
                results = inference_fn([prompt])
                r = results[0]
                if r.success:
                    if r.first_token_time_ms is not None:
                        run_ttfts.append(r.first_token_time_ms)
                    tpot = (r.total_time_ms - (r.first_token_time_ms or 0)) / max(r.output_tokens - 1, 1)
                    run_tpots.append(tpot)

                    if not is_warmup:
                        all_samples.append(SampleRecord(
                            request_id=i, batch_size=1, scenario="interactive",
                            input_tokens=self.suite["input_tokens"],
                            output_tokens=r.output_tokens,
                            ttft_ms=r.first_token_time_ms,
                            total_ms=r.total_time_ms, success=True
                        ))

            if not is_warmup:
                all_ttfts.extend(run_ttfts)
                all_tpots.extend(run_tpots)

        sampled = self._rng.sample(all_samples, min(MAX_SAMPLES_PER_CONFIG, len(all_samples)))
        self._write_samples(sampled)

        peak_memory = None  # platform script injects this

        return {"interactive": {
            "ttft_ms_p50": round(float(np.percentile(all_ttfts, 50)), 2) if all_ttfts else None,
            "ttft_ms_p90": round(float(np.percentile(all_ttfts, 90)), 2) if all_ttfts else None,
            "ttft_ms_p99": round(float(np.percentile(all_ttfts, 99)), 2) if all_ttfts else None,
            "tpot_ms_p50": round(float(np.percentile(all_tpots, 50)), 2) if all_tpots else None,
            "tpot_ms_p90": round(float(np.percentile(all_tpots, 90)), 2) if all_tpots else None,
            "tpot_ms_p99": round(float(np.percentile(all_tpots, 99)), 2) if all_tpots else None,
            "peak_memory_gb": peak_memory,
        }}

    # ------------------------------------------------------------------
    # Training scenario
    # ------------------------------------------------------------------

    def _run_training(self, inference_fn: Callable) -> dict:
        """
        Training throughput measurement.
        inference_fn here is a training step function, not inference.

        Training step function signature:
            def training_step_fn(step: int) -> TrainingStepResult
        where TrainingStepResult has: tokens_processed, step_time_ms, peak_memory_gb
        """
        # Training scenario is simpler: call the step function N times
        # LoadGen just handles timing aggregation
        # Platform script controls the actual training loop

        # This method is intentionally left for platform scripts to call directly.
        # See scripts/template/run_benchmark.py for the training scenario example.
        raise NotImplementedError(
            "Training scenario: use the helper in scripts/template/run_benchmark.py. "
            "LoadGen does not manage the training loop directly."
        )

    # ------------------------------------------------------------------
    # Multi-turn scenario
    # ------------------------------------------------------------------

    def _run_multiturn(self, inference_fn: Callable) -> dict:
        """
        Multi-turn conversation scenario.
        Groups requests by conversation_id and sends them sequentially.
        Measures how prefix caching affects performance.

        Only valid if requests.jsonl has conversation_id fields.
        """
        # Group by conversation
        from collections import defaultdict
        conversations = defaultdict(list)
        for r in self.requests:
            conv_id = r.get("conversation_id", str(r["request_id"]))
            conversations[conv_id].append(r)

        # Sort turns within each conversation
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda r: r.get("turn_index", 0))

        # Run conversations sequentially
        all_results = []
        for conv_id, turns in conversations.items():
            for turn in turns:
                result = inference_fn([turn["prompt"]])[0]
                all_results.append((conv_id, turn.get("turn_index", 0), result))

        # Metrics: compare first-turn vs subsequent-turn latency
        # (subsequent turns benefit from prefix cache)
        first_turn_ttfts = [r.first_token_time_ms for _, idx, r in all_results if idx == 0 and r.first_token_time_ms]
        later_turn_ttfts = [r.first_token_time_ms for _, idx, r in all_results if idx > 0 and r.first_token_time_ms]

        return {"multiturn": {
            "first_turn_ttft_p50": float(np.percentile(first_turn_ttfts, 50)) if first_turn_ttfts else None,
            "cached_turn_ttft_p50": float(np.percentile(later_turn_ttfts, 50)) if later_turn_ttfts else None,
            "cache_speedup_ratio": (
                float(np.percentile(first_turn_ttfts, 50)) / float(np.percentile(later_turn_ttfts, 50))
                if first_turn_ttfts and later_turn_ttfts else None
            ),
        }}

    # ------------------------------------------------------------------
    # Sustained scenario
    # ------------------------------------------------------------------

    def _run_sustained(self, inference_fn: Callable) -> dict:
        """
        Run at fixed QPS for suite.duration_minutes.
        Record throughput and latency every suite.sample_interval_seconds.
        Goal: detect thermal throttling, memory fragmentation, performance decay.
        """
        duration_s = self.suite["duration_minutes"] * 60
        interval_s = self.suite["sample_interval_seconds"]
        target_qps = self.suite["target_qps"]
        prompts = [r["prompt"] for r in self.requests]

        samples_over_time = []
        t_start = time.perf_counter()
        t_next_sample = t_start + interval_s
        t_next_request = t_start
        request_idx = 0

        while time.perf_counter() - t_start < duration_s:
            now = time.perf_counter()

            # Send request on schedule
            if now >= t_next_request:
                prompt = prompts[request_idx % len(prompts)]
                request_idx += 1
                result = inference_fn([prompt])[0]
                t_next_request += 1.0 / target_qps

            # Record sample at interval
            if time.perf_counter() >= t_next_sample:
                elapsed_min = (time.perf_counter() - t_start) / 60
                # Compute throughput over last interval
                samples_over_time.append({
                    "elapsed_minutes": round(elapsed_min, 1),
                    "throughput_tokens_per_sec": None,  # computed from recent requests
                    "ttft_ms_p99": None,
                })
                t_next_sample += interval_s

        # Compute stability metrics
        throughputs = [s["throughput_tokens_per_sec"] for s in samples_over_time if s["throughput_tokens_per_sec"]]
        if throughputs:
            initial = float(np.mean(throughputs[:3])) if len(throughputs) >= 3 else throughputs[0]
            final = float(np.mean(throughputs[-3:])) if len(throughputs) >= 3 else throughputs[-1]
            degradation = (initial - final) / initial if initial > 0 else 0
        else:
            initial = final = degradation = None

        return {"sustained": {
            "duration_minutes": self.suite["duration_minutes"],
            "target_qps": target_qps,
            "samples_over_time": samples_over_time,
            "initial_throughput": initial,
            "final_throughput": final,
            "degradation_pct": round(degradation * 100, 1) if degradation is not None else None,
        }}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_results(
        self,
        results: list[InferenceResult],
        batch_size: int,
        scenario: str,
    ) -> list[SampleRecord]:
        n = min(MAX_SAMPLES_PER_CONFIG, len(results))
        sampled_results = self._rng.sample(results, n)
        records = []
        for i, r in enumerate(sampled_results):
            records.append(SampleRecord(
                request_id=i, batch_size=batch_size, scenario=scenario,
                input_tokens=self.suite["input_tokens"],
                output_tokens=r.output_tokens,
                ttft_ms=r.first_token_time_ms,
                total_ms=r.total_ms if hasattr(r, 'total_ms') else r.total_time_ms,
                success=r.success,
            ))
        return records

    def _write_samples(self, samples: list[SampleRecord]) -> None:
        if not samples:
            return
        samples_path = self.output_dir / "samples.jsonl"
        with open(samples_path, "a") as f:
            for s in samples:
                f.write(json.dumps(s.__dict__) + "\n")
