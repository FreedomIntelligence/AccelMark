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

import asyncio
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: tqdm not installed, use a simple print wrapper
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(f"{desc} ...")
        return iterable

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
        chip_count: int = 1,
    ):
        """
        Args:
            suite:       Parsed contents of suite.json
            requests:    Parsed contents of requests.jsonl (list of {"prompt": str} dicts)
            scenario:    One of: offline, online, interactive, training
            output_dir:  Directory where samples.jsonl will be written
            chip_count:  Number of chips being used (affects throughput display per-chip)
        """
        self.suite = suite
        self.scenario = scenario
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(SAMPLE_SEED)
        self.chip_count = chip_count

        # Use different request counts per scenario
        # offline: use request_count (default 200, fast)
        # online/interactive: use online_request_count if set, else all requests
        if scenario == "offline":
            count = suite.get("request_count")
            self.warmup_runs = suite.get("warmup_runs", 1)
        elif scenario == "online":
            # online and interactive need more requests for reliable p99
            count = suite.get("online_request_count", suite.get("request_count"))
            self.warmup_runs = suite.get("online_warmup_runs", 0)
        elif scenario == "interactive":
            count = suite.get("interactive_request_count", suite.get("request_count"))
            self.warmup_runs = suite.get("interactive_warmup_runs", 0)
        else:
            count = suite.get("request_count")
            self.warmup_runs = suite.get("warmup_runs", 1)

        self.requests = requests[:count] if count else requests

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
            if not asyncio.iscoroutinefunction(inference_fn):
                raise TypeError(
                    "_run_interactive requires an async inference_fn(prompt: str) -> InferenceResult. "
                    "Pass an async coroutine (e.g. _run_one_streaming from run_vllm.py)."
                )
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._run_interactive_async(inference_fn))
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
        Send ALL requests to the engine at once for each configured concurrency level.
        The engine's internal scheduler handles batching optimally.

        batch_sizes in suite.json map to max concurrent request levels tested —
        they do NOT control client-side chunking. Higher values allow more
        in-flight sequences and improve GPU utilization up to hardware limits.

        Throughput = (total_input_tokens + total_output_tokens) / elapsed,
        which matches vLLM's own internal throughput metric.
        """
        results_by_batch_size = []
        all_samples: list[SampleRecord] = []
        prompts = [r["prompt"] for r in self.requests]

        total_runs = self.suite["num_runs"] + self.warmup_runs
        total_batch_sizes = len(self.suite["batch_sizes"])

        total_steps = self.suite["num_runs"] * total_batch_sizes
        print(f"\n{'='*60}")
        print(f"  AccelMark Offline Benchmark")
        print(f"  Requests  : {len(prompts)}")
        print(f"  Max concurrency levels: {self.suite['batch_sizes']}")
        print(f"  Runs      : {self.suite['num_runs']} (+{self.warmup_runs} warmup)")
        print(f"{'='*60}\n")

        with tqdm(total=total_steps, desc="Overall progress",
                  unit="run", position=0, leave=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}, {rate_fmt}]"
                  ) as overall_pbar:

            for bs_idx, batch_size in enumerate(self.suite["batch_sizes"]):
                run_throughputs = []
                run_elapsed_times = []
                run_samples: list[SampleRecord] = []

                for run_idx in range(total_runs):
                    is_warmup = run_idx < self.warmup_runs
                    run_label = "warmup" if is_warmup else \
                        f"run {run_idx - self.warmup_runs + 1}/{self.suite['num_runs']}"

                    desc = f"  concurrency={batch_size} ({bs_idx+1}/{total_batch_sizes}) {run_label}"
                    tqdm.write(f"{desc} — sending all {len(prompts)} requests...")

                    t_start = time.perf_counter()

                    # Send ALL requests at once — engine handles internal batching
                    all_results: list[InferenceResult] = inference_fn(prompts)

                    t_end = time.perf_counter()
                    elapsed = t_end - t_start

                    # Count both input and output tokens (matches vLLM's throughput metric)
                    total_input_tokens = sum(r.input_tokens for r in all_results if r.success)
                    total_output_tokens = sum(r.output_tokens for r in all_results if r.success)
                    total_tokens = total_input_tokens + total_output_tokens
                    throughput = total_tokens / elapsed if elapsed > 0 else 0
                    throughput_output_only = total_output_tokens / elapsed if elapsed > 0 else 0

                    if is_warmup:
                        tqdm.write(
                            f"  [warmup] concurrency={batch_size} — "
                            f"{throughput:.0f} tok/s total, "
                            f"{throughput_output_only:.0f} tok/s output (not recorded)"
                        )
                        continue

                    run_throughputs.append(throughput)
                    run_elapsed_times.append(elapsed)

                    per_chip_str = ""
                    if self.chip_count > 1:
                        per_chip_str = f"  ({throughput/self.chip_count:.0f} tok/s per chip)"

                    tqdm.write(
                        f"  [offline] concurrency={batch_size} {run_label} — "
                        f"{throughput:.0f} tok/s total "
                        f"({throughput_output_only:.0f} output only)"
                        f"{per_chip_str}  ({elapsed:.1f}s)"
                    )

                    overall_pbar.update(1)
                    overall_pbar.set_postfix({
                        "concurrency": batch_size,
                        "tok/s": f"{throughput:.0f}",
                    })

                    sampled = self._sample_results(all_results, batch_size, "offline")
                    run_samples.extend(sampled)

                all_samples.extend(run_samples)

                median_throughput = float(np.median(run_throughputs))
                per_chip_str = f"  ({median_throughput/self.chip_count:.0f} tok/s per chip)" \
                    if self.chip_count > 1 else ""
                tqdm.write(
                    f"  [offline] concurrency={batch_size} "
                    f"median={median_throughput:.0f} tok/s{per_chip_str}\n"
                )

                results_by_batch_size.append({
                    "batch_size": batch_size,
                    "throughput_tokens_per_sec": round(median_throughput, 2),
                    "throughput_tokens_per_sec_per_chip": round(median_throughput / self.chip_count, 2),
                    "elapsed_seconds_median": round(float(np.median(run_elapsed_times)), 1),
                    "peak_memory_gb": None,
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

        inference_fn must be an async coroutine: async def fn(prompt: str) -> InferenceResult
        Requests are dispatched concurrently according to Poisson arrival times so that
        the engine experiences realistic queueing pressure.
        """
        if not asyncio.iscoroutinefunction(inference_fn):
            raise TypeError(
                "_run_online requires an async inference_fn(prompt: str) -> InferenceResult. "
                "Pass an async coroutine (e.g. _run_one_streaming from run_vllm.py), "
                "not a sync wrapper."
            )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._run_online_async(inference_fn))

    async def _run_online_async(self, async_inference_fn) -> dict:
        """
        Async implementation of the online scenario.
        Generates Poisson arrival times upfront, then fires all requests
        concurrently via asyncio.gather so the engine sees real concurrent load.
        """
        loop = asyncio.get_event_loop()
        sla_ms = self.suite["online_sla_ttft_ms"]
        results_by_qps = []
        all_samples: list[SampleRecord] = []
        prompts = [r["prompt"] for r in self.requests]
        max_valid_qps = 0.0

        for target_qps in self.suite["online_qps_levels"]:
            print(f"[online] target_qps={target_qps}")
            run_ttfts: list[list[float]] = []
            run_tpots: list[list[float]] = []
            run_elapsed_times: list[float] = []

            for run_idx in range(self.suite["num_runs"]):
                run_label = f"run {run_idx + 1}/{self.suite['num_runs']}"
                n = len(prompts)

                # Generate all Poisson inter-arrival times upfront
                inter_arrivals = [self._rng.expovariate(target_qps) for _ in range(n)]
                arrival_times = list(itertools.accumulate(inter_arrivals))

                t_start = loop.time()

                async def send_request(p: str, t_arrival: float) -> InferenceResult:
                    delay = t_arrival - (loop.time() - t_start)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return await async_inference_fn(p)

                tqdm.write(
                    f"[online] qps={target_qps} {run_label} "
                    f"— dispatching {n} requests with Poisson arrivals..."
                )

                # Run all requests concurrently; each sleeps until its arrival time
                all_results: list[InferenceResult] = list(
                    await asyncio.gather(
                        *[send_request(p, t) for p, t in zip(prompts, arrival_times)]
                    )
                )

                t_run_end = loop.time()
                run_elapsed_times.append(t_run_end - t_start)

                ttfts: list[float] = []
                tpots: list[float] = []
                for r in all_results:
                    if r.success:
                        if r.first_token_time_ms is not None:
                            ttfts.append(r.first_token_time_ms)
                        tpot = (r.total_time_ms - (r.first_token_time_ms or 0)) / max(r.output_tokens - 1, 1)
                        tpots.append(tpot)

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

            sla_icon = "✓" if sla_met else "✗"
            chip_str = f"  ({self.chip_count} chips)" if self.chip_count > 1 else ""
            tqdm.write(f"  [online] qps={target_qps} TTFT_p99={ttft_p99:.0f}ms SLA={sla_ms}ms {sla_icon}{chip_str}")

            results_by_qps.append({
                "target_qps": target_qps,
                "achieved_qps": round(achieved_qps, 2),
                "ttft_ms_p50": round(ttft_p50, 2),
                "ttft_ms_p90": round(ttft_p90, 2),
                "ttft_ms_p99": round(ttft_p99, 2),
                "tpot_ms_p50": round(tpot_p50, 2),
                "tpot_ms_p90": round(tpot_p90, 2),
                "tpot_ms_p99": round(tpot_p99, 2),
                "elapsed_seconds_median": round(float(np.median(run_elapsed_times)), 1),
                "sla_met": sla_met,
            })

        self._write_samples(all_samples)
        return {"online": {"max_valid_qps": max_valid_qps, "results_by_qps": results_by_qps}}

    # ------------------------------------------------------------------
    # Interactive scenario
    # ------------------------------------------------------------------

    async def _run_interactive_async(self, async_inference_fn) -> dict:
        """
        Send one request at a time, waiting for completion before sending the next.
        Measures single-request latency in isolation (no queueing pressure).
        Uses the same async engine as online to ensure consistent TTFT measurement.
        """
        prompts = [r["prompt"] for r in self.requests]
        all_ttfts: list[float] = []
        all_tpots: list[float] = []
        all_samples: list[SampleRecord] = []
        run_elapsed_times: list[float] = []

        total_runs = self.warmup_runs + self.suite["num_runs"]

        for run_idx in range(total_runs):
            is_warmup = run_idx < self.warmup_runs
            run_label = "warmup" if is_warmup else \
                f"run {run_idx - self.warmup_runs + 1}/{self.suite['num_runs']}"

            run_ttfts: list[float] = []
            run_tpots: list[float] = []
            t_run_start = time.perf_counter()

            for i, prompt in enumerate(prompts):
                # Send one request, await completion before next — true serial
                r = await async_inference_fn(prompt)

                if r.success:
                    if r.first_token_time_ms is not None:
                        run_ttfts.append(r.first_token_time_ms)
                    tpot = (r.total_time_ms - (r.first_token_time_ms or 0)) / max(r.output_tokens - 1, 1)
                    run_tpots.append(tpot)

                    if not is_warmup:
                        all_samples.append(SampleRecord(
                            request_id=i, batch_size=1, scenario="interactive",
                            input_tokens=self.suite.get("input_tokens",
                                self.suite.get("request_distribution", {}).get("input_tokens_p50")),
                            output_tokens=r.output_tokens,
                            ttft_ms=r.first_token_time_ms,
                            total_ms=r.total_time_ms,
                            success=True,
                        ))

                # Print progress every 10 requests
                if (i + 1) % 10 == 0:
                    tqdm.write(
                        f"  [interactive] {run_label} {i+1}/{len(prompts)} "
                        f"— last TTFT: {r.first_token_time_ms:.0f}ms" if r.first_token_time_ms else ""
                    )

            t_run_end = time.perf_counter()
            run_elapsed = t_run_end - t_run_start

            if is_warmup:
                tqdm.write(f"  [warmup] interactive done ({run_elapsed:.0f}s, not recorded)")
                continue

            all_ttfts.extend(run_ttfts)
            all_tpots.extend(run_tpots)
            run_elapsed_times.append(run_elapsed)

            if run_ttfts:
                tqdm.write(
                    f"  [interactive] {run_label} — "
                    f"TTFT p50={float(np.percentile(run_ttfts, 50)):.0f}ms "
                    f"p99={float(np.percentile(run_ttfts, 99)):.0f}ms "
                    f"({run_elapsed:.0f}s)"
                )

        sampled = self._rng.sample(all_samples, min(MAX_SAMPLES_PER_CONFIG, len(all_samples)))
        self._write_samples(sampled)

        return {"interactive": {
            "ttft_ms_p50": round(float(np.percentile(all_ttfts, 50)), 2) if all_ttfts else None,
            "ttft_ms_p90": round(float(np.percentile(all_ttfts, 90)), 2) if all_ttfts else None,
            "ttft_ms_p99": round(float(np.percentile(all_ttfts, 99)), 2) if all_ttfts else None,
            "tpot_ms_p50": round(float(np.percentile(all_tpots, 50)), 2) if all_tpots else None,
            "tpot_ms_p90": round(float(np.percentile(all_tpots, 90)), 2) if all_tpots else None,
            "tpot_ms_p99": round(float(np.percentile(all_tpots, 99)), 2) if all_tpots else None,
            "peak_memory_gb": None,
            "elapsed_seconds_median": round(float(np.median(run_elapsed_times)), 1) if run_elapsed_times else None,
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
        total_turns = sum(len(turns) for turns in conversations.values())
        for conv_id, turns in tqdm(conversations.items(), desc="[multiturn] conversations", unit="conv"):
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

        print(f"[sustained] Running {self.suite['duration_minutes']} min at {target_qps} QPS...")
        with tqdm(total=self.suite["duration_minutes"], desc="[sustained]", unit="min", bar_format="{l_bar}{bar}| {n:.1f}/{total}min [{elapsed}<{remaining}]") as pbar:
            last_pbar_update = t_start

            while time.perf_counter() - t_start < duration_s:
                now = time.perf_counter()

                # Update progress bar every second
                elapsed_min = (now - t_start) / 60
                delta = elapsed_min - (last_pbar_update - t_start) / 60
                if delta >= 1/60:  # update every ~1 second
                    pbar.n = min(elapsed_min, self.suite["duration_minutes"])
                    pbar.refresh()
                    last_pbar_update = now

                # Send request on schedule
                if now >= t_next_request:
                    prompt = prompts[request_idx % len(prompts)]
                    request_idx += 1
                    result = inference_fn([prompt])[0]
                    t_next_request += 1.0 / target_qps

                # Record sample at interval
                if time.perf_counter() >= t_next_sample:
                    elapsed_min = (time.perf_counter() - t_start) / 60
                    samples_over_time.append({
                        "elapsed_minutes": round(elapsed_min, 1),
                        "throughput_tokens_per_sec": None,
                        "ttft_ms_p99": None,
                    })
                    print(f"  [sustained] {elapsed_min:.1f}min — sample recorded")
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
                input_tokens=self.suite.get("input_tokens", self.suite.get("request_distribution", {}).get("input_tokens_p50")),
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