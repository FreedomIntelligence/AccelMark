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


def _percentile(data: list, p: float):
    """Return the p-th percentile of data. Returns None if data is empty."""
    if not data:
        return None
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


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

    async def run_sustained(
        self,
        inference_fn,
        target_qps: float,
        duration_minutes: float,
        sample_interval_seconds: float,
        warmup_minutes: float = 2.0,
    ) -> dict:
        """
        Time-based, rate-controlled load test.

        Sends requests at target_qps for duration_minutes, recording a
        throughput/latency snapshot every sample_interval_seconds.

        Returns a metrics dict with a 'sustained' block containing:
          - samples: list of per-interval snapshots
          - sustained_throughput_tokens_per_sec: mean over all intervals
          - throttle_ratio: min_throughput / max_throughput (1.0 = no throttle)
          - throttle_onset_minute: first minute throughput drops >10% from peak
          - ttft_p99_drift_ms: ttft_p99 in last interval minus ttft_p99 in first
        """
        import asyncio
        import time as _time

        duration_seconds    = duration_minutes * 60
        interval_seconds    = sample_interval_seconds
        request_interval_s  = 1.0 / target_qps   # seconds between requests

        # Load requests (cycle through the suite's request pool)
        requests = self.requests   # already loaded in __init__
        if not requests:
            raise ValueError("run_sustained requires requests to be loaded.")

        samples          = []
        start_time       = _time.perf_counter()
        next_request_at  = start_time
        next_sample_at   = start_time + interval_seconds
        request_idx      = 0

        # Per-interval accumulators
        interval_tokens_out   = 0
        interval_tokens_in    = 0
        interval_ttfts        = []
        interval_start        = start_time
        interval_requests     = 0

        # Semaphore: allow up to 32 concurrent requests; let the event loop manage.
        sem = asyncio.Semaphore(32)

        async def _one_request(prompt: str):
            nonlocal interval_tokens_out, interval_tokens_in
            nonlocal interval_ttfts, interval_requests
            async with sem:
                result = await inference_fn(prompt)
            if result.success:
                interval_tokens_out += result.output_tokens
                interval_tokens_in  += result.input_tokens or 0
                if result.first_token_time_ms is not None:
                    interval_ttfts.append(result.first_token_time_ms)
                interval_requests += 1

        pending_tasks = set()

        while True:
            now = _time.perf_counter()
            elapsed = now - start_time

            # ── Check if duration exceeded ─────────────────────────────────
            if elapsed >= duration_seconds:
                break

            # ── Sample interval checkpoint ─────────────────────────────────
            if now >= next_sample_at:
                interval_elapsed = now - interval_start
                total_tokens = interval_tokens_out + interval_tokens_in
                throughput = total_tokens / interval_elapsed if interval_elapsed > 0 else 0

                ttft_p50 = _percentile(interval_ttfts, 50) if interval_ttfts else None
                ttft_p99 = _percentile(interval_ttfts, 99) if interval_ttfts else None

                samples.append({
                    "minute":                     round(elapsed / 60, 1),
                    "is_warmup":                  elapsed < warmup_minutes * 60,
                    "throughput_tokens_per_sec":  round(throughput, 1),
                    "tokens_out":                 interval_tokens_out,
                    "tokens_in":                  interval_tokens_in,
                    "requests_completed":         interval_requests,
                    "ttft_ms_p50":                round(ttft_p50, 1) if ttft_p50 else None,
                    "ttft_ms_p99":                round(ttft_p99, 1) if ttft_p99 else None,
                })

                # Reset interval accumulators
                interval_tokens_out  = 0
                interval_tokens_in   = 0
                interval_ttfts       = []
                interval_requests    = 0
                interval_start       = now
                next_sample_at      += interval_seconds

                # Log progress
                if ttft_p99:
                    print(
                        f"  [{elapsed/60:.1f}/{duration_minutes:.0f} min] "
                        f"{throughput:,.0f} tok/s | "
                        f"TTFT p99: {ttft_p99:.0f}ms"
                    )
                else:
                    print(
                        f"  [{elapsed/60:.1f}/{duration_minutes:.0f} min] "
                        f"{throughput:,.0f} tok/s"
                    )

            # ── Fire next request ──────────────────────────────────────────
            if now >= next_request_at:
                prompt_data = requests[request_idx % len(requests)]
                prompt      = prompt_data.get("prompt", "")
                request_idx += 1

                task = asyncio.create_task(_one_request(prompt))
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)

                next_request_at += request_interval_s

            # ── Yield control ──────────────────────────────────────────────
            sleep_until = min(next_request_at, next_sample_at)
            sleep_for   = max(0, sleep_until - _time.perf_counter())
            await asyncio.sleep(sleep_for)

        # Wait for in-flight requests to finish (up to 30s grace period)
        if pending_tasks:
            await asyncio.wait(pending_tasks, timeout=30)

        # ── Compute derived metrics ────────────────────────────────────────
        # Split warmup from analysis samples
        analysis_samples = [s for s in samples if not s.get("is_warmup")]

        # Use analysis_samples for all scalar metrics
        throughputs = [
            s["throughput_tokens_per_sec"]
            for s in analysis_samples
            if s["throughput_tokens_per_sec"]
        ]
        ttft_p99s = [
            s["ttft_ms_p99"]
            for s in analysis_samples
            if s["ttft_ms_p99"] is not None
        ]

        sustained_throughput = round(sum(throughputs) / len(throughputs), 1) if throughputs else None
        max_thr              = max(throughputs) if throughputs else None
        min_thr              = min(throughputs) if throughputs else None
        throttle_ratio       = round(min_thr / max_thr, 3) if max_thr and max_thr > 0 else None

        # Throttle onset: first minute where throughput drops >10% below peak
        throttle_onset_minute = None
        if max_thr:
            for s in analysis_samples:
                if s["throughput_tokens_per_sec"] < max_thr * 0.90:
                    throttle_onset_minute = s["minute"]
                    break

        # TTFT drift: last interval p99 minus first interval p99
        ttft_p99_drift_ms = None
        if len(ttft_p99s) >= 2:
            ttft_p99_drift_ms = round(ttft_p99s[-1] - ttft_p99s[0], 1)

        return {
            "sustained": {
                "target_qps":                      target_qps,
                "duration_minutes":                duration_minutes,
                "warmup_minutes":                  warmup_minutes,
                "sample_interval_seconds":         sample_interval_seconds,
                "samples":                         samples,
                "sustained_throughput_tokens_per_sec": sustained_throughput,
                "throttle_ratio":                  throttle_ratio,
                "throttle_onset_minute":           throttle_onset_minute,
                "ttft_p99_drift_ms":               ttft_p99_drift_ms,
            }
        }

    # ------------------------------------------------------------------
    # Offline scenario
    # ------------------------------------------------------------------

    def _run_offline(self, inference_fn: Callable) -> dict:
        """
        Send ALL requests to the engine at once for each configured concurrency level.
        The engine's internal scheduler handles batching optimally.

        concurrency_levels in suite.json define how many requests are sent simultaneously.
        They do NOT control client-side chunking or the engine's internal max_num_seqs.
        Higher client concurrency allows the engine to form larger internal batches,
        which may improve throughput up to the engine's scheduling limits.

        Throughput = (total_input_tokens + total_output_tokens) / elapsed,
        which matches vLLM's own internal throughput metric.

        NOTE — total_ms in samples.jsonl for offline:
        Each InferenceResult.total_time_ms is set by the runner to the wall-clock
        elapsed time of the entire batch (the time from sending all requests until
        the last one completes). Because LLM.generate() is a blocking call, all
        requests in a single run share the same total_ms value. This is by design —
        offline measures batch throughput, not per-request latency. Do not interpret
        offline total_ms as individual request completion times.
        """
        results_by_concurrency = []
        all_samples: list[SampleRecord] = []
        prompts = [r["prompt"] for r in self.requests]

        total_runs = self.suite["num_runs"] + self.warmup_runs
        total_concurrency_levels = len(self.suite["concurrency_levels"])

        total_steps = self.suite["num_runs"] * total_concurrency_levels
        print(f"\n{'='*60}")
        print(f"  AccelMark Offline Benchmark")
        print(f"  Requests  : {len(prompts)}")
        print(f"  Client concurrency levels: {self.suite['concurrency_levels']}")
        print(f"  Runs      : {self.suite['num_runs']} (+{self.warmup_runs} warmup)")
        print(f"{'='*60}\n")

        with tqdm(total=total_steps, desc="Overall progress",
                  unit="run", position=0, leave=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}, {rate_fmt}]"
                  ) as overall_pbar:

            for cc_idx, client_concurrency in enumerate(self.suite["concurrency_levels"]):
                run_throughputs = []
                run_elapsed_times = []
                run_samples: list[SampleRecord] = []

                for run_idx in range(total_runs):
                    is_warmup = run_idx < self.warmup_runs
                    run_label = "warmup" if is_warmup else \
                        f"run {run_idx - self.warmup_runs + 1}/{self.suite['num_runs']}"

                    desc = f"  client_concurrency={client_concurrency} ({cc_idx+1}/{total_concurrency_levels}) {run_label}"
                    tqdm.write(f"{desc} — sending all {len(prompts)} requests...")

                    t_start = time.perf_counter()

                    # Send ALL requests at once — engine handles internal batching
                    all_results: list[InferenceResult] = []
                    oom_occurred = False
                    try:
                        all_results = inference_fn(prompts)
                    except Exception as e:
                        err_str = str(e).lower()
                        if "out of memory" in err_str or "cuda" in err_str:
                            tqdm.write(
                                f"  [offline] client_concurrency={client_concurrency} OOM — recording and continuing"
                            )
                            oom_occurred = True
                        else:
                            raise

                    t_end = time.perf_counter()
                    elapsed = t_end - t_start

                    if oom_occurred:
                        results_by_concurrency.append({
                            "client_concurrency": client_concurrency,
                            "throughput_tokens_per_sec": None,
                            "throughput_tokens_per_sec_per_chip": None,
                            "elapsed_seconds_median": None,
                            "peak_memory_gb": None,
                            "power_watts_avg": None,
                            "power_watts_peak": None,
                            "oom": True,
                        })
                        try:
                            import torch as _torch
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                        break  # skip remaining runs for this client_concurrency

                    # Count both input and output tokens (matches vLLM's throughput metric)
                    total_input_tokens = sum(r.input_tokens for r in all_results if r.success)
                    total_output_tokens = sum(r.output_tokens for r in all_results if r.success)
                    total_tokens = total_input_tokens + total_output_tokens
                    throughput = total_tokens / elapsed if elapsed > 0 else 0
                    throughput_output_only = total_output_tokens / elapsed if elapsed > 0 else 0

                    if is_warmup:
                        tqdm.write(
                            f"  [warmup] client_concurrency={client_concurrency} — "
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
                        f"  [offline] client_concurrency={client_concurrency} {run_label} — "
                        f"{throughput:.0f} tok/s total "
                        f"({throughput_output_only:.0f} output only)"
                        f"{per_chip_str}  ({elapsed:.1f}s)"
                    )

                    overall_pbar.update(1)
                    overall_pbar.set_postfix({
                        "client_concurrency": client_concurrency,
                        "tok/s": f"{throughput:.0f}",
                    })

                    sampled = self._sample_results(all_results, client_concurrency, "offline")
                    run_samples.extend(sampled)

                all_samples.extend(run_samples)

                # Skip summary if OOM caused an early break (result already appended)
                if not run_throughputs:
                    continue

                median_throughput = float(np.median(run_throughputs))
                per_chip_str = f"  ({median_throughput/self.chip_count:.0f} tok/s per chip)" \
                    if self.chip_count > 1 else ""
                tqdm.write(
                    f"  [offline] client_concurrency={client_concurrency} "
                    f"median={median_throughput:.0f} tok/s{per_chip_str}\n"
                )

                results_by_concurrency.append({
                    "client_concurrency": client_concurrency,
                    "throughput_tokens_per_sec": round(median_throughput, 2),
                    "throughput_tokens_per_sec_per_chip": round(median_throughput / self.chip_count, 2),
                    "elapsed_seconds_median": round(float(np.median(run_elapsed_times)), 1),
                    "peak_memory_gb": None,
                    "power_watts_avg": None,
                    "power_watts_peak": None,
                    "oom": False,
                    "_concurrency_note": (
                        "client_concurrency is the number of requests sent simultaneously. "
                        "The inference engine batches internally; this does not directly "
                        "set engine parameters like max_num_seqs."
                    ),
                })

        self._write_samples(all_samples)
        return {"offline": {"results_by_concurrency": results_by_concurrency}}

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
        Training scenario is not supported in AccelMark inference benchmarks.
        Training benchmarks require a separate infrastructure (e.g. torchtitan).
        """
        raise NotImplementedError(
            "Training scenario is not implemented. "
            "AccelMark focuses on inference benchmarks. "
            "For training benchmarks, see AccelMark-Train (planned)."
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