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

# Import InferenceRequest for type hints — runners pass list[InferenceRequest]
# to inference_fn_offline and inference_fn_streaming
try:
    from runners.benchmark_runner import InferenceRequest
except ImportError:
    # Fallback if running loadgen standalone — define a minimal shim
    from dataclasses import dataclass as _dc, field as _field
    from typing import Optional as _Opt
    @_dc
    class InferenceRequest:
        prompt:       str
        request_id:   int           = 0
        input_tokens: int           = 0
        max_tokens:   _Opt[int]     = None
        temperature:  float         = 0.0
        extra:        dict          = _field(default_factory=dict)

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


# ── Reliability helpers ──────────────────────────────────────────────────────
#
# These produce the inter-run variability metrics consumed by the leaderboard
# UI's "Reliability" panel. They live here (not in types.py) because they are
# pure functions over already-collected per-run lists and are easier to
# regression-test alongside the scenario implementations.
#
# Coefficient of Variation (CV) = std / mean × 100 %, computed with ddof=1
# (sample std) when n ≥ 2. Returns None when input is too small or the mean
# is non-positive, in which case the frontend hides the badge entirely so
# users do not see a meaningless "stable ✓" on a single-run measurement.

# Stability thresholds. These are intentionally permissive on first launch —
# real-world hardware noise (especially memory thrashing on first cycle)
# regularly crosses 2 % even on healthy systems. Tune in the schema after we
# observe the first wave of submissions.
_STABILITY_THRESHOLD_STABLE_PCT = 2.0
_STABILITY_THRESHOLD_NOISY_PCT  = 5.0


def _cv_pct(values: list) -> Optional[float]:
    """Coefficient of variation as a percentage. None if too small / undefined."""
    if not values or len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return None
    mean = float(arr.mean())
    if mean <= 0:
        return None
    std = float(arr.std(ddof=1))
    return round(std / mean * 100.0, 2)


def _stability_label(cv_pct: Optional[float]) -> Optional[str]:
    """Map a CV percentage to a stable/noisy/unstable label, or None."""
    if cv_pct is None:
        return None
    if cv_pct <= _STABILITY_THRESHOLD_STABLE_PCT:
        return "stable"
    if cv_pct <= _STABILITY_THRESHOLD_NOISY_PCT:
        return "noisy"
    return "unstable"


def _reliability_block(values: list, *, decimals: int = 2) -> dict:
    """
    Build the standard {n, mean, std, cv_pct, stability, runs} block emitted
    per metric. Returns an empty dict (not None) so the result schema retains
    a consistent shape — frontend gates on `cv_pct` being numeric.
    """
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    cv   = _cv_pct(arr.tolist())
    return {
        "n":         int(len(arr)),
        "mean":      round(mean, decimals),
        "std":       round(std, decimals),
        "cv_pct":    cv,
        "stability": _stability_label(cv),
        "runs":      [round(float(v), decimals) for v in arr.tolist()],
    }


def _compute_recovery_time(
    arrivals: list,
    ttfts: list,
    *,
    threshold_ms: float,
    window_s: float = 3.0,
    min_samples: int = 5,
) -> Optional[float]:
    """
    Find the elapsed time (seconds, relative to the start of the post-burst
    steady window) at which a rolling-window p99 of TTFT first falls below
    `threshold_ms`. Returns None if it never recovers within the window or
    if there are too few samples to compute a stable percentile.

    `arrivals` and `ttfts` are parallel arrays — arrivals must be relative
    times in seconds from the start of the measurement window.
    """
    if not arrivals or len(arrivals) < min_samples:
        return None
    pairs = sorted(zip(arrivals, ttfts))
    a = [p[0] for p in pairs]
    t = [p[1] for p in pairs]
    n = len(a)
    j = 0
    for i in range(n):
        while j < i and a[j] < a[i] - window_s:
            j += 1
        if i - j + 1 < min_samples:
            continue
        window = t[j:i + 1]
        if float(np.percentile(window, 99)) < threshold_ms:
            return round(float(a[i]), 2)
    return None


class AccelMarkLoadGen:

    def __init__(
        self,
        suite: dict,
        requests: list,
        scenario: str,
        output_dir: str,
        chip_count: int = 1,
    ):
        """
        Args:
            suite:       Parsed contents of suite.json
            requests:    List of InferenceRequest objects (built from requests.jsonl
                         by benchmark_runner._run_single_scenario)
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
        #
        # Warmup semantics differ per scenario:
        #   offline / interactive : `warmup_runs` = number of full passes to discard
        #                           (interactive_warmup_runs may override for interactive)
        #   sustained             : `warmup_minutes` = time window discarded
        #   online / burst        : `online_warmup_requests` / `burst_warmup_requests`
        #                           = number of dummy requests fired sequentially before
        #                           the timed phase, used to JIT-compile kernels, allocate
        #                           CUDA graphs, prime the KV cache, etc. Results are
        #                           never recorded. Without this warmup, the first few
        #                           requests of the first QPS level inflate p99 by
        #                           hundreds of ms on cold engines.
        self.online_warmup_requests = 0
        self.burst_warmup_requests = 0
        if scenario == "offline":
            count = suite.get("request_count")
            self.warmup_runs = suite.get("warmup_runs", 1)
        elif scenario == "online":
            # online and interactive need more requests for reliable p99
            count = suite.get("online_request_count", suite.get("request_count"))
            self.warmup_runs = 0  # online doesn't use full-pass warmup
            self.online_warmup_requests = suite.get("online_warmup_requests", 10)
        elif scenario == "interactive":
            count = suite.get("interactive_request_count", suite.get("request_count"))
            self.warmup_runs = suite.get("interactive_warmup_runs", 0)
        elif scenario == "burst":
            count = suite.get("online_request_count", suite.get("request_count"))
            self.warmup_runs = 0  # burst doesn't use full-pass warmup
            self.burst_warmup_requests = suite.get("burst_warmup_requests", 10)
        elif scenario == "speculative":
            count = suite.get("request_count")
            self.warmup_runs = suite.get("warmup_runs", 1)
        else:
            count = suite.get("request_count")
            self.warmup_runs = suite.get("warmup_runs", 1)

        self.requests = requests[:count] if count else requests

    def run(self, inference_fn: Callable) -> dict:
        """
        Run the benchmark for the configured scenario.

        inference_fn signature:
            offline:  def inference_fn(requests: list[InferenceRequest]) -> list[InferenceResult]
            online/interactive/sustained: async def inference_fn(request: InferenceRequest) -> InferenceResult

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
                    "_run_interactive requires an async inference_fn(request: InferenceRequest) -> InferenceResult. "
                    "Pass an async coroutine (inference_fn_streaming)."
                )
            return asyncio.run(self._run_interactive_async(inference_fn))
        elif self.scenario == "training":
            return self._run_training(inference_fn)
        elif self.scenario == "multiturn":
            return self._run_multiturn(inference_fn)
        elif self.scenario == "sustained":
            return self._run_sustained(inference_fn)
        elif self.scenario == "burst":
            return self._run_burst(inference_fn)
        elif self.scenario == "speculative":
            result = self._run_offline(inference_fn)
            return {"speculative": result.get("offline", {})}
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    async def run_sustained(
        self,
        inference_fn,
        sustained_concurrency: int,
        duration_minutes: float,
        sample_interval_seconds: float,
        warmup_minutes: float = 2.0,
    ) -> dict:
        """
        Fixed-concurrency sustained load test.

        Keeps exactly `sustained_concurrency` requests in-flight at all times.
        A new request fires the moment one completes — no rate limiting, no queue
        buildup. The hardware is always busy.

        This cleanly separates hardware/memory degradation from scheduling effects:
        any throughput drop at constant concurrency is genuine degradation, not
        queue saturation.

        Returns a metrics dict with a 'sustained' block containing per-interval
        samples and derived scalar metrics (sustained_throughput, throttle_ratio,
        throttle_onset_minute, ttft_p99_drift_ms).
        """
        import asyncio
        import time as _time

        duration_seconds  = duration_minutes * 60
        interval_seconds  = sample_interval_seconds

        requests = list(self.requests)
        if not requests:
            raise ValueError("run_sustained requires requests to be loaded.")

        samples           = []
        start_time        = _time.perf_counter()
        next_sample_at    = start_time + interval_seconds
        request_idx       = 0

        # Per-interval accumulators
        interval_tokens_out  = 0
        interval_tokens_in   = 0
        interval_ttfts       = []
        interval_requests    = 0
        interval_start       = start_time

        # ── Fixed-concurrency semaphore ───────────────────────────────────────
        # Limits in-flight count to exactly sustained_concurrency.
        sem = asyncio.Semaphore(sustained_concurrency)

        async def _one_request(req: InferenceRequest) -> None:
            nonlocal interval_tokens_out, interval_tokens_in
            nonlocal interval_ttfts, interval_requests
            async with sem:
                result = await inference_fn(req)
            if result.success:
                interval_tokens_out += result.output_tokens
                interval_tokens_in  += result.input_tokens or 0
                if result.first_token_time_ms is not None:
                    interval_ttfts.append(result.first_token_time_ms)
                interval_requests += 1

        pending_tasks = set()

        def _fire_request() -> None:
            nonlocal request_idx
            req = requests[request_idx % len(requests)]
            request_idx += 1
            task = asyncio.create_task(_one_request(req))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        # ── Pre-fill to sustained_concurrency ────────────────────────────────
        for _ in range(sustained_concurrency):
            _fire_request()

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            now     = _time.perf_counter()
            elapsed = now - start_time

            # Duration check
            if elapsed >= duration_seconds:
                break

            # Sample interval checkpoint
            if now >= next_sample_at:
                interval_elapsed = now - interval_start
                total_tokens     = interval_tokens_out + interval_tokens_in
                throughput       = (
                    interval_tokens_out / interval_elapsed if interval_elapsed > 0 else 0
                )

                ttft_p50 = _percentile(interval_ttfts, 50) if interval_ttfts else None
                ttft_p99 = _percentile(interval_ttfts, 99) if interval_ttfts else None

                samples.append({
                    "minute":                    round(elapsed / 60, 1),
                    "is_warmup":                 elapsed < warmup_minutes * 60,
                    "throughput_tokens_per_sec": round(throughput, 1),
                    "tokens_out":                interval_tokens_out,
                    "tokens_in":                 interval_tokens_in,
                    "requests_completed":        interval_requests,
                    "ttft_ms_p50":               round(ttft_p50, 1) if ttft_p50 else None,
                    "ttft_ms_p99":               round(ttft_p99, 1) if ttft_p99 else None,
                })

                # Log progress
                msg = (
                    f"  [{elapsed/60:.1f}/{duration_minutes:.0f} min] "
                    f"{throughput:,.0f} tok/s"
                )
                if ttft_p99:
                    msg += f" | TTFT p99: {ttft_p99:.0f}ms"
                print(msg)

                # Reset interval accumulators
                interval_tokens_out  = 0
                interval_tokens_in   = 0
                interval_ttfts       = []
                interval_requests    = 0
                interval_start       = now
                next_sample_at      += interval_seconds

            # Keep concurrency topped up — fire new requests as slots open.
            # The semaphore limits actual in-flight count; we can queue extras
            # without concern because sem.acquire() blocks inside _one_request.
            while len(pending_tasks) < sustained_concurrency * 2:
                _fire_request()

            await asyncio.sleep(0.05)   # yield control, check again shortly

        # Grace period for in-flight requests
        if pending_tasks:
            await asyncio.wait(pending_tasks, timeout=60)

        # ── Derived metrics ───────────────────────────────────────────────────
        # Exclude warmup samples from scalar metrics
        analysis_samples = [s for s in samples if not s.get("is_warmup")]

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

        sustained_throughput = (
            round(sum(throughputs) / len(throughputs), 1) if throughputs else None
        )
        max_thr        = max(throughputs) if throughputs else None
        min_thr        = min(throughputs) if throughputs else None
        throttle_ratio = (
            round(min_thr / max_thr, 3) if max_thr and max_thr > 0 else None
        )

        throttle_onset_minute = None
        if max_thr:
            for s in analysis_samples:
                if s["throughput_tokens_per_sec"] < max_thr * 0.90:
                    throttle_onset_minute = s["minute"]
                    break

        ttft_p99_drift_ms = None
        if len(ttft_p99s) >= 2:
            ttft_p99_drift_ms = round(ttft_p99s[-1] - ttft_p99s[0], 1)

        # Inter-sample throughput stability across the post-warmup window.
        # This is conceptually distinct from `throttle_ratio` (min/max): CV
        # measures dispersion around the mean and is a better signal for
        # "the chip throttles intermittently" vs "the chip is degrading".
        throughput_cv_block = _reliability_block(throughputs, decimals=1)

        return {
            "sustained": {
                "sustained_concurrency":               sustained_concurrency,
                "duration_minutes":                    duration_minutes,
                "warmup_minutes":                      warmup_minutes,
                "sample_interval_seconds":             sample_interval_seconds,
                "samples":                             samples,
                "sustained_throughput_tokens_per_sec": sustained_throughput,
                "throttle_ratio":                      throttle_ratio,
                "throttle_onset_minute":               throttle_onset_minute,
                "ttft_p99_drift_ms":                   ttft_p99_drift_ms,
                "throughput_post_warmup_reliability":  throughput_cv_block,
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

        Throughput = total_output_tokens / elapsed (output tokens only).
        This is consistent with the sustained scenario and standard LLM throughput
        reporting practice. Input tokens are tracked internally but not recorded
        as the primary metric.

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

        total_runs = self.suite["num_runs"] + self.warmup_runs
        total_concurrency_levels = len(self.suite["concurrency_levels"])

        total_steps = self.suite["num_runs"] * total_concurrency_levels
        _scenario_label = "Speculative Decoding" if self.scenario == "speculative" else "Offline"
        print(f"\n{'='*60}")
        print(f"  AccelMark {_scenario_label} Benchmark")
        print(f"  Requests  : {len(self.requests)}")
        print(f"  Client concurrency levels: {self.suite['concurrency_levels']}")
        print(f"  Runs      : {self.suite['num_runs']} (+{self.warmup_runs} warmup)")
        print(f"{'='*60}\n")

        with tqdm(total=total_steps, desc="Overall progress",
                  unit="run", position=0, leave=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}, {rate_fmt}]"
                  ) as overall_pbar:

            for cc_idx, client_concurrency in enumerate(self.suite["concurrency_levels"]):
                run_throughputs = []
                run_total_throughputs = []
                run_elapsed_times = []
                run_samples: list[SampleRecord] = []

                for run_idx in range(total_runs):
                    is_warmup = run_idx < self.warmup_runs
                    run_label = "warmup" if is_warmup else \
                        f"run {run_idx - self.warmup_runs + 1}/{self.suite['num_runs']}"

                    desc = f"  client_concurrency={client_concurrency} ({cc_idx+1}/{total_concurrency_levels}) {run_label}"
                    tqdm.write(f"{desc} — sending all {len(self.requests)} requests...")

                    t_start = time.perf_counter()

                    # Send ALL requests at once — engine handles internal batching
                    all_results: list[InferenceResult] = []
                    oom_occurred = False
                    try:
                        all_results = inference_fn(self.requests)
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

                    # Count output tokens only for throughput (consistent with sustained/online).
                    # Input tokens are also tracked for reference but not used as the primary metric.
                    total_input_tokens = sum(r.input_tokens for r in all_results if r.success)
                    total_output_tokens = sum(r.output_tokens for r in all_results if r.success)
                    total_tokens = total_input_tokens + total_output_tokens
                    throughput_output_only = total_output_tokens / elapsed if elapsed > 0 else 0
                    throughput_total = total_tokens / elapsed if elapsed > 0 else 0

                    if is_warmup:
                        tqdm.write(
                            f"  [warmup] client_concurrency={client_concurrency} — "
                            f"{throughput_output_only:.0f} tok/s output, "
                            f"{throughput_total:.0f} tok/s total (input+output)"
                        )
                        continue

                    run_throughputs.append(throughput_output_only)
                    run_total_throughputs.append(throughput_total)
                    run_elapsed_times.append(elapsed)

                    per_chip_str = ""
                    if self.chip_count > 1:
                        per_chip_str = f"  ({throughput_output_only/self.chip_count:.0f} tok/s per chip)"

                    tqdm.write(
                        f"  [offline] client_concurrency={client_concurrency} {run_label} — "
                        f"{throughput_output_only:.0f} tok/s output "
                        f"({throughput_total:.0f} tok/s total input+output)"
                        f"{per_chip_str}  ({elapsed:.1f}s)"
                    )

                    overall_pbar.update(1)
                    overall_pbar.set_postfix({
                        "client_concurrency": client_concurrency,
                        "tok/s": f"{throughput_output_only:.0f}",
                    })

                    sampled = self._sample_results(all_results, client_concurrency, "offline")
                    run_samples.extend(sampled)

                all_samples.extend(run_samples)

                # Skip summary if OOM caused an early break (result already appended)
                if not run_throughputs:
                    continue

                median_throughput = float(np.median(run_throughputs))
                median_throughput_total = float(np.median(run_total_throughputs))
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
                    "throughput_tokens_per_sec_total": round(median_throughput_total, 2),
                    "elapsed_seconds_median": round(float(np.median(run_elapsed_times)), 1),
                    "peak_memory_gb": None,
                    "power_watts_avg": None,
                    "power_watts_peak": None,
                    "oom": False,
                    # Per-run throughput reliability: lets the UI show "stable ✓ /
                    # noisy ⚠ / unstable ✗" without forcing the user to download
                    # samples.jsonl. `runs` preserves the underlying values so
                    # future stability rules can be recomputed without a re-run.
                    "throughput_tokens_per_sec_reliability":
                        _reliability_block(run_throughputs, decimals=2),
                    "_throughput_note": "output_only",
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

        inference_fn must be an async coroutine: async def fn(request: InferenceRequest) -> InferenceResult
        Requests are dispatched concurrently according to Poisson arrival times so that
        the engine experiences realistic queueing pressure.
        """
        if not asyncio.iscoroutinefunction(inference_fn):
            raise TypeError(
                "_run_online requires an async inference_fn(request: InferenceRequest) -> InferenceResult. "
                "Pass an async coroutine (inference_fn_streaming), "
                "not a sync wrapper."
            )
        return asyncio.run(self._run_online_async(inference_fn))

    async def _warmup_requests(self, async_inference_fn, count: int, label: str) -> None:
        """
        Fire `count` dummy requests sequentially before timed measurement.

        Cycles through self.requests if count > len(requests). All results are
        discarded — purpose is to JIT-compile kernels, allocate CUDA graphs,
        prime the KV cache, and let the engine reach steady-state schedules
        before the timed phase. Without this, the first few timed requests on
        cold engines inflate p99 by hundreds of milliseconds.

        Exceptions during warmup are logged and swallowed; warmup failures
        must never abort the timed run.
        """
        if count <= 0 or not self.requests:
            return
        tqdm.write(
            f"[{label} warmup] firing {count} dummy requests "
            "(results discarded — engine JIT/cache warm-up)"
        )
        for i in range(count):
            req = self.requests[i % len(self.requests)]
            try:
                await async_inference_fn(req)
            except Exception as e:
                tqdm.write(f"[{label} warmup] request {i} failed (ignored): {e}")

    async def _run_online_async(self, async_inference_fn) -> dict:
        """
        Async implementation of the online scenario.
        Generates Poisson arrival times upfront, then fires all requests
        concurrently via asyncio.gather so the engine sees real concurrent load.

        A warmup phase fires `online_warmup_requests` dummy requests
        sequentially before the QPS sweep. Their latencies are not recorded
        in `results_by_qps`. This prevents cold-engine TTFT spikes from
        inflating p99 at the first QPS level.
        """
        loop = asyncio.get_event_loop()
        sla_ms = self.suite["online_sla_ttft_ms"]
        results_by_qps = []
        all_samples: list[SampleRecord] = []
        max_valid_qps = 0.0

        await self._warmup_requests(
            async_inference_fn, self.online_warmup_requests, "online"
        )

        for target_qps in self.suite["online_qps_levels"]:
            print(f"[online] target_qps={target_qps}")
            run_ttfts: list[list[float]] = []
            run_tpots: list[list[float]] = []
            run_elapsed_times: list[float] = []

            for run_idx in range(self.suite["num_runs"]):
                run_label = f"run {run_idx + 1}/{self.suite['num_runs']}"
                n = len(self.requests)

                # Generate all Poisson inter-arrival times upfront
                inter_arrivals = [self._rng.expovariate(target_qps) for _ in range(n)]
                arrival_times = list(itertools.accumulate(inter_arrivals))

                t_start = loop.time()

                async def send_request(req: InferenceRequest, t_arrival: float) -> InferenceResult:
                    delay = t_arrival - (loop.time() - t_start)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return await async_inference_fn(req)

                tqdm.write(
                    f"[online] qps={target_qps} {run_label} "
                    f"— dispatching {n} requests with Poisson arrivals..."
                )

                # Run all requests concurrently; each sleeps until its arrival time
                all_results: list[InferenceResult] = list(
                    await asyncio.gather(
                        *[send_request(req, t) for req, t in zip(self.requests, arrival_times)]
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
            achieved_qps = len(all_ttfts) / (self.suite["num_runs"] * len(self.requests) / target_qps) if target_qps > 0 else 0

            ttft_p50 = float(np.percentile(all_ttfts, 50)) if all_ttfts else 0
            ttft_p90 = float(np.percentile(all_ttfts, 90)) if all_ttfts else 0
            ttft_p99 = float(np.percentile(all_ttfts, 99)) if all_ttfts else 0
            tpot_p50 = float(np.percentile(all_tpots, 50)) if all_tpots else 0
            tpot_p90 = float(np.percentile(all_tpots, 90)) if all_tpots else 0
            tpot_p99 = float(np.percentile(all_tpots, 99)) if all_tpots else 0

            # Per-run p99s, used to surface inter-run TTFT variability.
            # We compute each run's p99 independently; the scenario's overall
            # `ttft_ms_p99` (above) is computed by pooling all per-request
            # TTFTs, which is the headline number, while this CV captures
            # whether that number is reproducible across `num_runs`.
            ttft_p99_per_run = [
                float(np.percentile(run, 99)) for run in run_ttfts if run
            ]

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
                "ttft_ms_p99_reliability":
                    _reliability_block(ttft_p99_per_run, decimals=2),
            })

        self._write_samples(all_samples)
        return {"online": {
            "sla_ttft_ms": sla_ms,
            "max_valid_qps": max_valid_qps,
            "results_by_qps": results_by_qps,
        }}

    # ------------------------------------------------------------------
    # Burst scenario
    # ------------------------------------------------------------------

    async def _run_burst_async(self, async_inference_fn) -> dict:
        """
        Two-state bursty load test.

        Alternates between STEADY (burst_steady_qps) and BURST (burst_peak_qps)
        arrival rates. Each state transition fires requests at the configured QPS
        using Poisson inter-arrivals within the state window.

        Suite parameters consumed:
            burst_steady_qps        float  — QPS during steady state
            burst_peak_qps          float  — QPS during burst windows
            burst_duration_seconds  float  — duration of each burst window (seconds)
            burst_interval_seconds  float  — duration of each steady window between bursts (seconds)
            num_runs                int    — number of complete steady+burst cycles to run
            online_sla_ttft_ms      int    — SLA threshold (same as online scenario)

        Metrics returned:
            steady_ttft_p99_ms      — p99 TTFT during steady-state windows
            burst_ttft_p99_ms       — p99 TTFT during burst windows
            steady_ttft_p50_ms
            burst_ttft_p50_ms
            steady_requests_total   — total requests that completed during all steady windows
            burst_requests_total    — total requests that completed during all burst windows
            sla_met_during_burst    — bool: p99 TTFT during burst < online_sla_ttft_ms
            burst_degradation_ratio — burst_ttft_p99 / steady_ttft_p99 (higher = worse)
            results_by_cycle        — per-cycle breakdown

        A warmup phase fires `burst_warmup_requests` dummy requests
        sequentially before the first cycle. Their latencies are excluded
        from steady/burst windows so the first cycle's steady-state
        measurement is not contaminated by cold-engine TTFT spikes.
        """
        loop = asyncio.get_event_loop()
        sla_ms = self.suite["online_sla_ttft_ms"]
        steady_qps = self.suite["burst_steady_qps"]
        burst_qps = self.suite["burst_peak_qps"]
        burst_dur = self.suite["burst_duration_seconds"]
        steady_dur = self.suite["burst_interval_seconds"]
        num_runs = self.suite.get("num_runs", 3)

        await self._warmup_requests(
            async_inference_fn, self.burst_warmup_requests, "burst"
        )

        all_steady_ttfts: list[float] = []
        all_burst_ttfts: list[float] = []
        results_by_cycle = []

        all_samples: list[SampleRecord] = []

        async def fire_window(qps: float, duration_secs: float, label: str):
            """
            Fire requests at Poisson QPS for duration_secs.

            Returns
                results       : list[InferenceResult] in arrival order
                elapsed       : wall-clock seconds the window took
                arrival_times : list[float] — each request's intended arrival
                                relative to window start (parallel to results).
                                Used to compute post-burst recovery_time_seconds.
            """
            n_expected = max(1, int(qps * duration_secs * 1.5))
            requests_pool = (self.requests * ((n_expected // len(self.requests)) + 2))[:n_expected]

            inter_arrivals = [self._rng.expovariate(qps) for _ in range(n_expected)]
            arrival_times = list(itertools.accumulate(inter_arrivals))

            pairs = [(req, t) for req, t in zip(requests_pool, arrival_times) if t < duration_secs]
            if not pairs:
                return [], 0.0, []

            t_start = loop.time()

            async def send(req, t_arrival):
                delay = t_arrival - (loop.time() - t_start)
                if delay > 0:
                    await asyncio.sleep(delay)
                return await async_inference_fn(req)

            results = list(await asyncio.gather(*[send(req, t) for req, t in pairs]))
            elapsed = loop.time() - t_start
            window_arrivals = [t for (_, t) in pairs]
            return results, elapsed, window_arrivals

        # Each cycle's per-request data, captured so we can compute
        # recovery_time_seconds in a single post-processing pass after
        # all cycles complete.
        cycle_data: list[dict] = []

        for cycle_idx in range(num_runs):
            tqdm.write(f"[burst] cycle {cycle_idx + 1}/{num_runs} — steady({steady_qps} qps)...")

            steady_results, steady_elapsed, steady_arrivals = await fire_window(
                steady_qps, steady_dur, "steady"
            )
            steady_ttfts_pairs = [
                (a, r.first_token_time_ms)
                for r, a in zip(steady_results, steady_arrivals)
                if r.success and r.first_token_time_ms is not None
            ]
            steady_ttfts = [v for _, v in steady_ttfts_pairs]

            tqdm.write(f"[burst] cycle {cycle_idx + 1}/{num_runs} — burst({burst_qps} qps)...")

            burst_results, burst_elapsed, burst_arrivals = await fire_window(
                burst_qps, burst_dur, "burst"
            )
            burst_ttfts_pairs = [
                (a, r.first_token_time_ms)
                for r, a in zip(burst_results, burst_arrivals)
                if r.success and r.first_token_time_ms is not None
            ]
            burst_ttfts = [v for _, v in burst_ttfts_pairs]

            all_steady_ttfts.extend(steady_ttfts)
            all_burst_ttfts.extend(burst_ttfts)

            cycle_steady_p99 = float(np.percentile(steady_ttfts, 99)) if steady_ttfts else None
            cycle_burst_p99  = float(np.percentile(burst_ttfts, 99)) if burst_ttfts else None

            cycle_data.append({
                "steady_pairs": steady_ttfts_pairs,
                "burst_pairs":  burst_ttfts_pairs,
            })

            results_by_cycle.append({
                "cycle": cycle_idx + 1,
                "steady_requests": len(steady_ttfts),
                "burst_requests": len(burst_ttfts),
                "steady_ttft_p99_ms": round(cycle_steady_p99, 2) if cycle_steady_p99 else None,
                "burst_ttft_p99_ms":  round(cycle_burst_p99, 2) if cycle_burst_p99 else None,
            })

            tqdm.write(
                f"  [burst] cycle {cycle_idx + 1} — "
                f"steady p99={cycle_steady_p99:.0f}ms  "
                f"burst p99={cycle_burst_p99:.0f}ms"
                if cycle_steady_p99 and cycle_burst_p99 else
                f"  [burst] cycle {cycle_idx + 1} — insufficient data"
            )

        # Aggregate
        steady_p50 = float(np.percentile(all_steady_ttfts, 50)) if all_steady_ttfts else None
        steady_p99 = float(np.percentile(all_steady_ttfts, 99)) if all_steady_ttfts else None
        burst_p50  = float(np.percentile(all_burst_ttfts, 50)) if all_burst_ttfts else None
        burst_p99  = float(np.percentile(all_burst_ttfts, 99)) if all_burst_ttfts else None

        sla_met_during_burst = (burst_p99 < sla_ms) if burst_p99 is not None else False
        degradation = round(burst_p99 / steady_p99, 3) if (burst_p99 and steady_p99) else None

        # ── Recovery time after burst ─────────────────────────────────────────
        # Definition: seconds elapsed within a post-burst steady window before
        # the rolling p99 TTFT drops below 1.5× the long-term steady baseline.
        #
        # Implementation: the loop above runs `steady → burst` per cycle, so
        # cycle (i+1)'s steady window is the post-burst recovery window for
        # cycle i's burst. We compute one recovery time per cycle that has a
        # successor steady window, then emit the median (more robust than
        # mean to a single outlier cycle).
        recovery_baseline_p99 = steady_p99  # long-term, post-warmup baseline
        cycle_recovery_times: list[float] = []
        if recovery_baseline_p99 and recovery_baseline_p99 > 0:
            threshold = 1.5 * recovery_baseline_p99
            for i in range(len(cycle_data) - 1):
                post = cycle_data[i + 1]["steady_pairs"]
                if not post:
                    continue
                arrivals = [a for a, _ in post]
                ttfts    = [t for _, t in post]
                rec = _compute_recovery_time(
                    arrivals, ttfts,
                    threshold_ms=threshold,
                    window_s=min(3.0, steady_dur / 2),
                    min_samples=5,
                )
                if rec is not None:
                    cycle_recovery_times.append(rec)

        recovery_time_seconds = (
            round(float(np.median(cycle_recovery_times)), 2)
            if cycle_recovery_times else None
        )

        sla_icon = "✓" if sla_met_during_burst else "✗"
        chip_str = f"  ({self.chip_count} chips)" if self.chip_count > 1 else ""
        tqdm.write(
            f"  [burst] burst_ttft_p99={burst_p99:.0f}ms  "
            f"degradation={degradation:.2f}×  SLA {sla_icon}{chip_str}"
            if burst_p99 and degradation else
            f"  [burst] insufficient data"
        )

        self._write_samples(all_samples)
        return {"burst": {
            "sla_ttft_ms": sla_ms,
            "burst_steady_qps": steady_qps,
            "burst_peak_qps": burst_qps,
            "burst_duration_seconds": burst_dur,
            "burst_interval_seconds": steady_dur,
            "steady_requests_total": sum(c["steady_requests"] for c in results_by_cycle),
            "burst_requests_total": sum(c["burst_requests"] for c in results_by_cycle),
            "steady_ttft_p50_ms": round(steady_p50, 2) if steady_p50 else None,
            "steady_ttft_p99_ms": round(steady_p99, 2) if steady_p99 else None,
            "burst_ttft_p50_ms":  round(burst_p50, 2) if burst_p50 else None,
            "burst_ttft_p99_ms":  round(burst_p99, 2) if burst_p99 else None,
            "sla_met_during_burst": sla_met_during_burst,
            "burst_degradation_ratio": degradation,
            "recovery_time_seconds": recovery_time_seconds,
            "recovery_time_seconds_per_cycle": [
                round(v, 2) for v in cycle_recovery_times
            ] if cycle_recovery_times else [],
            "_recovery_definition": (
                "Median seconds within the post-burst steady window before "
                "rolling TTFT p99 drops below 1.5x the long-term steady baseline. "
                "Lower is better; None means it never recovered within the window."
            ),
            "results_by_cycle": results_by_cycle,
        }}

    def _run_burst(self, inference_fn: Callable) -> dict:
        """Sync entry point for burst scenario. Wraps _run_burst_async."""
        if not asyncio.iscoroutinefunction(inference_fn):
            raise TypeError(
                "_run_burst requires an async inference_fn(request: InferenceRequest) -> InferenceResult."
            )
        return asyncio.run(self._run_burst_async(inference_fn))

    # ------------------------------------------------------------------
    # Interactive scenario
    # ------------------------------------------------------------------

    async def _run_interactive_async(self, async_inference_fn) -> dict:
        """
        Send one request at a time, waiting for completion before sending the next.
        Measures single-request latency in isolation (no queueing pressure).
        Uses the same async engine as online to ensure consistent TTFT measurement.

        Per-run TTFT p99s are captured so the result emits an inter-run
        reliability block alongside the pooled metrics.
        """
        all_ttfts: list[float] = []
        all_tpots: list[float] = []
        all_samples: list[SampleRecord] = []
        run_elapsed_times: list[float] = []
        ttft_p99_per_run: list[float] = []

        total_runs = self.warmup_runs + self.suite["num_runs"]

        for run_idx in range(total_runs):
            is_warmup = run_idx < self.warmup_runs
            run_label = "warmup" if is_warmup else \
                f"run {run_idx - self.warmup_runs + 1}/{self.suite['num_runs']}"

            run_ttfts: list[float] = []
            run_tpots: list[float] = []
            t_run_start = time.perf_counter()

            for i, req in enumerate(self.requests):
                # Send one request, await completion before next — true serial
                r = await async_inference_fn(req)

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
                        f"  [interactive] {run_label} {i+1}/{len(self.requests)} "
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
                ttft_p99_per_run.append(float(np.percentile(run_ttfts, 99)))

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
            "ttft_ms_p99_reliability":
                _reliability_block(ttft_p99_per_run, decimals=2),
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
            conv_id = r.extra.get("conversation_id", str(r.request_id)) if r.extra else str(r.request_id)
            conversations[conv_id].append(r)

        # Sort turns within each conversation
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda r: r.extra.get("turn_index", 0) if r.extra else 0)

        # Run conversations sequentially
        all_results = []
        total_turns = sum(len(turns) for turns in conversations.values())
        for conv_id, turns in tqdm(conversations.items(), desc="[multiturn] conversations", unit="conv"):
            for turn in turns:
                result = inference_fn([turn])[0]
                turn_index = turn.extra.get("turn_index", 0) if turn.extra else 0
                all_results.append((conv_id, turn_index, result))

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
        Run fixed-concurrency sustained load test.
        Delegates to run_sustained() for the actual implementation.
        """
        sustained_concurrency = self.suite.get("sustained_concurrency", 8)
        duration_minutes      = self.suite.get("duration_minutes", 30)
        sample_interval_s     = self.suite.get("sample_interval_seconds", 60)
        warmup_minutes        = self.suite.get("warmup_minutes", 2.0)

        print(
            f"[sustained] Running {duration_minutes} min at concurrency {sustained_concurrency}..."
        )
        return asyncio.run(
            self.run_sustained(
                inference_fn=inference_fn,
                sustained_concurrency=sustained_concurrency,
                duration_minutes=duration_minutes,
                sample_interval_seconds=sample_interval_s,
                warmup_minutes=warmup_minutes,
            )
        )

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