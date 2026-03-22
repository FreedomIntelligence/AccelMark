# AccelMark Suites

This directory contains the suite definitions for AccelMark benchmarks.
Each suite is a named, fully-specified benchmark configuration that enables
apples-to-apples comparison across different AI accelerators.

---

## Suite Overview

| Suite | Model | Chips | Scenarios | Purpose |
|-------|-------|-------|-----------|---------|
| [Suite A](#suite-a) | Llama-3-8B-Instruct | 1 | offline, online, interactive | Standard single-chip inference. **Required for leaderboard entry.** |
| [Suite B](#suite-b) | Llama-3-70B-Instruct | 8 | offline, online | Large model multi-chip inference |
| [Suite C](#suite-c) | Llama-3-8B | 8 | training | Training throughput |
| [Suite D](#suite-d) | Llama-3.1-8B-Instruct | 1 | offline, interactive | Long-context inference (32K tokens) |

---

## Time Budget

All suites are calibrated to complete within the following time budgets
on a **single NVIDIA A100 SXM4 80GB** running with vLLM, as the reference hardware.

| Suite | Scenario | Request Count | Target Time | Total |
|-------|----------|---------------|-------------|-------|
| A | offline | 200 | ~5 min | |
| A | online | 500 | ~19 min | |
| A | interactive | 100 | ~22 min | **~46 min** |
| B | offline | 200 | ~20 min | |
| B | online | 500 | ~20 min | **~40 min** |
| C | training | 50 steps | ~20 min | **~20 min** |
| D | offline | 100 | ~30 min | |
| D | interactive | 50 | ~15 min | **~45 min** |

**Note**: If your hardware is slower than A100, expect proportionally longer
run times. The request counts and run counts are calibrated to A100 as the
reference. Faster hardware (e.g. H100) will complete in less time.

---

## Request Count Design

Each scenario uses a different request count, optimized for both
statistical reliability and time budget:

```
offline:      200 requests
              ├── Throughput is a bulk metric — 200 requests is enough
              │   for a stable median across 3 runs
              └── More requests don't improve accuracy meaningfully

online:       500 requests per QPS level
              ├── p99 latency needs enough samples to be reliable
              │   (p99 of 500 = the 495th value, statistically stable)
              └── Each QPS level runs 3 times for median

interactive:  100 requests
              ├── Serial execution (one at a time) — 100 is the limit
              │   before the time budget is exceeded
              └── p99 of 100 = the 99th value, acceptable for latency
```

---

## Suite A

**Single chip inference — minimum required for leaderboard entry**

```
Model:    meta-llama/Meta-Llama-3-8B-Instruct
Chips:    1
Precision: BF16
```

### Offline scenario

Measures maximum throughput when all requests are sent at once.
vLLM's internal scheduler handles batching.

```
batch_sizes: [8, 32, 128]   — max concurrent requests (max_num_seqs)
request_count: 200
num_runs: 3 + 1 warmup
Primary metric: throughput_tokens_per_sec (input + output tokens)
```

The three batch sizes show how throughput scales with concurrency.
Diminishing returns above a certain point indicate the GPU's decode
throughput ceiling has been reached.

### Online scenario

Measures maximum sustainable QPS while meeting a latency SLA.
Requests arrive following a Poisson process (realistic service traffic).

```
online_qps_levels: [5, 10, 25, 50, 100]
online_sla_ttft_ms: 500    — p99 TTFT must be < 500ms to pass
online_request_count: 500
num_runs: 3 (no warmup)
Primary metric: max_valid_qps
```

`max_valid_qps` = the highest QPS level where p99 TTFT < 500ms.
This directly answers: "How many users per second can this chip serve
while keeping 99% of them waiting less than 500ms for the first token?"

### Interactive scenario

Measures single-request latency in isolation.
Requests are sent one at a time (no concurrency).

```
interactive_request_count: 100
num_runs: 3 (no warmup)
Primary metrics: ttft_ms_p50, ttft_ms_p99
```

This represents the best-case latency a single user experiences
when the system is not under load.

---

## Suite B

**Multi-chip inference with large model**

```
Model:    meta-llama/Meta-Llama-3-70B-Instruct
Chips:    8
Precision: BF16
```

Same scenario design as Suite A but with a 70B model across 8 chips.
Primary purpose: measure inter-chip communication overhead and
scaling efficiency compared to Suite A.

**Scaling efficiency** can be computed as:
```
efficiency = (Suite B throughput / 8) / (Suite A throughput / 1)
```
A value of 0.8 means 80% linear scaling — 8 chips give 6.4× the
throughput of 1 chip, not 8×, due to communication overhead.

---

## Suite C

**Training throughput**

```
Model:    meta-llama/Meta-Llama-3-8B (base, not instruct)
Chips:    8
Precision: BF16
Sequence length: 4096
Global batch size: 256
```

Measures training throughput in tokens/sec and MFU
(Model FLOPS Utilization).

```
num_steps: 50 (after 5 warmup steps)
num_runs: 2
Primary metrics: tokens_per_sec, mfu
```

MFU is the most hardware-agnostic metric — it measures what fraction
of the chip's theoretical peak compute is actually being used.
A high MFU indicates an efficient software stack regardless of chip speed.

---

## Suite D

**Long-context inference**

```
Model:    meta-llama/Llama-3.1-8B-Instruct  (128K context window)
Chips:    1
Precision: BF16
Input tokens: ~32,000
Output tokens: up to 256
```

Tests the chip's ability to handle long-context workloads.
Llama-3.1-8B-Instruct is used instead of Llama-3-8B because it
natively supports 128K context without rope scaling workarounds.

```
batch_sizes: [1, 4]
request_count: 100
num_runs: 2 + 1 warmup
Primary metric: throughput_tokens_per_sec
```

**OOM on some batch sizes is expected and recorded as valid data.**
A chip that OOMs at batch_size=4 but succeeds at batch_size=1 will
show `"oom": true` for batch_size=4 — this is useful information,
not a failure of the benchmark.

Long-context inference is dominated by the **prefill phase** (processing
the 32K input tokens), which is compute-bound and tests the chip's
raw FLOPS more than memory bandwidth.

---

## Adding a New Suite

To propose a new suite:

1. Open a GitHub Issue using the "Request new suite" template
2. Specify: model, chip count, scenarios, and rationale
3. Maintainers review and add to the roadmap
4. Internal data pipeline generates `requests.jsonl`
5. Suite is published with at least one reference result

New suites require a reference result on at least one chip before
they appear on the main leaderboard.

---

## requests.jsonl Format

Each suite ships with a fixed `requests.jsonl` file.
All platforms run against the exact same prompts to ensure comparability.

```jsonl
{"request_id": 0, "prompt": "...", "input_tokens_approx": 245, "prompt_type": "conversational"}
{"request_id": 1, "prompt": "...", "input_tokens_approx": 412, "prompt_type": "code_generation"}
```

**These files are generated by the internal data pipeline and must not
be edited manually.** Changing the prompts would invalidate comparisons
with existing results.

Source dataset: `shibing624/sharegpt_gpt4`

Prompt type distribution (Suite A/B):
```
conversational:  40%  — everyday dialogue, advice, Q&A
summarization:   30%  — long input, short output
code_generation: 20%  — write/fix/explain code
reasoning:       10%  — step-by-step analysis, math
```
