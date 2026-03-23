# AccelMark Suites

Each suite defines a fully-specified benchmark configuration for comparing
AI accelerators apples-to-apples.

---

## Suite Overview

| Suite | Model | Chips | Scenarios | Purpose |
|-------|-------|-------|-----------|---------|
| [Suite A](#suite-a) | Llama-3-8B-Instruct | 1 | offline, online, interactive | Standard single-chip inference. **Required for leaderboard entry.** |
| [Suite B](#suite-b) | Llama-3-70B-Instruct | 8 | offline, online | Large model multi-chip inference |
| [Suite C](#suite-c) | Llama-3-8B-Instruct | 1 | offline | Quantization efficiency (BF16/INT8/INT4) — **not yet implemented** |
| [Suite D](#suite-d) | Llama-3.1-8B-Instruct | 1 | offline, interactive | Long-context inference (32K tokens) |
| [Suite E](#suite-e) | Llama-3-8B-Instruct | 1×/2×/4×/8× | offline | Multi-chip scaling efficiency |

---

## Time Budget

The table below shows measured wall-clock times on a **single NVIDIA A100 SXM4 80GB** running with vLLM,
recorded in `meta.benchmark_elapsed_minutes` of each submitted result.json.
Times on other hardware will differ.

| Suite | Scenario | Request Count | Measured Time¹ | Total |
|-------|----------|---------------|----------------|-------|
| A | offline | 200 | ~5 min | |
| A | online | 500 | ~22 min | |
| A | interactive | 100 | ~23 min | **~50 min** |
| B | offline | 200 | ~9 min | |
| B | online | 500 | ~60 min | **~69 min** |
| C | offline (BF16) | 200 | — | |
| C | offline (INT8) | 200 | — | |
| C | offline (INT4) | 200 | — | **not yet implemented** |
| D | offline | 100 | ~47 min | |
| D | interactive | 50 | ~13 min | **~60 min** |
| E | offline (1×) | 500 | ~10 min | |
| E | offline (2×) | 500 | ~8 min | |
| E | offline (4×) | 500 | ~6 min | **~24 min** |

¹ Measured on NVIDIA A100 SXM4 80 GB with vLLM as reference. Recorded in `meta.benchmark_elapsed_minutes` of each result.json.

---

## Request Count Design

Each scenario uses a different request count, optimized for statistical
reliability within the time budget:

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

**Single-chip inference — minimum required for leaderboard entry**

```
Model:     meta-llama/Meta-Llama-3-8B-Instruct
Chips:     1
Precision: BF16
```

### Offline scenario

Measures maximum throughput when all requests are sent at once.
vLLM's internal scheduler handles batching.

```
concurrency_levels: [8, 32, 128]   — client-side concurrency (requests sent simultaneously)
request_count: 200
num_runs: 3 + 1 warmup
Primary metric: throughput_tokens_per_sec (input + output tokens)
```

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

### Interactive scenario

Measures single-request latency in isolation (no concurrency).

```
interactive_request_count: 100
num_runs: 3 (no warmup)
Primary metrics: ttft_ms_p50, ttft_ms_p99
```

### Accuracy scenario

Runs 100 MMLU multiple-choice questions through the same model and framework
as the benchmark. Runs automatically as the first step of `--scenario all`.

```
accuracy_questions: 100
accuracy_threshold_delta: 0.03  — valid if score ≥ BF16_baseline − 0.03
Primary metric: subset_score (fraction correct)
```

---

## Suite B

**Large model multi-chip inference**

```
Model:     meta-llama/Meta-Llama-3-70B-Instruct
Chips:     4+ (tensor parallelism — use however many your hardware needs)
Precision: BF16
```

Same scenario structure as Suite A (offline + online) at 70B scale.
The chip count is flexible — use however many chips your hardware needs.

**Scaling efficiency** vs Suite A (reference: N chips used for Suite B):
```
efficiency = (Suite B throughput / N) / (Suite A throughput / 1)
```
A value of 0.8 with N=4 means 4 chips deliver 3.2× the single-chip throughput.

---

## Suite C

**Quantization efficiency — speed vs quality tradeoff**

> **Status: Not yet implemented.** Suite C is defined and documented but the benchmark runner support (`_run_suite_c()`) is not finished. Do not submit Suite C results yet.
>
> *"How much faster does quantization make this chip, and what quality is lost?"*

Suite C runs the same workload as Suite A (Llama-3-8B, offline) at three
precision levels. Holding everything else constant isolates the effect of
quantization on throughput and output quality.

| | |
|---|---|
| **Model** | Llama-3-8B-Instruct |
| **Chips** | 1 |
| **Scenarios** | offline |
| **Precision levels** | BF16 · INT8 (W8A8) · INT4 (W4A16 AWQ) |
| **Primary metrics** | throughput per precision · speedup_vs_bf16 · quality_efficiency |
| **Run time** | ~15 min on A100 (5 min × 3 precision levels) |

```
concurrency_levels: [8, 32, 128]
request_count: 200
num_runs: 3 + 1 warmup
accuracy_threshold_delta: 0.05  — wider than other suites; quantization reduces accuracy
```

### Metrics

**speedup_vs_bf16**: throughput ratio vs BF16 baseline. 1.5 = 50% more throughput.

**quality_efficiency**: `throughput × accuracy_score`. Higher = better overall tradeoff.

```
Example result on A100:
Precision  Throughput    Accuracy  Speedup   Quality Eff
BF16       6,123 tok/s   0.62      1.00×     3,796
INT8       9,200 tok/s   0.60      1.50×     5,520  ← good tradeoff
INT4       13,500 tok/s  0.54      2.20×     7,290  ← aggressive
```

### Running Suite C

```bash
python run.py --runner nvidia_vllm_e0859b3c --suite suite_C
```

Runs BF16 → INT8 → INT4 in sequence. Each precision level is a separate
subprocess for clean GPU state. BF16 is required; INT8 and INT4 are optional.

### Requirements

```bash
# INT8: install bitsandbytes
pip install bitsandbytes

# INT4: requires AWQ quantized model weights
# e.g. meta-llama/Meta-Llama-3-8B-Instruct-AWQ
```

---

## Suite D

**Long-context inference**

```
Model:         meta-llama/Llama-3.1-8B-Instruct  (128K context window)
Chips:         1
Precision:     BF16
Input tokens:  ~32,000
Output tokens: up to 256
```

Tests the chip's ability to handle long-context workloads. Llama-3.1-8B
is used (not 3.0) because it natively supports 128K context.

```
concurrency_levels: [1, 4]
request_count: 100
num_runs: 2 + 1 warmup
Primary metric: throughput_tokens_per_sec
```

Long-context inference is dominated by the **prefill phase** (32K input tokens),
which is compute-bound and tests raw FLOPS more than memory bandwidth.

**OOM on some batch sizes is expected and recorded as valid data.**
A chip that OOMs at batch_size=4 but succeeds at batch_size=1 will show
`"oom": true` for that row — useful information, not a failure.

---

## Suite E

**Multi-chip scaling efficiency**

```
Model:    meta-llama/Meta-Llama-3-8B-Instruct
Chips:    1×, 2× required; 4×, 8× optional
Scenario: offline only
```

Holds the model constant (8B, fits on any single chip) and varies only
chip count. This isolates the scaling dimension from chip speed.

**Scaling efficiency** is the primary metric:

```
scaling_efficiency = N_chip_throughput / (1_chip_throughput × N)

1.00 = perfect linear scaling
0.85 = 4 chips give 3.4× speedup (15% lost to communication)
0.50 = 4 chips give only 2× speedup (poor interconnect)
```

```
concurrency_levels: [8, 32, 128]
request_count: 500
num_runs: 3 + 1 warmup
chip_counts_required: [1, 2]
chip_counts_all: [1, 2, 4, 8]
```

### Running Suite E

```bash
# 4-chip machine
python run.py --runner nvidia_vllm_e0859b3c --suite suite_E --max-chips 4

# 8-chip machine
python run.py --runner nvidia_vllm_e0859b3c --suite suite_E --max-chips 8
```

**Minimum requirement:** both 1× and 2× must succeed for the submission
to pass validation.

---

## requests.jsonl Format

Each suite ships with a fixed `requests.jsonl` file. All platforms run
against the exact same prompts to ensure comparability.

```jsonl
{"request_id": 0, "prompt": "...", "input_tokens_approx": 245, "prompt_type": "conversational"}
{"request_id": 1, "prompt": "...", "input_tokens_approx": 412, "prompt_type": "code_generation"}
```

**These files must not be edited manually.** Changing the prompts invalidates
comparisons with existing results.

Source dataset: `shibing624/sharegpt_gpt4`

Prompt type distribution (Suite A/B):
```
conversational:  40%  — everyday dialogue, advice, Q&A
summarization:   30%  — long input, short output
code_generation: 20%  — write/fix/explain code
reasoning:       10%  — step-by-step analysis, math
```

---

## Adding a New Suite

1. Open a GitHub Issue using the "Request new suite" template
2. Specify: model, chip count, scenarios, and rationale
3. Maintainers review and add to the roadmap
4. Internal data pipeline generates `requests.jsonl`
5. Suite is published with at least one reference result

New suites require a reference result on at least one chip before
they appear on the main leaderboard.
