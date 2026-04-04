# AccelMark Suites

Each suite defines a fully-specified benchmark configuration for comparing
AI accelerators apples-to-apples.

---

## Suite Overview

| Suite | Model | Chips | Scenarios | Purpose |
|-------|-------|-------|-----------|---------|
| [Suite A](#suite-a) | Llama-3-8B-Instruct | 1 | offline, online, interactive | Standard single-chip inference. **Required for leaderboard entry.** |
| [Suite B](#suite-b) | Llama-3-70B-Instruct | flexible | offline, online (+ interactive, sustained extra) | Large model multi-chip inference |
| [Suite C](#suite-c) | Llama-3.1-8B-Instruct | 1 | offline (+ online, sustained extra) | Quantization efficiency (BF16/FP8/W8A8/W8A16/W4A16) |
| [Suite D](#suite-d) | Llama-3.1-8B-Instruct | 1 | offline, interactive | Long-context inference (~28K input tokens) |
| [Suite E](#suite-e) | Llama-3-8B-Instruct | 1×/2×/4×/8× | offline | Multi-chip scaling efficiency |
| [Suite F](#suite-f) | Qwen2.5-0.5B-Instruct | 1 (recommended) | offline, online, interactive | Consumer/edge single-GPU inference |

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
| C | offline (BF16) | 200 | ~9 min | |
| C | offline (FP8) | 200 | ~9 min | |
| C | offline (W8A8) | 200 | ~8 min | |
| C | offline (W8A16) | 200 | ~10 min | |
| C | offline (W4A16) | 200 | ~9 min | **~45 min** |
| D | offline | 100 | ~47 min | |
| D | interactive | 50 | ~13 min | **~60 min** |
| E | offline (1×) | 500 | ~10 min | |
| E | offline (2×) | 500 | ~8 min | |
| E | offline (4×) | 500 | ~6 min | **~24 min** |
| F | offline | 200 | ~1 min | |
| F | online | 500 | ~18 min | |
| F | interactive | 100 | ~5 min | **~27 min** |

¹ Measured on NVIDIA A100 SXM4 80 GB with vLLM as reference. Recorded in `meta.benchmark_elapsed_minutes` of each result.json.
² Suite F measured on NVIDIA A100 SXM4 40 GB with vLLM 0.6.6.post1. Online is dominated by QPS=2 (low-rate Poisson arrival with 500 requests takes ~4 min per run × 3 runs).

> **Sustained scenario** (extra, opt-in): typically adds ~30 minutes wall time on datacenter suites (A–E); Suite F uses a **15-minute** sustained profile. Run with `--scenario sustained`. Not included in default suite runs.

---

## Request Count Design

Counts are defined per suite in `suites/<suite_id>/suite.json`. Typical patterns:

```
offline:      Often 200 requests (Suites A, B, C, F) — throughput is a bulk metric
              Suite D: 100 (long context). Suite E: 500 (scaling comparability).

online:       Often 500 per QPS level (A, B, C, F) — enough samples for stable p99
              Suite D: 50 (long context).

interactive:  Often 100 (A, F); Suite D: 50. Serial execution (one at a time).
```

Always use the suite’s `request_count`, `online_request_count`, and
`interactive_request_count` fields as the source of truth.

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
online_qps_levels: [5, 25, 100]
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
as the benchmark. Runs automatically as the first step when running a suite.

```
accuracy_questions: 100
accuracy_threshold_delta: 0.10  — valid if score ≥ baseline − 0.10 (see suite.json)
Primary metric: subset_score (fraction correct)
```

Suite C uses per-format `accuracy_thresholds` in `suite_C/suite.json` instead of a single delta.

### Sustained scenario (extra)

30-minute fixed-concurrency load test. Detects KV cache exhaustion, thermal
throttling, and memory fragmentation that point-in-time benchmarks miss.

```
sustained_concurrency: 8    — requests kept in-flight simultaneously
duration_minutes: 30
sample_interval_seconds: 60  — throughput snapshot every minute
warmup_minutes: 2
```

**A100 reference result:** 504 tok/s sustained, throttle ratio 0.87,
throttle onset at minute 12.

Key output metrics:
- `sustained_throughput_tokens_per_sec` — average post-warmup throughput
- `throttle_ratio` — min/max throughput ratio. 1.0 = no degradation. Lower = more throttling.
- `throttle_onset_minute` — when throughput first dropped below 90% of peak

Run explicitly with `--scenario sustained`. Not part of the default run.

```bash
python run.py --runner nvidia_vllm_2b3890cf --suite suite_A --scenario sustained
```

---

## Suite B

**Large model multi-chip inference**

```
Model:     meta-llama/Meta-Llama-3-70B-Instruct
Chips:     flexible — use however many your hardware requires
Precision: BF16
```

Default scenarios match Suite A’s **offline + online** workload at 70B scale.
Optional **interactive** and **sustained** scenarios are defined in `suite_B/suite.json`
(`scenarios.extra`). The chip count is flexible — use however many chips your hardware needs.

**Scaling efficiency** vs Suite A (reference: N chips used for Suite B):
```
efficiency = (Suite B throughput / N) / (Suite A throughput / 1)
```
A value of 0.8 with N=4 means 4 chips deliver 3.2× the single-chip throughput.

### Sustained scenario (extra)

```
sustained_concurrency: 4    — lower than Suite A due to higher memory pressure per request
duration_minutes: 30
```

Run with `--scenario sustained`. Concurrency set to 4 because the 70B model
occupies most GPU memory, leaving less room for KV cache than the 8B model.

---

## Suite C

**Quantization efficiency — speed vs quality tradeoff**

> *"How much faster does quantization make this chip, and what quality is lost?"*

Suite C runs a similar offline workload to Suite A (same dataset, `output_tokens_max` 512)
at five precision formats using
fixed pre-quantized HuggingFace checkpoints. All formats use the same
Llama-3.1-8B base model — accuracy differences reflect quantization only,
not model version differences.

| | |
|---|---|
| **Base model** | `meta-llama/Llama-3.1-8B-Instruct` |
| **Chips** | 1 |
| **Default scenarios** | accuracy, offline |
| **Extra scenarios** | online, sustained |
| **Primary metric** | `quality_efficiency` (best across all formats) |
| **Run time** | ~45 min on A100 (default scenarios, all 5 formats) |

### Precision formats

| Format | Checkpoint | Accuracy threshold | Notes |
|---|---|---|---|
| BF16 | `meta-llama/Llama-3.1-8B-Instruct` | ±0.03 | Baseline |
| FP8 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` | ±0.03 | Fast on H100/MI300X; emulated on A100 |
| W8A8 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` | ±0.04 | INT8 weights + activations |
| W8A16 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16` | ±0.03 | INT8 weights, FP16 activations |
| W4A16 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` | ±0.05 | INT4 weights (AWQ), FP16 activations |

Each format runs against the same 200 prompts with concurrency levels
`[1, 4, 16, 64]` from `suite_C/suite.json` (not the same sweep as Suite A’s
`[8, 32, 128]`). Format availability depends on the runner's `SUPPORTED_QUANTIZATIONS`
declaration — unsupported formats are skipped automatically.

### Metrics

**speedup_vs_bf16**: throughput ratio relative to BF16 baseline.
`1.20` = 20% more throughput than BF16.

**quality_efficiency**: `throughput × accuracy_score`. Rewards both
speed and accuracy simultaneously. The leaderboard primary metric is the
best quality_efficiency across all evaluated formats.

**accuracy per format**: each format has its own accuracy baseline and
threshold. Accuracy below threshold is flagged but does not block the run.

### Example result (A100, vLLM)

```
Format  Throughput    Accuracy  Speedup   Quality Eff   Compute dtype
BF16    5,336 tok/s   0.57      1.000×    3,042         bfloat16
FP8     5,179 tok/s   0.57      0.971×    2,952         bfloat16 (emulated)
W8A8    6,399 tok/s   0.59      1.199×    3,776         bfloat16
W8A16   4,939 tok/s   0.57      0.925×    2,815         bfloat16
W4A16   5,095 tok/s   0.57      0.955×    2,904         float16
```

W8A8 wins on A100 because it uses INT8 tensor cores. FP8 shows no speedup
because A100 lacks native FP8 hardware — compute falls back to BF16.
On H100, FP8 would show ~1.5-1.8× speedup.

### Runner requirements

Declare which formats your runner supports:

```python
# In your runner class:
SUPPORTED_QUANTIZATIONS = ["fp8", "w8a8", "w8a16", "w4a16"]  # H100
SUPPORTED_QUANTIZATIONS = ["w8a8", "w8a16", "w4a16"]          # A100 (no native FP8)
SUPPORTED_QUANTIZATIONS = []                                   # BF16 only
```

Each format's checkpoint must be available locally. Add to
`configs/models_local.yaml`:

```yaml
models:
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8":
    local_path: /data/models/llama31-8b-fp8
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8":
    local_path: /data/models/llama31-8b-w8a8
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16":
    local_path: /data/models/llama31-8b-w8a16
  "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16":
    local_path: /data/models/llama31-8b-w4a16
```

### Running Suite C

```bash
python run.py --runner nvidia_vllm_2b3890cf --suite suite_C
```

Runs all supported formats in sequence. Each format is a separate subprocess
for clean GPU state. BF16 always runs first as the baseline.

---

## Suite D

**Long-context inference**

```
Model:         meta-llama/Llama-3.1-8B-Instruct  (128K native context window)
Chips:         1
Precision:     BF16
max_model_len: 30208   (KV budget for this benchmark)
Input tokens:  p50 ~28,650 (dataset sharegpt_longctx_v1; p99 ~29,932)
Output tokens: up to 256
```

Tests the chip's ability to handle long-context workloads. Llama-3.1-8B
is used (not 3.0) because it natively supports 128K context. The suite caps
`max_model_len` at 30,208 and uses prompts near **~28K** tokens (not full 32K)
so runs remain reproducible and within practical memory limits on common GPUs.

```
concurrency_levels: [1, 4]
request_count: 100
num_runs: 2 + 1 warmup
Primary metric: throughput_tokens_per_sec
```

Long-context inference is dominated by the **prefill phase** (~28K input tokens),
which is compute-bound and tests raw FLOPS more than memory bandwidth.

**OOM on some batch sizes is expected and recorded as valid data.**
A chip that OOMs at batch_size=4 but succeeds at batch_size=1 will show
`"oom": true` for that row — useful information, not a failure.

### Sustained scenario (extra)

```
sustained_concurrency: 8
duration_minutes: 30
```

**A100 reference result:** 52 tok/s sustained, throttle ratio 0.85,
throttle onset at minute 13. Low absolute throughput is expected due to
the ~28K input token prefill overhead — what matters is the throttle ratio
relative to peak.

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
chip_counts_optional: [4, 8]
chip_counts_all: [1, 2, 4, 8]
```

### Running Suite E

```bash
# 4-chip machine
python run.py --runner nvidia_vllm_2b3890cf --suite suite_E --max-chips 4

# 8-chip machine
python run.py --runner nvidia_vllm_2b3890cf --suite suite_E --max-chips 8
```

**Minimum requirement:** both 1× and 2× must succeed for the submission
to pass validation.

---

## Datasets

Request datasets are stored in `datasets/` and shared across suites.
Each dataset is versioned and immutable — changing prompts creates a new
version rather than modifying the existing one.

| Dataset | Used by | Prompts | Input p50 | Output p50 |
|---|---|---|---|---|
| `sharegpt_standard_v1` | Suite A, B, C, E | 500 | ~280 tokens | ~310 tokens |
| `sharegpt_longctx_v1` | Suite D | 200 | p50 ~28,650 tokens | up to 256 (suite cap) |
| `sharegpt_edge_v1` | Suite F | 500 | ~95 tokens | ~150 tokens |

Suite JSON files reference datasets by name:
```json
"dataset": "sharegpt_standard_v1"
```

### Request format

Each line in `requests.jsonl`:

```jsonl
{"request_id": 0, "prompt": "...", "input_tokens": 245, "conversation_id": "sg_00001", "turn_index": 0, "prompt_type": "conversational"}
```

**These files must not be edited manually.** Changing prompts invalidates
comparisons with existing results.

Prompt type distribution (sharegpt_standard_v1):
```
conversational:  40%  — everyday dialogue, advice, Q&A
summarization:   30%  — long input, short output
code_generation: 20%  — write/fix/explain code
reasoning:       10%  — step-by-step analysis, math
```

---

## Suite F

**Consumer/edge single-GPU inference**

```
Model:     Qwen/Qwen2.5-0.5B-Instruct
Chips:     1 (recommended — no hard constraint)
Precision: BF16 (auto-fallback to FP16 on pre-Ampere)
```

Suite F is designed for consumer and edge GPUs: RTX 3090, RTX 4090, A10, L4,
and pre-Ampere hardware including V100 and T4. The model (0.5B parameters,
~1 GB in FP16) fits comfortably on any GPU with 4+ GB VRAM.

**Precision handling**
`precision_required: BF16` with `allowed_precisions: [FP16, BF16]` (order matches
`suite_F/suite.json`). Ampere+ GPUs
(RTX 3090/4090, A100, H100) use BF16 natively. Pre-Ampere GPUs (V100, T4, RTX 20xx)
automatically fall back to FP16 via `allowed_precisions` — no warning, no flag, since
FP16 is an explicitly accepted precision for this suite. Results are labeled with
the actual precision used.

**Why Qwen2.5-0.5B?**
- Smallest practical instruction-tuned model with full vLLM support since v0.4.0
- Fits in 4 GB VRAM in FP16 — accessible to the widest range of consumer hardware
- Stable `Qwen2ForCausalLM` architecture avoids newer-vLLM-only features
- Apache 2.0 licensed

**Accuracy note:** Absolute MMLU score for a 0.5B model is ~0.35–0.40, well below
larger models. The accuracy gate exists to detect broken quantization or misconfigured
precision — not to evaluate model quality. The threshold (±0.10) is intentionally
wider than datacenter suites.

### Scenarios

Same structure as Suite A — offline, online, and interactive. Concurrency levels
are smaller (4/16/64 vs 8/32/128) because a 0.5B model saturates consumer GPUs
at lower concurrency.

```
concurrency_levels:    [4, 16, 64]
online_qps_levels:     [2, 10, 40]
online_sla_ttft_ms:    500
request_count:         200 (offline), 500 (online), 100 (interactive)
num_runs:              3 + 1 warmup
```

### Sustained scenario (extra)

Shorter wall time than datacenter suites so consumer GPUs stay within a practical budget.

```
sustained_concurrency: 32
duration_minutes: 15
sample_interval_seconds: 60
warmup_minutes: 1
```

Run with `--scenario sustained`. Not part of the default run.

### Running Suite F

```bash
# Standard run (Ampere+)
python runners/nvidia_vllm_2b3890cf/runner.py --suite suite_F

# Pre-Ampere GPU (V100, T4, RTX 20xx) — required flag
python runners/nvidia_vllm_2b3890cf/runner.py --suite suite_F --enforce-eager
# Or set persistently: enforce_eager: true in
# configs/runner_configs/runner_nvidia_vllm_2b3890cf.yaml under suites.suite_F

# Single scenario
python runners/nvidia_vllm_2b3890cf/runner.py --suite suite_F --scenario offline
```

For runner-specific hardware compatibility details (including pre-Ampere guidance),
see `runners/nvidia_vllm_2b3890cf/README.md`.

### Multi-chip note

Suite F does not enforce a single-chip constraint. Developers are free to run
with TP > 1. However, Suite F is designed for single-chip consumer hardware —
multi-chip results over PCIe will show poor scaling efficiency and reflect
the interconnect bottleneck rather than GPU capability. For apples-to-apples
consumer comparisons, submit single-chip results.


## Adding a new suite

1. Open a GitHub Issue using the "Request new suite" template
2. Specify: model, chip count, scenarios, and rationale
3. Maintainers review and add to the roadmap
4. Create `suites/suite_X/suite.json` referencing a shared dataset
   (or add a new dataset to `datasets/`)
5. If custom orchestration is needed, add `suites/suite_X/suite.py`
   (see [DEVELOPMENT.md](../DEVELOPMENT.md) for the suite plugin interface)
6. Submit a reference result on at least one chip before the suite
   appears on the main leaderboard

See [DEVELOPMENT.md](../DEVELOPMENT.md) for the full guide.