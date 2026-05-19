# AccelMark Suites

Each suite defines a fully-specified benchmark configuration for comparing
AI accelerators apples-to-apples.

---

## Suite Overview

| Suite | Model | Chips | Scenarios | Purpose |
|-------|-------|-------|-----------|---------|
| [Suite A](#suite-a) | Llama-3-8B-Instruct | 1 | offline, online (+ interactive, sustained, speculative, burst extra) | Standard single-chip inference. **Required for leaderboard entry.** |
| [Suite B](#suite-b) | Llama-3-70B-Instruct | flexible | offline, online (+ interactive, sustained, burst extra) | Large model multi-chip inference |
| [Suite C](#suite-c) | Llama-3.1-8B-Instruct | 1 | offline (+ online, sustained extra) | Quantization efficiency (BF16/FP8/W8A8/W8A16/W4A16) |
| [Suite D](#suite-d) | Llama-3.1-8B-Instruct | 1 | offline (+ interactive, online, sustained, speculative extra) | Long-context inference (~28K input tokens) |
| [Suite E](#suite-e) | Llama-3-8B-Instruct | 1×/2×/4×/8× | offline | Multi-chip scaling efficiency |
| [Suite F](#suite-f) | Qwen2.5-0.5B-Instruct | 1 (recommended) | offline, online, interactive | Consumer/edge single-GPU inference |
| [Suite G](#suite-g) | Mixtral-8x7B-Instruct-v0.1 | ≥2 (auto) | offline, online (+ interactive, sustained extra) | MoE multi-chip inference |

---

## Time Budget

The table below shows measured wall-clock times on a **single NVIDIA A100 SXM4 80GB** running with vLLM,
recorded in `meta.benchmark_elapsed_minutes` of each submitted result.json.
Times on other hardware will differ.

Times below are measured wall-clock on **NVIDIA A100-SXM4-80GB** with vLLM 0.7.3.
`benchmark_elapsed_minutes` in each result.json is the **sum of per-scenario benchmark times**
(excludes model load and sleep gaps between scenarios).

Default scenarios only:

| Suite | Scenario | rc | Formula | Wall time |
|-------|----------|----|---------|-----------|
| A | offline | 100 | 10s/run × 4 × 3 conc | ~2 min |
| A | online | 300 | Σ(elapsed × 3) / 3 QPS | ~7 min |
| A | *(interactive — extra)* | 150 | 659s/run × 3 runs | *(~33 min)* |
| A | *(sustained — extra)* | — | 32 min fixed | *(~32 min)* |
| A | *(speculative — extra)* | 100 | same as offline, draft model loaded | *(~3 min)* |
| A | *(burst — extra)* | 300 | num_runs × (burst_interval + burst_duration) | *(~18 min)* |
| | | | **Suite A default total** | **~13 min** |
| B | offline | 100 | 21s/run × 4 × 3 conc | ~4 min |
| B | online | 200 | Σ(elapsed × 3) / 4 QPS | ~23 min |
| B | *(interactive — extra)* | 50 | 780s/run × 3 runs | *(~39 min)* |
| B | *(sustained — extra)* | — | 32 min fixed | *(~32 min)* |
| B | *(burst — extra)* | 200 | num_runs × (burst_interval + burst_duration) | *(~18 min)* |
| | | | **Suite B default total** | **~26 min** |
| C | offline (×5 formats) | 100 | 4s/run × 4 × 3 conc × 5 fmt | ~22 min |
| C | *(online — extra)* | 300 | Σ(elapsed × 3) / 4 QPS × 5 fmt | *(~48 min)* |
| C | *(sustained — extra)* | — | 15 min fixed × 5 fmt | *(~76 min)* |
| | | | **Suite C default total** | **~22 min** |
| D | offline | 50 | 220s/run × 3 × 2 conc | ~22 min |
| D | *(interactive — extra)* | 100 | 1124s/run × 2 runs | *(~37 min)* |
| D | *(online — extra)* | 200 | Σ(elapsed × 2) / 3 QPS | *(~38 min)* |
| D | *(sustained — extra)* | — | 32 min fixed | *(~32 min)* |
| D | *(speculative — extra)* | 50 | same as offline, draft model loaded | *(~24 min)* |
| | | | **Suite D default total** | **~22 min** |
| E | offline (1×/2×/4×) | 150 | per-chip runs × 4 × 3 conc | ~9 min |
| | | | **Suite E default total** | **~9 min** |
| F | offline | 200 | 8s/run × 4 × 3 conc | ~2 min |
| F | online | 300 | Σ(elapsed × 3) / 2 QPS | ~3 min |
| F | interactive | 150 | 94s/run × 3 runs | ~5 min |
| F | *(sustained — extra)* | — | 15 min fixed | *(~15 min)* |
| | | | **Suite F default total** | **~10 min** |

**Total default (A–F):** ~85 min · **Total all-scenarios (A–F):** ~420 min · **Suite G default:** ~35 min (2× chip; varies with MoE routing overhead)

`rc` = request count per run. `elapsed` = `elapsed_seconds_median` from result.json (one run).
Formula for offline: `elapsed × (num_runs + 1 warmup) × num_concurrency_levels`.
Formula for online/interactive: `elapsed × num_runs` (no warmup run).
Times in italics are extra scenarios — run with `--scenario all`.

> **Sustained scenario** (extra, opt-in): adds ~30 min on datacenter suites (A–E); Suite F uses a **15-minute** profile. Run with `--scenario sustained` or `--scenario all`. Not included in default suite runs.

---

## Request Count Design

Counts are defined per suite in `suites/<suite_id>/suite.json`. Typical patterns:

```
offline:      rc=100 (A, B, C); rc=50 (D, long-context); rc=150 (E, scaling); rc=200 (F, fast model)

online:       orc=300 (A, C, F) — robust p99 at practical QPS levels
              orc=200 (B, D)    — 70B/long-context; p95 is primary tail metric

interactive:  irc=150 (A, F); irc=100 (D, long-context p90 primary); irc=50 (B, 70B decode ~15s/req)
              Serial execution — one request at a time. interactive_warmup_runs=0 for all suites.
```

> Total wall time = `elapsed_seconds_median × num_runs` (interactive/online per QPS)
> or `elapsed_seconds_median × (num_runs + warmup_runs) × num_concurrency_levels` (offline).
> `elapsed_seconds_median` in result.json is **one run**, not the full suite.

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
request_count: 100
num_runs: 3 + 1 warmup
Primary metric: throughput_tokens_per_sec (input + output tokens)
```

### Online scenario

Measures maximum sustainable QPS while meeting a latency SLA.
Requests arrive following a Poisson process (realistic service traffic).

```
online_qps_levels: [5, 25, 100]
online_sla_ttft_ms: 500    — p99 TTFT must be < 500ms to pass
online_request_count: 300
num_runs: 3 (no warmup)
Primary metric: max_valid_qps
```

`max_valid_qps` = the highest QPS level where p99 TTFT < 500ms.

### Interactive scenario

Measures single-request latency in isolation (no concurrency).

```
interactive_request_count: 150
num_runs: 3 (no warmup)
Primary metrics: ttft_ms_p50, ttft_ms_p99

> Interactive is an **extra** scenario for Suite A. Run with `--scenario all` or `--scenario interactive`.
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

**A100 reference result:** 527 tok/s sustained, throttle ratio 0.91, no throttle onset detected.

Key output metrics:
- `sustained_throughput_tokens_per_sec` — average post-warmup throughput
- `throttle_ratio` — min/max throughput ratio. 1.0 = no degradation. Lower = more throttling.
- `throttle_onset_minute` — when throughput first dropped below 90% of peak

Run explicitly with `--scenario sustained`. Not part of the default run.

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --scenario sustained
```

### Speculative decoding scenario (extra)

Runs the offline workload with a draft model loaded for speculative token generation.
The loadgen path is identical to offline — only the engine configuration changes.

```
speculative_draft_model_id:       meta-llama/Llama-3.2-1B-Instruct
speculative_draft_model_revision: 9213176726f574b556790deb65791e0c5aa438b6
speculative_num_tokens:           4      — draft tokens proposed per step
request_count:  100
num_runs: 3 + 1 warmup
Primary metric: throughput_tokens_per_sec (offline)
```

The draft model path is resolved automatically via `_resolve_model_path()` (respects
`configs/models_local.yaml`). Runners may override `get_runtime_metrics()` to expose
`acceptance_rate` and `mean_accepted_tokens` in `task.runtime_metrics`.

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --scenario speculative
```

### Burst load scenario (extra)

Alternates between a steady arrival rate and a 5× burst. Tests KV cache eviction
behavior and scheduler responsiveness under transient overload.

```
burst_steady_qps:        5    — QPS during steady windows
burst_peak_qps:         25    — QPS during burst windows
burst_duration_seconds: 30    — duration of each burst window
burst_interval_seconds: 120   — duration of each steady window between bursts
num_runs: 3 (cycles)
online_request_count: 300     — request pool size (same as online)
Primary metric: burst_degradation_ratio (burst_ttft_p99 / steady_ttft_p99)
```

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_A --scenario burst
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

### Burst load scenario (extra)

Same two-state burst pattern as Suite A, using `online_request_count` (200) as the
request pool.

```
burst_steady_qps:        5
burst_peak_qps:         25
burst_duration_seconds: 30
burst_interval_seconds: 120
```

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_B --scenario burst
```

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
| **Run time** | ~31 min on A100 (default scenarios, all 5 formats) |

### Precision formats

| Format | Checkpoint | Accuracy threshold | Notes |
|---|---|---|---|
| BF16 | `meta-llama/Llama-3.1-8B-Instruct` | ±0.03 | Baseline |
| FP8 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` | ±0.03 | Fast on H100/MI300X; emulated on A100 |
| W8A8 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` | ±0.04 | INT8 weights + activations |
| W8A16 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16` | ±0.03 | INT8 weights, FP16 activations |
| W4A16 | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` | ±0.05 | INT4 weights (AWQ), FP16 activations |

Each format runs against the same 100 prompts with concurrency levels
`[1, 4, 16, 64]` from `suite_C/suite.json` (not the same sweep as Suite A’s
`[8, 32, 128]`). Format availability depends on the runner's
`SUPPORTED_QUANTIZATION_BACKENDS` declaration — unsupported formats are
skipped automatically by matching each entry's `engine_kwargs.quantization`
against the runner's backend list.

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

Declare which quantization backends your runner's framework supports. The
strings are the engine's own backend identifiers (vLLM names shown), NOT
suite precision tags such as W8A8/FP8/W4A16:

```python
# In your runner class:
SUPPORTED_QUANTIZATION_BACKENDS = ["fp8", "compressed-tensors", "gptq_marlin"]  # vLLM full
SUPPORTED_QUANTIZATION_BACKENDS = ["compressed-tensors", "gptq_marlin"]         # No native FP8
SUPPORTED_QUANTIZATION_BACKENDS = []                                            # BF16 only
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
python run.py --runner nvidia_vllm_47f5d58e --suite suite_C
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
request_count: 50
num_runs: 2 + 1 warmup
Primary metric: throughput_tokens_per_sec
```

Long-context inference is dominated by the **prefill phase** (~28K input tokens),
which is compute-bound and tests raw FLOPS more than memory bandwidth.

**OOM on some batch sizes is expected and recorded as valid data.**
A chip that OOMs at batch_size=4 but succeeds at batch_size=1 will show
`"oom": true` for that row — useful information, not a failure.

### Interactive scenario (extra)

```
interactive_request_count: 100   — each request ~11s sequential at 28K context
num_runs: 2 (no warmup)
Primary metrics: ttft_ms_p50, ttft_ms_p90   — p90 is primary; p95 marginal at 100 reqs
```

Interactive is **extra** for Suite D: 100 reqs × 2 runs ≈ 37 min — expensive at 28K context.
Run with `--scenario all` or `--scenario interactive`.

### Online scenario (extra)

```
online_qps_levels: [0.5, 1, 2]
online_request_count: 200
```

Extra due to cost: QPS=0.5 alone takes ~13 min (rate-bound: 200 reqs / 0.5 QPS × 2 runs).

### Sustained scenario (extra)

```
sustained_concurrency: 8
duration_minutes: 30
```

**A100 reference result:** 52 tok/s sustained, throttle ratio 0.85,
throttle onset at minute 13. Low absolute throughput is expected due to
the ~28K input token prefill overhead — what matters is the throttle ratio
relative to peak.

### Speculative decoding scenario (extra)

Runs the offline workload at ~28K context with a 1B draft model. Speculative
decoding at long context is prefill-bound — acceptance rate and speedup
will differ significantly from Suite A.

```
speculative_draft_model_id:       meta-llama/Llama-3.2-1B-Instruct
speculative_draft_model_revision: 9213176726f574b556790deb65791e0c5aa438b6
speculative_num_tokens:           4
request_count: 50
num_runs: 2 + 1 warmup
```

```bash
python run.py --runner nvidia_vllm_47f5d58e --suite suite_D --scenario speculative
```

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
request_count: 150
num_runs: 3 + 1 warmup
chip_counts_required: [1, 2]
chip_counts_optional: [4, 8]
chip_counts_all: [1, 2, 4, 8]
```

### Running Suite E

```bash
# 4-chip machine
python run.py --runner nvidia_vllm_47f5d58e --suite suite_E --max-chips 4

# 8-chip machine
python run.py --runner nvidia_vllm_47f5d58e --suite suite_E --max-chips 8
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
| `sharegpt_standard_v1` | Suite A, B, C, E, G | 500 | ~280 tokens | ~310 tokens |
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
online_qps_levels:     [10, 40]       — QPS=2 excluded; rate-bound at 0.5B scale, below practical range
online_sla_ttft_ms:    500
online_request_count:  300
request_count:         200 (offline)
interactive_request_count: 150
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
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_F

# Pre-Ampere GPU (V100, T4, RTX 20xx) — required flag
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_F --enforce-eager
# Or set persistently: enforce_eager: true in
# configs/runner_configs/runner_nvidia_vllm_47f5d58e.yaml under suites.suite_F

# Single scenario
python runners/nvidia_vllm_47f5d58e/runner.py --suite suite_F --scenario offline
```

For runner-specific hardware compatibility details (including pre-Ampere guidance),
see `runners/nvidia_vllm_47f5d58e/README.md`.

### Multi-chip note

Suite F does not enforce a single-chip constraint. Developers are free to run
with TP > 1. However, Suite F is designed for single-chip consumer hardware —
multi-chip results over PCIe will show poor scaling efficiency and reflect
the interconnect bottleneck rather than GPU capability. For apples-to-apples
consumer comparisons, submit single-chip results.


## Suite G

**MoE multi-chip inference**

```
Model:     mistralai/Mixtral-8x7B-Instruct-v0.1
Chips:     auto — minimum 2×A100-80GB or 4×A100-40GB (~90GB BF16)
Precision: BF16
```

Suite G targets Mixture-of-Experts architectures. Mixtral-8x7B uses 8 experts
per layer with top-2 routing (~47B total parameters, ~13B active per token).
The model requires at least two datacenter GPUs due to its memory footprint.

`required_chips` is set to `"auto"` — runners use all available GPUs via
tensor parallelism. The leaderboard groups results by chip type and chip count
naturally, so no scaling sweep is needed (unlike Suite E).

### Scenarios

Default scenarios match Suite A's offline + online workload at MoE scale.
Optional interactive and sustained are available via `--scenario all`.

```
concurrency_levels:      [4, 16, 64]  — lower than Suite A due to larger memory footprint
online_qps_levels:       [2, 10, 40]
online_sla_ttft_ms:      500
request_count:           100 (offline)
online_request_count:    300
interactive_request_count: 150
num_runs:                3 + 1 warmup
```

### Sustained scenario (extra)

```
sustained_concurrency: 8
duration_minutes: 30
sample_interval_seconds: 60
warmup_minutes: 2
```

### Runtime metrics

Runners that expose MoE-specific statistics should override `get_runtime_metrics()`
to return expert routing data:

```python
{
    "expert_load_balance": 0.12,    # std dev of expert activation frequency
    "mean_experts_per_token": 2.0   # mean number of experts activated per token
}
```

These are recorded in `task.runtime_metrics` and displayed on the leaderboard
but do not affect ranking.

### Running Suite G

```bash
# 2-GPU machine (A100-80GB)
python run.py --runner nvidia_vllm_47f5d58e --suite suite_G

# 4-GPU machine (A100-40GB)
python run.py --runner nvidia_vllm_47f5d58e --suite suite_G
```

### Accuracy baseline

The MMLU accuracy baseline for Mixtral-8x7B is pending — `bf16_baseline_score`
is set to `null` in `schema/accuracy_baselines.json`. Run the accuracy scenario
on 2×A100-80GB BF16 to establish the baseline before accepting community
submissions.

---

## Adding a new suite

1. Open a GitHub Issue using the [**Propose a new suite**](https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=new_suite.md) template
2. Specify: model, chip count, scenarios, and rationale
3. Discuss the proposal in the issue thread — interested contributors weigh in
4. Create `suites/suite_X/suite.json` referencing a shared dataset
   (or add a new dataset to `datasets/`)
5. If custom orchestration is needed, add `suites/suite_X/suite.py`
   (see [DEVELOPMENT.md](../DEVELOPMENT.md) for the suite plugin interface)
6. Submit a reference result on at least one chip before the suite
   appears on the main leaderboard

See [DEVELOPMENT.md](../DEVELOPMENT.md) for the full guide.