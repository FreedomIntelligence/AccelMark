# AccelMark Methodology & Protocol

AccelMark is designed around a single principle: **benchmark complete serving recipes under fixed, auditable contracts — never bare chips under ad-hoc scripts.**

---

## 1. What Is a Serving Recipe?

A production LLM deployment is defined by four coupled axes:

$$\text{Recipe } h = (\text{accelerator } a,\; \text{serving engine } s,\; \text{precision } q,\; \text{parallelism } p)$$

Changing any single axis — swapping `vLLM` for `SGLang`, enabling `FP8` vs `W8A8`, or scaling from `TP=1` to `TP=8` — can shift throughput or SLA capacity by 20% to 20×. AccelMark records all four axes explicitly in every submission's `result.json` and `env_info.json`.

---

## 2. Multi-Regime Spectrum Sampling (Suites A–H)

Because inference workloads span from **memory-bandwidth bound** (low arithmetic intensity, short-prompt decode) to **compute bound** (high arithmetic intensity, long-context prefill) and **communication / capacity bound** (multi-chip TP, sparse MoE), no single scalar score preserves hardware rankings.

AccelMark samples eight distinct bottleneck regions:

| Suite | Bottleneck Region | Fixed Model | Prompt Shape (in → out) | Dataset |
|---|---|---|---|---|
| **A** | Bandwidth decode + online SLA | `Meta-Llama-3-8B-Instruct` | ~280 → ~310 tok | `sharegpt_standard_v1` |
| **B** | VRAM capacity + multi-chip TP | `Meta-Llama-3-70B-Instruct` | ~280 → ~310 tok | `sharegpt_standard_v1` |
| **C** | Native quantization vs accuracy | `Llama-3.1-8B-Instruct` | ~280 → ~310 tok | `sharegpt_standard_v1` |
| **D** | Compute-bound long prefill | `Llama-3.1-8B-Instruct` | ~28K → ≤256 tok | `sharegpt_longctx_v1` |
| **E** | Interconnect Amdahl scaling | `Meta-Llama-3-8B-Instruct` | ~280 → ~310 tok | `sharegpt_standard_v1` |
| **F** | Single-die bandwidth & tok/$ | `Qwen2.5-0.5B-Instruct` | ~95 → ~150 tok | `sharegpt_edge_v1` |
| **G** | Sparse MoE routing + agg. HBM | `Mixtral-8x7B-Instruct-v0.1` | ~280 → ~310 tok | `sharegpt_standard_v1` |
| **H** | Sliding-window KV cache (4K win) | `Mistral-7B-Instruct-v0.1` | ~10K → ≤256 tok | `sharegpt_swa_v1` |

Full suite contracts live in [`suites/suite_*/suite.json`](../suites/) and are documented in [`suites/README.md`](../suites/README.md).

---

## 3. Seven Evaluation Scenarios

Each suite selects a required **default** subset of scenarios and optional **extra** scenarios:

1. **`accuracy` (Quality Gate — Always First):** Evaluates a pinned MMLU subset on the *exact same loaded model instance* before throughput runs begin. If accuracy drops beyond `accuracy_threshold_delta` relative to the BF16 baseline (`schema/accuracy_baselines.json`), the run aborts (or is permanently flagged if `--skip-accuracy-gate` is passed).
2. **`offline` (Peak Throughput):** All prompts dispatched concurrently with no SLA constraint; sweeps suite concurrency levels and reports median aggregate `tokens/sec` across repeated trials.
3. **`online` (SLA-Bound Capacity):** Sweeps Poisson arrival rates (`online_qps_levels`) and reports `max_valid_qps` — the highest request rate that maintains **p99 TTFT ≤ 500 ms**.
4. **`interactive` (Single-Stream Latency):** Concurrency = 1; isolates minimum TTFT and TPOT without queueing contention.
5. **`sustained` (Thermal & Drift Stability):** Fixed concurrency held for 15 min (Suite F) or 30 min (datacenter suites); reports `throttle_ratio = min_window_tps / max_window_tps` and onset minute.
6. **`speculative` (Draft-Assisted Decode, extra):** Offline decode with a pinned 1B draft model; records speedup and token acceptance rate.
7. **`burst` (KV-Cache Spike Resilience, extra):** Alternates steady baseline traffic with 5× spike windows; reports burst p99 TTFT degradation ratio.

---

## 4. Reproducibility & Integrity Architecture

### Content-Hashed Runner Pinning
Every runner lives in `runners/{platform}_{name}_{hash8}/`, where `hash8` is the first 8 hex characters of `SHA-256(runner.py)` (verified by `runners/hash_runner.py`). Modifying `runner.py` requires a new versioned folder with the new hash, ensuring historical leaderboard entries always map 1-to-1 to the exact code that produced them.

### Runner Tiers
To keep apples-to-apples baselines clean while welcoming systems innovation, `meta.json` tags each runner with `runner_tier`:
- **`reference`** *(default)*: Standard, out-of-the-box serving engine integration (`vLLM`, `SGLang`, `TensorRT-LLM`, `vllm-rocm`, `vllm-ascend`, `vllm-tpu`, `mlx-lm`, `vllm-musa`, `vllm-ixuca`).
- **`optimized`**: Tuned or patched engine variants (must use lightweight `patches/*.patch` + `apply_patches.sh` — vendoring upstream trees is blocked by CI).
- **`experimental`**: Custom research forks or early previews (e.g., `turboquant`, custom SM70 attention kernels).

### Deterministic `run_id` & Automated CI Validation
Every submission PR is checked automatically by `.github/workflows/validate_pr.yml` (`runners/validate_submission.py`):
- **Schema validation:** Strict checks against `schema/result.schema.json` and `schema/env_info.schema.json`.
- **Deterministic `run_id` check:** Recomputes the SHA-256 identity from hardware, `dies_per_card`, `chip_count`, `suite_id`, `implementation_id`, `precision`, and `start_time` to prevent manual tampering.
- **Protocol integrity checks:** Rejects inverted throughput fields (`output_tps > total_tps`), zero-token positive throughput claims, missing accuracy baselines, and Suite D truncated-context runs.
- **No-bloat runner guard:** Blocks `.vendor/` directories or >100 files inside runner folders.

### Two-Tier Verification (`community` → `verified`)
- **`community`:** Any PR that passes automated CI validation lands in `results/community/` and immediately appears on the leaderboard with full artifact links.
- **`verified`:** Any independent contributor with matching hardware and runner can rerun the suite (via [Reproduction Quests](https://freedomintelligence.github.io/AccelMark/#/reproduce)). When the primary metric matches within **±5%**, the result is promoted to `results/verified/`.

---

## 5. Agent Optimization Track

AccelMark also serves as the trusted black-box evaluator for [**AccelMark-Agent**](https://freedomintelligence.github.io/AccelMark/#/agent) — evaluating coding agents that optimize real LLM serving stacks under strict anti-gaming controls:
- In-place fresh-process baselines
- Mandatory MMLU accuracy gates ($\Delta \le 0.03$)
- Artifact fingerprinting (unchanged artifacts cannot claim speedups)
- 3× independent fresh-process re-measurements
- Workload integrity audits (blocking cached responses or edits to protected evaluation configs)
