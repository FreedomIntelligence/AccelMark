# Workload Suites Reference (A–H)

AccelMark uses **spectrum sampling** across the Roofline model so no single opaque composite score hides real-world hardware reversals.

![AccelMark 8-Suite Roofline Spectrum](../assets/roofline-suites-light.svg#only-light)
![AccelMark 8-Suite Roofline Spectrum](../assets/roofline-suites-dark.svg#only-dark)

## Suite Overview

| Suite | Model | Precision | Chips | Input → Output | Default Scenarios | Primary Metric | Ref Time (A100) |
|---|---|---|---|---|---|---|---|
| **A** · Single-Chip Chat | `Meta-Llama-3-8B` | BF16 | 1 | ~280 → ~310 | accuracy, offline, online | Offline tok/s & Max QPS | ~11 min |
| **B** · 70B Multi-Chip TP | `Meta-Llama-3-70B` | BF16 | 4 / 8 | ~280 → ~310 | accuracy, offline, online | Offline tok/s | ~20 min |
| **C** · Quantization | `Llama-3.1-8B` | BF16 + FP8/W8A8/W8A16/W4A16 | 1 | ~280 → ~310 | accuracy, offline (×5) | Quality efficiency | ~22 min |
| **D** · 28K Long Context | `Llama-3.1-8B` | BF16 | 1 | ~28K → ≤256 | accuracy, offline | Offline tok/s | ~22 min |
| **E** · Multi-Chip Scaling | `Meta-Llama-3-8B` | BF16 | 1 / 2 / 4 / 8 | ~280 → ~310 | offline sweep | 2× / 4× scaling eff. | ~9 min |
| **F** · Edge / Consumer | `Qwen2.5-0.5B` | BF16 / FP16 | 1 | ~95 → ~150 | accuracy, offline, interactive | Offline tok/s & tok/$/hr | ~10 min |
| **G** · Sparse MoE | `Mixtral-8x7B` | BF16 | ≥2 | ~280 → ~310 | accuracy, offline, sustained | Sustained / offline tok/s | ~35 min |
| **H** · Sliding Window | `Mistral-7B` | BF16 | 1 | ~10K (4K win) → 256 | accuracy, offline, online | Offline tok/s | ~14 min |

## Seven Evaluation Scenarios

- **`accuracy` (mandatory gate)**: 100-question MMLU subset on the exact loaded model instance (`Δ ≤ 0.03`).
- **`offline`**: Unbounded batch arrival measuring peak throughput (`tokens/sec`).
- **`online`**: Poisson arrival sweep reporting highest QPS sustaining **`p99 TTFT ≤ 500 ms`**.
- **`interactive`**: Single-stream concurrency=1 TTFT latency (`ms`).
- **`sustained`** _(extra)_: 15–30 min constant load measuring thermal/fragmentation throttle ratio.
- **`burst`** _(extra)_: Alternating 5× arrival pulses measuring KV admission resilience.
- **`speculative`** _(extra)_: Draft-assisted decoding speedup & acceptance rate.

Full JSON definitions live in [`suites/suite_*/suite.json`](https://github.com/FreedomIntelligence/AccelMark/tree/main/suites).
