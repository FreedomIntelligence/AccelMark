# Key Empirical Findings

> **Core thesis:** *There is no single fastest chip — only the best serving recipe `(accelerator, framework, precision, parallelism)` for a given workload regime.*

This document summarizes the primary empirical takeaways surfaced by AccelMark across its live leaderboard and paper evaluations. Every number below traces back to a pinned runner folder and public `result.json` artifact under [`results/`](../results/).

---

## 1. Regime Reversal: Short-Chat Winner Loses Long-Context Prefill

**Type I Roofline Reversal (Suite A vs Suite D, same `vLLM` reference stack):**

| Accelerator | Suite A (Short Chat · ~280 tok in) | Suite D (Long Context · ~28K tok in) | Suite A Value (`tok/$/hr`) |
|---|---:|---:|---:|
| **NVIDIA GeForce RTX 5090** | **3,488 tok/s** (+29%) | 54.3 tok/s (−23%) | **4,650** |
| **NVIDIA A100-SXM4-80GB** | 2,701 tok/s | **70.2 tok/s** (+29%) | 1,800 |

- **Why it happens:** Suite A (Llama-3-8B, ~280-token prompts) is **memory-bandwidth bound**, where the RTX 5090's high GDDR7 bandwidth shines. Suite D (~28K-token prefill) pushes arithmetic intensity past the roofline knee into a **compute-bound** regime, where datacenter tensor throughput and sustained prefill execution let A100 pull ahead despite the 5090's peak spec advantage.
- **Practical lesson:** Never procure hardware for RAG or long-document summarization using short-prompt decode numbers.

---

## 2. Offline Throughput Winner Fails the Production SLA Tier

**Type II Capacity & Scheduling Reversal (Suite A, verified `vLLM 0.7.3` reference stack):**

| Accelerator | Offline Throughput | Online Max Valid QPS (p99 TTFT ≤ 500 ms) |
|---|---:|---:|
| **NVIDIA H200 (141 GB)** | **5,731 tok/s** (#1 verified) | 5 QPS |
| **NVIDIA H100 80GB HBM3** | 5,128 tok/s | **25 QPS (5× higher)** |
| **NVIDIA RTX 5090 (32 GB)** | 3,488 tok/s | **0 QPS (SLA fail)** |

- **Why it happens:** Two distinct effects decouple offline throughput from production SLA capacity:
  1. **KV-cache capacity cliff:** Under Poisson arrivals, 24–32 GB consumer GPUs run out of KV headroom and queue requests past the 500 ms p99 TTFT budget even when their offline tok/s looks strong.
  2. **Scheduler batching effect:** H200's faster decode lets continuous-batching schedulers pack larger batches per step, pushing prefill/TTFT tail latency past 500 ms earlier than H100 under the same default vLLM scheduler policy.
- **Practical lesson:** Always inspect `online_max_qps` and `ttft_p99` alongside offline throughput for interactive services.

---

## 3. The Software Stack Can Move a Recipe More Than the Chip

**Type II Software-Stack Effect (same physical GPU, swapping only the runner):**

| Hardware | Suite | Runner A | Runner B | Measured Delta |
|---|---|---|---|---|
| **NVIDIA H20-3e** | Suite A (Offline) | vLLM 0.7.3 (`2,298 tok/s`) | SGLang 0.5.6 (`4,342 tok/s`) | **+89% throughput** |
| **NVIDIA A100-SXM4-80GB** | Suite A (Online SLA) | vLLM 0.7.3 (`5 QPS`) | SGLang 0.4.10 / 0.5.6 (`100 QPS`) | **20× SLA capacity** |
| **NVIDIA A800-SXM4-80GB** | Suite H (10K SWA) | vLLM 0.7.3 (`2,069 tok/s`) | TensorRT-LLM 0.21 (`2,316 tok/s`) | **+12% throughput** |

- **Why it matters:** A benchmark that locks one serving engine measures a *hardware-software pair*, not raw silicon. AccelMark pins every runner by content hash (`{platform}_{name}_{hash8}`) and surfaces framework filters so readers can separate silicon limits from stack maturity.
- **Check locally:** Run `python tools/advisor.py --engine-compare "A100"` to compare frameworks on any chip with multi-engine submissions.

---

## 4. Tensor-Parallel Scaling Inverts Interconnect Generation

**Type II Interconnect Amdahl Effect (Suite E · Llama-3-8B, verified `vLLM 0.7.3`):**

| Accelerator | Interconnect | 2× Scaling Efficiency ($\eta_2$) | 4× Scaling Efficiency ($\eta_4$) |
|---|---|---:|---:|
| **NVIDIA RTX 4090 D** | PCIe 4.0 | **94.5%** | **55.5%** |
| **NVIDIA H20-3e** | NVLink | 77.2% | 54.8% |
| **NVIDIA A100-SXM4-40GB** | NVLink 3.0 | 72.5% | 47.0% |
| **NVIDIA H200** | NVLink 4.0 | **60.2%** | **32.9%** |

- **Why it happens:** Scaling efficiency $\eta_N = T_N / (N \cdot T_1)$ is governed by the Amdahl ratio of per-die throughput to interconnect bandwidth. H200's per-die decode throughput grew faster than NVLink 4.0 all-reduce bandwidth, so communication consumes a larger fraction of each step than on slower-per-die chips.
- **Practical lesson:** For models that fit on 1–2 chips (<70B), prefer **data-parallel replicas** over wide tensor parallelism (`TP > 4`).

---

## 5. Sparse MoE Rewards Aggregate Bandwidth Over Compute

**Suite G (Mixtral-8x7B-Instruct, 8-chip multi-GPU):**

- Mixtral activates only 2 of 8 experts per token, keeping arithmetic intensity below dense 8B inference even at 8-chip scale.
- **H20-3e**, which trails A100-40GB by ~5% on dense Suite A, **beats A100-40GB ×8 by +17%** on Suite G (`3,757` vs `3,218 tok/s` offline) because its aggregate HBM bandwidth becomes the binding resource; **H200 ×8** leads verified vLLM runs at `5,334 tok/s`.

---

## 6. Cost-Efficiency & Quantization Depend on Architecture, Not Labels

- **Value Inversion (Suite F, verified vLLM):** **Tesla T4** achieves **51.3k tok/$/hr** — **6.7×** higher throughput per rental dollar than **H100** (`7.6k tok/$/hr`). Consumer **RTX 4090 D** and **RTX 5090** routinely deliver 2–3× the `tok/$/hr` of datacenter GPUs on edge and offline batch workloads.
- **Quantization Path Sensitivity (Suite C):**
  - On **Hopper / Ada / Blackwell** (native FP8 tensor cores), FP8 delivers ~1.5–1.8× speedup while clearing the MMLU accuracy gate.
  - On **Ampere** (A100/A800/3090), FP8 lacks a native tensor-core path (falling back or emulating), whereas **W8A8 (SmoothQuant / INT8)** engages native INT8 tensor cores and leads quality-adjusted efficiency (`2.30×` on RTX 3090, `1.20×` on A100).

---

## Workload-to-Recipe Decision Matrix

| Workload Regime | Best-Matched Recipe | Weaker Fit (for this regime) | Suite |
|---|---|---|---|
| **Interactive SLA chat** | H100 80GB · SGLang / vLLM | Consumer 24–32 GB GPUs (SLA failure) | Suite A |
| **Long-context / RAG (28K)** | B200, H200, H100 | H20-3e, RTX 5090 | Suite D |
| **Offline batch / edge** | RTX 5090, RTX 4090 D, Tesla T4 | Datacenter flagships (cost penalty) | Suite F |
| **Sparse MoE (Mixtral-8x7B)** | H200 ×8, H20-3e ×8 | Mid-tier compute SKUs, immature TP stacks | Suite G |
| **Dense <70B scaling** | Data-parallel replicas | Tensor parallelism > 4-way | Suite E |
| **Quantized serving (Ampere)** | W8A8 (SmoothQuant) | Emulated FP8, W4A16 at high batch | Suite C |
| **Quantized serving (Ada / Hopper / Blackwell)** | Native FP8 | Unchecked INT4 without MMLU gate | Suite C |
| **Sliding-window attention (10K ctx)** | Stack with SWA-aware KV allocation | Stacks sizing KV as full attention | Suite H |

> Explore live numbers interactively at [freedomintelligence.github.io/AccelMark](https://freedomintelligence.github.io/AccelMark) or locally via `python tools/advisor.py`.
