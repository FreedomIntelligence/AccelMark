# Frequently Asked Questions (FAQ)

### Do I need to tune flags or achieve SOTA speed to submit?
**No — out-of-the-box reference baselines are the most valuable submissions.** AccelMark is designed to measure what practitioners actually get with standard serving recipes, especially on consumer GPUs, older datacenter cards, AMD ROCm, Huawei Ascend, Apple Silicon, Google TPU, Moore Threads, and Iluvatar CoreX.

---

### How long does a benchmark run take?
- **Pre-flight check (`python run.py --init`):** ~10 seconds, no model download.
- **Suite F (Qwen2.5-0.5B — recommended first run):** ~10 minutes on most consumer or datacenter GPUs.
- **Suite A (Llama-3-8B):** ~11 minutes on A100-80GB.
- **Suite D / C / B:** ~20–22 minutes on A100-80GB.
- **Suite G (Mixtral-8x7B MoE):** ~35 minutes on multi-chip nodes.
- **Suite H (Mistral-7B 10K SWA):** ~14 minutes on A100-80GB.

Interrupted runs resume automatically when you re-run the same command.

---

### I'm behind a firewall / in mainland China — can I use a HuggingFace mirror or local weights?
Yes. Two options work out of the box:
1. **HF Mirror:** Export `HF_ENDPOINT=https://hf-mirror.com` (and `HF_TOKEN=...` for gated Llama models).
2. **Local / air-gapped weights:** Copy `configs/models_local.yaml.example` to `configs/models_local.yaml` and point each model ID to your local directory, or pass `--model-path /path/to/model`. AccelMark still records the canonical HuggingFace model ID and revision for apples-to-apples indexing.

---

### Why did my fast GPU get `0.00 QPS` (SLA FAIL) on Suite A Online?
That is **expected behavior, not a bug** — and one of AccelMark's core empirical findings!
The online scenario enforces a strict **p99 TTFT ≤ 500 ms** latency SLA under Poisson arrivals. On 24–32 GB GPUs (e.g., RTX 4090, RTX 5090), high concurrency saturates the KV cache and pushes queueing latency above 500 ms even when offline throughput exceeds 3,000 tok/s. Submit the result as-is — honest SLA limits are critical data for the community.

---

### Can I submit results without using `git` or opening a PR manually?
Yes! Run:
```bash
python run.py --submit results/community/<run_name> --zip
```
This validates your files, prints your projected leaderboard rank, and packages a clean `.zip` bundle you can attach directly to our [Result Submission Issue form](https://github.com/FreedomIntelligence/AccelMark/issues/new?template=result_submission.md).

---

### What is the difference between `community` vs `verified` and `reference` vs `optimized`?
- **`community` vs `verified` (trust tier):**
  - `community`: Passed all automated CI schema, accuracy, and integrity checks.
  - `verified`: Independently reproduced on the same hardware and runner by another run within **±5%**.
- **`reference` vs `optimized` vs `experimental` (runner tier):**
  - `reference`: Standard out-of-the-box engine runner (e.g., `nvidia_vllm_47f5d58e`, `nvidia_sglang_c43a8309`).
  - `optimized`: Performance-tuned or patched runner targeting higher throughput on a specific workload.
  - `experimental`: Research forks or early previews (e.g., custom quantization/attention kernels).

You can filter by `Runner` tier on the [Results page](https://freedomintelligence.github.io/AccelMark/#/rankings) or locally via `python tools/advisor.py --tier reference`.

---

### Why did CI reject my new runner PR?
The two most common reasons:
1. **Hash suffix mismatch:** A runner folder must be named `{platform}_{customname}_{hash8}` where `hash8` matches `python runners/hash_runner.py runners/<your_folder>/`.
2. **Vendored upstream source code:** CI rejects `.vendor/` directories or runner folders with >100 files so the repository stays lightweight. Keep custom framework edits as minimal `patches/*.patch` files applied via an `apply_patches.sh` script (see [CONTRIBUTING.md#adding-a-new-runner](../CONTRIBUTING.md#adding-a-new-runner)).

Note: You do **not** need to hand-edit the Supported Platforms table in `README.md` — maintainers regenerate it periodically.

---

### How do I query AccelMark data in my own script or AI agent?
Either fetch the static auth-free JSON endpoints published on GitHub Pages:
- `https://freedomintelligence.github.io/AccelMark/api/chips.json`
- `https://freedomintelligence.github.io/AccelMark/api/rank.json`
- `https://freedomintelligence.github.io/AccelMark/api/index.json`
- `https://freedomintelligence.github.io/AccelMark/api/suites.json`

Or use our zero-GPU CLI advisor locally:
```bash
python tools/advisor.py --suite suite_A --budget 2.0 --tier reference
python tools/advisor.py --engine-compare "A100"
python tools/advisor.py --compare "5090" "4090" "A100" --format markdown
```
