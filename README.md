# ⚡ AccelMark

**Open benchmark leaderboard for AI accelerators on LLM workloads.**

[**→ Live Leaderboard**](https://juhaoliang1997.github.io/AccelMark) · [Contributing](docs/CONTRIBUTING.md) · [Suites](suites/README.md) · [Development](docs/DEVELOPMENT.md)

---

## Why AccelMark?

- **MLPerf** is rigorous but slow to update — only large vendors participate
- **Vendor whitepapers** use different setups, making cross-vendor comparison impossible
- **Most benchmarks** cover only NVIDIA and only throughput

AccelMark is different: anyone with a GPU can run a benchmark and submit results in under an hour. A fixed schema and shared LoadGen ensure all results are directly comparable across NVIDIA, AMD, Huawei Ascend, Apple Silicon, and more.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/JuhaoLiang1997/AccelMark.git
cd AccelMark
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/nvidia/requirements.txt

# 2. One-time setup
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name

# 3. Run the full benchmark (~46 min on A100)
python scripts/nvidia/run_vllm.py --suite suite_A --scenario all

# 4. Validate and submit
python scripts/validate_submission.py --dir results/community/<your_dir>
# Open a GitHub Issue with the "Community Submission" template
```

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full guide.

---

## Suites

| Suite | Model | Chips | Primary question |
|-------|-------|-------|-----------------|
| A | Llama-3-8B | 1 | How fast is this chip at inference? |
| B | Llama-3-70B | flexible | Can this chip serve large models? |
| C | Llama-3-8B | 1 | Quantization speed/quality tradeoff? |
| D | Llama-3.1-8B | 1 | How does this chip handle 32K-token inputs? |
| E | Llama-3-8B | 1×/2×/4×/8× | How well does this chip scale? |

See [suites/README.md](suites/README.md) for full specs, time budgets, and metrics.

---

## Supported Platforms

| Platform | Framework | A | B | C | D | E |
|----------|-----------|---|---|---|---|---|
| NVIDIA (H100/A100/A800) | vLLM | ✓ | ✓ | ✓ | ✓ | ✓ |
| NVIDIA (H100/A100) | TensorRT-LLM | ✓ | — | — | — | — |
| AMD (MI300X) | vLLM ROCm | ✓ | — | — | — | — |
| Huawei Ascend | MindIE | ✓ | — | — | — | — |
| Apple Silicon | mlx-lm | ✓ | — | — | — | — |

Adding a new platform? See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md#adding-support-for-a-new-platform).

---

## Leaderboard Tiers

| Tier | How | Where |
|------|-----|-------|
| **community** | Submit via GitHub Issue, passes CI validation | Community tab |
| **verified** | Independently reproduced by maintainer within 5% | Main leaderboard |

---

## Repository Structure

```
AccelMark/
├── suites/              # Suite definitions — see suites/README.md
├── scripts/             # Platform benchmark runners
│   ├── benchmark_runner.py   # Shared base class
│   ├── nvidia/          # vLLM (NVIDIA)
│   ├── amd/             # vLLM ROCm (AMD)
│   └── ascend/          # MindIE (Huawei Ascend)
├── loadgen/             # Shared request sending and timing logic
├── schema/              # JSON schemas, accuracy subset, cloud pricing
├── results/             # Benchmark results — see results/README.md
│   ├── verified/
│   └── community/
├── leaderboard/         # Static leaderboard site (GitHub Pages)
├── docs/                # Documentation
│   ├── CONTRIBUTING.md
│   └── DEVELOPMENT.md
└── configs/             # Local config (gitignored)
```
