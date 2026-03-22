# ⚡ AccelMark

**Open benchmark leaderboard for AI accelerators on LLM workloads.**

[![Live Leaderboard](https://img.shields.io/badge/leaderboard-live-brightgreen)](https://juhaoliang1997.github.io/AccelMark)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](docs/CONTRIBUTING.md)

[**→ Live Leaderboard**](https://juhaoliang1997.github.io/AccelMark) · [Contributing](docs/CONTRIBUTING.md) · [Suites](suites/README.md) · [Development](docs/DEVELOPMENT.md)

---

## Why AccelMark?

| | The problem | AccelMark's answer |
|---|---|---|
| **MLPerf** | Rigorous but slow — only large vendors participate | Anyone with a GPU submits in under an hour |
| **Vendor whitepapers** | Different setups make cross-vendor comparison impossible | Fixed schema + shared LoadGen = apples-to-apples |
| **Most benchmarks** | Cover only NVIDIA and only throughput | NVIDIA, AMD, Huawei Ascend, Apple Silicon — throughput, latency, scaling, quantization |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/JuhaoLiang1997/AccelMark.git
cd AccelMark
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/nvidia/requirements.txt

# 2. One-time setup
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name

# 3. Run the benchmark (~46 min on A100)
python scripts/nvidia/run_vllm.py --suite suite_A --scenario all

# 4. Validate and submit
python scripts/validate_submission.py --dir results/community/<your_dir>
# Open a GitHub Issue with the "Community Submission" template
```

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the full guide.

---

## Suites

| Suite | Model | Chips | Question answered | Primary metric |
|-------|-------|-------|-------------------|----------------|
| **A** | Llama-3-8B | 1 | How fast is this chip at inference? | Offline tokens/sec |
| **B** | Llama-3-70B | flexible | Can this chip serve large models? | Offline tokens/sec |
| **C** | Llama-3-8B | 1 | Quantization speed/quality tradeoff? | Speedup vs BF16 |
| **D** | Llama-3.1-8B | 1 | How does this chip handle 32K-token inputs? | Interactive TTFT p99 |
| **E** | Llama-3-8B | 1×/2×/4×/8× | How well does this chip scale? | Scaling efficiency |

See [suites/README.md](suites/README.md) for full specs, time budgets, SLA definitions, and metric descriptions.

---

## Supported platforms

| Platform | Framework | A | B | C | D | E |
|----------|-----------|:-:|:-:|:-:|:-:|:-:|
| NVIDIA (H100/A100/A800) | vLLM | ✓ | ✓ | ✓ | ✓ | ✓ |
| NVIDIA (H100/A100) | TensorRT-LLM | ✓ | — | — | — | — |
| AMD (MI300X) | vLLM ROCm | ✓ | — | — | — | — |
| Huawei Ascend | MindIE | ✓ | — | — | — | — |
| Apple Silicon | mlx-lm | ✓ | — | — | — | — |

Adding a new platform? See [docs/CONTRIBUTING.md#adding-support-for-a-new-platform](docs/CONTRIBUTING.md#adding-support-for-a-new-platform).

---

## Leaderboard tiers

| Tier | How | Where |
|------|-----|-------|
| **community** | Submit via GitHub Issue, passes CI validation | Community tab |
| **verified** | Independently reproduced by maintainer within 5% | Main leaderboard |

Community results are fully visible and comparable — they just haven't been independently reproduced yet.

---

## Repository structure

```
AccelMark/
├── suites/              # Suite definitions — see suites/README.md
├── scripts/             # Platform benchmark runners
│   ├── benchmark_runner.py   # Shared base class — all orchestration logic
│   ├── collect_env.py        # Hardware/software detection → env_info.json
│   ├── validate_submission.py
│   ├── nvidia/          # vLLM (NVIDIA CUDA)
│   ├── amd/             # vLLM ROCm (AMD)
│   └── ascend/          # MindIE (Huawei Ascend)
├── loadgen/             # Shared request sending and timing logic
│   ├── loadgen.py       # Core timing engine — do not modify per-platform
│   └── types.py         # InferenceResult, SampleRecord
├── schema/              # JSON schemas, accuracy subset, cloud pricing
│   ├── result.schema.json
│   ├── accuracy_subset.jsonl    # immutable
│   └── cloud_pricing.json
├── results/             # Benchmark results
│   ├── verified/        # Maintainer-reproduced results
│   └── community/       # Community-submitted results
├── leaderboard/         # Static leaderboard site (GitHub Pages)
│   ├── generate.py      # Reads results/, writes leaderboard.js + api/
│   └── site/
│       ├── index.html
│       └── leaderboard.js   # Auto-generated — do not edit manually
├── docs/
│   ├── CONTRIBUTING.md
│   └── DEVELOPMENT.md
└── configs/             # Local config — gitignored
    └── submitter.yaml.example
```

---

## Contributing

The most valuable contribution is running the benchmark on hardware not yet in the leaderboard.

- **Submit a result** → [Community Submission guide](docs/CONTRIBUTING.md)
- **Report a bug** → [Open an issue](https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=bug_report.md)
- **Add platform support** → [Platform guide](docs/CONTRIBUTING.md#adding-support-for-a-new-platform)
- **Extend the leaderboard** → [Development guide](docs/DEVELOPMENT.md)

---

## Citation

If you use AccelMark results in research, please cite:

```bibtex
@misc{accelmark2026,
  title  = {AccelMark: Open Benchmark Leaderboard for AI Accelerators on LLM Workloads},
  author = {Liang, Juhao},
  year   = {2026},
  url    = {https://github.com/JuhaoLiang1997/AccelMark}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
Submitted benchmark results are contributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).