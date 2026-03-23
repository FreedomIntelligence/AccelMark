# ⚡ AccelMark

**Open benchmark leaderboard for AI accelerators on LLM workloads.**

[![Live Leaderboard](https://img.shields.io/badge/leaderboard-live-brightgreen)](https://juhaoliang1997.github.io/AccelMark)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](docs/CONTRIBUTING.md)

[**→ Live Leaderboard**](https://juhaoliang1997.github.io/AccelMark) · [Contributing](docs/CONTRIBUTING.md) · [Suites](suites/README.md) · [Serve](serve/README.md) · [Development](docs/DEVELOPMENT.md)

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
pip install -r runners/nvidia_vllm_e0859b3c/requirements.txt

# 2. One-time setup
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name

# 3. Run the benchmark (~46 min on A100)
python run.py --runner nvidia_vllm_e0859b3c --suite suite_A --scenario all

# 4. Validate and submit
python runners/validate_submission.py --dir results/community/<your_dir>
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

## Serve

Any runner can be started as an OpenAI-compatible inference server — the same
code that produced the benchmark result serves your API.

```bash
# Install serve dependencies
pip install -r serve/requirements.txt

# Option A — use a suite (model + generation params come from suite.json)
python run.py --runner nvidia_vllm_e0859b3c --suite suite_A --serve

# Option B — specify the model directly, no suite required
python run.py --runner nvidia_vllm_e0859b3c --model meta-llama/Llama-3.1-8B-Instruct --serve

# With options
python run.py --runner nvidia_vllm_e0859b3c --suite suite_A --serve \
    --port 8000 --workers 4 --api-key sk-mykey
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`.

See [serve/README.md](serve/README.md) for the full flag reference and client examples.

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
├── run.py               # Unified CLI entry point (benchmark + serve)
├── runners/             # Platform benchmark runners
│   ├── benchmark_runner.py   # Shared base class — all orchestration logic
│   ├── protocol.py           # RunnerProtocol — shared interface for runners and serve
│   ├── collect_env.py        # Hardware/software detection → env_info.json
│   ├── validate_submission.py
│   ├── hash_runner.py        # Compute runner ID before submission
│   ├── validate_runners.py   # CI: validate all runner folders
│   ├── meta.schema.json      # JSON schema for runner meta.json
│   └── nvidia_vllm_e0859b3c/ # Reference runner (NVIDIA + vLLM)
│       ├── runner.py
│       ├── requirements.txt
│       └── meta.json
├── serve/               # OpenAI-compatible inference server
│   ├── server.py        # FastAPI app, endpoints, start_server()
│   ├── adapter.py       # Pydantic request/response models
│   ├── capacity.py      # Capacity estimates from prior benchmark results
│   ├── requirements.txt
│   └── README.md
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
├── openclaw_skill/      # OpenClaw chat skill — benchmark from any chat app
│   ├── accelmark_skill.py   # Skill entry point (trigger phrases → benchmark → report)
│   ├── skill.json           # Trigger phrases, permissions, metadata
│   ├── requirements.txt
│   ├── README.md
│   ├── mini/                # Auto-adaptive benchmark (hardware detection + tier selection)
│   │   ├── mini_suite_selector.py  # Hardware → tier config (nano / mini / standard / pro)
│   │   ├── run_mini.py             # vLLM + mlx-lm backends, result.json builder
│   │   ├── hardware_assessment.py  # CPU-only analysis and model recommendations
│   │   └── requests.jsonl          # Fixed 200-prompt set (reproducible across hardware)
│   └── tests/
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
