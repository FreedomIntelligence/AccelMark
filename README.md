# ⚡ AccelMark

**Open benchmark leaderboard for AI accelerators on LLM workloads.**

[![Live Leaderboard](https://img.shields.io/badge/leaderboard-live-brightgreen)](https://juhaoliang1997.github.io/AccelMark)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)

[**→ Live Leaderboard**](https://juhaoliang1997.github.io/AccelMark) · [Contributing](CONTRIBUTING.md) · [Suites](suites/README.md) · [Development](DEVELOPMENT.md)

---

## Why AccelMark?

| | The problem | AccelMark's answer |
|---|---|---|
| **MLPerf** | Rigorous but slow — only large vendors participate | Community runs often finish quickly (e.g. Suite A default ~11 min; Suite D default ~22 min; full all-scenarios run ~7 h) |
| **Vendor whitepapers** | Different setups make cross-vendor comparison impossible | Fixed schema + shared LoadGen = apples-to-apples |
| **Most benchmarks** | Cover only NVIDIA and only throughput | NVIDIA, AMD, Huawei Ascend, Apple Silicon — throughput, latency, scaling, quantization |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/JuhaoLiang1997/AccelMark.git
cd AccelMark
pip install -e .                                              # installs framework dependencies (Python >=3.10 required)
pip install -r runners/nvidia_vllm_47f5d58e/requirements.txt # installs runner dependencies

# 2. One-time setup
cp configs/submitter.yaml.example configs/submitter.yaml
# Edit configs/submitter.yaml — add your name

# 3. Run the benchmark (~11 min on A100)
python run.py --runner nvidia_vllm_47f5d58e --suite suite_A

# 4. Submit — open a GitHub Issue and paste your result.json
# https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=community_submission.md
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## Suites

| Suite | Model | Chips | Question answered | Primary metric |
|-------|-------|-------|-------------------|----------------|
| **A** | Llama-3-8B | 1 | How fast is this chip at inference? | Offline tokens/sec |
| **B** | Llama-3-70B | flexible | Can this chip serve large models? | Offline tokens/sec |
| **C** | Llama-3.1-8B | 1 | Quantization speed/quality tradeoff? | Speedup vs BF16 |
| **D** | Llama-3.1-8B | 1 | How does this chip handle long-context (28K) inputs? | Offline tokens/sec |
| **E** | Llama-3-8B | 1×/2×/4×/8× | How well does this chip scale? | Scaling efficiency |
| **F** | Qwen2.5-0.5B | 1 | How fast is this consumer/edge GPU? | Offline tokens/sec |
| **G** | Mixtral-8x7B-Instruct | ≥2 (auto) | How efficiently does this chip handle sparse MoE inference? | Offline tokens/sec |

Suites A, B, and D also include optional **speculative decoding** and/or **burst load** extra scenarios — see [suites/README.md](suites/README.md) for per-suite details.

See [suites/README.md](suites/README.md) for full specs, time budgets, SLA definitions, and metric descriptions.

---

## Supported platforms

Reference runners live under `runners/` (see each folder’s `meta.json`). Checkmarks mark suites **implemented and runnable** with that runner in this repository.

| Hardware | Runner folder | Framework | A | B | C | D | E | F | G |
|----------|---------------|-----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| NVIDIA GPU | `nvidia_vllm_47f5d58e` | vLLM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| NVIDIA GPU | `nvidia_sglang_6da83845` | SGLang | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AMD GPU | `amd_vllm_rocm_5355c2c6` | vLLM (ROCm) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Huawei Ascend NPU | `ascend_vllm_ascend_605db33a` | vLLM-Ascend | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| Google TPU | `google_vllm_tpu_68cc9ffa` | vllm-tpu (JAX/XLA) | ✓ | — | — | ✓ | — | ✓ | — |

Other stacks (TensorRT-LLM, MindIE, mlx-lm, etc.) can be added as new runner folders; see the contributor guide.

Adding a new platform? See [CONTRIBUTING.md#adding-support-for-a-new-platform](CONTRIBUTING.md#adding-support-for-a-new-platform).

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
├── runners/             # Platform benchmark runners
│   ├── benchmark_runner.py   # Shared base class — all orchestration logic
│   ├── collect_env.py        # Hardware/software detection → env_info.json
│   ├── validate_submission.py
│   ├── validate_runners.py
│   ├── protocol.py           # RunnerProtocol interface (serve layer)
│   ├── template/             # Annotated starter template for new runners
│   └── nvidia_vllm_{hash8}/  # Example: NVIDIA vLLM runner
│       ├── runner.py
│       ├── meta.json
│       └── requirements.txt
├── loadgen/             # Shared request sending and timing logic
│   ├── loadgen.py       # Core timing engine — do not modify per-platform
│   └── types.py         # InferenceResult, SampleRecord
├── serve/               # OpenAI-compatible inference server
│   ├── server.py        # FastAPI app — wraps any runner as an HTTP API
│   └── adapter.py       # OpenAI request/response models
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
├── run.py               # Unified entry point — benchmark and serve
├── CONTRIBUTING.md
├── DEVELOPMENT.md
└── configs/             # Local config — gitignored
    └── submitter.yaml.example
```

---

## Contributing

The most valuable contribution is running the benchmark on hardware not yet in the leaderboard.

- **Submit a result** → [Community Submission guide](CONTRIBUTING.md)
- **Report a bug** → [Open an issue](https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=bug_report.md)
- **Add platform support** → [Platform guide](CONTRIBUTING.md#adding-support-for-a-new-platform)
- **Extend the leaderboard** → [Development guide](DEVELOPMENT.md)

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