# moorethreads_vllm_musa_f2f6f965 — Moore Threads MUSA Runner (vllm-musa)

AccelMark runner for Moore Threads MUSA GPUs using
[vllm-musa](https://github.com/MooreThreads/vllm-musa).

## Supported suites

| Suite | Description | Notes |
|-------|-------------|-------|
| Suite A | Single-chip, Llama-3-8B | Smoke tested on MTT S4000; accuracy not at baseline on vLLM 0.4.x |
| Suite B | Multi-chip, Llama-3-70B | MCCL tensor parallelism; set `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| Suite C | Quantization, Llama-3.1-8B | FP8 skipped (not supported); W8A8/W8A16 via compressed-tensors |
| Suite D | Long context ~28K input, Llama-3.1-8B | Reduce `max_num_seqs` / `gpu_memory_utilization` in runner config |
| Suite E | Multi-chip scaling, Llama-3-8B | MCCL tensor parallelism |
| Suite F | Edge, Qwen2.5-0.5B | Smoke tested on MTT S4000; recommended first run |
| Suite G | MoE multi-chip, Mixtral-8x7B | Unsupported |

## Hardware compatibility

| GPU | BF16 / FP16 | Multi-chip TP | FP8 | Notes |
|-----|-------------|---------------|-----|-------|
| MTT S4000 / S5000 | ✅ (BF16 → float16 on vLLM &lt; 0.10) | ✅ (MCCL) | ❌ | Tested with vLLM 0.4.x+musa |
| MTT S3000 / S80 | ✅ | ✅ | ❌ | May need `--enforce-eager` on Triton errors |

FP8 is excluded — not supported on this runner. FP32 inference fails with
FlashAttention on MUSA (use FP16 or BF16). Qwen3 requires a newer vLLM + MUSA port
(Qwen2.5 / Llama-3 work on 0.4.x).

## Prerequisites

Install in this order — **do not** `pip install torch` or `vllm` from PyPI on a
bare Linux host:

**1. MUSA toolkit + driver**

<https://developer.mthreads.com/musa/>

**2. vllm-musa (official build)**

| Resource | URL |
|----------|-----|
| Repository | <https://github.com/MooreThreads/vllm-musa> |
| Build guide | [README_vllm_musa.md](https://github.com/MooreThreads/vllm-musa/blob/main/README_vllm_musa.md) |
| PyTorch MUSA | <https://github.com/MooreThreads/torch_musa> |

```bash
git clone https://github.com/MooreThreads/vllm-musa.git
cd vllm-musa
bash build_musa.sh
python -c "from vllm import LLM; print('vllm ok')"
```

**3. Runner dependencies**

```bash
pip install -r runners/moorethreads_vllm_musa_f2f6f965/requirements.txt
```

Pin `transformers` to **4.40–4.46** (not 5.x) when on vLLM 0.4.x.

**Environment variables**

```bash
export MUSA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # when tensor_parallel_size > 1
```

## Smoke test

```bash
python runners/moorethreads_vllm_musa_f2f6f965/test_smoke.py
python runners/moorethreads_vllm_musa_f2f6f965/test_smoke.py /path/to/model
```

## Accuracy

AccelMark runs an integrated MMLU subset after each benchmark using the **same**
vLLM instance as the perf run. The runner sets `device=musa`, dtype, and
tokenizer correctly; low scores on vLLM **0.4.x+musa** reflect broken generation
in that stack, not missing AccelMark wiring.

| Model | Suite | Measured | Baseline |
|-------|-------|----------|----------|
| Qwen2.5-0.5B-Instruct | F | **~0.07** | 0.37 (FP16) / 0.38 (BF16) |
| Llama-3-8B-Instruct | A | **~0.07** | 0.60 (BF16) |

Throughput completes normally; answers are effectively random (repetition, system
prompt regurgitation, similar ~7% across different models).

While accuracy is broken on 0.4.x, use `--skip-accuracy-gate` to finish a perf run:

```bash
python run.py --runner moorethreads_vllm_musa_f2f6f965 \
  --suite suite_F --precision FP16 --skip-accuracy-gate
```

Likely fix: upgrade to vllm-musa aligned with vLLM **0.10+**, keep
`transformers` 4.40–4.46 on legacy forks, then re-run without
`--skip-accuracy-gate`.

## Usage

```bash
python run.py --runner moorethreads_vllm_musa_f2f6f965 --suite suite_F --precision FP16

VLLM_WORKER_MULTIPROC_METHOD=spawn \
python run.py --runner moorethreads_vllm_musa_f2f6f965 \
  --suite suite_B --tensor-parallel-size 8
```

Optional runner config (copy and edit):

```bash
cp configs/runner_configs/runner_moorethreads_vllm_musa_f2f6f965.yaml.example \
   configs/runner_configs/runner_moorethreads_vllm_musa_f2f6f965.yaml
```

| Field | Default | Notes |
|-------|---------|-------|
| `tensor_parallel_size` | 1 | MCCL tensor parallelism |
| `enforce_eager` | false | Only if Triton / graph capture errors |
| `max_num_seqs` | 256 | Lower on small HBM |
| `gpu_memory_utilization` | 0.85 | Lower if OOM |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `GLIBCXX_3.4.30` on import | Import `torch` before `transformers` (runner and smoke test do this) |
| `KeyError: 'type'` in rope_scaling | Pin `transformers==4.46.3` (not 5.x) |
| `Expected musa device, got cuda:0` | Use this runner (`device="musa"`) |
| MMLU ~0.07 | See [Accuracy](#accuracy); `--skip-accuracy-gate` for perf-only runs |
| OOM | Lower `gpu_memory_utilization` / `max_num_seqs` |
| Triton / graph errors | `--enforce-eager` or `enforce_eager: true` in runner YAML |

## Requirements

See `requirements.txt` for AccelMark extras. vLLM, torch_musa, and the MUSA
driver are installed per the official vllm-musa guide above (not from this file).

Minimum environment:

- Moore Threads GPU with MUSA driver
- Python 3.10+
- vllm-musa build per [MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa)
