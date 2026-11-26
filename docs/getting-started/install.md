# Installation

> **Prerequisites**: Python ≥ 3.10, git, and a working accelerator driver / runtime for your platform.

## 1. Clone & Install Core Harness

```bash
git clone https://github.com/FreedomIntelligence/AccelMark.git
cd AccelMark
pip install -e .
```

## 2. Install Runner Dependencies by Platform

=== "NVIDIA (vLLM)"

    ```bash
    pip install -r runners/nvidia_vllm_47f5d58e/requirements.txt
    ```

=== "NVIDIA (SGLang)"

    ```bash
    pip install -r runners/nvidia_sglang_c43a8309/requirements.txt
    ```

=== "AMD ROCm"

    ```bash
    pip install -r runners/amd_vllm_rocm_6c18cd8f/requirements.txt
    ```

=== "Huawei Ascend"

    ```bash
    pip install -r runners/ascend_vllm_ascend_d4aa9fda/requirements.txt
    ```

=== "Iluvatar CoreX"

    ```bash
    pip install -r runners/iluvatar_vllm_ixuca_4d3befc9/requirements.txt
    ```

=== "Moore Threads"

    ```bash
    pip install -r runners/moorethreads_vllm_musa_6607740b/requirements.txt
    ```

=== "Google TPU"

    ```bash
    pip install -r runners/google_vllm_tpu_68cc9ffa/requirements.txt
    ```

=== "Apple Silicon (MLX)"

    ```bash
    pip install -r runners/apple_mlx_lm_9546b8b5/requirements.txt
    ```

## 3. Mainland China / Offline Weights (Optional)

If HuggingFace Hub is slow or unreachable in your environment, use either:

```bash
# Option A: HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

Or map local checkpoints once in `configs/models_local.yaml` (git-ignored):

```yaml
Qwen/Qwen2.5-0.5B-Instruct: /data/models/Qwen2.5-0.5B-Instruct
meta-llama/Meta-Llama-3-8B-Instruct: /data/models/Meta-Llama-3-8B-Instruct
```

## 4. Verify Setup in 10 Seconds

```bash
python run.py --init
```

This creates your local `configs/submitter.yaml` handle and verifies hardware detection and runner imports without downloading any model weights. Next: [First Submission (~10 min)](first-submission.md).
