# 中文快速上手（约 10 分钟）

> **无需极致调参。** 消费级显卡、老卡及国产加速卡（昇腾、天数智芯、摩尔线程等）的诚实默认配置（`Closed` 参考栈）与数据中心旗舰卡同等珍贵。

## 1. 克隆并安装核心依赖

```bash
git clone https://github.com/FreedomIntelligence/AccelMark.git
cd AccelMark
pip install -e .
```

根据你的硬件生态安装对应 Runner 依赖（也可直接在 [一屏命令生成器 ↗](https://freedomintelligence.github.io/AccelMark/#/submit) 中点选生成）：

- **NVIDIA (vLLM)**：`pip install -r runners/nvidia_vllm_47f5d58e/requirements.txt`
- **NVIDIA (SGLang)**：`pip install -r runners/nvidia_sglang_c43a8309/requirements.txt`
- **华为昇腾 Ascend**：`pip install -r runners/ascend_vllm_ascend_d4aa9fda/requirements.txt`
- **天数智芯 Iluvatar**：`pip install -r runners/iluvatar_vllm_ixuca_4d3befc9/requirements.txt`
- **摩尔线程 MTT**：`pip install -r runners/moorethreads_vllm_musa_6607740b/requirements.txt`
- **AMD ROCm**：`pip install -r runners/amd_vllm_rocm_6c18cd8f/requirements.txt`
- **Apple Silicon**：`pip install -r runners/apple_mlx_lm_9546b8b5/requirements.txt`

## 2. 中国大陆镜像或本地权重配置

如直连 HuggingFace 较慢，二选一即可：

```bash
# 方式 A：使用国内 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

或 **方式 B**：在 `configs/models_local.yaml`（已被 `.gitignore` 忽略）中映射本地模型目录：

```yaml
Qwen/Qwen2.5-0.5B-Instruct: /data/models/Qwen2.5-0.5B-Instruct
meta-llama/Meta-Llama-3-8B-Instruct: /data/models/Meta-Llama-3-8B-Instruct
```

## 3. 初始化、运行并提交

```bash
# 10 秒初始化署名与环境预检（不下载模型）
python run.py --init

# 跑最快的 Suite F（单卡约 10 分钟，自动先过 MMLU 精度门禁）
python run.py --runner auto --suite suite_F

# 校验并预览预计排名（不想用 git 可加 --zip 打包后通过 GitHub Issue 表单上传）
python run.py --submit results/community/<run_dir>
```

遇到问题请查阅 [中文常见问题 FAQ](faq.md)。
