import test from "node:test";
import assert from "node:assert/strict";
import { buildDeployRecipe } from "../assets/js/deploy-recipe.js";

const SAMPLE_ROW = {
  run_id: "r_h100_vllm",
  suite: "suite_A",
  chip: "NVIDIA H100 80GB HBM3",
  chip_count: 1,
  framework: "vLLM",
  framework_version: "0.7.3",
  precision: "FP8",
  implementation_id: "nvidia_vllm_47f5d58e",
  detail: {
    hw_chip: "NVIDIA H100 80GB HBM3",
    hw_memory_gb: 80,
    sw_framework: "vLLM",
    sw_framework_version: "0.7.3",
    sw_runtime: "CUDA 12.4",
    sw_driver: "550.90",
    model_id: "meta-llama/Meta-Llama-3-8B-Instruct",
    run_tp: 2,
    run_extra_config: {
      max_num_seqs: 256,
      gpu_memory_utilization: 0.92,
    },
  },
  impl: {
    id: "nvidia_vllm_47f5d58e",
    has_requirements: true,
    has_patches: false,
  },
};

test("buildDeployRecipe generates isolated setup, YAML config, and --serve launch", () => {
  const rec = buildDeployRecipe(SAMPLE_ROW, { mode: "serve", mirror: false });
  assert.match(rec.setupCmd, /bash scripts\/setup_env\.sh nvidia_vllm_47f5d58e --serve/);
  assert.match(rec.setupCmd, /source \.venv-nvidia_vllm_47f5d58e\/bin\/activate/);
  assert.ok(rec.configCmd);
  assert.match(rec.configCmd, /max_num_seqs: 256/);
  assert.match(rec.launchCmd, /python run\.py --runner nvidia_vllm_47f5d58e --suite suite_A --tensor-parallel-size 2 --serve --port 8000/);
  assert.match(rec.testCmd, /\/v1\/chat\/completions/);
});

test("buildDeployRecipe supports mirror, custom model, and native vLLM CLI", () => {
  const rec = buildDeployRecipe(SAMPLE_ROW, {
    mode: "native",
    mirror: true,
    model: "Qwen/Qwen2.5-7B-Instruct",
  });
  assert.match(rec.setupCmd, /--mirror/);
  assert.match(rec.setupCmd, /HF_ENDPOINT=https:\/\/hf-mirror\.com/);
  assert.match(rec.launchCmd, /vllm serve Qwen\/Qwen2\.5-7B-Instruct/);
  assert.match(rec.launchCmd, /--tensor-parallel-size 2/);
  assert.match(rec.launchCmd, /--quantization fp8/);
  assert.match(rec.launchCmd, /--max-num-seqs 256/);
  assert.match(rec.testCmd, /Qwen\/Qwen2\.5-7B-Instruct/);
});

test("buildDeployRecipe supports SGLang native CLI and reproduce mode", () => {
  const sgRow = {
    ...SAMPLE_ROW,
    framework: "SGLang",
    implementation_id: "nvidia_sglang_c43a8309",
    detail: { ...SAMPLE_ROW.detail, sw_framework: "SGLang", run_tp: 1, run_extra_config: null },
  };
  const nat = buildDeployRecipe(sgRow, { mode: "native" });
  assert.match(nat.launchCmd, /python -m sglang\.launch_server/);
  assert.equal(nat.configCmd, null);

  const rep = buildDeployRecipe(sgRow, { mode: "reproduce" });
  assert.equal(rep.launchCmd, "python run.py --runner nvidia_sglang_c43a8309 --suite suite_A");
});
