// deploy-recipe.js — Pure command & config builder for the modal Deploy tab.
// Turns any benchmarked leaderboard row into copyable, isolated-venv setup,
// optional runner YAML config, OpenAI-compatible `--serve` launch, native engine
// CLI, and reproduction commands.

import { suiteMetaFor, baseSuiteId, modelTrackId } from "./data.js";
import { shortVersion } from "./utils.js";

function shellQuote(val) {
  const s = String(val ?? "");
  if (/^[a-zA-Z0-9._:/@-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, "'\\''")}'`;
}

function toYamlLines(obj, indent = 0) {
  const pad = " ".repeat(indent);
  const lines = [];
  for (const [k, v] of Object.entries(obj || {})) {
    if (v === undefined || v === null) continue;
    if (typeof v === "object" && !Array.isArray(v)) {
      lines.push(`${pad}${k}:`);
      lines.push(...toYamlLines(v, indent + 2));
    } else if (Array.isArray(v)) {
      lines.push(`${pad}${k}: [${v.map((x) => JSON.stringify(x)).join(", ")}]`);
    } else if (typeof v === "boolean" || typeof v === "number") {
      lines.push(`${pad}${k}: ${v}`);
    } else {
      lines.push(`${pad}${k}: ${JSON.stringify(String(v))}`);
    }
  }
  return lines;
}

function precisionFlagsForEngine(framework, precision) {
  const p = String(precision || "").toUpperCase();
  const fw = String(framework || "").toLowerCase();
  if (!p || p === "BF16" || p === "FP16") {
    return p === "FP16" && fw.includes("vllm") ? " --dtype float16" : "";
  }
  if (fw.includes("vllm")) {
    if (p === "FP8") return " --quantization fp8";
    if (p === "AWQ" || p === "W4A16") return " --quantization awq";
    if (p === "GPTQ") return " --quantization gptq";
    if (p === "W8A8" || p === "INT8") return " --quantization compressed-tensors";
  }
  if (fw.includes("sglang")) {
    if (p === "FP8") return " --quantization fp8";
    if (p === "AWQ" || p === "W4A16") return " --quantization awq";
  }
  return "";
}

function buildNativeCommand({ framework, modelId, tp, precision, extraConfig, port, runnerId, suiteId }) {
  const fw = String(framework || "").toLowerCase();
  const qFlag = precisionFlagsForEngine(framework, precision);
  const ec = extraConfig && typeof extraConfig === "object" ? extraConfig : {};

  if (fw.includes("vllm")) {
    const flags = [`vllm serve ${shellQuote(modelId)}`];
    if (tp > 1) flags.push(`--tensor-parallel-size ${tp}`);
    if (qFlag) flags.push(qFlag.trim());
    if (ec.max_num_seqs) flags.push(`--max-num-seqs ${ec.max_num_seqs}`);
    if (ec.gpu_memory_utilization) flags.push(`--gpu-memory-utilization ${ec.gpu_memory_utilization}`);
    if (ec.enforce_eager) flags.push("--enforce-eager");
    flags.push(`--port ${port}`);
    return { supported: true, engineLabel: "vLLM CLI", cmd: flags.join(" \\\n  ") };
  }

  if (fw.includes("sglang")) {
    const flags = [`python -m sglang.launch_server --model-path ${shellQuote(modelId)}`];
    if (tp > 1) flags.push(`--tp ${tp}`);
    if (qFlag) flags.push(qFlag.trim());
    if (ec.mem_fraction_static) flags.push(`--mem-fraction-static ${ec.mem_fraction_static}`);
    flags.push(`--port ${port}`);
    return { supported: true, engineLabel: "SGLang CLI", cmd: flags.join(" \\\n  ") };
  }

  if (fw.includes("tensorrt") || fw.includes("trt")) {
    const flags = [`trtllm-serve ${shellQuote(modelId)}`];
    if (tp > 1) flags.push(`--tp_size ${tp}`);
    flags.push(`--port ${port}`);
    return { supported: true, engineLabel: "TensorRT-LLM CLI", cmd: flags.join(" \\\n  ") };
  }

  if (fw.includes("mlx")) {
    return {
      supported: true,
      engineLabel: "mlx-lm CLI",
      cmd: `mlx_lm.server --model ${shellQuote(modelId)} --port ${port}`,
    };
  }

  // Fallback for custom / platform-specific runners
  return {
    supported: false,
    engineLabel: framework || "AccelMark CLI",
    cmd: `python run.py --runner ${runnerId} --suite ${suiteId}${tp > 1 ? ` --tensor-parallel-size ${tp}` : ""} --serve --port ${port}`,
  };
}

/**
 * Build a complete, copyable deployment recipe from a leaderboard row.
 */
export function buildDeployRecipe(row, opts = {}) {
  const d = row?.detail || {};
  const impl = row?.impl || {};
  const suiteId = row?.suite || "suite_A";
  const baseSuite = row?.base_suite || baseSuiteId(suiteId);
  const trackSlug = row?.model_track && row.model_track !== "default"
    ? row.model_track
    : (modelTrackId(suiteId) !== "default" ? modelTrackId(suiteId) : null);
  const suiteMeta = suiteMetaFor(suiteId) || {};
  const trackFlag = trackSlug ? ` --model-track ${trackSlug}` : "";

  const runnerId = row?.implementation_id || impl.id || "auto";
  const framework = row?.framework || d.sw_framework || "vLLM";
  const frameworkVer = shortVersion(row?.framework_version || d.sw_framework_version);
  const precision = row?.precision || d.model_precision || "BF16";
  const tp = Number(d.run_tp || row?.chip_count || 1) || 1;
  const defaultModel = d.model_id || suiteMeta?.workload?.model || "meta-llama/Meta-Llama-3-8B-Instruct";
  const customModel = (opts.model || "").trim();
  const modelId = customModel || defaultModel;
  const usingCustomModel = !!customModel && customModel !== defaultModel;
  const mirror = !!opts.mirror;
  const port = Number(opts.port) || 8000;
  const mode = opts.mode === "native" || opts.mode === "reproduce" ? opts.mode : "serve";
  const extraConfig = d.run_extra_config && typeof d.run_extra_config === "object" && Object.keys(d.run_extra_config).length > 0
    ? d.run_extra_config
    : null;

  // Step 1: Isolated environment setup
  const setupLines = [
    "git clone https://github.com/FreedomIntelligence/AccelMark.git && cd AccelMark",
    `bash scripts/setup_env.sh ${runnerId} --serve${mirror ? " --mirror" : ""}`,
    `source .venv-${runnerId}/bin/activate`,
  ];
  if (mirror) setupLines.push("export HF_ENDPOINT=https://hf-mirror.com");
  const setupCmd = setupLines.join("\n");

  // Optional Step 2: Runner config file when non-default extra_config was recorded
  let configCmd = null;
  let configPath = null;
  if (extraConfig) {
    configPath = `configs/runner_configs/runner_${runnerId}.yaml`;
    const yamlBody = toYamlLines(extraConfig).join("\n");
    configCmd = `mkdir -p configs/runner_configs\ncat <<'EOF' > ${configPath}\n${yamlBody}\nEOF`;
  }

  // Step 3: Launch / Reproduce command
  const tpFlag = tp > 1 ? ` --tensor-parallel-size ${tp}` : "";
  const serveCmd = usingCustomModel
    ? `python run.py --runner ${runnerId} --model ${shellQuote(modelId)}${tpFlag} --serve --port ${port}`
    : `python run.py --runner ${runnerId} --suite ${baseSuite}${trackFlag}${tpFlag} --serve --port ${port}`;

  const nativeInfo = buildNativeCommand({
    framework, modelId, tp, precision, extraConfig, port, runnerId, suiteId,
  });

  const reproduceCmd = `python run.py --runner ${runnerId} --suite ${baseSuite}${trackFlag}${tpFlag}`;

  const launchCmd = mode === "native"
    ? nativeInfo.cmd
    : (mode === "reproduce" ? reproduceCmd : serveCmd);

  // Step 4: Quick OpenAI-compatible smoke test (wrapped cleanly to avoid horizontal clipping)
  const testCmd = [
    `curl http://localhost:${port}/v1/chat/completions \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{`,
    `    "model": ${JSON.stringify(modelId)},`,
    `    "messages": [{"role": "user", "content": "Hello!"}]`,
    `  }'`,
  ].join("\n");

  // Verified environment baseline badges
  const envSpecs = [];
  if (d.hw_chip || row?.chip) {
    envSpecs.push({
      k: "Hardware",
      v: `${d.hw_chip || row.chip}${tp > 1 ? ` ×${tp}` : ""}${d.hw_memory_gb ? ` (${d.hw_memory_gb}GB)` : ""}`,
    });
  }
  envSpecs.push({ k: "Engine", v: [framework, frameworkVer].filter(Boolean).join(" ") });
  envSpecs.push({ k: "Precision", v: precision });
  if (d.sw_runtime) envSpecs.push({ k: "Runtime", v: d.sw_runtime });
  if (d.sw_driver) envSpecs.push({ k: "Driver", v: d.sw_driver });
  if (d.sw_pytorch) envSpecs.push({ k: "PyTorch", v: shortVersion(d.sw_pytorch) || d.sw_pytorch });

  return {
    runnerId,
    framework,
    frameworkVer,
    precision,
    tp,
    suiteId,
    defaultModel,
    modelId,
    usingCustomModel,
    mirror,
    port,
    mode,
    extraConfig,
    hasPatches: !!impl.has_patches,
    dockerImage: impl.deploy?.docker_image || null,
    setupCmd,
    configPath,
    configCmd,
    serveCmd,
    nativeCmd: nativeInfo.cmd,
    nativeLabel: nativeInfo.engineLabel,
    nativeSupported: nativeInfo.supported,
    reproduceCmd,
    launchCmd,
    testCmd,
    envSpecs,
  };
}
