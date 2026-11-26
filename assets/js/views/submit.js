// views/submit.js — PyTorch-style one-screen interactive benchmark command builder.
// Zero multi-step wizards: pick ecosystem × suite × submit mode, copy 4 lines, run.

import { SUITE_ORDER, SUITE_META } from "../data.js";
import { esc, copyToClipboard, flashButtonLabel } from "../utils.js";
import { suiteIcon, icon } from "../icons.js";

const isZh = () => typeof window._lang === "function" && window._lang() === "cn";

const ECOSYSTEMS = [
  { id: "nvidia_vllm",   label: "NVIDIA (vLLM)",      alias: "vllm",     req: "runners/nvidia_vllm_47f5d58e/requirements.txt",            note: "Default reference stack for CUDA GPUs (Ampere / Ada / Hopper / Blackwell)." },
  { id: "nvidia_sglang", label: "NVIDIA (SGLang)",    alias: "sglang",   req: "runners/nvidia_sglang_c43a8309/requirements.txt",          note: "High-throughput RadixAttention serving stack on CUDA GPUs." },
  { id: "amd",           label: "AMD ROCm",           alias: "rocm",     req: "runners/amd_vllm_rocm_6c18cd8f/requirements.txt",          note: "vLLM-ROCm for Instinct MI210 / MI250X / MI300X / MI325X / Radeon." },
  { id: "ascend",        label: "Huawei Ascend",      alias: "ascend",   req: "runners/ascend_vllm_ascend_d4aa9fda/requirements.txt",     note: "vllm-ascend on CANN for Ascend 910B / 910C NPUs." },
  { id: "iluvatar",      label: "Iluvatar CoreX",     alias: "iluvatar", req: "runners/iluvatar_vllm_ixuca_4d3befc9/requirements.txt",    note: "vllm-ixuca for MR-V100 / BI-V150 (Suites A–H supported)." },
  { id: "moorethreads",  label: "Moore Threads",      alias: "musa",     req: "runners/moorethreads_vllm_musa_6607740b/requirements.txt", note: "vllm-musa for MTT S4000 / S5000 GPUs." },
  { id: "google",        label: "Google TPU",         alias: "tpu",      req: "runners/google_vllm_tpu_68cc9ffa/requirements.txt",        note: "vllm-tpu (JAX/XLA) on Cloud TPU v5e / v6e." },
  { id: "apple",         label: "Apple Silicon",      alias: "mlx",      req: "runners/apple_mlx_lm_9546b8b5/requirements.txt",           note: "mlx-lm on macOS M1 / M2 / M3 / M4 unified memory." },
];

const SUITE_TIMES = {
  suite_F: { time: "~10 min", badge: "Starter", shortEn: "Edge & Consumer", shortZh: "边缘/消费级", multi: false },
  suite_A: { time: "~11 min", badge: "Core",    shortEn: "Single-Chip Chat", shortZh: "单卡对话", multi: false },
  suite_D: { time: "~22 min", badge: "",        shortEn: "28K Long Context", shortZh: "28K 长上下文", multi: false },
  suite_C: { time: "~22 min", badge: "",        shortEn: "Quantization", shortZh: "量化效率", multi: false },
  suite_H: { time: "~14 min", badge: "",        shortEn: "Sliding Window", shortZh: "滑动窗口 SWA", multi: false },
  suite_B: { time: "~20 min", badge: "Multi",   shortEn: "70B Multi-Chip", shortZh: "70B 多卡 TP", multi: true },
  suite_E: { time: "~9 min",  badge: "Multi",   shortEn: "TP Scaling", shortZh: "多卡扩展", multi: true },
  suite_G: { time: "~35 min", badge: "Multi",   shortEn: "Sparse MoE", shortZh: "稀疏 MoE", multi: true },
};

const SUITE_DISPLAY_ORDER = ["suite_F", "suite_A", "suite_D", "suite_C", "suite_H", "suite_B", "suite_E", "suite_G"];
const ISSUE_ZIP_URL = "https://github.com/FreedomIntelligence/AccelMark/issues/new?template=result_submission.yml";

function buildCommands(state) {
  const eco = ECOSYSTEMS.find((e) => e.id === state.eco) || ECOSYSTEMS[0];
  const sInfo = SUITE_TIMES[state.suite] || SUITE_TIMES.suite_F;
  const lines = [
    "# 1. Clone & install framework + runner deps",
    "git clone https://github.com/FreedomIntelligence/AccelMark.git",
    `cd AccelMark && pip install -e .`,
    `pip install -r ${eco.req}`,
    "",
  ];
  if (state.cnMirror) {
    lines.push("# Mainland China HuggingFace mirror");
    lines.push("export HF_ENDPOINT=https://hf-mirror.com");
    lines.push("");
  }
  lines.push("# 2. Handle setup & 10s pre-flight check (no download)");
  lines.push("python run.py --init");
  lines.push("");
  lines.push(`# 3. Run ${state.suite} (${sInfo.time}) — MMLU gate runs first`);
  const tpFlag = sInfo.multi ? " --tensor-parallel-size 4" : "";
  lines.push(`python run.py --runner ${eco.alias} --suite ${state.suite}${tpFlag}`);
  lines.push("");
  if (state.mode === "zip") {
    lines.push("# 4. Validate & package no-git ZIP for GitHub Issue");
    lines.push("python run.py --submit results/community/<run_dir> --zip");
  } else {
    lines.push("# 4. Validate, preview rank & get exact PR command");
    lines.push("python run.py --submit results/community/<run_dir>");
  }
  return lines.join("\n");
}

export function render({ el }) {
  const zh = isZh();
  const state = {
    eco: "nvidia_vllm",
    suite: "suite_F",
    mode: "pr",
    cnMirror: zh,
  };

  function paint() {
    const eco = ECOSYSTEMS.find((e) => e.id === state.eco) || ECOSYSTEMS[0];
    const sInfo = SUITE_TIMES[state.suite] || SUITE_TIMES.suite_F;
    const cmdText = buildCommands(state);

    el.innerHTML = `
      <header class="page-head-left">
        <span class="eyebrow">${zh ? "参与贡献" : "Contribute"}</span>
        <h1>${zh ? "10 分钟提交你的硬件实测" : "Submit a benchmark in ~10 minutes"}</h1>
        <p class="page-lead">${zh
          ? "无需极致调优。消费卡、老架构和国产卡的诚实默认配置（Closed Reference）同样宝贵。点选下方配置，直接复制右侧命令开跑。"
          : "No SOTA tuning required. Honest out-of-the-box reference baselines on consumer, older, or emerging chips are just as valuable. Select your setup and copy the commands."}</p>
      </header>

      <div class="sub-builder">
        <div class="sub-config">
          <div class="sub-group">
            <div class="sub-group-label mono">${zh ? "1 · 选择硬件与推理栈" : "1 · Hardware & Stack"}</div>
            <div class="sub-pills">
              ${ECOSYSTEMS.map((e) => `
                <button type="button" class="sub-pill${state.eco === e.id ? " active" : ""}" data-sub-eco="${esc(e.id)}">
                  ${esc(e.label)}
                </button>
              `).join("")}
            </div>
            <div class="sub-hint">${esc(eco.note)}</div>
          </div>

          <div class="sub-group">
            <div class="sub-group-label mono">${zh ? "2 · 选择基准套件" : "2 · Benchmark Suite"}</div>
            <div class="sub-suite-grid">
              ${SUITE_DISPLAY_ORDER.map((sid) => {
                const m = SUITE_META[sid];
                const info = SUITE_TIMES[sid];
                return `
                  <button type="button" class="sub-suite-btn${state.suite === sid ? " active" : ""}" data-sub-suite="${esc(sid)}">
                    <span class="sub-sb-top">
                      <span class="sub-sb-ic">${suiteIcon(sid, 14)}</span>
                      <span class="mono"><strong>${esc(m.letter)}</strong></span>
                      <span class="sub-sb-time mono">${esc(info.time)}</span>
                    </span>
                    <span class="sub-sb-name">${esc(zh ? info.shortZh : info.shortEn)}</span>
                  </button>
                `;
              }).join("")}
            </div>
          </div>

          <div class="sub-group">
            <div class="sub-group-label mono">${zh ? "3 · 提交方式与网络" : "3 · Delivery & Network"}</div>
            <div class="sub-row-opts">
              <div class="sub-seg">
                <button type="button" class="sub-seg-btn${state.mode === "pr" ? " active" : ""}" data-sub-mode="pr">
                  ${icon("gitPullRequest", 13)} ${zh ? "GitHub PR (推荐)" : "GitHub PR"}
                </button>
                <button type="button" class="sub-seg-btn${state.mode === "zip" ? " active" : ""}" data-sub-mode="zip">
                  ${icon("download", 13)} ${zh ? "免 Git 打包上传 (--zip)" : "No-Git ZIP Upload"}
                </button>
              </div>

              <label class="sub-check">
                <input type="checkbox" id="sub-cn-mirror" ${state.cnMirror ? "checked" : ""}>
                <span>${zh ? "使用国内 HuggingFace 镜像 (hf-mirror.com)" : "Use China HF mirror (hf-mirror.com)"}</span>
              </label>
            </div>
          </div>
        </div>

        <div class="sub-terminal-col">
          <div class="sub-terminal">
            <div class="sub-term-head">
              <span class="sub-term-title mono">${icon("terminal", 14)} Quick Start · ${esc(state.suite)} (${esc(sInfo.time)})</span>
              <button type="button" class="sub-copy-btn" id="sub-copy-cmd">
                ${icon("copy", 13)} <span>${zh ? "复制命令" : "Copy"}</span>
              </button>
            </div>
            <pre class="sub-term-pre mono" id="sub-cmd-pre">${esc(cmdText)}</pre>
            <div class="sub-term-foot">
              ${state.mode === "zip"
                ? `<span>${zh ? "生成 zip 后直接上传至" : "Attach the generated zip on"} <a href="${ISSUE_ZIP_URL}" target="_blank" rel="noopener">${zh ? "GitHub 提交表单 →" : "GitHub Submission Issue →"}</a></span>`
                : `<span>${zh ? "CI 自动校验 schema、MMLU 精度与 run_id 完整性，合并后即刻上榜。" : "CI automatically verifies schema, MMLU gate, and run_id integrity on PR."}</span>`}
            </div>
          </div>

          <div class="sub-quick-links">
            <a class="sub-qlink" href="#/wanted">
              <strong>${zh ? "查看征集硬件列表" : "Wanted Hardware"}</strong>
              <span>${zh ? "优先提交榜上空缺的芯片，获得首发归属标记 →" : "Fill missing chips for First-Result attribution →"}</span>
            </a>
            <a class="sub-qlink" href="#/reproduce">
              <strong>${zh ? "参与 ±5% 复现验证" : "Reproduction Quests"}</strong>
              <span>${zh ? "复现已有 community 提交并晋升为 Verified →" : "Independently rerun a community entry within ±5% →"}</span>
            </a>
          </div>
        </div>
      </div>
    `;
  }

  paint();

  if (!el.__subBound) {
    el.__subBound = true;
    el.addEventListener("click", async (ev) => {
      if (!location.hash.startsWith("#/submit")) return;
      const ecoBtn = ev.target.closest("[data-sub-eco]");
      if (ecoBtn) { state.eco = ecoBtn.dataset.subEco; paint(); return; }
      const suiteBtn = ev.target.closest("[data-sub-suite]");
      if (suiteBtn) { state.suite = suiteBtn.dataset.subSuite; paint(); return; }
      const modeBtn = ev.target.closest("[data-sub-mode]");
      if (modeBtn) { state.mode = modeBtn.dataset.subMode; paint(); return; }
      const copyBtn = ev.target.closest("#sub-copy-cmd");
      if (copyBtn) {
        const pre = el.querySelector("#sub-cmd-pre");
        if (pre) {
          const ok = await copyToClipboard(pre.textContent);
          flashButtonLabel(copyBtn, ok ? (isZh() ? "已复制" : "Copied!") : "Failed", {
            holdMs: 1400,
            labelSelector: "span",
          });
        }
      }
    });
    el.addEventListener("change", (ev) => {
      if (ev.target?.id === "sub-cn-mirror") {
        state.cnMirror = ev.target.checked;
        paint();
      }
    });
  }
}
