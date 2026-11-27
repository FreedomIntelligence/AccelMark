// agent.js — AccelMark-Agent Optimization Track view (A100 50-cell matrix,
// protocol audit cases, cross-accelerator portability, and open community quests).

import { esc } from "../utils.js";
import {
  AGENT_WORKLOADS,
  A100_MATRIX,
  CROSS_HARDWARE_TRACKS,
  PROTOCOL_AUDIT_CASES,
  AGENT_OPEN_QUESTS,
} from "../../data/agent-results.js";

const REPO_URL = "https://github.com/FreedomIntelligence/AccelMark";

function isZh() {
  if (typeof window === "undefined") return false;
  // window._lang is a getter function; the site's zh locale is keyed "cn".
  const lang = typeof window._lang === "function" ? window._lang() : window._lang;
  return lang === "cn" || lang === "zh";
}

function t(en, zh) {
  return isZh() ? zh : en;
}

function fmtGain(v) {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function renderCell(cell) {
  const classes = ["tnum", "agent-cell"];
  if (cell.unchanged) classes.push("agent-cell-unchanged");
  else if (cell.best) classes.push("agent-cell-best");
  else if (cell.gain > 2.0) classes.push("agent-cell-pos");

  const badges = [];
  if (cell.noTokens) badges.push('<sup title="No model tokens recorded across all 40 rounds">‡</sup>');
  if (cell.repeat) badges.push('<sup title="First search failed final validation; repeat shown">§</sup>');
  const titleParts = [
    cell.unchanged ? t("Artifact fingerprint identical to baseline", "交付产物指纹与基线一致（未改动）") : t("Changed artifact", "产物已修改"),
    cell.gain > 2.0 && !cell.unchanged ? t("Validated success (>2%)", "通过 >2% 有效优化阈值") : "",
  ].filter(Boolean);

  return `<td class="${classes.join(" ")}" title="${esc(titleParts.join(" · "))}">${esc(fmtGain(cell.gain))}${badges.join("")}</td>`;
}

export function render({ el }) {
  const backbones = [...new Set(A100_MATRIX.map((r) => r.backbone))];
  const harnesses = [...new Set(A100_MATRIX.map((r) => r.harness))];

  el.innerHTML = `
    <section class="page-hero">
      <span class="eyebrow">${t("OPTIMIZATION TRACK · ACCELMARK-AGENT", "推理优化赛道 · ACCELMARK-AGENT")}</span>
      <h1>${t("Inference-Optimization Agent Leaderboard", "推理优化 Agent 基准榜单")}</h1>
      <p class="page-lead">${t(
        "Evaluating coding agents that optimize complete LLM serving deployments under a trusted controller: in-place baselines, MMLU accuracy gates (Δ ≤ 0.03), artifact fingerprinting, and 3× independent re-measurements.",
        "在受控验证协议下评测自动优化 LLM 推理部署的 Coding Agent：原地基线测量、MMLU 精度门禁（Δ ≤ 0.03）、产物指纹校验与 3 次独立复测。"
      )}</p>
    </section>

    <section class="section">
      <div class="agent-kpi-grid">
        <div class="card agent-kpi">
          <span class="agent-kpi-label">${t("A100 Matrix Scope", "A100 完整矩阵")}</span>
          <strong class="agent-kpi-value">50 ${t("cells", "组")} · 2,000 ${t("rounds", "轮")}</strong>
          <span class="muted">${t("5 workloads × 5 backbones × 2 harnesses (3,401 A100 rounds total)", "5 负载 × 5 模型 × 2 框架（A100 累计 3,401 轮）")}</span>
        </div>
        <div class="card agent-kpi">
          <span class="agent-kpi-label">${t("Validated Successes", "有效优化通过率")}</span>
          <strong class="agent-kpi-value">26 / 50 (52%)</strong>
          <span class="muted">${t("Changed artifact fingerprint AND final gain > +2.0%", "产物指纹确有变更且 3 次复测中位数 > +2.0%")}</span>
        </div>
        <div class="card agent-kpi">
          <span class="agent-kpi-label">${t("Largest Validated Gain", "最高验证增益")}</span>
          <strong class="agent-kpi-value">+211.72%</strong>
          <span class="muted">${t("Suite H (SWA): window-aware KV admission + mem_fraction_static", "Suite H (SWA)：窗口 KV 准入与显存池协同优化")}</span>
        </div>
        <div class="card agent-kpi">
          <span class="agent-kpi-label">${t("Accelerator Tracks", "跨硬件覆盖")}</span>
          <strong class="agent-kpi-value">4 ${t("devices", "类芯片")}</strong>
          <span class="muted">${t("A100 · RTX 4090 · V100 · Ascend 910C (portability)", "A100 · RTX 4090 · V100 · 昇腾 910C（可移植性）")}</span>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${t("FULLY CROSSED MATRIX", "全交叉评测矩阵")}</span>
          <h2>${t("A100-SXM4-80GB Matrix (G_final %)", "A100-SXM4-80GB 完整矩阵（验证终测涨幅 %）")}</h2>
        </div>
        <span class="section-sub">${t(
          "Grey cells delivered an unchanged artifact (−1.95% to +1.48% noise band). Bold = best per workload.",
          "灰色单元格表示产物指纹未变化（噪声区间 −1.95% 至 +1.48%）；加粗为该负载最优。"
        )}</span>
      </div>

      <div class="agent-filters card">
        <label>
          <span>${t("Backbone", "基座模型")}</span>
          <select id="agent-backbone">
            <option value="">${t("All backbones", "全部模型")}</option>
            ${backbones.map((b) => `<option value="${esc(b)}">${esc(b)}</option>`).join("")}
          </select>
        </label>
        <label>
          <span>${t("Harness", "Agent 框架")}</span>
          <select id="agent-harness">
            <option value="">${t("All harnesses", "全部框架")}</option>
            ${harnesses.map((h) => `<option value="${esc(h)}">${esc(h)}</option>`).join("")}
          </select>
        </label>
        <label>
          <span>${t("Sort by", "排序依据")}</span>
          <select id="agent-sort">
            <option value="default">${t("Default order", "默认顺序")}</option>
            <option value="median">${t("Median gain (desc)", "中位数涨幅（降序）")}</option>
            <option value="successes">${t("Successes / 5 (desc)", "通过数 / 5（降序）")}</option>
            <option value="swa">SWA (Suite H)</option>
            <option value="precision">Precision (Suite C)</option>
            <option value="small">Small (Suite F)</option>
          </select>
        </label>
      </div>

      <div class="contrib-table-wrap card">
        <table class="contrib-table community-data-table agent-matrix-table">
          <thead>
            <tr>
              <th>${t("Backbone", "模型")}</th>
              <th>${t("Harness", "框架")}</th>
              ${AGENT_WORKLOADS.map((w) => `
                <th class="tnum" title="${esc(`${w.model} · ${w.profile}`)}">
                  ${esc(w.label)}<br><small class="muted">${esc(w.suite)}</small>
                </th>
              `).join("")}
              <th class="tnum">${t("Median", "中位数")}</th>
              <th class="tnum">${t("Succ.", "通过")}</th>
            </tr>
          </thead>
          <tbody id="agent-matrix-body"></tbody>
          <tfoot>
            <tr class="agent-summary-row">
              <td colspan="2"><strong>${t("Workload median", "负载中位数")}</strong></td>
              ${AGENT_WORKLOADS.map((w) => `<td class="tnum"><strong>${esc(fmtGain(w.median))}</strong></td>`).join("")}
              <td></td>
              <td></td>
            </tr>
            <tr class="agent-summary-row">
              <td colspan="2"><strong>${t("Successes / 10", "通过数 / 10")}</strong></td>
              ${AGENT_WORKLOADS.map((w) => `<td class="tnum">${w.successes} / 10</td>`).join("")}
              <td></td>
              <td class="tnum"><strong>26 / 50</strong></td>
            </tr>
          </tfoot>
        </table>
      </div>

      <p class="muted" style="margin-top:0.75rem;font-size:0.85rem">
        ${t(
          "Note: Each cell represents one 40-round search under identical per-workload prompts. ‡ No model tokens recorded in all 40 rounds; § first search failed final validation and its repeat is shown. Because single-search variance reached 11pp on repeated Decode runs, compare absolute throughput and delivered artifacts rather than treating single cells as fixed rankings.",
          "说明：每个单元格为同一负载提示词下的单次 40 轮搜索。‡ 表示 40 轮均无模型 token 记录；§ 表示首次终测失败、此处展示复跑结果。由于单次搜索在 Decode 复跑中存在最高 11 个百分点的波动，建议结合绝对吞吐与交付产物做分析，而非将单次结果视为固定排名。"
        )}
      </p>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${t("PROTOCOL AUDIT", "协议防刷榜审计")}</span>
          <h2>${t("Why Naive Peak Throughput Misleads", "为什么不能只看搜索峰值：协议审计案例")}</h2>
        </div>
      </div>
      <div class="contrib-table-wrap card">
        <table class="contrib-table community-data-table">
          <thead>
            <tr>
              <th>${t("Cohort", "测试批次")}</th>
              <th>${t("Backbone / Harness", "模型 / 框架")}</th>
              <th class="tnum">${t("Naive Reading", "原始峰值")}</th>
              <th>${t("Protocol Outcome", "协议审计结果")}</th>
              <th>${t("Check Responsible", "触发门禁")}</th>
            </tr>
          </thead>
          <tbody>
            ${PROTOCOL_AUDIT_CASES.map((c) => `
              <tr>
                <td><strong>${esc(c.cohort)}</strong></td>
                <td>${esc(c.setup)}</td>
                <td class="tnum"><code>${esc(c.naive)}</code></td>
                <td><span class="agent-badge agent-badge-${esc(c.severity)}">${esc(c.outcome)}</span></td>
                <td class="muted">${esc(t(c.gate, c.gateZh || c.gate))}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${t("PORTABILITY TRACKS", "跨硬件可移植性")}</span>
          <h2>${t("Cross-Accelerator Coverage", "多硬件验证覆盖情况")}</h2>
        </div>
        <span class="section-sub">${t(
          "Missing finals are reported as incomplete coverage, never as zero gain.",
          "未通过终测的条件如实记为缺失覆盖，不折算为 0% 增益。"
        )}</span>
      </div>
      <div class="contrib-table-wrap card">
        <table class="contrib-table community-data-table">
          <thead>
            <tr>
              <th>${t("Accelerator", "加速器")}</th>
              <th>${t("Track (Workloads · Precision)", "赛道（负载 · 精度）")}</th>
              <th class="tnum">${t("Validated / Registered", "有效终测 / 注册数")}</th>
              <th class="tnum">${t("Median", "中位数")}</th>
              <th class="tnum">${t("Min", "最小")}</th>
              <th class="tnum">${t("Max", "最大")}</th>
            </tr>
          </thead>
          <tbody>
            ${CROSS_HARDWARE_TRACKS.map((r) => `
              <tr>
                <td><strong>${esc(r.accelerator)}</strong></td>
                <td class="muted">${esc(r.track)}</td>
                <td class="tnum"><code>${esc(r.finals)}</code></td>
                <td class="tnum">${esc(r.median)}</td>
                <td class="tnum">${esc(r.min)}</td>
                <td class="tnum">${esc(r.max)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${t("HELP WANTED", "社区共建任务")}</span>
          <h2>${t("Contribute Missing Agent Benchmarks", "欢迎社区补充以下高价值实验")}</h2>
        </div>
      </div>
      <div class="agent-quest-grid">
        ${AGENT_OPEN_QUESTS.map((q) => {
          const url = `${REPO_URL}/issues/new?title=${encodeURIComponent(q.issueTitle)}&body=${encodeURIComponent(
            `I would like to contribute to **${q.titleEn}**.\n\n- **Hardware**: \n- **Backbone / Harness**: \n- **Notes**: `
          )}`;
          return `
            <div class="card agent-quest-card">
              <h3>${esc(isZh() ? q.titleZh : q.titleEn)}</h3>
              <p class="muted">${esc(isZh() ? q.descZh : q.descEn)}</p>
              <a class="btn small primary" href="${esc(url)}" target="_blank" rel="noopener">${t("Claim on GitHub →", "在 GitHub 认领 →")}</a>
            </div>
          `;
        }).join("")}
      </div>
    </section>
  `;

  const tbody = el.querySelector("#agent-matrix-body");
  const selBackbone = el.querySelector("#agent-backbone");
  const selHarness = el.querySelector("#agent-harness");
  const selSort = el.querySelector("#agent-sort");

  function renderRows() {
    const bVal = selBackbone.value;
    const hVal = selHarness.value;
    const sVal = selSort.value;

    let rows = A100_MATRIX.filter(
      (r) => (!bVal || r.backbone === bVal) && (!hVal || r.harness === hVal)
    );

    if (sVal === "median") rows = [...rows].sort((a, b) => b.median - a.median);
    else if (sVal === "successes") rows = [...rows].sort((a, b) => b.successes - a.successes || b.median - a.median);
    else if (["swa", "precision", "small"].includes(sVal)) {
      rows = [...rows].sort((a, b) => (b.cells[sVal]?.gain ?? -999) - (a.cells[sVal]?.gain ?? -999));
    }

    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td><strong>${esc(r.backbone)}</strong></td>
        <td><code>${esc(r.harness)}</code></td>
        ${AGENT_WORKLOADS.map((w) => renderCell(r.cells[w.id])).join("")}
        <td class="tnum"><strong>${esc(fmtGain(r.median))}</strong></td>
        <td class="tnum">${r.successes} / 5</td>
      </tr>
    `).join("");
  }

  selBackbone.addEventListener("change", renderRows);
  selHarness.addEventListener("change", renderRows);
  selSort.addEventListener("change", renderRows);
  renderRows();
}
