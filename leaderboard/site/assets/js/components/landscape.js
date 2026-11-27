// components/landscape.js — Zero-CDN interactive SVG multi-suite Recipe Landscape.
// Mounted inside Leaderboard → Landscape view; adapts instantly to light/dark theme.

import { SUITE_META, VENDOR_ORDER, VENDOR_COLORS, allRows, baseSuiteId, modelTrackId } from "../data.js";
import { esc, fmtNum, chipHref } from "../utils.js";

const LANDSCAPE_SUITES = ["suite_A", "suite_B", "suite_C", "suite_D", "suite_F", "suite_G", "suite_H"];

function getMetric(row, mode) {
  const base = row.base_suite || baseSuiteId(row.suite);
  if (mode === "qps") return row.online_max_qps != null && row.online_max_qps > 0 ? row.online_max_qps : null;
  if (mode === "value") return row.cost_efficiency_toks_per_dollar_hr > 0 ? row.cost_efficiency_toks_per_dollar_hr : null;
  if (base === "suite_C") return row.quant_best_throughput > 0 ? row.quant_best_throughput : (row.primary_metric > 0 ? row.primary_metric : null);
  return row.offline_throughput > 0 ? row.offline_throughput : (row.primary_metric > 0 ? row.primary_metric : null);
}

function deterministicJitter(str, span) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (((h >>> 0) % 1000) / 1000 - 0.5) * span;
}

export function renderLandscape(container, { closedOnly = true, vendor = "", framework = "", metric = "offline", track = "classic" } = {}) {
  if (!container) return;
  const zh = typeof window._lang === "function" && window._lang() === "cn";
  const trackMode = track === "2026" ? "2026" : "classic";

  const rows = allRows().filter((r) => {
    if (r.flagged) return false;
    const base = r.base_suite || baseSuiteId(r.suite);
    if (!LANDSCAPE_SUITES.includes(base)) return false;
    const trk = r.model_track || modelTrackId(r.suite);
    if (trackMode === "2026" && trk === "default") return false;
    if (trackMode === "classic" && trk !== "default") return false;
    if (closedOnly && (r.runner_tier || "reference") !== "reference") return false;
    if (vendor && r.vendor !== vendor) return false;
    if (framework && r.framework !== framework) return false;
    return getMetric(r, metric) !== null;
  });

  const suites = LANDSCAPE_SUITES.filter((s) => rows.some((r) => (r.base_suite || baseSuiteId(r.suite)) === s));
  const trackToggleHtml = `
    <div class="ls-metric-tabs" role="group" aria-label="Model track">
      <button type="button" class="ls-mtab${trackMode === "classic" ? " active" : ""}" data-ls-track="classic">${zh ? "经典基线" : "Classic"}</button>
      <button type="button" class="ls-mtab${trackMode === "2026" ? " active" : ""}" data-ls-track="2026">${zh ? "2026 新轨" : "2026 Track"}</button>
    </div>
  `;

  if (!rows.length || !suites.length) {
    container.innerHTML = `
      <div class="ls-card">
        <div class="ls-head">
          <div>
            <div class="ls-title">${zh ? "全景分布图" : "Recipe Landscape"}</div>
            <div class="ls-sub">${zh ? "按模型代际严格隔离对比，不混排不同参数量模型。" : "Strictly isolated by model track — never mixes different model sizes on one axis."}</div>
          </div>
          <div style="display:flex;gap:0.45rem;flex-wrap:wrap;">${trackToggleHtml}</div>
        </div>
        <div class="rk-empty"><p>${zh ? "当前模型代际与筛选条件下暂无全景数据。" : "No recipes match the current Landscape track & filters."}</p></div>
      </div>
    `;
    return;
  }

  const vals = rows.map((r) => getMetric(r, metric));
  const minVal = Math.min(...vals);
  const maxVal = Math.max(...vals);
  const logMin = Math.floor(Math.log10(Math.max(minVal, 1)));
  const logMax = Math.ceil(Math.log10(Math.max(maxVal, 10)));

  const W = 1040, H = 430;
  const padL = 64, padR = 28, padT = 32, padB = 64;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const colW = plotW / suites.length;

  const yOf = (v) => {
    const lv = Math.log10(Math.max(v, 1));
    const t = (lv - logMin) / Math.max(logMax - logMin, 1);
    return padT + plotH * (1 - t);
  };

  // Highlight best per (base_suite, vendor) with a slightly larger ring
  const bestKeys = new Set();
  for (const s of suites) {
    const byV = new Map();
    for (const r of rows) {
      if ((r.base_suite || baseSuiteId(r.suite)) !== s) continue;
      const v = getMetric(r, metric);
      const cur = byV.get(r.vendor);
      if (!cur || v > cur.v) byV.set(r.vendor, { r, v });
    }
    for (const { r } of byV.values()) bestKeys.add(r.run_id || r.submission);
  }

  // Grid ticks at powers of 10
  const ticks = [];
  for (let p = logMin; p <= logMax; p++) {
    const v = Math.pow(10, p);
    const y = yOf(v);
    const label = v >= 1000 ? `${v / 1000}k` : String(v);
    ticks.push(`
      <line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" class="ls-grid"/>
      <text x="${padL - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="ls-axis mono">${label}</text>
    `);
  }

  const SHORT_SUB = {
    suite_A: "Single-Chip Chat",
    suite_B: "70B Multi-Chip",
    suite_C: "Quantization",
    suite_D: "28K Prefill",
    suite_F: "Edge & Consumer",
    suite_G: "Sparse MoE",
    suite_H: "Sliding Window",
  };
  const colLabels = suites.map((sid, i) => {
    const m = SUITE_META[sid];
    const cx = padL + colW * (i + 0.5);
    const sampleRow = rows.find((r) => (r.base_suite || baseSuiteId(r.suite)) === sid);
    const subText = trackMode === "2026" && sampleRow?.model_track_label
      ? sampleRow.model_track_label
      : (SHORT_SUB[sid] || m.title);
    return `
      <line x1="${cx.toFixed(1)}" y1="${padT}" x2="${cx.toFixed(1)}" y2="${padT + plotH}" class="ls-col-guide"/>
      <text x="${cx.toFixed(1)}" y="${H - 34}" text-anchor="middle" class="ls-suite-lbl">Suite ${esc(m.letter)}</text>
      <text x="${cx.toFixed(1)}" y="${H - 18}" text-anchor="middle" class="ls-suite-sub">${esc(subText)}</text>
    `;
  });

  const dots = rows.map((r) => {
    const base = r.base_suite || baseSuiteId(r.suite);
    const idx = suites.indexOf(base);
    const cx = padL + colW * (idx + 0.5);
    const v = getMetric(r, metric);
    const key = r.run_id || r.submission || r.chip;
    const x = cx + deterministicJitter(`${key}:${r.framework}`, Math.min(colW * 0.56, 62));
    const y = yOf(v);
    const col = VENDOR_COLORS[r.vendor] || "#71717a";
    const isBest = bestKeys.has(key);
    const rad = isBest ? 6.2 : 4.2;
    const unit = metric === "qps" ? "QPS" : (metric === "value" ? "tok/$/hr" : "tok/s");
    const modelTag = r.model_track_label ? ` [${r.model_track_label}]` : "";
    const tip = `${r.framework || ""} · ${r.precision || ""} on ${r._chip_label || r.chip}${modelTag} — ${fmtNum(Math.round(v))} ${unit}`;
    return `
      <circle class="ls-dot${isBest ? " is-best" : ""}"
              cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${rad}"
              fill="${col}" fill-opacity="${isBest ? "0.92" : "0.62"}"
              data-tip="${esc(tip)}"
              data-href="${esc(chipHref(r))}"
              data-open-run="${esc(r.run_id || "")}"
              data-open-tab="deploy"
              tabindex="0" role="button" aria-label="${esc(tip)}"/>
    `;
  });

  const activeVendors = VENDOR_ORDER.filter((v) => rows.some((r) => r.vendor === v));
  const metricLabel = metric === "qps" ? "Online 500ms SLA (QPS)" : (metric === "value" ? "Cost Efficiency (tok/$/hr)" : "Offline Throughput (tok/s)");
  const trackLabel = trackMode === "2026" ? (zh ? "2026 新模型轨道" : "2026 Model Track") : (zh ? "经典基线轨道" : "Classic Baseline Track");

  container.innerHTML = `
    <div class="ls-card">
      <div class="ls-head">
        <div>
          <div class="ls-title">${esc(metricLabel)} · ${esc(trackLabel)}</div>
          <div class="ls-sub">${zh
            ? `对数坐标 · ${fmtNum(rows.length)} 套方案 · 严格按模型轨道隔离，大圆点为各厂商最优。点击圆点直接部署。`
            : `Log scale · ${fmtNum(rows.length)} recipes · Isolated by model track (no mixed model sizes). Click any dot to deploy.`}</div>
        </div>
        <div style="display:flex;gap:0.45rem;flex-wrap:wrap;align-items:center;">
          ${trackToggleHtml}
          <div class="ls-metric-tabs" role="group" aria-label="Landscape metric">
            <button type="button" class="ls-mtab${metric === "offline" ? " active" : ""}" data-ls-metric="offline">${zh ? "吞吐" : "Throughput"}</button>
            <button type="button" class="ls-mtab${metric === "qps" ? " active" : ""}" data-ls-metric="qps">SLA QPS</button>
            <button type="button" class="ls-mtab${metric === "value" ? " active" : ""}" data-ls-metric="value">Tok / $</button>
          </div>
        </div>
      </div>
      <div class="ls-svg-wrap">
        <svg viewBox="0 0 ${W} ${H}" class="ls-svg" role="img" aria-label="Multi-suite recipe distribution">
          ${ticks.join("")}
          ${colLabels.join("")}
          ${dots.join("")}
        </svg>
        <div class="ls-tooltip" hidden></div>
      </div>
      <div class="ls-legend">
        ${activeVendors.map((v) => `
          <span class="ls-leg-item">
            <span class="ls-leg-dot" style="background:${VENDOR_COLORS[v] || "#71717a"}"></span>
            <span>${esc(v)}</span>
          </span>
        `).join("")}
      </div>
    </div>
  `;

  const tipEl = container.querySelector(".ls-tooltip");
  const wrap = container.querySelector(".ls-svg-wrap");

  wrap.addEventListener("mousemove", (e) => {
    const dot = e.target.closest(".ls-dot");
    if (!dot) { tipEl.hidden = true; return; }
    const rect = wrap.getBoundingClientRect();
    tipEl.textContent = dot.dataset.tip;
    tipEl.hidden = false;
    const x = Math.min(Math.max(e.clientX - rect.left + 12, 8), rect.width - 260);
    const y = Math.max(e.clientY - rect.top - 36, 8);
    tipEl.style.left = `${x}px`;
    tipEl.style.top = `${y}px`;
  });
  wrap.addEventListener("mouseleave", () => { tipEl.hidden = true; });
}
