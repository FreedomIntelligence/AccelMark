// components/landscape.js — Zero-CDN interactive SVG multi-suite Recipe Landscape.
// Mounted inside Leaderboard → Landscape view; adapts instantly to light/dark theme.

import { SUITE_ORDER, SUITE_META, VENDOR_ORDER, VENDOR_COLORS, allRows } from "../data.js";
import { esc, fmtNum, chipHref } from "../utils.js";

const LANDSCAPE_SUITES = ["suite_A", "suite_B", "suite_C", "suite_D", "suite_F", "suite_G", "suite_H"];

function getMetric(row, mode) {
  if (mode === "qps") return row.online_max_qps != null && row.online_max_qps > 0 ? row.online_max_qps : null;
  if (mode === "value") return row.cost_efficiency_toks_per_dollar_hr > 0 ? row.cost_efficiency_toks_per_dollar_hr : null;
  if (row.suite === "suite_C") return row.quant_best_throughput > 0 ? row.quant_best_throughput : (row.primary_metric > 0 ? row.primary_metric : null);
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

export function renderLandscape(container, { closedOnly = true, vendor = "", framework = "", metric = "offline" } = {}) {
  if (!container) return;
  const rows = allRows().filter((r) => {
    if (r.flagged) return false;
    if (!LANDSCAPE_SUITES.includes(r.suite)) return false;
    if (closedOnly && (r.runner_tier || "reference") !== "reference") return false;
    if (vendor && r.vendor !== vendor) return false;
    if (framework && r.framework !== framework) return false;
    return getMetric(r, metric) !== null;
  });

  const suites = LANDSCAPE_SUITES.filter((s) => rows.some((r) => r.suite === s));
  if (!rows.length || !suites.length) {
    container.innerHTML = `<div class="rk-empty"><p>No recipes match the current Landscape filters.</p></div>`;
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

  // Highlight best per (suite, vendor) with a slightly larger ring
  const bestKeys = new Set();
  for (const s of suites) {
    const byV = new Map();
    for (const r of rows) {
      if (r.suite !== s) continue;
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
    return `
      <line x1="${cx.toFixed(1)}" y1="${padT}" x2="${cx.toFixed(1)}" y2="${padT + plotH}" class="ls-col-guide"/>
      <text x="${cx.toFixed(1)}" y="${H - 34}" text-anchor="middle" class="ls-suite-lbl">Suite ${esc(m.letter)}</text>
      <text x="${cx.toFixed(1)}" y="${H - 18}" text-anchor="middle" class="ls-suite-sub">${esc(SHORT_SUB[sid] || m.title)}</text>
    `;
  });

  const dots = rows.map((r) => {
    const idx = suites.indexOf(r.suite);
    const cx = padL + colW * (idx + 0.5);
    const v = getMetric(r, metric);
    const key = r.run_id || r.submission || r.chip;
    const x = cx + deterministicJitter(`${key}:${r.framework}`, Math.min(colW * 0.56, 62));
    const y = yOf(v);
    const col = VENDOR_COLORS[r.vendor] || "#71717a";
    const isBest = bestKeys.has(key);
    const rad = isBest ? 6.2 : 4.2;
    const unit = metric === "qps" ? "QPS" : (metric === "value" ? "tok/$/hr" : "tok/s");
    const tip = `${r._chip_label || r.chip} · ${r.framework || ""} · ${r.precision || ""} — ${fmtNum(Math.round(v))} ${unit}`;
    return `
      <circle class="ls-dot${isBest ? " is-best" : ""}"
              cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${rad}"
              fill="${col}" fill-opacity="${isBest ? "0.92" : "0.62"}"
              data-tip="${esc(tip)}"
              data-href="${esc(chipHref(r))}"
              data-run-id="${esc(r.run_id || "")}"
              tabindex="0" role="button" aria-label="${esc(tip)}"/>
    `;
  });

  const activeVendors = VENDOR_ORDER.filter((v) => rows.some((r) => r.vendor === v));
  const metricLabel = metric === "qps" ? "Online 500ms SLA (QPS)" : (metric === "value" ? "Cost Efficiency (tok/$/hr)" : "Offline Throughput (tok/s)");

  container.innerHTML = `
    <div class="ls-card">
      <div class="ls-head">
        <div>
          <div class="ls-title">${esc(metricLabel)} across ${suites.length} Suites</div>
          <div class="ls-sub">Log scale · ${fmtNum(rows.length)} recipes · Larger dots mark best-in-class per vendor. Hover to inspect, click to open run.</div>
        </div>
        <div class="ls-metric-tabs" role="group" aria-label="Landscape metric">
          <button type="button" class="ls-mtab${metric === "offline" ? " active" : ""}" data-ls-metric="offline">Throughput</button>
          <button type="button" class="ls-mtab${metric === "qps" ? " active" : ""}" data-ls-metric="qps">SLA QPS</button>
          <button type="button" class="ls-mtab${metric === "value" ? " active" : ""}" data-ls-metric="value">Tok / $</button>
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
  wrap.addEventListener("click", (e) => {
    const dot = e.target.closest(".ls-dot");
    if (!dot) return;
    if (dot.dataset.href) location.hash = dot.dataset.href;
  });
}
