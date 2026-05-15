// views/chip-detail.js — placeholder for commit 1.  Will be implemented in commit 3.

import { rowsForChip, bestPerSuiteForChip, SUITE_META, formatPrimary } from "../data.js";
import { esc } from "../utils.js";

export function render({ el, params }) {
  const slug = params.slug;
  const rs = rowsForChip(slug);
  if (rs.length === 0) {
    el.innerHTML = `
      <section class="state">
        <span class="state-icon">⚠</span>
        No chip found for slug <code>${esc(slug)}</code>.<br>
        <a class="btn primary" href="#/" style="margin-top:1rem">Back to home</a>
      </section>
    `;
    return;
  }
  const sample = rs[0];
  const bestPerSuite = bestPerSuiteForChip(slug);

  // Minimal but informative placeholder: hero + per-suite tiles.  No charts yet.
  let tiles = "";
  for (const [suiteId, row] of bestPerSuite) {
    const meta = SUITE_META[suiteId];
    if (!meta) continue;
    const v = row[meta.primary.key];
    tiles += `
      <a class="card is-link" href="#/rankings?suite=${suiteId}" style="text-decoration:none">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem">
          <div class="suite-letter-big" style="width:28px;height:28px;font-size:0.85rem">${meta.letter}</div>
          <strong>${esc(meta.title)}</strong>
        </div>
        <div class="kpi-value">${esc(formatPrimary(v, suiteId))}</div>
        <div class="kpi-label">${esc(meta.primary.label)}</div>
      </a>
    `;
  }

  el.innerHTML = `
    <section class="hero">
      <h1>${esc(sample._chip_label)}</h1>
      <p class="tagline">
        ${esc(sample.vendor)} ·
        ${esc(sample.memory_gb || "-")}GB memory ·
        ${rs.length} ${rs.length === 1 ? "run" : "runs"} across ${bestPerSuite.size} suite${bestPerSuite.size === 1 ? "" : "s"}
      </p>
    </section>
    <section class="section">
      <div class="section-header"><h2>Suite results</h2></div>
      <div class="grid grid-3">${tiles || "<div class='state'>No suite results.</div>"}</div>
    </section>
    <section class="state">
      <span class="state-icon">⚙</span>
      Detailed concurrency / latency / memory charts and raw-run table arrive in commit 3.
    </section>
  `;
}
