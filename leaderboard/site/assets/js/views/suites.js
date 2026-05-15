// views/suites.js — Suites explorer.  In commit 1 this is already informative
// (lightweight static reference page); commit 4 will add per-suite stats.

import { SUITE_ORDER, SUITE_META, rowsForSuite } from "../data.js";
import { esc, buildHash } from "../utils.js";

export function render({ el }) {
  const cards = SUITE_ORDER.map((id) => {
    const meta = SUITE_META[id];
    const rows = rowsForSuite(id);
    const submissions = rows.length;
    const chips = new Set(rows.map((r) => r._chip_slug)).size;
    return `
      <a class="card suite-card" data-suite="${esc(meta.letter)}"
         href="${buildHash("/rankings", { suite: id })}">
        <div class="suite-card-head">
          <div class="suite-head-left">
            <span class="suite-letter">${esc(meta.letter)}</span>
            <span class="suite-title">${esc(meta.title)}</span>
          </div>
          <span class="suite-metric-tag">${esc(meta.primary.label)}</span>
        </div>
        <div class="suite-card-tag">${esc(meta.tagline)}</div>
        <div class="suite-card-body" style="padding:0.5rem 1.1rem 1rem">
          <div class="suite-stats">
            <span><strong>${submissions}</strong> submission${submissions === 1 ? "" : "s"}</span>
            <span><strong>${chips}</strong> chip${chips === 1 ? "" : "s"}</span>
          </div>
        </div>
      </a>
    `;
  }).join("");

  el.innerHTML = `
    <section class="hero">
      <span class="eyebrow hero-eyebrow">Reference · Suites</span>
      <h1>Benchmark <em>suites</em></h1>
      <p class="tagline">
        Each suite targets a distinct deployment workload. Click into one to
        view its full ranking.
      </p>
    </section>
    <section class="section">
      <div class="grid grid-2">${cards}</div>
    </section>
    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">Methodology</span>
          <h2>Why per-suite, not a single score?</h2>
        </div>
      </div>
      <p class="muted" style="max-width:62ch;line-height:1.7;margin:0;font-size:1rem">
        AI workloads vary widely — a chip optimized for batched offline throughput
        may rank poorly on interactive latency.  Combining all dimensions into a
        single number hides those trade-offs.  AccelMark keeps every suite raw and
        lets you pick the one that matches your deployment.
      </p>
    </section>
  `;
}
