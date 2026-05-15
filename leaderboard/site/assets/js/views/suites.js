// views/suites.js — Suites explorer.  In commit 1 this is already informative
// (lightweight static reference page), and commit 3 will add per-suite stats.

import { SUITE_ORDER, SUITE_META, rowsForSuite } from "../data.js";
import { esc, buildHash } from "../utils.js";

export function render({ el }) {
  const cards = SUITE_ORDER.map((id) => {
    const meta = SUITE_META[id];
    const rows = rowsForSuite(id);
    const submissions = rows.length;
    const chips = new Set(rows.map((r) => r._chip_slug)).size;
    return `
      <a class="card is-link" href="${buildHash("/rankings", { suite: id })}" style="text-decoration:none">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem">
          <div class="suite-letter-big">${meta.letter}</div>
          <strong style="color:var(--fg-strong)">${esc(meta.title)}</strong>
        </div>
        <p class="muted" style="margin:0 0 0.65rem;font-size:0.88rem;line-height:1.45">${esc(meta.tagline)}</p>
        <div class="suite-card-meta">
          <span class="meta-item">${submissions} submission${submissions === 1 ? "" : "s"}</span>
          <span class="meta-item">${chips} chip${chips === 1 ? "" : "s"}</span>
          <span class="meta-item">primary: <span class="mono">${esc(meta.primary.label)}</span></span>
        </div>
      </a>
    `;
  }).join("");

  el.innerHTML = `
    <section class="hero">
      <h1>Benchmark suites</h1>
      <p class="tagline">
        Each suite targets a distinct deployment workload.  Click into one to view its full ranking.
      </p>
    </section>
    <section class="section">
      <div class="grid grid-2">${cards}</div>
    </section>
    <section class="section">
      <div class="section-header"><h2>Why per-suite, not a single score?</h2></div>
      <p class="muted" style="max-width:62ch;line-height:1.6;margin:0">
        AI workloads vary widely — a chip optimized for batched offline throughput
        may rank poorly on interactive latency.  Combining all dimensions into a
        single number hides those trade-offs.  AccelMark keeps every suite raw and
        lets you pick the one that matches your deployment.
      </p>
    </section>
  `;
}
