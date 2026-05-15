// views/suites.js — Suites explorer.  In commit 1 this is already informative
// (lightweight static reference page); commit 4 will add per-suite stats.

import { SUITE_ORDER, SUITE_META, suiteFacts } from "../data.js";
import { esc, buildHash, shortModel, fmtNum } from "../utils.js";

export function render({ el }) {
  const cards = SUITE_ORDER.map((id) => {
    const meta = SUITE_META[id];
    const facts = suiteFacts(id);
    return `
      <a class="card suite-card" data-suite="${esc(meta.letter)}"
         href="${buildHash("/rankings", { suite: id })}">
        <div class="suite-card-head">
          <div class="suite-head-row1">
            <div class="suite-head-left">
              <span class="suite-letter">${esc(meta.letter)}</span>
              <span class="suite-title">${esc(meta.title)}</span>
            </div>
            <span class="suite-metric-tag">${esc(meta.primary.label)}</span>
          </div>
          <p class="suite-head-tagline">${esc(meta.tagline)}</p>
          <div class="suite-head-meta">
            ${facts.model    ? `<span class="meta-item"><strong>${esc(shortModel(facts.model))}</strong></span>` : ""}
            ${facts.precision ? `<span class="meta-item">${esc(facts.precision)} baseline</span>` : ""}
            <span class="meta-item"><strong>${fmtNum(facts.submissions)}</strong> results</span>
            <span class="meta-item"><strong>${fmtNum(facts.chips)}</strong> chips</span>
          </div>
        </div>
      </a>
    `;
  }).join("");

  el.innerHTML = `
    <section class="hero">
      <h1>Benchmark Suites</h1>
      <p class="hero-sub">Seven workloads, each on a fixed model and protocol</p>
      <p class="tagline">
        Each suite targets a distinct deployment regime. Click into one to view
        its full ranking.
      </p>
    </section>
    <section class="section">
      <div class="suite-grid">${cards}</div>
    </section>
    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">Methodology</span>
          <h2>Why per-suite, not a single score?</h2>
        </div>
      </div>
      <p class="muted" style="max-width:62ch;line-height:1.7;margin:0;font-size:1rem">
        AI workloads vary widely. A chip optimized for batched offline throughput
        may rank poorly on interactive latency. Combining every dimension into a
        single number hides those trade-offs. AccelMark keeps every suite raw and
        lets you pick the one that matches your deployment.
      </p>
    </section>
  `;
}
