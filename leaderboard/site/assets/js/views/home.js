// views/home.js — Home page (multi-suite overview).

import {
  SUITE_ORDER, SUITE_META,
  rowsForSuite, bestPerChipForSuite,
  summary, recent, formatPrimary,
} from "../data.js";
import { esc, fmtNum, fmtDate, chipHref, buildHash } from "../utils.js";

export function render({ el }) {
  const s = summary();
  el.innerHTML = `
    <section class="hero">
      <h1>AI accelerator benchmark — open and reproducible</h1>
      <p class="tagline">
        Independent measurements of inference performance across vendors.
        Every result links back to the runner code that produced it.
      </p>
      <div class="hero-stats">
        <div class="kpi"><span class="kpi-value">${fmtNum(s.total)}</span><span class="kpi-label">results</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.chips)}</span><span class="kpi-label">chip configs</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.vendors)}</span><span class="kpi-label">vendors</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.suites)}</span><span class="kpi-label">suites</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.verified)}</span><span class="kpi-label">verified</span></div>
      </div>
      <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.5rem;">
        <a class="btn primary" href="#/rankings">Browse rankings →</a>
        <a class="btn" href="#/compare">Compare chips</a>
        <a class="btn ghost" href="#/suites">What are the suites?</a>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Rankings by workload</h2>
        <span class="section-sub">Each suite measures a different real-world workload. Pick one to dive in.</span>
      </div>
      <div class="grid grid-3" id="suite-grid"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Recent submissions</h2>
        <a class="btn ghost small" href="#/rankings">See all →</a>
      </div>
      <div class="recent-list" id="recent-list"></div>
    </section>
  `;

  const grid = el.querySelector("#suite-grid");
  for (const suiteId of SUITE_ORDER) {
    grid.appendChild(renderSuiteCard(suiteId));
  }

  const recentEl = el.querySelector("#recent-list");
  for (const row of recent(8)) {
    recentEl.appendChild(renderRecentRow(row));
  }
}

function renderSuiteCard(suiteId) {
  const meta = SUITE_META[suiteId];
  const all = rowsForSuite(suiteId);
  const podiumRows = bestPerChipForSuite(suiteId).slice(0, 3);
  const direction = meta.primary.direction;
  const submissions = all.length;
  const chipsCovered = new Set(all.map((r) => r._chip_slug)).size;

  const article = document.createElement("a");
  article.href = buildHash("/rankings", { suite: suiteId });
  article.className = "card is-link suite-card" + (podiumRows.length === 0 ? " empty" : "");

  if (podiumRows.length === 0) {
    article.innerHTML = `
      <div class="suite-card-header">
        <div class="suite-letter-big">${meta.letter}</div>
        <div>
          <div class="suite-title">${esc(meta.title)}</div>
          <div class="suite-tagline">${esc(meta.tagline)}</div>
        </div>
      </div>
      <div class="suite-card-meta">
        <span class="meta-item">No submissions yet</span>
      </div>
      <div class="podium">Awaiting first submission.</div>
      <span class="suite-card-cta">View suite →</span>
    `;
    return article;
  }

  const best = podiumRows[0];
  const bestVal = best[meta.primary.key];
  // Range for relative bars: pick min/max across podium.
  const vals = podiumRows.map((r) => r[meta.primary.key]);
  const max = Math.max(...vals);
  const min = Math.min(...vals);

  const podiumHtml = podiumRows.map((r, i) => {
    const v = r[meta.primary.key];
    const fillPct = direction === "asc"
      // For "lower is better" (e.g. latency), best gets full bar.
      ? (max === min ? 100 : ((max - v) / (max - min)) * 100)
      : (max === 0 ? 0 : (v / max) * 100);
    const medal = ["gold", "silver", "bronze"][i] || "";
    return `
      <li>
        <div class="podium-row">
          <span class="rank ${medal}">#${i + 1}</span>
          <span class="podium-name">${esc(r._chip_label)}</span>
          <span class="podium-value">${esc(formatPrimary(v, suiteId))}</span>
        </div>
        <div class="rel-bar"><div class="fill" style="width:${Math.max(4, Math.min(100, fillPct))}%"></div></div>
      </li>
    `;
  }).join("");

  article.innerHTML = `
    <div class="suite-card-header">
      <div class="suite-letter-big">${meta.letter}</div>
      <div>
        <div class="suite-title">${esc(meta.title)}</div>
        <div class="suite-tagline">${esc(meta.tagline)}</div>
      </div>
    </div>
    <div class="suite-card-meta">
      <span class="meta-item">${submissions} ${submissions === 1 ? "submission" : "submissions"}</span>
      <span class="meta-item">${chipsCovered} ${chipsCovered === 1 ? "chip" : "chips"}</span>
      <span class="meta-item">primary: <span class="mono">${esc(meta.primary.label)}</span></span>
    </div>
    <ul class="podium">${podiumHtml}</ul>
    <span class="suite-card-cta">View full ranking →</span>
  `;
  // Suppress underline on hover for the whole card.
  article.style.color = "inherit";
  return article;
}

function renderRecentRow(row) {
  const meta = SUITE_META[row.suite];
  const a = document.createElement("a");
  a.href = chipHref(row);
  a.className = "recent-row";
  const metricVal = meta ? row[meta.primary.key] : row.primary_metric;
  a.innerHTML = `
    <span><strong>${esc(row._chip_label)}</strong></span>
    <span class="recent-suite">${esc(row.suite.replace("suite_", "Suite "))}</span>
    <span class="recent-metric">${esc(formatPrimary(metricVal, row.suite))}</span>
    <span class="muted">${esc(row.framework)} ${esc(row.framework_version || "")} · ${esc(row.precision)}</span>
    <span class="recent-date">${fmtDate(row.date)}</span>
  `;
  return a;
}
