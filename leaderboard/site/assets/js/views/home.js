// views/home.js — Home page (multi-suite overview).
//
// Layout (editorial, per-suite color):
//   • Hero is centered (eyebrow + serif h1 + tagline + KPI strip + CTAs).
//   • Suite grid: 4 compact cards on row 1 (A B C D, span 3 of 12),
//     3 wider cards on row 2 (E F G, span 4 of 12).
//   • Each suite card declares data-suite="A".."G" so CSS scopes
//     --suite-color for the header bar, featured #1 row tint, and CTA.
//   • Recent submissions: same .lb-row, suite letter circle tinted in
//     that submission's suite color.

import {
  SUITE_ORDER, SUITE_META,
  rowsForSuite, bestPerChipForSuite,
  summary, recent, formatPrimary,
} from "../data.js";
import { esc, fmtNum, fmtDate, chipHref, buildHash, shortVersion, submitterHandle } from "../utils.js";

// Row-1 (compact) suites — narrower 25%-ish cards. Anything not here
// goes into row 2 (wider 33% cards).
const COMPACT_SUITES = new Set(["suite_A", "suite_B", "suite_C", "suite_D"]);

export function render({ el }) {
  const s = summary();
  el.innerHTML = `
    <section class="hero">
      <span class="eyebrow hero-eyebrow">AccelMark · Benchmark suite</span>
      <h1>AI accelerator benchmark — <em>open and reproducible</em></h1>
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
      <div class="hero-cta">
        <a class="btn primary" href="#/rankings">Browse rankings →</a>
        <a class="btn" href="#/compare">Compare chips</a>
        <a class="btn ghost" href="#/suites">What are the suites?</a>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">01 · Workloads</span>
          <h2>Rankings by workload</h2>
        </div>
        <span class="section-sub">Each suite measures a different real-world workload. Pick one to dive in.</span>
      </div>
      <div class="suite-grid" id="suite-grid"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">02 · Latest activity</span>
          <h2>Recent submissions</h2>
        </div>
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
  const compact = COMPACT_SUITES.has(suiteId);
  const limit = compact ? 4 : 5;
  const top = bestPerChipForSuite(suiteId).slice(0, limit);
  const empty = top.length === 0;

  const card = document.createElement("article");
  card.className = "card suite-card" + (compact ? " is-compact" : "") + (empty ? " empty" : "");
  card.setAttribute("data-suite", meta.letter);

  const rankingsHref = buildHash("/rankings", { suite: suiteId });

  const header = `
    <div class="suite-card-head">
      <div class="suite-head-left">
        <span class="suite-letter">${esc(meta.letter)}</span>
        <span class="suite-title">${esc(meta.title)}</span>
      </div>
      <span class="suite-metric-tag">${esc(meta.primary.label)}</span>
    </div>
    <div class="suite-card-tag">${esc(meta.tagline)}</div>
  `;

  if (empty) {
    card.innerHTML = `
      ${header}
      <div class="suite-card-body">Awaiting first submission.</div>
      <div class="suite-card-foot"><a class="cta" href="${rankingsHref}">View suite →</a></div>
    `;
    return card;
  }

  const body = top.map((r, i) => renderLbRow(r, suiteId, i + 1, compact)).join("");
  card.innerHTML = `
    ${header}
    <div class="suite-card-body">${body}</div>
    <div class="suite-card-foot"><a class="cta" href="${rankingsHref}">View full ranking →</a></div>
  `;
  return card;
}

// Single ranked row used by suite cards.  Anchor → chip detail page.
// `compact` flag hides the submitter line so narrow row-1 cards stay
// readable; row-2 (wider) cards show it.
function renderLbRow(row, suiteId, rank, compact = false) {
  const meta = SUITE_META[suiteId];
  const value = row[meta.primary.key];
  const display = formatPrimary(value, suiteId);
  const { num, unit } = splitNumUnit(display);
  const medal = rank === 1 ? "gold" : rank === 2 ? "silver" : rank === 3 ? "bronze" : "";
  const featured = rank === 1 ? " lb-row--featured" : "";
  const fwLine = renderFwSub(row, compact);
  return `
    <a class="lb-row${featured}" href="${chipHref(row)}">
      <span class="lb-row-rank ${medal}">${rank}</span>
      <span class="lb-row-main">
        <span class="lb-row-name">${esc(row._chip_label)}</span>
        ${fwLine}
      </span>
      <span class="lb-row-score">
        <span class="score-val">${esc(num)}</span>
        ${unit ? `<span class="score-unit">${esc(unit)}</span>` : ""}
      </span>
    </a>
  `;
}

// Sub line under chip name. Always shows vendor + framework@version +
// precision. Submitter is shown on wider rows only (recent list +
// row-2 suite cards) to keep compact rows tight.
function renderFwSub(row, compact = false) {
  const fw = row.framework || "";
  const ver = shortVersion(row.framework_version);
  const fwVer = ver ? `${esc(fw)} <span class="fw-ver">${esc(ver)}</span>` : esc(fw);
  const precision = row.precision ? ` · ${esc(row.precision)}` : "";
  const handle = submitterHandle(row.submitted_by);
  const submitter = (!compact && handle)
    ? `<span class="sub-sep">·</span><span class="submitter">@${esc(handle)}</span>`
    : "";
  return `
    <span class="lb-row-sub">
      <span class="vendor-dot" data-vendor="${esc(row.vendor)}"></span>
      <span class="vendor-name">${esc(row.vendor)}</span>
      <span class="sub-sep">·</span>
      <span>${fwVer}${precision}</span>
      ${submitter}
    </span>
  `;
}

function renderRecentRow(row) {
  const meta = SUITE_META[row.suite];
  const metricVal = meta ? row[meta.primary.key] : row.primary_metric;
  const display = formatPrimary(metricVal, row.suite);
  const { num, unit } = splitNumUnit(display);
  const suiteLabel = row.suite.replace("suite_", "Suite ");
  const letter = meta ? meta.letter : "·";
  const handle = submitterHandle(row.submitted_by);
  const ver = shortVersion(row.framework_version);
  const fwVer = ver ? `${esc(row.framework)} <span class="fw-ver">${esc(ver)}</span>` : esc(row.framework);

  const a = document.createElement("a");
  a.className = "lb-row";
  a.href = chipHref(row);
  a.setAttribute("data-suite", letter);
  a.innerHTML = `
    <span class="lb-row-rank suite-tag-rank" aria-hidden="true">${esc(letter)}</span>
    <span class="lb-row-main">
      <span class="lb-row-name">${esc(row._chip_label)}</span>
      <span class="lb-row-sub">
        <span class="vendor-dot" data-vendor="${esc(row.vendor)}"></span>
        <span class="vendor-name">${esc(row.vendor)}</span>
        <span class="sub-sep">·</span>
        <span>${fwVer}${row.precision ? " · " + esc(row.precision) : ""}</span>
        <span class="sub-sep">·</span>
        <span class="suite-tag">${esc(suiteLabel)}</span>
        ${handle ? `<span class="sub-sep">·</span><span class="submitter">@${esc(handle)}</span>` : ""}
        <span class="sub-sep">·</span>
        <span class="date">${esc(fmtDate(row.date))}</span>
      </span>
    </span>
    <span class="lb-row-score">
      <span class="score-val">${esc(num)}</span>
      ${unit ? `<span class="score-unit">${esc(unit)}</span>` : ""}
    </span>
  `;
  return a;
}

// "5,731 tok/s" → { num: "5,731", unit: "tok/s" }
// "94.5 %"     → { num: "94.5",  unit: "%" }
// "—"          → { num: "—",     unit: "" }
function splitNumUnit(s) {
  if (!s) return { num: "—", unit: "" };
  const idx = s.search(/\s[A-Za-z%]/);
  if (idx === -1) return { num: s, unit: "" };
  return { num: s.slice(0, idx), unit: s.slice(idx + 1) };
}
