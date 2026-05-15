// views/home.js — Home page (multi-suite overview).
//
// Layout:
//   • Hero  : centered, two-line title (h1 + sub), tagline, KPI strip, CTAs.
//             Background gradient lives on body::before (no hard edges).
//   • 01    : Suite grid — uniform 3-col, 7 cards, each with a colored
//             header (letter + title + metric tag + tagline + meta line:
//             model · precision · N results · M chips) and top-6 entries
//             in the body. CTA at the bottom.
//   • 02    : Coverage by vendor — auto-fit cards, one per vendor.
//   • 03    : Recent submissions — same .lb-row primitive, suite-tinted
//             letter circle in the rank slot.

import {
  SUITE_ORDER, SUITE_META,
  bestPerChipForSuite, suiteFacts, vendorBreakdown,
  summary, recent, formatPrimary,
} from "../data.js";
import {
  esc, fmtNum, fmtDate, chipHref, buildHash,
  shortVersion, shortModel, submitterHandle,
} from "../utils.js";

const TOP_N = 6;

export function render({ el }) {
  const s = summary();
  el.innerHTML = `
    <section class="hero">
      <h1>AccelMark Leaderboard</h1>
      <p class="hero-sub">A Reproducible Multi-Regime AI Accelerator Benchmark</p>
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
        <span class="section-sub">Seven workloads, each on a fixed model and protocol. Pick one to dive in.</span>
      </div>
      <div class="suite-grid" id="suite-grid"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">02 · Coverage</span>
          <h2>Submissions by vendor</h2>
        </div>
        <span class="section-sub">Who shows up, how many chips, and where they compete.</span>
      </div>
      <div class="vendor-grid" id="vendor-grid"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">03 · Latest activity</span>
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

  const vendorGrid = el.querySelector("#vendor-grid");
  for (const v of vendorBreakdown()) {
    vendorGrid.appendChild(renderVendorCard(v));
  }

  const recentEl = el.querySelector("#recent-list");
  for (const row of recent(8)) {
    recentEl.appendChild(renderRecentRow(row));
  }
}

function renderSuiteCard(suiteId) {
  const meta = SUITE_META[suiteId];
  const facts = suiteFacts(suiteId);
  const top = bestPerChipForSuite(suiteId).slice(0, TOP_N);
  const empty = top.length === 0;

  const card = document.createElement("article");
  card.className = "card suite-card" + (empty ? " empty" : "");
  card.setAttribute("data-suite", meta.letter);

  const rankingsHref = buildHash("/rankings", { suite: suiteId });

  const metaLine = renderSuiteMeta(facts);
  const header = `
    <div class="suite-card-head">
      <div class="suite-head-row1">
        <div class="suite-head-left">
          <span class="suite-letter">${esc(meta.letter)}</span>
          <span class="suite-title">${esc(meta.title)}</span>
        </div>
        <span class="suite-metric-tag">${esc(meta.primary.label)}</span>
      </div>
      <p class="suite-head-tagline">${esc(meta.tagline)}</p>
      <div class="suite-head-meta">${metaLine}</div>
    </div>
  `;

  if (empty) {
    card.innerHTML = `
      ${header}
      <div class="suite-card-body">Awaiting first submission.</div>
      <div class="suite-card-foot"><a class="cta" href="${rankingsHref}">View suite →</a></div>
    `;
    return card;
  }

  const body = top.map((r, i) => renderLbRow(r, suiteId, i + 1)).join("");
  card.innerHTML = `
    ${header}
    <div class="suite-card-body">${body}</div>
    <div class="suite-card-foot"><a class="cta" href="${rankingsHref}">View full ranking →</a></div>
  `;
  return card;
}

function renderSuiteMeta(facts) {
  const items = [];
  if (facts.model) {
    items.push(`<span class="meta-item"><strong>${esc(shortModel(facts.model))}</strong></span>`);
  }
  if (facts.precision) {
    items.push(`<span class="meta-item">${esc(facts.precision)} baseline</span>`);
  }
  if (facts.submissions) {
    items.push(`<span class="meta-item"><strong>${fmtNum(facts.submissions)}</strong> results</span>`);
  }
  if (facts.chips) {
    items.push(`<span class="meta-item"><strong>${fmtNum(facts.chips)}</strong> chips</span>`);
  }
  return items.join("");
}

// Single ranked row used by suite cards.  Anchor → chip detail page.
function renderLbRow(row, suiteId, rank) {
  const meta = SUITE_META[suiteId];
  const value = row[meta.primary.key];
  const display = formatPrimary(value, suiteId);
  const { num, unit } = splitNumUnit(display);
  const medal = rank === 1 ? "gold" : rank === 2 ? "silver" : rank === 3 ? "bronze" : "";
  const featured = rank === 1 ? " lb-row--featured" : "";
  return `
    <a class="lb-row${featured}" href="${chipHref(row)}">
      <span class="lb-row-rank ${medal}">${rank}</span>
      <span class="lb-row-main">
        <span class="lb-row-name">${esc(row._chip_label)}</span>
        ${renderFwSub(row)}
      </span>
      <span class="lb-row-score">
        <span class="score-val">${esc(num)}</span>
        ${unit ? `<span class="score-unit">${esc(unit)}</span>` : ""}
      </span>
    </a>
  `;
}

// Sub line under chip name: vendor + framework@version + precision + submitter.
function renderFwSub(row) {
  const fw = row.framework || "";
  const ver = shortVersion(row.framework_version);
  const fwVer = ver ? `${esc(fw)} <span class="fw-ver">${esc(ver)}</span>` : esc(fw);
  const precision = row.precision ? ` · ${esc(row.precision)}` : "";
  const handle = submitterHandle(row.submitted_by);
  const submitter = handle
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

function renderVendorCard(v) {
  const div = document.createElement("article");
  div.className = "card vendor-card";
  div.setAttribute("data-vendor", v.vendor);

  const topMeta = v.topRow && v.topSuite ? SUITE_META[v.topSuite] : null;
  const topScore = topMeta && v.topRow
    ? formatPrimary(v.topRow[topMeta.primary.key], v.topSuite)
    : null;
  const topLine = v.topRow
    ? `${esc(v.topRow._chip_label)}${topScore ? `<span class="muted"> · ${esc(topScore)}</span>` : ""}`
    : `<span class="muted">No submissions yet</span>`;

  const pills = v.suites.map((l) => `
    <span class="vendor-suite-pill" data-suite="${esc(l)}" title="Suite ${esc(l)}">${esc(l)}</span>
  `).join("");

  div.innerHTML = `
    <div class="vendor-head">
      <span class="vendor-name-main">${esc(v.vendor)}</span>
    </div>
    <div class="vendor-stats">
      <div class="stat"><strong>${fmtNum(v.chips)}</strong>chips</div>
      <div class="stat"><strong>${fmtNum(v.submissions)}</strong>submissions</div>
    </div>
    <div class="vendor-best">
      <span class="vendor-best-label">Top entry</span>
      <span class="vendor-best-chip">${topLine}</span>
    </div>
    <div class="vendor-suites" aria-label="Suites this vendor appears in">${pills}</div>
  `;
  return div;
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
function splitNumUnit(s) {
  if (!s) return { num: "—", unit: "" };
  const idx = s.search(/\s[A-Za-z%]/);
  if (idx === -1) return { num: s, unit: "" };
  return { num: s.slice(0, idx), unit: s.slice(idx + 1) };
}
