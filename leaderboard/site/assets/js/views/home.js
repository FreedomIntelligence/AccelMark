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
  bestPerChipForSuite, suiteFacts, chipCloudData,
  summary, recent, recentSince, formatPrimary,
} from "../data.js";
import {
  esc, fmtNum, fmtDate, chipHref, buildHash,
  shortVersion, shortModel, submitterHandle,
} from "../utils.js";

const TOP_N = 8;
const RECENT_WINDOW_DAYS = 7;

export function render({ el }) {
  const s = summary();
  // 7-day momentum stat for the hero strip + standalone activity
  // ribbon below the KPIs.  The ribbon is suppressed when zero so
  // post-launch quiet weeks don't read as a regression to viewers.
  const recentCount = recentSince(RECENT_WINDOW_DAYS);
  const recentRibbon = recentCount > 0
    ? renderRecentRibbon(recentCount)
    : "";
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
        ${recentCount > 0 ? `
          <div class="kpi kpi--fresh">
            <span class="kpi-value">${fmtNum(recentCount)}</span>
            <span class="kpi-label">this week</span>
          </div>
        ` : ""}
      </div>
      ${recentRibbon}
      <div class="hero-cta">
        <a class="btn primary" href="#/rankings">Browse rankings →</a>
        <a class="btn" href="#/compare">Compare chips</a>
        <a class="btn" href="#/suites">What are the suites?</a>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">01 · Workloads</span>
          <h2>Rankings by workload</h2>
        </div>
        <span class="section-sub">Each suite is a fixed model + protocol. Pick one to dive in.</span>
      </div>
      <div class="suite-grid" id="suite-grid"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">02 · Coverage</span>
          <h2>Chips on the leaderboard</h2>
        </div>
        <span class="section-sub">Tile size = submission count. Color = vendor.</span>
      </div>
      <div class="chip-cloud" id="chip-cloud"></div>
      <div class="cloud-legend" id="cloud-legend"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">03 · Latest activity</span>
          <h2>Recent submissions</h2>
        </div>
        <a class="btn small" href="#/rankings">See all →</a>
      </div>
      <div class="recent-list" id="recent-list"></div>
    </section>

    <section class="section submit-section">
      <div class="submit-card">
        <span class="eyebrow">04 · Contribute</span>
        <h2 class="submit-title">Submit your result</h2>
        <p class="submit-body">
          Benchmark your hardware with a runner script, open a pull request,
          and CI re-runs the validation suite before a maintainer reviews. Once
          merged, your result lands in the leaderboard within minutes.
        </p>
        <div class="submit-cta">
          <a class="btn primary"
             href="https://github.com/JuhaoLiang1997/AccelMark/blob/main/CONTRIBUTING.md"
             target="_blank" rel="noopener">Contributor guide →</a>
          <a class="btn"
             href="https://github.com/JuhaoLiang1997/AccelMark/issues/new?template=submission.md"
             target="_blank" rel="noopener">Open a submission</a>
        </div>
      </div>
    </section>
  `;

  const grid = el.querySelector("#suite-grid");
  for (const suiteId of SUITE_ORDER) {
    grid.appendChild(renderSuiteCard(suiteId));
  }

  const cloud = el.querySelector("#chip-cloud");
  const legend = el.querySelector("#cloud-legend");
  renderChipCloud(cloud, legend);

  const recentEl = el.querySelector("#recent-list");
  for (const row of recent(8)) {
    recentEl.appendChild(renderRecentRow(row));
  }
}

// Hero ribbon directly below the KPI strip.  Tells contributors that
// fresh submissions land on the front page (not buried in a per-suite
// view) and gives a one-click way to scan the latest activity sorted
// by date across the canonical default suite.
function renderRecentRibbon(n) {
  const label = n === 1 ? "submission" : "submissions";
  const href = buildHash("/rankings", { suite: SUITE_ORDER[0], sort: "date:desc" });
  return `
    <a class="hero-recent-ribbon"
       href="${esc(href)}"
       title="Open rankings sorted by submission date.">
      <span class="hero-recent-dot" aria-hidden="true"></span>
      <span class="hero-recent-text">
        <strong>${fmtNum(n)}</strong> new ${label}
        <span class="hero-recent-window">in the last ${RECENT_WINDOW_DAYS} days</span>
      </span>
      <span class="hero-recent-cta" aria-hidden="true">See latest →</span>
    </a>
  `;
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

  const metaLine = renderSuiteMeta(suiteId, facts);
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

function renderSuiteMeta(suiteId, facts) {
  const wl = SUITE_META[suiteId] && SUITE_META[suiteId].workload;
  const items = [];
  if (facts.model) {
    items.push(`<span class="meta-item"><strong>${esc(shortModel(facts.model))}</strong></span>`);
  }
  if (facts.precision) {
    items.push(`<span class="meta-item">${esc(facts.precision)} baseline</span>`);
  }
  if (wl && wl.inputTokens && wl.outputTokens) {
    items.push(`<span class="meta-item">${esc(wl.inputTokens)} → ${esc(wl.outputTokens)} tok</span>`);
  }
  if (facts.submissions) {
    items.push(`<span class="meta-item"><strong>${fmtNum(facts.submissions)}</strong> results</span>`);
  }
  if (facts.chips) {
    items.push(`<span class="meta-item"><strong>${fmtNum(facts.chips)}</strong> chips</span>`);
  }
  return items.join("");
}

// Single ranked row used by suite cards.  Plain click → run modal.
// Anchor href stays pointed at the chip detail page so Cmd-click still
// opens the chip-level overview in a new tab.
function renderLbRow(row, suiteId, rank) {
  const meta = SUITE_META[suiteId];
  const value = row[meta.primary.key];
  const display = formatPrimary(value, suiteId);
  const { num, unit } = splitNumUnit(display);
  const medal = rank === 1 ? "gold" : rank === 2 ? "silver" : rank === 3 ? "bronze" : "";
  const featured = rank === 1 ? " lb-row--featured" : "";
  const runId = row.run_id || row.submission || "";
  // Row anatomy: the surrounding <div> is the run-modal trigger; the
  // chip name is a nested <a> that the modal listener lets through to
  // its native href, so users get two distinct affordances on one row:
  //   • click on chip name        → /chip/<slug> overview page
  //   • click anywhere else in row → run-detail modal
  return `
    <div class="lb-row${featured}" data-open-run="${esc(runId)}">
      <span class="lb-row-rank ${medal}">${rank}</span>
      <span class="lb-row-main">
        <a class="lb-row-name" href="${chipHref(row)}">${esc(row._chip_label)}</a>
        ${renderFwSub(row)}
      </span>
      <span class="lb-row-score">
        <span class="score-val">${esc(num)}</span>
        ${unit ? `<span class="score-unit">${esc(unit)}</span>` : ""}
      </span>
    </div>
  `;
}

// Sub block under chip name — two predictable lines so every row in
// a suite card lines up the same way regardless of framework / version
// string length:
//   line 1: vendor · framework version · precision
//   line 2: @submitter (omitted if absent)
function renderFwSub(row) {
  const fw = row.framework || "";
  const ver = shortVersion(row.framework_version);
  const fwVer = ver ? `${esc(fw)} <span class="fw-ver">${esc(ver)}</span>` : esc(fw);
  const precision = row.precision ? ` · ${esc(row.precision)}` : "";
  const handle = submitterHandle(row.submitted_by);
  const byline = handle
    ? `<span class="lb-row-byline">@${esc(handle)}</span>`
    : "";
  return `
    <span class="lb-row-sub">
      <span class="vendor-dot" data-vendor="${esc(row.vendor)}"></span>
      <span class="vendor-name">${esc(row.vendor)}</span>
      <span class="sub-sep">·</span>
      <span class="fw-line">${fwVer}${precision}</span>
    </span>
    ${byline}
  `;
}

function renderChipCloud(container, legendEl) {
  const chips = chipCloudData();
  for (const c of chips) {
    const a = document.createElement("a");
    a.className = `chip-tile size-${c.size}`;
    // Each tile is a navigational link to the chip's overview page.
    // From there users can drill into any specific run or jump to
    // Compare with this chip pre-selected, but the home landing stays
    // a "browse chips" experience rather than a "start comparing"
    // funnel — matches user expectations for a clickable chip name.
    a.href = `#/chip/${c.slug}`;
    a.setAttribute("data-vendor", c.vendor);
    const subL = c.submissions === 1 ? "submission" : "submissions";
    const suiteL = c.suites.length === 1 ? "suite" : "suites";
    const variantPart = c.variants > 1 ? ` · ${c.variants} chip-count variants` : "";
    a.setAttribute("title",
      `${c.label}: ${c.submissions} ${subL} across ` +
      `${c.suites.length} ${suiteL}${variantPart}`);
    a.innerHTML = `
      <span class="chip-tile-name">${esc(c.label)}</span>
      <span class="chip-tile-count">${fmtNum(c.submissions)}</span>
    `;
    container.appendChild(a);
  }

  // Vendor legend below the cloud
  if (!legendEl) return;
  const byVendor = new Map();
  for (const c of chips) {
    const v = byVendor.get(c.vendor) || { vendor: c.vendor, chips: 0, submissions: 0 };
    v.chips += 1;
    v.submissions += c.submissions;
    byVendor.set(c.vendor, v);
  }
  const vendors = Array.from(byVendor.values()).sort((a, b) => b.submissions - a.submissions);
  legendEl.innerHTML = vendors.map((v) => {
    const chipsLbl = v.chips === 1 ? "chip" : "chips";
    const subsLbl  = v.submissions === 1 ? "result" : "results";
    return `
      <span class="cloud-legend-item" data-vendor="${esc(v.vendor)}">
        <span class="dot"></span>
        <span class="name">${esc(v.vendor)}</span>
        <span class="meta">${fmtNum(v.chips)} ${chipsLbl} · ${fmtNum(v.submissions)} ${subsLbl}</span>
      </span>
    `;
  }).join("");
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

  // Mirrors renderLbRow: outer <div> = run-modal trigger, inner chip-name
  // <a> escapes via modal.js nested-anchor rule to navigate to /chip/<slug>.
  const wrap = document.createElement("div");
  wrap.className = "lb-row";
  wrap.setAttribute("data-suite", letter);
  wrap.setAttribute("data-open-run", row.run_id || row.submission || "");
  const bylineBits = [];
  if (handle) bylineBits.push(`@${esc(handle)}`);
  bylineBits.push(esc(fmtDate(row.date)));
  const byline = `<span class="lb-row-byline">${bylineBits.join(" · ")}</span>`;
  wrap.innerHTML = `
    <span class="lb-row-rank suite-tag-rank" aria-hidden="true">${esc(letter)}</span>
    <span class="lb-row-main">
      <a class="lb-row-name" href="${chipHref(row)}">${esc(row._chip_label)}</a>
      <span class="lb-row-sub">
        <span class="vendor-dot" data-vendor="${esc(row.vendor)}"></span>
        <span class="vendor-name">${esc(row.vendor)}</span>
        <span class="sub-sep">·</span>
        <span class="fw-line">${fwVer}${row.precision ? " · " + esc(row.precision) : ""}</span>
        <span class="sub-sep">·</span>
        <span class="suite-tag">${esc(suiteLabel)}</span>
      </span>
      ${byline}
    </span>
    <span class="lb-row-score">
      <span class="score-val">${esc(num)}</span>
      ${unit ? `<span class="score-unit">${esc(unit)}</span>` : ""}
    </span>
  `;
  return wrap;
}

// "5,731 tok/s" → { num: "5,731", unit: "tok/s" }
function splitNumUnit(s) {
  if (!s) return { num: "-", unit: "" };
  const idx = s.search(/\s[A-Za-z%]/);
  if (idx === -1) return { num: s, unit: "" };
  return { num: s.slice(0, idx), unit: s.slice(idx + 1) };
}
