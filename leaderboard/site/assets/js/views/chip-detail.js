// views/chip-detail.js — Chip-level overview page.
//
// Reachable via `#/chip/<slug>` where the slug encodes a specific chip
// + chip-count variant (e.g. `nvidia-h100-80gb-hbm3-x1` vs `…-x8`).
// Single-run details live in the modal; this page is the place to see
// everything about ONE hardware configuration in aggregate:
//
//   • Hero           : chip name, vendor, memory, suite/run/framework
//                      coverage facts, and a "Compare this chip" CTA
//                      that seeds the basket with the most recent run.
//   • Per-suite grid : one card per suite — the best primary-metric run
//                      the chip has on file there, with a `data-open-run`
//                      handle so a click pops the run modal.  Suites the
//                      chip never submitted to render as dim placeholders.
//   • Runs table     : every submission, sorted newest-first, each row
//                      `data-open-run` for the same modal flow.
//
// Layout primitives (.hero, .section, .grid, .card, .data-table) come
// from the global stylesheet; per-page tweaks live in chip-detail.css.

import {
  SUITE_ORDER, SUITE_META,
  rowsForChip, bestPerSuiteForChip, formatPrimary,
  rankChipInSuite, similarChipsTo,
} from "../data.js";
import {
  esc, fmtDate, shortVersion, submitterHandle,
} from "../utils.js";

export function render({ el, params }) {
  const slug = params.slug;
  const rs = rowsForChip(slug);

  if (rs.length === 0) {
    // Mirror `.rk-empty` / `.cmp-empty` so a stale chip-detail link feels
    // like the same "nothing here" surface as an over-filtered rankings
    // view rather than a dramatic 404.
    el.innerHTML = `
      <section class="chip-empty">
        <span class="state-icon" aria-hidden="true">⚠</span>
        <p>No chip found for <code>${esc(slug)}</code>.</p>
        <p class="chip-empty-sub">It may have been removed, or the link is from an older revision of the dataset.</p>
        <div class="hero-cta" style="justify-content:center;margin-top:1rem">
          <a class="btn primary" href="#/">Back to home</a>
          <a class="btn" href="#/rankings">Browse rankings</a>
        </div>
      </section>
    `;
    return;
  }

  // Pick a sample row for hero attribution (vendor / memory / label) and
  // a separate "latest run" for the Compare CTA so users land on the
  // freshest configuration when they jump to compare.
  const sample = rs[0];
  const latestRun = rs.reduce(
    (a, b) => (String(b.date || "") > String(a.date || "") ? b : a)
  );
  const latestRid = latestRun.run_id || latestRun.submission || "";

  const bestPerSuite = bestPerSuiteForChip(slug);
  const activeSuites = SUITE_ORDER.filter((sid) => bestPerSuite.has(sid));
  const frameworks   = new Set(rs.map((r) => r.framework).filter(Boolean));
  const precisions   = new Set(rs.map((r) => r.precision).filter(Boolean));

  const memoryStr = sample.memory_gb ? `${sample.memory_gb} GB` : "";
  const factPills = [
    `${activeSuites.length} suite${activeSuites.length === 1 ? "" : "s"}`,
    `${rs.length} run${rs.length === 1 ? "" : "s"}`,
    `${frameworks.size} framework${frameworks.size === 1 ? "" : "s"}`,
    `${precisions.size} precision${precisions.size === 1 ? "" : "s"}`,
  ];

  el.innerHTML = `
    <section class="hero chip-hero" data-vendor="${esc(sample.vendor)}">
      <span class="eyebrow chip-hero-eyebrow">
        <span class="vendor-dot" data-vendor="${esc(sample.vendor)}"></span>
        ${esc(sample.vendor)}${memoryStr ? " · " + esc(memoryStr) : ""}
      </span>
      <h1>${esc(sample._chip_label)}</h1>
      <p class="hero-sub">${factPills.map(esc).join(" · ")}</p>
      <div class="hero-cta">
        ${latestRid
          ? `<a class="btn primary" href="#/compare?runs=${encodeURIComponent(latestRid)}">Compare this chip</a>`
          : ""}
        <a class="btn" href="#/rankings?vendor=${encodeURIComponent(sample.vendor)}">Browse ${esc(sample.vendor)} rankings</a>
      </div>
    </section>

    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">01 · Best per suite</span>
          <h2>Where this chip lands</h2>
        </div>
        <p class="section-sub">Top primary-metric run in each suite. Click a card to open its details.</p>
      </div>
      <div class="chip-suite-grid">
        ${SUITE_ORDER.map((sid) => renderSuiteCard(sid, bestPerSuite.get(sid), slug)).join("")}
      </div>
    </section>

    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">02 · Every submission</span>
          <h2>${rs.length} run${rs.length === 1 ? "" : "s"} on file</h2>
        </div>
        <p class="section-sub">Sorted newest first. Click a row to open the run detail.</p>
      </div>
      <div class="chip-runs-wrap">
        ${renderRunsTable(rs)}
      </div>
    </section>

    ${renderSimilarChipsSection(slug, latestRid)}
  `;
}

// ── 03 · Compare with similar chips ──
//
// Surfaces chips with the largest suite-coverage overlap so users have
// a one-click jump-off to peer hardware they'd realistically compare
// against.  Each tile links to that chip's detail page; the section as
// a whole is skipped when the dataset is too sparse for a meaningful
// recommendation (e.g. the chip is the only entry in its suites).
function renderSimilarChipsSection(slug, latestRid) {
  const peers = similarChipsTo(slug, { limit: 6 });
  if (!peers.length) return "";

  const tiles = peers.map((p) => {
    const sharedLetters = p.sharedSuites
      .filter((sid) => SUITE_META[sid])
      .sort((a, b) => SUITE_ORDER.indexOf(a) - SUITE_ORDER.indexOf(b))
      .map((sid) => `
        <span class="chip-peer-suite" data-suite="${esc(SUITE_META[sid].letter)}"
              title="${esc(SUITE_META[sid].title)}">${esc(SUITE_META[sid].letter)}</span>
      `).join("");
    const subL = p.totalRuns === 1 ? "run" : "runs";
    return `
      <a class="chip-peer-card" data-vendor="${esc(p.vendor)}" href="#/chip/${esc(p.slug)}">
        <span class="chip-peer-vendor">
          <span class="vendor-dot" data-vendor="${esc(p.vendor)}"></span>
          ${esc(p.vendor)}
        </span>
        <span class="chip-peer-name">${esc(p.label)}</span>
        <span class="chip-peer-meta">
          <span class="chip-peer-suites">${sharedLetters}</span>
          <span class="chip-peer-runs">${p.totalRuns} ${subL}</span>
        </span>
      </a>
    `;
  }).join("");

  // A "Compare this chip against the strip" shortcut: seed compare with
  // both the source chip and each peer's freshest run.  Cheaper than
  // making every peer card itself a basket toggle and keeps the section
  // a navigation surface rather than a hidden mutation.
  return `
    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">03 · Peers</span>
          <h2>Compare with similar chips</h2>
        </div>
        <p class="section-sub">Chips that compete on the same workload suites — sorted by suite overlap, same-vendor first.</p>
      </div>
      <div class="chip-peer-grid">${tiles}</div>
    </section>
  `;
}

// Per-suite KPI card.  Active suite → linkable card with the run's best
// primary metric, framework / precision / date.  Inactive suite → muted
// placeholder so users can see at a glance which suites the chip skipped.
//
// `chipSlug` threads through so the card's fallback href (Cmd-click /
// middle-click) lands the user on the rankings view already filtered to
// this exact chip variant — saves a few clicks compared to dumping them
// at the unfiltered suite page.
function renderSuiteCard(sid, row, chipSlug) {
  const meta = SUITE_META[sid];
  if (!meta) return "";

  if (!row) {
    return `
      <div class="chip-suite-card chip-suite-card--empty" data-suite="${esc(meta.letter)}">
        <div class="chip-suite-head">
          <span class="chip-suite-letter">${esc(meta.letter)}</span>
          <span class="chip-suite-title">${esc(meta.title)}</span>
        </div>
        <div class="chip-suite-empty">Not submitted</div>
      </div>
    `;
  }

  const value = row[meta.primary.key];
  const display = formatPrimary(value, sid);
  const num   = display ? display.replace(/\s.+$/, "") : "—";
  const unit  = display && /\s/.test(display) ? display.replace(/^[^\s]+\s/, "") : meta.primary.unit;

  const rid = row.run_id || row.submission || "";
  const ver = shortVersion(row.framework_version);
  const fwLine = ver
    ? `${esc(row.framework)} <span class="fw-ver">${esc(ver)}</span>`
    : esc(row.framework || "");

  const rankingsHref = `#/rankings?suite=${encodeURIComponent(sid)}${chipSlug ? `&chip=${encodeURIComponent(chipSlug)}` : ""}`;

  // Rank among unique chips in this suite (not among raw rows — a chip
  // with 4 vLLM versions shouldn't take up 4 ranking slots in its own
  // badge).  Highlight top-3 with the same medal palette as Home / Rankings.
  const rank = chipSlug ? rankChipInSuite(chipSlug, sid) : null;
  const medal = rank
    ? (rank.rank === 1 ? " is-gold"
       : rank.rank === 2 ? " is-silver"
       : rank.rank === 3 ? " is-bronze" : "")
    : "";

  return `
    <a class="chip-suite-card"
       data-suite="${esc(meta.letter)}"
       data-open-run="${esc(rid)}"
       href="${esc(rankingsHref)}">
      <div class="chip-suite-head">
        <span class="chip-suite-letter">${esc(meta.letter)}</span>
        <span class="chip-suite-title">${esc(meta.title)}</span>
        ${rank ? `
          <span class="chip-suite-rank${medal}"
                title="Ranked #${rank.rank} of ${rank.total} chips in Suite ${esc(meta.letter)}">
            #${rank.rank}<span class="chip-suite-rank-total"> / ${rank.total}</span>
          </span>
        ` : ""}
      </div>
      <div class="chip-suite-metric">
        <span class="chip-suite-val">${esc(num)}</span>
        ${unit ? `<span class="chip-suite-unit">${esc(unit)}</span>` : ""}
      </div>
      <div class="chip-suite-meta">
        ${fwLine}${row.precision ? ` · ${esc(row.precision)}` : ""}${row.date ? ` · ${esc(fmtDate(row.date))}` : ""}
      </div>
    </a>
  `;
}

function renderRunsTable(rs) {
  const sorted = rs.slice().sort((a, b) =>
    String(b.date || "").localeCompare(String(a.date || ""))
  );
  return `
    <table class="data-table chip-runs">
      <thead>
        <tr>
          <th class="col-suite">Suite</th>
          <th class="col-framework">Framework</th>
          <th class="col-precision">Precision</th>
          <th class="col-primary">Primary metric</th>
          <th class="col-date">Date</th>
          <th class="col-submitter">Submitter</th>
          <th class="col-tier">Tier</th>
        </tr>
      </thead>
      <tbody>
        ${sorted.map(renderRunRow).join("")}
      </tbody>
    </table>
  `;
}

function renderRunRow(row) {
  const meta = SUITE_META[row.suite];
  const rid = row.run_id || row.submission || "";
  const ver = shortVersion(row.framework_version);
  const fwLine = ver
    ? `${esc(row.framework)} <span class="fw-ver">${esc(ver)}</span>`
    : esc(row.framework || "");
  const v = meta ? row[meta.primary.key] : row.primary_metric;
  const display = meta ? formatPrimary(v, row.suite) : (v != null ? String(v) : "");
  const handle = submitterHandle(row.submitted_by);
  const tierClass = row.tier ? ` tier-${esc(row.tier)}` : "";

  return `
    <tr data-open-run="${esc(rid)}" data-suite="${meta ? esc(meta.letter) : ""}">
      <td class="col-suite">
        <span class="chip-runs-suite">
          <span class="chip-suite-letter chip-suite-letter--inline">${meta ? esc(meta.letter) : "·"}</span>
          <span class="chip-runs-suite-title">${meta ? esc(meta.title) : esc(row.suite || "")}</span>
        </span>
      </td>
      <td class="col-framework">${fwLine}</td>
      <td class="col-precision">${esc(row.precision || "—")}</td>
      <td class="col-primary"><span class="chip-runs-metric">${esc(display || "—")}</span></td>
      <td class="col-date">${esc(fmtDate(row.date))}</td>
      <td class="col-submitter">${handle ? `@${esc(handle)}` : "—"}</td>
      <td class="col-tier"><span class="badge${tierClass}">${esc(row.tier || "—")}</span></td>
    </tr>
  `;
}
