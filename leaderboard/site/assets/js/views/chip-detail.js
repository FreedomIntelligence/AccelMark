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
  rankChipInSuite, similarChipsTo, chipCountsForChip,
  suiteFingerprint, chipCountScaling, vendorColor,
} from "../data.js";
import {
  esc, fmtDate, shortVersion, submitterHandle,
  copyToClipboard, flashButtonLabel, downloadCanvasAsPng,
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

  // Pick a sample row for hero attribution (vendor / memory / model
  // name) and a separate "latest run" for the Compare CTA so users land
  // on the freshest configuration when they jump to compare.  We use
  // `sample.chip` (not `_chip_label`) for the hero h1 because the page
  // is now a chip-model page; ×1/×4/×8 variants live inside.
  const sample = rs[0];
  const latestRun = rs.reduce(
    (a, b) => (String(b.date || "") > String(a.date || "") ? b : a)
  );
  const latestRid = latestRun.run_id || latestRun.submission || "";

  const bestPerSuite = bestPerSuiteForChip(slug);
  const activeSuites = SUITE_ORDER.filter((sid) => bestPerSuite.has(sid));
  const frameworks   = new Set(rs.map((r) => r.framework).filter(Boolean));
  const precisions   = new Set(rs.map((r) => r.precision).filter(Boolean));
  const chipCounts   = chipCountsForChip(slug);

  // Section numbering — scaling section (03) is conditional on the
  // chip having ≥2 chip_count variants with data on at least one
  // shared suite.  Precompute once so we can renumber every
  // downstream eyebrow consistently and we only call the data helper
  // once per render.
  const scalingHtml = renderScalingSection(slug, sample);
  const runsNum  = scalingHtml ? "04" : "03";
  const peersNum = scalingHtml ? "05" : "04";

  const memoryStr = sample.memory_gb ? `${sample.memory_gb} GB` : "";
  const factPills = [
    `${activeSuites.length} suite${activeSuites.length === 1 ? "" : "s"}`,
    `${rs.length} run${rs.length === 1 ? "" : "s"}`,
    `${frameworks.size} framework${frameworks.size === 1 ? "" : "s"}`,
    `${precisions.size} precision${precisions.size === 1 ? "" : "s"}`,
  ];
  // Chip-count fact only adds noise for single-variant chips; only
  // surface it when the chip has been deployed at >1 fan-out.
  if (chipCounts.length > 1) {
    factPills.push(`${chipCounts.length} chip-count variants (${chipCounts.map((c) => `×${c}`).join(", ")})`);
  } else if (chipCounts.length === 1 && chipCounts[0] > 1) {
    factPills.push(`deployed at ×${chipCounts[0]}`);
  }

  el.innerHTML = `
    <section class="hero chip-hero" data-vendor="${esc(sample.vendor)}">
      <span class="eyebrow chip-hero-eyebrow">
        <span class="vendor-dot" data-vendor="${esc(sample.vendor)}"></span>
        ${esc(sample.vendor)}${memoryStr ? " · " + esc(memoryStr) : ""}
      </span>
      <h1>${esc(sample.chip)}</h1>
      <p class="hero-sub">${factPills.map(esc).join(" · ")}</p>
      <div class="hero-cta">
        ${latestRid
          ? `<a class="btn primary" href="#/compare?runs=${encodeURIComponent(latestRid)}">Compare this chip</a>`
          : ""}
        <a class="btn" href="#/rankings?vendor=${encodeURIComponent(sample.vendor)}">Browse ${esc(sample.vendor)} rankings</a>
        <button class="btn copy-btn chip-share-btn"
                type="button"
                data-chip-share="1"
                title="Copy a link to this chip's overview page.">
          <span class="copy-btn-icon" aria-hidden="true">↗</span>
          <span class="copy-btn-label">Copy link</span>
        </button>
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

    ${renderFingerprintSection(slug, sample)}

    ${scalingHtml}

    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">${runsNum} · Every submission</span>
          <h2>${rs.length} run${rs.length === 1 ? "" : "s"} on file</h2>
        </div>
        <p class="section-sub">Sorted newest first. Click a row to open the run detail.</p>
      </div>
      <div class="chip-runs-wrap">
        ${renderRunsTable(rs)}
      </div>
    </section>

    ${renderSimilarChipsSection(slug, latestRid, peersNum)}
  `;

  bindClicks(el);
  // Mount Chart.js instances after innerHTML lands so the canvases
  // are attached.  setTimeout(0) instead of rAF so the first paint
  // already includes the charts (no layout flash).
  setTimeout(() => {
    _mountFingerprintChart(el, slug, sample);
    _mountScalingChart(el, slug, sample);
  }, 0);
}

// Click delegation — once-attached on the view container.  The router
// rebuilds `el.innerHTML` on every visit so handlers must live on the
// container itself; the guard flag prevents listener stacking on
// re-renders.  Currently scoped to the hero share button; expand here
// when other in-view affordances need view-local logic (anything that
// runs before modal.js's document delegation should land here too).
function bindClicks(el) {
  if (el.__chipDetailClicksAttached) return;
  el.__chipDetailClicksAttached = true;

  el.addEventListener("click", (ev) => {
    if (!location.hash.startsWith("#/chip/")) return;
    const t = ev.target;

    // Hero share button — copy a clean URL to this chip's overview
    // page (no query string, so the recipient lands on the canonical
    // view rather than a filtered slice).
    const shareBtn = t.closest("[data-chip-share]");
    if (shareBtn) {
      ev.preventDefault();
      _copyChipShareLink(shareBtn);
      return;
    }

    // Download chart as PNG.  Each download button sits inside the
    // chart's canvas wrapper; we resolve the canvas via the data
    // attribute the wrapper template owns, so adding a new chart
    // just needs the wrapper + the matching button — no rebinding.
    const dlBtn = t.closest("[data-chart-dl]");
    if (dlBtn) {
      ev.preventDefault();
      _downloadChipChart(dlBtn);
      return;
    }
  });
}

// Read the active chip slug from the URL hash rather than capturing
// it in bindClicks's closure — bindClicks attaches once per mounted
// view container (`__chipDetailClicksAttached` guard), so a captured
// slug would go stale the first time the user navigates to a
// different chip without remounting the view.  `#/chip/<slug>(?…)`
// is the only shape this view ever runs under, so the regex is safe.
function _activeChipSlug() {
  const m = (location.hash || "").match(/#\/chip\/([^?]+)/);
  return m ? decodeURIComponent(m[1]) : "chip";
}

async function _downloadChipChart(btn) {
  const kind = btn.dataset.chartDl; // "radar" | "scaling"
  const wrap = btn.closest(".chip-fp-canvas, .chip-scl-canvas");
  const canvas = wrap && wrap.querySelector("canvas");
  if (!canvas) {
    flashButtonLabel(btn, "Failed", { holdMs: 2000, className: "is-failed", labelSelector: ".chart-dl-btn-label" });
    return;
  }
  const slug = _activeChipSlug();
  const filename = `${slug}-${kind === "radar" ? "fingerprint" : "scaling"}.png`;
  const ok = await downloadCanvasAsPng(canvas, { filename });
  flashButtonLabel(btn, ok ? "Saved" : "Failed", {
    holdMs: ok ? 1400 : 2200,
    className: ok ? "is-saved" : "is-failed",
    labelSelector: ".chart-dl-btn-label",
  });
}

function _chipShareUrl() {
  // Strip any in-flight query string (`?foo=…` after `#/chip/<slug>`)
  // so the shared link points at the canonical chip-detail view.
  // Anything past `#/chip/<slug>` is current-session UI state, not
  // semantically part of the chip identity.
  const hash = location.hash || "";
  const hashPath = hash.split("?")[0];
  return location.origin + location.pathname + location.search + hashPath;
}

async function _copyChipShareLink(btn) {
  const url = _chipShareUrl();
  const ok = await copyToClipboard(url);
  flashButtonLabel(btn, ok ? "Copied!" : "Copy failed — select & ⌘C", {
    holdMs: ok ? 1600 : 3500,
    className: ok ? "is-copied" : "is-copy-failed",
    labelSelector: ".copy-btn-label",
  });
}

// ── 02 · Performance fingerprint (radar) ──────────────────────
//
// Tells the user where this chip is strong and weak across the suite
// spectrum at a glance.  Each axis = one suite, each tick = % of the
// global best primary metric for that suite (asc-direction metrics
// inverted so the leader still reads as 100 %).
//
// We render a section + canvas, then mount Chart.js after innerHTML
// lands.  When the chip has data on fewer than 2 suites the polygon
// degenerates to a point and the chart adds no signal — the section
// is skipped.  When Chart.js isn't available (rare: tests, blocked
// CDN) we render a structured table fallback so the data still
// reaches keyboard / screen-reader users.
function renderFingerprintSection(slug, sample) {
  const fp = suiteFingerprint(slug);
  const activeSuites = SUITE_ORDER.filter((sid) => {
    const cell = fp.get(sid);
    return cell && !cell.missing;
  });
  if (activeSuites.length < 2) return "";

  const missing = SUITE_ORDER.filter((sid) => fp.get(sid)?.missing);
  const cells = SUITE_ORDER.map((sid) => {
    const meta = SUITE_META[sid];
    const cell = fp.get(sid);
    if (!meta || !cell) return "";
    const pct = cell.missing ? "—" : `${Math.round(cell.normalized * 100)}%`;
    const value = cell.missing ? "no data" : formatPrimary(cell.value, sid);
    return `
      <li class="chip-fp-row" data-suite="${esc(meta.letter)}">
        <span class="chip-fp-letter" aria-hidden="true">${esc(meta.letter)}</span>
        <span class="chip-fp-name">${esc(meta.title)}</span>
        <span class="chip-fp-value tnum">${esc(value)}</span>
        <span class="chip-fp-pct tnum${cell.missing ? " is-missing" : ""}">${esc(pct)}</span>
      </li>
    `;
  }).join("");

  return `
    <section class="section chip-fp-section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">02 · Performance fingerprint</span>
          <h2>How this chip sits across the spectrum</h2>
        </div>
        <p class="section-sub">
          Each axis is one suite.  100 % is the global best primary metric
          for that suite — your chip's normalised score sits inside.
          ${missing.length
            ? `Suites without a submission collapse to the centre (${missing.map((sid) => SUITE_META[sid]?.letter).filter(Boolean).join(", ")}).`
            : ""}
        </p>
      </div>
      <div class="chip-fp-wrap" data-vendor="${esc(sample.vendor)}">
        <div class="chip-fp-canvas">
          <canvas data-chip-radar
                  aria-label="Radar chart of ${esc(sample.chip)} performance across all suites"
                  role="img"></canvas>
          <button class="chart-dl-btn"
                  type="button"
                  data-chart-dl="radar"
                  title="Download this radar as a PNG image">
            <span class="chart-dl-btn-icon" aria-hidden="true">↓</span>
            <span class="chart-dl-btn-label">PNG</span>
          </button>
        </div>
        <ol class="chip-fp-legend" aria-label="Per-suite normalised scores">
          ${cells}
        </ol>
      </div>
    </section>
  `;
}

let _activeFpChart = null;

function _mountFingerprintChart(el, slug, sample) {
  const canvas = el.querySelector("[data-chip-radar]");
  if (!canvas) return;
  if (typeof window.Chart !== "function") return; // table fallback already shown.

  if (_activeFpChart) {
    try { _activeFpChart.destroy(); } catch (_) { /* noop */ }
    _activeFpChart = null;
  }

  const fp = suiteFingerprint(slug);
  const labels = SUITE_ORDER
    .map((sid) => SUITE_META[sid]?.letter || sid)
    .filter(Boolean);
  const data = SUITE_ORDER.map((sid) => {
    const cell = fp.get(sid);
    return cell && !cell.missing ? Math.round(cell.normalized * 100) : 0;
  });
  const reference = SUITE_ORDER.map(() => 100);

  const cs = getComputedStyle(document.documentElement);
  const textColor   = (cs.getPropertyValue("--fg-muted").trim()    || "#8b949e");
  const gridColor   = (cs.getPropertyValue("--border-soft").trim() || "rgba(127,127,127,0.18)");
  const refColor    = (cs.getPropertyValue("--fg-faint").trim()    || "#888780");
  const accentHex   = vendorColor(sample.vendor);

  _activeFpChart = new window.Chart(canvas, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "Global best",
          data: reference,
          borderColor: refColor,
          borderDash: [4, 4],
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 1,
        },
        {
          label: sample.chip,
          data,
          borderColor: accentHex,
          backgroundColor: accentHex + "33",
          pointBackgroundColor: accentHex,
          pointBorderColor: accentHex,
          pointRadius: 3.5,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.r}%`,
          },
        },
      },
      scales: {
        r: {
          suggestedMin: 0,
          suggestedMax: 100,
          ticks: {
            color: textColor,
            backdropColor: "transparent",
            font: { size: 10 },
            stepSize: 25,
            showLabelBackdrop: false,
            callback: (v) => `${v}%`,
          },
          grid: { color: gridColor },
          angleLines: { color: gridColor },
          pointLabels: {
            color: textColor,
            font: { size: 12, weight: "600" },
          },
        },
      },
    },
  });
}

// ── 03 · Scaling across chip-counts (grouped bar) ─────────────
//
// Only meaningful when the chip has been deployed at >1 fan-out
// (e.g. ×1 + ×8 for an A100, ×1 + ×4 for a 4090D); otherwise this
// section returns "" and the rest of the page renumbers downward.
// The chart answers: "is this chip linearly faster at ×8 than at
// ×1?  On which suites does going wide actually help?"
//
// Y-axis is "% of this chip's best on this suite" — within a single
// suite cluster, the bars are normalised to the highest-throughput
// chip-count for that suite (asc-direction primary metrics inverted
// so the lowest-latency variant still reads as 100 %).  Cross-suite
// comparison would mix tok/s with ms with %, which is exactly what
// the fingerprint radar is for; here we want intra-suite scaling
// behaviour to read at a glance instead.
function renderScalingSection(slug, sample) {
  const data = chipCountScaling(slug);
  if (data.suites.length === 0) return "";

  const chipCounts = data.chipCounts;
  // Build a compact legend list — also serves as the keyboard /
  // screen-reader fallback when Chart.js can't load.
  const legendItems = chipCounts.map((c, i) => `
    <li class="chip-scl-legend-item" data-count-idx="${i}">
      <span class="chip-scl-legend-swatch" data-count-idx="${i}" aria-hidden="true"></span>
      <span class="chip-scl-legend-label">×${c}</span>
    </li>
  `).join("");

  // Per-suite breakdown row underneath the chart — visible on every
  // viewport and the only surface for the absolute primary-metric
  // value (chart bars are normalised %).
  const breakdownRows = data.suites.map((s) => {
    const bestPerCount = chipCounts.map((c) => {
      const cell = s.perCount.get(c);
      if (!cell || cell.value == null) return `<span class="chip-scl-cell-empty" title="No submission at ×${c}">—</span>`;
      const pct = Math.round(cell.normalized * 100);
      const display = formatPrimary(cell.value, s.sid);
      return `
        <span class="chip-scl-cell" title="${esc(`×${c}: ${display} (${pct}% of this chip's best in Suite ${s.letter})`)}">
          <span class="chip-scl-cell-pct tnum">${pct}%</span>
          <span class="chip-scl-cell-val tnum">${esc(display || "—")}</span>
        </span>
      `;
    }).join("");
    return `
      <div class="chip-scl-row" data-suite="${esc(s.letter)}">
        <span class="chip-scl-row-letter" aria-hidden="true">${esc(s.letter)}</span>
        <span class="chip-scl-row-title">${esc(s.title)}</span>
        <span class="chip-scl-row-cells">${bestPerCount}</span>
      </div>
    `;
  }).join("");

  return `
    <section class="section chip-scl-section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">03 · Scaling across chip-counts</span>
          <h2>Does going wide actually pay off?</h2>
        </div>
        <p class="section-sub">
          Bars are normalised to this chip's best result on each suite —
          ×N at 100 % means that fan-out wins the suite among this chip's
          variants.  Chip-counts without a submission for a suite show as
          gaps; zoom out via Compare to put another chip on the same axes.
        </p>
      </div>
      <div class="chip-scl-wrap" data-vendor="${esc(sample.vendor)}">
        <div class="chip-scl-canvas">
          <canvas data-chip-scaling
                  aria-label="Grouped bar chart of ${esc(sample.chip)} scaling across chip-counts ${chipCounts.map((c) => `×${c}`).join(", ")}"
                  role="img"></canvas>
          <button class="chart-dl-btn"
                  type="button"
                  data-chart-dl="scaling"
                  title="Download this scaling chart as a PNG image">
            <span class="chart-dl-btn-icon" aria-hidden="true">↓</span>
            <span class="chart-dl-btn-label">PNG</span>
          </button>
        </div>
        <ol class="chip-scl-legend" aria-label="Chip-count series">
          ${legendItems}
        </ol>
      </div>
      <div class="chip-scl-breakdown" role="table" aria-label="Per-suite scaling breakdown">
        ${breakdownRows}
      </div>
    </section>
  `;
}

let _activeSclChart = null;

function _mountScalingChart(el, slug, sample) {
  const canvas = el.querySelector("[data-chip-scaling]");
  if (!canvas) return;
  if (typeof window.Chart !== "function") return; // breakdown table is the fallback.

  if (_activeSclChart) {
    try { _activeSclChart.destroy(); } catch (_) { /* noop */ }
    _activeSclChart = null;
  }

  const data = chipCountScaling(slug);
  if (!data.suites.length) return;

  const chipCounts = data.chipCounts;
  const labels = data.suites.map((s) => `Suite ${s.letter}`);
  const accentHex = vendorColor(sample.vendor);

  // Per-chip-count colour: tint the vendor accent by lightening it
  // for higher counts, so the bars within a cluster read as a
  // gradient ("more chips = brighter" matches scaling intuition)
  // instead of needing a separate palette per chip family.
  const datasets = chipCounts.map((c, i) => {
    const tint = chipCounts.length === 1 ? 0 : i / (chipCounts.length - 1);
    // Mix from accent (i=0) → vendor accent at 50% (i=last).  We use
    // explicit hex for opacity so old browsers without color-mix in
    // canvas still get a reasonable colour.
    const alphaHex = (200 - Math.round(tint * 90)).toString(16).padStart(2, "0");
    return {
      label: `×${c}`,
      data: data.suites.map((s) => {
        const cell = s.perCount.get(c);
        return cell && cell.value != null ? Math.round(cell.normalized * 100) : null;
      }),
      backgroundColor: accentHex + alphaHex,
      borderColor: accentHex,
      borderWidth: 1,
      borderRadius: 3,
    };
  });

  // Push the same per-chip-count palette into the legend swatches so
  // the canvas + DOM legend stay visually in sync without a runtime
  // observer.
  const legendItems = el.querySelectorAll(".chip-scl-legend-swatch");
  legendItems.forEach((sw, i) => {
    sw.style.background = datasets[i]?.backgroundColor || accentHex;
    sw.style.borderColor = datasets[i]?.borderColor || accentHex;
  });

  const cs = getComputedStyle(document.documentElement);
  const textColor = (cs.getPropertyValue("--fg-muted").trim()    || "#8b949e");
  const gridColor = (cs.getPropertyValue("--border-soft").trim() || "rgba(127,127,127,0.18)");

  _activeSclChart = new window.Chart(canvas, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const sIdx = ctx.dataIndex;
              const cIdx = ctx.datasetIndex;
              const suite = data.suites[sIdx];
              const cnt = chipCounts[cIdx];
              const cell = suite?.perCount.get(cnt);
              if (!cell || cell.value == null) return `×${cnt}: no submission`;
              const display = formatPrimary(cell.value, suite.sid);
              return `×${cnt}: ${display} (${ctx.parsed.y}% of best)`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: textColor, font: { size: 11, weight: "600" } },
          grid:  { color: gridColor, display: false },
        },
        y: {
          beginAtZero: true,
          suggestedMax: 100,
          ticks: { color: textColor, callback: (v) => `${v}%` },
          grid:  { color: gridColor },
        },
      },
    },
  });
}

// ── 04 · Compare with similar chips ──
//
// Surfaces chips with the largest suite-coverage overlap so users have
// a one-click jump-off to peer hardware they'd realistically compare
// against.  Each tile links to that chip's detail page; the section as
// a whole is skipped when the dataset is too sparse for a meaningful
// recommendation (e.g. the chip is the only entry in its suites).
function renderSimilarChipsSection(slug, latestRid, sectionNum = "04") {
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
          <span class="eyebrow">${sectionNum} · Peers</span>
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
//
// The dual-affordance (plain click → run modal, modified click → filtered
// rankings) isn't visually obvious, so each active card carries both a
// `title` tooltip and a small bottom-row hint so the cmd-click path is
// discoverable without bloating the layout.
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

  // Two affordances on one element. Spell them out in the tooltip and
  // (more visibly) in a tiny hint footer so the modifier-click path is
  // findable without a separate help layer.
  const chipLabel = row._chip_label || "this chip";
  const cardTitle = `Click to open this run · Cmd/Ctrl-click to see all ${chipLabel} runs in Suite ${meta.letter}`;

  // Now that chip_count variants share a chip-detail page, the "best
  // per suite" run can land on any fan-out (×1 vs ×4 vs ×8).  Surface
  // that explicitly so users don't read the metric as "single-card
  // throughput".  Only render the badge when the chip has multiple
  // variants — otherwise it's noise.
  const allCounts = chipSlug ? chipCountsForChip(chipSlug) : [];
  const bestCount = row.chip_count || 1;
  const showCountBadge = allCounts.length > 1;

  return `
    <a class="chip-suite-card"
       data-suite="${esc(meta.letter)}"
       data-open-run="${esc(rid)}"
       href="${esc(rankingsHref)}"
       title="${esc(cardTitle)}">
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
        ${showCountBadge
          ? `<span class="chip-suite-count" title="Best score in this suite came from a ×${bestCount} deployment">×${bestCount}</span>`
          : ""}
      </div>
      <div class="chip-suite-meta">
        ${fwLine}${row.precision ? ` · ${esc(row.precision)}` : ""}${row.date ? ` · ${esc(fmtDate(row.date))}` : ""}
      </div>
      <div class="chip-suite-hint" aria-hidden="true">
        <span class="chip-suite-hint-primary">Open run</span>
        <span class="chip-suite-hint-sep">·</span>
        <span class="chip-suite-hint-secondary"><kbd>⌘</kbd>+click for all in suite</span>
      </div>
    </a>
  `;
}

function renderRunsTable(rs) {
  // Group by chip_count then date desc; runs at the same fan-out land
  // next to each other so users can scan "what does ×1 look like vs ×8"
  // without flipping rows.  Within a fan-out, newest first.
  const sorted = rs.slice().sort((a, b) => {
    const ca = a.chip_count || 1;
    const cb = b.chip_count || 1;
    if (ca !== cb) return ca - cb;
    return String(b.date || "").localeCompare(String(a.date || ""));
  });
  // Show the Chips column only when this chip actually has variants —
  // adding a column that only ever reads "×1" is just visual debt.
  const counts = new Set(rs.map((r) => r.chip_count || 1));
  const showChipCol = counts.size > 1;
  return `
    <table class="data-table chip-runs">
      <thead>
        <tr>
          <th class="col-suite">Suite</th>
          ${showChipCol ? `<th class="col-chips">Chips</th>` : ""}
          <th class="col-framework">Framework</th>
          <th class="col-precision">Precision</th>
          <th class="col-primary">Primary metric</th>
          <th class="col-date">Date</th>
          <th class="col-submitter">Submitter</th>
          <th class="col-tier">Tier</th>
        </tr>
      </thead>
      <tbody>
        ${sorted.map((r) => renderRunRow(r, showChipCol)).join("")}
      </tbody>
    </table>
  `;
}

function renderRunRow(row, showChipCol) {
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
  const cnt = row.chip_count || 1;

  // a11y: tabindex makes the row keyboard-reachable; modal.js's
  // keydown delegate fires openModal on Enter/Space.  Native <tr>
  // semantics stay so screen-reader column headers still pair with
  // each cell.
  const a11yLabel = `Open run details: ${meta ? meta.title + " · " : ""}${row.framework || ""} ${display || ""}`.trim();
  return `
    <tr data-open-run="${esc(rid)}"
        data-suite="${meta ? esc(meta.letter) : ""}"
        tabindex="0"
        aria-label="${esc(a11yLabel)}">
      <td class="col-suite">
        <span class="chip-runs-suite">
          <span class="chip-suite-letter chip-suite-letter--inline">${meta ? esc(meta.letter) : "·"}</span>
          <span class="chip-runs-suite-title">${meta ? esc(meta.title) : esc(row.suite || "")}</span>
        </span>
      </td>
      ${showChipCol ? `<td class="col-chips tnum"><span class="chip-runs-count">×${cnt}</span></td>` : ""}
      <td class="col-framework">${fwLine}</td>
      <td class="col-precision">${esc(row.precision || "—")}</td>
      <td class="col-primary"><span class="chip-runs-metric">${esc(display || "—")}</span></td>
      <td class="col-date">${esc(fmtDate(row.date))}</td>
      <td class="col-submitter">${handle ? `@${esc(handle)}` : "—"}</td>
      <td class="col-tier"><span class="badge${tierClass}">${esc(row.tier || "—")}</span></td>
    </tr>
  `;
}
