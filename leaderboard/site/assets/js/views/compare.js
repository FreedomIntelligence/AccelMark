// views/compare.js — Side-by-side comparison of basket runs on one suite.
//
// Page anatomy:
//   • Compact hero with the active suite's eyebrow + title + tagline.
//   • Basket strip listing each compared run (vendor-tinted dot, chip
//     name + framework, × remove button) and a "Clear all" affordance.
//   • Suite switcher (re-uses .rk-suite-pill from rankings).  Switching
//     the suite does NOT clear the basket — instead each run resolves
//     to its same-hardware best row in the newly-selected suite, so the
//     user can compare the same set of configurations across metrics.
//   • Per-metric comparison table:
//       rows  = metrics in SUITE_COLUMNS[suiteId]
//       cols  = runs (one per basket entry)
//       cells = horizontal bar normalized to the row's best run + value,
//               with a ★ on the winning run.  For "lower is better"
//               metrics the bar uses min / value so the winner still
//               renders at 100 % width.
//
// Empty state when the basket has zero runs, with a deep link back to
// rankings to start picking.
//
// On basket changes the view re-renders itself, but only when /compare
// is still the active route — otherwise we'd clobber whatever view the
// user just navigated to.

import {
  SUITE_ORDER, SUITE_META, SUITE_COLUMNS, formatMetric,
  rowByRunId, bestRowForRunInSuite, chipCloudData,
  representativeRunForChip,
} from "../data.js";
import {
  esc, fmtNum, buildHash, chipHref, parseHash, shortVersion,
  copyToClipboard, flashButtonLabel, downloadCanvasAsPng,
} from "../utils.js";
import {
  basketGet, basketHas, basketToggle, basketOnChange,
} from "../router.js";

export function render({ el, query }) {
  // Seed from ?runs=a,b,c (back-compat: ?chips=…).  Used both for
  // shareable links and for the Home chip-cloud quick-add affordance,
  // which sends users here with a single run pre-baked into the URL.
  //
  // Behaviour:
  //   • Add each listed run to the basket if it isn't already there —
  //     existing entries are preserved so users coming from Home can
  //     stack a chip on top of a basket they already built up.
  //   • Strip ?runs= / ?chips= from the URL after consuming so that
  //     subsequent re-renders (suite toggles, basket edits) don't
  //     reprocess the same param.  This also keeps the address bar
  //     clean once the state has moved into in-memory basket.
  const seed = query.runs || query.chips;
  if (seed) {
    const added = String(seed).split(",").map((x) => x.trim()).filter(Boolean);
    // Strip ?runs= / ?chips= from the URL BEFORE mutating the basket.
    // basketToggle fires the listener synchronously which re-renders by
    // re-parsing location.hash — if we don't strip first, that re-render
    // sees the same seed param and recurses through render() again.
    if (added.length > 0) {
      const cleaned = { ...query };
      delete cleaned.runs;
      delete cleaned.chips;
      const newHash = buildHash("/compare", cleaned);
      if (location.hash !== newHash) {
        history.replaceState(null, "", newHash);
      }
    }
    for (const s of added) {
      if (!basketHas(s)) basketToggle(s);
    }
  }

  const runIds = basketGet();

  if (runIds.length === 0) {
    el.innerHTML = `
      <section class="cmp-hero">
        <span class="eyebrow">Compare</span>
        <h1 class="cmp-hero-title">Side-by-Side Comparison</h1>
        <p class="cmp-hero-sub">
          Pick chips below to start a head-to-head across every metric.
          You can also tick runs from
          <a class="cmp-hero-link" href="#/rankings">any rankings page</a>
          to compare specific framework / precision configurations.
        </p>
      </section>
      ${renderChipCloudBlock({
        title: "Pick chips to compare",
        hint: "Each click adds that chip's most recent run.  Choose any two or more.",
        compact: false,
      })}
    `;
    bindClicks(el);
    attachBasketListener(el);
    return;
  }

  // Resolve each basket entry to its source (seed) row.  Runs that no
  // longer exist in the dataset (stale shared link) are silently dropped.
  const seeds = runIds
    .map((rid) => ({ rid, row: rowByRunId(rid) }))
    .filter((x) => x.row);

  // Edge case: the basket has entries but every one of them resolves to
  // null — almost always a shared comparison URL whose runs have since
  // been re-uploaded under new ids or pruned.  Without this short-
  // circuit the page renders a header-only table and an empty chart
  // grid with no explanation, which reads like a broken UI.
  if (seeds.length === 0) {
    el.innerHTML = `
      <section class="cmp-hero">
        <span class="eyebrow">Compare</span>
        <h1 class="cmp-hero-title">Side-by-Side Comparison</h1>
        <p class="cmp-hero-sub">
          This comparison link refers to ${runIds.length === 1 ? "a run" : `${runIds.length} runs`}
          that ${runIds.length === 1 ? "is" : "are"} no longer in the dataset
          (re-uploaded or pruned). Pick chips below to start a new comparison.
        </p>
      </section>
      <div class="cmp-basket cmp-basket--empty">
        <span class="cmp-basket-label">Comparing</span>
        <span class="cmp-basket-stale">${runIds.length} stale ${runIds.length === 1 ? "run" : "runs"}</span>
        <div class="cmp-basket-actions">
          <button class="cmp-basket-clear" data-basket-clear="1" type="button">Clear &amp; start over</button>
        </div>
      </div>
      ${renderChipCloudBlock({
        title: "Pick chips to compare",
        hint: "Each click adds that chip's most recent run.  Choose any two or more.",
        compact: false,
      })}
    `;
    bindClicks(el);
    attachBasketListener(el);
    return;
  }

  // Default suite: first one in canonical order that any selected run
  // has data for; otherwise the seed's own suite.  Lets the page open
  // on something useful even without a ?suite= param.
  let suiteId = SUITE_ORDER.includes(query.suite) ? query.suite : null;
  if (!suiteId) {
    for (const sid of SUITE_ORDER) {
      const hasData = seeds.some(({ rid }) => bestRowForRunInSuite(rid, sid));
      if (hasData) { suiteId = sid; break; }
    }
    if (!suiteId) suiteId = (seeds[0] && seeds[0].row.suite) || SUITE_ORDER[0];
  }
  const meta = SUITE_META[suiteId];
  const cols = SUITE_COLUMNS[suiteId];

  // For each basket run resolve the equivalent row in the active suite.
  // The seed row (where the user originally ticked the checkbox) is
  // used purely for display attribution — chip name, vendor, framework
  // chip in the basket strip — so the user sees the configuration they
  // selected even when this suite has no matching data.
  const chips = seeds.map(({ rid, row }) => ({
    rid,
    label: row._chip_label,
    vendor: row.vendor,
    seedRow: row,
    suiteRow: bestRowForRunInSuite(rid, suiteId),
  }));

  // When none of the selected chips have data for the active suite the
  // page would otherwise read as a wall of "-" dashes with no charts.
  // Suggest the suites that DO have data so users have a clear next step.
  const suitesWithData = SUITE_ORDER.filter((sid) =>
    chips.some((c) => bestRowForRunInSuite(c.rid, sid))
  );
  const suiteEmpty = chips.length > 0 && chips.every((c) => !c.suiteRow);

  el.innerHTML = `
    <section class="cmp-hero">
      <span class="eyebrow">Compare</span>
      <h1 class="cmp-hero-title">${esc(meta.title)}</h1>
      <p class="cmp-hero-sub">${esc(meta.tagline)}</p>
    </section>

    ${renderChipCloudBlock({
      title: "Pick chips to compare",
      hint: "Click a chip to add or remove its most recent run.  Already-selected chips are highlighted.",
      compact: true,
    })}

    <div class="cmp-basket">
      <span class="cmp-basket-label">Comparing</span>
      ${chips.map((c) => renderBasketChip(c)).join("")}
      <div class="cmp-basket-actions">
        <button class="copy-btn cmp-basket-share"
                data-basket-share="1"
                type="button"
                title="Copy a URL that pre-loads this comparison.">
          <span class="copy-btn-icon" aria-hidden="true">↗</span>
          <span class="copy-btn-label">Copy share link</span>
        </button>
        <button class="cmp-basket-clear" data-basket-clear="1" type="button">Clear all</button>
      </div>
    </div>

    <div class="cmp-suite-row">
      <span class="rk-facet-label">Suite</span>
      <div class="rk-suite-pills">
        ${SUITE_ORDER.map((sid) => renderSuitePill(sid, sid === suiteId)).join("")}
      </div>
    </div>

    ${suiteEmpty ? `
      <div class="cmp-suite-empty">
        <span class="state-icon" aria-hidden="true">∅</span>
        <p>None of the selected chips have <strong>Suite ${esc(meta.letter)} · ${esc(meta.title)}</strong> data.</p>
        ${suitesWithData.length ? `
          <p class="cmp-suite-empty-sub">Try
            ${suitesWithData.map((sid) => `
              <a class="cmp-suite-empty-link" href="${esc(buildHash("/compare", { ...query, suite: sid }))}">
                Suite ${esc(SUITE_META[sid].letter)}
              </a>
            `).join(" · ")}
            instead.
          </p>
        ` : `<p class="cmp-suite-empty-sub">The chips you picked have no submissions on file.</p>`}
      </div>
    ` : `
      <div class="cmp-table-wrap">
        ${renderCmpTable(suiteId, cols, chips)}
      </div>

      <div class="cmp-charts-wrap" data-suite="${esc(meta.letter)}"></div>
    `}
  `;

  bindClicks(el);
  attachBasketListener(el);
  renderCmpCharts(el.querySelector(".cmp-charts-wrap"), suiteId, chips);
}

// ── Chip cloud — quick-add affordance ──
//
// Reuses the same chip-tile styling as the home page so users get a
// familiar visual rhythm.  Tiles with an `.in-basket` modifier mark
// chips that already have at least one row in the basket (compared by
// chip name, not slug — so e.g. an H100×8 chip already in the basket
// will mark the H100 tile as selected even though the cloud surfaces
// the most popular variant of each chip).
function renderChipCloudBlock({ title, hint, compact }) {
  const chips = chipCloudData();
  const basketChipNames = new Set(
    basketGet()
      .map((rid) => rowByRunId(rid))
      .filter(Boolean)
      .map((r) => r.chip)
  );
  if (!chips.length) return "";
  const tiles = chips.map((c) => {
    const inBasket = basketChipNames.has(c.label);
    const subL  = c.submissions === 1 ? "submission" : "submissions";
    const suiteL = c.suites.length === 1 ? "suite" : "suites";
    const variantPart = c.variants > 1 ? ` · ${c.variants} chip-count variants` : "";
    // Left-click is intercepted by bindClicks for the toggle add/remove
    // basket behaviour; the href is the middle-click / Cmd-click /
    // copy-link fallback and points at the chip's overview page so it
    // matches every other chip-name link on the site.
    //
    // a11y: tile doubles as a toggle (left-click) and a navigation link
    // (modifier-click).  We treat the toggle as the primary action for
    // assistive tech — `role="button"` + `aria-pressed` mirrors the
    // visual "in-basket" state.  Modifier-click is documented in the
    // tooltip so non-mouse users know about the secondary affordance.
    const a11yTitle = `${c.label}: ${c.submissions} ${subL} across ${c.suites.length} ${suiteL}${variantPart}. Click to ${inBasket ? "remove from" : "add to"} compare basket; Cmd / Ctrl-click to open chip overview.`;
    return `
      <a class="chip-tile size-${esc(c.size)}${inBasket ? " in-basket" : ""}"
         href="#/chip/${esc(c.slug)}"
         data-vendor="${esc(c.vendor)}"
         data-cmp-add-slug="${esc(c.slug)}"
         role="button"
         aria-pressed="${inBasket ? "true" : "false"}"
         title="${esc(a11yTitle)}">
        <span class="chip-tile-name">${esc(c.label)}</span>
        <span class="chip-tile-count">${fmtNum(c.submissions)}</span>
      </a>
    `;
  }).join("");
  return `
    <section class="cmp-cloud${compact ? " is-compact" : ""}">
      <div class="cmp-cloud-head">
        <h2 class="cmp-cloud-title">${esc(title)}</h2>
        ${hint ? `<p class="cmp-cloud-hint">${esc(hint)}</p>` : ""}
      </div>
      <div class="chip-cloud">${tiles}</div>
    </section>
  `;
}

// ── Per-suite head-to-head charts ──────────────────────────────
//
// Lives below the comparison table and renders one or more Chart.js
// charts that overlay every basket run on the same canvas.  Each suite
// picks its most distinctive visualization (offline throughput curves,
// quantization-format throughput, scaling efficiency, etc.) instead of
// recycling the same chart everywhere.
//
// Charts are torn down on every render so re-attaching never leaks
// canvases.  When the active suite has no chart data for the current
// basket the section is hidden entirely — rather than show empty
// canvases — to keep the page short.

const _activeCmpCharts = [];

// Distinct chip palette — same hues as legacy main so users get a
// stable visual ID across pages.  Cycles modulo length when there are
// more chips than colors.
const CMP_PALETTE = [
  "#3b82f6", "#14b8a6", "#f97316", "#ec4899",
  "#a855f7", "#84cc16", "#f59e0b", "#06b6d4",
];
function _palette(i) { return CMP_PALETTE[i % CMP_PALETTE.length]; }

function _destroyCmpCharts() {
  while (_activeCmpCharts.length) {
    const c = _activeCmpCharts.pop();
    try { c.destroy(); } catch (_) { /* noop */ }
  }
}

function _cmpChartColors() {
  const cs = getComputedStyle(document.documentElement);
  return {
    text: (cs.getPropertyValue("--fg-muted").trim()    || "#8b949e"),
    grid: (cs.getPropertyValue("--border-soft").trim() || "rgba(127,127,127,0.18)"),
  };
}

function renderCmpCharts(wrap, suiteId, chips) {
  if (!wrap) return;
  _destroyCmpCharts();
  wrap.innerHTML = "";

  if (typeof window.Chart !== "function") return;
  // Only chips that actually have a suite row contribute to the chart;
  // ones missing data for this suite are listed separately as a note.
  const active = chips.filter((c) => c.suiteRow);
  if (active.length < 1) return;
  const missing = chips.filter((c) => !c.suiteRow);

  const renderer = CMP_CHART_RENDERERS[suiteId];
  if (!renderer) return;

  const charts = renderer(active);
  if (!charts || !charts.length) return;

  wrap.innerHTML = `
    <div class="cmp-charts-head">
      <h2 class="cmp-charts-title">Head-to-head charts</h2>
      <p class="cmp-charts-hint">${esc(
        active.length === chips.length
          ? "Each chip overlaid on the same axes."
          : `${active.length} of ${chips.length} chips have Suite ${SUITE_META[suiteId].letter} data — others are listed below.`
      )}</p>
    </div>
  `;

  for (const spec of charts) {
    const card = document.createElement("div");
    card.className = "cmp-chart-card";
    card.innerHTML = `
      <div class="cmp-chart-card-head">
        <h3 class="cmp-chart-card-title">${esc(spec.title)}</h3>
        ${spec.subtitle ? `<p class="cmp-chart-card-sub">${esc(spec.subtitle)}</p>` : ""}
      </div>
      <div class="cmp-chart-card-body"></div>
    `;
    const body = card.querySelector(".cmp-chart-card-body");
    const canvasWrap = document.createElement("div");
    canvasWrap.className = "cmp-chart-canvas";
    canvasWrap.style.height = (spec.height || 220) + "px";
    const canvas = document.createElement("canvas");
    canvasWrap.appendChild(canvas);
    // Download-as-PNG button — anchors against the canvas wrapper
    // (which is `position: relative` in rankings.css).  data-chart-dl
    // value is the chart spec's title slug so the saved file reads
    // as e.g. "compare-suite-a-throughput-by-concurrency.png".
    const dlBtn = document.createElement("button");
    dlBtn.className = "chart-dl-btn";
    dlBtn.type = "button";
    dlBtn.dataset.chartDl = (spec.title || "chart").toLowerCase()
      .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
    dlBtn.title = "Download this chart as a PNG image";
    dlBtn.innerHTML = `
      <span class="chart-dl-btn-icon" aria-hidden="true">↓</span>
      <span class="chart-dl-btn-label">PNG</span>
    `;
    canvasWrap.appendChild(dlBtn);
    body.appendChild(canvasWrap);
    body.appendChild(_cmpLegend(spec.legend || []));
    wrap.appendChild(card);
    try {
      _activeCmpCharts.push(new window.Chart(canvas, spec.config));
    } catch (e) {
      body.innerHTML = `<div class="cmp-chart-empty">Chart failed to render.</div>`;
    }
  }

  if (missing.length) {
    const note = document.createElement("div");
    note.className = "cmp-charts-missing";
    const names = missing.map((m) => esc(m.label)).join(", ");
    note.innerHTML = `<strong>No Suite ${esc(SUITE_META[suiteId].letter)} data:</strong> ${names}`;
    wrap.appendChild(note);
  }
}

function _cmpLegend(items) {
  const div = document.createElement("div");
  div.className = "cmp-chart-legend";
  div.innerHTML = items.map(({ color, label }) => `
    <span class="cmp-chart-legend-item">
      <span class="cmp-chart-legend-swatch" style="background:${esc(color)}"></span>
      ${esc(label)}
    </span>
  `).join("");
  return div;
}

// ── Per-suite chart specs ─────────────────────────────────────
//
// Each entry returns an array of chart specs ({title, subtitle, config,
// legend}).  Config goes straight to new Chart(canvas, config).

const CMP_CHART_RENDERERS = {
  suite_A: (chips) => _offlineConcurrencyBars(chips, "Single-chip offline throughput",  "tok/s by concurrency"),
  suite_F: (chips) => _offlineConcurrencyBars(chips, "Edge offline throughput",          "tok/s by concurrency"),
  suite_G: (chips) => _offlineConcurrencyBars(chips, "MoE offline throughput",           "tok/s by concurrency"),
  suite_B: (chips) => [
    ..._offlineConcurrencyBars(chips, "Multi-chip total throughput", "tok/s by concurrency"),
    ..._perChipThroughputBars(chips),
  ],
  suite_C: (chips) => _quantThroughputBars(chips),
  suite_D: (chips) => _longContextLatency(chips),
  suite_E: (chips) => _scalingCurves(chips),
};

// Shared: grouped bar where x = concurrency level, groups = chips.
// Each chip's offline.throughput array contributes one bar at each
// concurrency level.  Falls back to a single primary-metric bar when
// the run lacks per-concurrency series.
function _offlineConcurrencyBars(chips, title, subtitle) {
  const C = _cmpChartColors();
  // Collect the union of concurrency labels across all chips.
  const labelSet = new Set();
  for (const c of chips) {
    const off = c.suiteRow.viz && c.suiteRow.viz.offline;
    if (off && Array.isArray(off.labels)) {
      for (const l of off.labels) labelSet.add(String(l));
    }
  }
  if (labelSet.size === 0) {
    // No per-concurrency data; render simple primary-metric bar across chips.
    const labels = chips.map((c) => c.label);
    const data = chips.map((c) => c.suiteRow.offline_throughput ?? null);
    return [{
      title,
      subtitle: "Peak offline throughput",
      height: 200,
      config: {
        type: "bar",
        data: {
          labels,
          datasets: [{
            data,
            backgroundColor: chips.map((_, i) => _palette(i) + "99"),
            borderColor:     chips.map((_, i) => _palette(i)),
            borderWidth: 1, borderRadius: 3,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid } },
            y: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v.toLocaleString() }, grid: { color: C.grid }, title: { display: true, text: "tokens / sec", color: C.text, font: { size: 11 } } },
          },
        },
      },
      legend: [],
    }];
  }
  const labels = Array.from(labelSet).sort((a, b) => Number(a) - Number(b));
  const datasets = chips.map((c, i) => {
    const off = c.suiteRow.viz && c.suiteRow.viz.offline;
    let data = [];
    if (off && Array.isArray(off.labels)) {
      const idx = new Map(off.labels.map((l, j) => [String(l), j]));
      data = labels.map((l) => {
        const j = idx.get(String(l));
        return j == null ? null : off.throughput[j];
      });
    } else {
      data = labels.map(() => null);
    }
    return {
      label: c.label,
      data,
      backgroundColor: _palette(i) + "99",
      borderColor:     _palette(i),
      borderWidth: 1, borderRadius: 3,
    };
  });
  return [{
    title,
    subtitle,
    height: 230,
    config: {
      type: "bar",
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid }, title: { display: true, text: "concurrency", color: C.text, font: { size: 11 } } },
          y: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v.toLocaleString() }, grid: { color: C.grid }, title: { display: true, text: "tokens / sec", color: C.text, font: { size: 11 } } },
        },
      },
    },
    legend: chips.map((c, i) => ({ color: _palette(i), label: c.label })),
  }];
}

function _perChipThroughputBars(chips) {
  const C = _cmpChartColors();
  const labels = chips.map((c) => c.label);
  const data   = chips.map((c) => c.suiteRow.tokens_per_sec_per_chip ?? null);
  if (data.every((v) => v == null)) return [];
  return [{
    title: "Per-chip throughput",
    subtitle: "tok/s per accelerator, scales with hardware count",
    height: 200,
    config: {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: chips.map((_, i) => _palette(i) + "99"),
          borderColor:     chips.map((_, i) => _palette(i)),
          borderWidth: 1, borderRadius: 3,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid } },
          y: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v.toLocaleString() }, grid: { color: C.grid }, title: { display: true, text: "tok / sec / chip", color: C.text, font: { size: 11 } } },
        },
      },
    },
    legend: [],
  }];
}

function _quantThroughputBars(chips) {
  const C = _cmpChartColors();
  // Collect the union of precision formats across all chips.
  const fmtSet = new Set();
  for (const c of chips) {
    const v = c.suiteRow.viz;
    if (v && Array.isArray(v.precisions)) {
      for (const p of v.precisions) fmtSet.add(p);
    }
  }
  if (fmtSet.size === 0) return [];
  const PREC_ORDER = ["BF16", "FP16", "FP8", "W8A8", "W8A16", "W4A16"];
  const labels = Array.from(fmtSet).sort((a, b) => {
    const ai = PREC_ORDER.indexOf(a), bi = PREC_ORDER.indexOf(b);
    if (ai === -1 && bi === -1) return a.localeCompare(b);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
  const datasets = chips.map((c, i) => {
    const v = c.suiteRow.viz;
    let data = [];
    if (v && Array.isArray(v.precisions)) {
      const idx = new Map(v.precisions.map((p, j) => [p, j]));
      data = labels.map((p) => {
        const j = idx.get(p);
        return j == null ? null : v.throughput[j];
      });
    } else {
      data = labels.map(() => null);
    }
    return {
      label: c.label,
      data,
      backgroundColor: _palette(i) + "99",
      borderColor:     _palette(i),
      borderWidth: 1, borderRadius: 3,
    };
  });
  return [{
    title: "Throughput across quantization formats",
    subtitle: "tok/s by precision · grouped by chip",
    height: 230,
    config: {
      type: "bar",
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid }, title: { display: true, text: "precision", color: C.text, font: { size: 11 } } },
          y: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }) }, grid: { color: C.grid }, title: { display: true, text: "tokens / sec", color: C.text, font: { size: 11 } } },
        },
      },
    },
    legend: chips.map((c, i) => ({ color: _palette(i), label: c.label })),
  }];
}

function _longContextLatency(chips) {
  const C = _cmpChartColors();
  const buckets = [
    { key: "ttft_p50", label: "TTFT p50" },
    { key: "ttft_p90", label: "TTFT p90" },
    { key: "ttft_p99", label: "TTFT p99" },
    { key: "tpot_p50", label: "TPOT p50" },
    { key: "tpot_p90", label: "TPOT p90" },
    { key: "tpot_p99", label: "TPOT p99" },
  ];
  const datasets = chips.map((c, i) => {
    const v = (c.suiteRow.viz && c.suiteRow.viz.interactive) || {};
    return {
      label: c.label,
      data: buckets.map((b) => (v[b.key] == null ? null : v[b.key])),
      backgroundColor: _palette(i) + "99",
      borderColor:     _palette(i),
      borderWidth: 1, borderRadius: 3,
    };
  });
  // Drop datasets that are entirely null.
  const filtered = datasets.filter((ds) => ds.data.some((v) => v != null));
  if (!filtered.length) return [];
  return [{
    title: "Long-context latency",
    subtitle: "ms · TTFT (prefill) and TPOT (decode) percentiles",
    height: 280,
    config: {
      type: "bar",
      data: { labels: buckets.map((b) => b.label), datasets: filtered },
      options: {
        indexAxis: "y",
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v >= 1000 ? (v / 1000).toFixed(1) + "s" : v + "ms" }, grid: { color: C.grid }, title: { display: true, text: "latency", color: C.text, font: { size: 11 } } },
          y: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid } },
        },
      },
    },
    legend: chips.filter((_, i) => filtered.some((ds) => ds === datasets[i]))
      .map((c, i) => ({ color: _palette(i), label: c.label })),
  }];
}

function _scalingCurves(chips) {
  const C = _cmpChartColors();
  // Collect the union of chip counts across all chips.
  const countSet = new Set();
  for (const c of chips) {
    const v = c.suiteRow.viz;
    if (v && Array.isArray(v.chip_counts)) for (const n of v.chip_counts) countSet.add(Number(n));
  }
  if (!countSet.size) return [];
  const counts = Array.from(countSet).sort((a, b) => a - b);
  const labels = counts.map((n) => n + "× GPU");
  const throughputDs = chips.map((c, i) => {
    const v = c.suiteRow.viz;
    let data = [];
    if (v && Array.isArray(v.chip_counts)) {
      const idx = new Map(v.chip_counts.map((n, j) => [Number(n), j]));
      data = counts.map((n) => {
        const j = idx.get(n);
        return j == null ? null : v.throughput[j];
      });
    } else {
      data = counts.map(() => null);
    }
    return {
      label: c.label,
      data,
      borderColor:     _palette(i),
      backgroundColor: _palette(i) + "22",
      pointRadius: 4, tension: 0.25, fill: false,
    };
  });
  const effDs = chips.map((c, i) => {
    const v = c.suiteRow.viz;
    let data = [];
    if (v && Array.isArray(v.chip_counts)) {
      const idx = new Map(v.chip_counts.map((n, j) => [Number(n), j]));
      data = counts.map((n) => {
        const j = idx.get(n);
        return j == null ? null : v.efficiency_pct[j];
      });
    } else {
      data = counts.map(() => null);
    }
    return {
      label: c.label,
      data,
      borderColor:     _palette(i),
      backgroundColor: _palette(i) + "22",
      pointRadius: 4, tension: 0.25, fill: false,
    };
  });
  return [
    {
      title: "Throughput by chip count",
      subtitle: "tok/s · one line per chip",
      height: 240,
      config: {
        type: "line",
        data: { labels, datasets: throughputDs },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid } },
            y: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v.toLocaleString() }, grid: { color: C.grid }, title: { display: true, text: "tokens / sec", color: C.text, font: { size: 11 } } },
          },
        },
      },
      legend: chips.map((c, i) => ({ color: _palette(i), label: c.label })),
    },
    {
      title: "Scaling efficiency vs linear ideal",
      subtitle: "% · 100 % is perfect linear scaling",
      height: 240,
      config: {
        type: "line",
        data: {
          labels,
          datasets: [
            ...effDs,
            { label: "Linear ideal", data: counts.map(() => 100),
              borderColor: C.text, borderDash: [6, 4], pointRadius: 0, fill: false, tension: 0 },
            { label: "Good (80 %)", data: counts.map(() => 80),
              borderColor: "#2dd4bf", borderDash: [3, 3], pointRadius: 0, fill: false, tension: 0 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid } },
            y: { min: 0, max: 110, ticks: { color: C.text, font: { size: 11 }, callback: (v) => v + "%" }, grid: { color: C.grid }, title: { display: true, text: "efficiency %", color: C.text, font: { size: 11 } } },
          },
        },
      },
      legend: chips.map((c, i) => ({ color: _palette(i), label: c.label }))
        .concat([{ color: "#2dd4bf", label: "Good (80 %)" }]),
    },
  ];
}

function renderSuitePill(sid, active) {
  const m = SUITE_META[sid];
  return `
    <button class="rk-suite-pill ${active ? "active" : ""}"
            data-suite="${esc(sid)}"
            type="button"
            aria-pressed="${active ? "true" : "false"}"
            title="${esc(m.tagline)}">
      <span class="rk-suite-letter">${esc(m.letter)}</span>
      <span class="rk-suite-name">${esc(m.title)}</span>
    </button>
  `;
}

// Basket strip pill — chip name on top line, framework + version + precision
// on the second so the user can tell two same-chip runs apart at a glance.
// Plain click on the name opens the seed run's detail modal; Cmd-click
// falls through to the chip-level overview page.
function renderBasketChip(c) {
  const r = c.seedRow;
  const ver = shortVersion(r.framework_version);
  const fwLine = [r.framework, ver].filter(Boolean).join(" ");
  const detail = [fwLine, r.precision].filter(Boolean).join(" · ");
  // Chip-name link goes to the chip overview (every chip name on the
  // site should navigate to /chip/<slug>).  The × button still removes
  // the specific run from the basket — that's the basket-local action.
  return `
    <span class="cmp-basket-chip" data-vendor="${esc(c.vendor)}">
      <span class="vendor-dot"></span>
      <a href="${chipHref(r)}" class="cmp-basket-name">
        <span class="cmp-basket-chip-name">${esc(c.label)}</span>
        ${detail ? `<span class="cmp-basket-chip-meta">${esc(detail)}</span>` : ""}
      </a>
      <button class="cmp-basket-remove"
              data-remove-run="${esc(c.rid)}"
              aria-label="Remove ${esc(c.label)} from compare"
              type="button">×</button>
    </span>
  `;
}

function renderCmpTable(suiteId, cols, chips) {
  const meta = SUITE_META[suiteId];
  return `
    <table class="cmp-table" data-suite="${esc(meta.letter)}">
      <thead>
        <tr>
          <th class="cmp-row-header" scope="col">Metric</th>
          ${chips.map((c) => renderChipHead(c, meta)).join("")}
        </tr>
      </thead>
      <tbody>
        ${cols.map((col) => renderCmpRow(col, chips)).join("")}
      </tbody>
    </table>
  `;
}

function renderChipHead(c, meta) {
  const r = c.suiteRow;
  let metaLine = `No Suite ${esc(meta.letter)} data`;
  if (r) {
    const fw = r.framework || "";
    const ver = shortVersion(r.framework_version);
    const fwLine = ver ? `${fw} ${ver}` : fw;
    const parts = [fwLine, r.precision].filter(Boolean);
    metaLine = esc(parts.join(" · "));
  }
  // The header links to the chip's overview page so users can drill
  // into every run for that chip.  We use c.seedRow (the row the user
  // ticked from rankings) rather than c.suiteRow for the slug, so the
  // link stays stable when switching across suites.
  const href = chipHref(c.seedRow);
  return `
    <th class="cmp-chip-head" data-vendor="${esc(c.vendor)}" scope="col">
      <a class="cmp-chip-name" href="${esc(href)}">
        <span class="vendor-dot"></span>${esc(c.label)}
      </a>
      <span class="cmp-chip-meta">${metaLine}</span>
    </th>
  `;
}

function renderCmpRow(col, chips) {
  // Pull each chip's raw value for this metric.
  const values = chips.map((c) => {
    if (!c.suiteRow) return null;
    const v = c.suiteRow[col.key];
    if (v === null || v === undefined) return null;
    if (typeof v === "number" && Number.isNaN(v)) return null;
    return v;
  });

  // Find the winning index (skipped for textual columns).
  let bestIdx = -1;
  if (!col.textual) {
    let best = null;
    values.forEach((v, i) => {
      if (typeof v !== "number") return;
      if (best === null) { best = { v, i }; return; }
      if (col.direction === "asc"  && v < best.v) best = { v, i };
      if (col.direction === "desc" && v > best.v) best = { v, i };
    });
    if (best) bestIdx = best.i;
  }

  // Normalize bar fractions so the winner sits at 100 % regardless of
  // metric direction.  For "lower is better" we use min / value.
  const numerics = values.filter((v) => typeof v === "number");
  const max = numerics.length ? Math.max(...numerics) : 0;
  const min = numerics.length ? Math.min(...numerics) : 0;
  const fracOf = (v) => {
    if (typeof v !== "number") return 0;
    if (col.textual) return 0;
    if (col.direction === "asc") return v > 0 ? min / v : 0;
    return max > 0 ? v / max : 0;
  };

  const dirLabel = col.direction === "asc" ? "lower is better" : "higher is better";

  return `
    <tr>
      <td class="cmp-metric-label${col.primary ? " is-primary" : ""}">
        ${esc(col.label)}
        ${col.unit ? `<span class="cmp-metric-unit">${esc(col.unit)}</span>` : ""}
        <span class="cmp-metric-dir">${dirLabel}</span>
      </td>
      ${chips.map((_, i) => renderCmpCell(values[i], col, i === bestIdx, fracOf(values[i]))).join("")}
    </tr>
  `;
}

function renderCmpCell(value, col, isWinner, frac) {
  if (value === null || value === undefined) {
    return `
      <td class="cmp-cell is-missing">
        <div class="cmp-cell-value">-</div>
      </td>
    `;
  }
  const formatted = formatMetric(value, col);
  const pct = Math.max(2, Math.min(100, frac * 100));
  return `
    <td class="cmp-cell${isWinner ? " is-winner" : ""}">
      ${col.textual ? "" : `
        <div class="cmp-bar${isWinner ? " is-winner" : ""}">
          <div class="cmp-bar-fill" style="width:${pct}%"></div>
        </div>
      `}
      <div class="cmp-cell-value">
        ${esc(formatted)}
        ${col.unit && !col.textual ? `<span class="cmp-cell-unit">${esc(col.unit)}</span>` : ""}
      </div>
    </td>
  `;
}

// ── Share-link copy ───────────────────────────────────────────
//
// Build a self-contained URL that, when opened, restores the current
// basket on the active suite.  We always include `?runs=` (basket
// state isn't in `location.hash` after seed consumption) and pin the
// suite so the recipient lands on the exact view the sender saw.
function _shareUrlForBasket(suiteId) {
  const rids = basketGet();
  if (!rids.length) return location.href;
  const params = new URLSearchParams();
  params.set("runs", rids.join(","));
  if (suiteId) params.set("suite", suiteId);
  const base = location.origin + location.pathname + location.search;
  return `${base}#/compare?${params.toString()}`;
}

async function _copyShareLink(btn, suiteId) {
  const url = _shareUrlForBasket(suiteId);
  const ok = await copyToClipboard(url);
  flashButtonLabel(btn, ok ? "Copied!" : "Copy failed — select & ⌘C", {
    holdMs: ok ? 1600 : 3500,
    className: ok ? "is-copied" : "is-copy-failed",
    labelSelector: ".copy-btn-label",
  });
}

// ── Click delegation ──
//
// Attached once per mounted view; we re-derive the active suite from
// the URL on every click so a render-induced re-attach can never stack
// stale closures on top of each other.

function bindClicks(el) {
  if (el.__cmpClicksAttached) return;
  el.__cmpClicksAttached = true;

  el.addEventListener("click", (ev) => {
    if (!location.hash.startsWith("#/compare")) return;
    const { params: query } = parseHash(location.hash);
    const suiteId = SUITE_ORDER.includes(query.suite) ? query.suite : SUITE_ORDER[0];
    const t = ev.target;

    // Remove a single run from the basket.
    const removeBtn = t.closest("[data-remove-run]");
    if (removeBtn) {
      ev.preventDefault();
      basketToggle(removeBtn.dataset.removeRun);
      return;
    }

    // Clear all runs.
    if (t.closest("[data-basket-clear]")) {
      ev.preventDefault();
      for (const s of basketGet()) basketToggle(s);
      return;
    }

    // Copy a shareable URL.  The basket lives in-memory on the
    // /compare route (the seed ?runs= is stripped from the address bar
    // after consumption), so we have to recompose the canonical link
    // from the current basket + active suite at click time.
    const shareBtn = t.closest("[data-basket-share]");
    if (shareBtn) {
      ev.preventDefault();
      _copyShareLink(shareBtn, suiteId);
      return;
    }

    // Chip cloud quick-add: plain click on a tile adds (or removes, if
    // already in the basket) the chip's representative run.  Modifier
    // clicks fall through to the anchor's href so users can still pop
    // open the chip overview page in a new tab.
    const cloudTile = t.closest("[data-cmp-add-slug]");
    if (cloudTile) {
      if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button === 1) return;
      ev.preventDefault();
      const slug = cloudTile.dataset.cmpAddSlug;
      // If any basket run already shares this chip's name, treat the
      // click as a toggle: remove every matching basket entry instead
      // of adding a duplicate run for the same chip.
      const targetName = cloudTile.querySelector(".chip-tile-name")?.textContent || "";
      const matching = basketGet()
        .map((rid) => ({ rid, row: rowByRunId(rid) }))
        .filter((x) => x.row && x.row.chip === targetName);
      if (matching.length) {
        for (const m of matching) basketToggle(m.rid);
        return;
      }
      const rid = representativeRunForChip(slug);
      if (rid) basketToggle(rid);
      return;
    }

    // Suite switcher.  Always pin the explicit suite to the URL because
    // the default suite for compare is dynamic (first suite where any
    // basket run has data), so omitting the param would round-trip
    // back to the default instead of honoring the user's pick.
    const suitePill = t.closest(".rk-suite-pill");
    if (suitePill) {
      ev.preventDefault();
      const sid = suitePill.dataset.suite;
      if (sid && sid !== suiteId) {
        location.hash = buildHash("/compare", { suite: sid });
      }
      return;
    }

    // Chart download — sits inside any .cmp-chart-canvas wrapper
    // built by renderCmpCharts.  Resolve the canvas via DOM proximity.
    const dlBtn = t.closest("[data-chart-dl]");
    if (dlBtn) {
      ev.preventDefault();
      _downloadCmpChart(dlBtn, suiteId);
      return;
    }
  });
}

async function _downloadCmpChart(btn, suiteId) {
  const wrap = btn.closest(".cmp-chart-canvas");
  const canvas = wrap && wrap.querySelector("canvas");
  if (!canvas) {
    flashButtonLabel(btn, "Failed", { holdMs: 2000, className: "is-failed", labelSelector: ".chart-dl-btn-label" });
    return;
  }
  const sectionSlug = btn.dataset.chartDl || "chart";
  const ok = await downloadCanvasAsPng(canvas, {
    filename: `compare-${suiteId.replace(/^suite_/, "suite-")}-${sectionSlug}.png`,
  });
  flashButtonLabel(btn, ok ? "Saved" : "Failed", {
    holdMs: ok ? 1400 : 2200,
    className: ok ? "is-saved" : "is-failed",
    labelSelector: ".chart-dl-btn-label",
  });
}

// Subscribe once per mounted view container; when the shared basket
// store changes we re-render — but only if /compare is still the active
// route, otherwise we'd clobber whatever view the user navigated to.
function attachBasketListener(el) {
  if (el.__cmpBasketAttached) return;
  basketOnChange(() => {
    if (!location.hash.startsWith("#/compare")) return;
    const { params } = parseHash(location.hash);
    render({ el, query: params });
  });
  el.__cmpBasketAttached = true;
}
