// modal.js — Single-run detail modal overlay.
//
// Triggered by any element with `data-open-run="<run_id>"` anywhere in
// the app via document-level delegation, plus the public openModal()
// API for programmatic callers and a URL deep-link (`?run=<rid>`).
//
// Layout: 3 tabs.
//   • Details — hardware / software / model / run-settings / accuracy
//               / metadata key-value sections from row.detail.
//   • Visualize — Chart.js charts per suite type, driven by row.viz
//                 (offline bars, online latency lines, sustained
//                 throughput-over-time, scaling efficiency, etc.).
//   • Implementation — runner-code metadata from row.impl, including
//                      deprecation banner and links to runner.py.
//
// Inert when no `<script src="chart.umd*.js">` is present — the Viz tab
// silently falls back to "Charting library failed to load".  Modal
// itself never crashes the app.

import { rowByRunId, SUITE_META } from "./data.js";
import { esc, shortVersion, fmtDate, submitterHandle, chipSlug } from "./utils.js";

// ── Module state ──
let _modalEl = null;
let _activeCharts = [];
let _activeTab = "details";
let _lastFocus = null;
let _currentRow = null;
let _vizRenderedForRun = null;

// Suite letter -> CSS variable used for accent color binding on the
// modal shell.  The same tokens are already defined in base.css.
const SUITE_LETTERS = { suite_A: "A", suite_B: "B", suite_C: "C", suite_D: "D", suite_E: "E", suite_F: "F", suite_G: "G" };

// Chart palette — pulled from CSS variables so the modal reacts to the
// active light / dark scheme.  Resolved lazily so we read after the
// stylesheet has been parsed.
let _C = null;
function chartColors() {
  if (_C) return _C;
  const cs = getComputedStyle(document.documentElement);
  const v = (name, fallback) => (cs.getPropertyValue(name).trim() || fallback);
  _C = {
    blue:  "#378ADD",
    teal:  "#1D9E75",
    amber: "#EF9F27",
    coral: "#D85A30",
    purple:"#c084fc",
    green: "#56d364",
    red:   "#f85149",
    yellow:"#fbbf24",
    gray:  "#888780",
    text:  v("--fg-muted",   "#8b949e"),
    grid:  v("--border-soft","rgba(127,127,127,0.18)"),
  };
  return _C;
}

// ── Public API ──

// Inject the modal DOM and wire global listeners.  Idempotent.
export function initModal() {
  if (_modalEl) return;
  _modalEl = document.createElement("div");
  _modalEl.className = "modal-overlay";
  _modalEl.setAttribute("aria-hidden", "true");
  _modalEl.innerHTML = `
    <div class="modal-shell" role="dialog" aria-modal="true" aria-labelledby="run-modal-title" tabindex="-1">
      <header class="modal-header">
        <div class="modal-titles">
          <h2 class="modal-title" id="run-modal-title">—</h2>
          <a class="modal-chip-link" href="#/" title="See every run for this chip">View chip overview →</a>
          <p class="modal-subtitle">—</p>
        </div>
        <button class="modal-close" type="button" aria-label="Close">×</button>
      </header>
      <nav class="modal-tabs" role="tablist" aria-label="Run details">
        <button class="modal-tab" data-tab="details" type="button" role="tab" id="run-modal-tab-details" aria-controls="run-modal-panel-details">Details</button>
        <button class="modal-tab" data-tab="viz"     type="button" role="tab" id="run-modal-tab-viz"     aria-controls="run-modal-panel-viz">Visualize</button>
        <button class="modal-tab" data-tab="impl"    type="button" role="tab" id="run-modal-tab-impl"    aria-controls="run-modal-panel-impl">Implementation</button>
      </nav>
      <div class="modal-body">
        <section class="modal-panel" data-panel="details" role="tabpanel" id="run-modal-panel-details" aria-labelledby="run-modal-tab-details" tabindex="0"></section>
        <section class="modal-panel" data-panel="viz"     role="tabpanel" id="run-modal-panel-viz"     aria-labelledby="run-modal-tab-viz"     tabindex="0"></section>
        <section class="modal-panel" data-panel="impl"    role="tabpanel" id="run-modal-panel-impl"    aria-labelledby="run-modal-tab-impl"    tabindex="0"></section>
      </div>
      <footer class="modal-footer">
        <span class="modal-submission">—</span>
        <a class="modal-script-link" target="_blank" rel="noopener">View reproduce script ↗</a>
      </footer>
    </div>
  `;
  document.body.appendChild(_modalEl);

  // Backdrop click closes (anywhere outside the shell).
  _modalEl.addEventListener("click", (ev) => {
    if (ev.target === _modalEl) closeModal();
  });

  // Header / tab clicks within the modal.
  _modalEl.addEventListener("click", (ev) => {
    if (ev.target.closest(".modal-close")) { closeModal(); return; }
    const tab = ev.target.closest(".modal-tab");
    if (tab) { _setTab(tab.dataset.tab); return; }
  });

  // Esc closes from anywhere.
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && _modalEl.classList.contains("open")) {
      ev.preventDefault();
      closeModal();
    }
  });

  // Document-wide trigger.  Any element marked data-open-run="<rid>"
  // opens the modal — works inside any view without each view having
  // to wire its own handler.
  document.addEventListener("click", (ev) => {
    const t = ev.target.closest("[data-open-run]");
    if (!t) return;

    // Nested-anchor escape: when the click lands on a real `<a href>`
    // strictly inside the trigger (e.g. the chip-name link inside an
    // lb-row), let the browser follow that link instead of popping the
    // modal.  This is what lets `chip name → /chip/<slug>` and `row
    // body → run modal` coexist on the same row.
    const inner = ev.target.closest("a[href]");
    if (inner && inner !== t && t.contains(inner)) return;

    // Allow modifier-clicks (cmd/ctrl/middle) to behave like normal links
    // when the element is also an anchor with an href.
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button === 1) return;
    ev.preventDefault();
    openModal(t.dataset.openRun);
  });

  // Keyboard activation for the same `[data-open-run]` triggers.
  // Mirrors the click handler so non-mouse users can pop the modal
  // by tabbing onto a row / card and pressing Enter or Space — the
  // ARIA-recommended activation keys for role="button" surfaces.  We
  // skip text inputs / contenteditable so typing inside, say, a
  // future modal search field doesn't accidentally fire it.
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" && ev.key !== " " && ev.key !== "Spacebar") return;
    const t = ev.target.closest("[data-open-run]");
    if (!t) return;
    // If focus is on an inner link/button, let that element's own
    // activation handler win (matches the click nested-anchor escape).
    const inner = ev.target.closest("a[href], button, input, textarea, select, [contenteditable=true]");
    if (inner && inner !== t && t.contains(inner)) return;
    // Modifiers fall through to the browser's default (e.g. Cmd+Enter
    // on a focused link still opens in a new tab).
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
    ev.preventDefault();
    openModal(t.dataset.openRun);
  });

  // URL deep-link: `?run=<rid>` in the hash query opens the modal on
  // boot and on any subsequent hashchange (so e.g. switching from
  // /rankings?run=<rid> to /compare?run=<rid> still pops the modal).
  // An optional `&tab=viz|impl` lets a deep link target a specific tab.
  const syncFromHash = () => {
    const q = location.hash.split("?")[1] || "";
    const params = new URLSearchParams(q);
    const rid = params.get("run");
    const tab = params.get("tab");
    if (rid) {
      // Open in a microtask so the current view has time to render first.
      setTimeout(() => {
        openModal(rid);
        // Only honour the tab hint when the modal actually opened — if the
        // run_id was stale, openModal will have stripped ?run= for us and
        // poking a tab on a closed modal is just noise.
        if (_modalEl.classList.contains("open") && (tab === "viz" || tab === "impl")) {
          _setTab(tab);
        }
      }, 0);
    } else if (_modalEl.classList.contains("open")) {
      closeModal();
    }
  };
  window.addEventListener("hashchange", syncFromHash);
  syncFromHash();
}

export function openModal(runId) {
  if (!_modalEl) initModal();
  const row = rowByRunId(runId);
  if (!row) {
    // Stale or invalid run_id (e.g. a shared link to a run that has
    // since been removed from the dataset).  Strip the param so we
    // don't keep retrying on every hashchange and don't leave a
    // misleading ?run= in the address bar.
    if (runId) _setHashRunParam(null);
    return;
  }
  _currentRow = row;
  _vizRenderedForRun = null;
  _lastFocus = document.activeElement;
  _fillModal(row);
  // Default to Details on every open; Viz tab is lazy.
  _setTab("details");
  _modalEl.classList.add("open");
  _modalEl.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  const closeBtn = _modalEl.querySelector(".modal-close");
  if (closeBtn) closeBtn.focus();
  _setHashRunParam(runId);
}

export function closeModal() {
  if (!_modalEl || !_modalEl.classList.contains("open")) return;
  _destroyCharts();
  _modalEl.classList.remove("open");
  _modalEl.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  if (_lastFocus && typeof _lastFocus.focus === "function") {
    _lastFocus.focus();
  }
  _lastFocus = null;
  _currentRow = null;
  _vizRenderedForRun = null;
  _setHashRunParam(null);
}

// Update (or strip) the `run` query param inside the current hash using
// history.replaceState so we don't fire a hashchange event — the router
// would otherwise re-dispatch the current view and reset its scroll on
// every modal open / close.
function _setHashRunParam(runId) {
  const raw = location.hash || "";
  const stripped = raw.startsWith("#") ? raw.slice(1) : raw;
  const [path, qs = ""] = stripped.split("?");
  const params = new URLSearchParams(qs);
  if (runId) params.set("run", runId);
  else       params.delete("run");
  const nextQs = params.toString();
  const next = "#" + (path || "/") + (nextQs ? "?" + nextQs : "");
  if (next !== raw) {
    history.replaceState(null, "", location.pathname + location.search + next);
  }
}

// ── Header / footer ──

function _fillModal(row) {
  const d = row.detail || {};
  const meta = SUITE_META[row.suite];

  _modalEl.querySelector(".modal-shell").setAttribute(
    "data-suite", SUITE_LETTERS[row.suite] || ""
  );

  const titleEl = _modalEl.querySelector(".modal-title");
  titleEl.textContent =
    (d.hw_chip || row.chip || "—") +
    (row.chip_count > 1 ? ` ×${row.chip_count}` : "");

  // Drill-in link to the chip-detail page so users can see every run
  // for this hardware configuration without leaving via the URL bar.
  // Clicking the link triggers a hashchange which auto-closes the modal
  // (see syncFromHash in initModal).
  const chipLink = _modalEl.querySelector(".modal-chip-link");
  chipLink.setAttribute("href", `#/chip/${chipSlug(row)}`);

  const subEl = _modalEl.querySelector(".modal-subtitle");
  const parts = [];
  if (meta) {
    parts.push(
      `<span class="modal-suite-pill" data-suite="${esc(meta.letter)}">` +
      `Suite ${esc(meta.letter)} · ${esc(meta.title)}` +
      `</span>`
    );
  }
  const fwLine = [row.framework, shortVersion(row.framework_version)].filter(Boolean).join(" ");
  if (fwLine) parts.push(esc(fwLine));
  if (row.precision) {
    const fallback = row.precision_fallback
      ? ` <span class="modal-warn">(fallback)</span>`
      : "";
    parts.push(esc(row.precision) + fallback);
  }
  if (row.date) parts.push(esc(fmtDate(row.date)));
  subEl.innerHTML = parts.filter(Boolean).join(' <span class="sep">·</span> ');

  // Footer: submission + script link.
  const subInfo = _modalEl.querySelector(".modal-submission");
  const handle = submitterHandle(row.submitted_by);
  subInfo.textContent =
    `Submission: ${row.submission || row.run_id || "—"}` +
    (handle ? ` · by @${handle}` : "");

  const scriptLink = _modalEl.querySelector(".modal-script-link");
  const impl = row.impl || {};
  let scriptUrl = null;
  if (d.meta_reproduce_script) {
    scriptUrl = `https://github.com/JuhaoLiang1997/AccelMark/blob/main/${d.meta_reproduce_script}`;
    // If the result references an old (superseded) runner path, point to
    // the current runner.py instead.
    if (impl.supersedes_chain && Array.isArray(impl.supersedes_chain)) {
      const parts = d.meta_reproduce_script.split("/");
      const oldRunnerId = parts.length >= 2 ? parts[parts.length - 2] : null;
      if (oldRunnerId && impl.supersedes_chain.includes(oldRunnerId) && impl.runner_url) {
        scriptUrl = impl.runner_url;
      }
    }
  }
  if (scriptUrl) {
    scriptLink.href = scriptUrl;
    scriptLink.style.display = "";
  } else {
    scriptLink.removeAttribute("href");
    scriptLink.style.display = "none";
  }

  // Hide tabs whose data is missing.
  const vizTab  = _modalEl.querySelector('[data-tab="viz"]');
  const implTab = _modalEl.querySelector('[data-tab="impl"]');
  const hasViz = row.viz && row.viz.type && row.viz.type !== "none" && _vizHasAnyData(row.viz);
  vizTab.style.display  = hasViz ? "" : "none";
  implTab.style.display = row.impl ? "" : "none";

  // Render Details up-front; Viz is lazy in _setTab().
  _renderDetails(row, _modalEl.querySelector('[data-panel="details"]'));
  _renderImpl(row,    _modalEl.querySelector('[data-panel="impl"]'));
  // Clear viz panel; it'll be filled on first tab activation.
  _modalEl.querySelector('[data-panel="viz"]').innerHTML = "";
}

function _vizHasAnyData(viz) {
  if (!viz) return false;
  for (const k of Object.keys(viz)) {
    if (k === "type") continue;
    const v = viz[k];
    if (v && typeof v === "object" && Object.keys(v).length > 0) return true;
  }
  return false;
}

function _setTab(name) {
  _activeTab = name;
  for (const btn of _modalEl.querySelectorAll(".modal-tab")) {
    btn.classList.toggle("active", btn.dataset.tab === name);
    btn.setAttribute("aria-selected", btn.dataset.tab === name ? "true" : "false");
  }
  for (const panel of _modalEl.querySelectorAll(".modal-panel")) {
    panel.classList.toggle("active", panel.dataset.panel === name);
  }
  if (name === "viz" && _currentRow) {
    if (_vizRenderedForRun !== _currentRow.run_id) {
      _renderViz(_currentRow);
      _vizRenderedForRun = _currentRow.run_id;
    }
  }
  // Reset body scroll on tab change so users see the top of each panel.
  const body = _modalEl.querySelector(".modal-body");
  if (body) body.scrollTop = 0;
}

// ── Details panel ──

function _detailRow(label, v, opts = {}) {
  if (v === null || v === undefined || v === "" || v === "unknown" || v === "Unknown") {
    return `<div class="detail-row"><span class="detail-label">${esc(label)}</span><span class="detail-value null">—</span></div>`;
  }
  const cls = opts.cls ? " " + opts.cls : "";
  const mono = opts.mono ? " mono" : "";
  const display = opts.html ? String(v) : (opts.format ? opts.format(v) : esc(String(v)));
  return `<div class="detail-row"><span class="detail-label">${esc(label)}</span><span class="detail-value${mono}${cls}">${display}</span></div>`;
}

function _detailSection(title, rows) {
  const content = rows.filter(Boolean).join("");
  if (!content) return "";
  return `
    <div class="detail-section">
      <div class="detail-section-title">${esc(title)}</div>
      ${content}
    </div>
  `;
}

function _renderDetails(row, panel) {
  const d = row.detail || {};

  const html = [
    _detailSection("Hardware", [
      _detailRow("Chip",     d.hw_chip),
      _detailRow("Vendor",   d.hw_vendor),
      _detailRow("Count",    d.hw_count),
      _detailRow("Memory per chip", d.hw_memory_gb, { format: (v) => `${v} GB` }),
      _detailRow("Intra-node interconnect", d.hw_interconnect_intra),
      _detailRow("Inter-node interconnect", d.hw_interconnect_inter),
      _detailRow("CPU", d.hw_cpu),
      _detailRow("System memory", d.hw_system_memory_gb, { format: (v) => `${Number(v).toFixed(0)} GB` }),
      _detailRow("PCIe generation", d.hw_pcie),
      _detailRow("Network", d.hw_network),
    ]),
    _detailSection("Software", [
      _detailRow("Framework",
        d.sw_framework
          ? (d.sw_framework + (d.sw_framework_version ? " " + d.sw_framework_version : ""))
          : null),
      _detailRow("PyTorch",        d.sw_pytorch, { mono: true }),
      _detailRow("Driver version", d.sw_driver,  { mono: true }),
      _detailRow("Runtime",        d.sw_runtime, { mono: true }),
      _detailRow("OS",             d.sw_os),
      _detailRow("Python",         d.sw_python,  { mono: true }),
    ]),
    _detailSection("Model", [
      _detailRow("Model ID",   d.model_id,       { mono: true }),
      _detailRow("Revision",   d.model_revision, { mono: true }),
      d.model_name
        ? _detailRow("Actual model",
            d.model_name + ` <span class="muted-note">(substituted variant)</span>`,
            { html: true })
        : null,
      d.model_note ? _detailRow("Model note", d.model_note) : null,
      _detailRow("Architecture", d.model_arch),
      _detailRow("Parameters",   d.model_params_b, { format: (v) => `${v}B` }),
      _detailRow("Precision",    d.model_precision),
      d.model_effective_dtype
        ? _detailRow("Compute dtype",
            d.model_effective_dtype +
              (row.precision_emulated
                ? ` <span class="muted-note">(${esc(row.precision)} weights, ${esc(d.model_effective_dtype)} compute)</span>`
                : ""),
            { html: true })
        : null,
      d.model_quant_method ? _detailRow("Quantization", d.model_quant_method) : null,
      _detailRow("Format", d.model_format),
      d.model_source === "local"
        ? _detailRow("Model source", "local override", {})
        : null,
    ]),
    _detailSection("Run settings", [
      d.run_scenarios
        ? _detailRow("Scenarios", Array.isArray(d.run_scenarios) ? d.run_scenarios.join(", ") : d.run_scenarios)
        : null,
      d.run_chip_counts
        ? _detailRow("Chip counts tested", Array.isArray(d.run_chip_counts) ? d.run_chip_counts.join(", ") : d.run_chip_counts)
        : null,
      _detailRow("Runs per config",       d.run_num_runs),
      d.run_tp != null ? _detailRow("Tensor parallel size",   d.run_tp) : null,
      d.run_pp != null ? _detailRow("Pipeline parallel size", d.run_pp) : null,
      d.run_dp != null ? _detailRow("Data parallel size",     d.run_dp) : null,
    ]),
    _detailSection("Accuracy", [
      _detailRow("Subset score", d.acc_score, {
        format: (v) => Number(v).toFixed(2),
        cls: d.acc_valid ? "good" : "warn",
      }),
      _detailRow("Baseline delta", d.acc_baseline_delta, {
        format: (v) => (Number(v) >= 0 ? "+" : "") + Number(v).toFixed(3),
      }),
      d.acc_valid != null
        ? _detailRow("Valid",
            d.acc_valid ? `<span class="detail-value good">Yes</span>` : `<span class="detail-value warn">No</span>`,
            { html: true })
        : null,
      d.acc_notes ? _detailRow("Notes", d.acc_notes) : null,
    ]),
    _detailSection("Metadata", [
      _detailRow("Submitted by",    d.meta_submitted_by),
      _detailRow("Submission type", d.meta_submission_type),
      _detailRow("Date",            d.meta_date),
      d.meta_elapsed_min != null
        ? _detailRow("Benchmark duration", d.meta_elapsed_min, { format: (v) => `${Number(v).toFixed(1)} min` })
        : null,
      d.meta_model_load_sec != null
        ? _detailRow("Model load time", d.meta_model_load_sec, { format: (v) => `${Number(v).toFixed(1)} s` })
        : null,
      d.meta_notes ? _detailRow("Notes", d.meta_notes) : null,
    ]),
  ].join("");

  panel.innerHTML = html;
}

// ── Implementation panel ──

function _renderImpl(row, panel) {
  const impl = row.impl;
  if (!impl) { panel.innerHTML = ""; return; }
  const isDeprecated = !!impl.deprecated_by;

  panel.innerHTML = [
    isDeprecated
      ? `<div class="impl-deprecation">
           ⚠ This runner has been superseded by
           <span class="mono">${esc(impl.deprecated_by)}</span>.
           Results using this runner remain valid — the code is unchanged.
         </div>`
      : "",
    _detailSection("Runner", [
      _detailRow("ID",           impl.id,           { mono: true }),
      _detailRow("Platform",     impl.platform),
      _detailRow("Framework",    impl.framework),
      _detailRow("Name",         impl.name),
      _detailRow("Submitted by", impl.submitted_by),
      _detailRow("Created",      impl.created),
      impl.supersedes_chain && impl.supersedes_chain.length
        ? _detailRow("Replaces", impl.supersedes_chain[0], { mono: true })
        : null,
      impl.deprecated_by ? _detailRow("Deprecated by", impl.deprecated_by, { mono: true }) : null,
    ]),
    (impl.description || impl.notes)
      ? `<div class="detail-section">
           <div class="detail-section-title">Description</div>
           ${impl.description ? `<div class="impl-prose">${esc(impl.description)}</div>` : ""}
           ${impl.notes ? `<div class="detail-row"><span class="detail-label">Notes</span><span class="detail-value">${esc(impl.notes)}</span></div>` : ""}
         </div>`
      : "",
    impl.github_url || impl.runner_url
      ? `<div class="detail-section">
           <div class="detail-section-title">Source</div>
           ${impl.github_url ? `
             <div class="detail-row">
               <span class="detail-label">Runner folder</span>
               <a class="detail-link mono" href="${esc(impl.github_url)}" target="_blank" rel="noopener">
                 runners/${esc(impl.id)} ↗
               </a>
             </div>` : ""}
           ${impl.runner_url ? `
             <div class="detail-row">
               <span class="detail-label">runner.py</span>
               <a class="detail-link" href="${esc(impl.runner_url)}" target="_blank" rel="noopener">View source ↗</a>
             </div>` : ""}
         </div>`
      : "",
  ].filter(Boolean).join("");
}

// ── Visualize panel ──

// _renderViz lazily fills the Visualize tab with the right chart bundle
// for `row.viz.type`.  `panel` defaults to the modal's viz panel for the
// production code path; tests pass a fake element to drive the dispatch
// + per-suite renderer fallbacks without standing up the full modal DOM.
function _renderViz(row, panel) {
  if (!panel) panel = _modalEl.querySelector('[data-panel="viz"]');
  panel.innerHTML = "";

  if (typeof window.Chart !== "function") {
    panel.innerHTML = `<div class="viz-empty">Charting library failed to load.</div>`;
    return;
  }

  const viz = row.viz || {};
  if (!_vizHasAnyData(viz)) {
    panel.innerHTML = `<div class="viz-empty">No visualization data for this run.</div>`;
    return;
  }

  switch (viz.type) {
    case "suite_A": _renderSuiteAB(panel, row, viz, "blue"); break;
    case "suite_B": _renderSuiteB (panel, row, viz);          break;
    case "suite_C": _renderSuiteC (panel, row, viz);          break;
    case "suite_D": _renderSuiteD (panel, row, viz);          break;
    case "suite_E": _renderSuiteE (panel, row, viz);          break;
    case "suite_F": _renderSuiteAB(panel, row, viz, "purple"); break;
    case "suite_G": _renderSuiteG (panel, row, viz);          break;
    case "sustained": _renderSustained(panel, row, viz); break;
    default:
      panel.innerHTML = `<div class="viz-empty">Visualization for type "${esc(viz.type || "unknown")}" is not yet supported.</div>`;
  }

  // Suite-level rows often nest a sustained run inside the main viz.
  if (viz.type !== "sustained" && viz.sustained) {
    _renderSustained(panel, row, viz.sustained);
  }
}

function _destroyCharts() {
  for (const c of _activeCharts) {
    try { c.destroy(); } catch (e) { /* ignore */ }
  }
  _activeCharts = [];
}

// ── Chart utilities ──

function _section(label) {
  const d = document.createElement("div");
  d.className = "viz-section-title";
  d.textContent = label;
  return d;
}

function _statChips(items) {
  const grid = document.createElement("div");
  grid.className = "viz-stat-chips";
  for (const { label, value, cls } of items) {
    const card = document.createElement("div");
    card.className = "viz-stat-chip" + (cls ? " " + cls : "");
    card.innerHTML = `
      <div class="viz-stat-label">${esc(label)}</div>
      <div class="viz-stat-value">${value != null ? esc(String(value)) : "—"}</div>
    `;
    grid.appendChild(card);
  }
  return grid;
}

function _mkCanvas(height) {
  const wrap = document.createElement("div");
  wrap.className = "viz-chart-wrap";
  wrap.style.height = height + "px";
  const canvas = document.createElement("canvas");
  wrap.appendChild(canvas);
  return { wrap, canvas };
}

function _baseOpts(yLabel, yFormat, xLabel) {
  const C = chartColors();
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: C.text, font: { size: 11 } },
        grid:  { color: C.grid },
        title: { display: !!xLabel, text: xLabel || "", color: C.text, font: { size: 11 } },
      },
      y: {
        ticks: { color: C.text, font: { size: 11 }, callback: yFormat || ((v) => v) },
        grid:  { color: C.grid },
        title: { display: !!yLabel, text: yLabel || "", color: C.text, font: { size: 11 } },
      },
    },
  };
}

function _legend(items) {
  const leg = document.createElement("div");
  leg.className = "viz-legend";
  leg.innerHTML = items.map(({ color, label }) => `
    <span class="viz-legend-item">
      <span class="viz-legend-swatch" style="background:${color}"></span>
      ${esc(label)}
    </span>
  `).join("");
  return leg;
}

function _fmtKMs(v) {
  if (v == null) return "";
  if (v >= 1000) return (v / 1000).toFixed(1) + "s";
  return v + "ms";
}

// ── Per-suite renderers ──

// Shared offline + online + interactive + speculative + burst block.
// Used for suite_A and suite_F (just changes the accent color).
function _renderSuiteAB(el, row, viz, accent) {
  const C = chartColors();
  const accentHex = accent === "purple" ? C.purple : C.blue;

  if (viz.offline && viz.offline.labels && viz.offline.labels.length) {
    el.appendChild(_section("Offline — throughput by concurrency"));
    el.appendChild(_statChips([
      { label: "Peak throughput", value: row.offline_throughput
          ? row.offline_throughput.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "accent" },
      { label: "Peak memory",  value: row.peak_memory_gb ? row.peak_memory_gb.toFixed(1) + " GB" : null },
      { label: "Memory util.", value: row.memory_utilization_pct ? row.memory_utilization_pct.toFixed(1) + "%" : null },
    ]));
    const { wrap, canvas } = _mkCanvas(190);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "bar",
      data: {
        labels: viz.offline.labels,
        datasets: [{
          label: "Throughput (tok/s)",
          data: viz.offline.throughput,
          backgroundColor: accentHex + "99",
          borderColor:     accentHex,
          borderWidth: 1, borderRadius: 3,
        }],
      },
      options: _baseOpts("tokens / sec", (v) => v.toLocaleString(), "concurrency"),
    }));
  }

  if (viz.online && viz.online.labels && viz.online.labels.length) {
    el.appendChild(_section("Online — TTFT & TPOT by QPS"));
    _appendSlaRow(el, viz.online);
    const { wrap, canvas } = _mkCanvas(210);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "line",
      data: {
        labels: viz.online.labels,
        datasets: [
          { label: "TTFT p50", data: viz.online.ttft_p50, borderColor: accentHex, backgroundColor: accentHex + "33", tension: 0.3, pointRadius: 4, fill: false },
          { label: "TTFT p90", data: viz.online.ttft_p90, borderColor: C.coral,   backgroundColor: C.coral + "33",   tension: 0.3, pointRadius: 4, fill: false, borderDash: [4, 3] },
          { label: "TPOT p50", data: viz.online.tpot_p50, borderColor: C.teal,    backgroundColor: C.teal + "33",    tension: 0.3, pointRadius: 4, fill: false },
        ],
      },
      options: _baseOpts("latency (ms)", _fmtKMs, "target QPS"),
    }));
    el.appendChild(_legend([
      { color: accentHex, label: "TTFT p50" },
      { color: C.coral,   label: "TTFT p90" },
      { color: C.teal,    label: "TPOT p50" },
    ]));
  }

  if (viz.interactive && viz.interactive.ttft_p50 != null) {
    el.appendChild(_section("Interactive — single request latency"));
    el.appendChild(_statChips([
      { label: "TTFT p50", value: _ms(viz.interactive.ttft_p50), cls: "accent" },
      { label: "TTFT p90", value: _ms(viz.interactive.ttft_p90) },
      { label: "TTFT p99", value: _ms(viz.interactive.ttft_p99) },
      { label: "TPOT p50", value: _ms(viz.interactive.tpot_p50), cls: "good" },
      { label: "TPOT p99", value: _ms(viz.interactive.tpot_p99) },
    ]));
  }

  if (viz.speculative && viz.speculative.offline_tok_per_sec != null) {
    el.appendChild(_section("Speculative decoding"));
    const accPct = viz.speculative.acceptance_rate != null
      ? (viz.speculative.acceptance_rate * 100).toFixed(1) + "%"
      : null;
    el.appendChild(_statChips([
      { label: "Speculative tok/s", value: viz.speculative.offline_tok_per_sec
          ? viz.speculative.offline_tok_per_sec.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "warn" },
      { label: "Acceptance rate", value: accPct, cls: accPct ? "good" : null },
      { label: "Mean accepted",   value: viz.speculative.mean_accepted_tokens != null
          ? viz.speculative.mean_accepted_tokens.toFixed(2) + " tok" : null },
    ]));
  }

  if (viz.burst) {
    el.appendChild(_section("Burst load"));
    const degr = viz.burst.burst_degradation_ratio;
    const degrCls = degr != null ? (degr < 2 ? "good" : degr < 5 ? "warn" : "bad") : null;
    el.appendChild(_statChips([
      { label: "Degradation ratio", value: degr != null ? degr.toFixed(2) + "×" : null, cls: degrCls },
      { label: "Steady p99 TTFT",   value: viz.burst.steady_ttft_p99_ms != null ? viz.burst.steady_ttft_p99_ms.toFixed(0) + " ms" : null },
      { label: "Burst p99 TTFT",    value: viz.burst.burst_ttft_p99_ms  != null ? viz.burst.burst_ttft_p99_ms.toFixed(0)  + " ms" : null, cls: "bad" },
      { label: "SLA met during burst", value: viz.burst.sla_met_during_burst != null ? (viz.burst.sla_met_during_burst ? "✓" : "✗") : null,
        cls: viz.burst.sla_met_during_burst ? "good" : "bad" },
    ]));
  }
}

function _renderSuiteB(el, row, viz) {
  const C = chartColors();
  if (viz.offline && viz.offline.labels && viz.offline.labels.length) {
    el.appendChild(_section("Offline — total & per-chip throughput by concurrency"));
    el.appendChild(_statChips([
      { label: "Peak total",   value: row.offline_throughput      ? row.offline_throughput.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "accent" },
      { label: "Per chip",     value: row.tokens_per_sec_per_chip ? row.tokens_per_sec_per_chip.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "good" },
      { label: "Peak memory",  value: row.peak_memory_gb          ? row.peak_memory_gb.toFixed(1) + " GB" : null },
      { label: "Memory util.", value: row.memory_utilization_pct  ? row.memory_utilization_pct.toFixed(1) + "%" : null },
    ]));
    const { wrap, canvas } = _mkCanvas(210);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "bar",
      data: {
        labels: viz.offline.labels,
        datasets: [
          { label: "Total (tok/s)",   data: viz.offline.throughput,           backgroundColor: C.blue + "99", borderColor: C.blue, borderWidth: 1, borderRadius: 3 },
          { label: "Per chip (tok/s)",data: viz.offline.throughput_per_chip,  backgroundColor: C.teal + "99", borderColor: C.teal, borderWidth: 1, borderRadius: 3 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: true, labels: { color: C.text, font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid }, title: { display: true, text: "concurrency", color: C.text, font: { size: 11 } } },
          y: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v.toLocaleString() }, grid: { color: C.grid }, title: { display: true, text: "tokens / sec", color: C.text, font: { size: 11 } } },
        },
      },
    }));
  }

  if (viz.online && viz.online.labels && viz.online.labels.length) {
    el.appendChild(_section("Online — TTFT & TPOT by QPS"));
    _appendSlaRow(el, viz.online);
    const { wrap, canvas } = _mkCanvas(210);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "line",
      data: {
        labels: viz.online.labels,
        datasets: [
          { label: "TTFT p50", data: viz.online.ttft_p50, borderColor: C.blue,  backgroundColor: C.blue  + "33", tension: 0.3, pointRadius: 4, fill: false },
          { label: "TTFT p90", data: viz.online.ttft_p90, borderColor: C.coral, backgroundColor: C.coral + "33", tension: 0.3, pointRadius: 4, fill: false, borderDash: [4, 3] },
          { label: "TPOT p50", data: viz.online.tpot_p50, borderColor: C.teal,  backgroundColor: C.teal  + "33", tension: 0.3, pointRadius: 4, fill: false },
        ],
      },
      options: _baseOpts("latency (ms)", _fmtKMs, "target QPS"),
    }));
    el.appendChild(_legend([
      { color: C.blue,  label: "TTFT p50" },
      { color: C.coral, label: "TTFT p90" },
      { color: C.teal,  label: "TPOT p50" },
    ]));
  }

  if (viz.burst) {
    el.appendChild(_section("Burst load"));
    const degr = viz.burst.burst_degradation_ratio;
    const degrCls = degr != null ? (degr < 2 ? "good" : degr < 5 ? "warn" : "bad") : null;
    el.appendChild(_statChips([
      { label: "Degradation ratio", value: degr != null ? degr.toFixed(2) + "×" : null, cls: degrCls },
      { label: "Steady p99 TTFT",   value: viz.burst.steady_ttft_p99_ms != null ? viz.burst.steady_ttft_p99_ms.toFixed(0) + " ms" : null },
      { label: "Burst p99 TTFT",    value: viz.burst.burst_ttft_p99_ms  != null ? viz.burst.burst_ttft_p99_ms.toFixed(0)  + " ms" : null, cls: "bad" },
      { label: "SLA met during burst", value: viz.burst.sla_met_during_burst != null ? (viz.burst.sla_met_during_burst ? "✓" : "✗") : null,
        cls: viz.burst.sla_met_during_burst ? "good" : "bad" },
    ]));
  }
}

function _renderSuiteC(el, row, viz) {
  if (!viz.precisions || !viz.precisions.length) {
    el.innerHTML = `<div class="viz-empty">No quantization data available.</div>`;
    return;
  }
  const C = chartColors();
  const PREC_COLORS = { BF16: C.teal, FP8: C.blue, W8A8: C.amber, W8A16: C.coral, W4A16: C.green };
  const precColor = (p) => PREC_COLORS[p] || C.gray;

  el.appendChild(_section("Offline — quantization efficiency"));

  // Per-format summary table.
  const tbl = document.createElement("table");
  tbl.className = "viz-quant-table";
  tbl.innerHTML = `
    <thead>
      <tr>
        <th>Format</th>
        <th>Throughput</th>
        <th>Speedup</th>
        <th>Accuracy</th>
        <th>Quality eff.</th>
      </tr>
    </thead>
    <tbody>
      ${viz.precisions.map((prec, i) => {
        const thr   = viz.throughput[i];
        const spd   = viz.speedup ? viz.speedup[i] : null;
        const acc   = viz.accuracies ? viz.accuracies[i] : null;
        const qe    = viz.quality_efficiency ? viz.quality_efficiency[i] : null;
        const valid = viz.acc_valid ? viz.acc_valid[i] : null;
        const accCls = valid === false ? "bad" : valid === true ? "good" : "muted";
        return `
          <tr>
            <td><strong>${esc(prec)}</strong></td>
            <td class="tnum">${thr != null ? esc(thr.toLocaleString(undefined, { maximumFractionDigits: 0 })) + " tok/s" : "—"}</td>
            <td class="tnum">${spd != null ? esc(spd.toFixed(2)) + "×" : "—"}</td>
            <td class="tnum ${accCls}">${acc != null ? esc(acc.toFixed(4)) : "—"}</td>
            <td class="tnum">${qe  != null ? esc(qe.toLocaleString(undefined, { maximumFractionDigits: 0 })) : "—"}</td>
          </tr>
        `;
      }).join("")}
    </tbody>
  `;
  el.appendChild(tbl);

  // Throughput bar by format.
  const { wrap: w1, canvas: c1 } = _mkCanvas(200);
  el.appendChild(w1);
  _activeCharts.push(new window.Chart(c1, {
    type: "bar",
    data: {
      labels: viz.precisions,
      datasets: [{
        label: "Throughput",
        data: viz.throughput,
        backgroundColor: viz.precisions.map((p) => precColor(p) + "99"),
        borderColor:     viz.precisions.map((p) => precColor(p)),
        borderWidth: 1, borderRadius: 3,
      }],
    },
    options: _baseOpts("tokens / sec", (v) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }), "precision"),
  }));

  if (viz.online_by_precision && viz.online_by_precision.length) {
    el.appendChild(_section("Online — TTFT p99 by QPS across formats"));
    const qpsLabels = viz.online_by_precision[0].qps_labels;
    const { wrap, canvas } = _mkCanvas(240);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "line",
      data: {
        labels: qpsLabels,
        datasets: viz.online_by_precision.map((fp) => ({
          label: fp.precision,
          data:  fp.ttft_p99,
          borderColor:     precColor(fp.precision),
          backgroundColor: precColor(fp.precision) + "22",
          pointBackgroundColor: fp.sla_met.map((m) => (m === false ? C.red : precColor(fp.precision))),
          pointRadius: 5, tension: 0.2, fill: false,
        })),
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: true, labels: { color: C.text, font: { size: 11 }, boxWidth: 12 } } },
        scales: {
          x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid }, title: { display: true, text: "target QPS", color: C.text, font: { size: 11 } } },
          y: { ticks: { color: C.text, font: { size: 11 }, callback: (v) => v + "ms" }, grid: { color: C.grid }, title: { display: true, text: "TTFT p99 (ms)", color: C.text, font: { size: 11 } } },
        },
      },
    }));
  }
}

function _renderSuiteD(el, row, viz) {
  const C = chartColors();

  if (viz.interactive && viz.interactive.ttft_p50 != null) {
    const iv = viz.interactive;
    el.appendChild(_section("Long-context latency (~28K input tokens)"));
    el.appendChild(_statChips([
      { label: "TTFT p50", value: _ms(iv.ttft_p50, 0), cls: "accent" },
      { label: "TTFT p90", value: _ms(iv.ttft_p90, 0), cls: "accent" },
      { label: "TTFT p99", value: _ms(iv.ttft_p99, 0), cls: "accent" },
      { label: "TPOT p50", value: _ms(iv.tpot_p50, 1), cls: "good" },
      { label: "TPOT p90", value: _ms(iv.tpot_p90, 1), cls: "good" },
      { label: "TPOT p99", value: _ms(iv.tpot_p99, 1), cls: "good" },
    ]));
    const { wrap, canvas } = _mkCanvas(240);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "bar",
      data: {
        labels: ["TTFT p50", "TTFT p90", "TTFT p99", "TPOT p50", "TPOT p90", "TPOT p99"],
        datasets: [
          { label: "TTFT — prefill", data: [iv.ttft_p50, iv.ttft_p90, iv.ttft_p99, null, null, null],
            backgroundColor: C.blue + "99", borderColor: C.blue, borderWidth: 1, borderRadius: 3 },
          { label: "TPOT — decode",  data: [null, null, null, iv.tpot_p50, iv.tpot_p90, iv.tpot_p99],
            backgroundColor: C.teal + "99", borderColor: C.teal, borderWidth: 1, borderRadius: 3 },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: true, labels: { color: C.text, font: { size: 11 } } } },
        scales: {
          x: { ticks: { color: C.text, font: { size: 11 }, callback: _fmtKMs }, grid: { color: C.grid }, title: { display: true, text: "latency", color: C.text, font: { size: 11 } } },
          y: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid } },
        },
      },
    }));
  }

  if (viz.offline && viz.offline.labels && viz.offline.labels.length) {
    el.appendChild(_section("Offline throughput (~28K context)"));
    el.appendChild(_statChips([
      { label: "Peak throughput", value: row.offline_throughput ? row.offline_throughput.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "accent" },
      { label: "Peak memory",     value: row.peak_memory_gb    ? row.peak_memory_gb.toFixed(1) + " GB" : null },
    ]));
    const { wrap, canvas } = _mkCanvas(160);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "bar",
      data: {
        labels: viz.offline.labels,
        datasets: [{ data: viz.offline.throughput, backgroundColor: C.blue + "99", borderColor: C.blue, borderWidth: 1, borderRadius: 3 }],
      },
      options: _baseOpts("tokens / sec", (v) => v.toLocaleString(), "concurrency"),
    }));
  }
}

function _renderSuiteE(el, row, viz) {
  const C = chartColors();
  const counts = viz.chip_counts || [];
  const labels = counts.map((c) => c + "× GPU");

  el.appendChild(_section("Multi-chip scaling"));
  el.appendChild(_statChips([
    { label: "1× baseline",     value: viz.throughput[0] ? viz.throughput[0].toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "accent" },
    { label: "2× throughput",   value: viz.throughput[1] ? viz.throughput[1].toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null },
    { label: "4× throughput",   value: viz.throughput[2] ? viz.throughput[2].toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null },
    { label: "2× efficiency",   value: row.scaling_efficiency_2x != null ? (row.scaling_efficiency_2x * 100).toFixed(1) + "%" : null, cls: row.scaling_efficiency_2x >= 0.8 ? "good" : "warn" },
    { label: "4× efficiency",   value: row.scaling_efficiency_4x != null ? (row.scaling_efficiency_4x * 100).toFixed(1) + "%" : null,
      cls: row.scaling_efficiency_4x >= 0.8 ? "good" : row.scaling_efficiency_4x >= 0.5 ? "warn" : "bad" },
  ]));

  const { wrap: w1, canvas: c1 } = _mkCanvas(190);
  el.appendChild(w1);
  _activeCharts.push(new window.Chart(c1, {
    type: "bar",
    data: { labels, datasets: [{ data: viz.throughput, backgroundColor: C.blue + "99", borderColor: C.blue, borderWidth: 1, borderRadius: 3 }] },
    options: _baseOpts("tokens / sec", (v) => v ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : ""),
  }));

  el.appendChild(_section("Scaling efficiency vs ideal"));
  const ideal  = counts.map(() => 100);
  const target = counts.map(() => 80);
  const { wrap: w2, canvas: c2 } = _mkCanvas(220);
  el.appendChild(w2);
  _activeCharts.push(new window.Chart(c2, {
    type: "line",
    data: { labels, datasets: [
      { label: "Actual",         data: viz.efficiency_pct, borderColor: C.blue, backgroundColor: C.blue + "33", tension: 0.3, pointRadius: 5, pointBackgroundColor: C.blue, fill: true },
      { label: "Linear ideal",   data: ideal,              borderColor: C.gray, borderDash: [6, 4], tension: 0, pointRadius: 0, fill: false },
      { label: "Good (80%)",     data: target,             borderColor: C.teal, borderDash: [3, 3], tension: 0, pointRadius: 0, fill: false },
    ]},
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: C.text, font: { size: 11 } }, grid: { color: C.grid } },
        y: { min: 0, max: 110, ticks: { color: C.text, font: { size: 11 }, callback: (v) => v + "%" }, grid: { color: C.grid }, title: { display: true, text: "efficiency %", color: C.text, font: { size: 11 } } },
      },
    },
  }));
  el.appendChild(_legend([
    { color: C.blue, label: "Actual"        },
    { color: C.gray, label: "Linear ideal"  },
    { color: C.teal, label: "Good (80%)"    },
  ]));

  if (viz.throughput_per_chip && viz.throughput_per_chip.some((v) => v != null)) {
    el.appendChild(_section("Throughput per chip"));
    const { wrap: w3, canvas: c3 } = _mkCanvas(160);
    el.appendChild(w3);
    _activeCharts.push(new window.Chart(c3, {
      type: "bar",
      data: { labels, datasets: [{ data: viz.throughput_per_chip, backgroundColor: C.teal + "99", borderColor: C.teal, borderWidth: 1, borderRadius: 3 }] },
      options: _baseOpts("tok/s per chip", (v) => v ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : ""),
    }));
  }
}

function _renderSuiteG(el, row, viz) {
  const C = chartColors();
  const accentHex = "#2dd4bf";

  if (viz.offline && viz.offline.labels && viz.offline.labels.length) {
    el.appendChild(_section("Offline — throughput by concurrency"));
    el.appendChild(_statChips([
      { label: "Peak throughput", value: row.offline_throughput ? row.offline_throughput.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "good" },
      { label: "Peak memory",     value: row.peak_memory_gb    ? row.peak_memory_gb.toFixed(1) + " GB" : null },
    ]));
    const { wrap, canvas } = _mkCanvas(190);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "bar",
      data: {
        labels: viz.offline.labels,
        datasets: [{ label: "Throughput (tok/s)", data: viz.offline.throughput, backgroundColor: accentHex + "99", borderColor: accentHex, borderWidth: 1, borderRadius: 3 }],
      },
      options: _baseOpts("tokens / sec", (v) => v.toLocaleString(), "concurrency"),
    }));
  }

  if (viz.online && viz.online.labels && viz.online.labels.length) {
    el.appendChild(_section("Online — TTFT & TPOT by QPS"));
    _appendSlaRow(el, viz.online);
    const { wrap, canvas } = _mkCanvas(210);
    el.appendChild(wrap);
    _activeCharts.push(new window.Chart(canvas, {
      type: "line",
      data: {
        labels: viz.online.labels,
        datasets: [
          { label: "TTFT p50", data: viz.online.ttft_p50, borderColor: accentHex, backgroundColor: accentHex + "33", tension: 0.3, pointRadius: 4, fill: false },
          { label: "TTFT p90", data: viz.online.ttft_p90, borderColor: C.coral,    backgroundColor: C.coral + "33",   tension: 0.3, pointRadius: 4, fill: false, borderDash: [4, 3] },
          { label: "TPOT p50", data: viz.online.tpot_p50, borderColor: C.teal,     backgroundColor: C.teal + "33",    tension: 0.3, pointRadius: 4, fill: false },
        ],
      },
      options: _baseOpts("latency (ms)", _fmtKMs, "target QPS"),
    }));
    el.appendChild(_legend([
      { color: accentHex, label: "TTFT p50" },
      { color: C.coral,   label: "TTFT p90" },
      { color: C.teal,    label: "TPOT p50" },
    ]));
  }

  if (viz.interactive && viz.interactive.ttft_p50 != null) {
    el.appendChild(_section("Interactive — single request latency"));
    el.appendChild(_statChips([
      { label: "TTFT p50", value: _ms(viz.interactive.ttft_p50, 1), cls: "good" },
      { label: "TTFT p90", value: _ms(viz.interactive.ttft_p90, 1) },
      { label: "TTFT p99", value: _ms(viz.interactive.ttft_p99, 1) },
      { label: "TPOT p50", value: _ms(viz.interactive.tpot_p50, 1) },
      { label: "TPOT p99", value: _ms(viz.interactive.tpot_p99, 1) },
    ]));
  }

  if (viz.runtime_metrics) {
    const rm = viz.runtime_metrics;
    const items = [];
    if (rm.expert_load_balance != null) items.push({ label: "Expert load balance (std)", value: rm.expert_load_balance.toFixed(3) });
    if (rm.mean_experts_per_token != null) items.push({ label: "Mean experts/token", value: rm.mean_experts_per_token.toFixed(1) });
    if (items.length) {
      el.appendChild(_section("MoE runtime metrics"));
      el.appendChild(_statChips(items));
    }
  }
}

function _renderSustained(el, row, viz) {
  const C = chartColors();
  const dur  = viz.duration_minutes != null ? viz.duration_minutes : "?";
  const conc = viz.sustained_concurrency != null ? viz.sustained_concurrency : "?";
  const mean  = viz.sustained_throughput;
  const thr   = viz.throttle_ratio;
  const onset = viz.throttle_onset_minute;

  el.appendChild(_section(`Sustained load — ${dur} min @ concurrency ${conc}`));
  el.appendChild(_statChips([
    { label: "Sustained throughput", value: mean != null
        ? mean.toLocaleString(undefined, { maximumFractionDigits: 0 }) + " tok/s" : null, cls: "accent" },
    { label: "Concurrency",          value: viz.sustained_concurrency != null ? viz.sustained_concurrency + " in-flight" : null },
    { label: "Throttle ratio",       value: thr != null ? (thr * 100).toFixed(1) + "%" : null,
      cls: thr >= 0.95 ? "good" : thr >= 0.85 ? "warn" : "bad" },
    { label: "Throttle onset",       value: onset != null ? "min " + onset.toFixed(1) : "No throttling detected",
      cls: onset == null ? "good" : "warn" },
  ]));

  const samples = viz.samples;
  if (!samples || !samples.length) return;
  const minutes = samples.map((s) => s.minute);
  const tputs   = samples.map((s) => (s.throughput_tokens_per_sec == null ? null : s.throughput_tokens_per_sec));
  const warmup  = samples.filter((s) => s.is_warmup).length;
  const pointColors = minutes.map((_, i) => i < warmup ? C.gray : C.blue);
  const pointRadii  = minutes.map((_, i) => i < warmup ? 2 : 3);

  el.appendChild(_section("Throughput over time"));
  const { wrap: w1, canvas: c1 } = _mkCanvas(200);
  el.appendChild(w1);
  const datasets = [{
    label: "Throughput", data: tputs,
    borderColor: C.blue, backgroundColor: C.blue + "22",
    fill: true, tension: 0.3,
    pointBackgroundColor: pointColors, pointRadius: pointRadii,
  }];
  if (mean != null) {
    datasets.push({
      label: "Sustained mean", data: new Array(tputs.length).fill(mean),
      borderColor: C.teal, borderDash: [6, 3], borderWidth: 1.5, pointRadius: 0,
    });
  }
  _activeCharts.push(new window.Chart(c1, {
    type: "line",
    data: { labels: minutes.map((m) => m + " min"), datasets },
    options: {
      ..._baseOpts("tok/s", (v) => (v != null ? v.toFixed(0) : "")),
      plugins: { legend: { display: true, labels: { color: C.text, font: { size: 11 } } } },
    },
  }));
  if (warmup > 0) {
    const note = document.createElement("div");
    note.className = "viz-note";
    note.textContent = `First ${warmup} sample${warmup === 1 ? "" : "s"} are warmup (gray) and excluded from scalar metrics.`;
    el.appendChild(note);
  }

  const ttfts = samples.map((s) => (s.ttft_ms_p99 == null ? null : s.ttft_ms_p99));
  if (ttfts.some((v) => v != null)) {
    el.appendChild(_section("TTFT p99 over time"));
    const { wrap: w2, canvas: c2 } = _mkCanvas(160);
    el.appendChild(w2);
    _activeCharts.push(new window.Chart(c2, {
      type: "line",
      data: { labels: minutes.map((m) => m + " min"),
        datasets: [{ label: "TTFT p99", data: ttfts, borderColor: C.yellow, backgroundColor: C.yellow + "22", fill: true, tension: 0.3, pointRadius: 2 }] },
      options: { ..._baseOpts("ms", (v) => (v != null ? v.toFixed(0) : "")), plugins: { legend: { display: false } } },
    }));
  }
}

// ── Small shared bits ──

function _appendSlaRow(el, online) {
  const C = chartColors();
  const slaDiv = document.createElement("div");
  slaDiv.className = "viz-sla-row";
  for (let i = 0; i < online.labels.length; i++) {
    const qps = online.labels[i];
    const pass = online.sla_met[i];
    const cls = pass ? "pass" : "fail";
    slaDiv.innerHTML += `<span class="viz-sla-item">
      <span class="viz-sla-dot ${cls}"></span>
      <span class="qps">${esc(String(qps))} QPS</span>
      <span class="${cls}">${pass ? "SLA met" : "SLA fail"}</span>
    </span>`;
  }
  el.appendChild(slaDiv);
}

function _ms(v, decimals = 1) {
  if (v == null) return null;
  return Number(v).toFixed(decimals) + " ms";
}

// Test-only escape hatch.  Kept namespaced under an underscore so it
// reads as "internals — don't depend on this from app code"; node tests
// import it to drive _renderViz against fake panels + a stubbed Chart
// constructor without booting the full modal shell.
export const _test = {
  vizHasAnyData: _vizHasAnyData,
  renderViz: (panel, row) => _renderViz(row, panel),
  destroyCharts: _destroyCharts,
  // Lets a test reset the cached chartColors() result between cases —
  // the cache reads CSS custom-properties on first access and would
  // otherwise pin to whatever the first test happened to expose.
  resetColorCache: () => { _C = null; },
};
