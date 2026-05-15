// views/rankings.js — Per-suite rankings table with filters and compare basket.
//
// Page anatomy:
//   • Compact hero with the active suite's eyebrow + title + tagline.
//   • Sticky toolbar:
//       row 1 — 7 suite filter pills (color-bound to --suite-X).
//       row 2 — multi-select vendor / precision / framework facet pills,
//               populated from the rows in the active suite (filters that
//               would empty the set get hidden, not greyed).
//       row 3 — status line: "N of M results · sorted by …"
//   • Data table:
//       columns: ☐ · # · Chip · Vendor · Framework · Precision · primary
//                metric · secondary metrics · Date · Tier.
//       Basket key is run_id so every row toggles independently — same
//       chip with vLLM vs SGLang (or a different precision / version)
//       never share state.  The whole leftmost cell wraps the checkbox
//       so the hit-area is generous and the user doesn't have to land
//       on the tiny input itself.
//       click chip link → chip detail; click checkbox cell → toggle
//       compare basket; click sortable header → toggle / set sort
//       order; click any other inert area on the row → chip detail.
//   • Empty state appears when filters return no rows.
//
// All state (suite, sort, vendor/precision/framework filters) lives in the
// URL hash so views are bookmarkable and shareable.

import {
  SUITE_ORDER, SUITE_META, VENDOR_ORDER,
  SUITE_COLUMNS, formatMetric,
  rowsForSuite,
} from "../data.js";
import {
  esc, fmtNum, fmtDate, chipHref, buildHash, parseHash,
  shortVersion,
} from "../utils.js";
import {
  basketHas, basketToggle, basketGet, basketOnChange,
} from "../router.js";

export function render({ el, query }) {
  const suiteId = SUITE_ORDER.includes(query.suite) ? query.suite : SUITE_ORDER[0];
  const meta = SUITE_META[suiteId];
  const cols = SUITE_COLUMNS[suiteId];
  const primary = cols.find((c) => c.primary) || cols[0];

  const vendorFilter    = parseCsv(query.vendor);
  const precisionFilter = parseCsv(query.precision);
  const frameworkFilter = parseCsv(query.framework);
  // `?chip=<slug>` lets the chip-detail KPI cards deep-link straight to
  // a rankings view scoped to that chip variant.  We don't expose it as
  // a facet pill — it's a soft filter set externally and announced via
  // a banner that offers a one-click "show all chips" escape.
  const chipFilter      = (query.chip || "").trim();

  // Sort: ?sort=key:dir, validated against the active suite's columns.
  let sortKey = primary.key;
  let sortDir = primary.direction;
  if (query.sort) {
    const [k, d] = String(query.sort).split(":");
    if (cols.some((c) => c.key === k)) {
      sortKey = k;
      sortDir = d === "asc" ? "asc" : "desc";
    }
  }

  // Facet sets are derived from the suite's rows (not the post-filter
  // subset) so toggling one filter doesn't erase the others.
  const allRows = rowsForSuite(suiteId);
  const facets = buildFacets(allRows);

  const filtered = allRows.filter((r) =>
    matchSet(vendorFilter,    r.vendor) &&
    matchSet(precisionFilter, r.precision) &&
    matchSet(frameworkFilter, r.framework) &&
    (!chipFilter || r._chip_slug === chipFilter)
  );

  const sorted = [...filtered].sort((a, b) => safeCompare(a[sortKey], b[sortKey], sortDir));

  const filtersActive = !!(query.vendor || query.precision || query.framework || chipFilter);

  // Banner label for the chip-focus mode: now that `?chip=<slug>`
  // matches every chip_count variant of a single chip model (×1 / ×4 /
  // ×8 share a slug), use the bare chip name rather than `_chip_label`
  // — the latter carries "×N" which would mislabel the banner as
  // scoped to a specific fan-out.  We also surface how many distinct
  // chip-count variants the filter is showing so the count isn't
  // surprising.
  const chipFocusLabel = chipFilter
    ? (filtered[0]?.chip || allRows.find((r) => r._chip_slug === chipFilter)?.chip || chipFilter)
    : "";
  const chipFocusVariants = chipFilter
    ? new Set(filtered.map((r) => r.chip_count || 1)).size
    : 0;

  el.innerHTML = `
    <section class="rk-hero" data-suite="${esc(meta.letter)}">
      <span class="eyebrow">Rankings · Suite ${esc(meta.letter)}</span>
      <h1 class="rk-hero-title">${esc(meta.title)}</h1>
      <p class="rk-hero-sub">${esc(meta.tagline)}</p>
    </section>

    <section class="rk-toolbar">
      <div class="rk-suite-pills" role="tablist" aria-label="Workload suite">
        ${SUITE_ORDER.map((sid) => renderSuitePill(sid, sid === suiteId)).join("")}
      </div>

      <div class="rk-filter-row">
        ${renderFacetGroup("vendor",    "Vendor",    facets.vendor,    vendorFilter,    suiteId, query)}
        ${renderFacetGroup("precision", "Precision", facets.precision, precisionFilter, suiteId, query)}
        ${renderFacetGroup("framework", "Framework", facets.framework, frameworkFilter, suiteId, query)}
        ${filtersActive
          ? `<a class="rk-clear-all" data-clear-all="1" href="${esc(buildHash("/rankings", suiteUrlParam(suiteId)))}">Clear filters</a>`
          : ""}
      </div>

      <div class="rk-status">
        <span class="rk-count">
          <strong class="tnum">${fmtNum(sorted.length)}</strong>
          of <span class="tnum">${fmtNum(allRows.length)}</span> results
        </span>
        <span class="rk-sortby">
          Sorted by
          <strong>${esc(colLabel(cols, sortKey))}</strong>
          <span class="rk-sort-arrow">${sortDir === "asc" ? "↑" : "↓"}</span>
        </span>
      </div>
    </section>

    ${chipFilter ? `
      <div class="rk-chip-focus" data-suite="${esc(meta.letter)}">
        <span class="rk-chip-focus-msg">
          Showing only <strong>${esc(chipFocusLabel)}</strong>${chipFocusVariants > 1 ? ` <span class="rk-chip-focus-variants">(across ${chipFocusVariants} chip-count variants)</span>` : ""} in Suite ${esc(meta.letter)}.
        </span>
        <span class="rk-chip-focus-actions">
          <a class="btn ghost small" href="#/chip/${esc(chipFilter)}">View chip overview</a>
          <a class="btn ghost small" href="${esc(buildHash("/rankings", { ...query, chip: undefined }))}">Show all chips</a>
        </span>
      </div>
    ` : ""}

    <div class="rk-basket-bar ${basketGet().length ? "show" : ""}">
      <span class="rk-basket-msg">
        <strong class="tnum">${fmtNum(basketGet().length)}</strong>
        ${basketGet().length === 1 ? "run" : "runs"} in your compare basket
      </span>
      <span class="rk-basket-actions">
        <a class="btn primary small" href="#/compare">Open compare →</a>
        <button class="btn ghost small" data-basket-clear="1" type="button">Clear basket</button>
      </span>
    </div>

    <section class="rk-table-section">
      ${sorted.length === 0
        ? renderEmpty(meta, filtersActive)
        : renderTable(suiteId, sorted, cols, sortKey, sortDir)}
    </section>
  `;

  bindClicks(el);

  // Keep the table's checked state, the basket banner, and the in-basket
  // row highlight in sync with the shared basket store.  Attach once per
  // mounted view container to avoid stacking listeners across re-renders.
  if (!el.__rkBasketAttached) {
    basketOnChange(() => {
      const tbody = el.querySelector(".data-table tbody");
      if (tbody) {
        for (const tr of tbody.querySelectorAll("tr[data-run-id]")) {
          const rid = tr.dataset.runId;
          const inBasket = basketHas(rid);
          tr.classList.toggle("in-basket", inBasket);
          const cb = tr.querySelector(".compare-checkbox");
          if (cb) cb.checked = inBasket;
        }
      }
      const bar = el.querySelector(".rk-basket-bar");
      if (bar) {
        const n = basketGet().length;
        bar.classList.toggle("show", n > 0);
        const msg = bar.querySelector(".rk-basket-msg");
        if (msg) {
          msg.innerHTML = `
            <strong class="tnum">${fmtNum(n)}</strong>
            ${n === 1 ? "run" : "runs"} in your compare basket
          `;
        }
      }
    });
    el.__rkBasketAttached = true;
  }
}

// ── Toolbar bits ──

function renderSuitePill(sid, active) {
  const m = SUITE_META[sid];
  return `
    <button class="rk-suite-pill ${active ? "active" : ""}"
            data-suite="${esc(sid)}"
            data-suite-letter="${esc(m.letter)}"
            type="button"
            aria-pressed="${active ? "true" : "false"}"
            title="${esc(m.tagline)}">
      <span class="rk-suite-letter">${esc(m.letter)}</span>
      <span class="rk-suite-name">${esc(m.title)}</span>
    </button>
  `;
}

function renderFacetGroup(facetKey, label, options, selected, suiteId, query) {
  if (options.length === 0) return "";
  const sel = new Set(selected);
  // Each pill is an anchor whose `href` is the URL the click would
  // navigate to (toggling this value into the comma-separated facet
  // CSV).  That gives middle-click and Cmd/Ctrl-click native open-in-
  // new-tab semantics for free; the listener still preventDefaults on
  // plain left-clicks and routes through the SPA in-place.
  return `
    <div class="rk-facet" data-facet-group="${esc(facetKey)}">
      <span class="rk-facet-label">${esc(label)}</span>
      <div class="rk-facet-pills">
        ${options.map((opt) => {
          const next = toggleInCsv(query[facetKey], opt);
          const href = rebuildHash(suiteId, query, { [facetKey]: next || undefined });
          const active = sel.has(opt);
          return `
            <a class="rk-facet-pill ${active ? "active" : ""}"
               href="${esc(href)}"
               data-toggle-facet="${esc(facetKey)}"
               data-value="${esc(opt)}"
               role="button"
               aria-pressed="${active ? "true" : "false"}">
              ${facetKey === "vendor"
                ? `<span class="vendor-dot" data-vendor="${esc(opt)}"></span>`
                : ""}
              <span>${esc(opt)}</span>
            </a>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

// ── Table ──

function renderTable(suiteId, rows, cols, sortKey, sortDir) {
  const meta = SUITE_META[suiteId];
  return `
    <div class="data-table-wrap">
      <table class="data-table" data-suite="${esc(meta.letter)}">
        <thead>
          <tr>
            <th class="col-compare" scope="col"><span class="visually-hidden">Compare</span></th>
            <th class="col-rank" scope="col">#</th>
            <th class="col-chip" scope="col">Chip</th>
            <th class="col-vendor" scope="col">Vendor</th>
            <th class="col-fw" scope="col">Framework</th>
            <th class="col-precision" scope="col">Precision</th>
            ${cols.map((c) => `
              <th class="col-metric sortable${c.primary ? " col-primary" : ""}${sortKey === c.key ? " is-sort" : ""}"
                  data-sort-key="${esc(c.key)}"
                  data-sort-dir-default="${esc(c.direction)}"
                  aria-sort="${sortKey === c.key ? (sortDir === "asc" ? "ascending" : "descending") : "none"}"
                  scope="col">
                <span class="th-label">${esc(c.label)}</span>
                ${c.unit ? `<span class="th-unit">${esc(c.unit)}</span>` : ""}
                <span class="th-sort-icon">${sortKey === c.key ? (sortDir === "asc" ? "↑" : "↓") : ""}</span>
              </th>
            `).join("")}
            <th class="col-date sortable${sortKey === "date" ? " is-sort" : ""}"
                data-sort-key="date" data-sort-dir-default="desc"
                aria-sort="${sortKey === "date" ? (sortDir === "asc" ? "ascending" : "descending") : "none"}"
                scope="col">
              <span class="th-label">Date</span>
              <span class="th-sort-icon">${sortKey === "date" ? (sortDir === "asc" ? "↑" : "↓") : ""}</span>
            </th>
            <th class="col-tier" scope="col">Tier</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((r, i) => renderRow(suiteId, r, cols, sortKey, i + 1)).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderRow(suiteId, row, cols, sortKey, rank) {
  const slug = row._chip_slug;
  const runId = row.run_id || row.submission || "";
  const inBasket = basketHas(runId);
  const ver = shortVersion(row.framework_version);
  const fw = row.framework || "";
  // a11y: tabindex on the row makes the run-detail trigger reachable
  // via keyboard.  We deliberately keep the native `<tr>` role so
  // assistive tech still announces row context (column → cell mapping
  // would break under role="button").  modal.js's keydown handler
  // turns Enter / Space on a focused row into the same openModal call
  // a click would make.
  const a11yLabel = `Open run details for ${row._chip_label}${fw ? " on " + fw : ""}`;
  return `
    <tr data-run-id="${esc(runId)}"
        data-chip-slug="${esc(slug)}"
        data-open-run="${esc(runId)}"
        tabindex="0"
        aria-label="${esc(a11yLabel)}"
        class="${inBasket ? "in-basket" : ""}">
      <td class="col-compare" title="Add to compare basket">
        <label class="compare-cell" aria-label="Add ${esc(row._chip_label)} run to compare basket">
          <input type="checkbox" class="compare-checkbox" tabindex="-1" ${inBasket ? "checked" : ""}>
        </label>
      </td>
      <td class="col-rank tnum">${rank}</td>
      <td class="col-chip">
        <a class="rk-chip-link" href="${chipHref(row)}">
          <span class="rk-chip-name">${esc(row._chip_label)}</span>
          ${row.memory_gb
            ? `<span class="rk-chip-meta">${esc(fmtNum(row.memory_gb))} GB</span>`
            : ""}
        </a>
      </td>
      <td class="col-vendor">
        <span class="vendor-dot" data-vendor="${esc(row.vendor)}"></span>
        <span class="vendor-name">${esc(row.vendor || "-")}</span>
      </td>
      <td class="col-fw">
        ${esc(fw)}${ver ? ` <span class="fw-ver tnum">${esc(ver)}</span>` : ""}
      </td>
      <td class="col-precision">${esc(row.precision || "-")}</td>
      ${cols.map((c) => {
        const v = formatMetric(row[c.key], c);
        const isSort = sortKey === c.key;
        return `
          <td class="col-metric${c.primary ? " col-primary" : ""}${isSort ? " is-sort" : ""} tnum">
            ${v !== null
              ? `<span class="metric-val">${esc(v)}</span>`
              : `<span class="metric-empty">-</span>`}
          </td>
        `;
      }).join("")}
      <td class="col-date tnum${sortKey === "date" ? " is-sort" : ""}">${esc(fmtDate(row.date))}</td>
      <td class="col-tier">
        <span class="badge tier-${esc(row.tier || "community")}">${esc(row.tier || "community")}</span>
      </td>
    </tr>
  `;
}

function renderEmpty(meta, filtersActive) {
  if (!filtersActive) {
    return `
      <div class="rk-empty">
        <span class="rk-empty-icon" aria-hidden="true">∅</span>
        <p>No submissions yet for <strong>Suite ${esc(meta.letter)} · ${esc(meta.title)}</strong>.</p>
        <p class="rk-empty-sub">${esc(meta.tagline)}</p>
      </div>
    `;
  }
  // Resolve the active suite from the URL so the "Clear filters" href
  // strips facets while keeping the user on this same suite.
  const { params: q } = parseHash(location.hash);
  const sid = SUITE_ORDER.includes(q.suite) ? q.suite : SUITE_ORDER[0];
  return `
    <div class="rk-empty">
      <span class="rk-empty-icon" aria-hidden="true">∅</span>
      <p>No submissions match the current filters in Suite ${esc(meta.letter)}.</p>
      <a class="btn" data-clear-all="1" href="${esc(buildHash("/rankings", suiteUrlParam(sid)))}">Clear filters</a>
    </div>
  `;
}

// ── Click delegation ──
//
// The listener is attached exactly once per mounted view container.
// Stale closures (a re-render that re-attached the handler would
// otherwise capture stale suite/query and double-fire updates) are
// avoided by re-reading the URL hash on every click.
//
// The first thing the listener does is bail when the active route is
// no longer /rankings.  Without that guard, navigating Rankings →
// Compare leaves this listener subscribed to the same view container
// and intercepts compare-page clicks (e.g. the suite pills, which
// share a class name) and redirects them back to /rankings.

function bindClicks(el) {
  if (el.__rkClicksAttached) return;
  el.__rkClicksAttached = true;

  el.addEventListener("click", (ev) => {
    if (!location.hash.startsWith("#/rankings")) return;
    const { params: query } = parseHash(location.hash);
    const suiteId = SUITE_ORDER.includes(query.suite) ? query.suite : SUITE_ORDER[0];
    const t = ev.target;

    // Compare basket clear.
    if (t.closest("[data-basket-clear]")) {
      ev.preventDefault();
      const ids = basketGet();
      for (const id of ids) basketToggle(id);
      return;
    }

    // Anchor-driven actions (clear filters, facet pills) are real <a>
    // tags so middle-click / Cmd-click / Ctrl-click / Shift-click can
    // open the toggled URL in a new tab.  The plain-click case still
    // preventDefaults and routes through the SPA in-place.
    const isModifiedNav = ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button !== 0;

    // Clear all filters anchor.
    if (t.closest("[data-clear-all]")) {
      if (isModifiedNav) return;
      ev.preventDefault();
      location.hash = buildHash("/rankings", suiteUrlParam(suiteId));
      return;
    }

    // Suite pill (still a button — switching suite resets sort/filters
    // so opening "the same suite with different filters" in a new tab
    // doesn't make sense).
    const suitePill = t.closest(".rk-suite-pill");
    if (suitePill) {
      ev.preventDefault();
      const sid = suitePill.dataset.suite;
      if (sid && sid !== suiteId) {
        location.hash = buildHash("/rankings", suiteUrlParam(sid));
      }
      return;
    }

    // Facet pill anchor.  Toggle the facet value in the CSV and update
    // the URL — but only on plain left-click; modified clicks fall
    // through to the browser so they open the precomputed href in a
    // new tab.
    const facetPill = t.closest("[data-toggle-facet]");
    if (facetPill) {
      if (isModifiedNav) return;
      ev.preventDefault();
      const facet = facetPill.dataset.toggleFacet;
      const value = facetPill.dataset.value;
      const next = toggleInCsv(query[facet], value);
      location.hash = rebuildHash(suiteId, query, { [facet]: next || undefined });
      return;
    }

    // Compare cell.  Clicking anywhere inside .col-compare toggles the
    // basket — not just the tiny checkbox itself — to forgive misclicks.
    const compareCell = t.closest(".col-compare");
    if (compareCell) {
      ev.preventDefault();
      ev.stopPropagation();
      const tr = compareCell.closest("tr");
      const rid = tr && tr.dataset.runId;
      if (rid) basketToggle(rid);
      return;
    }

    // Sort header.
    const th = t.closest("th.sortable");
    if (th) {
      ev.preventDefault();
      const key = th.dataset.sortKey;
      const defaultDir = th.dataset.sortDirDefault;
      const cur = String(query.sort || "");
      let nextDir = defaultDir;
      if (cur.startsWith(key + ":")) {
        const curDir = cur.split(":")[1];
        nextDir = curDir === "desc" ? "asc" : "desc";
      }
      location.hash = rebuildHash(suiteId, query, { sort: `${key}:${nextDir}` });
      return;
    }

    // Plain row body clicks fall through to the document-level
    // data-open-run handler in modal.js, which pops up the run detail.
  });
}

// ── Helpers ──

function parseCsv(s) {
  if (!s) return [];
  return String(s).split(",").map((x) => x.trim()).filter(Boolean);
}

function toggleInCsv(csv, value) {
  const arr = parseCsv(csv);
  const idx = arr.indexOf(value);
  if (idx >= 0) arr.splice(idx, 1);
  else arr.push(value);
  return arr.join(",");
}

function matchSet(filter, value) {
  if (filter.length === 0) return true;
  if (value === null || value === undefined) return false;
  return filter.includes(value);
}

function safeCompare(a, b, dir) {
  const aMissing = a === null || a === undefined || (typeof a === "number" && Number.isNaN(a));
  const bMissing = b === null || b === undefined || (typeof b === "number" && Number.isNaN(b));
  // Missing values always sink to the bottom regardless of direction.
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;
  if (typeof a === "string" || typeof b === "string") {
    const c = String(a).localeCompare(String(b));
    return dir === "asc" ? c : -c;
  }
  return dir === "asc" ? a - b : b - a;
}

function buildFacets(rows) {
  const vendor = new Set();
  const precision = new Set();
  const framework = new Set();
  for (const r of rows) {
    if (r.vendor)    vendor.add(r.vendor);
    if (r.precision) precision.add(r.precision);
    if (r.framework) framework.add(r.framework);
  }
  // Vendors follow the canonical order; precision/framework alphabetical.
  const sortedVendor = VENDOR_ORDER.filter((v) => vendor.has(v))
    .concat(Array.from(vendor).filter((v) => !VENDOR_ORDER.includes(v)).sort());
  return {
    vendor: sortedVendor,
    precision: Array.from(precision).sort(),
    framework: Array.from(framework).sort(),
  };
}

function colLabel(cols, key) {
  if (key === "date") return "Date";
  const c = cols.find((x) => x.key === key);
  return c ? c.label : key;
}

function suiteUrlParam(suiteId) {
  // suite_A is the default so we omit it from the URL for tidiness.
  return suiteId === SUITE_ORDER[0] ? {} : { suite: suiteId };
}

function rebuildHash(suiteId, currentQuery, updates) {
  const merged = { ...currentQuery, ...updates, ...suiteUrlParam(suiteId) };
  const out = {};
  for (const [k, v] of Object.entries(merged)) {
    if (v === "" || v === null || v === undefined) continue;
    out[k] = v;
  }
  return buildHash("/rankings", out);
}
