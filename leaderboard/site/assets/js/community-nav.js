// community-nav.js — shared navigation for community sub-pages.

import { esc } from "./utils.js";
const _i = (k, r) => (window._i ? window._i(k, r) : k);

function getTabs() { return [
  { id: "contributors", href: "#/contributors", label: _i('nav.contributors') },
  { id: "wanted",       href: "#/wanted",       label: _i('nav.wanted') },
  { id: "reproduce",    href: "#/reproduce",    label: _i('nav.reproduce') },
]; }

/** Underline tab strip — one row, no subtitles. */
export function communityTabs(activeId) {
  return `
    <nav class="community-tabs" aria-label="Community">
      ${getTabs().map((t) => `
        <a class="community-tab${t.id === activeId ? " active" : ""}" href="${esc(t.href)}">${esc(t.label)}</a>
      `).join("")}
    </nav>
  `;
}

/** Compact academic page header (tabs + title + lead). */
export function communityHeader(activeId, title, leadHtml) {
  return `
    <header class="community-header">
      ${communityTabs(activeId)}
      <h1>${title}</h1>
      ${leadHtml ? `<p class="community-lead">${leadHtml}</p>` : ""}
    </header>
  `;
}

/** Standard filter row used on Wanted / Reproduce tables. */
export function communityFilterRow(fields) {
  return `
    <div class="community-filters" role="search">
      ${fields.map((f) => `
        <label class="community-filter">
          <span>${esc(f.label)}</span>
          ${f.html}
        </label>
      `).join("")}
      <span class="community-filter-count muted" id="${esc(fields[0]?.countId || "filter-count")}"></span>
    </div>
  `;
}

/**
 * Client-side table filter.
 * Rows need data-vendor, data-tier (or data-suite), and data-search attributes.
 * If suiteSel is set, it filters on data-suite instead of data-tier.
 */
export function wireTableFilters(root, opts) {
  const tbody = root.querySelector(opts.tbodySel || "tbody[data-filterable]");
  if (!tbody) return;

  const vendorEl = opts.vendorSel ? root.querySelector(opts.vendorSel) : null;
  const tierEl = opts.tierSel ? root.querySelector(opts.tierSel) : null;
  const suiteEl = opts.suiteSel ? root.querySelector(opts.suiteSel) : null;
  const searchEl = opts.searchSel ? root.querySelector(opts.searchSel) : null;
  const countEl = opts.countSel ? root.querySelector(opts.countSel) : null;
  const secondaryEl = suiteEl || tierEl;
  const secondaryAttr = suiteEl ? "suite" : "tier";
  const rows = [...tbody.querySelectorAll("tr[data-vendor]")];

  function apply() {
    const vendor = vendorEl?.value || "";
    const secondary = secondaryEl?.value || "";
    const q = (searchEl?.value || "").trim().toLowerCase();
    let visible = 0;
    for (const row of rows) {
      const okVendor = !vendor || row.dataset.vendor === vendor;
      const okSecondary = !secondary || row.dataset[secondaryAttr] === secondary;
      const hay = row.dataset.search || "";
      const okSearch = !q || hay.includes(q);
      const show = okVendor && okSecondary && okSearch;
      row.hidden = !show;
      if (show) visible += 1;
    }
    if (countEl) {
      countEl.textContent = visible === rows.length
        ? `${visible} ${_i('community.entries')}`
        : `${visible} ${_i('community.of')} ${rows.length} ${_i('community.entries')}`;
    }
  }

  vendorEl?.addEventListener("change", apply);
  secondaryEl?.addEventListener("change", apply);
  searchEl?.addEventListener("input", apply);
  apply();
}
