// wanted.js — Hardware gaps from accelerator catalog (filterable table).
import { esc, fmtNum } from "../utils.js";
import { chipCloudData } from "../data.js";
import { wantedFromCatalog, tierLabel, VENDORS } from "../hardware-catalog.js";
import { communityHeader, communityFilterRow, wireTableFilters } from "../community-nav.js";
const _i = (k, r) => (window._i ? window._i(k, r) : k);

function tierLabels() { return [
  { id: "", label: _i('wanted.allTiers') },
  { id: "datacenter", label: _i('wanted.datacenter') },
  { id: "cloud", label: _i('wanted.cloud') },
  { id: "workstation", label: _i('wanted.workstation') },
  { id: "consumer", label: _i('wanted.consumer') },
  { id: "edge", label: _i('wanted.edge') },
]; }

function searchKey(w) {
  return `${w.name} ${w.vendorLabel} ${w.vendor} ${w.tier || ""} ${w.memoryGb || ""}`.toLowerCase();
}

export function render({ query, el }) {
  const liveLabels = chipCloudData().map((c) => c.label);
  const open = wantedFromCatalog(liveLabels);
  const covered = liveLabels.length;
  const preVendor = query?.vendor || "";

  const vendorOptions = [
    `<option value="">${_i('wanted.allVendors')}</option>`,
    ...VENDORS.filter((v) => v.id !== "Other").map((v) => {
      const sel = v.id === preVendor ? " selected" : "";
      return `<option value="${esc(v.id)}"${sel}>${esc(v.label)}</option>`;
    }),
  ].join("");

  const TIERS = tierLabels();
  const tierOptions = TIERS.map((t) =>
    `<option value="${esc(t.id)}">${esc(t.label)}</option>`
  ).join("");

  el.innerHTML = `
    ${communityHeader(
      "wanted",
      _i('wanted.title'),
      `${_i('wanted.desc1')} (${fmtNum(open.length)} ${_i('wanted.open')}, ${fmtNum(covered)} ${_i('wanted.covered')}). ` +
      `${_i('wanted.desc2')} ` +
      `<a href="#/contributors">${_i('wanted.contribIndex')}</a>.`
    )}

    <section class="section community-section">
      ${communityFilterRow([
        {
          label: _i('wanted.filterVendor'),
          html: `<select id="wanted-vendor">${vendorOptions}</select>`,
          countId: "wanted-count",
        },
        {
          label: _i('wanted.filterTier'),
          html: `<select id="wanted-tier">${tierOptions}</select>`,
        },
        {
          label: _i('wanted.filterSearch'),
          html: `<input type="search" id="wanted-search" placeholder="${_i('wanted.searchPlaceholder')}" autocomplete="off">`,
        },
      ])}

      <div class="contrib-table-wrap card">
        ${open.length ? `
          <table class="contrib-table community-data-table">
            <thead>
              <tr>
                <th>${_i('wanted.thPlatform')}</th>
                <th>${_i('wanted.thVendor')}</th>
                <th>${_i('wanted.thTier')}</th>
                <th>${_i('wanted.thMemory')}</th>
                <th>${_i('wanted.thSuites')}</th>
                <th></th>
              </tr>
            </thead>
            <tbody data-filterable>
              ${open.map((w) => `
                <tr data-vendor="${esc(w.vendor)}" data-tier="${esc(w.tier || "")}" data-search="${esc(searchKey(w))}">
                  <td><strong>${esc(w.name)}</strong></td>
                  <td>${esc(w.vendorLabel)}</td>
                  <td class="muted">${esc(tierLabel(w.tier))}</td>
                  <td class="tnum">${w.memoryGb ? `${w.memoryGb} GB` : "—"}</td>
                  <td>${(w.suites || ["suite_A", "suite_F"]).map((s) => `<code>${esc(s)}</code>`).join(" ")}</td>
                  <td class="tnum">
                    <a class="text-link" href="#/submit?vendor=${encodeURIComponent(w.vendor)}&chip=${encodeURIComponent(w.name)}">${_i('wanted.submit')}</a>
                  </td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        ` : `<p class="state">${_i('wanted.allCovered')}</p>`}
      </div>
    </section>
  `;

  wireTableFilters(el, {
    vendorSel: "#wanted-vendor",
    tierSel: "#wanted-tier",
    searchSel: "#wanted-search",
    countSel: "#wanted-count",
  });
}
