// wanted.js — Hardware gaps from accelerator catalog (filterable table).

import { esc, fmtNum } from "../utils.js";
import { chipCloudData } from "../data.js";
import { wantedFromCatalog, tierLabel, VENDORS } from "../hardware-catalog.js";
import { communityHeader, communityFilterRow, wireTableFilters } from "../community-nav.js";

const TIERS = [
  { id: "", label: "All tiers" },
  { id: "datacenter", label: "Datacenter" },
  { id: "cloud", label: "Cloud" },
  { id: "workstation", label: "Workstation" },
  { id: "consumer", label: "Consumer / edge" },
  { id: "edge", label: "Edge" },
];

function searchKey(w) {
  return `${w.name} ${w.vendorLabel} ${w.vendor} ${w.tier || ""} ${w.memoryGb || ""}`.toLowerCase();
}

export function render({ query, el }) {
  const liveLabels = chipCloudData().map((c) => c.label);
  const open = wantedFromCatalog(liveLabels);
  const covered = liveLabels.length;
  const preVendor = query?.vendor || "";

  const vendorOptions = [
    `<option value="">All vendors</option>`,
    ...VENDORS.filter((v) => v.id !== "Other").map((v) => {
      const sel = v.id === preVendor ? " selected" : "";
      return `<option value="${esc(v.id)}"${sel}>${esc(v.label)}</option>`;
    }),
  ].join("");

  const tierOptions = TIERS.map((t) =>
    `<option value="${esc(t.id)}">${esc(t.label)}</option>`
  ).join("");

  el.innerHTML = `
    ${communityHeader(
      "wanted",
      "Wanted hardware",
      `Accelerators present in our catalog but not yet covered on the leaderboard (${fmtNum(open.length)} open, ${fmtNum(covered)} covered). ` +
      `The first published result for a platform receives a <em>First result</em> attribution on the ` +
      `<a href="#/contributors">contributor index</a>.`
    )}

    <section class="section community-section">
      ${communityFilterRow([
        {
          label: "Vendor",
          html: `<select id="wanted-vendor">${vendorOptions}</select>`,
          countId: "wanted-count",
        },
        {
          label: "Tier",
          html: `<select id="wanted-tier">${tierOptions}</select>`,
        },
        {
          label: "Search",
          html: `<input type="search" id="wanted-search" placeholder="Platform name…" autocomplete="off">`,
        },
      ])}

      <div class="contrib-table-wrap card">
        ${open.length ? `
          <table class="contrib-table community-data-table">
            <thead>
              <tr>
                <th>Platform</th>
                <th>Vendor</th>
                <th>Tier</th>
                <th>Memory</th>
                <th>Recommended suites</th>
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
                    <a class="text-link" href="#/submit?vendor=${encodeURIComponent(w.vendor)}&chip=${encodeURIComponent(w.name)}">Submit</a>
                  </td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        ` : `<p class="state">All catalog platforms are covered. Propose additional hardware via GitHub Discussions.</p>`}
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
