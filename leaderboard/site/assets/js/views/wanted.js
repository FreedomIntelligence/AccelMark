// wanted.js — Hardware gaps from accelerator catalog (filterable table).
import { esc, fmtNum, copyToClipboard, flashButtonLabel } from "../utils.js";
import { chipCloudData } from "../data.js";
import { wantedFromCatalog, tierLabel, VENDORS } from "../hardware-catalog.js";
import { communityHeader, communityFilterRow, wireTableFilters } from "../community-nav.js";
const _i = (k, r) => (window._i ? window._i(k, r) : k);

const REPO_URL = "https://github.com/FreedomIntelligence/AccelMark";

const VENDOR_DEFAULT_RUNNER = {
  NVIDIA: "nvidia_vllm_47f5d58e",
  AMD: "amd_vllm_rocm_6c18cd8f",
  Huawei: "ascend_vllm_ascend_d4aa9fda",
  Apple: "apple_mlx_lm_9546b8b5",
  Google: "google_vllm_tpu_68cc9ffa",
  "Moore Threads": "moorethreads_vllm_musa_6607740b",
  Iluvatar: "iluvatar_vllm_ixuca_4d3befc9",
};

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

function buildRunCmd(w) {
  const runner = VENDOR_DEFAULT_RUNNER[w.vendor] || "<your_runner_folder>";
  const suite = (w.suites && w.suites[0]) || "suite_A";
  return `python run.py --runner ${runner} --suite ${suite}`;
}

function buildClaimUrl(w) {
  const suites = (w.suites || ["suite_A", "suite_F"]).join(", ");
  const title = `[Wanted] ${w.name} (${w.vendorLabel}) benchmark submission`;
  const body = [
    `I plan to run AccelMark on **${w.name}** (${w.vendorLabel}).`,
    "",
    `- **Recommended suites**: ${suites}`,
    `- **Command**: \`${buildRunCmd(w)}\``,
    "- **Expected timeline**: ",
  ].join("\n");
  return `${REPO_URL}/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
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
              ${open.map((w) => {
                const cmd = buildRunCmd(w);
                const claimUrl = buildClaimUrl(w);
                return `
                <tr data-vendor="${esc(w.vendor)}" data-tier="${esc(w.tier || "")}" data-search="${esc(searchKey(w))}">
                  <td><strong>${esc(w.name)}</strong></td>
                  <td>${esc(w.vendorLabel)}</td>
                  <td class="muted">${esc(tierLabel(w.tier))}</td>
                  <td class="tnum">${w.memoryGb ? `${w.memoryGb} GB` : "—"}</td>
                  <td>${(w.suites || ["suite_A", "suite_F"]).map((s) => `<code>${esc(s)}</code>`).join(" ")}</td>
                  <td class="repro-actions tnum">
                    <button type="button" class="btn small" data-copy-cmd="${esc(cmd)}" title="${esc(cmd)}"><span class="copy-btn-label">${_i('wanted.copyCmd')}</span></button>
                    <a class="text-link" href="${esc(claimUrl)}" target="_blank" rel="noopener">${_i('wanted.claim')}</a>
                    <a class="text-link" href="#/submit?vendor=${encodeURIComponent(w.vendor)}&chip=${encodeURIComponent(w.name)}">${_i('wanted.submit')}</a>
                  </td>
                </tr>
              `;
              }).join("")}
            </tbody>
          </table>
        ` : `<p class="state">${_i('wanted.allCovered')}</p>`}
      </div>
    </section>
  `;

  el.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button[data-copy-cmd]");
    if (!btn) return;
    const ok = await copyToClipboard(btn.getAttribute("data-copy-cmd") || "");
    flashButtonLabel(btn, ok ? _i('wanted.copied') : "Copy failed");
  });

  wireTableFilters(el, {
    vendorSel: "#wanted-vendor",
    tierSel: "#wanted-tier",
    searchSel: "#wanted-search",
    countSel: "#wanted-count",
  });
}
