// reproduce.js — Reproduction quests (community → verified), filterable table.

import { esc, fmtNum, fmtDate, shortVersion } from "../utils.js";

import { rowByRunId, SUITE_META } from "../data.js";

import {

  reproductionQuests,

  reproductionQuestExplain,

  crossVerifiedRecords,

} from "../contributors.js";

import { discussResultUrl } from "../cite.js";

import { communityHeader, communityFilterRow, wireTableFilters } from "../community-nav.js";
const _i = (k, r) => (window._i ? window._i(k, r) : k);

function suiteLabel(id) {

  const m = SUITE_META[id];

  return m ? `Suite ${m.letter}` : id;

}

function recipeLabel(row) {

  if (!row?.framework) return "—";

  const ver = shortVersion(row.framework_version);

  const base = `${row.framework}${ver ? ` ${ver}` : ""}`;

  return row.precision ? `${base} · ${row.precision}` : base;

}

function searchKey(q) {

  return `${q.chip} ${q.vendor} ${q.suite} ${q.framework || ""} ${q.submitted_by} ${q.run_id}`.toLowerCase();

}

function renderCriteria(ex) {

  const questWord = ex.openQuestCount === 1 ? _i("reproduce.quest") : _i("reproduce.quests");

  const pairWord = ex.openPairCount === 1 ? _i("reproduce.pair") : _i("reproduce.pairs");

  const recordWord = ex.crossVerifiedRecordCount === 1 ? _i("reproduce.record") : _i("reproduce.records");

  return `

    <aside class="community-note card repro-criteria">

      <h2>${_i("reproduce.whyTitle")}</h2>

      <p class="repro-criteria-lead">

        ${_i('reproduce.criteriaLead')}

      </p>

      <ol class="repro-steps repro-criteria-rules">

        <li>${_i("reproduce.criteriaRule1")}</li>

        <li>${_i("reproduce.criteriaRule2")}</li>

      </ol>

      <p class="repro-criteria-example muted">

        <strong>${_i("reproduce.criteriaNote")}</p>

      <p class="repro-criteria-stats">

        <span class="repro-stat"><strong>${fmtNum(ex.openQuestCount)}</strong> ${_i('reproduce.statOpen', {n: fmtNum(ex.openQuestCount), w: questWord})}</span>

        <span class="repro-stat-sep">·</span>

        <span class="repro-stat"><strong>${fmtNum(ex.crossVerifiedRecordCount)}</strong>${_i("reproduce.statLinked", {n: fmtNum(ex.crossVerifiedRecordCount), w: recordWord})}</span>

        <span class="repro-stat-sep">·</span>

        <span class="repro-stat muted">${_i('reproduce.statPairs', {np: fmtNum(ex.openPairCount), pw: pairWord, nv: fmtNum(ex.verifiedPairs)})}</span>

      </p>

    </aside>

  `;

}

function renderCrossVerified(records) {

  if (!records.length) {

    return `

      <section class="section community-section repro-verified-section">

        <div class="section-header">

          <div class="section-title">

            <span class="eyebrow">${_i("reproduce.verified")}</span>

            <h2>${_i("reproduce.verifiedTitle")}</h2>

          </div>

        </div>

        <div class="community-note card repro-verified-empty-card">

          <p class="repro-verified-empty">

            ${_i('reproduce.emptyVerified')}

          </p>

          <pre class="repro-cmd-hint"><code>python runners/.../runner.py --suite suite_A --reproduces-run-id &lt;original_run_id&gt; --tier verified</code></pre>

          <p class="repro-verified-empty muted">

            ${_i('reproduce.emptyVerifiedNote')}

          </p>

        </div>

      </section>

    `;

  }

  return `

    <section class="section community-section repro-verified-section">

      <div class="section-header">

        <div class="section-title">

          <span class="eyebrow">${_i("reproduce.verified")}</span>

          <h2>${_i("reproduce.verifiedTitle")}</h2>

        </div>

        <span class="section-sub">${fmtNum(records.length)} explicit verification ${records.length === 1 ? "link" : "links"} via <code>meta.reproduces_run_id</code>.</span>

      </div>

      <div class="contrib-table-wrap card">

        <table class="contrib-table community-data-table repro-verified-table">

          <thead>

            <tr>

              <th>${_i("reproduce.thOriginal")}</th>

              <th>${_i("reproduce.thVerified")}</th>

              <th>${_i("reproduce.thSuite")}</th>

              <th>${_i("reproduce.thConfirmed")}</th>

              <th></th>

            </tr>

          </thead>

          <tbody>

            ${records.map((rec) => {

              const orig = rec.original;

              const ver = rec.verification;

              return `

                <tr>

                  <td class="repro-party">

                    <strong>${esc(orig.chip)}</strong>

                    <span class="repro-party-meta">${esc(recipeLabel(orig))}</span>

                    ${orig.submitted_by ? `<a class="text-link" href="#/contributor/${esc(orig.submitted_by)}">@${esc(orig.submitted_by)}</a>` : ""}

                    <span class="repro-party-date muted">${esc(fmtDate(orig.date))} · <code>${esc(rec.reproduces_run_id)}</code></span>

                  </td>

                  <td class="repro-party">

                    <strong>${esc(ver.chip)}</strong>

                    <span class="repro-party-meta">${esc(recipeLabel(ver))}</span>

                    ${ver.submitted_by ? `<a class="text-link" href="#/contributor/${esc(ver.submitted_by)}">@${esc(ver.submitted_by)}</a>` : ""}

                    <span class="tier-pill ${esc(ver.tier || "verified")}">${esc(ver.tier || "verified")}</span>

                  </td>

                  <td>${esc(suiteLabel(orig.suite))}</td>

                  <td class="muted">${esc(fmtDate(ver.date))}</td>

                  <td class="repro-actions">

                    ${orig.run_id ? `<button type="button" class="btn small" data-open-run="${esc(orig.run_id)}">${_i("reproduce.btnOriginal")}</button>` : ""}

                    ${ver.run_id ? `<button type="button" class="btn small" data-open-run="${esc(ver.run_id)}">${_i("reproduce.verified")}</button>` : ""}

                  </td>

                </tr>

              `;

            }).join("")}

          </tbody>

        </table>

      </div>

    </section>

  `;

}

export function render({ el }) {

  const ex = reproductionQuestExplain();

  const quests = reproductionQuests(200);

  const crossVerified = crossVerifiedRecords(100);

  const vendors = [...new Set(quests.map((q) => q.vendor).filter(Boolean))].sort();

  const suites = [...new Set(quests.map((q) => q.suite).filter(Boolean))].sort();

  const vendorOptions = [

    `<option value="">${_i("reproduce.allVendors")}</option>`,

    ...vendors.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`),

  ].join("");

  const suiteOptions = [

    `<option value="">${_i("reproduce.allSuites")}</option>`,

    ...suites.map((s) => `<option value="${esc(s)}">${esc(suiteLabel(s))}</option>`),

  ].join("");

  el.innerHTML = `

    ${communityHeader(

      "reproduce",

      _i('reproduce.title'),
      _i('reproduce.desc')
    )}

    ${renderCriteria(ex)}

    <section class="section community-section">

      <div class="section-header">

        <div class="section-title">

          <span class="eyebrow">${_i("reproduce.open")}</span>

          <h2>${_i("reproduce.awaitingTitle")}</h2>

        </div>

        ${quests.length ? `<span class="section-sub">${fmtNum(quests.length)} ${_i('reproduce.countLine')}</span>` : ""}

      </div>

      ${quests.length ? communityFilterRow([

        {

          label: _i("reproduce.filterVendor"),

          html: `<select id="repro-vendor">${vendorOptions}</select>`,

          countId: "repro-count",

        },

        {

          label: _i("reproduce.filterSuite"),

          html: `<select id="repro-suite">${suiteOptions}</select>`,

        },

        {

          label: _i("reproduce.filterSearch"),

          html: `<input type="search" id="repro-search" placeholder="${_i("reproduce.searchPlaceholder")}" autocomplete="off">`,

        },

      ]) : ""}

      <div class="contrib-table-wrap card">

        ${quests.length ? `

          <table class="contrib-table community-data-table">

            <thead>

              <tr>

                <th>${_i("reproduce.thPlatform")}</th>

                <th>${_i("reproduce.thRecipe")}</th>

                <th>${_i("reproduce.thSuite")}</th>

                <th>${_i("reproduce.thSubmitter")}</th>

                <th>${_i("reproduce.thDate")}</th>

                <th>${_i("reproduce.thTier")}</th>

                <th></th>

              </tr>

            </thead>

            <tbody data-filterable>

              ${quests.map((q) => {

                const row = rowByRunId(q.run_id);

                const reproUrl = row ? discussResultUrl(row, "reproduction") : "#";

                const ver = shortVersion(q.framework_version);

                const recipe = q.framework

                  ? `${esc(q.framework)}${ver ? ` ${esc(ver)}` : ""}${q.precision ? ` · ${esc(q.precision)}` : ""}`

                  : "—";

                return `

                  <tr data-vendor="${esc(q.vendor || "")}" data-suite="${esc(q.suite || "")}" data-search="${esc(searchKey(q))}">

                    <td><strong>${esc(q.chip)}${q.chip_count > 1 ? ` ×${q.chip_count}` : ""}</strong></td>

                    <td class="muted">${recipe}</td>

                    <td>${esc(suiteLabel(q.suite))}</td>

                    <td>${q.submitted_by ? `<a class="text-link" href="#/contributor/${esc(q.submitted_by)}">@${esc(q.submitted_by)}</a>` : "—"}</td>

                    <td class="muted">${esc(fmtDate(q.date))}</td>

                    <td><span class="tier-pill community">${esc(q.tier)}</span></td>

                    <td class="repro-actions">

                      ${q.run_id ? `<button type="button" class="btn small" data-open-run="${esc(q.run_id)}">${_i("reproduce.btnDetails")}</button>` : ""}

                      <a class="text-link" href="${esc(reproUrl)}" target="_blank" rel="noopener">Request</a>

                    </td>

                  </tr>

                `;

              }).join("")}

            </tbody>

          </table>

        ` : `

          <p class="state">No open reproduction quests. Every community platform–suite pair either has a verified sibling or the dataset is still growing.</p>

        `}

      </div>

      <aside class="community-note card">

        <h2>${_i("reproduce.procedure")}</h2>

        <ol class="repro-steps">

          <li>${_i("reproduce.procedure1")}</li>

          <li>${_i("reproduce.procedure2")}

          <li>${_i("reproduce.procedure3")}</li>

          <li>${_i("reproduce.procedure4")}</li>

        </ol>

      </aside>

    </section>

    ${renderCrossVerified(crossVerified)}

  `;

  if (quests.length) {

    wireTableFilters(el, {

      vendorSel: "#repro-vendor",

      suiteSel: "#repro-suite",

      searchSel: "#repro-search",

      countSel: "#repro-count",

    });

  }

}

