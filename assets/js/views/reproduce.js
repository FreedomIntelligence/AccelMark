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

  const questWord = ex.openQuestCount === 1 ? "quest" : "quests";

  const pairWord = ex.openPairCount === 1 ? "pair" : "pairs";

  const recordWord = ex.crossVerifiedRecordCount === 1 ? "record" : "records";

  return `

    <aside class="community-note card repro-criteria">

      <h2>Why these quests?</h2>

      <p class="repro-criteria-lead">

        The open list is <strong>automatic</strong> — nothing is hand-picked.

        A row appears when <em>both</em> conditions hold:

      </p>

      <ol class="repro-steps repro-criteria-rules">

        <li>The submission is <span class="tier-pill community">community</span> tier (published, not yet independently verified).</li>

        <li>No <span class="tier-pill verified">verified</span> result exists for the <strong>same hardware platform + benchmark suite</strong> — same chip model and workload, regardless of framework or chip count.</li>

      </ol>

      <p class="repro-criteria-example muted">

        <strong>Cross-verified coverage</strong> (below) is separate: it only lists runs whose

        <code>meta.reproduces_run_id</code> explicitly cites the original community

        <code>run_id</code>. That link is set at submit time (runner flag

        <code>--reproduces-run-id</code>) — never inferred from chip or suite alone.

      </p>

      <p class="repro-criteria-stats">

        <span class="repro-stat"><strong>${fmtNum(ex.openQuestCount)}</strong> open ${questWord}</span>

        <span class="repro-stat-sep">·</span>

        <span class="repro-stat"><strong>${fmtNum(ex.crossVerifiedRecordCount)}</strong> linked verification ${recordWord}</span>

        <span class="repro-stat-sep">·</span>

        <span class="repro-stat muted">${fmtNum(ex.openPairCount)} platform–suite ${pairWord} awaiting any verified run · ${fmtNum(ex.verifiedPairs)} pairs with verified coverage</span>

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

            <span class="eyebrow">Verified</span>

            <h2>Cross-verified coverage</h2>

          </div>

        </div>

        <div class="community-note card repro-verified-empty-card">

          <p class="repro-verified-empty">

            No linked verifications in this snapshot yet. When someone independently reruns a

            community result, they set the original <code>run_id</code> on submit:

          </p>

          <pre class="repro-cmd-hint"><code>python runners/.../runner.py --suite suite_A --reproduces-run-id &lt;original_run_id&gt; --tier verified</code></pre>

          <p class="repro-verified-empty muted">

            The verifying run's <code>result.json</code> stores

            <code>meta.reproduces_run_id</code>, and this table fills automatically after the

            next <code>leaderboard/generate.py</code> refresh — no manual page edits.

          </p>

        </div>

      </section>

    `;

  }



  return `

    <section class="section community-section repro-verified-section">

      <div class="section-header">

        <div class="section-title">

          <span class="eyebrow">Verified</span>

          <h2>Cross-verified coverage</h2>

        </div>

        <span class="section-sub">${fmtNum(records.length)} explicit verification ${records.length === 1 ? "link" : "links"} via <code>meta.reproduces_run_id</code>.</span>

      </div>

      <div class="contrib-table-wrap card">

        <table class="contrib-table community-data-table repro-verified-table">

          <thead>

            <tr>

              <th>Original (community)</th>

              <th>Verified rerun</th>

              <th>Suite</th>

              <th>Confirmed</th>

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

                    ${orig.run_id ? `<button type="button" class="btn small" data-open-run="${esc(orig.run_id)}">Original</button>` : ""}

                    ${ver.run_id ? `<button type="button" class="btn small" data-open-run="${esc(ver.run_id)}">Verified</button>` : ""}

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

    `<option value="">All vendors</option>`,

    ...vendors.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`),

  ].join("");



  const suiteOptions = [

    `<option value="">All suites</option>`,

    ...suites.map((s) => `<option value="${esc(s)}">${esc(suiteLabel(s))}</option>`),

  ].join("");



  el.innerHTML = `

    ${communityHeader(

      "reproduce",

      "Reproduction quests",

      `Independent reruns that confirm a community submission within 5% earn <strong>verifier credit</strong> for both parties. ` +

      `Reference the original <code>run_id</code> in your pull request and in <code>meta.reproduces_run_id</code>.`

    )}



    ${renderCriteria(ex)}



    <section class="section community-section">

      <div class="section-header">

        <div class="section-title">

          <span class="eyebrow">Open</span>

          <h2>Awaiting verification</h2>

        </div>

        ${quests.length ? `<span class="section-sub">${fmtNum(quests.length)} community ${quests.length === 1 ? "run" : "runs"} on platforms without verified coverage yet.</span>` : ""}

      </div>



      ${quests.length ? communityFilterRow([

        {

          label: "Vendor",

          html: `<select id="repro-vendor">${vendorOptions}</select>`,

          countId: "repro-count",

        },

        {

          label: "Suite",

          html: `<select id="repro-suite">${suiteOptions}</select>`,

        },

        {

          label: "Search",

          html: `<input type="search" id="repro-search" placeholder="Platform, recipe, submitter…" autocomplete="off">`,

        },

      ]) : ""}



      <div class="contrib-table-wrap card">

        ${quests.length ? `

          <table class="contrib-table community-data-table">

            <thead>

              <tr>

                <th>Platform</th>

                <th>Recipe</th>

                <th>Suite</th>

                <th>Submitter</th>

                <th>Date</th>

                <th>Tier</th>

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

                      ${q.run_id ? `<button type="button" class="btn small" data-open-run="${esc(q.run_id)}">Details</button>` : ""}

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

        <h2>Procedure</h2>

        <ol class="repro-steps">

          <li>Select a quest above and note its <code>run_id</code> from the run details modal.</li>

          <li>Rerun the same suite on matching hardware with <code>--reproduces-run-id &lt;run_id&gt;</code>.</li>

          <li>Open a pull request; after merge and data regen, the pair appears in Cross-verified coverage below.</li>

          <li>Both submitters receive credit on the <a href="#/contributors">contributor index</a>.</li>

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


