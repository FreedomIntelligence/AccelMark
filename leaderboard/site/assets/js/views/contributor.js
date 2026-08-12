// contributor.js (view) — single contributor profile by GitHub handle.

import { esc, fmtNum, fmtDate } from "../utils.js";
import { contributorByHandle, BADGE_DEFS, contributorIndex } from "../contributors.js";
import { SUITE_META } from "../data.js";
const _i = (k, r) => (window._i ? window._i(k, r) : k);

function badgeHtml(ids) {
  return (ids || []).map((id) => {
    const b = BADGE_DEFS[id];
    if (!b) return "";
    return `<span class="contrib-badge" title="${esc(b.desc)}">${esc(b.label)}</span>`;
  }).join("");
}

function suiteShort(id) {
  const m = SUITE_META[id];
  return m ? `Suite ${m.letter}` : id;
}

export function render({ params, el }) {
  const handle = params.handle || "";
  const c = contributorByHandle(handle);

  if (!c) {
    el.innerHTML = `
      <section class="state">
        No published results for <code>@${esc(handle)}</code>.<br>
        <span class="muted">${_i('contributor.githubHint')} <code>submitted_by</code> in the PR.</span><br>
        <a href="#/contributors" class="btn" style="margin-top:1rem">← All contributors</a>
        <a href="#/submit" class="btn primary" style="margin-top:1rem">${_i('contributor.submitResult')}</a>
      </section>
    `;
    return;
  }

  const rank = c.rank;
  const totalContributors = contributorIndex().length;

  el.innerHTML = `
    <section class="page-hero contrib-profile-hero">
      <a class="back-link" href="#/contributors">← Contributors</a>
      <div class="contrib-profile-head">
        <img class="contrib-avatar" src="https://github.com/${esc(c.handle)}.png?size=96" width="72" height="72" alt="" loading="lazy">
        <div>
          <h1>@${esc(c.handle)}</h1>
          <p class="page-lead">
            Rank <strong>#${rank}</strong> of ${fmtNum(totalContributors)} ·
            <a href="${esc(c.githubUrl)}" target="_blank" rel="noopener">GitHub profile ↗</a>
          </p>
          <div class="contrib-badges-row">${badgeHtml(c.badges)}</div>
        </div>
      </div>
      <div class="hero-stats compact">
        <div class="kpi"><span class="kpi-value">${fmtNum(c.total)}</span><span class="kpi-label">runs</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(c.verified)}</span><span class="kpi-label">verified</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(c.uniqueChips.size)}</span><span class="kpi-label">platforms</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(c.score)}</span><span class="kpi-label">score</span></div>
      </div>
    </section>

    ${c.firstChips.length ? `
      <section class="section">
        <div class="section-header">
          <div class="section-title"><h2>${_i('contributor.firstResults')}</h2></div>
          <span class="section-sub">${_i('contributor.firstBenchmark')}</span>
        </div>
        <div class="first-chips-row">
          ${c.firstChips.map((chip) => `
            <span class="contrib-badge first-chip-badge">${esc(chip)}</span>
          `).join("")}
        </div>
      </section>
    ` : ""}

    <section class="section">
      <div class="section-header">
        <div class="section-title"><h2>${_i('contributor.publishedRuns')}</h2></div>
        <span class="section-sub">Latest activity: ${esc(fmtDate(c.latestDate))}</span>
      </div>
      <div class="contrib-runs card">
        <table class="contrib-table compact">
          <thead>
            <tr><th>Chip</th><th>Suite</th><th>Tier</th><th>Date</th><th></th></tr>
          </thead>
          <tbody>
            ${c.runs.slice().sort((a, b) => String(b.date).localeCompare(String(a.date))).map((r) => `
              <tr>
                <td>${esc(r.chip)}${r.chip_count > 1 ? ` ×${r.chip_count}` : ""}</td>
                <td>${esc(suiteShort(r.suite))}</td>
                <td><span class="tier-pill ${esc(r.tier || "community")}">${esc(r.tier || "community")}</span></td>
                <td class="muted">${esc(fmtDate(r.date))}</td>
                <td>${r.run_id ? `<button type="button" class="btn small" data-open-run="${esc(r.run_id)}">Details</button>` : ""}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title"><h2>Coverage</h2></div>
      </div>
      <div class="contrib-coverage card">
        <p><strong>Vendors:</strong> ${[...c.vendors].map((v) => esc(v)).join(", ") || "—"}</p>
        <p><strong>Platforms:</strong> ${[...c.chips].map((v) => esc(v)).join(" · ") || "—"}</p>
        <p><strong>Suites:</strong> ${[...c.suites].map((s) => esc(suiteShort(s))).join(", ") || "—"}</p>
      </div>
    </section>
  `;
}
