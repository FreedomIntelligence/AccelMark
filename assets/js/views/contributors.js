// contributors.js (view) — public contributor leaderboard.

import { esc, fmtNum } from "../utils.js";
import { contributorIndex, BADGE_DEFS } from "../contributors.js";
import { summary } from "../data.js";
import { communityHeader } from "../community-nav.js";

function badgeHtml(ids) {
  if (!ids?.length) return "";
  return ids.map((id) => {
    const b = BADGE_DEFS[id];
    if (!b) return "";
    return `<span class="contrib-badge" title="${esc(b.desc)}">${esc(b.label)}</span>`;
  }).join("");
}

export function render({ el }) {
  const list = contributorIndex();
  const s = summary();

  el.innerHTML = `
    ${communityHeader(
      "contributors",
      "Contributor index",
      `${fmtNum(list.length)} contributors · ${fmtNum(s.total)} published runs · ${fmtNum(s.verified)} verified. ` +
      `Profiles are derived from <code>submitted_by</code> in merged results (GitHub handle); no separate registration.`
    )}

    <section class="section community-section">
      <div class="section-header">
        <div class="section-title"><h2>Ranking</h2></div>
        <span class="section-sub">Score = runs + verified×3 + platforms×4 + first-result×12 + runner×5</span>
      </div>
      ${list.length ? `
        <div class="contrib-table-wrap card">
          <table class="contrib-table community-data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Contributor</th>
                <th>Runs</th>
                <th>Verified</th>
                <th>Platforms</th>
                <th>Attributions</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              ${list.map((c) => `
                <tr>
                  <td class="tnum rank-cell">${esc(String(c.rank))}</td>
                  <td>
                    <a class="contrib-handle" href="#/contributor/${esc(c.handle)}">@${esc(c.handle)}</a>
                    <a class="contrib-gh" href="${esc(c.githubUrl)}" target="_blank" rel="noopener" aria-label="GitHub">↗</a>
                  </td>
                  <td class="tnum">${fmtNum(c.total)}</td>
                  <td class="tnum">${fmtNum(c.verified)}</td>
                  <td class="tnum">${fmtNum(c.uniqueChips.size)}</td>
                  <td class="contrib-badges-cell">${badgeHtml(c.badges)}</td>
                  <td class="tnum score-cell">${fmtNum(c.score)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      ` : `<p class="state">No contributors yet — <a href="#/submit">submit the first result</a>.</p>`}

      <aside class="community-note card">
        <h2>Attribution definitions</h2>
        <dl class="badge-dl">
          ${Object.values(BADGE_DEFS).map((b) => `
            <dt>${esc(b.label)}</dt>
            <dd>${esc(b.desc)}</dd>
          `).join("")}
        </dl>
      </aside>
    </section>
  `;
}
