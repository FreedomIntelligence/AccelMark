// contributors.js (view) — public contributor leaderboard.
import { esc, fmtNum } from "../utils.js";
import { contributorIndex, BADGE_DEFS } from "../contributors.js";
import { summary } from "../data.js";
import { communityHeader } from "../community-nav.js";
const _i = (k, r) => (window._i ? window._i(k, r) : k);

function badgeHtml(ids) {
  if (!ids?.length) return "";
  return ids.map((id) => {
    const b = BADGE_DEFS[id];
    if (!b) return "";
    const labelKey = 'badge.' + id + '.label';
    const descKey = 'badge.' + id + '.desc';
    return `<span class="contrib-badge" title="${esc(_i(descKey, b.desc))}">${esc(_i(labelKey, b.label))}</span>`;
  }).join("");
}

export function render({ el }) {
  const list = contributorIndex();
  const s = summary();

  el.innerHTML = `
    ${communityHeader(
      "contributors",
      _i('contributors.title'),
      `${fmtNum(list.length)} ${_i('contributors.count')}`
    )}

    <section class="section community-section">
      <div class="section-header">
        <div class="section-title"><h2>${_i('contributors.ranking')}</h2></div>
        <span class="section-sub">${_i('contributors.scoreDesc')}</span>
      </div>
      ${list.length ? `
        <div class="contrib-table-wrap card">
          <table class="contrib-table community-data-table">
            <thead>
              <tr>
                <th>${_i('contributors.rank')}</th>
                <th>${_i('contributors.contributor')}</th>
                <th>${_i('contributors.runs')}</th>
                <th>${_i('contributors.verified')}</th>
                <th>${_i('contributors.platforms')}</th>
                <th>${_i('contributors.attribs')}</th>
                <th>${_i('contributors.score')}</th>
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
      ` : `<p class="state">${_i('contributors.empty')} — <a href="#/submit">${_i('contributors.submitFirst')}</a>.</p>`}

      <aside class="community-note card">
        <h2>${_i('contributors.attribDefs')}</h2>
        <dl class="badge-dl">
          ${Object.values(BADGE_DEFS).map((b) => {
            const lk = 'badge.' + b.id + '.label';
            const dk = 'badge.' + b.id + '.desc';
            return `<dt>${esc(_i(lk, b.label))}</dt><dd>${esc(_i(dk, b.desc))}</dd>`;
          }).join("")}
        </dl>
      </aside>
    </section>
  `;
}
