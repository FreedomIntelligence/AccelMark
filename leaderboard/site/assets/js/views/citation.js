// citation.js — How to cite AccelMark at project, dataset, and run level.

import { esc } from "../utils.js";
import { PROJECT_BIBTEX, datasetBibTeX } from "../cite.js";
import { copyToClipboard, flashButtonLabel } from "../utils.js";

const DATASET_VERSION = "2026.07.08";
const API_BASE = typeof location !== "undefined" ? `${location.origin}${location.pathname}` : "";

export function render({ el }) {
  const dsBib = datasetBibTeX(DATASET_VERSION, `${API_BASE}api/manifest.json`);
  const plainDataset = `AccelMark Results Dataset v${DATASET_VERSION}. AccelMark Contributors. ${DATASET_VERSION.slice(0, 4)}. ${API_BASE}api/manifest.json (CC BY 4.0).`;

  el.innerHTML = `
    <section class="page-hero">
      <span class="eyebrow">Reference</span>
      <h1>Cite the dataset</h1>
      <p class="page-lead">Three levels of citation — framework, versioned snapshot, and individual reproducible runs. All community results are <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a>.</p>
    </section>

    <section class="section cite-section">
      <article class="cite-block card">
        <div class="cite-head"><h2>1 · Project</h2><button type="button" class="btn small copy-btn" data-copy-target="proj-bib">Copy BibTeX</button></div>
        <p class="muted">Cite the AccelMark benchmarking framework and methodology.</p>
        <pre class="cite-pre" id="proj-bib">${esc(PROJECT_BIBTEX)}</pre>
      </article>

      <article class="cite-block card">
        <div class="cite-head"><h2>2 · Dataset snapshot</h2>
          <span>
            <button type="button" class="btn small copy-btn" data-copy-target="ds-bib">BibTeX</button>
            <button type="button" class="btn small copy-btn" data-copy-target="ds-plain">Plain text</button>
          </span>
        </div>
        <p class="muted">Cite a specific leaderboard export (<code>v${esc(DATASET_VERSION)}</code>). Checksums in <a href="api/manifest.json">manifest.json</a>.</p>
        <pre class="cite-pre" id="ds-bib">${esc(dsBib)}</pre>
        <pre class="cite-pre faint" id="ds-plain">${esc(plainDataset)}</pre>
      </article>

      <article class="cite-block card">
        <div class="cite-head"><h2>3 · Single result</h2></div>
        <p class="muted">Open any result on the Results page → use <strong>Copy link</strong>, <strong>Copy Markdown</strong>, or <strong>Copy BibTeX</strong> in the detail panel. Each run links to <code>result.json</code>, runner hash, and reproduction script.</p>
        <a class="btn" href="#/rankings">Browse results →</a>
      </article>
    </section>

    <section class="section">
      <div class="section-header"><div class="section-title"><h2>Machine-readable API</h2></div></div>
      <div class="api-links card">
        <a href="api/manifest.json"><code>/api/manifest.json</code></a>
        <a href="api/schema.json"><code>/api/schema.json</code></a>
        <a href="api/rank.json"><code>/api/rank.json</code></a>
        <a href="api/chips.json"><code>/api/chips.json</code></a>
        <a href="api/index.json"><code>/api/index.json</code></a>
        <a href="api/suites.json"><code>/api/suites.json</code></a>
        <p class="muted" style="margin-top:0.75rem">Future: <code>/api/runs.json</code>, <code>/api/runs/&lt;run_id&gt;.json</code>, versioned snapshots, Zenodo DOI per release.</p>
      </div>
    </section>
  `;

  el.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy-target]");
    if (!btn) return;
    const pre = el.querySelector("#" + btn.dataset.copyTarget);
    if (!pre) return;
    const ok = await copyToClipboard(pre.textContent);
    flashButtonLabel(btn, ok ? "Copied!" : "Failed", 1400, ok ? "is-copied" : "is-copy-failed");
  });
}
