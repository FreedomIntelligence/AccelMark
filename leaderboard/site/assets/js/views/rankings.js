// views/rankings.js — placeholder for commit 1.  Will be implemented in commit 2.

import { esc } from "../utils.js";

export function render({ el, query }) {
  const suiteId = query.suite || "suite_A";
  el.innerHTML = `
    <section class="hero">
      <h1>Rankings</h1>
      <p class="tagline">Coming up in the next commit: full per-suite ranking, filters, and compare basket.</p>
    </section>
    <section class="state">
      <span class="state-icon">⚙</span>
      The rankings view for <strong>${esc(suiteId)}</strong> is under construction.<br>
      In the meantime, browse the <a href="#/">home page</a> to see top-3 per suite.
    </section>
  `;
}
