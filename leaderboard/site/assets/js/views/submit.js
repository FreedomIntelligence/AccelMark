// submit.js — Submit wizard (static guided flow).

import { esc } from "../utils.js";
import { SUITE_ORDER, SUITE_META } from "../data.js";
import {
  VENDORS, chipsForVendor, vendorById, chipOptionsHtml,
  recommendedSuites,
} from "../hardware-catalog.js";

const STEPS = ["Hardware", "Suite", "Run", "Submit"];

const SUITE_HINTS = {
  suite_A: { mins: 11, note: "Best first benchmark · single-chip inference" },
  suite_F: { mins: 10, note: "Edge / consumer GPU · offline + online" },
  suite_B: { mins: 20, note: "Large-model multi-chip" },
  suite_C: { mins: 22, note: "Quantization efficiency" },
  suite_D: { mins: 22, note: "Long-context inference" },
  suite_E: { mins: 9, note: "Multi-chip scaling" },
  suite_G: { mins: 35, note: "MoE multi-chip" },
};

const FLOW = ["Detect", "Run", "Validate", "Preview", "Submit", "Published"];

function parsePrefill(query) {
  if (!query) return { vendor: "", chip: "" };
  return {
    vendor: query.vendor || "",
    chip: query.chip || "",
  };
}

export function render({ query, el }) {
  const prefill = parsePrefill(query);

  el.innerHTML = `
    <section class="page-hero">
      <span class="eyebrow">Contribute</span>
      <h1>Submit your benchmark</h1>
      <p class="page-lead">From hardware detection to a published, citable result — pick from the full accelerator catalog or enter custom hardware.</p>
    </section>

    <div class="flow-bar" aria-label="Submission pipeline">
      ${FLOW.map((s, i) => `
        <span class="flow-step${i < 4 ? " active" : ""}"><span class="flow-num">${i + 1}</span>${esc(s)}</span>
        ${i < FLOW.length - 1 ? '<span class="flow-arrow" aria-hidden="true">→</span>' : ""}
      `).join("")}
    </div>

    <div class="wizard-shell card" id="submit-wizard">
      <nav class="wizard-tabs" role="tablist">
        ${STEPS.map((s, i) => `<button type="button" class="wizard-tab${i === 0 ? " active" : ""}" data-step="${i}" role="tab">${i + 1}. ${esc(s)}</button>`).join("")}
      </nav>

      <div class="wizard-panel active" data-panel="0">
        <h2>Choose your hardware</h2>
        <p class="muted">Full catalog — datacenter, workstation, consumer, cloud TPU, and domestic accelerators.</p>
        <label class="wizard-field wizard-field-wide"><span>Search catalog</span>
          <input type="search" id="sw-search" placeholder="e.g. H100, MI350, Ascend 910C, RTX 5090…" autocomplete="off">
        </label>
        <div class="wizard-grid">
          <label class="wizard-field"><span>Vendor</span>
            <select id="sw-vendor">${VENDORS.map((v) => `<option value="${esc(v.id)}">${esc(v.label)}</option>`).join("")}</select>
          </label>
          <label class="wizard-field wizard-field-wide"><span>GPU / accelerator</span>
            <select id="sw-chip"></select>
          </label>
          <label class="wizard-field"><span>Or custom name</span>
            <input type="text" id="sw-custom" placeholder="If not listed above">
          </label>
          <label class="wizard-field"><span>Chip count</span>
            <select id="sw-count"><option value="1">1</option><option value="2">2</option><option value="4">4</option><option value="8">8</option></select>
          </label>
        </div>
        <p class="muted catalog-hint" id="sw-chip-hint"></p>
      </div>

      <div class="wizard-panel" data-panel="1">
        <h2>Pick a suite</h2>
        <p class="muted">New contributors: start with <strong>Suite A</strong> or <strong>Suite F</strong> (~10 min).</p>
        <div class="suite-pick-grid" id="sw-suites"></div>
      </div>

      <div class="wizard-panel" data-panel="2">
        <h2>Run &amp; validate</h2>
        <p class="muted" id="sw-time-hint">First benchmark in ~11 minutes on datacenter GPUs.</p>
        <div class="cmd-block">
          <div class="cmd-head"><span>Install</span><button type="button" class="btn small copy-btn" data-copy="install">Copy</button></div>
          <pre id="sw-cmd-install">git clone https://github.com/FreedomIntelligence/AccelMark.git
cd AccelMark
pip install -e .</pre>
        </div>
        <div class="cmd-block">
          <div class="cmd-head"><span>Benchmark</span><button type="button" class="btn small copy-btn" data-copy="run">Copy</button></div>
          <pre id="sw-cmd-run"></pre>
        </div>
        <div class="cmd-block">
          <div class="cmd-head"><span>Validate locally</span><button type="button" class="btn small copy-btn" data-copy="validate">Copy</button></div>
          <pre id="sw-cmd-validate"></pre>
        </div>
      </div>

      <div class="wizard-panel" data-panel="3">
        <h2>Submit for publication</h2>
        <p class="muted">Every merged result links to <code>result.json</code>, <code>env_info.json</code>, runner hash, and reproduction instructions. Your GitHub ID becomes your <a href="#/contributors">contributor profile</a>.</p>
        <div class="submit-options">
          <a class="submit-option card is-link" href="https://github.com/FreedomIntelligence/AccelMark/blob/main/CONTRIBUTING.md#submitting-a-result" target="_blank" rel="noopener">
            <strong>Pull request (recommended)</strong>
            <span class="muted">Fork → run benchmark → open PR. CI validates schema, accuracy, and runner hash.</span>
          </a>
          <a class="submit-option card is-link" href="https://github.com/FreedomIntelligence/AccelMark/issues/new?template=result_submission.md" target="_blank" rel="noopener">
            <strong>Upload via GitHub Issue</strong>
            <span class="muted">Attach <code>result.json</code> + <code>env_info.json</code>. Maintainers help land your PR.</span>
          </a>
          <a class="submit-option card is-link" href="https://github.com/FreedomIntelligence/AccelMark/discussions/new?category=show-and-tell" target="_blank" rel="noopener">
            <strong>Share in Discussions</strong>
            <span class="muted">Show and tell — hardware tips, optimization notes, reproduction requests.</span>
          </a>
        </div>
        <div class="trust-evidence compact">
          <span>Artifacts per run:</span>
          <code>result.json</code><code>env_info.json</code><code>runner hash</code><code>accuracy receipt</code>
        </div>
      </div>

      <div class="wizard-nav">
        <button type="button" class="btn" id="sw-prev" disabled>← Back</button>
        <button type="button" class="btn primary" id="sw-next">Next →</button>
      </div>
    </div>
  `;

  let step = 0;
  const defaultVendor = prefill.vendor && VENDORS.some((v) => v.id === prefill.vendor)
    ? prefill.vendor
    : "NVIDIA";
  const state = {
    vendor: defaultVendor,
    chip: chipsForVendor(defaultVendor)[0]?.name || "Custom accelerator",
    custom: "",
    count: "1",
    suite: "suite_A",
  };

  function effectiveChip() {
    return state.custom.trim() || state.chip;
  }

  function selectedChipEntry() {
    const list = chipsForVendor(state.vendor);
    return list.find((c) => c.name === state.chip) || list[0];
  }

  function fillChips(keepSelection) {
    const sel = el.querySelector("#sw-chip");
    const prev = keepSelection ? state.chip : null;
    sel.innerHTML = chipOptionsHtml(state.vendor, prev || state.chip);
    const chips = chipsForVendor(state.vendor);
    if (prefill.chip && chips.some((c) => c.name === prefill.chip)) {
      state.chip = prefill.chip;
      sel.value = prefill.chip;
      prefill.chip = "";
    } else if (prev && chips.some((c) => c.name === prev)) {
      state.chip = prev;
      sel.value = prev;
    } else {
      state.chip = chips[0]?.name || "Custom accelerator";
      sel.value = state.chip;
    }
    updateChipHint();
  }

  function updateChipHint() {
    const entry = selectedChipEntry();
    const hint = el.querySelector("#sw-chip-hint");
    if (!entry || !hint) return;
    const mem = entry.memoryGb ? `${entry.memoryGb} GB · ` : "";
    const suites = (entry.suites || recommendedSuites(entry, state.count)).join(", ");
    hint.textContent = `${mem}${entry.tier || "accelerator"} · suggested: ${suites}`;
  }

  function fillSuites() {
    const grid = el.querySelector("#sw-suites");
    const entry = selectedChipEntry();
    const rec = recommendedSuites(entry, state.count);
    grid.innerHTML = SUITE_ORDER.map((id) => {
      const meta = SUITE_META[id];
      const hint = SUITE_HINTS[id];
      const isRec = rec.includes(id);
      return `
        <label class="suite-pick${state.suite === id ? " selected" : ""}${isRec ? " recommended" : ""}">
          <input type="radio" name="suite" value="${esc(id)}"${state.suite === id ? " checked" : ""}>
          <span class="suite-pick-letter">Suite ${esc(meta?.letter || "?")}</span>
          <span class="suite-pick-title">${esc(meta?.title || id)}</span>
          <span class="suite-pick-time">~${hint?.mins || "?"} min</span>
          ${isRec ? '<span class="suite-pick-badge">Recommended</span>' : ""}
        </label>`;
    }).join("");
  }

  function vendorRunner() {
    return vendorById(state.vendor)?.runner || VENDORS[0].runner;
  }

  function slugify(s) {
    return s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  }

  function updateCommands() {
    const runner = vendorRunner();
    const chipName = effectiveChip();
    const runName = `${slugify(state.vendor)}_${slugify(chipName)}_${state.count}x1_${state.suite}_${runner.slice(0, 12)}`;
    const hint = SUITE_HINTS[state.suite];
    el.querySelector("#sw-time-hint").textContent =
      hint ? `Estimated runtime: ~${hint.mins} min · ${hint.note}` : "";

    el.querySelector("#sw-cmd-run").textContent =
      `python run.py --runner ${runner} --suite ${state.suite}` +
      (state.count !== "1" ? ` --chip-count ${state.count}` : "");

    el.querySelector("#sw-cmd-validate").textContent =
      `python runners/validate_submission.py results/community/${runName}/result.json`;
  }

  function showStep(n) {
    step = Math.max(0, Math.min(STEPS.length - 1, n));
    el.querySelectorAll(".wizard-tab").forEach((t, i) => t.classList.toggle("active", i === step));
    el.querySelectorAll(".wizard-panel").forEach((p, i) => p.classList.toggle("active", i === step));
    el.querySelector("#sw-prev").disabled = step === 0;
    el.querySelector("#sw-next").textContent = step === STEPS.length - 1 ? "Done" : "Next →";
    if (step === 2) updateCommands();
  }

  function applySearch(q) {
    const needle = q.trim().toLowerCase();
    if (!needle) return;
    outer: for (const v of VENDORS) {
      for (const c of chipsForVendor(v.id)) {
        const hay = `${v.label} ${c.name} ${c.memoryGb || ""}`.toLowerCase();
        if (hay.includes(needle)) {
          state.vendor = v.id;
          el.querySelector("#sw-vendor").value = v.id;
          fillChips(false);
          state.chip = c.name;
          el.querySelector("#sw-chip").value = c.name;
          state.custom = "";
          el.querySelector("#sw-custom").value = "";
          fillSuites();
          updateCommands();
          updateChipHint();
          break outer;
        }
      }
    }
  }

  if (prefill.vendor) {
    el.querySelector("#sw-vendor").value = state.vendor;
  }

  fillChips(false);
  fillSuites();
  updateCommands();

  el.querySelector("#sw-vendor").addEventListener("change", (e) => {
    state.vendor = e.target.value;
    fillChips(false);
    fillSuites();
    updateCommands();
  });
  el.querySelector("#sw-chip").addEventListener("change", (e) => {
    state.chip = e.target.value;
    state.custom = "";
    el.querySelector("#sw-custom").value = "";
    fillSuites();
    updateCommands();
    updateChipHint();
  });
  el.querySelector("#sw-custom").addEventListener("input", (e) => {
    state.custom = e.target.value;
    updateCommands();
  });
  el.querySelector("#sw-count").addEventListener("change", (e) => {
    state.count = e.target.value;
    fillSuites();
    updateCommands();
    updateChipHint();
  });
  el.querySelector("#sw-suites").addEventListener("change", (e) => {
    if (e.target.name === "suite") {
      state.suite = e.target.value;
      fillSuites();
      updateCommands();
    }
  });
  el.querySelector("#sw-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      applySearch(e.target.value);
    }
  });
  el.querySelector("#sw-prev").addEventListener("click", () => showStep(step - 1));
  el.querySelector("#sw-next").addEventListener("click", () => {
    if (step < STEPS.length - 1) showStep(step + 1);
    else location.hash = "#/rankings";
  });
  el.querySelectorAll(".wizard-tab").forEach((t) => {
    t.addEventListener("click", () => showStep(Number(t.dataset.step)));
  });

  el.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    const key = btn.dataset.copy;
    const pre = el.querySelector(key === "install" ? "#sw-cmd-install" : key === "run" ? "#sw-cmd-run" : "#sw-cmd-validate");
    if (!pre) return;
    const { copyToClipboard, flashButtonLabel } = await import("../utils.js");
    const ok = await copyToClipboard(pre.textContent);
    flashButtonLabel(btn, ok ? "Copied!" : "Failed", 1400, ok ? "is-copied" : "is-copy-failed");
  });

  if (prefill.chip || prefill.vendor) {
    applySearch(prefill.chip || prefill.vendor);
  }
}
