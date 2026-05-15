// views/suites.js — Suites explainer.  This is the "what is this
// benchmark?" page: roofline argument + diagram, scenarios catalog
// with per-scenario specs, per-suite specifications, datasets reference.

import {
  SUITE_ORDER, SUITE_META, suiteFacts, bestRowByMetric,
} from "../data.js";
import { esc, fmtNum, buildHash, shortModel } from "../utils.js";

const GH_BASE = "https://github.com/JuhaoLiang1997/AccelMark";

// One concrete finding per suite, distilled from the paper.  Kept short
// enough to fit inside a single card but specific enough to be useful.
const SUITE_FINDINGS = {
  suite_A: {
    headline: "Offline winner is not the SLA winner",
    body:
      "H200 leads Suite A offline at 5,731 tokens/sec, but on the same suite's " +
      "online tier A100 and A800 sustain 25 queries/sec while H200 caps " +
      "at 10. The 500 ms p99 TTFT SLA binds well before the throughput " +
      "ceiling, so the chip that leads offline loses the production tier.",
  },
  suite_B: {
    headline: "Bottom tier is decided by software stack, not VRAM",
    body:
      "With 8 chips serving 70B BF16, aggregate VRAM is no longer binding. " +
      "Ascend 910C ×16 supplies 1,024 GB and still delivers only 723 tokens/sec, " +
      "below Ascend 910B2 ×8 at 770 tokens/sec. Doubling the hardware buys " +
      "nothing once vllm-ascend's 70B path is the binding constraint.",
  },
  suite_C: {
    headline: "Speed without quality is meaningless",
    body:
      "Quality efficiency multiplies throughput by accuracy. On A100, W8A8 " +
      "wins at 3,776 (1.20× speedup, +2 pts accuracy) because INT8 tensor " +
      "cores engage. FP8 shows zero speedup on A100 because the hardware " +
      "path is absent and compute falls back to BF16. On H100 the same FP8 " +
      "column flips to roughly 1.5 to 1.8× speedup.",
  },
  suite_D: {
    headline: "Long context inverts the bandwidth-bound ranking",
    body:
      "Suite D pushes arithmetic intensity past the roofline knee with " +
      "~28K-token prefill, making the workload compute-bound rather than " +
      "memory-bandwidth-bound. Rankings invert relative to Suite A: chips " +
      "that win on short-prompt decode lose to chips with higher raw FLOPS.",
  },
  suite_E: {
    headline: "Newest flagship has the worst 2× efficiency",
    body:
      "RTX 4090 D tops the Suite E 2× efficiency leaderboard because its " +
      "lower per-die throughput keeps the communication share small. H200, " +
      "the newest flagship, shows the worst NVIDIA 2× efficiency because " +
      "per-chip throughput grew faster than NVLink 4.0 bandwidth did.",
  },
  suite_F: {
    headline: "Edge isolates the pure HBM bandwidth ceiling",
    body:
      "Suite F uses Qwen2.5-0.5B with ~95-token prompts. Stripping residual " +
      "compute-path interference exposes raw memory-bandwidth headroom; " +
      "this is where commodity hardware (RTX 4090, A6000) is most " +
      "competitive on a per-dollar basis.",
  },
  suite_G: {
    headline: "Sparse routing rewards bandwidth over FLOPS",
    body:
      "Mixtral activates only 2 of 8 experts per token, keeping arithmetic " +
      "intensity below dense 8B inference even at 8-chip scale. H20-3e " +
      "trails A100-40G by ~5% on dense Suite A but beats A100-40G ×8 by " +
      "17% on Suite G; its 4,000 GB/s aggregate bandwidth pays off more " +
      "than its compute.",
  },
};

// Three cross-suite ranking inversions distilled from the paper.
const INVERSION_CARDS = [
  {
    eyebrow: "Offline vs Online",
    title: "The throughput winner loses the SLA tier",
    body:
      "Suite A: H200 leads offline at 5,731 tokens/sec, but caps at 10 queries/sec " +
      "once the 500 ms p99 TTFT SLA is enforced. A100 and A800 sustain " +
      "25 queries/sec on the same hardware tier. Offline and online are not the " +
      "same race.",
    suite: "A",
  },
  {
    eyebrow: "Dense vs MoE",
    title: "H20-3e: minus 5% on dense, plus 17% on MoE",
    body:
      "On dense Suite A, H20-3e trails A100-40G by ~5%. On Suite G's " +
      "sparse Mixtral routing the same chip leads A100-40G ×8 by 17%. " +
      "Sparse activation holds arithmetic intensity below dense 8B, so " +
      "bandwidth (not FLOPS) sets the ceiling.",
    suite: "G",
  },
  {
    eyebrow: "Multi-chip Amdahl",
    title: "Newest flagship has the worst 2× scaling",
    body:
      "RTX 4090 D tops the Suite E 2× efficiency table precisely because " +
      "its per-die throughput is low; communication occupies a small share " +
      "of wall time. H200's per-chip throughput outgrew NVLink 4.0 " +
      "bandwidth, so its 2× scaling sits at the bottom of NVIDIA's range.",
    suite: "E",
  },
];

// Shared scenarios catalog.  Each card has:
//   icon         — small SVG illustrating the load shape
//   role         — uppercase tagline (what the scenario reveals)
//   description  — 2 sentences in plain prose
//   spec         — key/value rows (metric / direction / setting)
//   appliesTo    — which suite letters use it (default vs. extra)
const SCENARIO_CATALOG = [
  {
    name: "accuracy",
    icon: "gate",
    role: "Quality gate",
    description:
      "MMLU subset score against the suite's baseline model. Runs before any throughput scenario; a chip that drops accuracy beyond the threshold has every other number on this suite invalidated.",
    spec: [
      { k: "Metric",    v: "MMLU score (0-100)" },
      { k: "Direction", v: "Higher is better" },
      { k: "Threshold", v: "Suite-specific baseline" },
      { k: "Cost",      v: "~5 min / chip" },
    ],
    appliesTo: { default: ["A", "B", "C", "D", "F", "G"], extra: [] },
  },
  {
    name: "offline",
    icon: "batch",
    role: "Peak throughput",
    description:
      "All requests submitted at once, no SLA, no concurrency cap. The pure capability number that establishes the chip's ceiling on this workload.",
    spec: [
      { k: "Metric",    v: "Aggregate tokens/sec" },
      { k: "Direction", v: "Higher is better" },
      { k: "Concurrency", v: "Unbounded (vendor-tuned)" },
      { k: "Cost",      v: "~10 to 15 min / chip" },
    ],
    appliesTo: { default: ["A", "B", "C", "D", "E", "F", "G"], extra: [] },
  },
  {
    name: "online",
    icon: "poisson",
    role: "SLA-bound capacity",
    description:
      "Sweeps offered load under Poisson arrivals and reports the highest queries/sec that still meets the 500 ms p99 TTFT SLA. The number production traffic actually has to honour.",
    spec: [
      { k: "Metric",    v: "Max queries/sec" },
      { k: "Direction", v: "Higher is better" },
      { k: "SLA",       v: "p99 TTFT ≤ 500 ms" },
      { k: "Arrivals",  v: "Poisson, vendor sweep" },
    ],
    appliesTo: { default: ["A", "B", "F", "G"], extra: ["C", "D"] },
  },
  {
    name: "interactive",
    icon: "single",
    role: "Single-stream latency",
    description:
      "One request in-flight at a time, no concurrency. The chat-window UX baseline; minimal queueing, dominated by decode latency and software overhead.",
    spec: [
      { k: "Metric",    v: "TTFT p99 (milliseconds)" },
      { k: "Direction", v: "Lower is better" },
      { k: "Concurrency", v: "1 stream" },
      { k: "Streams",   v: "Many short conversations" },
    ],
    appliesTo: { default: ["F"], extra: ["A", "B", "D", "G"] },
  },
  {
    name: "sustained",
    icon: "longblock",
    role: "Stability under load",
    description:
      "Fixed-concurrency load held for 15 to 30 minutes. Reports the throttle ratio between the first and last 60 s windows so thermal throttling and memory fragmentation surface.",
    spec: [
      { k: "Metric",    v: "Throttle ratio (peak → end)" },
      { k: "Direction", v: "Smaller drop is better" },
      { k: "Duration",  v: "15 to 30 minutes" },
      { k: "Load",      v: "Fixed concurrency" },
    ],
    appliesTo: { default: [], extra: ["A", "B", "C", "D", "F", "G"] },
  },
  {
    name: "speculative",
    icon: "spec",
    role: "Draft-assisted decode",
    description:
      "Offline workload with a 1B draft model loaded alongside the target. Reports speculative decoding acceptance rate and end-to-end speedup; tells you whether spec-decode is worth the VRAM cost.",
    spec: [
      { k: "Metric",    v: "tokens/sec + acceptance rate" },
      { k: "Direction", v: "Higher is better" },
      { k: "Draft model", v: "1B (fits beside target)" },
      { k: "Mode",      v: "Offline (no SLA)" },
    ],
    appliesTo: { default: [], extra: ["A", "D"] },
  },
  {
    name: "burst",
    icon: "burst",
    role: "KV pressure",
    description:
      "Alternates 5× steady arrival rate (short windows) with steady traffic and reports TTFT p99 during the burst. Stresses the KV cache, admission control, and warm-up paths.",
    spec: [
      { k: "Metric",    v: "TTFT p99 during burst" },
      { k: "Direction", v: "Lower is better" },
      { k: "Burst",     v: "5× steady traffic" },
      { k: "Window",    v: "Short pulses + recovery" },
    ],
    appliesTo: { default: [], extra: ["A", "B"] },
  },
];

const DATASETS = [
  {
    name: "sharegpt_standard_v1",
    used: "A · B · C · E · G",
    prompts: "500",
    inputP50: "~280 tok",
    outputP50: "~310 tok",
    notes: "Curated to match production LLM-API traffic; token-length p99 ≈ 2,100.",
  },
  {
    name: "sharegpt_longctx_v1",
    used: "D",
    prompts: "200",
    inputP50: "~28,650 tok",
    outputP50: "≤256 tok",
    notes: "Multi-turn dialogues concatenated to push prefill past the roofline knee.",
  },
  {
    name: "sharegpt_edge_v1",
    used: "F",
    prompts: "500",
    inputP50: "~95 tok",
    outputP50: "~150 tok",
    notes: "Short single-turn prompts; keeps the edge suite bandwidth-isolated.",
  },
];

// Tiny SVG icons keyed by scenario.icon — drawn with currentColor so they
// inherit the surrounding text colour and stay legible in both themes.
const SCN_ICONS = {
  gate: `
    <svg viewBox="0 0 28 28" aria-hidden="true">
      <path d="M14 3 L23 6 L23 14 C23 19.5 19.2 23.5 14 25.5 C8.8 23.5 5 19.5 5 14 L5 6 Z"
            fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M9.5 14.5 L13 18 L19 11" fill="none" stroke="currentColor"
            stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`,
  batch: `
    <svg viewBox="0 0 28 28" aria-hidden="true">
      <rect x="4"  y="6"  width="20" height="2.6" rx="1" fill="currentColor"/>
      <rect x="4"  y="11" width="20" height="2.6" rx="1" fill="currentColor"/>
      <rect x="4"  y="16" width="20" height="2.6" rx="1" fill="currentColor"/>
      <rect x="4"  y="21" width="20" height="2.6" rx="1" fill="currentColor"/>
    </svg>`,
  poisson: `
    <svg viewBox="0 0 28 28" aria-hidden="true">
      <line x1="4"  y1="22" x2="4"  y2="14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="8"  y1="22" x2="8"  y2="11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="13" y1="22" x2="13" y2="16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="17" y1="22" x2="17" y2="10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="21" y1="22" x2="21" y2="14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="25" y1="22" x2="25" y2="18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="3" y1="23" x2="26" y2="23" stroke="currentColor" stroke-width="1" opacity="0.4"/>
    </svg>`,
  single: `
    <svg viewBox="0 0 28 28" aria-hidden="true">
      <line x1="3" y1="22" x2="25" y2="22" stroke="currentColor" stroke-width="1" opacity="0.4"/>
      <line x1="14" y1="22" x2="14" y2="8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <circle cx="14" cy="6" r="2" fill="currentColor"/>
    </svg>`,
  longblock: `
    <svg viewBox="0 0 28 28" aria-hidden="true">
      <rect x="3"  y="10" width="18" height="8" rx="1.5" fill="currentColor"/>
      <rect x="22" y="13" width="4"  height="5" rx="1"   fill="currentColor" opacity="0.55"/>
    </svg>`,
  spec: `
    <svg viewBox="0 0 28 28" aria-hidden="true">
      <path d="M3 9 L17 9 L17 6 L25 12 L17 18 L17 15 L3 15 Z"
            fill="currentColor" opacity="0.45"/>
      <path d="M3 17 L21 17 L21 14 L26 18 L21 22 L21 20 L3 20 Z"
            fill="currentColor"/>
    </svg>`,
  burst: `
    <svg viewBox="0 0 28 28" aria-hidden="true">
      <line x1="3"  y1="22" x2="25" y2="22" stroke="currentColor" stroke-width="1" opacity="0.4"/>
      <line x1="3"  y1="22" x2="3"  y2="18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="7"  y1="22" x2="7"  y2="18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="11" y1="22" x2="11" y2="6"  stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="15" y1="22" x2="15" y2="8"  stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="19" y1="22" x2="19" y2="18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="23" y1="22" x2="23" y2="18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    </svg>`,
};

// Roofline mini-diagram.  Coordinates are hand-tuned so the seven dots
// land at roughly the right region of the spectrum for each suite.
const ROOFLINE_POINTS = [
  { letter: "F", x:  68, y: 168, label: "right" },  // far left, bandwidth-bound, smallest model
  { letter: "A", x: 102, y: 132, label: "right" },  // bandwidth-bound, 8B decode
  { letter: "G", x: 118, y: 122, label: "right" },  // MoE, similar region as A
  { letter: "B", x: 138, y:  96, label: "left"  },  // 70B multi-chip
  { letter: "C", x: 168, y:  68, label: "left"  },  // quantization, transition
  { letter: "E", x: 192, y:  56, label: "right" },  // scaling, near the knee
  { letter: "D", x: 268, y:  56, label: "left"  },  // compute-bound long context
];

export function render({ el }) {
  el.innerHTML = `
    <section class="hero suites-hero">
      <h1>Workload Suites</h1>
      <p class="hero-sub">
        Each suite anchors a distinct bottleneck region, together sampling
        the full inference workload spectrum.
      </p>
      <div class="hero-cta">
        <a class="btn primary" href="#/rankings">Browse rankings →</a>
        <a class="btn" href="${GH_BASE}/tree/main/suites" target="_blank" rel="noopener">Suite spec on GitHub</a>
      </div>
    </section>

    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">01 · Methodology</span>
          <h2>Why per-suite, not a single score?</h2>
        </div>
      </div>

      <div class="why-grid">
        <div class="why-prose">
          <p>
            AI inference workloads span a wide range of arithmetic intensity.
            The roofline model makes the consequence concrete: a chip's
            effective performance is set by whichever of memory bandwidth or
            compute is binding for the workload. Because different workloads
            occupy different regions of that spectrum, hardware rankings
            <em>are not preserved</em> across them.
          </p>
          <details class="why-prose-more">
            <summary>
              <span class="why-more-show">Read the full argument</span>
              <span class="why-more-hide">Show less</span>
            </summary>
            <p>
              A chip optimized for one region, say bandwidth-bound 8B decode,
              diverges from a chip optimized for another, say compute-bound
              long-context prefill, as soon as the workload moves. Collapsing
              heterogeneous workloads into a single composite score hides
              exactly the trade-offs a buyer needs to see.
            </p>
            <p>
              AccelMark operationalizes <strong>spectrum sampling</strong>: a
              set of suites, each anchored to a qualitatively distinct
              bottleneck region. Bandwidth-bound serving at 8B (A) and 0.5B
              (F); capacity-then-stack-bound 70B multi-chip (B); the
              bandwidth-to-compute transition via quantization (C);
              compute-bound long-context prefill (D); multi-chip communication
              overhead (E); and sparse MoE routing (G).
            </p>
          </details>
        </div>
        ${renderRoofline()}
      </div>

      <div class="inversion-grid">
        ${INVERSION_CARDS.map(renderInversion).join("")}
      </div>
    </section>

    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">02 · Scenarios</span>
          <h2>Seven protocols, one suite at a time</h2>
        </div>
      </div>
      <p class="section-lede">
        Each suite picks a subset of these seven protocols. The metric,
        direction, and setting are pinned here once; per-suite cards below
        just say which apply.
        <strong>Default</strong> scenarios are required for a valid
        submission; <strong>extras</strong> are opt-in for vendors who want
        to characterize a regime further.
      </p>
      <ul class="scn-catalog">
        ${SCENARIO_CATALOG.map(renderScenarioCard).join("")}
      </ul>
    </section>

    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">03 · Specifications</span>
          <h2>Each suite, in detail</h2>
        </div>
      </div>
      <p class="section-lede">
        One self-contained card per suite. The header pins the primary
        metric and direction; the workload strip pins the model, hardware
        budget, precision, and dataset; the protocols row shows which
        scenarios apply; current leaders surface the top chip per metric.
      </p>
      <div class="suite-spec-list">
        ${SUITE_ORDER.map(renderSuiteSpec).join("")}
      </div>
    </section>

    <section class="section">
      <div class="section-header section-header--stacked">
        <div class="section-title">
          <span class="eyebrow">04 · Datasets</span>
          <h2>Three immutable prompt sets</h2>
        </div>
      </div>
      <p class="section-lede">
        Datasets are content-hash-pinned: once a name is published, the
        bytes never change. Revising a dataset means a new version
        (<code>_v2</code>, etc.). Every result is tied to a dataset hash so
        comparisons stay apples-to-apples across years.
      </p>
      <div class="dataset-table">
        <div class="dataset-row dataset-row--head">
          <span>Dataset</span>
          <span>Used by</span>
          <span>Prompts</span>
          <span>Input p50</span>
          <span>Output p50</span>
          <span class="dt-notes">Why</span>
        </div>
        ${DATASETS.map(renderDataset).join("")}
      </div>
    </section>

    <section class="section submit-section">
      <div class="submit-card">
        <span class="eyebrow">05 · Extend</span>
        <h2 class="submit-title">Propose a new suite</h2>
        <p class="submit-body">
          Have a workload regime AccelMark doesn't cover yet: long-context
          serving, speculative decoding economics, a domain-specific
          fine-tune? Open a discussion with a one-page sketch of the
          bottleneck region and reference SLAs. The contribution flow is
          the same as a new result.
        </p>
        <div class="submit-cta">
          <a class="btn primary" href="${GH_BASE}/discussions/new?category=ideas"
             target="_blank" rel="noopener">Propose a suite →</a>
          <a class="btn" href="${GH_BASE}/blob/main/CONTRIBUTING.md"
             target="_blank" rel="noopener">Read the contributor guide</a>
        </div>
      </div>
    </section>
  `;

  // Wire up smooth scroll for the scenario catalog's "Used by" suite-letter
  // chips.  These use #suite-X anchors, but the SPA hash router would
  // otherwise interpret the click as a route change.  Intercept here and
  // scroll the matching <article id="suite-X"> into view.
  //
  // Use a once-attach guard: the router rebuilds the view's innerHTML on
  // each visit, but the listener is on `el` itself and would accumulate
  // across re-renders without this flag.
  if (!el.__suitesScrollAttached) {
    el.addEventListener("click", (ev) => {
      const a = ev.target.closest(".scn-suite-letter");
      if (!a) return;
      const href = a.getAttribute("href") || "";
      if (!href.startsWith("#suite-")) return;
      ev.preventDefault();
      const id = href.slice(1);
      const target = el.querySelector(`#${CSS.escape(id)}`);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      // Brief highlight so the user sees which card they landed on.
      target.classList.remove("suite-spec--flash");
      // Force reflow so the animation restarts when re-clicked.
      void target.offsetWidth;
      target.classList.add("suite-spec--flash");
    });
    el.__suitesScrollAttached = true;
  }

  // Default the methodology "Read the full argument" disclosure based
  // on viewport width: desktop readers see the whole essay up front,
  // mobile readers see the opening paragraph + an opt-in expander so
  // the rest of the page stays reachable without a long scroll.  Pure
  // mount-time decision — we don't react to viewport changes since
  // toggling `open` mid-read would yank the user's scroll position.
  for (const det of el.querySelectorAll(".why-prose-more")) {
    const isDesktop = typeof window.matchMedia === "function"
      && window.matchMedia("(min-width: 880px)").matches;
    det.open = isDesktop;
  }
}

function renderRoofline() {
  const dots = ROOFLINE_POINTS.map((p) => {
    const labelDx = p.label === "left" ? -10 : 10;
    const textAnchor = p.label === "left" ? "end" : "start";
    return `
      <g class="rfl-dot" data-suite="${esc(p.letter)}">
        <circle cx="${p.x}" cy="${p.y}" r="7"/>
        <text x="${p.x + labelDx}" y="${p.y + 3.5}" class="rfl-letter" text-anchor="${textAnchor}">${esc(p.letter)}</text>
      </g>
    `;
  }).join("");
  return `
    <aside class="roofline-diagram" aria-label="Roofline diagram of AccelMark suites">
      <span class="eyebrow roofline-eyebrow">Roofline</span>
      <svg class="roofline-svg" viewBox="0 0 340 220" xmlns="http://www.w3.org/2000/svg" role="img">
        <!-- axes -->
        <line x1="40" y1="190" x2="320" y2="190" class="rfl-axis"/>
        <line x1="40" y1="20"  x2="40"  y2="190" class="rfl-axis"/>

        <!-- roofline (bandwidth-bound slope → compute-bound ceiling) -->
        <path d="M 40 178 L 190 50 L 320 50" class="rfl-roof"/>

        <!-- region annotations -->
        <text x="58"  y="116" class="rfl-region rfl-region--bw">bandwidth-bound</text>
        <text x="200" y="42"  class="rfl-region rfl-region--cp">compute-bound</text>

        <!-- knee marker -->
        <line x1="190" y1="50" x2="190" y2="190" class="rfl-knee"/>
        <text x="194" y="200" class="rfl-knee-label">roofline knee</text>

        ${dots}

        <!-- axis labels -->
        <text x="42"  y="14"  class="rfl-axis-label">throughput</text>
        <text x="320" y="208" class="rfl-axis-label" text-anchor="end">arithmetic intensity →</text>
      </svg>
      <p class="roofline-caption">
        Each suite sits at a different point on the roofline. Bandwidth-bound
        regimes (left of the knee) reward HBM throughput; compute-bound
        regimes (right) reward raw FLOPS. A chip's ranking changes as the
        workload moves.
      </p>
    </aside>
  `;
}

function renderInversion(card) {
  return `
    <article class="inversion-card" data-suite="${esc(card.suite)}">
      <span class="eyebrow">${esc(card.eyebrow)}</span>
      <h3>${esc(card.title)}</h3>
      <p>${esc(card.body)}</p>
    </article>
  `;
}

function renderScenarioCard(scn) {
  const defaults = scn.appliesTo.default || [];
  const extras   = scn.appliesTo.extra   || [];
  const icon = SCN_ICONS[scn.icon] || "";
  return `
    <li class="scn-card">
      <header class="scn-card-head">
        <span class="scn-icon" aria-hidden="true">${icon}</span>
        <div class="scn-card-id">
          <h3 class="scn-card-name">${esc(scn.name)}</h3>
          <span class="scn-card-role">${esc(scn.role)}</span>
        </div>
      </header>
      <p class="scn-card-desc">${esc(scn.description)}</p>
      <dl class="scn-card-spec">
        ${scn.spec.map((row) => `
          <div class="scn-spec-row">
            <dt>${esc(row.k)}</dt>
            <dd>${esc(row.v)}</dd>
          </div>
        `).join("")}
      </dl>
      <div class="scn-card-applies">
        <span class="scn-applies-label">Used by</span>
        <span class="scn-applies-letters">
          ${defaults.map((l) => `<a class="scn-suite-letter" data-suite="${esc(l)}" href="#suite-${esc(l)}">${esc(l)}</a>`).join("")}
          ${extras.length && defaults.length ? `<span class="scn-applies-sep">·</span>` : ""}
          ${extras.map((l) => `<a class="scn-suite-letter scn-suite-letter--extra" data-suite="${esc(l)}" href="#suite-${esc(l)}">${esc(l)}</a>`).join("")}
        </span>
        ${extras.length ? `<span class="scn-applies-note">${defaults.length ? "extras" : "all extra"}</span>` : ""}
      </div>
    </li>
  `;
}

function renderSuiteSpec(suiteId) {
  const meta = SUITE_META[suiteId];
  if (!meta) return "";
  const facts = suiteFacts(suiteId);
  const finding = SUITE_FINDINGS[suiteId];
  const wl = meta.workload || {};
  const scenarios = meta.scenarios || [];

  const directionLabel = meta.primary.direction === "asc"
    ? "Lower is better"
    : "Higher is better";

  const leaderScenarios = scenarios.filter((s) => s.metric);

  return `
    <article class="suite-spec" data-suite="${esc(meta.letter)}" id="suite-${esc(meta.letter)}">
      <header class="suite-spec-head">
        <span class="suite-spec-letter">${esc(meta.letter)}</span>
        <div class="suite-spec-title">
          <span class="eyebrow">Suite ${esc(meta.letter)}</span>
          <h3>${esc(meta.title)}</h3>
          <p class="suite-spec-tagline">${esc(meta.tagline)}</p>
        </div>
        <span class="suite-spec-metric">
          <span class="metric-label">${esc(meta.primary.label)}</span>
          <span class="metric-direction">${esc(directionLabel)}</span>
        </span>
      </header>

      <div class="suite-spec-body">
        <div class="suite-intro-row">
          ${meta.description ? `<p class="suite-spec-intro">${esc(meta.description)}</p>` : `<div></div>`}
          ${finding ? `
            <aside class="suite-finding suite-finding--side">
              <span class="finding-eyebrow">Concrete finding</span>
              <p class="finding-headline">${esc(finding.headline)}</p>
              <p class="finding-body">${esc(finding.body)}</p>
            </aside>
          ` : ""}
        </div>

        <ul class="spec-strip">
          <li><span class="strip-k">Model</span><span class="strip-v">${esc(shortModel(wl.model) || "-")}</span></li>
          <li><span class="strip-k">Chips</span><span class="strip-v">${esc(wl.chips || "-")}</span></li>
          <li><span class="strip-k">Precision</span><span class="strip-v">${esc(wl.precision || "-")}</span></li>
          <li><span class="strip-k">Dataset</span><span class="strip-v"><code>${esc(wl.dataset || "-")}</code></span></li>
          <li><span class="strip-k">Tokens (in / out)</span><span class="strip-v tnum">${esc(wl.inputTokens || "-")} / ${esc(wl.outputTokens || "-")}</span></li>
          <li><span class="strip-k">Coverage</span><span class="strip-v tnum">${fmtNum(facts.submissions)} results &middot; ${fmtNum(facts.chips)} chips</span></li>
        </ul>

        <div class="suite-scns">
          <span class="suite-scns-label">Protocols</span>
          <ul class="suite-scns-list">
            ${scenarios.map((s) => `
              <li class="scn-pill ${s.isExtra ? "scn-extra" : "scn-default"}" title="${esc(s.isExtra ? "Extra protocol (opt-in)" : "Default protocol")}">
                ${esc(s.name)}${s.isExtra ? `<span class="scn-extra-tag">extra</span>` : ""}
              </li>
            `).join("")}
          </ul>
        </div>

        ${leaderScenarios.length ? `
          <div class="suite-leaders">
            <span class="suite-leaders-label">Current leaders</span>
            <ul class="leader-list">
              ${leaderScenarios.map((s) => renderLeaderRow(suiteId, s)).join("")}
            </ul>
          </div>
        ` : ""}

        <div class="suite-spec-cta">
          <a class="btn primary small"
             href="${esc(buildHash("/rankings", { suite: suiteId }))}">
            Open ranking →
          </a>
          <a class="btn small"
             href="${GH_BASE}/blob/main/suites/${esc(suiteId)}/suite.json"
             target="_blank" rel="noopener">
            View suite.json
          </a>
        </div>
      </div>
    </article>
  `;
}

function renderLeaderRow(suiteId, scn) {
  const row = bestRowByMetric(suiteId, scn.metric.key, scn.metric.direction);
  const pillClass = scn.isExtra ? "scn-extra" : "scn-default";
  if (!row) {
    return `
      <li class="leader-row leader-row--empty">
        <span class="scn-pill ${pillClass}">${esc(scn.name)}</span>
        <span class="leader-empty">No qualifying submissions yet</span>
      </li>
    `;
  }
  const val = formatScnMetric(row[scn.metric.key], scn.metric);
  return `
    <li class="leader-row">
      <span class="scn-pill ${pillClass}">${esc(scn.name)}</span>
      <a class="leader-chip"
         href="${esc(buildHash("/rankings", { suite: suiteId }))}"
         title="View ${esc(scn.name)} ranking">
        <span class="leader-vendor" data-vendor="${esc(row.vendor)}"></span>
        <span class="leader-chip-name">${esc(row._chip_label)}</span>
      </a>
      <span class="leader-val tnum">${esc(val)}</span>
    </li>
  `;
}

function formatScnMetric(value, m) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  const scale = m.scale || 1;
  const decimals = m.decimals != null
    ? m.decimals
    : Math.abs(value * scale) >= 100 ? 0 : 1;
  const num = (value * scale).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return m.unit ? `${num} ${m.unit}` : num;
}

function renderDataset(d) {
  return `
    <div class="dataset-row">
      <span class="dt-name"><code>${esc(d.name)}</code></span>
      <span class="dt-used">${esc(d.used)}</span>
      <span class="dt-count tnum">${esc(d.prompts)}</span>
      <span class="dt-in tnum">${esc(d.inputP50)}</span>
      <span class="dt-out tnum">${esc(d.outputP50)}</span>
      <span class="dt-notes">${esc(d.notes)}</span>
    </div>
  `;
}
