// data.js — single source of truth for leaderboard rows.
//
// The legacy generator emits leaderboard.js which sets `window.LEADERBOARD_DATA`
// as a side-effect when included via <script>. We import it through that route
// to keep the generator output unchanged (zero-build pipeline).
//
// All views consume LeaderboardData from this module.  Do not access
// window.LEADERBOARD_DATA directly anywhere else.

import { groupBy, maxBy, chipSlug, toTitleCase } from "./utils.js";

// Each suite has a "primary metric" most relevant to a buyer's question.
// This drives default sort on the rankings page and the top-3 podium on home.
//
// `primary.scale` multiplies raw value at display (e.g. 0.945 → 94.5 %).
// `primary.decimals` overrides automatic decimal selection.
// Suite workload constants — fixed per suite definition (suites/README.md).
// `inputTokens` / `outputTokens` are the dataset p50s used at benchmark
// time and are NOT derived from data files; they're part of the suite
// contract and only change with a suite revision.
export const SUITE_META = {
  suite_A: {
    letter: "A",
    title: "Single-chip throughput",
    tagline: "How fast can one accelerator serve an 8B model?",
    description:
      "The canonical bandwidth-bound regime. 8B Llama on a single accelerator is small enough to fit comfortably in HBM, large enough that decode is memory-bandwidth-bound rather than compute-bound. This is the bread-and-butter serving workload that anchors most other LLM benchmarks, and the suite where vendor marketing numbers usually land.",
    primary: { key: "offline_throughput",     label: "tokens/sec",      direction: "desc", unit: "tokens/sec" },
    workload: {
      model: "meta-llama/Meta-Llama-3-8B-Instruct",
      chips: "1",
      precision: "BF16",
      dataset: "sharegpt_standard_v1",
      inputTokens: "~280",
      outputTokens: "~310",
    },
    scenarios: [
      { name: "accuracy",    isExtra: false,
        desc: "MMLU subset score against the baseline. Gate for a valid submission." },
      { name: "offline",     isExtra: false,
        desc: "Max throughput with all requests batched at once.",
        metric: { key: "offline_throughput", label: "tokens/sec", direction: "desc", unit: "tokens/sec" } },
      { name: "online",      isExtra: false,
        desc: "Highest QPS that meets the 500 ms p99 TTFT SLA under Poisson arrivals.",
        metric: { key: "online_max_qps", label: "queries/sec", direction: "desc", unit: "queries/sec" } },
      { name: "interactive", isExtra: true,
        desc: "Single-stream first-token latency. No concurrency.",
        metric: { key: "interactive_ttft_p99", label: "TTFT p99", direction: "asc", unit: "ms", decimals: 0 } },
      { name: "sustained",   isExtra: true,
        desc: "30 min fixed-concurrency load. Reports throughput stability and throttle ratio.",
        metric: { key: "sustained_throughput", label: "sustained throughput", direction: "desc", unit: "tokens/sec" } },
      { name: "speculative", isExtra: true,
        desc: "Offline workload with a 1B draft model loaded. Reports acceptance rate." },
      { name: "burst",       isExtra: true,
        desc: "TTFT p99 during 5x burst windows versus steady. KV pressure test." },
    ],
  },
  suite_B: {
    letter: "B",
    title: "Multi-chip throughput",
    tagline: "Large-model serving across multiple chips.",
    description:
      "70B Llama distributed across multiple accelerators. Two effects compound: the model itself no longer fits on one chip (capacity-bound) and tensor-parallel inference shards KV cache, activations, and all-reduce traffic over the interconnect. Both the framework's TP path and the chip's NVLink / Infinity Fabric / scale-out fabric come under test here.",
    primary: { key: "offline_throughput",     label: "tokens/sec",      direction: "desc", unit: "tokens/sec" },
    workload: {
      model: "meta-llama/Meta-Llama-3-70B-Instruct",
      chips: "flexible (typ. 4 / 8)",
      precision: "BF16",
      dataset: "sharegpt_standard_v1",
      inputTokens: "~280",
      outputTokens: "~310",
    },
    scenarios: [
      { name: "accuracy",    isExtra: false,
        desc: "MMLU subset score against the 70B baseline." },
      { name: "offline",     isExtra: false,
        desc: "Aggregate throughput across N chips serving the 70B model.",
        metric: { key: "offline_throughput", label: "tokens/sec", direction: "desc", unit: "tokens/sec" } },
      { name: "online",      isExtra: false,
        desc: "Highest QPS that meets the 500 ms p99 TTFT SLA at 70B scale.",
        metric: { key: "online_max_qps", label: "queries/sec", direction: "desc", unit: "queries/sec" } },
      { name: "interactive", isExtra: true,
        desc: "Single-stream TTFT at 70B. Decode-bound." },
      { name: "sustained",   isExtra: true,
        desc: "30 min fixed load; concurrency 4 (70B leaves less KV headroom than 8B).",
        metric: { key: "sustained_throughput", label: "sustained throughput", direction: "desc", unit: "tokens/sec" } },
      { name: "burst",       isExtra: true,
        desc: "Burst vs steady TTFT p99 at 70B scale." },
    ],
  },
  suite_C: {
    letter: "C",
    title: "Quantization efficiency",
    tagline: "Quality-adjusted throughput across precision formats.",
    description:
      "The bandwidth-to-compute transition. The same 8B model is run at five precision formats (BF16, FP8, W8A8, W8A16, W4A16); quality efficiency multiplies throughput speedup by the accuracy drop so a chip can't trade quality for speed silently. Reveals which chips have working low-precision tensor cores and which fall back to BF16 on the same instruction.",
    primary: { key: "quant_quality_eff",      label: "quality efficiency", direction: "desc", unit: "" },
    workload: {
      model: "meta-llama/Llama-3.1-8B-Instruct",
      chips: "1",
      precision: "BF16, FP8, W8A8, W8A16, W4A16",
      dataset: "sharegpt_standard_v1",
      inputTokens: "~280",
      outputTokens: "~310",
    },
    scenarios: [
      { name: "accuracy",          isExtra: false,
        desc: "Per-format accuracy gate (each format has its own threshold)." },
      { name: "offline (×5 formats)", isExtra: false,
        desc: "Offline throughput at each precision. Quality efficiency = throughput × accuracy.",
        metric: { key: "quant_quality_eff", label: "quality efficiency", direction: "desc", unit: "" } },
      { name: "online",            isExtra: true,
        desc: "Online QPS sweep per format. Extra: 5 formats × QPS levels is expensive." },
      { name: "sustained",         isExtra: true,
        desc: "15 min sustained load per format." },
    ],
  },
  suite_D: {
    letter: "D",
    title: "Long-context inference",
    tagline: "28K-token prefill, compute-bound regime.",
    description:
      "Compute-bound prefill. ~28K-token prompts push arithmetic intensity past the roofline knee, so chips with more raw FLOPS pull ahead of bandwidth-rich ones. The output cap (256 tokens) keeps decode short on purpose; this suite isolates the prefill side and is where Suite A's bandwidth-bound rankings begin to invert.",
    primary: { key: "offline_throughput",     label: "tokens/sec",      direction: "desc", unit: "tokens/sec" },
    workload: {
      model: "meta-llama/Llama-3.1-8B-Instruct",
      chips: "1",
      precision: "BF16; max_model_len 30,208",
      dataset: "sharegpt_longctx_v1",
      inputTokens: "~28K",
      outputTokens: "≤256",
    },
    scenarios: [
      { name: "accuracy",    isExtra: false,
        desc: "MMLU gate against the 8B Llama-3.1 baseline." },
      { name: "offline",     isExtra: false,
        desc: "Offline throughput at ~28K input tokens. Prefill-bound, tests raw FLOPS.",
        metric: { key: "offline_throughput", label: "tokens/sec", direction: "desc", unit: "tokens/sec" } },
      { name: "interactive", isExtra: true,
        desc: "Long-context TTFT (~11 s per request at 28K). p90 is primary." },
      { name: "online",      isExtra: true,
        desc: "Sub-QPS levels (0.5 / 1 / 2). Rate-bound at long context." },
      { name: "sustained",   isExtra: true,
        desc: "30 min sustained at concurrency 8. Throttle ratio is the headline." },
      { name: "speculative", isExtra: true,
        desc: "Long-context offline with 1B draft model. Prefill-bound speculative." },
    ],
  },
  suite_E: {
    letter: "E",
    title: "Multi-chip scaling efficiency",
    tagline: "How well does 8B throughput scale to 2 / 4 / 8 chips?",
    description:
      "The Amdahl penalty in numbers. The same 8B model runs at 1×, 2×, and (optionally) 4× / 8× chip counts; the headline metric is 2× scaling efficiency = T_2× / (2 · T_1×). Reveals NVLink / Infinity Fabric / PCIe ceilings, and exposes flagships whose per-chip throughput grew faster than the interconnect did.",
    primary: { key: "scaling_efficiency_2x",  label: "2× scaling efficiency", direction: "desc", unit: "%", scale: 100, decimals: 1 },
    workload: {
      model: "meta-llama/Meta-Llama-3-8B-Instruct",
      chips: "1× / 2× required; 4× / 8× optional",
      precision: "BF16",
      dataset: "sharegpt_standard_v1",
      inputTokens: "~280",
      outputTokens: "~310",
    },
    scenarios: [
      { name: "offline (1× / 2×)", isExtra: false,
        desc: "Two-chip scaling efficiency vs single chip. Required for a valid submission.",
        metric: { key: "scaling_efficiency_2x", label: "2× scaling efficiency", direction: "desc", unit: "%", scale: 100, decimals: 1 } },
      { name: "offline (4×)",      isExtra: false,
        desc: "Four-chip scaling efficiency. Optional but commonly reported.",
        metric: { key: "scaling_efficiency_4x", label: "4× scaling efficiency", direction: "desc", unit: "%", scale: 100, decimals: 1 } },
      { name: "offline (8×)",      isExtra: false,
        desc: "Eight-chip scaling. Communication overhead is the binding constraint here." },
    ],
  },
  suite_F: {
    letter: "F",
    title: "Edge / consumer hardware",
    tagline: "Small models on single-GPU edge hardware.",
    description:
      "The pure-bandwidth lower bound. Qwen2.5-0.5B with ~95-token prompts strips away residual compute interference and short-circuits prefill, exposing raw HBM headroom and software overhead. Commodity GPUs (RTX 4090, A6000) tend to be most competitive per dollar here, and the suite doubles as a regression check for low-VRAM deployments.",
    primary: { key: "offline_throughput",     label: "tokens/sec",      direction: "desc", unit: "tokens/sec" },
    workload: {
      model: "Qwen/Qwen2.5-0.5B-Instruct",
      chips: "1 (≥4 GB VRAM)",
      precision: "BF16",
      dataset: "sharegpt_edge_v1",
      inputTokens: "~95",
      outputTokens: "~150",
    },
    scenarios: [
      { name: "accuracy",    isExtra: false,
        desc: "MMLU gate against the 0.5B baseline." },
      { name: "offline",     isExtra: false,
        desc: "Offline throughput on the edge dataset (~95 tok prompts).",
        metric: { key: "offline_throughput", label: "tokens/sec", direction: "desc", unit: "tokens/sec" } },
      { name: "online",      isExtra: false,
        desc: "Max QPS at the standard 500 ms p99 TTFT SLA.",
        metric: { key: "online_max_qps", label: "queries/sec", direction: "desc", unit: "queries/sec" } },
      { name: "interactive", isExtra: false,
        desc: "Single-stream TTFT on consumer hardware.",
        metric: { key: "interactive_ttft_p99", label: "TTFT p99", direction: "asc", unit: "ms", decimals: 0 } },
      { name: "sustained",   isExtra: true,
        desc: "15 min sustained load (shorter than datacenter suites)." },
    ],
  },
  suite_G: {
    letter: "G",
    title: "Mixture-of-Experts (MoE)",
    tagline: "Sparse routing; bandwidth-bound multi-chip serving.",
    description:
      "Sparse activation. Mixtral 8×7B activates only 2 of 8 experts per token, which keeps arithmetic intensity below dense 8B inference even at multi-chip scale. Chips with high aggregate HBM bandwidth (HBM3e generation) pay off here; pure-FLOPS advantages from compute-bound suites don't translate.",
    primary: { key: "sustained_throughput",   label: "tokens/sec",      direction: "desc", unit: "tokens/sec" },
    workload: {
      model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
      chips: "≥2 (auto)",
      precision: "BF16",
      dataset: "sharegpt_standard_v1",
      inputTokens: "~280",
      outputTokens: "~310",
    },
    scenarios: [
      { name: "accuracy",    isExtra: false,
        desc: "MMLU gate against the Mixtral baseline." },
      { name: "offline",     isExtra: false,
        desc: "Aggregate MoE throughput. Only 2 of 8 experts activate per token.",
        metric: { key: "offline_throughput", label: "tokens/sec", direction: "desc", unit: "tokens/sec" } },
      { name: "online",      isExtra: false,
        desc: "Max QPS under the 500 ms p99 TTFT SLA on MoE serving.",
        metric: { key: "online_max_qps", label: "queries/sec", direction: "desc", unit: "queries/sec" } },
      { name: "interactive", isExtra: true,
        desc: "Single-stream TTFT on MoE inference." },
      { name: "sustained",   isExtra: true,
        desc: "30 min sustained MoE load. Several chips show thermal onset on this suite.",
        metric: { key: "sustained_throughput", label: "sustained throughput", direction: "desc", unit: "tokens/sec" } },
    ],
  },
};

// House style: headline-style Title Case for all suite titles so they
// look correct everywhere they surface (home cards, rankings hero,
// rankings suite pills, compare hero, compare suite pills, modal
// header, suites explainer).  The source strings above stay in
// natural sentence case for readability; we transform once at module
// load so consumers don't have to remember to call toTitleCase().
for (const meta of Object.values(SUITE_META)) {
  meta.title = toTitleCase(meta.title);
}

// Single source of truth for "render the primary metric for this suite".
// Used by home, rankings, chip-detail, compare — keeps unit / scale rules
// in one place.  Returns "—" for null / missing values.
export function formatPrimary(value, suiteId) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const meta = SUITE_META[suiteId];
  if (!meta) return String(value);
  const p = meta.primary;
  const scaled = Number(value) * (p.scale || 1);
  const decimals = p.decimals !== undefined
    ? p.decimals
    : (Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2);
  const num = scaled.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return p.unit ? `${num} ${p.unit}` : num;
}

export const SUITE_ORDER = ["suite_A", "suite_B", "suite_C", "suite_D", "suite_E", "suite_F", "suite_G"];

// ── Vendor metadata (single source of truth) ────────────────────────
//
// Adding a new vendor used to require touching ~70 lines across 6 CSS
// files (one `[data-vendor="X"] { --vendor-color: var(--v-x); }` per
// scoped selector).  Now everything lives here:
//
//   • VENDOR_COLORS — known brand colour per vendor name (1 entry per
//                     vendor; key matches the `vendor` field on rows).
//   • vendorColor(name) — returns the brand colour, or a deterministic
//                         fallback from FALLBACK_PALETTE when the
//                         vendor isn't in the table yet.
//   • injectVendorStyles() — reads every vendor seen in the loaded
//                            dataset and writes one `[data-vendor="X"]
//                            { --vendor-color: ... }` rule into a
//                            singleton `<style>` tag.  Components keep
//                            consuming `var(--vendor-color)` like before.
//
// Net result: a brand-new vendor gets a stable colour out of the box.
// To pin its actual brand colour, add one entry below — that's it.
export const VENDOR_COLORS = {
  "NVIDIA":        "#76b900",
  "AMD":           "#ed1c24",
  "Apple":         "#a1a1aa",
  "Google":        "#4285f4",
  "Huawei":        "#ff4d4d",
  "Moore Threads": "#c084fc",
  "Intel":         "#0071c5",
};

// Spread across the colour wheel so two unknown vendors picked up in
// the same dataset are unlikely to collide.  Tuned for legibility on
// both light and dark backgrounds.
const FALLBACK_PALETTE = [
  "#5e9bff", "#ffa94d", "#56d364", "#f87171",
  "#a78bfa", "#2dd4bf", "#fbbf24", "#fb7185", "#22d3ee",
];

export function vendorColor(name) {
  if (!name) return "#888780";
  if (VENDOR_COLORS[name]) return VENDOR_COLORS[name];
  // Deterministic per-name pick: same vendor name always lands on the
  // same fallback so the colour doesn't shuffle between page loads.
  let h = 0;
  const s = String(name);
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return FALLBACK_PALETTE[Math.abs(h) % FALLBACK_PALETTE.length];
}

// Display order for vendor filter pills, derived from VENDOR_COLORS in
// declaration order so adding a vendor to the colour table is the only
// step needed to give it a stable rank in the UI.  Vendors that show
// up in the data without a colour entry are appended in alphabetical
// order at the end.
export const VENDOR_ORDER = Object.keys(VENDOR_COLORS);

function _cssEscapeAttr(v) {
  // CSS attribute selectors use double-quoted strings; double quotes
  // and backslashes need escaping.  Vendor names in the wild are very
  // unlikely to contain either, but better to be robust than to write
  // unparseable CSS the moment a vendor like `Foo "Bar"` shows up.
  return String(v).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function injectVendorStyles(vendors) {
  // Quietly no-op when the DOM isn't available — node tests import
  // data.js without standing up the full document/window globals and
  // we don't want them to blow up.
  if (typeof document === "undefined" || !document.head) return;
  const css = Array.from(vendors)
    .filter(Boolean)
    .map((v) => `[data-vendor="${_cssEscapeAttr(v)}"]{--vendor-color:${vendorColor(v)};}`)
    .join("\n");
  let style = document.getElementById("__vendor-color-styles");
  if (!style) {
    style = document.createElement("style");
    style.id = "__vendor-color-styles";
    document.head.appendChild(style);
  }
  style.textContent = css;
}

// Per-suite column / metric specs used by the rankings table and the
// compare view.  Decoupled from SUITE_META.scenarios because the UI
// columns are a curated subset (some scenarios are extras, some have
// no numeric metric at all), and the column ordering matters.
//
// First entry is the primary metric (default sort + featured row).
// `scale` multiplies the raw value at display (e.g. 0.945 -> 94.5 %).
// `direction` selects asc / desc default sort and informs compare bars.
// `textual` marks string-valued columns (e.g. quant_best_precision).
export const SUITE_COLUMNS = {
  suite_A: [
    { key: "offline_throughput",   label: "Offline",        unit: "tok/s", direction: "desc", primary: true },
    { key: "online_max_qps",       label: "Online QPS",     unit: "qps",   direction: "desc" },
    { key: "interactive_ttft_p99", label: "TTFT p99",       unit: "ms",    direction: "asc", decimals: 0 },
  ],
  suite_B: [
    { key: "offline_throughput",   label: "Offline",        unit: "tok/s", direction: "desc", primary: true },
    { key: "online_max_qps",       label: "Online QPS",     unit: "qps",   direction: "desc" },
  ],
  suite_C: [
    { key: "quant_quality_eff",    label: "Quality eff.",   unit: "",      direction: "desc", primary: true, decimals: 2 },
    { key: "quant_best_throughput", label: "Best tok/s",    unit: "tok/s", direction: "desc" },
    { key: "quant_best_precision", label: "Best format",    unit: "",      direction: "desc", textual: true },
  ],
  suite_D: [
    { key: "offline_throughput",   label: "Offline",        unit: "tok/s", direction: "desc", primary: true },
    { key: "interactive_ttft_p99", label: "TTFT p99",       unit: "ms",    direction: "asc", decimals: 0 },
  ],
  suite_E: [
    { key: "scaling_efficiency_2x", label: "2× eff.",       unit: "%",     direction: "desc", primary: true, scale: 100, decimals: 1 },
    { key: "scaling_efficiency_4x", label: "4× eff.",       unit: "%",     direction: "desc", scale: 100, decimals: 1 },
  ],
  suite_F: [
    { key: "offline_throughput",   label: "Offline",        unit: "tok/s", direction: "desc", primary: true },
    { key: "online_max_qps",       label: "Online QPS",     unit: "qps",   direction: "desc" },
    { key: "interactive_ttft_p99", label: "TTFT p99",       unit: "ms",    direction: "asc", decimals: 0 },
  ],
  suite_G: [
    { key: "sustained_throughput", label: "Sustained",      unit: "tok/s", direction: "desc", primary: true },
    { key: "offline_throughput",   label: "Offline",        unit: "tok/s", direction: "desc" },
  ],
};

// Format a numeric value against a column spec.  Returns null for
// missing values so callers can render the "-" themselves.
export function formatMetric(value, col) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  if (col.textual) return String(value);
  const scaled = Number(value) * (col.scale || 1);
  const decimals = col.decimals !== undefined
    ? col.decimals
    : (Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2);
  return scaled.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

let _rows = null;
let _byChip = null;
let _byRunId = null;
let _bySuite = null;
let _ready = false;

export function ready() { return _ready; }

export function rows() {
  if (!_rows) throw new Error("data.js: rows() called before init()");
  return _rows;
}

export function init() {
  if (_ready) return;
  mergeSuiteSpecs();

  const data = (typeof window !== "undefined" && Array.isArray(window.LEADERBOARD_DATA))
    ? window.LEADERBOARD_DATA
    : [];
  // Attach derived fields so views don't recompute.
  for (const r of data) {
    r._chip_slug = chipSlug(r);
    r._chip_label = r.chip_count && r.chip_count > 1
      ? `${r.chip} ×${r.chip_count}`
      : r.chip;
  }
  _rows = data;
  _bySuite = groupBy(data, (r) => r.suite);
  _byChip  = groupBy(data, (r) => r._chip_slug);
  _byRunId = new Map(data.filter((r) => r.run_id).map((r) => [r.run_id, r]));

  // Resolve vendor colours for every vendor seen in the loaded data
  // and inject them as CSS custom properties.  Components consume
  // `var(--vendor-color)` and stay agnostic to the colour table.
  const vendors = new Set(data.map((r) => r.vendor).filter(Boolean));
  injectVendorStyles(vendors);

  _ready = true;
}

// Merge canonical suite spec from suites/suite_X/suite.json (baked into
// the generated leaderboard.js as window.SUITE_SPECS) into SUITE_META.
//
// Only factual fields (model, dataset, precision baseline, prompt-token
// p50, default/extra scenario split) are merged in; editorial fields
// (title, tagline, description, primary metric label, chip-count copy)
// stay hardcoded since they aren't part of the suite contract.
//
// The merge is conservative: a spec value only overrides the existing
// editorial value when the existing value looks like a simple token
// (no comma, no semicolon).  This preserves intentionally composite
// strings — e.g. suite_C's "BF16, FP8, W8A8, W8A16, W4A16" and
// suite_D's "BF16; max_model_len 30,208".  If maintainers want those
// suites to auto-track suite.json too, they should split the composite
// into separate fields in SUITE_META.
function mergeSuiteSpecs() {
  const specs = (typeof window !== "undefined" && window.SUITE_SPECS) || {};
  for (const [sid, spec] of Object.entries(specs)) {
    const meta = SUITE_META[sid];
    if (!meta) continue;
    const wl = meta.workload || (meta.workload = {});

    if (spec.model_id) wl.model = spec.model_id;
    if (spec.dataset)  wl.dataset = spec.dataset;
    if (spec.precision_required && _isSimpleToken(wl.precision)) {
      wl.precision = spec.precision_required;
    }

    const inT  = _fmtTokenCount(spec.input_tokens_p50);
    const outT = _fmtTokenCount(spec.output_tokens_p50);
    if (inT  && _isTokenCountToken(wl.inputTokens))  wl.inputTokens  = inT;
    if (outT && _isTokenCountToken(wl.outputTokens)) wl.outputTokens = outT;

    // Scenario name in SUITE_META may be qualified ("offline (×5 formats)",
    // "offline (1× / 2×)"); match by leading bare word.
    const extras   = new Set(spec.scenarios_extra   || []);
    const defaults = new Set(spec.scenarios_default || []);
    for (const s of meta.scenarios) {
      const base = s.name.split(" ")[0].trim();
      if (extras.has(base))   s.isExtra = true;
      if (defaults.has(base)) s.isExtra = false;
    }
  }
}

function _isSimpleToken(v) {
  // True for "BF16", "FP16", "FP8", etc.; false for "BF16, FP8, ..."
  // (composite list) and "BF16; max_model_len 30,208" (composite line).
  if (!v) return true;
  const s = String(v);
  return !s.includes(",") && !s.includes(";");
}

function _isTokenCountToken(v) {
  // Auto-format only when the existing value is a plain "~280" / "~28K"
  // style.  Preserves anything more elaborate the editorial layer set.
  if (!v) return true;
  return /^~[\d.]+[KM]?$/.test(String(v));
}

function _fmtTokenCount(n) {
  if (n === null || n === undefined) return null;
  const v = Number(n);
  if (!Number.isFinite(v)) return null;
  if (v >= 10000) return `~${Math.round(v / 1000)}K`;
  if (v >= 1000)  return `~${(v / 1000).toFixed(1)}K`;
  return `~${Math.round(v)}`;
}

// Returns rows in a given suite, sorted by suite primary metric.
export function rowsForSuite(suiteId) {
  if (!_ready) init();
  const suite = SUITE_META[suiteId];
  if (!suite) return [];
  const raw = _bySuite.get(suiteId) || [];
  const direction = suite.primary.direction;
  const key = suite.primary.key;

  // Filter to rows that actually have the primary metric.
  const valid = raw.filter((r) => {
    const v = r[key];
    return v !== null && v !== undefined && !Number.isNaN(v);
  });

  const cmp = direction === "asc"
    ? (a, b) => a[key] - b[key]
    : (a, b) => b[key] - a[key];

  return [...valid].sort(cmp);
}

// One row per chip for a suite — picks best by primary metric.
// Used for "is this chip the leader in this suite?" comparisons.
export function bestPerChipForSuite(suiteId) {
  const all = rowsForSuite(suiteId);
  const seen = new Map();
  for (const r of all) {
    if (!seen.has(r._chip_slug)) seen.set(r._chip_slug, r);
  }
  return Array.from(seen.values());
}

// Rows for a given chip slug, grouped by suite (so a chip detail page
// can show each suite as a tile).  Each entry has one "best" row.
export function rowsForChip(slug) {
  if (!_ready) init();
  return _byChip.get(slug) || [];
}

// Chip-count variants this chip has been deployed at, sorted asc.
// Returns e.g. [1, 4, 8] for a chip with single-card, 4-card, and
// 8-card submissions.  chip-detail uses this in the hero facts row
// and the suite-card "won at ×N" badge.
export function chipCountsForChip(slug) {
  const rs = rowsForChip(slug);
  const counts = new Set();
  for (const r of rs) counts.add(r.chip_count || 1);
  return Array.from(counts).sort((a, b) => a - b);
}

// Pick a representative run_id for a given chip slug — used by the
// chip-cloud "quick add" affordance on Home and Compare so that a single
// click on a chip seeds the compare basket with the freshest run for
// that chip-count variant.  Returns null when no rows exist (e.g. the
// chip was removed from the dataset since a shared link was created).
export function representativeRunForChip(slug) {
  const rs = rowsForChip(slug);
  if (!rs.length) return null;
  const best = rs.reduce((a, b) => (String(b.date) > String(a.date) ? b : a));
  return best.run_id || best.submission || null;
}

// Find a row by its run_id (the compare-basket primary key).
// Returns null if the run no longer exists in the dataset (e.g. the
// user reloaded a shared URL after the submission was withdrawn).
export function rowByRunId(runId) {
  if (!_ready) init();
  return _byRunId.get(runId) || null;
}

// Resolve a basket run_id to the most appropriate row to display for a
// given suite:
//   • exact run in the same suite     → return it
//   • same chip + count + framework + version → best by suite primary
//   • same chip + count + framework             → best by suite primary
//   • otherwise                                  → null (no data in suite)
//
// This is what makes the compare page meaningful across suites: the
// user picks a specific run from Rankings, and when they switch suite
// we resurface that same hardware/software configuration's best run
// in the new suite rather than dropping the chip from the basket.
// Resolve a basket entry to its "matching" row in another suite.  The
// match cascades through ever-laxer constraints so suites that fix
// chip_count (e.g. suite_B / suite_G are ×8) still surface a relevant
// row when the user originally picked a single-chip variant:
//
//   1. exact run_id in the target suite (rare; same submission ran the
//      whole suite family)
//   2. chip + chip_count + framework + framework_version
//   3. chip + chip_count + framework (looser framework_version)
//   4. chip + framework (loosen chip_count; this is how an H200×1 pick
//      maps onto suite_B's H200×8 row, which is the *only* H200 row in
//      that suite anyway)
//   5. chip (loosen framework — last-resort cross-framework fallback)
//
// Within whichever pool wins we pick the row with the best value on
// the target suite's primary metric.
export function bestRowForRunInSuite(runId, suiteId) {
  const seed = rowByRunId(runId);
  if (!seed) return null;
  if (seed.suite === suiteId) return seed;
  const inSuite = _rows.filter((r) => r.suite === suiteId);
  if (inSuite.length === 0) return null;

  const cascade = [
    (r) => r.chip === seed.chip
        && (r.chip_count || 1) === (seed.chip_count || 1)
        && r.framework === seed.framework
        && r.framework_version === seed.framework_version,
    (r) => r.chip === seed.chip
        && (r.chip_count || 1) === (seed.chip_count || 1)
        && r.framework === seed.framework,
    (r) => r.chip === seed.chip
        && r.framework === seed.framework,
    (r) => r.chip === seed.chip,
  ];

  let pool = null;
  for (const pred of cascade) {
    const candidates = inSuite.filter(pred);
    if (candidates.length > 0) { pool = candidates; break; }
  }
  if (!pool) return null;

  const meta = SUITE_META[suiteId];
  if (!meta) return pool[0];
  const pk = meta.primary.key;
  const valid = pool.filter((r) =>
    r[pk] !== null && r[pk] !== undefined && !Number.isNaN(r[pk])
  );
  if (valid.length === 0) return pool[0];
  return meta.primary.direction === "asc"
    ? valid.reduce((a, b) => (a[pk] <= b[pk] ? a : b))
    : valid.reduce((a, b) => (a[pk] >= b[pk] ? a : b));
}

// Best per suite for a chip.  Returns Map<suite_id, best_row>.
export function bestPerSuiteForChip(slug) {
  const rs = rowsForChip(slug);
  const out = new Map();
  for (const suiteId of SUITE_ORDER) {
    const inSuite = rs.filter((r) => r.suite === suiteId);
    if (inSuite.length === 0) continue;
    const meta = SUITE_META[suiteId];
    const key = meta.primary.key;
    const direction = meta.primary.direction;
    const valid = inSuite.filter((r) => r[key] !== null && r[key] !== undefined);
    if (valid.length === 0) {
      // Fall back to first row even without primary, so user still sees a tile.
      out.set(suiteId, inSuite[0]);
      continue;
    }
    const best = direction === "asc"
      ? valid.reduce((a, b) => (a[key] <= b[key] ? a : b))
      : valid.reduce((a, b) => (a[key] >= b[key] ? a : b));
    out.set(suiteId, best);
  }
  return out;
}

// Unique chips list (representative row per chip slug).
export function uniqueChips() {
  if (!_ready) init();
  return Array.from(_byChip.values()).map((rs) => rs[0]);
}

// High-level stats for hero strip.
export function summary() {
  if (!_ready) init();
  const chips = new Set(_rows.map((r) => r._chip_slug));
  const vendors = new Set(_rows.map((r) => r.vendor));
  const suites = new Set(_rows.map((r) => r.suite));
  const verified = _rows.filter((r) => r.tier === "verified").length;
  return {
    total: _rows.length,
    verified,
    chips: chips.size,
    vendors: vendors.size,
    suites: suites.size,
  };
}

// Last N submissions, sorted by date desc.
export function recent(limit = 6) {
  if (!_ready) init();
  return [..._rows]
    .filter((r) => r.date)
    .sort((a, b) => String(b.date).localeCompare(String(a.date)))
    .slice(0, limit);
}

// Rank a *chip* within a suite leaderboard (one entry per chip slug,
// scored by each chip's best primary metric).  Used by chip-detail to
// surface "ranked #N of M" badges per suite — far more useful than the
// raw row rank, which would penalise chips that submit many variants.
export function rankChipInSuite(slug, suiteId) {
  const board = bestPerChipForSuite(suiteId);
  if (!board.length) return null;
  const idx = board.findIndex((r) => r._chip_slug === slug);
  if (idx < 0) return null;
  return { rank: idx + 1, total: board.length };
}

// Find chips with the most overlapping suite coverage with `slug`.
// Returns a small array sorted by (shared suite count desc, same-vendor
// preferred, total run count desc) — i.e. "you might also care about
// these because they compete on the same workloads".  Chips with no
// shared suite are excluded; the source chip itself is excluded.
export function similarChipsTo(slug, { limit = 5 } = {}) {
  if (!_ready) init();
  const myRows = rowsForChip(slug);
  if (!myRows.length) return [];
  const mySuites = new Set(myRows.map((r) => r.suite));
  const myVendor = myRows[0].vendor;

  const candidates = [];
  for (const [otherSlug, rows] of _byChip.entries()) {
    if (otherSlug === slug || !rows.length) continue;
    const shared = rows.filter((r) => mySuites.has(r.suite));
    if (shared.length === 0) continue;
    const otherSuites = new Set(shared.map((r) => r.suite));
    candidates.push({
      slug: otherSlug,
      sample: rows[0],
      // Peer cards link to a chip-detail page (which is now per-model,
      // not per-fan-out) so the displayed label is the bare chip name
      // — `_chip_label` would arbitrarily pick whichever variant came
      // first in `rows`, which would read as "RTX 4090D ×4" sometimes
      // and "RTX 4090D" other times depending on insertion order.
      label: rows[0].chip,
      vendor: rows[0].vendor,
      sharedSuites: Array.from(otherSuites),
      sameVendor: rows[0].vendor === myVendor,
      totalRuns: rows.length,
    });
  }

  candidates.sort((a, b) => {
    if (b.sharedSuites.length !== a.sharedSuites.length) {
      return b.sharedSuites.length - a.sharedSuites.length;
    }
    // Same-vendor first when overlap is tied — keeps recommendations
    // intra-vendor for vendor lock-in shoppers without hard-filtering
    // cross-vendor matches off the strip.
    if (a.sameVendor !== b.sameVendor) return a.sameVendor ? -1 : 1;
    return b.totalRuns - a.totalRuns;
  });

  return candidates.slice(0, limit);
}

// Rank of a row within its suite (1-indexed).
export function rankWithinSuite(row) {
  const list = rowsForSuite(row.suite);
  const idx = list.findIndex((r) => r.submission === row.submission);
  return idx >= 0 ? { rank: idx + 1, total: list.length } : null;
}

// Suite-level facts derived from current data — model used,
// baseline precision, total submissions, distinct chips.  Used in
// the home suite-card header to give buyers immediate context
// without having to dive in.
export function suiteFacts(suiteId) {
  if (!_ready) init();
  const rows = _bySuite.get(suiteId) || [];
  if (rows.length === 0) {
    return { model: null, precision: null, submissions: 0, chips: 0 };
  }
  const model = mode(rows.map((r) => r.model).filter(Boolean));
  const precision = mode(rows.map((r) => r.precision).filter(Boolean));
  const chips = new Set(rows.map((r) => r._chip_slug)).size;
  return { model, precision, submissions: rows.length, chips };
}

// Suite leader — best chip in this suite by the suite's primary metric
// (one row per chip; the same chip can also appear at other chip counts
// elsewhere on the rankings page).  Returns null if the suite has no
// numeric data yet.
export function suiteLeader(suiteId) {
  const rows = bestPerChipForSuite(suiteId);
  return rows.length ? rows[0] : null;
}

// Find the best row in a suite by an arbitrary metric key + direction.
// Used by the suites page to surface a leader per scenario, not just
// for the suite-level primary metric.  Returns null if no rows have a
// numeric value for that key.
export function bestRowByMetric(suiteId, key, direction) {
  if (!_ready) init();
  const rows = (_bySuite.get(suiteId) || [])
    .filter((r) => r[key] !== null && r[key] !== undefined && Number.isFinite(r[key]));
  if (rows.length === 0) return null;
  return direction === "asc"
    ? rows.reduce((a, b) => (a[key] <= b[key] ? a : b))
    : rows.reduce((a, b) => (a[key] >= b[key] ? a : b));
}

// Chip cloud data — one entry per chip (variants like ×1 / ×4 / ×8
// are aggregated under the base chip name; the linked detail row
// picks the most-submitted variant so the link still resolves).
// Each entry gets a size bucket (sm/md/lg/xl) for the home cloud.
//
// Ordering is intentionally NOT by submission count: tiles are
// shuffled by a deterministic hash so the resulting layout reads
// like a real cloud rather than a sorted bar chart.
export function chipCloudData() {
  if (!_ready) init();
  // Aggregate by base chip name across chip-count variants.
  const byBase = new Map();
  for (const r of _rows) {
    const base = r.chip;
    if (!base) continue;
    let agg = byBase.get(base);
    if (!agg) {
      agg = {
        label: base,
        vendor: r.vendor,
        submissions: 0,
        _bestSlug: null,
        _bestVariantSubs: -1,
        _variantSubs: new Map(),
        _suites: new Set(),
      };
      byBase.set(base, agg);
    }
    agg.submissions += 1;
    agg._suites.add(r.suite);
    const cur = (agg._variantSubs.get(r._chip_slug) || 0) + 1;
    agg._variantSubs.set(r._chip_slug, cur);
    if (cur > agg._bestVariantSubs) {
      agg._bestVariantSubs = cur;
      agg._bestSlug = r._chip_slug;
    }
  }
  const out = [];
  for (const agg of byBase.values()) {
    out.push({
      slug: agg._bestSlug,
      label: agg.label,
      vendor: agg.vendor,
      submissions: agg.submissions,
      variants: agg._variantSubs.size,
      suites: Array.from(agg._suites).sort().map((s) =>
        (SUITE_META[s] && SUITE_META[s].letter) || null).filter(Boolean),
    });
  }
  // Bucket by submission count — thresholds tuned to spread current
  // distribution (max ~12 subs) into 4 buckets.
  for (const c of out) {
    c.size = c.submissions >= 8 ? "xl"
           : c.submissions >= 4 ? "lg"
           : c.submissions >= 2 ? "md"
                                : "sm";
  }
  // Deterministic shuffle: hash of label seeds a stable but
  // chaotic-looking order.  Stable across reloads.
  for (const c of out) c._h = hashString(c.label);
  out.sort((a, b) => a._h - b._h);
  for (const c of out) delete c._h;
  return out;
}

function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return h;
}

function mode(arr) {
  if (!arr || arr.length === 0) return null;
  const c = new Map();
  for (const x of arr) c.set(x, (c.get(x) || 0) + 1);
  let best = null, bestC = -1;
  for (const [k, n] of c.entries()) if (n > bestC) { best = k; bestC = n; }
  return best;
}
