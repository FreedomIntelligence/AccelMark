// data.js — single source of truth for leaderboard rows.
//
// The legacy generator emits leaderboard.js which sets `window.LEADERBOARD_DATA`
// as a side-effect when included via <script>. We import it through that route
// to keep the generator output unchanged (zero-build pipeline).
//
// All views consume LeaderboardData from this module.  Do not access
// window.LEADERBOARD_DATA directly anywhere else.

import { groupBy, maxBy, chipSlug } from "./utils.js";

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

// Vendor display order — controls filter pill order on rankings.
export const VENDOR_ORDER = ["NVIDIA", "AMD", "Apple", "Google", "Huawei", "Moore Threads", "Intel"];

let _rows = null;
let _byChip = null;
let _bySuite = null;
let _ready = false;

export function ready() { return _ready; }

export function rows() {
  if (!_rows) throw new Error("data.js: rows() called before init()");
  return _rows;
}

export function init() {
  if (_ready) return;
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
  _byChip = groupBy(data, (r) => r._chip_slug);
  _ready = true;
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
