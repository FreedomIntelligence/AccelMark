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
export const SUITE_META = {
  suite_A: {
    letter: "A",
    title: "Single-chip throughput",
    tagline: "How fast can one accelerator serve an 8B model?",
    primary: { key: "offline_throughput",     label: "tok/s",           direction: "desc", unit: "tok/s" },
  },
  suite_B: {
    letter: "B",
    title: "Multi-chip throughput",
    tagline: "Aggregate throughput across 8 chips.",
    primary: { key: "offline_throughput",     label: "tok/s",           direction: "desc", unit: "tok/s" },
  },
  suite_C: {
    letter: "C",
    title: "Quantization efficiency",
    tagline: "Quality-adjusted throughput across quantization formats.",
    primary: { key: "quant_quality_eff",      label: "quality eff.",    direction: "desc", unit: "" },
  },
  suite_D: {
    letter: "D",
    title: "Interactive latency",
    tagline: "Single-stream first-token & per-token latency.",
    primary: { key: "interactive_ttft_p99",   label: "TTFT p99",        direction: "asc",  unit: "ms",  decimals: 0 },
  },
  suite_E: {
    letter: "E",
    title: "Scaling efficiency",
    tagline: "How well does throughput scale to 2 / 4 / 8 chips?",
    primary: { key: "scaling_efficiency_2x",  label: "2× efficiency",   direction: "desc", unit: "%",   scale: 100, decimals: 1 },
  },
  suite_F: {
    letter: "F",
    title: "Edge / low-power",
    tagline: "Smaller models on commodity & edge hardware.",
    primary: { key: "offline_throughput",     label: "tok/s",           direction: "desc", unit: "tok/s" },
  },
  suite_G: {
    letter: "G",
    title: "Sustained load",
    tagline: "Throughput stability under prolonged load (30 min).",
    primary: { key: "sustained_throughput",   label: "tok/s sustained", direction: "desc", unit: "tok/s" },
  },
};

// Single source of truth for "render the primary metric for this suite".
// Used by home, rankings, chip-detail, compare — keeps unit / scale rules
// in one place.  Returns "—" for null / missing values.
export function formatPrimary(value, suiteId) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
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
