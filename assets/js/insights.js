// insights.js — live "key findings" computed from the leaderboard rows.
//
// The home page used to hard-code the headline numbers from the paper
// (e.g. "A100 vs 5090: −29% / +29%").  Those drift as soon as new runs
// land, and they silently disagree with the tables a visitor clicks
// through to.  Everything here is recomputed from `window.LEADERBOARD_DATA`
// on every page load, so a finding card always matches the evidence it
// links to — and simply disappears when the data can no longer support it.
//
// Pure module: no DOM, no globals.  Each finder takes the flat row array
// and returns either `null` (not enough data) or a plain object that the
// view layer formats.  Unit-tested in test/insights.test.mjs.

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

// Runner tiers: only "reference" runners are the apples-to-apples
// baseline.  Optimized / experimental runners stay on the board but are
// excluded from cross-chip findings so a tuned fork can't manufacture a
// "reversal" that the reference stack doesn't show.
const isReference = (r) => (r.runner_tier || "reference") === "reference";
const isSingleChip = (r) => (r.chip_count || 1) === 1;
const isBaselinePrecision = (r) => !r.precision || r.precision === "BF16" || r.precision === "FP16";

function chipKey(r) {
  const n = r.chip_count || 1;
  return n > 1 ? `${r.chip} ×${n}` : r.chip;
}

function suiteRows(rows, suiteId) {
  return rows.filter((r) => r.suite === suiteId && !r.flagged);
}

// Prefer independently-verified rows when there are enough of them to
// support a comparison; otherwise fall back to the full pool.
function preferVerified(pool, min = 3) {
  const v = pool.filter((r) => r.tier === "verified");
  return v.length >= min ? v : pool;
}

// Best row per chip (by `key`, higher is better).
function bestPerChip(pool, key) {
  const m = new Map();
  for (const r of pool) {
    if (!isNum(r[key])) continue;
    const k = chipKey(r);
    const cur = m.get(k);
    if (!cur || r[key] > cur[key]) m.set(k, r);
  }
  return m;
}

// Most common framework in a pool — used to hold the engine axis fixed.
function dominantFramework(pool) {
  const counts = new Map();
  for (const r of pool) {
    if (!r.framework) continue;
    counts.set(r.framework, (counts.get(r.framework) || 0) + 1);
  }
  let best = null, n = 0;
  for (const [fw, c] of counts) if (c > n) { best = fw; n = c; }
  return best;
}

function runRef(r) {
  return {
    chip: chipKey(r),
    vendor: r.vendor || "",
    framework: r.framework || "",
    run_id: r.run_id || r.submission || "",
    tier: r.tier || "",
  };
}

// ── 1. Offline leader ≠ SLA leader (Suite A) ─────────────────────────
export function findSlaGap(rows) {
  let pool = suiteRows(rows, "suite_A").filter(
    (r) => isReference(r) && isSingleChip(r) && isNum(r.offline_throughput) && isNum(r.online_max_qps)
  );
  pool = preferVerified(pool);
  if (pool.length < 2) return null;
  const offLeader = pool.reduce((a, b) => (b.offline_throughput > a.offline_throughput ? b : a));
  const qpsLeader = pool.reduce((a, b) =>
    b.online_max_qps > a.online_max_qps ||
    (b.online_max_qps === a.online_max_qps && b.offline_throughput > a.offline_throughput) ? b : a
  );
  if (chipKey(offLeader) === chipKey(qpsLeader)) return null;
  if (!(qpsLeader.online_max_qps > offLeader.online_max_qps)) return null;
  return {
    id: "sla_gap",
    suite: "suite_A",
    offline: { ...runRef(offLeader), throughput: offLeader.offline_throughput, qps: offLeader.online_max_qps },
    sla: { ...runRef(qpsLeader), throughput: qpsLeader.offline_throughput, qps: qpsLeader.online_max_qps },
  };
}

// ── 2. Ranking reversal between short chat (A) and long context (D) ──
export function findReversal(rows, { a = "suite_A", b = "suite_D" } = {}) {
  const base = (r) => isReference(r) && isSingleChip(r) && isBaselinePrecision(r) && isNum(r.offline_throughput);
  const poolA0 = suiteRows(rows, a).filter(base);
  const poolB0 = suiteRows(rows, b).filter(base);
  const fw = dominantFramework([...poolA0, ...poolB0]);
  if (!fw) return null;
  const pick = (p) => {
    const sameFw = p.filter((r) => r.framework === fw);
    return preferVerified(sameFw);
  };
  const A = bestPerChip(pick(poolA0), "offline_throughput");
  const B = bestPerChip(pick(poolB0), "offline_throughput");
  const chips = [...A.keys()].filter((k) => B.has(k));
  let best = null;
  for (const x of chips) {
    for (const y of chips) {
      if (x === y) continue;
      const rA = A.get(x).offline_throughput / A.get(y).offline_throughput; // x wins A
      const rB = B.get(y).offline_throughput / B.get(x).offline_throughput; // y wins B
      if (!(rA > 1.05 && rB > 1.05)) continue;
      const score = Math.min(rA, rB);
      if (!best || score > best.score) best = { x, y, score };
    }
  }
  if (!best) return null;
  const xa = A.get(best.x), ya = A.get(best.y), xb = B.get(best.x), yb = B.get(best.y);
  const xCost = xa.cost_efficiency_toks_per_dollar_hr;
  const yCost = ya.cost_efficiency_toks_per_dollar_hr;
  return {
    id: "reversal",
    suites: [a, b],
    framework: fw,
    x: { ...runRef(xa), a: xa.offline_throughput, b: xb.offline_throughput, run_b: xb.run_id || "", costA: isNum(xCost) ? xCost : null },
    y: { ...runRef(ya), a: ya.offline_throughput, b: yb.offline_throughput, run_b: yb.run_id || "", costA: isNum(yCost) ? yCost : null },
    gainA: xa.offline_throughput / ya.offline_throughput - 1,
    lossB: 1 - xb.offline_throughput / yb.offline_throughput,
  };
}

// ── 3. Same chip, different engine (Suite A) ─────────────────────────
export function findEngineGap(rows, suiteId = "suite_A") {
  const pool = suiteRows(rows, suiteId).filter(
    (r) => isReference(r) && isSingleChip(r) && isBaselinePrecision(r) && isNum(r.offline_throughput) && r.framework
  );
  const byChip = new Map();
  for (const r of pool) {
    const k = chipKey(r);
    if (!byChip.has(k)) byChip.set(k, new Map());
    const fws = byChip.get(k);
    const cur = fws.get(r.framework);
    if (!cur || r.offline_throughput > cur.offline_throughput) fws.set(r.framework, r);
  }
  let best = null;
  for (const [, fws] of byChip) {
    if (fws.size < 2) continue;
    const list = [...fws.values()].sort((p, q) => q.offline_throughput - p.offline_throughput);
    const hi = list[0], lo = list[list.length - 1];
    const gain = hi.offline_throughput / lo.offline_throughput - 1;
    if (!best || gain > best.gain) best = { hi, lo, gain };
  }
  if (!best || best.gain < 0.1) return null;
  return {
    id: "engine_gap",
    suite: suiteId,
    chip: chipKey(best.hi),
    vendor: best.hi.vendor || "",
    fast: { ...runRef(best.hi), throughput: best.hi.offline_throughput },
    slow: { ...runRef(best.lo), throughput: best.lo.offline_throughput },
    gain: best.gain,
  };
}

// ── 4. Scaling efficiency spread (Suite E) ───────────────────────────
export function findScalingSpread(rows) {
  let pool = suiteRows(rows, "suite_E").filter((r) => isReference(r) && isNum(r.scaling_efficiency_2x));
  const fw = dominantFramework(pool);
  if (fw) pool = pool.filter((r) => r.framework === fw);
  pool = preferVerified(pool);
  if (pool.length < 2) return null;
  const hi = pool.reduce((a, b) => (b.scaling_efficiency_2x > a.scaling_efficiency_2x ? b : a));
  const lo = pool.reduce((a, b) => (b.scaling_efficiency_2x < a.scaling_efficiency_2x ? b : a));
  if (hi === lo || hi.scaling_efficiency_2x - lo.scaling_efficiency_2x < 0.1) return null;
  return {
    id: "scaling",
    suite: "suite_E",
    framework: fw,
    best: { ...runRef(hi), chip: hi.chip, eff: hi.scaling_efficiency_2x },
    worst: { ...runRef(lo), chip: lo.chip, eff: lo.scaling_efficiency_2x },
  };
}

// ── 5. Fastest ≠ best value (Suite F, tokens per $/hr) ───────────────
export function findValueGap(rows, suiteId = "suite_F") {
  let pool = suiteRows(rows, suiteId).filter(
    (r) => isReference(r) && isSingleChip(r) && isNum(r.offline_throughput) && isNum(r.cost_efficiency_toks_per_dollar_hr)
  );
  const fw = dominantFramework(pool);
  if (fw) pool = pool.filter((r) => r.framework === fw);
  pool = preferVerified(pool);
  if (pool.length < 2) return null;
  const fastest = pool.reduce((a, b) => (b.offline_throughput > a.offline_throughput ? b : a));
  const value = pool.reduce((a, b) =>
    (b.cost_efficiency_toks_per_dollar_hr > a.cost_efficiency_toks_per_dollar_hr ? b : a));
  if (chipKey(fastest) === chipKey(value)) return null;
  const ratio = value.cost_efficiency_toks_per_dollar_hr / fastest.cost_efficiency_toks_per_dollar_hr;
  if (!(ratio > 1.2)) return null;
  return {
    id: "value",
    suite: suiteId,
    framework: fw,
    value: { ...runRef(value), throughput: value.offline_throughput, perDollar: value.cost_efficiency_toks_per_dollar_hr },
    fastest: { ...runRef(fastest), throughput: fastest.offline_throughput, perDollar: fastest.cost_efficiency_toks_per_dollar_hr },
    ratio,
  };
}

// ── 6. Quantization speed-up depends on the chip (Suite C) ───────────
export function findQuantSpread(rows) {
  let pool = suiteRows(rows, "suite_C").filter((r) =>
    isReference(r) && isSingleChip(r) &&
    isNum(r.quant_bf16_throughput) && r.quant_bf16_throughput > 0 &&
    isNum(r.quant_best_throughput) && r.quant_best_precision
  );
  const fw = dominantFramework(pool);
  if (fw) pool = pool.filter((r) => r.framework === fw);
  pool = preferVerified(pool);
  const withSpeed = pool.map((r) => ({ r, s: r.quant_best_throughput / r.quant_bf16_throughput }));
  if (withSpeed.length < 2) return null;
  const hi = withSpeed.reduce((a, b) => (b.s > a.s ? b : a));
  const lo = withSpeed.reduce((a, b) => (b.s < a.s ? b : a));
  if (hi.s - lo.s < 0.25) return null;
  return {
    id: "quant",
    suite: "suite_C",
    framework: fw,
    top: { ...runRef(hi.r), speedup: hi.s, format: hi.r.quant_best_precision },
    low: { ...runRef(lo.r), speedup: lo.s, format: lo.r.quant_best_precision },
  };
}

// All findings, in display order, with nulls dropped.
export function computeInsights(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  const finders = [findReversal, findSlaGap, findEngineGap, findScalingSpread, findValueGap, findQuantSpread];
  const out = [];
  for (const f of finders) {
    try {
      const v = f(rows);
      if (v) out.push(v);
    } catch (e) {
      // A malformed row must never take the home page down.
      if (typeof console !== "undefined") console.warn("insight failed", f.name, e);
    }
  }
  return out;
}
