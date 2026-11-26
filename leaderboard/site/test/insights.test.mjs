// insights.test.mjs — unit tests for live key-findings computation.

import test from "node:test";
import assert from "node:assert/strict";

import {
  findReversal,
  findSlaGap,
  findEngineGap,
  findScalingSpread,
  findValueGap,
  findQuantSpread,
  computeInsights,
} from "../assets/js/insights.js";

const SAMPLE_ROWS = [
  // Suite A: 5090 beats A100 offline; H200 leads offline but loses SLA to H100; SGLang beats vLLM on H20-3e
  { suite: "suite_A", chip: "RTX 5090", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 3488, online_max_qps: 0, cost_efficiency_toks_per_dollar_hr: 4650, run_id: "r5090a" },
  { suite: "suite_A", chip: "A100-80GB", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 2701, online_max_qps: 5, cost_efficiency_toks_per_dollar_hr: 1800, run_id: "ra100a" },
  { suite: "suite_A", chip: "H200", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 5731, online_max_qps: 5, run_id: "rh200a" },
  { suite: "suite_A", chip: "H100", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 5128, online_max_qps: 25, run_id: "rh100a" },
  { suite: "suite_A", chip: "H20-3e", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 2298, online_max_qps: 5, run_id: "rh20v" },
  { suite: "suite_A", chip: "H20-3e", vendor: "NVIDIA", chip_count: 1, framework: "SGLang", precision: "BF16", tier: "community", runner_tier: "reference", offline_throughput: 4342, online_max_qps: 100, run_id: "rh20s" },
  // Experimental runner should NOT pollute cross-chip reference findings
  { suite: "suite_A", chip: "B200", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "experimental", offline_throughput: 14717, online_max_qps: 25, run_id: "rb200exp" },
  // Suite D: A100 beats 5090
  { suite: "suite_D", chip: "RTX 5090", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 54.3, run_id: "r5090d" },
  { suite: "suite_D", chip: "A100-80GB", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 70.2, run_id: "ra100d" },
  { suite: "suite_D", chip: "H100", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", precision: "BF16", tier: "verified", runner_tier: "reference", offline_throughput: 160.4, run_id: "rh100d" },
  // Suite E: 4090 D scales better than H200
  { suite: "suite_E", chip: "RTX 4090 D", vendor: "NVIDIA", chip_count: 4, framework: "vLLM", tier: "verified", runner_tier: "reference", scaling_efficiency_2x: 0.945, run_id: "r4090e" },
  { suite: "suite_E", chip: "H200", vendor: "NVIDIA", chip_count: 8, framework: "vLLM", tier: "verified", runner_tier: "reference", scaling_efficiency_2x: 0.602, run_id: "rh200e" },
  { suite: "suite_E", chip: "A100-40GB", vendor: "NVIDIA", chip_count: 8, framework: "vLLM", tier: "verified", runner_tier: "reference", scaling_efficiency_2x: 0.725, run_id: "ra100e" },
  // Suite F: Tesla T4 beats H100 on tok/$
  { suite: "suite_F", chip: "Tesla T4", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", tier: "verified", runner_tier: "reference", offline_throughput: 5126, cost_efficiency_toks_per_dollar_hr: 51256, run_id: "rt4f" },
  { suite: "suite_F", chip: "H100", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", tier: "verified", runner_tier: "reference", offline_throughput: 18911, cost_efficiency_toks_per_dollar_hr: 7595, run_id: "rh100f" },
  { suite: "suite_F", chip: "A100-80GB", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", tier: "verified", runner_tier: "reference", offline_throughput: 11200, cost_efficiency_toks_per_dollar_hr: 7467, run_id: "ra100f" },
  // Suite C: Quantization speedup spread
  { suite: "suite_C", chip: "RTX 3090", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", tier: "verified", runner_tier: "reference", quant_bf16_throughput: 900, quant_best_throughput: 2070, quant_best_precision: "W8A8", run_id: "r3090c" },
  { suite: "suite_C", chip: "H200", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", tier: "verified", runner_tier: "reference", quant_bf16_throughput: 5300, quant_best_throughput: 6150, quant_best_precision: "FP8", run_id: "rh200c" },
  { suite: "suite_C", chip: "H100", vendor: "NVIDIA", chip_count: 1, framework: "vLLM", tier: "verified", runner_tier: "reference", quant_bf16_throughput: 4200, quant_best_throughput: 6300, quant_best_precision: "FP8", run_id: "rh100c" },
];

test("findReversal identifies RTX 5090 vs A100-80GB flip across Suites A and D", () => {
  const rev = findReversal(SAMPLE_ROWS);
  assert.ok(rev);
  assert.equal(rev.x.chip, "RTX 5090");
  assert.equal(rev.y.chip, "A100-80GB");
  assert.ok(rev.gainA > 0.25);
  assert.ok(rev.lossB > 0.2);
});

test("findSlaGap excludes experimental runners and detects H200 vs H100 SLA gap", () => {
  const gap = findSlaGap(SAMPLE_ROWS);
  assert.ok(gap);
  assert.equal(gap.offline.chip, "H200");
  assert.equal(gap.sla.chip, "H100");
  assert.equal(gap.sla.qps, 25);
});

test("findEngineGap detects SGLang vs vLLM gap on same chip", () => {
  const eg = findEngineGap(SAMPLE_ROWS);
  assert.ok(eg);
  assert.equal(eg.chip, "H20-3e");
  assert.equal(eg.fast.framework, "SGLang");
  assert.equal(eg.slow.framework, "vLLM");
  assert.ok(eg.gain > 0.8);
});

test("findScalingSpread, findValueGap, and findQuantSpread all return valid findings", () => {
  const sc = findScalingSpread(SAMPLE_ROWS);
  assert.equal(sc.best.chip, "RTX 4090 D");
  assert.equal(sc.worst.chip, "H200");

  const val = findValueGap(SAMPLE_ROWS);
  assert.equal(val.value.chip, "Tesla T4");
  assert.equal(val.fastest.chip, "H100");
  assert.ok(val.ratio > 6);

  const q = findQuantSpread(SAMPLE_ROWS);
  assert.equal(q.top.chip, "RTX 3090");
  assert.equal(q.low.chip, "H200");
});

test("computeInsights returns all 6 findings on rich data and [] on empty input", () => {
  assert.deepEqual(computeInsights([]), []);
  const all = computeInsights(SAMPLE_ROWS);
  assert.equal(all.length, 6);
  assert.deepEqual(all.map((x) => x.id), ["reversal", "sla_gap", "engine_gap", "scaling", "value", "quant"]);
});
