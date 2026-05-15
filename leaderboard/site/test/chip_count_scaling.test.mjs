// chip_count_scaling.test.mjs — guard the chip-detail scaling chart.
//
// chipCountScaling(slug) feeds the grouped bar chart on the chip-detail
// 03 · Scaling section.  Two regressions to lock down:
//
//   1. Single-fan-out chips return an empty `suites` list — the
//      section depends on `data.suites.length` to decide whether to
//      render at all, and rendering an empty bar chart for, say, an
//      Apple M2 (always ×1) would look like a layout bug.
//   2. Per-suite normalization is intra-cluster only.  Each chip-count
//      cell on a suite is normalised to that suite's max value across
//      the chip's own variants — NOT to a global max.  A regression
//      that conflates them would make every bar read 100% on bandwidth
//      suites and ~0% on multi-chip suites.

import test from "node:test";
import assert from "node:assert/strict";

import { installDom } from "./dom_stub.mjs";

installDom();

// Two synthetic chips:
//   ChipScale: ×1, ×4, ×8 on suite_A; only ×1 on suite_B (so ×4/×8
//             cells should be zero-filled, not absent).
//   ChipSolo: only ×1 on suite_A — drives the "empty suites list"
//             branch via chipCounts < 2.
globalThis.window.LEADERBOARD_DATA = [
  // ChipScale on suite_A
  { run_id: "scale-a-1",  date: "2026-04-01", suite: "suite_A", chip: "ChipScale", chip_count: 1, vendor: "V",
    offline_throughput: 100, primary_metric: 100 },
  { run_id: "scale-a-4",  date: "2026-04-02", suite: "suite_A", chip: "ChipScale", chip_count: 4, vendor: "V",
    offline_throughput: 380, primary_metric: 380 },
  { run_id: "scale-a-8",  date: "2026-04-03", suite: "suite_A", chip: "ChipScale", chip_count: 8, vendor: "V",
    offline_throughput: 700, primary_metric: 700 },
  // ChipScale on suite_B (only ×1 — should still render a row with ×4/×8 cells nullified)
  { run_id: "scale-b-1",  date: "2026-04-04", suite: "suite_B", chip: "ChipScale", chip_count: 1, vendor: "V",
    offline_throughput: 50,  primary_metric: 50 },
  // ChipSolo on suite_A — single fan-out → scaling section should suppress.
  { run_id: "solo-a-1",   date: "2026-04-05", suite: "suite_A", chip: "ChipSolo",  chip_count: 1, vendor: "V",
    offline_throughput: 200, primary_metric: 200 },
];

const { chipCountScaling, init } = await import("../assets/js/data.js");
init();

test("chipCountScaling: chip with a single fan-out returns empty suites + chipCounts of length 1", () => {
  const out = chipCountScaling("chipsolo");
  assert.deepEqual(out.chipCounts, [1]);
  assert.equal(out.suites.length, 0);
});

test("chipCountScaling: multi-fan-out chip surfaces every chip_count and normalises within suite", () => {
  const out = chipCountScaling("chipscale");
  assert.deepEqual(out.chipCounts, [1, 4, 8]);
  // suite_A should be present with a perCount Map covering 1/4/8.
  const suiteA = out.suites.find((s) => s.sid === "suite_A");
  assert.ok(suiteA, "suite_A entry should exist");
  const cellA1 = suiteA.perCount.get(1);
  const cellA4 = suiteA.perCount.get(4);
  const cellA8 = suiteA.perCount.get(8);
  assert.equal(cellA1.value, 100);
  assert.equal(cellA4.value, 380);
  assert.equal(cellA8.value, 700);
  // Normalised against the cluster max (700).
  assert.equal(cellA1.normalized, 100 / 700);
  assert.equal(cellA4.normalized, 380 / 700);
  assert.equal(cellA8.normalized, 1.0);
});

test("chipCountScaling: cells without a submission for that chip-count come back as null + zero", () => {
  const out = chipCountScaling("chipscale");
  const suiteB = out.suites.find((s) => s.sid === "suite_B");
  assert.ok(suiteB, "suite_B should still render so the chart shows the gap");
  // ×1 has data, ×4 and ×8 don't — must be null/zero, not missing keys.
  assert.equal(suiteB.perCount.get(1).value, 50);
  assert.equal(suiteB.perCount.get(1).normalized, 1.0);
  assert.equal(suiteB.perCount.get(4).value, null);
  assert.equal(suiteB.perCount.get(4).normalized, 0);
  assert.equal(suiteB.perCount.get(8).value, null);
  assert.equal(suiteB.perCount.get(8).normalized, 0);
});

test("chipCountScaling: suites the chip never submitted to (any count) are omitted entirely", () => {
  const out = chipCountScaling("chipscale");
  // ChipScale never touched suites C/D/E/F/G — they should NOT appear,
  // saving the chart from rendering empty clusters.
  for (const sid of ["suite_C", "suite_D", "suite_E", "suite_F", "suite_G"]) {
    assert.equal(out.suites.find((s) => s.sid === sid), undefined,
      `${sid} should be omitted when the chip has no data on it at any chip_count`);
  }
});

test("chipCountScaling: unknown chip slug returns a stable shape (no throw, empty suites)", () => {
  const out = chipCountScaling("does-not-exist");
  assert.deepEqual(out.chipCounts, []);
  assert.deepEqual(out.suites, []);
});
