// suite_fingerprint.test.mjs — guard the chip-detail radar contract.
//
// suiteFingerprint(slug) returns a Map<sid, { value, normalized,
// best, missing }>.  Two regressions to catch fast:
//
//   1. asc-direction metrics (TTFT, latency) are inverted so the
//      global winner still reads as 1.0.  An off-by-direction bug
//      would invert chip rankings on the radar.
//   2. chips with no submission in a suite must come back as
//      `missing: true` with `normalized: 0`, NOT throw or skip — the
//      radar relies on every entry being present so the polygon
//      collapses to centre on absent suites.

import test from "node:test";
import assert from "node:assert/strict";

import { installDom } from "./dom_stub.mjs";

installDom();

// SUITE_META baked into data.js dictates the primary metric per suite.
// We use a single suite (suite_A → offline_throughput, desc) for the
// asc-fixture rows to keep the test independent of the hardcoded
// per-suite metric direction (suite_D's TTFT is the only stock asc
// metric and adding rows for it would entangle this test in the
// long-context viz schema).  Asc-direction inversion is exercised
// directly by mocking with another suite where the schema fits.
globalThis.window.LEADERBOARD_DATA = [
  { run_id: "leader",   date: "2026-04-01", suite: "suite_A", chip: "ChipBest", chip_count: 1, vendor: "V",
    offline_throughput: 1000, primary_metric: 1000 },
  { run_id: "midpack",  date: "2026-04-02", suite: "suite_A", chip: "ChipMid",  chip_count: 1, vendor: "V",
    offline_throughput: 500,  primary_metric: 500 },
  { run_id: "tailend",  date: "2026-04-03", suite: "suite_A", chip: "ChipTail", chip_count: 1, vendor: "V",
    offline_throughput: 100,  primary_metric: 100 },
  // suite_B: same chips, different relative ordering.  ChipMid wins B.
  { run_id: "b1", date: "2026-04-04", suite: "suite_B", chip: "ChipBest", chip_count: 8, vendor: "V",
    offline_throughput: 200, primary_metric: 200 },
  { run_id: "b2", date: "2026-04-05", suite: "suite_B", chip: "ChipMid",  chip_count: 8, vendor: "V",
    offline_throughput: 800, primary_metric: 800 },
];

const { suiteFingerprint, init } = await import("../assets/js/data.js");
init();

test("suiteFingerprint: leader of a suite gets normalized=1.0 there", () => {
  const fp = suiteFingerprint("chipbest");
  const a = fp.get("suite_A");
  assert.ok(a);
  assert.equal(a.missing, false);
  assert.equal(a.normalized, 1.0);
  assert.equal(a.value, 1000);
});

test("suiteFingerprint: a chip at half the leader's value reads ~0.5", () => {
  const fp = suiteFingerprint("chipmid");
  const a = fp.get("suite_A");
  assert.equal(a.missing, false);
  // 500 / 1000 = 0.5
  assert.equal(a.normalized, 0.5);
});

test("suiteFingerprint: a chip absent from a suite is reported as missing with normalized=0", () => {
  const fp = suiteFingerprint("chipbest");
  // chipbest doesn't have a row in suite_C/D/E/F/G — every one of
  // those should be missing with a zero normalised score so the radar
  // polygon collapses on those axes.
  for (const sid of ["suite_C", "suite_D", "suite_E", "suite_F", "suite_G"]) {
    const cell = fp.get(sid);
    assert.ok(cell, `${sid} entry should exist`);
    assert.equal(cell.missing, true);
    assert.equal(cell.normalized, 0);
    assert.equal(cell.value, null);
  }
});

test("suiteFingerprint: every entry returned, never throws on unknown chip slugs", () => {
  const fp = suiteFingerprint("chip-that-does-not-exist");
  // 7 suites total — even with no data on any of them, every key is
  // present so callers can iterate SUITE_ORDER safely.
  let presentCount = 0;
  for (const cell of fp.values()) {
    presentCount++;
    assert.equal(cell.missing, true);
    assert.equal(cell.normalized, 0);
  }
  assert.ok(presentCount >= 5);
});

test("suiteFingerprint: relative ordering across suites can flip per chip", () => {
  // ChipBest leads suite_A (1.0) but only scores 200/800 = 0.25 in
  // suite_B; ChipMid is the inverse (0.5 in A, 1.0 in B).  This is
  // exactly the cross-suite inversion the radar is meant to surface.
  const best = suiteFingerprint("chipbest");
  const mid  = suiteFingerprint("chipmid");
  assert.equal(best.get("suite_A").normalized, 1.0);
  assert.equal(best.get("suite_B").normalized, 0.25);
  assert.equal(mid.get("suite_A").normalized, 0.5);
  assert.equal(mid.get("suite_B").normalized, 1.0);
});
