// recent_since.test.mjs — guard the home hero "this week" counter.
//
// recentSince(days, now) is what the home hero ribbon and the
// "this week" KPI both call.  Two regressions we want to catch:
//
//   1. cutoff drift — an off-by-one in the days math would either
//      double-count or under-count, both visible to contributors.
//   2. silent failure when rows are missing dates — the loader
//      sometimes drops `date` for runs imported from older formats;
//      those should be ignored, not crash the hero render.
//
// We inject a hand-built fixture as `window.LEADERBOARD_DATA` *before*
// importing data.js so init() picks it up on the very first call.

import test from "node:test";
import assert from "node:assert/strict";

import { installDom } from "./dom_stub.mjs";

installDom();

// Fixture spans a 6-month window so we can exercise multiple cutoff
// distances without contriving wall-clock dates.
globalThis.window.LEADERBOARD_DATA = [
  { run_id: "old",   date: "2026-01-01", suite: "suite_A", chip: "X", chip_count: 1, vendor: "V" },
  { run_id: "edge",  date: "2026-05-09", suite: "suite_A", chip: "X", chip_count: 1, vendor: "V" },
  { run_id: "fresh1", date: "2026-05-15", suite: "suite_A", chip: "X", chip_count: 1, vendor: "V" },
  { run_id: "fresh2", date: "2026-05-10", suite: "suite_A", chip: "X", chip_count: 1, vendor: "V" },
  { run_id: "fresh3", date: "2026-05-13", suite: "suite_A", chip: "X", chip_count: 1, vendor: "V" },
  { run_id: "today", date: "2026-05-16", suite: "suite_A", chip: "X", chip_count: 1, vendor: "V" },
  { run_id: "no-date", suite: "suite_A", chip: "X", chip_count: 1, vendor: "V" },
];

const { recentSince, init } = await import("../assets/js/data.js");
init();

// Pin "now" so the cutoff math is reproducible regardless of when the
// suite runs in CI.
const NOW = new Date("2026-05-16T00:00:00Z");

test("recentSince: 7-day window catches every row newer than today-7", () => {
  // cutoff = 2026-05-09 (inclusive via lexicographic >=)
  // matches: edge, fresh1, fresh2, fresh3, today = 5
  assert.equal(recentSince(7, NOW), 5);
});

test("recentSince: 1-day window is just today + yesterday", () => {
  // cutoff = 2026-05-15
  // matches: fresh1 (2026-05-15), today (2026-05-16) = 2
  assert.equal(recentSince(1, NOW), 2);
});

test("recentSince: huge window catches every dated row, never the undated one", () => {
  // 365 days catches everything dated, and the `no-date` row stays
  // skipped because the body explicitly guards on `r.date`.
  assert.equal(recentSince(365, NOW), 6);
});

test("recentSince: 0, negative, NaN, and non-numeric days return 0", () => {
  assert.equal(recentSince(0, NOW), 0);
  assert.equal(recentSince(-1, NOW), 0);
  assert.equal(recentSince(NaN, NOW), 0);
  assert.equal(recentSince("not-a-number", NOW), 0);
});

test("recentSince: defaults `now` to wall clock so callers don't have to thread it", () => {
  // We can't pin Date.now() here without monkey-patching, but we can
  // verify the call doesn't throw and returns a non-negative integer.
  const n = recentSince(7);
  assert.ok(Number.isInteger(n));
  assert.ok(n >= 0);
});
