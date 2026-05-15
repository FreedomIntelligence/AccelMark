// modal_viz.test.mjs — node:test coverage for the Visualize tab.
//
// modal.js's Visualize panel is the most failure-prone surface in the
// frontend: its per-suite chart renderers all dereference deeply-nested
// fields on `row.viz` and would happily blow up on a slightly malformed
// row.  These tests guard the three regressions we'd actually catch in
// review:
//
//   1. happy-path  — full suite_A row renders without throwing and
//                    actually creates Chart.js instances.
//   2. missing-viz — `viz` is present but every sub-block is empty;
//                    we should fall back to the "No visualization
//                    data" placeholder and skip the renderers entirely.
//   3. non-suite   — `viz.type` we don't know about renders the
//                    "not yet supported" placeholder rather than
//                    crashing the dispatch.
//
// We additionally smoke-test every per-suite renderer with a partially-
// populated viz block so future edits to a single renderer can't slip a
// `Cannot read properties of undefined` past CI.

import test from "node:test";
import assert from "node:assert/strict";

import { installDom } from "./dom_stub.mjs";

// IMPORTANT: install the DOM stub before importing modal.js — modal.js
// references `window` / `document` at module init time (e.g. inside
// chartColors() which the renderers call).
const dom = installDom();

const { _test } = await import("../assets/js/modal.js");

function freshPanel() {
  dom.reset();
  _test.destroyCharts();
  _test.resetColorCache();
  return dom.makePanel();
}

// ── 1. happy-path ─────────────────────────────────────────────
test("renderViz: full suite_A row creates charts and does not throw", () => {
  const panel = freshPanel();
  const row = {
    run_id: "test-run-1",
    suite: "suite_A",
    offline_throughput: 12345,
    peak_memory_gb: 70.2,
    memory_utilization_pct: 88,
    viz: {
      type: "suite_A",
      offline:    { labels: [1, 2, 4, 8], throughput: [1000, 1900, 3500, 6000] },
      online:     {
        labels:   [1, 2, 4],
        ttft_p50: [80, 95, 130],
        ttft_p90: [120, 150, 220],
        tpot_p50: [25, 28, 32],
        sla_met:  [true, true, false],
      },
      interactive: {
        ttft_p50: 100, ttft_p90: 130, ttft_p99: 180,
        tpot_p50: 25,  tpot_p99: 35,
      },
      speculative: {
        offline_tok_per_sec: 18000,
        acceptance_rate: 0.72,
        mean_accepted_tokens: 2.4,
      },
      burst: {
        burst_degradation_ratio: 1.5,
        steady_ttft_p99_ms: 200,
        burst_ttft_p99_ms: 300,
        sla_met_during_burst: true,
      },
    },
  };

  assert.doesNotThrow(() => _test.renderViz(panel, row),
    "happy-path renderViz must not throw");
  // Offline + Online lines, plus optional interactive stat-only block;
  // we only assert ≥1 to stay decoupled from chart-count tweaks.
  assert.ok(dom.chartsCreated() >= 2,
    `expected at least 2 charts, got ${dom.chartsCreated()}`);
  // Panel should have been populated with section titles, stat chips,
  // canvases — so at minimum the children list is non-empty.
  assert.ok(panel.children.length > 0,
    "panel should have children after a successful render");
});

// ── 2. missing-viz-block ──────────────────────────────────────
test("renderViz: viz with only `type` shows the no-data placeholder", () => {
  const panel = freshPanel();
  const row = { run_id: "missing-viz", suite: "suite_A", viz: { type: "suite_A" } };

  assert.doesNotThrow(() => _test.renderViz(panel, row));
  assert.equal(dom.chartsCreated(), 0,
    "no charts should be created when viz has no actual data");
  assert.match(panel.innerHTML, /No visualization data/,
    "expected the No-visualization-data placeholder");
});

// ── 3. non-suite-row (unknown viz.type) ───────────────────────
test("renderViz: unknown viz.type falls back to a friendly message", () => {
  const panel = freshPanel();
  const row = {
    run_id: "weird",
    suite: "suite_unknown",
    viz: { type: "suite_zzz", offline: { labels: [1], throughput: [9] } },
  };

  assert.doesNotThrow(() => _test.renderViz(panel, row));
  assert.equal(dom.chartsCreated(), 0,
    "unknown types must not invent charts");
  assert.match(panel.innerHTML, /not yet supported/i,
    "expected the unsupported-type placeholder");
});

// ── 4. each per-suite renderer survives a partial viz block ──
//
// Some real rows submit only the offline block (no online / interactive)
// or only the interactive block.  This battery confirms each per-suite
// dispatch handles partial input without throwing.
const partialCases = [
  ["suite_A", { offline:    { labels: [1, 2], throughput: [10, 20] } }],
  ["suite_B", { offline:    { labels: [1], throughput: [10], throughput_per_chip: [5] } }],
  ["suite_C", { precisions: ["BF16", "FP8"], throughput: [100, 200] }],
  ["suite_D", { interactive:{ ttft_p50: 50, ttft_p90: 80, ttft_p99: 100, tpot_p50: 10, tpot_p90: 12, tpot_p99: 15 } }],
  ["suite_E", { chip_counts: [1, 2, 4], throughput: [100, 190, 350], efficiency_pct: [100, 95, 88] }],
  ["suite_F", { offline:    { labels: [1], throughput: [10] } }],
  ["suite_G", { offline:    { labels: [1], throughput: [10] }, runtime_metrics: { expert_load_balance: 0.12 } }],
];

for (const [type, vizBody] of partialCases) {
  test(`renderViz: ${type} renderer accepts a partial viz block without throwing`, () => {
    const panel = freshPanel();
    const row = {
      run_id: `partial-${type}`,
      suite: type,
      viz: { type, ...vizBody },
    };
    assert.doesNotThrow(() => _test.renderViz(panel, row),
      `${type} renderer threw on partial viz`);
    // Either some charts were created or a graceful empty-state was
    // appended.  Either is fine; the key invariant is "no throw".
    assert.ok(panel.children.length > 0 || panel.innerHTML.length > 0,
      `${type} renderer produced no output at all`);
  });
}

// ── 5. vizHasAnyData: the gatekeeper before dispatch ──────────
test("vizHasAnyData: distinguishes empty vs populated viz objects", () => {
  assert.equal(_test.vizHasAnyData(null), false);
  assert.equal(_test.vizHasAnyData({}), false);
  assert.equal(_test.vizHasAnyData({ type: "suite_A" }), false);
  assert.equal(_test.vizHasAnyData({ type: "suite_A", offline: {} }), false,
    "an empty sub-object should not count as data");
  assert.equal(_test.vizHasAnyData({ type: "suite_A", offline: { labels: [1] } }), true);
});
