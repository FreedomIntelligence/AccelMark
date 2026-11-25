import test from "node:test";
import assert from "node:assert/strict";
import {
  AGENT_WORKLOADS,
  A100_MATRIX,
  CROSS_HARDWARE_TRACKS,
  PROTOCOL_AUDIT_CASES,
  AGENT_OPEN_QUESTS,
} from "../assets/data/agent-results.js";

test("A100_MATRIX covers 10 setups x 5 workloads = 50 cells with 26 validated successes", () => {
  assert.equal(AGENT_WORKLOADS.length, 5);
  assert.equal(A100_MATRIX.length, 10);

  let totalCells = 0;
  let totalSuccesses = 0;
  for (const row of A100_MATRIX) {
    let rowSucc = 0;
    for (const w of AGENT_WORKLOADS) {
      const cell = row.cells[w.id];
      assert.ok(cell && typeof cell.gain === "number", `Missing cell ${row.backbone}/${row.harness}/${w.id}`);
      totalCells += 1;
      if (!cell.unchanged && cell.gain > 2.0) {
        rowSucc += 1;
        totalSuccesses += 1;
      }
    }
    assert.equal(rowSucc, row.successes, `Success count mismatch for ${row.backbone}/${row.harness}`);
  }
  assert.equal(totalCells, 50);
  assert.equal(totalSuccesses, 26);
});

test("CROSS_HARDWARE_TRACKS, PROTOCOL_AUDIT_CASES, and AGENT_OPEN_QUESTS are populated", () => {
  assert.equal(CROSS_HARDWARE_TRACKS.length, 6);
  assert.equal(PROTOCOL_AUDIT_CASES.length, 7);
  assert.ok(AGENT_OPEN_QUESTS.length >= 3);
});
