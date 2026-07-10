// contributors.test.mjs — contributor index + reproduction quests.



import test from "node:test";

import assert from "node:assert/strict";



import { installDom } from "./dom_stub.mjs";



installDom();



// Minimal leaderboard fixture for contributor aggregation.

globalThis.window = globalThis.window || {};

globalThis.window.LEADERBOARD_DATA = [

  {

    run_id: "run_a1",

    chip: "H100 80GB",

    vendor: "NVIDIA",

    suite: "suite_A",

    tier: "community",

    date: "2026-01-01",

    submitted_by: "alice",

    framework: "vLLM",

    framework_version: "0.7.3",

  },

  {

    run_id: "run_a2",

    chip: "H100 80GB",

    vendor: "NVIDIA",

    suite: "suite_A",

    tier: "verified",

    date: "2026-02-01",

    submitted_by: "bob",

    framework: "vLLM",

    framework_version: "0.7.3",

    reproduces_run_id: "run_a1",

  },

  {

    run_id: "run_b1",

    chip: "MI300X",

    vendor: "AMD",

    suite: "suite_B",

    tier: "community",

    date: "2026-03-01",

    submitted_by: "alice",

  },

];

globalThis.window.SUITE_SPECS = {};



const { _resetContributorCache, contributorIndex, reproductionQuests, reproductionQuestExplain, crossVerifiedRecords, firstResultBadgeForChip } =

  await import("../assets/js/contributors.js");

const { wantedFromCatalog, flattenCatalog, normChipKey } =

  await import("../assets/js/hardware-catalog.js");

const { discussResultUrl, DISCUSS_TEMPLATES } =

  await import("../assets/js/cite.js");



test("contributorIndex ranks alice with first-result on MI300X", () => {

  _resetContributorCache();

  const list = contributorIndex();

  assert.ok(list.length >= 2);

  const alice = list.find((c) => c.handle === "alice");

  assert.ok(alice);

  assert.ok(alice.badges.includes("first-result"));

  assert.ok(alice.firstChips.includes("MI300X"));

});



test("crossVerifiedRecords only follows explicit reproduces_run_id links", () => {

  _resetContributorCache();

  const records = crossVerifiedRecords();

  assert.equal(records.length, 1);

  assert.equal(records[0].reproduces_run_id, "run_a1");

  assert.equal(records[0].original.submitted_by, "alice");

  assert.equal(records[0].verification.submitted_by, "bob");

  assert.equal(records[0].original.framework, "vLLM");

  assert.equal(records[0].verification.framework, "vLLM");

});



test("crossVerifiedRecords ignores chip+suite overlap without reproduces_run_id", () => {

  _resetContributorCache();

  const data = globalThis.window.LEADERBOARD_DATA;

  data.push({

    run_id: "run_x1",

    chip: "MI300X",

    vendor: "AMD",

    suite: "suite_B",

    tier: "verified",

    date: "2026-04-01",

    submitted_by: "carol",

    framework: "SGLang",

  });

  _resetContributorCache();

  const records = crossVerifiedRecords();

  assert.equal(records.length, 1);

  assert.ok(!records.some((r) => r.verification.submitted_by === "carol"));

  data.pop();

});



test("reproductionQuestExplain reports open vs verified coverage", () => {

  _resetContributorCache();

  const ex = reproductionQuestExplain();

  assert.equal(ex.openQuestCount, 1);

  assert.equal(ex.openPairCount, 1);

  assert.equal(ex.crossVerifiedRecordCount, 1);

  assert.equal(ex.verifiedPairs, 1);

  assert.equal(ex.communityRuns, 2);

});



test("reproductionQuests excludes verified chip+suite pairs", () => {

  _resetContributorCache();

  const quests = reproductionQuests();

  const h100Verified = quests.find((q) => normChipKey(q.chip).includes("h100") && q.suite === "suite_A");

  assert.equal(h100Verified, undefined);

  const mi300 = quests.find((q) => q.chip === "MI300X");

  assert.ok(mi300);

});



test("firstResultBadgeForChip returns earliest submitter", () => {

  _resetContributorCache();

  assert.equal(firstResultBadgeForChip("MI300X"), "alice");

  assert.equal(firstResultBadgeForChip("H100 80GB"), "alice");

});



test("wantedFromCatalog includes accelerators not on leaderboard", () => {

  const wanted = wantedFromCatalog(["H100 80GB", "MI300X"]);

  assert.ok(wanted.some((w) => w.name.includes("RTX 5090")));

  assert.ok(!wanted.some((w) => w.name === "H100 80GB"));

});



test("flattenCatalog has many vendors beyond tested hardware", () => {

  const flat = flattenCatalog();

  assert.ok(flat.length > 40);

  const vendors = new Set(flat.map((f) => f.vendor));

  assert.ok(vendors.has("Intel"));

  assert.ok(vendors.has("Qualcomm"));

});



test("DISCUSS_TEMPLATES exposes five discussion entry points", () => {

  assert.equal(DISCUSS_TEMPLATES.length, 5);

  const row = globalThis.window.LEADERBOARD_DATA[0];

  const url = discussResultUrl(row, "reproduction");

  assert.ok(url.includes("discussions/new"));

  assert.ok(url.includes("Reproduction"));

});


