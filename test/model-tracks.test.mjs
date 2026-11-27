import test from "node:test";
import assert from "node:assert/strict";
import { baseSuiteId, modelTrackId, suiteMetaFor, suiteColumnsFor, rowsForSuite, tracksForSuite } from "../assets/js/data.js";
import { buildDeployRecipe } from "../assets/js/deploy-recipe.js";

test("baseSuiteId and modelTrackId parse canonical track suite_ids", () => {
  assert.equal(baseSuiteId("suite_A"), "suite_A");
  assert.equal(modelTrackId("suite_A"), "default");

  assert.equal(baseSuiteId("suite_A_qwen38_27b"), "suite_A");
  assert.equal(modelTrackId("suite_A_qwen38_27b"), "qwen38_27b");

  assert.equal(baseSuiteId("suite_F_qwen3_06b"), "suite_F");
  assert.equal(modelTrackId("suite_F_qwen3_06b"), "qwen3_06b");
});

test("suiteMetaFor and suiteColumnsFor inherit base suite specs for model tracks", () => {
  assert.equal(suiteMetaFor("suite_A_qwen38_27b")?.letter, "A");
  assert.equal(suiteMetaFor("suite_F_qwen3_06b")?.letter, "F");
  assert.equal(suiteColumnsFor("suite_F_qwen3_06b").length, suiteColumnsFor("suite_F").length);
});

test("rowsForSuite returns rows for non-default model track suite_ids", () => {
  globalThis.window = {
    LEADERBOARD_DATA: [
      { run_id: "r1", suite: "suite_F", base_suite: "suite_F", model_track: "default", chip: "RTX 6000D", framework: "vLLM", offline_throughput: 10000 },
      { run_id: "r2", suite: "suite_F_qwen3_06b", base_suite: "suite_F", model_track: "qwen3_06b", model_track_label: "Qwen3-0.6B", chip: "RTX 6000D", framework: "SGLang", offline_throughput: 18500 },
      { run_id: "r3", suite: "suite_F_qwen3_06b", base_suite: "suite_F", model_track: "qwen3_06b", model_track_label: "Qwen3-0.6B", chip: "RTX 6000D", framework: "LMDeploy", offline_throughput: 16200 },
    ],
    SUITE_SPECS: {},
  };
  const qwenRows = rowsForSuite("suite_F_qwen3_06b");
  assert.equal(qwenRows.length, 2);
  assert.equal(qwenRows[0].framework, "SGLang");
  const tracks = tracksForSuite("suite_F");
  assert.ok(tracks.some((t) => t.suite_id === "suite_F_qwen3_06b" && t.count === 2));
});

test("buildDeployRecipe emits --suite <base> --model-track <slug> for model tracks", () => {
  const row = {
    run_id: "f0a2d2ff",
    suite: "suite_F_qwen3_06b",
    base_suite: "suite_F",
    model_track: "qwen3_06b",
    chip: "NVIDIA RTX 6000D",
    framework: "SGLang",
    precision: "BF16",
    implementation_id: "nvidia_sglang_c43a8309",
    detail: { model_id: "Qwen/Qwen3-0.6B" },
  };
  const rec = buildDeployRecipe(row, { mode: "serve" });
  assert.match(rec.launchCmd, /--suite suite_F --model-track qwen3_06b --serve/);
  assert.match(rec.reproduceCmd, /--suite suite_F --model-track qwen3_06b/);
});
