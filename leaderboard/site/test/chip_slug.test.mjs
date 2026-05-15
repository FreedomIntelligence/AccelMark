// chip_slug.test.mjs — guard the chip-detail URL contract.
//
// chipSlug deliberately does NOT encode chip_count: 4090D, 4090D ×4,
// 4090D ×8 are the same hardware and share a chip-detail page.  This
// is a behaviour change from the original implementation, where the
// slug used to carry an "-x<N>" suffix.  Two regressions we'd want to
// catch fast:
//
//   1. someone re-introducing chip_count into chipSlug — would re-fork
//      ×1 / ×4 / ×8 back into separate detail pages.
//   2. someone removing the legacy "-x<N>" → bare-model normaliser —
//      would 404 every shared link / bookmark from before the change.

import test from "node:test";
import assert from "node:assert/strict";

import { chipSlug, normalizeChipSlug, slugify } from "../assets/js/utils.js";

test("chipSlug: same chip at different chip_count produces the same slug", () => {
  const x1 = { chip: "RTX 4090D", chip_count: 1 };
  const x4 = { chip: "RTX 4090D", chip_count: 4 };
  const x8 = { chip: "RTX 4090D", chip_count: 8 };
  assert.equal(chipSlug(x1), chipSlug(x4));
  assert.equal(chipSlug(x4), chipSlug(x8));
  assert.equal(chipSlug(x1), slugify("RTX 4090D"));
});

test("chipSlug: missing chip / row returns empty string", () => {
  assert.equal(chipSlug(null), "");
  assert.equal(chipSlug(undefined), "");
  assert.equal(chipSlug({}), "");
});

test("chipSlug: precomputed _chip_slug wins over recomputation", () => {
  const row = { chip: "Will Be Ignored", _chip_slug: "precomputed-value" };
  assert.equal(chipSlug(row), "precomputed-value");
});

test("normalizeChipSlug: legacy -x<N> shape is mapped back to the bare model", () => {
  assert.equal(normalizeChipSlug("nvidia-rtx-4090d-x4"), "nvidia-rtx-4090d");
  assert.equal(normalizeChipSlug("apple-m4-max-x1"),     "apple-m4-max");
  assert.equal(normalizeChipSlug("h200-x16"),            "h200");
});

test("normalizeChipSlug: new bare-model slugs are reported unchanged (null)", () => {
  // Returning null lets the router skip the rewrite + replaceState
  // dance for slugs that are already in the new shape.
  assert.equal(normalizeChipSlug("nvidia-rtx-4090d"), null);
  assert.equal(normalizeChipSlug("apple-m4-max"),     null);
  assert.equal(normalizeChipSlug(""),                  null);
  assert.equal(normalizeChipSlug(null),                null);
});

test("normalizeChipSlug: known boundary — slug tokens that look like -x<digits>", () => {
  // The regex matches any `-x<digits>$` tail, so a hypothetical chip
  // ending in "x86" or "x64" would also be rewritten.  Document this
  // here so a future refactor doesn't widen the regex by accident.
  // No real chip in the dataset has this shape today; if one ever
  // ships, switch the redirect to a known-suffix allowlist or check
  // the rewrite target actually exists in `_byChip` before applying.
  assert.equal(normalizeChipSlug("foo-x86"), "foo");
  // Suffix with non-digits stays put — no rewrite.
  assert.equal(normalizeChipSlug("foo-xy"),  null);
  assert.equal(normalizeChipSlug("foo-x"),   null);
});
