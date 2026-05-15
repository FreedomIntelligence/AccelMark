// share_helpers.test.mjs — guard the cross-view copy/share UX helpers.
//
// `copyToClipboard` and `flashButtonLabel` live in utils.js so both the
// compare basket and the chip-detail hero share button behave the same.
// Two regressions we want fast feedback on:
//
//   1. copyToClipboard never throws — even when none of the three
//      paths (navigator.clipboard / execCommand / window.prompt) are
//      available, it must return a boolean and log nothing.
//   2. flashButtonLabel restores the original label on the second call
//      to the same button — without that, two rapid clicks would lock
//      the button into "Copied!" forever (the new flash would think
//      "Copied!" is the original label).

import test from "node:test";
import assert from "node:assert/strict";

import { installDom } from "./dom_stub.mjs";

installDom();

const { copyToClipboard, flashButtonLabel } = await import("../assets/js/utils.js");

test("copyToClipboard: returns false (and never throws) when no clipboard path is available", async () => {
  // The DOM stub doesn't provide navigator.clipboard or execCommand,
  // so the helper should fall through to the prompt branch and report
  // failure cleanly.  globalThis.window.prompt is undefined, that's ok.
  const ok = await copyToClipboard("https://example.test/share");
  assert.equal(ok, false);
});

test("copyToClipboard: succeeds via navigator.clipboard.writeText when the API is present", async () => {
  // Node's `navigator` is a non-configurable getter on globalThis, so
  // we monkey-patch the `clipboard` property on the existing object
  // (which IS configurable) for the duration of the assertion.
  const captured = [];
  const desc = Object.getOwnPropertyDescriptor(globalThis.navigator, "clipboard");
  Object.defineProperty(globalThis.navigator, "clipboard", {
    value: { writeText: async (s) => { captured.push(s); } },
    configurable: true,
    writable: true,
  });
  try {
    const ok = await copyToClipboard("hello");
    assert.equal(ok, true);
    assert.deepEqual(captured, ["hello"]);
  } finally {
    if (desc) Object.defineProperty(globalThis.navigator, "clipboard", desc);
    else delete globalThis.navigator.clipboard;
  }
});

test("copyToClipboard: surfaces a string-coerced empty value cleanly", async () => {
  // Two-call assertion: null and undefined must both reach a clipboard
  // path; we check `false` because the stub still has no clipboard API
  // — the point is just "no exception".
  await assert.doesNotReject(() => copyToClipboard(null));
  await assert.doesNotReject(() => copyToClipboard(undefined));
});

test("flashButtonLabel: restores the original label after holdMs and is re-entrant", async () => {
  // FakeEl from dom_stub doesn't carry a real querySelector — but
  // flashButtonLabel falls back to the button itself when the selector
  // resolves to null, so we can use the button as its own labelEl.
  const btn = document.createElement("button");
  btn.textContent = "Copy share link";
  // First flash — should swap label to "Copied!" and arm the timer.
  flashButtonLabel(btn, "Copied!", { holdMs: 30, labelSelector: ".missing" });
  assert.equal(btn.textContent, "Copied!");
  // Second flash before the first restored — should clear the existing
  // timer, NOT capture "Copied!" as the new "original" label.
  flashButtonLabel(btn, "Copy failed — select & ⌘C", { holdMs: 30, labelSelector: ".missing" });
  assert.equal(btn.textContent, "Copy failed — select & ⌘C");
  // Wait for the second timer to fire, then verify we landed on the
  // genuine original ("Copy share link") not the intermediate flash.
  await new Promise((r) => setTimeout(r, 80));
  assert.equal(btn.textContent, "Copy share link");
});

test("flashButtonLabel: does nothing (and doesn't throw) for a null button", () => {
  assert.doesNotThrow(() => flashButtonLabel(null, "x"));
  assert.doesNotThrow(() => flashButtonLabel(undefined, "x"));
});
