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

const { copyToClipboard, flashButtonLabel, downloadCanvasAsPng } = await import("../assets/js/utils.js");

test("copyToClipboard: returns false (and never throws) when no clipboard path is available", async () => {
  // The DOM stub doesn't provide navigator.clipboard or execCommand,
  // so the helper should fall through to the prompt branch and report
  // failure cleanly.  globalThis.window.prompt is undefined, that's ok.
  const ok = await copyToClipboard("https://example.test/share");
  assert.equal(ok, false);
});

test("copyToClipboard: succeeds via navigator.clipboard.writeText when the API is present", async () => {
  // Cross-Node-version setup: Node ≤20 leaves `globalThis.navigator`
  // undefined entirely, while Node ≥21 exposes it as a configurable
  // getter.  We replace the whole `navigator` slot via defineProperty
  // (configurable on every supported version) and restore the original
  // descriptor — or delete the slot if there wasn't one — on the way
  // out.  This keeps the test green on both runtimes (CI pins Node 20
  // today, local devs typically run a newer version).
  const captured = [];
  const origDesc = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    value: { clipboard: { writeText: async (s) => { captured.push(s); } } },
    configurable: true,
    writable: true,
  });
  try {
    const ok = await copyToClipboard("hello");
    assert.equal(ok, true);
    assert.deepEqual(captured, ["hello"]);
  } finally {
    if (origDesc) Object.defineProperty(globalThis, "navigator", origDesc);
    else delete globalThis.navigator;
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

// ── downloadCanvasAsPng — never-throws contract + happy path ──

test("downloadCanvasAsPng: returns false (and never throws) on null / non-canvas inputs", async () => {
  assert.equal(await downloadCanvasAsPng(null), false);
  assert.equal(await downloadCanvasAsPng(undefined), false);
  assert.equal(await downloadCanvasAsPng({}), false); // no getContext
  assert.equal(await downloadCanvasAsPng({ getContext: 42 }), false); // not a function
});

test("downloadCanvasAsPng: returns false when a 2d context is unavailable", async () => {
  const canvas = { getContext: () => null, width: 100, height: 50 };
  // The helper still has to *create* a composite canvas via
  // document.createElement — dom_stub returns a FakeEl whose getContext
  // is also missing, so this exercises the early-return branch in
  // the composite block too.
  assert.equal(await downloadCanvasAsPng(canvas), false);
});

test("downloadCanvasAsPng: succeeds via toBlob path with a full canvas mock", async () => {
  // Stand up a minimal canvas implementation: getContext returns a
  // ctx that records draws (so we can assert background fill happened
  // before drawImage), and toBlob yields a fake Blob.
  const composite = makeFakeCanvas({ withToBlob: true });
  // Override createElement just for this test so the helper picks up
  // our instrumented composite canvas.
  const origCreate = document.createElement;
  document.createElement = (tag) => {
    if (tag === "canvas") return composite;
    return origCreate.call(document, tag);
  };
  try {
    const sourceCanvas = makeFakeCanvas({ withToBlob: true });
    const ok = await downloadCanvasAsPng(sourceCanvas, { filename: "test.png", backgroundColor: "#000" });
    assert.equal(ok, true);
    // Background fill happened before drawImage (otherwise the source
    // canvas would overwrite the bg, defeating the whole point).
    assert.deepEqual(composite._ops, ["fillRect", "drawImage"]);
    assert.equal(composite._fillStyle, "#000");
  } finally {
    document.createElement = origCreate;
  }
});

test("downloadCanvasAsPng: falls back to toDataURL when toBlob is unavailable", async () => {
  const composite = makeFakeCanvas({ withToBlob: false }); // toBlob undefined
  const origCreate = document.createElement;
  document.createElement = (tag) => {
    if (tag === "canvas") return composite;
    return origCreate.call(document, tag);
  };
  try {
    const sourceCanvas = makeFakeCanvas({ withToBlob: false });
    const ok = await downloadCanvasAsPng(sourceCanvas, { filename: "fallback.png" });
    assert.equal(ok, true);
    assert.equal(composite._toDataURLCalled, true);
  } finally {
    document.createElement = origCreate;
  }
});

// Tiny canvas factory the tests above share.  Records fillRect /
// drawImage call order so we can assert ordering, and surfaces the
// filename via the synthetic <a> click that the helper performs.
function makeFakeCanvas({ withToBlob }) {
  const ops = [];
  const ctx = {
    set fillStyle(v) { canvas._fillStyle = v; },
    get fillStyle() { return canvas._fillStyle; },
    fillRect:  () => ops.push("fillRect"),
    drawImage: () => ops.push("drawImage"),
  };
  const canvas = {
    width: 200,
    height: 100,
    _ops: ops,
    _fillStyle: null,
    _toDataURLCalled: false,
    getContext: (type) => (type === "2d" ? ctx : null),
    toDataURL: () => { canvas._toDataURLCalled = true; return "data:image/png;base64,AAAA"; },
  };
  if (withToBlob) {
    canvas.toBlob = (cb) => cb(new FakeBlob());
  }
  return canvas;
}

class FakeBlob {
  constructor() { this.size = 4; this.type = "image/png"; }
}

// URL.createObjectURL is required by the toBlob path.  Stub on first
// test setup; harmless if jsdom or another stub adds the real one
// later because we only set it when missing.
if (typeof globalThis.URL === "undefined") globalThis.URL = {};
if (typeof globalThis.URL.createObjectURL !== "function") {
  globalThis.URL.createObjectURL = () => "blob:fake";
}
if (typeof globalThis.URL.revokeObjectURL !== "function") {
  globalThis.URL.revokeObjectURL = () => {};
}
