// vendor_color.test.mjs — guard the vendor-colour single-source-of-truth.
//
// Adding a new vendor used to mean editing 70 lines of CSS across 6
// files; now it's one entry in `VENDOR_COLORS` (or zero entries if you
// don't care about a brand colour — the deterministic fallback picks a
// stable shade for you).  These tests pin that contract:
//
//   • known vendors return their brand colour exactly.
//   • unknown vendors get a colour from the fallback palette.
//   • the same unknown vendor always lands on the same fallback shade
//     across calls (no per-pageload reshuffling).
//   • two different unknown vendors usually land on different shades
//     (no degenerate "everyone is teal" failure mode).

import test from "node:test";
import assert from "node:assert/strict";

import { VENDOR_COLORS, vendorColor, VENDOR_ORDER } from "../assets/js/data.js";

test("vendorColor: known vendors return the brand colour from VENDOR_COLORS", () => {
  for (const [name, hex] of Object.entries(VENDOR_COLORS)) {
    assert.equal(vendorColor(name), hex,
      `expected ${name} to map to ${hex}, got ${vendorColor(name)}`);
  }
});

test("vendorColor: unknown vendor falls back to a deterministic palette colour", () => {
  const c1 = vendorColor("Cerebras");
  const c2 = vendorColor("Cerebras");
  assert.equal(c1, c2, "same vendor name must return the same colour");
  // The fallback palette is hex-coded — the result should look like a
  // 6-digit hex colour, not "undefined" or an empty string.
  assert.match(c1, /^#[0-9a-fA-F]{6}$/);
});

test("vendorColor: empty / null vendor returns a neutral grey, never throws", () => {
  assert.match(vendorColor(""),        /^#[0-9a-fA-F]{6}$/);
  assert.match(vendorColor(null),      /^#[0-9a-fA-F]{6}$/);
  assert.match(vendorColor(undefined), /^#[0-9a-fA-F]{6}$/);
});

test("vendorColor: most pairs of unknown vendors land on different fallback shades", () => {
  const probes = ["Cerebras", "Tenstorrent", "Graphcore", "Groq", "SambaNova", "Mythic", "Etched"];
  const colours = probes.map(vendorColor);
  const distinct = new Set(colours).size;
  // 7 probes against a 9-colour palette: the deterministic hash will
  // very occasionally collide, but a healthy spread is ≥4 distinct
  // shades.  This is a "no degenerate clumping" guard, not a hard
  // uniqueness contract.
  assert.ok(distinct >= 4,
    `expected at least 4 distinct shades across ${probes.length} unknown vendors, got ${distinct}`);
});

test("VENDOR_ORDER: derived from VENDOR_COLORS so adding a vendor pins its rank", () => {
  // The rankings facet pill row uses VENDOR_ORDER to lay out the
  // brand-name vendors in a stable order; vendors NOT in this list
  // get appended alphabetically (see buildFacets in rankings.js).
  // If VENDOR_COLORS and VENDOR_ORDER ever drift apart, we'd start
  // showing brand-coloured pills at unexpected positions.
  assert.deepEqual(VENDOR_ORDER, Object.keys(VENDOR_COLORS));
});
