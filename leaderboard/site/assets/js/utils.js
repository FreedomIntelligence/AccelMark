// utils.js — pure helpers (no DOM/state side-effects)

// HTML escape — used everywhere we interpolate text into HTML.
export function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Format big numbers compactly: 12453 -> "12,453"; 1234567 -> "1.23M"
// Integers (e.g. counts) render with no decimals automatically.
export function fmtNum(v, opts = {}) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  const n = Number(v);
  const { compact = false, decimals } = opts;
  if (compact && Math.abs(n) >= 1000) {
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + "K";
  }
  // Integer fast-path: counts like 97 / 32 / 4 must not render as "97.0".
  if (decimals === undefined && Number.isInteger(n)) {
    return n.toLocaleString();
  }
  const d = decimals === undefined
    ? (Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 10 ? 1 : 2)
    : decimals;
  return n.toLocaleString(undefined, {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

export function fmtPct(v, decimals = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return `${Number(v).toFixed(decimals)}%`;
}

export function fmtMs(v, decimals = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return `${Number(v).toFixed(decimals)} ms`;
}

export function fmtDate(s) {
  if (!s) return "-";
  // Accept YYYY-MM-DD or ISO timestamp; return short YYYY-MM-DD.
  return String(s).slice(0, 10);
}

// Trim long framework versions to something display-friendly.
//   "0.7.3"                         → "0.7.3"
//   "0.19.1rc1.dev339+gedc364896"   → "0.19.1rc1"
//   "0.18.0rc1"                     → "0.18.0rc1"
// Strategy: cut at "+" (drops git hash), then at ".dev" (drops dev
// build counter). If still long, hard-cap at 12 chars.
// Title-case a string while preserving common separators (hyphen,
// slash, parens) so hero copy reads as proper headlines.
//   "Multi-chip throughput"   -> "Multi-Chip Throughput"
//   "Edge / consumer hardware"-> "Edge / Consumer Hardware"
//   "Mixture-of-Experts (MoE)"-> "Mixture-of-Experts (MoE)"
// Words that already contain *any* uppercase letter are left intact —
// covers acronyms like MoE / GPU / TPU / FP8 as well as branded camel
// casing.  Lowercase "stop words" stay lowercase unless they're the
// first token (matches AP / Chicago house-style minor words).
export function toTitleCase(s) {
  if (!s) return "";
  const STOP = new Set(["a","an","the","and","but","or","nor","for","of","on","in","to","by","at","as","vs"]);
  return String(s).replace(/[A-Za-z][A-Za-z0-9]*/g, (word, offset, full) => {
    if (/[A-Z]/.test(word)) return word;
    const isFirst = offset === 0 || /^[\s]+$/.test(full.slice(0, offset));
    if (!isFirst && STOP.has(word)) return word;
    return word.charAt(0).toUpperCase() + word.slice(1);
  });
}

export function shortVersion(v) {
  if (!v) return "";
  let s = String(v).split("+")[0];
  s = s.split(".dev")[0];
  if (s.length > 12) s = s.slice(0, 12);
  return s;
}

// Abbreviate verbose HF-style model names for tight UI slots.
//   "Meta-Llama-3-8B-Instruct"        → "Llama 3 · 8B"
//   "Meta-Llama-3-70B-Instruct"       → "Llama 3 · 70B"
//   "Llama-3.1-8B-Instruct"           → "Llama 3.1 · 8B"
//   "Qwen2.5-0.5B-Instruct"           → "Qwen 2.5 · 0.5B"
//   "Mixtral-8x7B-Instruct-v0.1"      → "Mixtral 8×7B"
//   anything-else                     → unchanged
export function shortModel(name) {
  if (!name) return "";
  let s = String(name);
  s = s.replace(/^Meta-/, "");                  // Meta-Llama-3 → Llama-3
  s = s.replace(/-Instruct(?:-v[\d.]+)?$/, ""); // drop trailing variant
  // Llama / Qwen with a "-<size>" suffix → split family · size
  const m = s.match(/^(Llama|Qwen)[-]?(\d+(?:\.\d+)?)-([\d.]+B)$/i);
  if (m) return `${m[1]} ${m[2]} · ${m[3]}`;
  // Mixtral 8x7B style → keep
  s = s.replace(/(\d+)x(\d+B)/i, "$1×$2");
  return s;
}

// Normalize a submitter field to a bare handle (no leading @).
// Accepts GitHub login, email, or "Name <email>" — falls back to the
// part before the first @ for emails.
export function submitterHandle(s) {
  if (!s) return "";
  let h = String(s).trim();
  if (!h) return "";
  if (h.startsWith("@")) h = h.slice(1);
  if (h.includes("<") && h.includes(">")) {
    const m = h.match(/<([^>]+)>/);
    if (m) h = m[1];
  }
  if (h.includes("@")) h = h.split("@")[0]; // email → local part
  return h;
}

// Stable slug for URLs.  Lowercase, hyphenated, ASCII-safe.
export function slugify(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// Build a chip slug that's stable across pages.
//
// Slug is the chip *model* only — chip_count is intentionally NOT
// encoded.  4090D, 4090D ×4, 4090D ×8 are the same hardware deployed in
// different fan-out configurations; they belong on the same chip-detail
// page where the page itself can break them out by chip-count.  Ranking
// tables and compare baskets continue to disambiguate per-row via
// `_chip_label` (which still carries "×N") and `run_id` respectively.
//
// `data.js` precomputes `_chip_slug` for every loaded row — we prefer
// that so the link target stays in lock-step with the grouping key the
// rest of the app uses (no risk of slug drift if slugify ever changes).
export function chipSlug(row) {
  if (!row) return "";
  if (row._chip_slug) return row._chip_slug;
  if (!row.chip) return "";
  return slugify(row.chip);
}

// Detect a stale "<chip>-x<N>" slug from the pre-2026-05 era and return
// the bare-model slug it should now be normalised to.  Returns null
// when `slug` is already in the new shape.  Used by the router to
// redirect old shared links instead of 404-ing them.
export function normalizeChipSlug(slug) {
  if (!slug) return null;
  const m = String(slug).match(/^(.+)-x\d+$/);
  return m ? m[1] : null;
}

// Build a URL hash for chip detail.  Falls back to the rankings landing
// when the row can't produce a slug — better to bounce somewhere useful
// than to leave a dangling `#/chip/` that 404s into the empty state.
export function chipHref(row) {
  const slug = chipSlug(row);
  return slug ? `#/chip/${slug}` : "#/rankings";
}

// Group by helper.
export function groupBy(arr, keyFn) {
  const m = new Map();
  for (const item of arr) {
    const k = keyFn(item);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(item);
  }
  return m;
}

// Pick max by metric, treating null as -Infinity.
export function maxBy(arr, fn) {
  let best = null;
  let bestV = -Infinity;
  for (const x of arr) {
    const v = fn(x);
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (v > bestV) { bestV = v; best = x; }
  }
  return best;
}

// Hash-route param parsing: takes "#/foo/bar?a=1&b=2" → { path:"/foo/bar", params:{a,b} }
export function parseHash(hash) {
  const raw = hash || "";
  const stripped = raw.startsWith("#") ? raw.slice(1) : raw;
  const [pathRaw, queryRaw] = stripped.split("?");
  const path = pathRaw || "/";
  const params = {};
  if (queryRaw) {
    for (const part of queryRaw.split("&")) {
      if (!part) continue;
      const [k, v = ""] = part.split("=");
      params[decodeURIComponent(k)] = decodeURIComponent(v);
    }
  }
  return { path, params };
}

export function buildHash(path, params = {}) {
  const qs = Object.entries(params)
    .filter(([, v]) => v !== "" && v !== null && v !== undefined)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join("&");
  return `#${path}${qs ? "?" + qs : ""}`;
}

// Empty wrapper to keep view modules consistent.
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v !== null && v !== undefined) {
      node.setAttribute(k, v);
    }
  }
  for (const c of children) {
    if (c === null || c === undefined) continue;
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else node.appendChild(c);
  }
  return node;
}
