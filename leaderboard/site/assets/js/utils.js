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
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
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
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${Number(v).toFixed(decimals)}%`;
}

export function fmtMs(v, decimals = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${Number(v).toFixed(decimals)} ms`;
}

export function fmtDate(s) {
  if (!s) return "—";
  // Accept YYYY-MM-DD or ISO timestamp; return short YYYY-MM-DD.
  return String(s).slice(0, 10);
}

// Trim long framework versions to something display-friendly.
//   "0.7.3"                         → "0.7.3"
//   "0.19.1rc1.dev339+gedc364896"   → "0.19.1rc1"
//   "0.18.0rc1"                     → "0.18.0rc1"
// Strategy: cut at "+" (drops git hash), then at ".dev" (drops dev
// build counter). If still long, hard-cap at 12 chars.
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
// Includes count to keep H100 ×1 distinct from H100 ×8.
export function chipSlug(row) {
  if (!row) return "";
  const parts = [row.chip, "x" + (row.chip_count || 1)];
  return slugify(parts.join("-"));
}

// Build a URL hash for chip detail.
export function chipHref(row) {
  return `#/chip/${chipSlug(row)}`;
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
