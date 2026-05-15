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
export function fmtNum(v, opts = {}) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const n = Number(v);
  const { compact = false, decimals } = opts;
  if (compact && Math.abs(n) >= 1000) {
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + "K";
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
