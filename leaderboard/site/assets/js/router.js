// router.js — minimal hash router.
//
// Routes are registered with patterns like:
//   "/"                  → home
//   "/rankings"          → rankings (suite via ?suite=...)
//   "/chip/:slug"        → chip detail
//   "/compare"           → compare (chips via ?chips=a,b,c)
//   "/suites"            → suite explorer
//
// Each handler receives ({ params, query, el }) and is responsible for
// rendering into `el` (the main view container).

import { parseHash, normalizeChipSlug } from "./utils.js";

const routes = [];
let mountEl = null;
let _compareBasket = new Set();

export function mount(el) { mountEl = el; }

export function register(pattern, handler) {
  const parts = pattern.split("/").filter(Boolean);
  routes.push({ pattern, parts, handler });
}

function matchRoute(path) {
  const target = path.split("/").filter(Boolean);
  for (const r of routes) {
    if (r.parts.length !== target.length) continue;
    const params = {};
    let ok = true;
    for (let i = 0; i < r.parts.length; i++) {
      const expected = r.parts[i];
      const actual = target[i];
      if (expected.startsWith(":")) {
        params[expected.slice(1)] = decodeURIComponent(actual);
      } else if (expected !== actual) {
        ok = false; break;
      }
    }
    if (ok) return { route: r, params };
  }
  return null;
}

export function dispatch() {
  if (!mountEl) return;
  const { path, params: query } = parseHash(location.hash);

  // Backward-compat for the pre-2026-05 chip slug shape (`<chip>-x<N>`)
  // — chip-detail pages are now per-model, with chip-count variants
  // surfaced inside the page.  Old shared links / bookmarks like
  // `#/chip/nvidia-rtx-4090d-x4` would otherwise hit the empty state;
  // rewrite to the new bare-model slug and let the dispatch continue.
  // We use replaceState so a back-button click goes to wherever the
  // user came from rather than the legacy URL itself.
  const chipMatch = path.match(/^\/chip\/(.+)$/);
  if (chipMatch) {
    const normalised = normalizeChipSlug(chipMatch[1]);
    if (normalised) {
      const qs = location.hash.includes("?")
        ? "?" + location.hash.split("?").slice(1).join("?")
        : "";
      const next = `#/chip/${encodeURIComponent(normalised)}${qs}`;
      history.replaceState(null, "", location.pathname + location.search + next);
      // Re-dispatch with the corrected hash so this turn renders the
      // right view immediately rather than waiting for a hashchange.
      return dispatch();
    }
  }

  const match = matchRoute(path);
  // Reset scroll on route change.
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });

  if (!match) {
    mountEl.innerHTML = `
      <section class="state">
        <span class="state-icon">⚠</span>
        Unknown route: <code>${path}</code><br>
        <a href="#/" class="btn primary" style="margin-top:1rem">Back to home</a>
      </section>
    `;
    syncNav(path);
    return;
  }

  try {
    match.route.handler({
      params: match.params,
      query,
      el: mountEl,
    });
  } catch (err) {
    console.error("Route handler failed:", err);
    mountEl.innerHTML = `
      <section class="state">
        <span class="state-icon">⚠</span>
        Something went wrong rendering this view.<br>
        <small class="muted">${(err && err.message) || err}</small>
      </section>
    `;
  }
  syncNav(path);
}

function syncNav(path) {
  const links = document.querySelectorAll(".topnav .nav-link");
  // Map the route to the nav entry that should appear active.
  //   • /chip/:slug    → no nav entry highlighted; it's a drill-in page
  //                      reached from chip-cloud / lb-row, not part of
  //                      the rankings hub.
  //   • everything else falls through to its obvious top-level link.
  let active = "/";
  if (path.startsWith("/rankings"))      active = "/rankings";
  else if (path.startsWith("/compare"))  active = "/compare";
  else if (path.startsWith("/suites"))   active = "/suites";
  else if (path.startsWith("/chip"))     active = null;
  for (const a of links) {
    const href = a.getAttribute("href") || "";
    a.classList.toggle("active", active !== null && href === `#${active}`);
  }
}

export function start() {
  window.addEventListener("hashchange", dispatch);
  if (!location.hash) location.hash = "#/";
  else dispatch();
}

// ── Compare basket (shared state across views) ──
const basketListeners = new Set();

export function basketGet() { return Array.from(_compareBasket); }
export function basketHas(slug) { return _compareBasket.has(slug); }
export function basketToggle(slug) {
  if (_compareBasket.has(slug)) _compareBasket.delete(slug);
  else _compareBasket.add(slug);
  notifyBasket();
}
export function basketClear() { _compareBasket.clear(); notifyBasket(); }
export function basketOnChange(fn) { basketListeners.add(fn); return () => basketListeners.delete(fn); }
function notifyBasket() {
  for (const fn of basketListeners) {
    try { fn(basketGet()); } catch (e) { console.warn("basket listener failed", e); }
  }
}
