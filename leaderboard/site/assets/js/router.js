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

import { parseHash } from "./utils.js";

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
  let active = "/";
  if (path.startsWith("/rankings")) active = "/rankings";
  else if (path.startsWith("/chip"))     active = "/rankings";
  else if (path.startsWith("/compare"))  active = "/compare";
  else if (path.startsWith("/suites"))   active = "/suites";
  for (const a of links) {
    const href = a.getAttribute("href") || "";
    a.classList.toggle("active", href === `#${active}`);
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
