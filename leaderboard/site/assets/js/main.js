// main.js — entry point.  Wires up router, data, and the top nav.

import { init as initData } from "./data.js";
import {
  mount, register, start, dispatch,
  basketGet, basketOnChange, basketClear,
} from "./router.js";
import { render as renderHome }     from "./views/home.js";
import { render as renderRankings } from "./views/rankings.js";
import { render as renderChip }     from "./views/chip-detail.js";
import { render as renderCompare }  from "./views/compare.js";
import { render as renderSuites }   from "./views/suites.js";

function boot() {
  initData();

  const appEl = document.getElementById("view");
  mount(appEl);

  register("/",            renderHome);
  register("/rankings",    renderRankings);
  register("/chip/:slug",  renderChip);
  register("/compare",     renderCompare);
  register("/suites",      renderSuites);

  setupCompareBadge();
  start();
}

function setupCompareBadge() {
  const pill = document.querySelector(".compare-pill");
  if (!pill) return;
  const sync = () => {
    const n = basketGet().length;
    pill.classList.toggle("show", n > 0);
    pill.querySelector(".count").textContent = String(n);
  };
  pill.addEventListener("click", (e) => {
    // The pill itself is a link — let the browser handle navigation;
    // we only intercept the "x" clear button if present.
    if (e.target.matches(".compare-clear")) {
      e.preventDefault();
      basketClear();
    }
  });
  basketOnChange(sync);
  sync();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
