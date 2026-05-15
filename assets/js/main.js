// main.js — entry point.  Wires up router, data, and the top nav.

import { init as initData } from "./data.js";
import { mount, register, start } from "./router.js";
import { initModal } from "./modal.js";
import { render as renderHome }     from "./views/home.js";
import { render as renderRankings } from "./views/rankings.js";
import { render as renderChip }     from "./views/chip-detail.js";
import { render as renderCompare }  from "./views/compare.js";
import { render as renderSuites }   from "./views/suites.js";

function boot() {
  initData();
  initModal();

  const appEl = document.getElementById("view");
  mount(appEl);

  register("/",            renderHome);
  register("/rankings",    renderRankings);
  register("/chip/:slug",  renderChip);
  register("/compare",     renderCompare);
  register("/suites",      renderSuites);

  start();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
