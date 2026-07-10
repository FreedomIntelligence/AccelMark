// main.js — entry point.  Wires up router, data, and the top nav.

import { init as initData } from "./data.js";
import { mount, register, start } from "./router.js";
import { initModal } from "./modal.js";
import { render as renderHome }     from "./views/home.js";
import { render as renderRankings } from "./views/rankings.js";
import { render as renderChip }     from "./views/chip-detail.js";
import { render as renderCompare }  from "./views/compare.js";
import { render as renderSuites }        from "./views/suites.js";
import { render as renderSubmit }   from "./views/submit.js";
import { render as renderWanted }   from "./views/wanted.js";
import { render as renderCitation } from "./views/citation.js";
import { render as renderContributors } from "./views/contributors.js";
import { render as renderContributor }  from "./views/contributor.js";
import { render as renderReproduce }    from "./views/reproduce.js";

function boot() {
  initData();
  initModal();

  const appEl = document.getElementById("view");
  mount(appEl);

  register("/",             renderHome);
  register("/rankings",     renderRankings);
  register("/chip/:slug",   renderChip);
  register("/compare",      renderCompare);
  register("/suites",       renderSuites);
  register("/submit",       renderSubmit);
  register("/wanted",       renderWanted);
  register("/citation",     renderCitation);
  register("/contributors", renderContributors);
  register("/contributor/:handle", renderContributor);
  register("/reproduce",     renderReproduce);

  start();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
