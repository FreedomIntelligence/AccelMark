// command-palette.js — Global ⌘K / Ctrl+K quick-jump palette for chips, suites, and pages.

import { SUITE_ORDER, SUITE_META, chipCloudData, allRows } from "./data.js";
import { esc, shortVersion } from "./utils.js";
import { suiteIcon, icon } from "./icons.js";

const _i = (k, r) => (window._i ? window._i(k, r) : k);

let _overlay = null;
let _input = null;
let _list = null;
let _items = [];
let _activeIdx = 0;

function buildIndex() {
  const isZh = typeof window._lang === "function" && window._lang() === "cn";
  const pages = [
    { group: isZh ? "页面" : "Pages", label: _i("nav.results", "Leaderboard"), sub: "#/rankings", href: "#/rankings", iconHtml: icon("table", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("nav.findings", "Key Findings"), sub: "#/findings", href: "#/findings", iconHtml: icon("scatter", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("nav.compare", "Compare"), sub: "#/compare", href: "#/compare", iconHtml: icon("arrowRight", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("nav.suites", "Methodology"), sub: "#/suites", href: "#/suites", iconHtml: icon("book", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("nav.agent", "Agent Track"), sub: "#/agent", href: "#/agent", iconHtml: icon("terminal", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("nav.submit", "Submit a result"), sub: "#/submit", href: "#/submit", iconHtml: icon("gitPullRequest", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("nav.wanted", "Wanted hardware"), sub: "#/wanted", href: "#/wanted", iconHtml: icon("cpu", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("nav.reproduce", "Reproduction quests"), sub: "#/reproduce", href: "#/reproduce", iconHtml: icon("shieldCheck", 15) },
    { group: isZh ? "页面" : "Pages", label: _i("footer.cite", "Cite dataset"), sub: "#/citation", href: "#/citation", iconHtml: icon("copy", 15) },
  ];

  const suites = SUITE_ORDER.map((sid) => {
    const m = SUITE_META[sid];
    const title = _i("suite." + sid + ".title", m.title);
    return {
      group: isZh ? "测试套件 (A–H)" : "Workload Suites",
      label: `Suite ${m.letter} · ${title}`,
      sub: m.workload?.model ? m.workload.model.split("/").pop() : m.tagline,
      href: sid === SUITE_ORDER[0] ? "#/rankings" : `#/rankings?suite=${sid}`,
      iconHtml: suiteIcon(sid, 15),
      hay: `suite ${m.letter} ${sid} ${title} ${m.title} ${m.tagline} ${m.workload?.model || ""}`.toLowerCase(),
    };
  });

  const recipeRows = allRows()
    .filter((r) => !r.flagged)
    .slice(0, 60)
    .map((r) => {
      const m = SUITE_META[r.suite];
      const fw = [r.framework, shortVersion(r.framework_version)].filter(Boolean).join(" ");
      const prec = r.precision ? ` · ${r.precision}` : "";
      const tp = (r.detail?.run_tp || r.chip_count || 1) > 1 ? ` · TP${r.detail?.run_tp || r.chip_count}` : "";
      const hw = r._chip_label || r.chip || "";
      return {
        group: isZh ? "部署方案 (Recipes)" : "Serving Recipes",
        label: `${fw}${prec}${tp}`,
        sub: `on ${hw}${m ? ` · Suite ${m.letter}` : ""}`,
        href: `#/rankings?suite=${encodeURIComponent(r.suite)}&run=${encodeURIComponent(r.run_id)}&tab=deploy`,
        iconHtml: icon("terminal", 14),
        hay: `${fw} ${r.precision || ""} ${hw} ${r.vendor || ""} ${r.suite || ""} ${r.implementation_id || ""} ${r.detail?.model_id || ""}`.toLowerCase(),
      };
    });

  const chips = chipCloudData().map((c) => ({
    group: isZh ? "按硬件查方案" : "Hardware Playbooks",
    label: c.label,
    sub: `${c.vendor} · ${c.count} ${isZh ? "套方案" : (c.count === 1 ? "recipe" : "recipes")}`,
    href: `#/chip/${c.slug}`,
    iconHtml: icon("cpu", 15),
    hay: `${c.label} ${c.vendor} ${c.slug}`.toLowerCase(),
  }));

  return [...recipeRows, ...chips, ...suites, ...pages].map((item) => ({
    ...item,
    hay: item.hay || `${item.label} ${item.sub}`.toLowerCase(),
  }));
}

function renderList(query) {
  const q = (query || "").trim().toLowerCase();
  const all = buildIndex();
  const matched = q
    ? all.filter((x) => q.split(/\s+/).every((tok) => x.hay.includes(tok))).slice(0, 14)
    : all.slice(0, 12);

  _items = matched;
  _activeIdx = 0;

  if (!matched.length) {
    _list.innerHTML = `<div class="cmdk-empty">No matches for “${esc(query)}”</div>`;
    return;
  }

  let lastGroup = "";
  const html = [];
  matched.forEach((item, i) => {
    if (item.group !== lastGroup) {
      lastGroup = item.group;
      html.push(`<div class="cmdk-group">${esc(item.group)}</div>`);
    }
    html.push(`
      <a class="cmdk-item${i === 0 ? " is-active" : ""}" href="${esc(item.href)}" data-cmdk-idx="${i}">
        <span class="cmdk-ic">${item.iconHtml}</span>
        <span class="cmdk-label">${esc(item.label)}</span>
        <span class="cmdk-sub mono">${esc(item.sub)}</span>
      </a>
    `);
  });
  _list.innerHTML = html.join("");
}

function syncActive() {
  if (!_list) return;
  const rows = _list.querySelectorAll(".cmdk-item");
  rows.forEach((r, i) => {
    r.classList.toggle("is-active", i === _activeIdx);
    if (i === _activeIdx) r.scrollIntoView({ block: "nearest" });
  });
}

export function openCommandPalette() {
  if (!_overlay) initCommandPalette();
  _overlay.hidden = false;
  document.body.style.overflow = "hidden";
  _input.value = "";
  renderList("");
  requestAnimationFrame(() => _input.focus());
}

export function closeCommandPalette() {
  if (!_overlay || _overlay.hidden) return;
  _overlay.hidden = true;
  document.body.style.overflow = "";
}

export function initCommandPalette() {
  if (_overlay) return;
  _overlay = document.createElement("div");
  _overlay.className = "cmdk-backdrop";
  _overlay.hidden = true;
  _overlay.innerHTML = `
    <div class="cmdk-dialog" role="dialog" aria-modal="true" aria-label="Quick search">
      <div class="cmdk-head">
        <span class="cmdk-search-ic">${icon("search", 16)}</span>
        <input class="cmdk-input" type="search" placeholder="Jump to chip (H200, 5090, 910B2), suite, or page…" autocomplete="off" spellcheck="false">
        <kbd class="cmdk-kbd">ESC</kbd>
      </div>
      <div class="cmdk-list"></div>
      <div class="cmdk-foot">
        <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>
        <span><kbd>↵</kbd> open</span>
        <span><kbd>ESC</kbd> close</span>
      </div>
    </div>
  `;
  document.body.appendChild(_overlay);
  _input = _overlay.querySelector(".cmdk-input");
  _list = _overlay.querySelector(".cmdk-list");

  _overlay.addEventListener("click", (e) => {
    if (e.target === _overlay) { closeCommandPalette(); return; }
    const a = e.target.closest(".cmdk-item");
    if (a) closeCommandPalette();
  });

  _input.addEventListener("input", () => renderList(_input.value));

  _input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); closeCommandPalette(); return; }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (_items.length) { _activeIdx = (_activeIdx + 1) % _items.length; syncActive(); }
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (_items.length) { _activeIdx = (_activeIdx - 1 + _items.length) % _items.length; syncActive(); }
      return;
    }
    if (e.key === "Enter" && _items[_activeIdx]) {
      e.preventDefault();
      const href = _items[_activeIdx].href;
      closeCommandPalette();
      location.hash = href;
    }
  });

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (_overlay.hidden) openCommandPalette();
      else closeCommandPalette();
    } else if (e.key === "Escape" && _overlay && !_overlay.hidden) {
      closeCommandPalette();
    }
  });

  const trigger = document.getElementById("cmdk-trigger");
  if (trigger) trigger.addEventListener("click", openCommandPalette);
}
