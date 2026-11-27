// views/rankings.js — Developer-tool minimalist Leaderboard (Table + Landscape).
// Defaults to Closed (reference-only) division with one-click Open toggle.
// Compact header puts the ranked table directly above the fold.

import {
  SUITE_ORDER, SUITE_META, VENDOR_ORDER,
  SUITE_COLUMNS, formatMetric, rowsForSuite,
  baseSuiteId, suiteMetaFor, suiteColumnsFor, tracksForSuite,
} from "../data.js";
import {
  esc, fmtNum, fmtDate, chipHref, buildHash, parseHash,
  shortVersion, shortModel,
} from "../utils.js";
import {
  basketHas, basketToggle, basketGet, basketOnChange,
} from "../router.js";
import { suiteIcon, icon } from "../icons.js";
import { renderLandscape } from "../components/landscape.js";

const _i = (k, r) => (window._i ? window._i(k, r) : k);
const isZh = () => typeof window._lang === "function" && window._lang() === "cn";

const TAB_SHORT_EN = {
  suite_A: "Chat",
  suite_B: "70B TP",
  suite_C: "Quant",
  suite_D: "28K Ctx",
  suite_E: "Scaling",
  suite_F: "Edge",
  suite_G: "MoE",
  suite_H: "SWA",
};
const TAB_SHORT_ZH = {
  suite_A: "短对话",
  suite_B: "70B 多卡",
  suite_C: "量化",
  suite_D: "28K 长文",
  suite_E: "多卡扩展",
  suite_F: "边缘/消费",
  suite_G: "MoE",
  suite_H: "滑动窗口",
};

export function render({ el, query }) {
  const rawSuite = query.suite || SUITE_ORDER[0];
  const candidateBase = baseSuiteId(rawSuite);
  const baseSuite = SUITE_ORDER.includes(candidateBase) ? candidateBase : SUITE_ORDER[0];
  const tracks = tracksForSuite(baseSuite);
  const activeTrack = tracks.find((t) => t.suite_id === rawSuite) || tracks.find((t) => t.suite_id === baseSuite) || tracks[0];
  const suiteId = activeTrack ? activeTrack.suite_id : baseSuite;
  const meta = suiteMetaFor(suiteId) || SUITE_META[baseSuite];
  const cols = suiteColumnsFor(suiteId);
  const primary = cols.find((c) => c.primary) || cols[0];
  const viewMode = query.view === "landscape" ? "landscape" : "table";
  const zh = isZh();

  const vendorFilter = (query.vendor || "").trim();
  const frameworkFilter = (query.framework || "").trim();
  const precisionFilter = (query.precision || "").trim();
  const chipFilter = (query.chip || "").trim();
  const searchQuery = (query.q || "").trim();
  const searchNeedle = searchQuery.toLowerCase();

  // Closed (reference-only) by default; `?division=open` (or legacy runner_tier param) shows all.
  const openDivision = query.division === "open" || (query.runner_tier && query.runner_tier !== "reference");
  const closedOnly = !openDivision;
  // Grouping: default shows all serving recipes; `?group=chip` collapses to best recipe per chip.
  const bestPerChip = query.group === "chip";

  let sortKey = primary.key;
  let sortDir = primary.direction;
  if (query.sort) {
    const [k, d] = String(query.sort).split(":");
    if (cols.some((c) => c.key === k) || k === "date") {
      sortKey = k;
      sortDir = d === "asc" ? "asc" : "desc";
    }
  }

  const suiteAllRows = rowsForSuite(suiteId);
  const nonRefCount = suiteAllRows.filter((r) => (r.runner_tier || "reference") !== "reference").length;
  const divisionRows = closedOnly
    ? suiteAllRows.filter((r) => (r.runner_tier || "reference") === "reference")
    : suiteAllRows;

  const facets = buildFacets(divisionRows);

  const filtered = divisionRows.filter((r) => {
    if (vendorFilter && r.vendor !== vendorFilter) return false;
    if (frameworkFilter && r.framework !== frameworkFilter) return false;
    if (precisionFilter && r.precision !== precisionFilter) return false;
    if (chipFilter && r._chip_slug !== chipFilter) return false;
    if (searchNeedle) {
      const hay = `${r._chip_label || ""} ${r.vendor || ""} ${r.framework || ""} ${r.precision || ""} ${r.implementation_id || ""} ${r.submitted_by || ""}`.toLowerCase();
      if (!hay.includes(searchNeedle)) return false;
    }
    return true;
  });

  let sorted = [...filtered].sort((a, b) => safeCompare(a[sortKey], b[sortKey], sortDir));
  const totalFilteredCount = sorted.length;
  if (bestPerChip) {
    const seen = new Set();
    sorted = sorted.filter((r) => {
      const k = r._chip_slug || r.chip || r.run_id;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }

  // Hide empty secondary metric columns automatically
  const displayCols = cols.filter((c) => {
    if (c.primary || c.key === sortKey) return true;
    return sorted.length === 0 || sorted.some((r) => r[c.key] != null);
  });

  // Leader value for inline proportional bar on primary metric
  const primaryNums = sorted.map((r) => Number(r[primary.key])).filter((n) => Number.isFinite(n) && n > 0);
  const maxPrimary = primaryNums.length ? Math.max(...primaryNums) : 0;

  const filtersActive = !!(vendorFilter || frameworkFilter || precisionFilter || chipFilter || searchQuery || bestPerChip);
  const unitLabel = bestPerChip
    ? (zh ? "款芯片最优方案" : "chips (best recipe)")
    : (closedOnly ? (zh ? "套参考部署方案" : "reference recipes") : (zh ? "套部署方案" : "recipes"));
  const countText = !bestPerChip && totalFilteredCount === divisionRows.length
    ? `${fmtNum(sorted.length)} ${unitLabel}`
    : `${fmtNum(sorted.length)} ${zh ? "/" : "of"} ${fmtNum(divisionRows.length)} ${unitLabel}`;

  const suiteTitle = _i("suite." + baseSuite + ".title", meta.title);
  const activeModelBadge = activeTrack && tracks.length > 1 ? ` · ${activeTrack.label}` : "";
  const chipFocusLabel = chipFilter
    ? (filtered[0]?.chip || suiteAllRows.find((r) => r._chip_slug === chipFilter)?.chip || chipFilter)
    : "";

  const basketN = basketGet().length;

  el.innerHTML = `
    <header class="lb-head">
      <div class="lb-head-left">
        <div class="lb-title-row">
          <h1>${zh ? "推理部署方案榜" : "Serving Recipes"}</h1>
          <span class="lb-suite-pill-active mono">
            <span class="lb-dot" style="background:var(--suite-${esc(meta.letter)})"></span>
            Suite ${esc(meta.letter)} · ${esc(suiteTitle)}${esc(activeModelBadge)}
          </span>
        </div>
        <div class="lb-meta-row">
          <span class="lb-count mono tnum" id="lb-count">${esc(countText)}</span>
          <span class="lb-meta-sep">·</span>
          <span class="lb-meta-sub">${esc(_i("suite." + baseSuite + ".tagline", meta.tagline))}</span>
        </div>
      </div>

      <div class="lb-view-switch" role="tablist" aria-label="View mode">
        <button type="button" class="lb-vtab${viewMode === "table" ? " active" : ""}" data-view-mode="table" role="tab" aria-selected="${viewMode === "table"}">
          ${icon("table", 14)} <span>${zh ? "方案列表" : "Recipes"}</span>
        </button>
        <button type="button" class="lb-vtab${viewMode === "landscape" ? " active" : ""}" data-view-mode="landscape" role="tab" aria-selected="${viewMode === "landscape"}">
          ${icon("scatter", 14)} <span>${zh ? "全景图" : "Landscape"}</span>
        </button>
      </div>
    </header>

    <div class="lb-tabs-bar" role="tablist" aria-label="Workload suite">
      ${SUITE_ORDER.map((sid) => {
        const m = SUITE_META[sid];
        const active = sid === baseSuite && viewMode === "table";
        const shortName = (zh ? TAB_SHORT_ZH : TAB_SHORT_EN)[sid] || m.title;
        return `
          <button type="button"
                  class="lb-stab${active ? " active" : ""}"
                  data-suite="${esc(sid)}"
                  role="tab"
                  aria-selected="${active}"
                  title="Suite ${esc(m.letter)} · ${esc(_i("suite." + sid + ".title", m.title))}">
            <span class="lb-stab-ic">${suiteIcon(sid, 14)}</span>
            <span class="lb-stab-letter mono">${esc(m.letter)}</span>
            <span class="lb-stab-name">${esc(shortName)}</span>
          </button>
        `;
      }).join("")}
    </div>

    ${viewMode === "table" && tracks.length > 1 ? `
      <div class="lb-track-bar" role="tablist" aria-label="Model track">
        <span class="lb-track-label mono">${zh ? "模型代际：" : "MODEL TRACK:"}</span>
        ${tracks.map((t) => {
          const isCur = t.suite_id === suiteId;
          const statusTag = t.track_id !== "default" && t.status === "current"
            ? `<span class="lb-track-tag is-new mono">${zh ? "新版" : "2026"}</span>`
            : (t.status === "legacy" ? `<span class="lb-track-tag is-legacy mono">${zh ? "经典" : "Classic"}</span>` : "");
          return `
            <button type="button"
                    class="lb-track-pill${isCur ? " active" : ""}"
                    data-track-suite="${esc(t.suite_id)}"
                    role="tab"
                    aria-selected="${isCur}"
                    title="${esc(t.model_id || t.label)}">
              <span class="lb-track-name">${esc(t.label)}</span>
              ${statusTag}
              <span class="lb-track-cnt mono tnum">${fmtNum(t.count)}</span>
            </button>
          `;
        }).join("")}
      </div>
    ` : ""}

    <div class="lb-controls">
      <div class="lb-search-wrap">
        <span class="lb-search-ic">${icon("search", 14)}</span>
        <input id="rk-search"
               class="lb-search"
               type="search"
               value="${esc(searchQuery)}"
               placeholder="${zh ? "搜索芯片、框架或 @提交者…" : "Filter by chip, framework, or @submitter…"}"
               aria-label="Filter results">
      </div>

      <select class="lb-select" id="lb-filter-framework" aria-label="Framework">
        <option value="">${zh ? "全部推理框架 / 引擎" : "All engines"}</option>
        ${facets.framework.map((f) => `<option value="${esc(f)}"${f === frameworkFilter ? " selected" : ""}>${esc(f)}</option>`).join("")}
      </select>

      ${facets.precision.length > 1 ? `
        <select class="lb-select" id="lb-filter-precision" aria-label="Precision">
          <option value="">${zh ? "全部精度" : "All precisions"}</option>
          ${facets.precision.map((p) => `<option value="${esc(p)}"${p === precisionFilter ? " selected" : ""}>${esc(p)}</option>`).join("")}
        </select>
      ` : ""}

      <select class="lb-select" id="lb-filter-chip" aria-label="Hardware">
        <option value="">${zh ? "全部硬件" : "All hardware"}</option>
        ${facets.chip.map((c) => `<option value="${esc(c.slug)}"${c.slug === chipFilter ? " selected" : ""}>${esc(c.label)}</option>`).join("")}
      </select>

      <select class="lb-select" id="lb-filter-vendor" aria-label="Vendor">
        <option value="">${zh ? "全部厂商" : "All vendors"}</option>
        ${facets.vendor.map((v) => `<option value="${esc(v)}"${v === vendorFilter ? " selected" : ""}>${esc(v)}</option>`).join("")}
      </select>

      ${viewMode === "table" ? `
        <button type="button"
                class="lb-group-btn${bestPerChip ? " is-active" : ""}"
                id="lb-group-toggle"
                title="${zh ? "切换显示全部部署方案，或每款芯片仅展示最强方案" : "Toggle between all serving recipes or best recipe per hardware"}">
          ${icon("layers", 13)}
          <span>${bestPerChip
            ? (zh ? "每卡最优" : "Best / Chip")
            : (zh ? "全部方案" : "All Recipes")}</span>
        </button>
      ` : ""}

      <button type="button"
              class="lb-division-btn${closedOnly ? " is-closed" : " is-open"}"
              id="lb-division-toggle"
              title="${zh ? "Closed 仅含开箱即用参考栈；Open 含优化与实验栈" : "Closed shows out-of-the-box reference stacks; Open includes optimized & experimental forks"}">
        ${closedOnly ? icon("lock", 13) : icon("flask", 13)}
        <span>${closedOnly
          ? (zh ? "仅参考栈 (Closed)" : "Closed (Ref)")
          : (zh ? "全部栈 (Open)" : "Open (All)")}</span>
        ${closedOnly && nonRefCount > 0 ? `<span class="lb-div-badge mono">+${nonRefCount}</span>` : ""}
      </button>

      ${viewMode === "table" ? `
        <button type="button" class="lb-info-btn" id="lb-protocol-toggle" aria-expanded="false">
          ${icon("info", 14)} <span>${zh ? "协议" : "Protocol"}</span>
        </button>
      ` : ""}

      ${filtersActive ? `
        <button type="button" class="lb-clear-btn" data-clear-all="1">
          ${icon("x", 13)} ${zh ? "重置" : "Reset"}
        </button>
      ` : ""}
    </div>

    ${renderProtocolDrawer(suiteId, meta)}

    ${chipFilter ? `
      <div class="lb-chip-banner">
        <span>${zh ? "当前仅展示芯片：" : "Filtered to chip:"} <strong>${esc(chipFocusLabel)}</strong></span>
        <span class="lb-chip-banner-actions">
          <a href="#/chip/${esc(chipFilter)}">${zh ? "查看芯片档案" : "Chip profile"} →</a>
          <a href="${esc(rebuildHash(suiteId, query, { chip: undefined }))}">${zh ? "显示全部" : "Show all"}</a>
        </span>
      </div>
    ` : ""}

    <section class="lb-body">
      ${viewMode === "landscape"
        ? `<div id="lb-landscape-mount"></div>`
        : (sorted.length === 0
            ? renderEmpty(meta, filtersActive, nonRefCount, closedOnly)
            : renderTable(suiteId, sorted, displayCols, sortKey, sortDir, primary, maxPrimary, openDivision))}
    </section>

    <div class="lb-float-basket${basketN ? " show" : ""}" id="lb-float-basket">
      <span class="lb-basket-count mono tnum"><strong>${fmtNum(basketN)}</strong> ${zh ? "项已选" : (basketN === 1 ? "selected" : "selected")}</span>
      <a class="lb-basket-go" href="#/compare">${zh ? "开始对比" : "Compare"} ${icon("arrowRight", 14)}</a>
      <button type="button" class="lb-basket-clear" data-basket-clear="1" aria-label="Clear selection">${icon("x", 14)}</button>
    </div>
  `;

  if (viewMode === "landscape") {
    const mount = el.querySelector("#lb-landscape-mount");
    const metric = query.ls_metric || "offline";
    renderLandscape(mount, { closedOnly, vendor: vendorFilter, framework: frameworkFilter, metric });
    mount.addEventListener("click", (ev) => {
      const btn = ev.target.closest("[data-ls-metric]");
      if (!btn) return;
      location.hash = rebuildHash(suiteId, query, { view: "landscape", ls_metric: btn.dataset.lsMetric });
    });
  }

  bindInteractions(el);

  if (!el.__rkBasketAttached) {
    basketOnChange(() => {
      const tbody = el.querySelector(".lb-table tbody");
      if (tbody) {
        for (const tr of tbody.querySelectorAll("tr[data-run-id]")) {
          const inB = basketHas(tr.dataset.runId);
          tr.classList.toggle("in-basket", inB);
          const cb = tr.querySelector(".compare-checkbox");
          if (cb) cb.checked = inB;
        }
      }
      const bar = el.querySelector("#lb-float-basket");
      if (bar) {
        const n = basketGet().length;
        bar.classList.toggle("show", n > 0);
        const cnt = bar.querySelector(".lb-basket-count");
        if (cnt) cnt.innerHTML = `<strong>${fmtNum(n)}</strong> ${isZh() ? "项已选" : "selected"}`;
      }
    });
    el.__rkBasketAttached = true;
  }
}

function renderProtocolDrawer(suiteId, meta) {
  const wl = meta.workload || {};
  const model = wl.model ? shortModel(wl.model) : "—";
  const tokens = wl.inputTokens && wl.outputTokens ? `${wl.inputTokens} → ${wl.outputTokens} tok` : "";
  const bn = _i(`rankings.bn.${suiteId}`, meta.tagline);
  const takeaway = _i(`suite.${suiteId}.finding.headline`, "");
  const zh = isZh();

  return `
    <aside class="lb-protocol-drawer" id="lb-protocol-drawer" hidden>
      <div class="lb-proto-grid">
        <div>
          <div class="lb-proto-lbl mono">${zh ? "隔离瓶颈" : "BOTTLENECK"}</div>
          <div class="lb-proto-val">${esc(bn)}</div>
        </div>
        <div>
          <div class="lb-proto-lbl mono">${zh ? "固定负载" : "WORKLOAD"}</div>
          <div class="lb-proto-val mono"><strong>${esc(model)}</strong>${wl.precision ? ` · ${esc(wl.precision)}` : ""}${tokens ? ` · ${esc(tokens)}` : ""}</div>
        </div>
        <div>
          <div class="lb-proto-lbl mono">${zh ? "关键发现" : "KEY TAKEAWAY"}</div>
          <div class="lb-proto-val">${esc(takeaway || meta.tagline)} <a href="#/suites">${zh ? "完整协议 →" : "Full protocol →"}</a></div>
        </div>
      </div>
    </aside>
  `;
}

function renderTable(suiteId, rows, cols, sortKey, sortDir, primary, maxPrimary, openDivision) {
  const zh = isZh();
  return `
    <div class="lb-table-wrap">
      <table class="lb-table data-table">
        <thead>
          <tr>
            <th class="col-compare" scope="col" title="${zh ? "勾选对比" : "Select to compare"}"><span class="visually-hidden">Compare</span></th>
            <th class="col-rank" scope="col">#</th>
            <th class="col-recipe" scope="col">${zh ? "部署方案 (引擎 · 精度 · 配置)" : "Serving Recipe"}</th>
            <th class="col-hw" scope="col">${zh ? "目标硬件" : "Hardware"}</th>
            ${cols.map((c) => `
              <th class="col-metric sortable${c.primary ? " col-primary" : ""}${sortKey === c.key ? " is-sort" : ""}"
                  data-sort-key="${esc(c.key)}"
                  data-sort-dir-default="${esc(c.direction)}"
                  aria-sort="${sortKey === c.key ? (sortDir === "asc" ? "ascending" : "descending") : "none"}"
                  scope="col">
                <span class="th-label">${esc(c.label)}</span>
                ${c.unit ? `<span class="th-unit mono">${esc(c.unit)}</span>` : ""}
                <span class="th-sort-arrow">${sortKey === c.key ? (sortDir === "asc" ? "↑" : "↓") : ""}</span>
              </th>
            `).join("")}
            <th class="col-trust" scope="col" title="${zh ? "±5% 独立复现验证" : "Independent ±5% verification"}">${zh ? "验证" : "Trust"}</th>
            <th class="col-deploy" scope="col">${zh ? "部署" : "Deploy"}</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((r, i) => renderRow(r, cols, sortKey, primary, maxPrimary, i + 1, openDivision)).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderRow(row, cols, sortKey, primary, maxPrimary, rank, openDivision) {
  const runId = row.run_id || row.submission || "";
  const inBasket = basketHas(runId);
  const ver = shortVersion(row.framework_version);
  const fwName = row.framework || "vLLM";
  const tp = Number(row.detail?.run_tp || row.chip_count || 1) || 1;
  const modelName = row.detail?.model_id ? shortModel(row.detail.model_id) : "";
  const rTier = row.runner_tier || "reference";
  const isVerified = row.tier === "verified";
  const ec = row.detail?.run_extra_config;
  const tunedCount = ec && typeof ec === "object" ? Object.keys(ec).length : 0;
  const zh = isZh();

  const hwSub = [
    row.vendor,
    row.memory_gb ? `${fmtNum(row.memory_gb)}GB` : "",
  ].filter(Boolean).join(" · ");

  return `
    <tr data-run-id="${esc(runId)}"
        data-chip-slug="${esc(row._chip_slug)}"
        data-open-run="${esc(runId)}"
        tabindex="0"
        class="${inBasket ? "in-basket" : ""}">
      <td class="col-compare" title="${zh ? "加入对比" : "Toggle compare"}">
        <label class="compare-cell">
          <input type="checkbox" class="compare-checkbox" tabindex="-1" ${inBasket ? "checked" : ""}>
        </label>
      </td>
      <td class="col-rank mono tnum${rank <= 3 ? ` is-top-${rank}` : ""}">${rank}</td>
      <td class="col-recipe">
        <div class="lb-recipe-cell">
          <span class="lb-recipe-line">
            <span class="lb-recipe-engine">${esc(fwName)}</span>
            ${ver ? `<span class="lb-recipe-ver mono">${esc(ver)}</span>` : ""}
            ${row.precision ? `<span class="lb-recipe-pill mono">${esc(row.precision)}</span>` : ""}
            ${tp > 1 ? `<span class="lb-recipe-pill mono">TP${tp}</span>` : ""}
            ${tunedCount > 0 ? `<span class="lb-tier-tag is-tuned" title="${zh ? "含显式调优参数" : "Includes explicit config overrides"}">+${tunedCount} cfg</span>` : ""}
            ${openDivision && rTier !== "reference" ? `<span class="lb-tier-tag is-${esc(rTier)}">${esc(rTier === "experimental" ? "exp" : "opt")}</span>` : ""}
          </span>
          <span class="lb-recipe-sub mono">
            ${modelName ? esc(modelName) : esc(row.implementation_id || "—")}
            ${row.date ? ` · ${esc(fmtDate(row.date))}` : ""}
          </span>
        </div>
      </td>
      <td class="col-hw">
        <a class="lb-chip-link" href="${chipHref(row)}">
          <span class="lb-hw-name">${esc(row._chip_label)}</span>
          ${hwSub ? `<span class="lb-hw-sub mono">${esc(hwSub)}</span>` : ""}
        </a>
      </td>
      ${cols.map((c) => renderMetricCell(row, c, sortKey, primary, maxPrimary)).join("")}
      <td class="col-trust">
        ${isVerified
          ? `<span class="trust-verified" title="${zh ? "已通过 ±5% 独立复现验证" : "Independently verified within ±5%"}">${icon("shieldCheck", 14)}<span>${zh ? "已验证" : "Verified"}</span></span>`
          : `<span class="trust-community" title="${zh ? "CI 已通过，等待独立复现" : "CI validated · community submission"}">—</span>`}
      </td>
      <td class="col-deploy">
        <button type="button"
                class="lb-deploy-btn"
                data-open-run="${esc(runId)}"
                data-open-tab="deploy"
                title="${zh ? "一键复制安装与部署命令" : "Copy setup & --serve deploy recipe"}">
          ${icon("terminal", 12)} <span>${zh ? "部署" : "Deploy"}</span>
        </button>
      </td>
    </tr>
  `;
}

function renderMetricCell(row, col, sortKey, primary, maxPrimary) {
  const raw = row[col.key];
  const isSort = sortKey === col.key;

  // Special semantic rendering for Online SLA QPS
  if (col.key === "online_max_qps") {
    if (raw == null) {
      return `<td class="col-metric${isSort ? " is-sort" : ""} mono tnum"><span class="metric-empty">—</span></td>`;
    }
    const n = Number(raw);
    if (n === 0) {
      return `<td class="col-metric${isSort ? " is-sort" : ""} mono"><span class="sla-fail" title="Failed 500ms p99 TTFT SLA under Poisson load">${isZh() ? "未达 SLA" : "SLA fail"}</span></td>`;
    }
    const disp = n >= 10 ? Math.round(n).toString() : n.toFixed(n % 1 === 0 ? 0 : 1);
    return `<td class="col-metric${isSort ? " is-sort" : ""} mono tnum"><span class="metric-val">${esc(disp)}</span></td>`;
  }

  const formatted = formatMetric(raw, col);
  if (formatted === null) {
    return `<td class="col-metric${col.primary ? " col-primary" : ""}${isSort ? " is-sort" : ""} mono tnum"><span class="metric-empty">—</span></td>`;
  }

  if (col.primary && !col.textual && maxPrimary > 0 && Number.isFinite(Number(raw)) && Number(raw) > 0) {
    const pct = Math.max(4, Math.min(100, Math.round((Number(raw) / maxPrimary) * 100)));
    return `
      <td class="col-metric col-primary${isSort ? " is-sort" : ""} mono tnum">
        <div class="metric-bar-cell">
          <span class="metric-val">${esc(formatted)}</span>
          <div class="metric-bar-track"><div class="metric-bar-fill" style="width:${pct}%"></div></div>
        </div>
      </td>
    `;
  }

  return `<td class="col-metric${col.primary ? " col-primary" : ""}${isSort ? " is-sort" : ""} mono tnum"><span class="metric-val">${esc(formatted)}</span></td>`;
}

function renderEmpty(meta, filtersActive, nonRefCount, closedOnly) {
  const zh = isZh();
  return `
    <div class="rk-empty">
      <p>${filtersActive
        ? (zh ? `Suite ${esc(meta.letter)} 下没有匹配当前筛选条件的结果。` : `No results match the current filters in Suite ${esc(meta.letter)}.`)
        : (zh ? `Suite ${esc(meta.letter)} 暂无参考栈提交。` : `No reference submissions yet in Suite ${esc(meta.letter)}.`)}</p>
      <div class="rk-empty-actions">
        ${filtersActive ? `<button type="button" class="btn-outline" data-clear-all="1">${zh ? "清除筛选" : "Reset filters"}</button>` : ""}
        ${closedOnly && nonRefCount > 0 ? `<button type="button" class="btn-solid" id="lb-open-empty">${zh ? `查看 ${nonRefCount} 条 Open 栈结果` : `Show ${nonRefCount} Open-division runs`}</button>` : ""}
      </div>
    </div>
  `;
}

function bindInteractions(el) {
  if (el.__rkBound) return;
  el.__rkBound = true;

  el.addEventListener("click", (ev) => {
    if (!location.hash.startsWith("#/rankings")) return;
    const { params: query } = parseHash(location.hash);
    const rawS = query.suite || SUITE_ORDER[0];
    const baseS = SUITE_ORDER.includes(baseSuiteId(rawS)) ? baseSuiteId(rawS) : SUITE_ORDER[0];
    const suiteId = tracksForSuite(baseS).some((tr) => tr.suite_id === rawS) ? rawS : baseS;
    const t = ev.target;

    const trackBtn = t.closest("[data-track-suite]");
    if (trackBtn) {
      ev.preventDefault();
      const targetSuite = trackBtn.dataset.trackSuite;
      if (targetSuite) {
        location.hash = buildHash("/rankings", {
          ...suiteUrlParam(targetSuite),
          division: query.division === "open" ? "open" : undefined,
        });
      }
      return;
    }

    if (t.closest("[data-basket-clear]")) {
      ev.preventDefault();
      for (const id of basketGet()) basketToggle(id);
      return;
    }

    if (t.closest("[data-clear-all]")) {
      ev.preventDefault();
      location.hash = buildHash("/rankings", suiteUrlParam(suiteId));
      return;
    }

    const vtab = t.closest("[data-view-mode]");
    if (vtab) {
      ev.preventDefault();
      const mode = vtab.dataset.viewMode;
      location.hash = rebuildHash(suiteId, query, { view: mode === "table" ? undefined : mode });
      return;
    }

    const stab = t.closest(".lb-stab");
    if (stab) {
      ev.preventDefault();
      const sid = stab.dataset.suite;
      if (sid) {
        location.hash = buildHash("/rankings", {
          ...suiteUrlParam(sid),
          division: query.division === "open" ? "open" : undefined,
        });
      }
      return;
    }

    if (t.closest("#lb-division-toggle") || t.closest("#lb-open-empty")) {
      ev.preventDefault();
      const isOpen = query.division === "open" || (query.runner_tier && query.runner_tier !== "reference");
      location.hash = rebuildHash(suiteId, query, {
        division: isOpen ? undefined : "open",
        runner_tier: undefined,
      });
      return;
    }

    if (t.closest("#lb-group-toggle")) {
      ev.preventDefault();
      location.hash = rebuildHash(suiteId, query, {
        group: query.group === "chip" ? undefined : "chip",
      });
      return;
    }

    const protoBtn = t.closest("#lb-protocol-toggle");
    if (protoBtn) {
      ev.preventDefault();
      const drawer = el.querySelector("#lb-protocol-drawer");
      if (drawer) {
        const nextHidden = !drawer.hidden;
        drawer.hidden = nextHidden;
        protoBtn.setAttribute("aria-expanded", String(!nextHidden));
        protoBtn.classList.toggle("active", !nextHidden);
      }
      return;
    }

    const compareCell = t.closest(".col-compare");
    if (compareCell) {
      ev.preventDefault();
      ev.stopPropagation();
      const tr = compareCell.closest("tr");
      if (tr?.dataset.runId) basketToggle(tr.dataset.runId);
      return;
    }

    const th = t.closest("th.sortable");
    if (th) {
      ev.preventDefault();
      const key = th.dataset.sortKey;
      const defaultDir = th.dataset.sortDirDefault;
      const cur = String(query.sort || "");
      let nextDir = defaultDir;
      if (cur.startsWith(key + ":")) {
        nextDir = cur.split(":")[1] === "desc" ? "asc" : "desc";
      }
      location.hash = rebuildHash(suiteId, query, { sort: `${key}:${nextDir}` });
      return;
    }
  });

  el.addEventListener("change", (ev) => {
    if (!location.hash.startsWith("#/rankings")) return;
    const id = ev.target?.id;
    if (!id || !id.startsWith("lb-filter-")) return;
    const { params: query } = parseHash(location.hash);
    const rawS = query.suite || SUITE_ORDER[0];
    const baseS = SUITE_ORDER.includes(baseSuiteId(rawS)) ? baseSuiteId(rawS) : SUITE_ORDER[0];
    const suiteId = tracksForSuite(baseS).some((tr) => tr.suite_id === rawS) ? rawS : baseS;
    const key = id.replace("lb-filter-", "");
    location.hash = rebuildHash(suiteId, query, { [key]: ev.target.value || undefined });
  });

  el.addEventListener("input", (ev) => {
    const inp = ev.target?.id === "rk-search" ? ev.target : null;
    if (!inp) return;
    const needle = inp.value.trim().toLowerCase();
    const tbody = el.querySelector(".lb-table tbody");
    if (!tbody) return;
    let visible = 0;
    let total = 0;
    for (const tr of tbody.querySelectorAll("tr[data-run-id]")) {
      total += 1;
      const match = !needle || tr.textContent.toLowerCase().includes(needle);
      tr.style.display = match ? "" : "none";
      if (match) visible += 1;
    }
    const cnt = el.querySelector("#lb-count");
    if (cnt) {
      const zh = isZh();
      cnt.textContent = visible === total
        ? `${fmtNum(total)} ${zh ? "条结果" : "runs"}`
        : `${fmtNum(visible)} ${zh ? "/" : "of"} ${fmtNum(total)} ${zh ? "条结果" : "runs"}`;
    }
  });

  el.addEventListener("keydown", (ev) => {
    const inp = ev.target?.id === "rk-search" ? ev.target : null;
    if (!inp || ev.key !== "Enter") return;
    ev.preventDefault();
    const { params: query } = parseHash(location.hash);
    const rawS = query.suite || SUITE_ORDER[0];
    const baseS = SUITE_ORDER.includes(baseSuiteId(rawS)) ? baseSuiteId(rawS) : SUITE_ORDER[0];
    const suiteId = tracksForSuite(baseS).some((tr) => tr.suite_id === rawS) ? rawS : baseS;
    location.hash = rebuildHash(suiteId, query, { q: inp.value.trim() || undefined });
  });
}

function buildFacets(rows) {
  const vendor = new Set();
  const precision = new Set();
  const framework = new Set();
  const chipMap = new Map();
  for (const r of rows) {
    if (r.vendor) vendor.add(r.vendor);
    if (r.precision) precision.add(r.precision);
    if (r.framework) framework.add(r.framework);
    if (r._chip_slug && !chipMap.has(r._chip_slug)) {
      chipMap.set(r._chip_slug, r.chip || r._chip_label || r._chip_slug);
    }
  }
  const sortedVendor = VENDOR_ORDER.filter((v) => vendor.has(v))
    .concat(Array.from(vendor).filter((v) => !VENDOR_ORDER.includes(v)).sort());
  const chip = Array.from(chipMap.entries())
    .map(([slug, label]) => ({ slug, label }))
    .sort((a, b) => a.label.localeCompare(b.label));
  return {
    chip,
    vendor: sortedVendor,
    precision: Array.from(precision).sort(),
    framework: Array.from(framework).sort(),
  };
}

function safeCompare(a, b, dir) {
  const aM = a === null || a === undefined || (typeof a === "number" && Number.isNaN(a));
  const bM = b === null || b === undefined || (typeof b === "number" && Number.isNaN(b));
  if (aM && bM) return 0;
  if (aM) return 1;
  if (bM) return -1;
  if (typeof a === "string" || typeof b === "string") {
    const c = String(a).localeCompare(String(b));
    return dir === "asc" ? c : -c;
  }
  return dir === "asc" ? a - b : b - a;
}

function suiteUrlParam(suiteId) {
  return suiteId === SUITE_ORDER[0] ? {} : { suite: suiteId };
}

function rebuildHash(suiteId, currentQuery, updates) {
  const merged = { ...currentQuery, ...updates, ...suiteUrlParam(suiteId) };
  const out = {};
  for (const [k, v] of Object.entries(merged)) {
    if (v === "" || v === null || v === undefined) continue;
    out[k] = v;
  }
  return buildHash("/rankings", out);
}
