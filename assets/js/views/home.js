// views/home.js — Developer-tool minimalist Home page (~3.5 screens).
// Screen 1: Left-aligned thesis + interactive live Rank-Shift trajectory chart.
// Screen 2: 8 compact Workload tiles (Lucide 2-4 stroke icons, zero rainbow banners, Closed #1 leader).
// Screen 3: Top 3 live Key Findings + deep link to #/findings.
// Screen 4: 4-pillar Trust Architecture + 7-Ecosystem strip + Contribute banner.

import {
  SUITE_ORDER, SUITE_META, SUITE_COLUMNS,
  rowsForSuite, allRows, formatMetric, summary,
} from "../data.js";
import { computeInsights } from "../insights.js";
import { esc, fmtNum, buildHash, shortVersion, shortModel, chipHref } from "../utils.js";
import { suiteIcon, icon } from "../icons.js";

const _i = (k, r) => (window._i ? window._i(k, r) : k);
const isZh = () => typeof window._lang === "function" && window._lang() === "cn";

const SHORT_SUITE_NAMES_EN = {
  suite_A: "Single-Chip Chat",
  suite_B: "70B Multi-Chip TP",
  suite_C: "Quantization",
  suite_D: "28K Long Context",
  suite_E: "TP Scaling",
  suite_F: "Edge & Consumer",
  suite_G: "Sparse MoE",
  suite_H: "Sliding Window",
};

const SHORT_SUITE_NAMES_ZH = {
  suite_A: "单卡对话吞吐",
  suite_B: "70B 多卡并行",
  suite_C: "量化效率",
  suite_D: "28K 长上下文",
  suite_E: "多卡扩展效率",
  suite_F: "边缘与消费级",
  suite_G: "稀疏 MoE",
  suite_H: "滑动窗口 SWA",
};

const ECOSYSTEMS = [
  "NVIDIA CUDA",
  "AMD ROCm",
  "Huawei Ascend",
  "Apple Silicon",
  "Google TPU",
  "Moore Threads",
  "Iluvatar CoreX",
];

function computeRankShiftData(rows) {
  const ref = rows.filter(
    (r) => (r.runner_tier || "reference") === "reference" && !r.flagged && (r.chip_count || 1) === 1 && r.framework === "vLLM"
  );
  const byChip = new Map();
  for (const r of ref) {
    if (!r.chip) continue;
    if (!byChip.has(r.chip)) byChip.set(r.chip, { chip: r.chip, slug: r._chip_slug });
    const cur = byChip.get(r.chip);
    if (r.suite === "suite_A" && Number.isFinite(r.offline_throughput)) {
      if (cur.A == null || r.offline_throughput > cur.A) {
        cur.A = r.offline_throughput;
        cur.QPS = Number.isFinite(r.online_max_qps) ? r.online_max_qps : 0;
        cur.runA = r.run_id || "";
      }
    } else if (r.suite === "suite_D" && Number.isFinite(r.offline_throughput)) {
      if (cur.D == null || r.offline_throughput > cur.D) {
        cur.D = r.offline_throughput;
        cur.runD = r.run_id || "";
      }
    } else if (r.suite === "suite_F" && Number.isFinite(r.cost_efficiency_toks_per_dollar_hr)) {
      if (cur.F_val == null || r.cost_efficiency_toks_per_dollar_hr > cur.F_val) {
        cur.F_val = r.cost_efficiency_toks_per_dollar_hr;
      }
    }
  }

  const preferred = [
    ["NVIDIA H200", "H200", "neutral-strong"],
    ["NVIDIA H100 80GB HBM3", "H100", "emerald"],
    ["NVIDIA GeForce RTX 5090", "RTX 5090", "accent"],
    ["NVIDIA A100-SXM4-80GB", "A100", "amber"],
    ["NVIDIA RTX A6000", "A6000", "muted"],
  ];
  const picked = [];
  for (const [full, short, tone] of preferred) {
    const d = byChip.get(full);
    if (d && d.A != null && d.D != null && d.F_val != null) {
      picked.push({ ...d, short, tone });
    }
  }
  if (picked.length < 4) return [];

  const rankAxis = (key, cmp) => {
    const ord = [...picked].sort(cmp);
    ord.forEach((item, idx) => { item[`r_${key}`] = idx + 1; });
  };
  rankAxis("A", (a, b) => b.A - a.A);
  rankAxis("QPS", (a, b) => (b.QPS - a.QPS) || (b.A - a.A));
  rankAxis("D", (a, b) => b.D - a.D);
  rankAxis("F_val", (a, b) => b.F_val - a.F_val);
  return picked;
}

function fmtShortK(v) {
  if (v >= 10000) return `${(v / 1000).toFixed(1)}k`;
  if (v >= 1000) return Math.round(v).toLocaleString();
  if (v >= 100) return Math.round(v).toString();
  return v.toFixed(1);
}

function renderHeroRankShift(rows) {
  const series = computeRankShiftData(rows);
  if (!series.length) return "";
  const zh = isZh();

  const W = 560, H = 312;
  const xs = [116, 232, 348, 464];
  const ys = [92, 136, 180, 224, 268];
  const cols = [
    { key: "A", title: zh ? "短对话" : "Chat", sub: "Suite A" },
    { key: "QPS", title: "500ms SLA", sub: "Max QPS" },
    { key: "D", title: zh ? "28K 预填充" : "28K Ctx", sub: "Suite D" },
    { key: "F_val", title: zh ? "性价比" : "Tok / $", sub: "Suite F" },
  ];

  const colGuides = cols.map((c, i) => `
    <line x1="${xs[i]}" y1="68" x2="${xs[i]}" y2="282" class="rs-guide"/>
    <text x="${xs[i]}" y="42" text-anchor="middle" class="rs-col-title">${esc(c.title)}</text>
    <text x="${xs[i]}" y="56" text-anchor="middle" class="rs-col-sub mono">${esc(c.sub)}</text>
  `).join("");

  const rankMarks = ys.map((y, i) => `
    <text x="16" y="${y + 4}" class="rs-rank mono">#${i + 1}</text>
  `).join("");

  const drawOrder = [...series].sort((a, b) => (a.tone === "accent" ? 1 : 0) - (b.tone === "accent" ? 1 : 0));
  const paths = drawOrder.map((item) => {
    const pts = cols.map((c, i) => [xs[i], ys[item[`r_${c.key}`] - 1]]);
    const cmds = [`M ${pts[0][0]} ${pts[0][1]}`];
    for (let i = 0; i < 3; i++) {
      const mx = (pts[i][0] + pts[i + 1][0]) / 2;
      cmds.push(`C ${mx} ${pts[i][1]}, ${mx} ${pts[i + 1][1]}, ${pts[i + 1][0]} ${pts[i + 1][1]}`);
    }
    const d = cmds.join(" ");
    const vals = [
      fmtShortK(item.A),
      item.QPS === 0 ? "SLA ✗" : `${Math.round(item.QPS)}`,
      fmtShortK(item.D),
      fmtShortK(item.F_val),
    ];
    const nodes = pts.map(([px, py], i) => {
      const fail = i === 1 && item.QPS === 0;
      const pw = fail ? 44 : 40;
      return `
        <rect x="${px - pw / 2}" y="${py - 10}" width="${pw}" height="20" rx="10" class="rs-pill${fail ? " is-fail" : ""}"/>
        <text x="${px}" y="${py + 3.5}" text-anchor="middle" class="rs-val mono${fail ? " is-fail" : ""}">${esc(vals[i])}</text>
      `;
    }).join("");

    return `
      <g class="rs-series tone-${item.tone}" data-chip="${esc(item.short)}" data-slug="${esc(item.slug || "")}" tabindex="0" role="button" aria-label="${esc(item.chip)} trajectory">
        <path d="${d}" class="rs-halo"/>
        <path d="${d}" class="rs-line"/>
        <text x="${pts[0][0] - 26}" y="${pts[0][1] + 4}" text-anchor="end" class="rs-lbl">${esc(item.short)}</text>
        <text x="${pts[3][0] + 26}" y="${pts[3][1] + 4}" text-anchor="start" class="rs-lbl">${esc(item.short)}</text>
        ${nodes}
      </g>
    `;
  }).join("");

  const compareIds = [
    series.find((s) => s.short === "RTX 5090")?.runA,
    series.find((s) => s.short === "A100")?.runA,
  ].filter(Boolean).join(",");

  return `
    <div class="hero-visual-card">
      <div class="hero-visual-head">
        <span class="hero-visual-kicker mono">${zh ? "实测排名反转 · 参考栈 vLLM" : "LIVE RANK INVERSION · vLLM"}</span>
        <a class="hero-visual-link" href="${compareIds ? `#/compare?runs=${encodeURIComponent(compareIds)}` : "#/compare"}">
          ${zh ? "去对比" : "Compare"} ${icon("arrowRight", 13)}
        </a>
      </div>
      <svg class="rs-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Hardware rankings invert across workloads">
        ${colGuides}
        ${rankMarks}
        ${paths}
      </svg>
      <div class="hero-visual-foot">
        <span>${zh ? "悬停芯片高亮轨迹 · 点击查看芯片详情" : "Hover a chip to trace its path · Click to inspect"}</span>
        <span class="mono">#1 = ${zh ? "最优" : "best"}</span>
      </div>
    </div>
  `;
}

const RECIPE_TABS = [
  { id: "featured", en: "Featured",        zh: "精选方案" },
  { id: "suite_A",  en: "8B Chat",         zh: "单卡对话" },
  { id: "sla",      en: "500ms SLA",       zh: "500ms SLA" },
  { id: "suite_D",  en: "28K Long Ctx",    zh: "28K 长上下文" },
  { id: "suite_B",  en: "70B Multi-Chip",  zh: "70B 多卡" },
  { id: "suite_C",  en: "Quantization",    zh: "量化加速" },
  { id: "suite_F",  en: "Edge & Value",    zh: "边缘 / 性价比" },
  { id: "suite_G",  en: "Sparse MoE",      zh: "稀疏 MoE" },
];

function cleanRows(rows) {
  const ref = rows.filter((r) => !r.flagged && (r.runner_tier || "reference") === "reference");
  return ref.length ? ref : rows.filter((r) => !r.flagged);
}

function topBy(rows, key, dir = "desc") {
  const valid = rows.filter((r) => r[key] != null && Number.isFinite(Number(r[key])) && Number(r[key]) > 0);
  valid.sort((a, b) => (dir === "asc" ? a[key] - b[key] : b[key] - a[key]));
  return valid;
}

function pickFeaturedRecipes(rows) {
  const zh = isZh();
  const pool = cleanRows(rows);
  const picks = [];
  const used = new Set();
  const add = (row, badgeEn, badgeZh, metricKey, metricLabel, metricUnit) => {
    if (!row) return;
    const rid = row.run_id || row.submission;
    if (!rid || used.has(rid)) return;
    used.add(rid);
    picks.push({ row, badge: zh ? badgeZh : badgeEn, metricKey, metricLabel, metricUnit });
  };

  // 1. Single-chip chat #1
  add(topBy(pool.filter((r) => r.suite === "suite_A"), "offline_throughput")[0],
    "Chat · #1 Throughput", "单卡对话 · 吞吐冠军", "offline_throughput", zh ? "离线吞吐" : "Offline", "tok/s");
  // 2. 500ms SLA #1 (prefer distinct from #1 if possible)
  const slaSorted = topBy(pool.filter((r) => r.suite === "suite_A"), "online_max_qps");
  add(slaSorted.find((r) => !used.has(r.run_id)) || slaSorted[0],
    "500ms SLA · #1 Capacity", "500ms SLA · 并发冠军", "online_max_qps", "Max QPS", "QPS");
  // 3. 28K Long Context #1
  add(topBy(pool.filter((r) => r.suite === "suite_D"), "offline_throughput")[0],
    "28K Prefill · #1", "28K 长上下文 · 最强", "offline_throughput", zh ? "预填充吞吐" : "Long Ctx", "tok/s");
  // 4. 70B Multi-Chip TP #1
  add(topBy(pool.filter((r) => r.suite === "suite_B"), "offline_throughput")[0],
    "70B TP · #1", "70B 多卡并行 · 最强", "offline_throughput", zh ? "70B 吞吐" : "70B TPS", "tok/s");
  // 5. Best Value (Tok / $)
  add(topBy(pool.filter((r) => r.suite === "suite_F"), "cost_efficiency_toks_per_dollar_hr")[0],
    "Best Value · Tok/$", "性价比之王 · Tok/$", "cost_efficiency_toks_per_dollar_hr", "Value", "tok/$/hr");
  // 6. Non-NVIDIA / Multi-Ecosystem Spotlight
  const nonNv = topBy(pool.filter((r) => r.suite === "suite_A" && r.vendor !== "NVIDIA"), "offline_throughput");
  add(nonNv[0],
    "Non-NVIDIA · #1", "非 NVIDIA 生态 · 最强", "offline_throughput", zh ? "离线吞吐" : "Offline", "tok/s");

  return picks;
}

function pickRecipesForTab(tabId, rows) {
  if (tabId === "featured") return pickFeaturedRecipes(rows);
  const zh = isZh();
  const pool = cleanRows(rows);
  if (tabId === "sla") {
    return topBy(pool.filter((r) => r.suite === "suite_A"), "online_max_qps")
      .slice(0, 6)
      .map((row, idx) => ({
        row,
        badge: `#${idx + 1} · 500ms SLA`,
        metricKey: "online_max_qps",
        metricLabel: "Max QPS",
        metricUnit: "QPS",
      }));
  }
  const cols = SUITE_COLUMNS[tabId] || [];
  const primary = cols.find((c) => c.primary) || cols[0];
  const m = SUITE_META[tabId] || {};
  const list = primary ? topBy(pool.filter((r) => r.suite === tabId), primary.key, primary.direction) : [];
  return list.slice(0, 6).map((row, idx) => ({
    row,
    badge: `#${idx + 1} · Suite ${m.letter || ""}`,
    metricKey: primary.key,
    metricLabel: primary.label,
    metricUnit: primary.unit || "",
  }));
}

function renderRecipeCard(item) {
  const { row, badge, metricKey, metricLabel, metricUnit } = item;
  const zh = isZh();
  const rid = row.run_id || row.submission || "";
  const fw = row.framework || "vLLM";
  const ver = shortVersion(row.framework_version);
  const tp = Number(row.detail?.run_tp || row.chip_count || 1) || 1;
  const prec = row.precision || "BF16";
  const chip = (row._chip_label || row.chip || "—");
  const mem = row.memory_gb ? `${fmtNum(row.memory_gb)}GB` : "";
  const model = row.detail?.model_id ? shortModel(row.detail.model_id) : "";
  const isVerified = row.tier === "verified";
  const runnerId = row.implementation_id || row.impl?.id || "auto";
  const cols = SUITE_COLUMNS[row.suite] || [];
  const colSpec = cols.find((c) => c.key === metricKey) || cols.find((c) => c.primary) || { key: metricKey, format: "int" };
  const valStr = formatMetric(row[metricKey], colSpec) || "—";
  const acc = row.detail?.acc_score != null ? `MMLU ${(Number(row.detail.acc_score) * 100).toFixed(0)}%` : "";
  const serveHint = `python run.py --runner ${runnerId}${tp > 1 ? ` --tensor-parallel-size ${tp}` : ""} --serve`;

  return `
    <article class="rc-card" data-open-run="${esc(rid)}" data-open-tab="deploy" tabindex="0">
      <div class="rc-card-top">
        <span class="rc-badge mono">${esc(badge)}</span>
        ${isVerified
          ? `<span class="rc-trust is-verified">${icon("shieldCheck", 12)} ${zh ? "已验证" : "Verified"}</span>`
          : `<span class="rc-trust mono">CI ✓</span>`}
      </div>

      <div class="rc-head">
        <div class="rc-engine-row">
          <span class="rc-engine">${esc(fw)}</span>
          ${ver ? `<span class="rc-ver mono">${esc(ver)}</span>` : ""}
          <span class="rc-pill mono">${esc(prec)}</span>
          ${tp > 1 ? `<span class="rc-pill mono">TP${tp}</span>` : ""}
        </div>
        <div class="rc-hw-row">
          <span class="rc-on">${zh ? "运行于" : "on"}</span>
          <a class="rc-hw-link" href="${chipHref(row)}">${esc(chip)}${mem ? ` · ${esc(mem)}` : ""}</a>
        </div>
      </div>

      <div class="rc-metric-band">
        <div class="rc-kpi">
          <span class="rc-kpi-val mono tnum">${esc(valStr)}</span>
          ${metricUnit ? `<span class="rc-kpi-unit mono">${esc(metricUnit)}</span>` : ""}
        </div>
        <div class="rc-kpi-sub mono">
          <span>${esc(metricLabel)}</span>
          ${acc ? `<span>· ${esc(acc)} ✓</span>` : ""}
          ${model ? `<span>· ${esc(model)}</span>` : ""}
        </div>
      </div>

      <div class="rc-cmd-preview mono" title="${esc(serveHint)}">
        <span class="rc-prompt">$</span> ${esc(serveHint)}
      </div>

      <div class="rc-actions">
        <button type="button" class="rc-deploy-btn" data-open-run="${esc(rid)}" data-open-tab="deploy">
          ${icon("terminal", 13)} <span>${zh ? "一键部署此方案" : "Deploy Recipe"}</span>
        </button>
        <button type="button" class="rc-detail-btn" data-open-run="${esc(rid)}" data-open-tab="details">
          ${zh ? "详情" : "Details"}
        </button>
      </div>
    </article>
  `;
}

function renderFeaturedRecipesSection(rows, activeTab = "featured") {
  const zh = isZh();
  const cards = pickRecipesForTab(activeTab, rows);
  const suiteHref = activeTab === "featured"
    ? "#/rankings"
    : (activeTab === "sla" ? "#/rankings?suite=suite_A&sort=online_max_qps:desc" : buildHash("/rankings", { suite: activeTab }));

  return `
    <section class="section" id="home-recipes-section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${zh ? "01 · 精选部署方案" : "01 · Featured Serving Recipes"}</span>
          <h2>${zh ? "按场景挑一套实测最强方案，直接复制上线" : "Battle-tested serving recipes, ready to deploy"}</h2>
        </div>
        <a class="section-action" href="${esc(suiteHref)}">
          ${zh ? "查看全部方案榜" : "Browse all recipes"} ${icon("arrowRight", 14)}
        </a>
      </div>

      <div class="rc-tabs" role="tablist" aria-label="Deployment scenarios">
        ${RECIPE_TABS.map((t) => `
          <button type="button"
                  class="rc-tab${t.id === activeTab ? " active" : ""}"
                  data-rc-tab="${esc(t.id)}"
                  role="tab"
                  aria-selected="${t.id === activeTab}">
            ${esc(zh ? t.zh : t.en)}
          </button>
        `).join("")}
      </div>

      <div class="rc-grid" id="home-rc-grid">
        ${cards.map(renderRecipeCard).join("")}
      </div>

      <div class="rc-suite-strip">
        <span class="rc-strip-label mono">${zh ? "按 8 大评测负载直达：" : "JUMP BY SUITE:"}</span>
        ${SUITE_ORDER.map((sid) => {
          const m = SUITE_META[sid];
          const shortName = (zh ? SHORT_SUITE_NAMES_ZH : SHORT_SUITE_NAMES_EN)[sid] || m.title;
          const href = sid === SUITE_ORDER[0] ? "#/rankings" : buildHash("/rankings", { suite: sid });
          return `<a class="rc-suite-chip" href="${esc(href)}"><span class="mono">${esc(m.letter)}</span> ${esc(shortName)}</a>`;
        }).join("")}
      </div>
    </section>
  `;
}

function renderTop3Findings(rows) {
  const findings = computeInsights(rows).slice(0, 3);
  if (!findings.length) return "";
  const zh = isZh();

  const cards = findings.map((f) => {
    if (f.id === "reversal") {
      const xS = f.x.chip.replace(/^NVIDIA\s+(GeForce\s+)?/, "");
      const yS = f.y.chip.replace(/^NVIDIA\s+/, "").replace(/-SXM4-/, "-");
      const gain = Math.round(f.gainA * 100);
      const loss = Math.round(f.lossB * 100);
      const href = `#/compare?runs=${encodeURIComponent([f.x.run_id, f.y.run_id].filter(Boolean).join(","))}`;
      return `
        <a class="f-card" href="${esc(href)}">
          <div class="f-card-top">
            <span class="f-kicker mono">${zh ? "跨负载反转" : "REGIME REVERSAL"}</span>
            <span class="f-tag mono">A ↔ D</span>
          </div>
          <div class="f-stat mono tnum">+${gain}% → −${loss}%</div>
          <div class="f-head">${zh ? `${esc(xS)} 赢短对话，输长上下文` : `${esc(xS)} wins short chat, loses 28K prefill`}</div>
          <p class="f-body">${zh
            ? `同为 ${esc(f.framework)}，${esc(xS)} 在 Suite A 领先 ${esc(yS)} +${gain}%，到了 28K 长上下文 Suite D 反落后 ${loss}%。`
            : `On identical ${esc(f.framework)}, ${esc(xS)} beats ${esc(yS)} by +${gain}% on Suite A, then trails by −${loss}% on 28K prefill (Suite D).`}</p>
          <div class="f-bars">
            <div class="f-bar-row"><span class="mono">Suite A</span><div class="f-track"><div class="f-fill accent" style="width:100%"></div></div><span class="mono">${fmtNum(Math.round(f.x.a))}</span></div>
            <div class="f-bar-row"><span class="mono">Suite D</span><div class="f-track"><div class="f-fill muted" style="width:${Math.round(f.x.b / f.y.b * 100)}%"></div></div><span class="mono">${f.x.b.toFixed(1)} vs ${f.y.b.toFixed(1)}</span></div>
          </div>
          <span class="f-cta">${zh ? "对比这组实测" : "Compare these runs"} ${icon("arrowRight", 13)}</span>
        </a>
      `;
    }
    if (f.id === "sla_gap") {
      const oS = f.offline.chip.replace(/^NVIDIA\s+/, "");
      const sS = f.sla.chip.replace(/^NVIDIA\s+/, "").replace(/\s+HBM3$/, "");
      const ratio = Math.round(f.sla.qps / Math.max(f.offline.qps, 0.1));
      return `
        <a class="f-card" href="#/rankings?suite=suite_A&sort=online_max_qps:desc">
          <div class="f-card-top">
            <span class="f-kicker mono">${zh ? "离线 vs SLA" : "OFFLINE VS SLA"}</span>
            <span class="f-tag mono">Suite A</span>
          </div>
          <div class="f-stat mono tnum">${Math.round(f.sla.qps)} vs ${Math.round(f.offline.qps)} QPS</div>
          <div class="f-head">${zh ? "离线吞吐第一未必赢生产 SLA" : "Peak offline speed fails to win SLA"}</div>
          <p class="f-body">${zh
            ? `${esc(oS)} 离线吞吐最高（${fmtNum(Math.round(f.offline.throughput))} tok/s），但在 500ms p99 TTFT 约束下仅撑住 ${Math.round(f.offline.qps)} QPS；${esc(sS)} 可达 ${Math.round(f.sla.qps)} QPS（${ratio}×）。`
            : `${esc(oS)} tops offline (${fmtNum(Math.round(f.offline.throughput))} tok/s) yet caps at ${Math.round(f.offline.qps)} QPS under 500 ms TTFT; ${esc(sS)} sustains ${Math.round(f.sla.qps)} QPS (${ratio}×).`}</p>
          <div class="f-bars">
            <div class="f-bar-row"><span class="mono">${esc(sS)}</span><div class="f-track"><div class="f-fill accent" style="width:100%"></div></div><span class="mono">${Math.round(f.sla.qps)} QPS</span></div>
            <div class="f-bar-row"><span class="mono">${esc(oS)}</span><div class="f-track"><div class="f-fill muted" style="width:${Math.max(12, Math.round(f.offline.qps / f.sla.qps * 100))}%"></div></div><span class="mono">${Math.round(f.offline.qps)} QPS</span></div>
          </div>
          <span class="f-cta">${zh ? "按 Online QPS 排序" : "Inspect SLA ranking"} ${icon("arrowRight", 13)}</span>
        </a>
      `;
    }
    if (f.id === "engine_gap") {
      const cS = f.chip.replace(/^NVIDIA\s+/, "");
      const gain = Math.round(f.gain * 100);
      const href = `#/compare?runs=${encodeURIComponent([f.fast.run_id, f.slow.run_id].filter(Boolean).join(","))}`;
      return `
        <a class="f-card" href="${esc(href)}">
          <div class="f-card-top">
            <span class="f-kicker mono">${zh ? "推理引擎差距" : "ENGINE > CHIP"}</span>
            <span class="f-tag mono">Suite A</span>
          </div>
          <div class="f-stat mono tnum">+${gain}%</div>
          <div class="f-head">${zh ? "同一张卡换推理引擎，提升近一倍" : "Software stack moves more than silicon"}</div>
          <p class="f-body">${zh
            ? `同一张 ${esc(cS)} 上，将 ${esc(f.slow.framework)}（${fmtNum(Math.round(f.slow.throughput))} tok/s）切换为 ${esc(f.fast.framework)}（${fmtNum(Math.round(f.fast.throughput))} tok/s）直接带来 +${gain}% 吞吐提升。`
            : `On the exact same ${esc(cS)}, switching ${esc(f.slow.framework)} (${fmtNum(Math.round(f.slow.throughput))}) → ${esc(f.fast.framework)} (${fmtNum(Math.round(f.fast.throughput))} tok/s) lifts throughput by +${gain}%.`}</p>
          <div class="f-bars">
            <div class="f-bar-row"><span class="mono">${esc(f.fast.framework)}</span><div class="f-track"><div class="f-fill accent" style="width:100%"></div></div><span class="mono">${fmtNum(Math.round(f.fast.throughput))}</span></div>
            <div class="f-bar-row"><span class="mono">${esc(f.slow.framework)}</span><div class="f-track"><div class="f-fill muted" style="width:${Math.round(f.slow.throughput / f.fast.throughput * 100)}%"></div></div><span class="mono">${fmtNum(Math.round(f.slow.throughput))}</span></div>
          </div>
          <span class="f-cta">${zh ? "对比两个引擎" : "Compare engines"} ${icon("arrowRight", 13)}</span>
        </a>
      `;
    }
    return "";
  }).filter(Boolean).join("");

  return `
    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${zh ? "02 · 实测洞察" : "02 · Key Findings"}</span>
          <h2>${zh ? "为什么单一分数会误导选型" : "What the live data actually shows"}</h2>
        </div>
        <a class="section-action" href="#/findings">
          ${zh ? "查看全部 6 大发现与选型矩阵" : "All 6 findings & procurement matrix"} ${icon("arrowRight", 14)}
        </a>
      </div>
      <div class="findings-grid-3">${cards}</div>
    </section>
  `;
}

export function render({ el }) {
  const rows = allRows();
  const stats = summary();
  const zh = isZh();

  el.innerHTML = `
    <section class="home-hero">
      <div class="hero-copy">
        <div class="hero-badge mono">
          <span class="hero-badge-dot"></span>
          ${zh ? "实测验证的 LLM 推理部署方案库 · 一键复现与上线" : "Verified LLM Serving Recipes · Benchmarked & Ready to Deploy"}
        </div>
        <h1 class="hero-h1">
          ${zh
            ? "没有唯一的“最快芯片”。<br>只有最适合你负载的<span>部署方案</span>。"
            : "No single fastest chip.<br>Only the best <span>serving recipe</span>."}
        </h1>
        <p class="hero-lead">
          ${zh
            ? "硬件排名会随上下文长度、SLA 延迟和引擎栈剧烈反转。按你的芯片与业务场景查找经过 MMLU 精度验证的完整推理方案，一键隔离装环境并拉起 OpenAI 兼容服务。"
            : "Pick your hardware and workload to find MMLU-verified chip × engine × precision recipes — with isolated setup scripts and one-command OpenAI-compatible --serve deployment."}
        </p>
        <div class="hero-actions">
          <a class="btn-solid" href="#/rankings">
            ${zh ? "查找部署方案" : "Browse Recipes"} ${icon("arrowRight", 15)}
          </a>
          <a class="btn-outline" href="#/submit">
            ${icon("terminal", 14)} ${zh ? "10 分钟贡献方案" : "Submit a Recipe"}
          </a>
        </div>
        <div class="hero-kpi-line mono tnum">
          <span><strong>${fmtNum(stats.total)}</strong> ${zh ? "套实测方案" : "recipes"}</span>
          <span class="kpi-sep">·</span>
          <span><strong>${fmtNum(stats.chips)}</strong> ${zh ? "款硬件" : "chips"}</span>
          <span class="kpi-sep">·</span>
          <span><strong>7</strong> ${zh ? "大生态" : "ecosystems"}</span>
          <span class="kpi-sep">·</span>
          <span><strong>${fmtNum(stats.verified)}</strong> ${zh ? "条已复现" : "verified"}</span>
        </div>
      </div>

      <div class="hero-visual">
        ${renderHeroRankShift(rows)}
      </div>
    </section>

    ${renderFeaturedRecipesSection(rows, "featured")}

    ${renderTop3Findings(rows)}

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${zh ? "03 · 可信保障" : "03 · Reproducibility"}</span>
          <h2>${zh ? "每一行榜单数据都可审计、可复现" : "Why you can trust every row"}</h2>
        </div>
      </div>
      <div class="trust-grid">
        <div class="trust-item">
          <span class="trust-ic">${icon("hash", 17)}</span>
          <h3>${zh ? "内容哈希锁定 Runner" : "Content-Hashed Runners"}</h3>
          <p>${zh ? "目录后缀为 runner.py 的 SHA-256，改一行代码即生成新 ID，绝不悄悄污染历史数据。" : "Every runner folder embeds its runner.py SHA-256 prefix so stack changes never silently drift."}</p>
        </div>
        <div class="trust-item">
          <span class="trust-ic">${icon("checkCircle", 17)}</span>
          <h3>${zh ? "强制 MMLU 精度门禁" : "MMLU Accuracy Gate"}</h3>
          <p>${zh ? "测速前先在同一加载实例上跑精度校验，防止以牺牲模型质量换取虚假吞吐。" : "Runs on the exact loaded model instance before throughput; degraded quality is permanently flagged."}</p>
        </div>
        <div class="trust-item">
          <span class="trust-ic">${icon("lock", 17)}</span>
          <h3>${zh ? "Closed / Open 分层" : "Closed vs Open Tiers"}</h3>
          <p>${zh ? "默认榜单只呈现开箱即用的 reference 栈，调优版与实验版清晰隔离、一键切换。" : "Leaderboard defaults to out-of-the-box reference stacks; tuned and experimental forks stay cleanly separated."}</p>
        </div>
        <div class="trust-item">
          <span class="trust-ic">${icon("shieldCheck", 17)}</span>
          <h3>${zh ? "±5% 独立复现认证" : "±5% Verification"}</h3>
          <p>${zh ? "CI 校验 schema 与硬件指纹后即刻上榜；独立复现误差 ≤5% 自动晋升 Verified。" : "Automated CI validates receipts on PR; independent reruns within ±5% promote entries to Verified."}</p>
        </div>
      </div>
    </section>

    <section class="section home-bottom-strip">
      <div class="eco-bar">
        <div class="eco-label mono">${zh ? "已覆盖 7 大算力生态" : "SUPPORTED ECOSYSTEMS"}</div>
        <div class="eco-list">
          ${ECOSYSTEMS.map((e) => `<span class="eco-item">${esc(e)}</span>`).join("")}
        </div>
        <a class="eco-link" href="#/wanted">${zh ? "查看征集硬件" : "Wanted hardware"} ${icon("arrowRight", 13)}</a>
      </div>

      <div class="cta-banner">
        <div class="cta-copy">
          <h3>${zh ? "手上有 GPU / NPU / TPU？10 分钟就能贡献一条权威基线" : "Have a GPU, NPU, or TPU? Contribute a baseline in ~10 minutes."}</h3>
          <p>${zh ? "无需极致调参。消费卡、老卡和国产卡的诚实默认配置同样宝贵，支持免 git 打包上传。" : "No SOTA tuning required. Honest out-of-the-box baselines on consumer, older, or emerging chips are just as valuable."}</p>
        </div>
        <div class="cta-btns">
          <a class="btn-solid" href="#/submit">${zh ? "开始提交" : "Submit a result"} ${icon("arrowRight", 14)}</a>
          <a class="btn-outline" href="#/reproduce">${zh ? "参与复现认证" : "Verify a result"}</a>
        </div>
      </div>
    </section>
  `;

  // Wire scenario filter tabs on Featured Serving Recipes
  const rcSection = el.querySelector("#home-recipes-section");
  if (rcSection) {
    rcSection.addEventListener("click", (ev) => {
      const tabBtn = ev.target.closest("[data-rc-tab]");
      if (!tabBtn) return;
      ev.preventDefault();
      const tabId = tabBtn.dataset.rcTab;
      rcSection.querySelectorAll("[data-rc-tab]").forEach((b) => {
        const on = b === tabBtn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-selected", String(on));
      });
      const grid = rcSection.querySelector("#home-rc-grid");
      if (grid) {
        grid.innerHTML = pickRecipesForTab(tabId, rows).map(renderRecipeCard).join("");
      }
      const actionLink = rcSection.querySelector(".section-action");
      if (actionLink) {
        actionLink.href = tabId === "featured"
          ? "#/rankings"
          : (tabId === "sla" ? "#/rankings?suite=suite_A&sort=online_max_qps:desc" : buildHash("/rankings", { suite: tabId }));
      }
    });
  }

  // Wire interactive hover highlighting & click-to-chip on the Hero Rank-Shift chart
  const svg = el.querySelector(".rs-svg");
  if (svg) {
    const groups = svg.querySelectorAll(".rs-series");
    const setHighlight = (target) => {
      svg.classList.toggle("is-hovering", !!target);
      groups.forEach((g) => g.classList.toggle("is-hot", g === target));
    };
    groups.forEach((g) => {
      g.addEventListener("mouseenter", () => setHighlight(g));
      g.addEventListener("focus", () => setHighlight(g));
      g.addEventListener("mouseleave", () => setHighlight(null));
      g.addEventListener("blur", () => setHighlight(null));
      g.addEventListener("click", () => {
        if (g.dataset.slug) location.hash = `#/chip/${g.dataset.slug}`;
      });
    });
  }
}
