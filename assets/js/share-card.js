// share-card.js — Studio-grade 2× Retina (1200×630) PNG share cards for
// single benchmark runs and side-by-side comparisons.
// Zero external network calls (works offline and behind firewalls).

import { SUITE_META } from "./data.js";
import { runCanonicalUrl } from "./cite.js";

const FONT_SANS = '-apple-system, BlinkMacSystemFont, "Inter", "SF Pro Display", "Segoe UI", sans-serif';
const FONT_MONO = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace';

function currentTheme() {
  const explicit = typeof document !== "undefined" ? document.documentElement.getAttribute("data-theme") : null;
  if (explicit === "dark" || explicit === "light") return explicit;
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)")?.matches
    ? "dark"
    : "light";
}

function palette(theme) {
  if (theme === "dark") {
    return {
      bg: "#09090B",
      surface: "#111114",
      surface2: "#18181D",
      border: "#27272A",
      grid: "#18181D",
      fg: "#FAFAFA",
      fgBody: "#D4D4D8",
      muted: "#A1A1AA",
      faint: "#71717A",
      accent: "#3B82F6",
      accentSoft: "rgba(59,130,246,0.16)",
      good: "#4ADE80",
      goodSoft: "rgba(74,222,128,0.14)",
      bad: "#F87171",
      warn: "#FBBF24",
    };
  }
  return {
    bg: "#FFFFFF",
    surface: "#FAFAFA",
    surface2: "#F4F4F5",
    border: "#E4E4E7",
    grid: "#F4F4F5",
    fg: "#09090B",
    fgBody: "#3F3F46",
    muted: "#71717A",
    faint: "#A1A1AA",
    accent: "#2563EB",
    accentSoft: "rgba(37,99,235,0.09)",
    good: "#16A34A",
    goodSoft: "rgba(22,163,74,0.10)",
    bad: "#DC2626",
    warn: "#D97706",
  };
}

function drawRoundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
  }
  ctx.closePath();
}

/** Draw the exact Gen-1 Roofline "A" vector mark on Canvas at (x, y) with given size. */
function drawGen1Logo(ctx, x, y, size, fg, accent) {
  const s = size / 64;
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(s, s);

  const drawPoly = (pts, color) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    pts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
    ctx.closePath();
    ctx.fill();
  };

  drawPoly([[30.03, 10.5], [59.0, 10.5], [54.55, 16.71], [33.05, 16.71], [13.57, 53.41], [5.0, 53.41]], fg);
  drawPoly([[34.81, 19.57], [37.92, 19.57], [59.0, 53.41], [50.27, 53.41], [43.63, 41.66], [21.8, 41.66], [39.26, 34.01], [33.13, 22.68]], fg);

  // Cobalt droplet
  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.moveTo(31.58, 26.2);
  ctx.bezierCurveTo(31.58, 26.2, 34.14, 29.73, 34.14, 31.37);
  ctx.arc(31.58, 31.37, 2.56, 0, Math.PI, false);
  ctx.bezierCurveTo(29.02, 29.73, 31.58, 26.2, 31.58, 26.2);
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}

function drawBackground(ctx, W, H, p) {
  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, W, H);
  // Subtle architectural grid
  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  for (let gx = 40; gx < W; gx += 40) {
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
  }
  for (let gy = 40; gy < H; gy += 40) {
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
  }
  // Outer hairline frame
  ctx.strokeStyle = p.border;
  ctx.lineWidth = 2;
  drawRoundRect(ctx, 1, 1, W - 2, H - 2, 16);
  ctx.stroke();
}

function fmtMetricValue(v) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  if (Math.abs(n) >= 100) return Math.round(n).toLocaleString();
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

function suiteTitle(row) {
  const m = SUITE_META[row.suite];
  return m ? `Suite ${m.letter} · ${m.title}` : (row.suite || "Benchmark");
}

function triggerDownload(canvas, filename) {
  return new Promise((resolve) => {
    if (canvas.toBlob) {
      canvas.toBlob((blob) => {
        if (!blob) { resolve(false); return; }
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        resolve(true);
      }, "image/png");
    } else {
      try {
        const a = document.createElement("a");
        a.href = canvas.toDataURL("image/png");
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        resolve(true);
      } catch {
        resolve(false);
      }
    }
  });
}

/** Render & download a studio-grade 1200×630 PNG share card for one benchmark run. */
export async function downloadResultShareCard(row, { filename, theme } = {}) {
  if (!row) return false;

  const W = 1200;
  const H = 630;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return false;

  const p = palette(theme || currentTheme());
  drawBackground(ctx, W, H, p);

  // Top header: Gen-1 logo + wordmark + right-aligned suite badge
  drawGen1Logo(ctx, 56, 46, 44, p.fg, p.accent);
  ctx.fillStyle = p.fg;
  ctx.font = `700 24px ${FONT_SANS}`;
  ctx.fillText("AccelMark", 110, 76);

  const sLabel = suiteTitle(row);
  ctx.font = `600 15px ${FONT_MONO}`;
  const sW = ctx.measureText(sLabel).width + 28;
  drawRoundRect(ctx, W - 56 - sW, 48, sW, 34, 17);
  ctx.fillStyle = p.accentSoft;
  ctx.fill();
  ctx.fillStyle = p.accent;
  ctx.fillText(sLabel, W - 56 - sW + 14, 70);

  // Chip headline
  const chipLine = (row.chip || "Unknown Chip") + (row.chip_count > 1 ? ` ×${row.chip_count}` : "");
  ctx.fillStyle = p.fg;
  ctx.font = `700 46px ${FONT_SANS}`;
  ctx.fillText(chipLine, 56, 172);

  // Recipe subtitle: framework · precision · vendor · tier badge
  const fw = [row.framework, row.framework_version].filter(Boolean).join(" ");
  const recipeLine = [fw, row.precision, row.vendor].filter(Boolean).join("  ·  ");
  ctx.fillStyle = p.muted;
  ctx.font = `500 20px ${FONT_MONO}`;
  ctx.fillText(recipeLine, 56, 212);

  // Trust & runner tier pill
  const isVerified = row.tier === "verified";
  const badgeText = isVerified ? "✓ VERIFIED (±5%)" : "COMMUNITY";
  ctx.font = `700 13px ${FONT_MONO}`;
  const bW = ctx.measureText(badgeText).width + 24;
  drawRoundRect(ctx, 56, 234, bW, 30, 8);
  ctx.fillStyle = isVerified ? p.goodSoft : p.accentSoft;
  ctx.fill();
  ctx.fillStyle = isVerified ? p.good : p.accent;
  ctx.fillText(badgeText, 68, 254);

  const runnerTier = row.runner_tier || "reference";
  const tierText = runnerTier === "reference" ? "CLOSED · REFERENCE" : `OPEN · ${runnerTier.toUpperCase()}`;
  const tW = ctx.measureText(tierText).width + 24;
  drawRoundRect(ctx, 56 + bW + 12, 234, tW, 30, 8);
  ctx.fillStyle = p.surface2;
  ctx.fill();
  ctx.strokeStyle = p.border;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = p.muted;
  ctx.fillText(tierText, 56 + bW + 24, 254);

  // Hero Primary Metric Card + 3 KPI sub-cards
  const primaryVal = row.primary_metric ?? row.offline_throughput ?? row.primary_value;
  const primaryUnit = row.primary_metric_label || row.metric_unit || "tokens/sec";

  drawRoundRect(ctx, 56, 294, 460, 210, 14);
  ctx.fillStyle = p.surface;
  ctx.fill();
  ctx.strokeStyle = p.border;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = p.accent;
  ctx.font = `600 13px ${FONT_MONO}`;
  ctx.fillText("PRIMARY METRIC", 84, 332);

  ctx.fillStyle = p.fg;
  ctx.font = `700 68px ${FONT_MONO}`;
  ctx.fillText(fmtMetricValue(primaryVal), 84, 418);

  ctx.fillStyle = p.muted;
  ctx.font = `500 20px ${FONT_SANS}`;
  ctx.fillText(primaryUnit, 84, 456);

  // Right 3 KPI tiles
  const qps = row.online_max_qps;
  const qpsVal = qps == null ? "—" : (Number(qps) === 0 ? "SLA fail" : `${fmtMetricValue(qps)} QPS`);
  const ttft = row.interactive_ttft_p99 ?? row.ttft_p99_ms ?? row.interactive_ttft_p99_ms ?? row.online_ttft_p99_ms;
  const ttftVal = ttft != null ? `${Math.round(Number(ttft))} ms` : "—";
  const mmlu = row.mmlu_score ?? row.accuracy_score;
  const mmluVal = mmlu != null ? `${(Number(mmlu) <= 1 ? Number(mmlu) * 100 : Number(mmlu)).toFixed(1)}%` : (row.accuracy_passed === false ? "Flagged" : "Pass");
  const tokDollar = row.cost_efficiency_toks_per_dollar_hr;
  const thirdKpi = tokDollar != null
    ? { label: "COST EFFICIENCY", val: `${fmtMetricValue(tokDollar)}`, sub: "tok / $/hr" }
    : { label: "MMLU ACCURACY GATE", val: mmluVal, sub: "verified before run" };

  const kpis = [
    { label: "500 MS SLA CAPACITY", val: qpsVal, sub: "Poisson online sweep", isBad: qps != null && Number(qps) === 0 },
    { label: "TTFT P99 LATENCY", val: ttftVal, sub: "time to first token" },
    thirdKpi,
  ];

  kpis.forEach((k, i) => {
    const ky = 294 + i * 74;
    drawRoundRect(ctx, 540, ky, 604, 62, 10);
    ctx.fillStyle = p.surface;
    ctx.fill();
    ctx.strokeStyle = p.border;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = p.muted;
    ctx.font = `600 11.5px ${FONT_MONO}`;
    ctx.fillText(k.label, 562, ky + 25);
    ctx.fillStyle = p.faint;
    ctx.font = `400 12px ${FONT_SANS}`;
    ctx.fillText(k.sub, 562, ky + 45);

    ctx.fillStyle = k.isBad ? p.bad : p.fg;
    ctx.font = `700 24px ${FONT_MONO}`;
    const vw = ctx.measureText(k.val).width;
    ctx.fillText(k.val, 1122 - vw, ky + 39);
  });

  // Footer provenance bar
  ctx.strokeStyle = p.border;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(56, 544); ctx.lineTo(W - 56, 544); ctx.stroke();

  const runnerId =
    row.runner_id ||
    row.implementation_id ||
    row.impl?.runner_id ||
    (row.reproduce_script ? row.reproduce_script.split("/")[1] : null) ||
    "—";
  ctx.fillStyle = p.muted;
  ctx.font = `500 14px ${FONT_MONO}`;
  ctx.fillText(`runner: ${runnerId}   ·   run_id: ${row.run_id || "—"}   ·   ${row.date || ""}`, 56, 578);

  const canonical = "freedomintelligence.github.io/AccelMark";
  ctx.fillStyle = p.accent;
  ctx.font = `600 14px ${FONT_MONO}`;
  const cw = ctx.measureText(canonical).width;
  ctx.fillText(canonical, W - 56 - cw, 578);

  ctx.fillStyle = p.faint;
  ctx.font = `400 12px ${FONT_SANS}`;
  ctx.fillText("CC BY 4.0 · Auditable result.json + env_info.json + content-hashed runner", 56, 602);

  const name = filename || `accelmark-${(row.run_id || row.chip || "result").replace(/[^a-zA-Z0-9_-]+/g, "_")}.png`;
  return triggerDownload(canvas, name);
}

/** Render & download a studio-grade 1200×630 side-by-side comparison share card. */
export async function downloadCompareShareCard(rows, { title, metricLabel, metricFn, filename, theme } = {}) {
  const list = (rows || []).filter(Boolean).slice(0, 6);
  if (list.length < 2) return false;

  const W = 1200;
  const H = 630;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return false;

  const p = palette(theme || currentTheme());
  drawBackground(ctx, W, H, p);

  // Header
  drawGen1Logo(ctx, 56, 44, 42, p.fg, p.accent);
  ctx.fillStyle = p.fg;
  ctx.font = `700 24px ${FONT_SANS}`;
  ctx.fillText("AccelMark", 108, 73);

  ctx.fillStyle = p.muted;
  ctx.font = `600 13px ${FONT_MONO}`;
  const tag = "HEAD-TO-HEAD COMPARISON";
  const tw = ctx.measureText(tag).width;
  ctx.fillText(tag, W - 56 - tw, 71);

  ctx.fillStyle = p.fg;
  ctx.font = `700 34px ${FONT_SANS}`;
  ctx.fillText(title || suiteTitle(list[0]), 56, 136);

  const unit = metricLabel || list[0].primary_metric_label || "tokens/sec";
  ctx.fillStyle = p.muted;
  ctx.font = `400 17px ${FONT_SANS}`;
  ctx.fillText(`Measured ${unit} (higher is better) · Sorted by primary metric`, 56, 166);

  // Extract values and sort descending
  const getter = metricFn || ((r) => Number(r.primary_metric ?? r.offline_throughput ?? 0));
  const scored = list.map((r) => ({ row: r, val: getter(r) })).sort((a, b) => b.val - a.val);
  const maxVal = Math.max(...scored.map((s) => s.val), 1);

  const rowH = Math.min(62, Math.floor(320 / scored.length));
  const startY = 200;
  const barX = 430;
  const barMaxW = 560;

  scored.forEach((s, i) => {
    const y = startY + i * (rowH + 10);
    const r = s.row;
    const isTop = i === 0;
    const chipText = (r.chip || "—") + (r.chip_count > 1 ? ` ×${r.chip_count}` : "");
    const subText = [r.framework, r.precision, r.tier === "verified" ? "✓" : ""].filter(Boolean).join(" · ");

    ctx.fillStyle = isTop ? p.fg : p.fgBody;
    ctx.font = `${isTop ? 700 : 600} 18px ${FONT_SANS}`;
    ctx.fillText(chipText.slice(0, 28), 56, y + 24);

    ctx.fillStyle = p.muted;
    ctx.font = `500 13px ${FONT_MONO}`;
    ctx.fillText(subText, 56, y + 44);

    // Track + bar
    drawRoundRect(ctx, barX, y + 12, barMaxW, 28, 6);
    ctx.fillStyle = p.surface2;
    ctx.fill();

    const bw = Math.max(12, Math.round((s.val / maxVal) * barMaxW));
    drawRoundRect(ctx, barX, y + 12, bw, 28, 6);
    ctx.fillStyle = isTop ? p.accent : (theme === "dark" ? "#3F3F46" : "#A1A1AA");
    ctx.fill();

    // Numeric value + relative ratio
    const ratio = i === 0 ? "1.00×" : `${(s.val / maxVal).toFixed(2)}×`;
    ctx.fillStyle = isTop ? p.accent : p.fg;
    ctx.font = `700 18px ${FONT_MONO}`;
    ctx.fillText(`${fmtMetricValue(s.val)}`, barX + barMaxW + 18, y + 32);

    ctx.fillStyle = p.muted;
    ctx.font = `500 13px ${FONT_MONO}`;
    ctx.fillText(ratio, W - 56 - ctx.measureText(ratio).width, y + 32);
  });

  // Footer
  ctx.strokeStyle = p.border;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(56, 552); ctx.lineTo(W - 56, 552); ctx.stroke();

  ctx.fillStyle = p.muted;
  ctx.font = `500 14px ${FONT_MONO}`;
  ctx.fillText("AccelMark Open LLM Inference Benchmarks · CC BY 4.0", 56, 588);

  const canonical = "freedomintelligence.github.io/AccelMark/#/compare";
  ctx.fillStyle = p.accent;
  ctx.font = `600 14px ${FONT_MONO}`;
  ctx.fillText(canonical, W - 56 - ctx.measureText(canonical).width, 588);

  const name = filename || `accelmark-compare-${Date.now()}.png`;
  return triggerDownload(canvas, name);
}
