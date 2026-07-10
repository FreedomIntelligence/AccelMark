// share-card.js — render & download a social/share PNG for a benchmark run.

import { SUITE_META } from "./data.js";
import { runCanonicalUrl } from "./cite.js";

function suiteLabel(row) {
  const meta = SUITE_META[row.suite];
  return meta ? `Suite ${meta.letter}` : row.suite || "—";
}

function primaryMetric(row) {
  const v = row.primary_metric ?? row.offline_throughput ?? row.primary_value;
  const u = row.primary_metric_label || row.metric_unit || "tok/s";
  if (v == null) return "—";
  const n = Number(v);
  const formatted = Number.isFinite(n)
    ? n.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : String(v);
  return `${formatted} ${u}`;
}

function tierLabel(tier) {
  return tier === "verified" ? "VERIFIED" : "COMMUNITY";
}

function tierColor(tier) {
  return tier === "verified" ? "#1D9E75" : "#378ADD";
}

function loadQrImage(url, size) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src =
      `https://api.qrserver.com/v1/create-qr-code/?size=${size}x${size}` +
      `&margin=1&data=${encodeURIComponent(url)}`;
  });
}

/** Draw share card to canvas and trigger download. */
export async function downloadResultShareCard(row, { filename } = {}) {
  if (!row) return false;

  const W = 720;
  const H = 400;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return false;

  const isDark = window.matchMedia?.("(prefers-color-scheme: dark)")?.matches;
  const bg = isDark ? "#0b0c10" : "#faf8f3";
  const fg = isDark ? "#e6edf3" : "#1a1a1a";
  const muted = isDark ? "#8b949e" : "#5c5c5c";
  const accent = "#378ADD";
  const gold = "#F59E0B";

  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = accent;
  ctx.fillRect(0, 0, W, 6);

  ctx.fillStyle = gold;
  ctx.font = "bold 22px system-ui, sans-serif";
  ctx.fillText("AccelMark", 28, 44);

  ctx.fillStyle = muted;
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText("Reproducible LLM inference benchmark", 28, 64);

  const chipLine = row.chip + (row.chip_count > 1 ? ` ×${row.chip_count}` : "");
  ctx.fillStyle = fg;
  ctx.font = "bold 36px system-ui, sans-serif";
  ctx.fillText(chipLine, 28, 118);

  ctx.font = "20px system-ui, sans-serif";
  ctx.fillStyle = muted;
  ctx.fillText(`${suiteLabel(row)} · ${primaryMetric(row)}`, 28, 152);

  const tier = row.tier || "community";
  const pill = tierLabel(tier);
  ctx.font = "bold 12px system-ui, sans-serif";
  const pillW = ctx.measureText(pill).width + 20;
  ctx.fillStyle = tierColor(tier);
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(28, 168, pillW, 26, 6);
    ctx.fill();
  } else {
    ctx.fillRect(28, 168, pillW, 26);
  }
  ctx.fillStyle = "#fff";
  ctx.fillText(pill, 38, 186);

  ctx.fillStyle = muted;
  ctx.font = "13px ui-monospace, monospace";
  ctx.fillText(`ID: ${row.run_id || "—"}`, 28, 228);
  ctx.font = "14px system-ui, sans-serif";
  if (row.vendor) ctx.fillText(row.vendor, 28, 252);

  const url = runCanonicalUrl(row.run_id);
  try {
    const qr = await loadQrImage(url, 120);
    ctx.drawImage(qr, W - 148, 28, 120, 120);
  } catch {
    ctx.strokeStyle = muted;
    ctx.strokeRect(W - 148, 28, 120, 120);
    ctx.fillStyle = muted;
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("Scan link", W - 118, 92);
  }

  ctx.fillStyle = accent;
  ctx.font = "12px system-ui, sans-serif";
  const linkShort = url.length > 42 ? url.slice(0, 40) + "…" : url;
  ctx.fillText(linkShort, 28, H - 36);

  ctx.fillStyle = muted;
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("CC BY 4.0 · result.json · env_info.json · runner hash", 28, H - 16);

  const name = filename || `accelmark-${(row.run_id || "result").replace(/[^a-zA-Z0-9_-]+/g, "_")}.png`;

  return new Promise((resolve) => {
    if (canvas.toBlob) {
      canvas.toBlob((blob) => {
        if (!blob) { resolve(false); return; }
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = name;
        a.click();
        URL.revokeObjectURL(a.href);
        resolve(true);
      }, "image/png");
    } else {
      try {
        const a = document.createElement("a");
        a.href = canvas.toDataURL("image/png");
        a.download = name;
        a.click();
        resolve(true);
      } catch {
        resolve(false);
      }
    }
  });
}
