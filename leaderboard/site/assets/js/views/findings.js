// views/findings.js — Dedicated Key Findings & Workload-to-Recipe Procurement Matrix.
// Powered live by insights.js so every number matches the live leaderboard.

import { allRows } from "../data.js";
import { computeInsights } from "../insights.js";
import { esc, fmtNum } from "../utils.js";
import { icon } from "../icons.js";

const isZh = () => typeof window._lang === "function" && window._lang() === "cn";

function renderFindingBlock(f, index) {
  const zh = isZh();
  const num = String(index + 1).padStart(2, "0");

  if (f.id === "reversal") {
    const xS = f.x.chip.replace(/^NVIDIA\s+(GeForce\s+)?/, "");
    const yS = f.y.chip.replace(/^NVIDIA\s+/, "").replace(/-SXM4-/, "-");
    const gain = Math.round(f.gainA * 100);
    const loss = Math.round(f.lossB * 100);
    const href = `#/compare?runs=${encodeURIComponent([f.x.run_id, f.y.run_id].filter(Boolean).join(","))}`;
    return cardHtml(num, zh ? "跨负载排名反转" : "REGIME REVERSAL", "Suite A ↔ D",
      `+${gain}% → −${loss}%`,
      zh ? `${xS} 赢短对话，在 28K 长上下文反输 ${yS}` : `${xS} wins short chat, flips and loses 28K prefill`,
      zh ? `相同 ${f.framework} 栈下，${xS} 在带宽受限的 Suite A 领先 ${yS} +${gain}%（${fmtNum(Math.round(f.x.a))} vs ${fmtNum(Math.round(f.y.a))} tok/s），但进入算力受限的 28K 长上下文 Suite D 后反转落后 ${loss}%（${f.x.b.toFixed(1)} vs ${f.y.b.toFixed(1)} tok/s）。`
         : `On identical ${f.framework}, ${xS} beats ${yS} by +${gain}% on memory-bandwidth-bound Suite A (${fmtNum(Math.round(f.x.a))} vs ${fmtNum(Math.round(f.y.a))} tok/s), then flips and trails by −${loss}% on compute-bound 28K prefill (${f.x.b.toFixed(1)} vs ${f.y.b.toFixed(1)} tok/s).`,
      [
        { label: `A · ${xS}`, pct: 100, val: `${fmtNum(Math.round(f.x.a))} tok/s`, accent: true },
        { label: `A · ${yS}`, pct: Math.round(f.y.a / f.x.a * 100), val: `${fmtNum(Math.round(f.y.a))} tok/s` },
        { label: `D · ${yS}`, pct: 100, val: `${f.y.b.toFixed(1)} tok/s`, accent: true },
        { label: `D · ${xS}`, pct: Math.round(f.x.b / f.y.b * 100), val: `${f.x.b.toFixed(1)} tok/s` },
      ],
      href, zh ? "对比这组实测" : "Compare runs");
  }

  if (f.id === "sla_gap") {
    const oS = f.offline.chip.replace(/^NVIDIA\s+/, "");
    const sS = f.sla.chip.replace(/^NVIDIA\s+/, "").replace(/\s+HBM3$/, "");
    const ratio = Math.round(f.sla.qps / Math.max(f.offline.qps, 0.1));
    return cardHtml(num, zh ? "离线吞吐 vs SLA 容量" : "OFFLINE VS SLA", "Suite A",
      `${Math.round(f.sla.qps)} vs ${Math.round(f.offline.qps)} QPS`,
      zh ? "离线吞吐第一未能守住 500ms 延迟 SLA" : "Offline leader loses the 500 ms TTFT SLA tier",
      zh ? `${oS} 离线吞吐高达 ${fmtNum(Math.round(f.offline.throughput))} tok/s 位居第一，但 500 ms p99 TTFT 约束下仅撑住 ${Math.round(f.offline.qps)} QPS；同栈下 ${sS} 离线略低却可稳定支撑 ${Math.round(f.sla.qps)} QPS（${ratio}×）。`
         : `${oS} leads offline at ${fmtNum(Math.round(f.offline.throughput))} tok/s yet caps at ${Math.round(f.offline.qps)} QPS under the 500 ms p99 TTFT SLA; ${sS} sustains ${Math.round(f.sla.qps)} QPS (${ratio}× higher) on the same stack.`,
      [
        { label: `${sS} SLA`, pct: 100, val: `${Math.round(f.sla.qps)} QPS`, accent: true },
        { label: `${oS} SLA`, pct: Math.max(12, Math.round(f.offline.qps / f.sla.qps * 100)), val: `${Math.round(f.offline.qps)} QPS` },
      ],
      "#/rankings?suite=suite_A&sort=online_max_qps:desc", zh ? "按 Online QPS 查看" : "Inspect SLA ranking");
  }

  if (f.id === "engine_gap") {
    const cS = f.chip.replace(/^NVIDIA\s+/, "");
    const gain = Math.round(f.gain * 100);
    const href = `#/compare?runs=${encodeURIComponent([f.fast.run_id, f.slow.run_id].filter(Boolean).join(","))}`;
    return cardHtml(num, zh ? "软件栈 > 芯片" : "ENGINE > SILICON", "Suite A",
      `+${gain}%`,
      zh ? `同卡 ${cS} 切换引擎，吞吐提升 +${gain}%` : `Switching engine on ${cS} lifts throughput +${gain}%`,
      zh ? `在同一张 ${cS} 上，从 ${f.slow.framework}（${fmtNum(Math.round(f.slow.throughput))} tok/s）切换到 ${f.fast.framework}（${fmtNum(Math.round(f.fast.throughput))} tok/s），无需换硬件即获得 +${gain}% 提升。`
         : `On the exact same ${cS}, switching ${f.slow.framework} (${fmtNum(Math.round(f.slow.throughput))} tok/s) → ${f.fast.framework} (${fmtNum(Math.round(f.fast.throughput))} tok/s) delivers +${gain}% without touching hardware.`,
      [
        { label: f.fast.framework, pct: 100, val: `${fmtNum(Math.round(f.fast.throughput))} tok/s`, accent: true },
        { label: f.slow.framework, pct: Math.round(f.slow.throughput / f.fast.throughput * 100), val: `${fmtNum(Math.round(f.slow.throughput))} tok/s` },
      ],
      href, zh ? "对比两个引擎" : "Compare engines");
  }

  if (f.id === "scaling") {
    const bS = f.best.chip.replace(/^NVIDIA\s+(GeForce\s+)?/, "");
    const wS = f.worst.chip.replace(/^NVIDIA\s+/, "");
    const bP = (f.best.eff * 100).toFixed(1);
    const wP = (f.worst.eff * 100).toFixed(1);
    return cardHtml(num, zh ? "多卡阿姆达尔定律" : "MULTI-CHIP AMDAHL", "Suite E",
      `${bP}% vs ${wP}%`,
      zh ? "最强旗舰反而多卡扩展效率最低" : "Fastest flagship scales worst at 2× TP",
      zh ? `2× 张量并行（${f.framework}）下，PCIe 上的 ${bS} 保持 ${bP}% 线性效率，而 NVLink 4.0 上的 ${wS} 仅 ${wP}%——因为单卡算力跑得太快，通信占比反而更高。`
         : `At 2× tensor parallelism (${f.framework}), PCIe ${bS} retains ${bP}% linear efficiency while NVLink 4.0 ${wS} retains ${wP}% — faster per-die compute amplifies interconnect Amdahl overhead.`,
      [
        { label: bS, pct: Math.round(f.best.eff * 100), val: `${bP}%`, accent: true },
        { label: wS, pct: Math.round(f.worst.eff * 100), val: `${wP}%` },
      ],
      "#/rankings?suite=suite_E", zh ? "查看 Suite E 扩展榜" : "Inspect Suite E");
  }

  if (f.id === "value") {
    const vS = f.value.chip.replace(/^NVIDIA\s+/, "");
    const fS = f.fastest.chip.replace(/^NVIDIA\s+/, "").replace(/\s+HBM3$/, "");
    return cardHtml(num, zh ? "算力性价比反转" : "COST EFFICIENCY", "Suite F",
      `${f.ratio.toFixed(1)}× tok/$`,
      zh ? "最快的卡很少是每美元产出最高的卡" : "Fastest chip is rarely best tokens-per-dollar",
      zh ? `边缘 Suite F 上，${vS} 达到 ${fmtNum(Math.round(f.value.perDollar))} tok/$/hr，是旗舰 ${fS}（${fmtNum(Math.round(f.fastest.perDollar))} tok/$/hr）单位成本吞吐的 ${f.ratio.toFixed(1)} 倍。`
         : `On edge Suite F, ${vS} delivers ${fmtNum(Math.round(f.value.perDollar))} tok/$/hr — ${f.ratio.toFixed(1)}× the cost efficiency of flagship ${fS} (${fmtNum(Math.round(f.fastest.perDollar))} tok/$/hr).`,
      [
        { label: vS, pct: 100, val: `${fmtNum(Math.round(f.value.perDollar))} tok/$`, accent: true },
        { label: fS, pct: Math.max(10, Math.round(f.fastest.perDollar / f.value.perDollar * 100)), val: `${fmtNum(Math.round(f.fastest.perDollar))} tok/$` },
      ],
      "#/rankings?suite=suite_F", zh ? "查看 Suite F 边缘榜" : "Inspect Suite F");
  }

  if (f.id === "quant") {
    const tS = f.top.chip.replace(/^NVIDIA\s+(GeForce\s+)?/, "");
    const lS = f.low.chip.replace(/^NVIDIA\s+/, "");
    return cardHtml(num, zh ? "量化收益看硬件通路" : "QUANTIZATION PATH", "Suite C",
      `${f.top.speedup.toFixed(2)}× vs ${f.low.speedup.toFixed(2)}×`,
      zh ? "量化加速取决于原生张量核，而非格式标签" : "Quantization speedup lives in silicon, not labels",
      zh ? `Suite C 实测中，${tS} 的 ${f.top.format} 带来 ${f.top.speedup.toFixed(2)}× 加速，而带宽充裕或缺乏原生通路的硬件（如 ${lS} ${f.low.format}）仅 ${f.low.speedup.toFixed(2)}×。`
         : `Across Suite C, best quantization speedup spans ${f.top.speedup.toFixed(2)}× (${tS} · ${f.top.format}) down to ${f.low.speedup.toFixed(2)}× (${lS} · ${f.low.format}). Always verify the MMLU accuracy gate.`,
      [
        { label: `${tS} ${f.top.format}`, pct: 100, val: `${f.top.speedup.toFixed(2)}×`, accent: true },
        { label: `${lS} ${f.low.format}`, pct: Math.round(f.low.speedup / f.top.speedup * 100), val: `${f.low.speedup.toFixed(2)}×` },
      ],
      "#/rankings?suite=suite_C", zh ? "查看 Suite C 量化榜" : "Inspect Suite C");
  }

  return "";
}

function cardHtml(num, kicker, tag, stat, head, body, bars, href, cta) {
  return `
    <article class="f-card">
      <div class="f-card-top">
        <span class="f-kicker mono">${num} · ${esc(kicker)}</span>
        <span class="f-tag mono">${esc(tag)}</span>
      </div>
      <div class="f-stat mono tnum">${esc(stat)}</div>
      <div class="f-head">${esc(head)}</div>
      <p class="f-body">${esc(body)}</p>
      <div class="f-bars">
        ${bars.map((b) => `
          <div class="f-bar-row">
            <span class="mono" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(b.label)}</span>
            <div class="f-track"><div class="f-fill ${b.accent ? "accent" : "muted"}" style="width:${b.pct}%"></div></div>
            <span class="mono">${esc(b.val)}</span>
          </div>
        `).join("")}
      </div>
      <a class="f-cta" href="${esc(href)}">${esc(cta)} ${icon("arrowRight", 13)}</a>
    </article>
  `;
}

export function render({ el }) {
  const rows = allRows();
  const findings = computeInsights(rows);
  const zh = isZh();

  const matrix = [
    {
      regime: zh ? "低延迟在线对话 (SLA)" : "Latency-SLA Chat",
      bn: zh ? "KV 容量与批调度" : "KV capacity & scheduler",
      best: "H100 80GB · SGLang / vLLM",
      watch: zh ? "24–32 GB 消费卡在泊松高并发下易触发 500ms TTFT 超时" : "24–32 GB consumer GPUs breach 500 ms p99 TTFT under Poisson load",
      href: "#/rankings?suite=suite_A&sort=online_max_qps:desc",
      cta: "Suite A",
    },
    {
      regime: zh ? "28K 长上下文 / RAG" : "Long-Context / RAG (28K)",
      bn: zh ? "算力受限 Prefill" : "Compute-bound prefill",
      best: "B200, H200, H100 (high BF16 FLOPS)",
      watch: zh ? "RTX 5090 与 H20-3e 预填充掉至 A100 级别或更低" : "RTX 5090 & H20-3e drop to or below A100-class prefill",
      href: "#/rankings?suite=suite_D",
      cta: "Suite D",
    },
    {
      regime: zh ? "离线批处理 / 边缘部署" : "Offline Batch / Edge",
      bn: zh ? "单卡带宽与 $/hr" : "Memory bandwidth & $/hr",
      best: "RTX 5090, RTX 4090 D, Tesla T4",
      watch: zh ? "数据中心旗舰卡单位美元吞吐贵 3–6×" : "Datacenter flagships pay a 3–6× tokens-per-dollar penalty",
      href: "#/rankings?suite=suite_F",
      cta: "Suite F",
    },
    {
      regime: zh ? "稀疏 MoE (Mixtral-8x7B)" : "Sparse MoE (Mixtral-8x7B)",
      bn: zh ? "聚合 HBM 显存带宽" : "Aggregate HBM bandwidth",
      best: "H200 ×8, H20-3e ×8",
      watch: zh ? "中端算力卡与早期多卡软件栈路由效率低" : "Mid-tier compute SKUs & immature multi-chip stacks",
      href: "#/rankings?suite=suite_G",
      cta: "Suite G",
    },
    {
      regime: zh ? "稠密 <70B 多卡扩容" : "Dense <70B Multi-Chip",
      bn: zh ? "All-reduce 通信开销" : "All-reduce communication",
      best: zh ? "优先数据并行副本而非 TP > 4" : "Data-parallel replicas over TP > 4",
      watch: zh ? "8 路 TP 即使在 NVLink 上也会损失 >50% 单卡效率" : "8-way TP loses >50% per-chip efficiency even on NVLink",
      href: "#/rankings?suite=suite_E",
      cta: "Suite E",
    },
    {
      regime: zh ? "低精度量化部署" : "Quantized Serving",
      bn: zh ? "原生低精度张量核通路" : "Native tensor-core path",
      best: "Native FP8 (Ada/Hopper/Blackwell) · W8A8 (Ampere)",
      watch: zh ? "Ampere 模拟 FP8 几乎无加速；务必检查 MMLU 精度门禁" : "Emulated FP8 on Ampere gives ~0% gain; verify MMLU gate",
      href: "#/rankings?suite=suite_C",
      cta: "Suite C",
    },
  ];

  el.innerHTML = `
    <section class="page-head-left">
      <span class="eyebrow">${zh ? "实证发现" : "Empirical Findings"}</span>
      <h1>${zh ? "六个改写硬件选型的实测结论" : "Six structural findings from live data"}</h1>
      <p class="page-lead">${zh
        ? "以下每个结论均直接由榜单上的已验证参考栈（Reference / Verified）实时计算生成，点击任意卡片即可核查原始运行记录。"
        : "Computed automatically from verified & reference submissions on this leaderboard. Click any card to inspect the underlying receipts."}</p>
    </section>

    <section class="section">
      <div class="findings-grid-3">
        ${findings.map((f, i) => renderFindingBlock(f, i)).join("")}
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${zh ? "选型指南" : "Procurement Guide"}</span>
          <h2>${zh ? "哪种推理方案最匹配你的业务负载？" : "Which serving recipe fits your workload?"}</h2>
        </div>
      </div>
      <div class="matrix-card">
        <table class="matrix-tbl">
          <thead>
            <tr>
              <th>${zh ? "业务负载" : "Workload Regime"}</th>
              <th>${zh ? "推荐方案" : "Best-Matched Recipe"}</th>
              <th>${zh ? "常见陷阱" : "Watch Out For"}</th>
              <th>${zh ? "实时榜单" : "Ranking"}</th>
            </tr>
          </thead>
          <tbody>
            ${matrix.map((r) => `
              <tr>
                <td>
                  <div class="m-regime">${esc(r.regime)}</div>
                  <div class="m-bn">${esc(r.bn)}</div>
                </td>
                <td class="m-best">${esc(r.best)}</td>
                <td class="m-watch">${esc(r.watch)}</td>
                <td><a class="m-link mono" href="${esc(r.href)}">${esc(r.cta)} →</a></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;
}
