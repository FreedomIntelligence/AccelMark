// cite.js — citation & share card builders for individual benchmark runs.

import { SUITE_META } from "./data.js";
import { submitterHandle } from "./utils.js";

const REPO = "https://github.com/FreedomIntelligence/AccelMark";

/** Canonical in-app URL for a run (deep-links the result modal). */
export function runCanonicalUrl(runId, path = "/rankings") {
  const base = typeof location !== "undefined"
    ? `${location.origin}${location.pathname}`
    : REPO;
  return `${base}#${path}?run=${encodeURIComponent(runId)}`;
}

/** GitHub path to result.json for a submission folder name. */
export function resultJsonUrl(submission) {
  if (!submission) return REPO;
  return `${REPO}/blob/main/results/community/${submission}/result.json`;
}

function suiteLabel(row) {
  const meta = SUITE_META[row.suite];
  return meta ? `Suite ${meta.letter} · ${meta.title}` : row.suite || "—";
}

function primaryMetric(row) {
  const v = row.primary_metric ?? row.offline_throughput ?? row.primary_value;
  const u = row.primary_metric_label || row.metric_unit || "tokens/s";
  if (v == null) return "—";
  return `${v} ${u}`;
}

function runnerId(row) {
  return row.runner_id || row.impl?.runner_id ||
    (row.reproduce_script ? row.reproduce_script.split("/")[1] : "—");
}

/** Markdown share card (GitHub, blogs, papers). */
export function runMarkdownCard(row) {
  const handle = submitterHandle(row.submitted_by);
  const lines = [
    `**AccelMark result:** ${row.chip || "—"} on ${suiteLabel(row)}`,
    `- Throughput: **${primaryMetric(row)}**`,
    `- Runner: \`${runnerId(row)}\``,
    `- Tier: **${row.tier || "community"}**`,
    `- Result ID: \`${row.run_id || "—"}\``,
    `- Submitted by: ${handle ? `@${handle}` : "community"}`,
    `- Reproduce: ${runCanonicalUrl(row.run_id)}`,
  ];
  if (row.submission) {
    lines.push(`- Artifacts: ${resultJsonUrl(row.submission)}`);
  }
  return lines.join("\n");
}

/** Plain-text one-liner citation. */
export function runPlainCitation(row, accessed = new Date()) {
  const handle = submitterHandle(row.submitted_by) || "community";
  const date = accessed.toISOString().slice(0, 10);
  return (
    `AccelMark result ${row.run_id}, ${row.chip}, ${suiteLabel(row)}, ` +
    `${primaryMetric(row)}, submitted by ${handle}, accessed ${date}. ` +
    runCanonicalUrl(row.run_id)
  );
}

/** BibTeX for a single run. */
export function runBibTeX(row) {
  const id = (row.run_id || "run").replace(/[^a-zA-Z0-9_]/g, "_");
  const handle = submitterHandle(row.submitted_by) || "community";
  const runner = runnerId(row);
  const sha = row.impl?.git_sha || row.detail?.meta_git_sha || "—";
  return `@misc{accelmark_${id},
  title        = {AccelMark result: ${row.chip} on ${suiteLabel(row)}},
  author       = {${handle}},
  year         = {${(row.date || "2026").slice(0, 4)}},
  howpublished = {AccelMark benchmark result},
  url          = {${runCanonicalUrl(row.run_id)}},
  note         = {Runner: ${runner}; Tier: ${row.tier || "community"}; License: CC-BY-4.0}
}`;
}

/** Project-level BibTeX (from README). */
export const PROJECT_BIBTEX = `@misc{accelmark2026,
  title  = {Beyond NVIDIA! A Multi-Regime Framework for Benchmarking Heterogeneous AI Accelerators},
  author = {Liang, Juhao and Zhang, Zhiyuan and Li, Siyu and Lin, Zhihang and Yu, Minchen and Zeng, Li and Chen, Zizhong and Sun, Ruoyu and Wang, Benyou},
  year   = {2026},
  url    = {https://github.com/FreedomIntelligence/AccelMark}
}`;

/** Dataset snapshot BibTeX template. */
export function datasetBibTeX(version, url) {
  return `@misc{accelmark_results_${version.replace(/\./g, "_")},
  title        = {AccelMark Results Dataset v${version}},
  author       = {AccelMark Contributors},
  year         = {${version.slice(0, 4)}},
  howpublished = {Open benchmark dataset},
  url          = {${url || REPO}},
  note         = {License: CC-BY-4.0; Generated from community and verified submissions}
}`;
}

const DISCUSS_BASE = "https://github.com/FreedomIntelligence/AccelMark/discussions/new";

function resultContextBlock(row) {
  const handle = submitterHandle(row.submitted_by);
  return (
    `## Hardware\n${row.chip}${row.chip_count > 1 ? ` ×${row.chip_count}` : ""}` +
    (row.vendor ? ` · ${row.vendor}` : "") +
    `\n\n## Suite\n${suiteLabel(row)}\n\n` +
    `## Result\n- Throughput: ${primaryMetric(row)}\n- Runner: \`${runnerId(row)}\`\n` +
    `- Tier: ${row.tier || "community"}\n- Run ID: \`${row.run_id || "—"}\`\n` +
    (handle ? `- Submitted by: @${handle}\n` : "") +
    `\n## Link\n${runCanonicalUrl(row.run_id)}\n`
  );
}

function discussLink(category, title, body) {
  return `${DISCUSS_BASE}?category=${category}&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}

/** All discussion template entry points for a run. */
export const DISCUSS_TEMPLATES = [
  {
    id: "show-and-tell",
    label: "Show & tell",
    short: "Share tips",
    category: "show-and-tell",
    title: (row) => `[Show and tell] ${row.chip} · ${suiteLabel(row)}`,
    body: (row) =>
      resultContextBlock(row) +
      `\n## What I ran\n(Hardware setup, runner version, anything non-obvious)\n\n` +
      `## What worked / what didn't\n\n` +
      `## Tips for others\n`,
  },
  {
    id: "ask-submitter",
    label: "Ask submitter",
    short: "Ask @them",
    category: "q-a",
    title: (row) => `[Q&A] Question about ${row.chip} · ${suiteLabel(row)}`,
    body: (row) => {
      const handle = submitterHandle(row.submitted_by);
      return (
        resultContextBlock(row) +
        `\n## Question for ${handle ? `@${handle}` : "the submitter"}\n` +
        `(Your question about configuration, environment, or reproduction)\n\n` +
        `## My setup\n(Optional — helps them answer)\n`
      );
    },
  },
  {
    id: "reproduction",
    label: "Request reproduction",
    short: "Reproduce",
    category: "q-a",
    title: (row) => `[Reproduction request] ${row.chip} · ${suiteLabel(row)}`,
    body: (row) =>
      resultContextBlock(row) +
      `\n## Request\nWho has matching hardware to independently reproduce this **community** result?\n\n` +
      `## Matching criteria\n- Chip: ${row.chip}\n- Suite: ${suiteLabel(row)}\n` +
      `- Runner: \`${runnerId(row)}\`\n\n` +
      `## Offer\nI'll help validate / review if you open a PR referencing run \`${row.run_id}\`.\n`,
  },
  {
    id: "optimization",
    label: "Optimization notes",
    short: "Optimize",
    category: "ideas",
    title: (row) => `[Optimization] ${row.chip} · ${suiteLabel(row)} · ${primaryMetric(row)}`,
    body: (row) =>
      resultContextBlock(row) +
      `\n## Changes tried\n(Parameters, kernels, batch sizes, quantization, etc.)\n\n` +
      `## Throughput impact\n(Before → after, same accuracy gate)\n\n` +
      `## Accuracy impact\n\n` +
      `## Reproducible?\n(Yes / partial / no — link to branch or config if yes)\n`,
  },
  {
    id: "report",
    label: "Report concern",
    short: "Report",
    category: "q-a",
    title: (row) => `[Review] Concern about ${row.chip} · ${row.run_id}`,
    body: (row) =>
      resultContextBlock(row) +
      `\n## Concern\n(Describe what looks inconsistent — env mismatch, outlier metric, missing artifacts, etc.)\n\n` +
      `## Evidence\n(Links to your reproduction attempt, logs, or diff)\n\n` +
      `## Suggested action\n(Re-run, downgrade tier, request env_info clarification, etc.)\n`,
  },
];

export function discussResultUrl(row, templateId = "show-and-tell") {
  const tpl = DISCUSS_TEMPLATES.find((t) => t.id === templateId) || DISCUSS_TEMPLATES[0];
  return discussLink(tpl.category, tpl.title(row), tpl.body(row));
}

export function discussTemplateById(id) {
  return DISCUSS_TEMPLATES.find((t) => t.id === id) || null;
}
