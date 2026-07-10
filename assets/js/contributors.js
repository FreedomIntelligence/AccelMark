// contributors.js — aggregate submitter stats, badges, and leaderboard ranking.
// Profiles are keyed by GitHub handle parsed from submitted_by — no site registration.

import { init as initData } from "./data.js";
import { submitterHandle } from "./utils.js";
import { normChipKey, flattenCatalog } from "./hardware-catalog.js";

export const BADGE_DEFS = {
  "first-result": { id: "first-result", label: "First result", desc: "First published benchmark for a hardware platform" },
  pioneer:        { id: "pioneer",        label: "Pioneer",        desc: "Among the first three contributors on the leaderboard" },
  verified:       { id: "verified",       label: "Verified",       desc: "At least one verified-tier result" },
  verifier:       { id: "verifier",       label: "Verifier",       desc: "Independent reproduction published as verified" },
  prolific:       { id: "prolific",       label: "Prolific",       desc: "Ten or more published benchmark runs" },
  "multi-chip":   { id: "multi-chip",     label: "Multi-platform", desc: "Results on three or more distinct hardware platforms" },
  "multi-vendor": { id: "multi-vendor",   label: "Cross-vendor",   desc: "Results across two or more vendors" },
  runner:         { id: "runner",         label: "Runner author",  desc: "Credited on a runner implementation" },
};

let _index = null;
let _firstByChip = null;

function rows() {
  initData();
  const data = (typeof window !== "undefined" && Array.isArray(window.LEADERBOARD_DATA))
    ? window.LEADERBOARD_DATA
    : [];
  return data;
}

function submitterFromRow(r) {
  return submitterHandle(r.submitted_by || r.detail?.meta_submitted_by);
}

function computeFirstByChip() {
  if (_firstByChip) return _firstByChip;
  const map = new Map();
  for (const r of rows()) {
    const chipKey = normChipKey(r.chip);
    if (!chipKey) continue;
    const handle = submitterFromRow(r);
    if (!handle) continue;
    const prev = map.get(chipKey);
    const date = String(r.date || "");
    if (!prev || date < prev.date) {
      map.set(chipKey, { handle, date, chip: r.chip });
    }
  }
  _firstByChip = map;
  return map;
}

function badgeListFor(stats) {
  const badges = [];
  if (stats.firstChips.length) badges.push("first-result");
  if (stats.verified > 0) badges.push("verified");
  if (stats.verified > 0) badges.push("verifier");
  if (stats.total >= 10) badges.push("prolific");
  if (stats.uniqueChips.size >= 3) badges.push("multi-chip");
  if (stats.vendors.size >= 2) badges.push("multi-vendor");
  if (stats.runnerCredits > 0) badges.push("runner");
  if (stats.rank <= 3 && stats.total > 0) badges.push("pioneer");
  return [...new Set(badges)];
}

function scoreFor(stats) {
  return (
    stats.total +
    stats.verified * 3 +
    stats.uniqueChips.size * 4 +
    stats.firstChips.length * 12 +
    stats.runnerCredits * 5
  );
}

export function contributorIndex() {
  if (_index) return _index;

  const firstByChip = computeFirstByChip();
  const byHandle = new Map();

  for (const r of rows()) {
    const handle = submitterFromRow(r);
    if (!handle) continue;

    if (!byHandle.has(handle)) {
      byHandle.set(handle, {
        handle,
        githubUrl: `https://github.com/${encodeURIComponent(handle)}`,
        total: 0,
        verified: 0,
        community: 0,
        vendors: new Set(),
        chips: new Set(),
        uniqueChips: new Set(),
        suites: new Set(),
        firstChips: [],
        runs: [],
        runnerCredits: 0,
        latestDate: "",
      });
    }
    const c = byHandle.get(handle);
    c.total += 1;
    if (r.tier === "verified") c.verified += 1;
    else c.community += 1;
    if (r.vendor) c.vendors.add(r.vendor);
    if (r.chip) {
      c.chips.add(r.chip);
      c.uniqueChips.add(normChipKey(r.chip));
    }
    if (r.suite) c.suites.add(r.suite);
    if (r.date && String(r.date) > c.latestDate) c.latestDate = String(r.date);
    c.runs.push({
      run_id: r.run_id,
      chip: r.chip,
      chip_count: r.chip_count,
      vendor: r.vendor,
      suite: r.suite,
      tier: r.tier,
      date: r.date,
      submitted_by: r.submitted_by,
    });

    const chipKey = normChipKey(r.chip);
    const first = firstByChip.get(chipKey);
    if (first && first.handle === handle && !c.firstChips.includes(r.chip)) {
      c.firstChips.push(r.chip);
    }

    const implBy = submitterHandle(r.impl?.submitted_by);
    if (implBy === handle) c.runnerCredits += 1;
  }

  const list = [...byHandle.values()];
  list.sort((a, b) => scoreFor(b) - scoreFor(a) || b.total - a.total || a.handle.localeCompare(b.handle));
  list.forEach((c, i) => {
    c.rank = i + 1;
    c.score = scoreFor(c);
    c.badges = badgeListFor(c);
  });

  _index = list;
  return list;
}

export function contributorByHandle(handle) {
  const h = submitterHandle(handle);
  if (!h) return null;
  return contributorIndex().find((c) => c.handle.toLowerCase() === h.toLowerCase()) || null;
}

export function firstResultBadgeForChip(chipName) {
  const key = normChipKey(chipName);
  const first = computeFirstByChip().get(key);
  return first ? first.handle : null;
}

export function reproductionQuestExplain() {
  const all = rows();
  const verifiedKeys = new Set(
    all
      .filter((r) => r.tier === "verified")
      .map((r) => `${normChipKey(r.chip)}|${r.suite}`)
  );
  const communityRuns = all.filter((r) => r.tier !== "verified");
  const openQuests = communityRuns.filter(
    (r) => !verifiedKeys.has(`${normChipKey(r.chip)}|${r.suite}`)
  );
  const openPairs = new Set(openQuests.map((r) => `${normChipKey(r.chip)}|${r.suite}`));
  return {
    totalRuns: all.length,
    communityRuns: communityRuns.length,
    verifiedPairs: verifiedKeys.size,
    openQuestCount: openQuests.length,
    openPairCount: openPairs.size,
    crossVerifiedRecordCount: crossVerifiedRecords(9999).length,
  };
}

/**
 * Explicit verification links only — a row appears when meta.reproduces_run_id
 * points at an existing run (set via --reproduces-run-id on the benchmark runner
 * or in result.json meta). No chip+suite guessing.
 */
export function crossVerifiedRecords(limit = 100) {
  const all = rows();
  const byRunId = new Map();
  for (const r of all) {
    if (r.run_id) byRunId.set(r.run_id, r);
  }

  function rowSnapshot(r) {
    return {
      run_id: r.run_id,
      chip: r.chip,
      vendor: r.vendor,
      suite: r.suite,
      framework: r.framework,
      framework_version: r.framework_version,
      precision: r.precision,
      tier: r.tier,
      submitted_by: submitterFromRow(r),
      date: r.date,
    };
  }

  const records = [];
  for (const r of all) {
    const reproId = r.reproduces_run_id || r.detail?.meta_reproduces_run_id;
    if (!reproId) continue;
    const original = byRunId.get(reproId);
    if (!original) continue;
    records.push({
      reproduces_run_id: reproId,
      original: rowSnapshot(original),
      verification: rowSnapshot(r),
    });
  }

  records.sort((a, c) =>
    String(c.verification.date).localeCompare(String(a.verification.date))
  );
  return records.slice(0, limit);
}

/**
 * Community rows lacking a verified sibling on the same platform + suite.
 * Key = normalized chip name + suite id; does not require matching runner or chip count.
 * Sorted newest-first; not a manual curator pick.
 */
export function reproductionQuests(limit = 24) {
  const all = rows();
  const verifiedKeys = new Set(
    all
      .filter((r) => r.tier === "verified")
      .map((r) => `${normChipKey(r.chip)}|${r.suite}`)
  );

  return all
    .filter((r) => r.tier !== "verified")
    .filter((r) => !verifiedKeys.has(`${normChipKey(r.chip)}|${r.suite}`))
    .sort((a, b) => String(b.date).localeCompare(String(a.date)))
    .slice(0, limit)
    .map((r) => ({
      run_id: r.run_id,
      chip: r.chip,
      chip_count: r.chip_count,
      vendor: r.vendor,
      suite: r.suite,
      framework: r.framework,
      framework_version: r.framework_version,
      precision: r.precision,
      tier: r.tier || "community",
      submitted_by: submitterFromRow(r),
      date: r.date,
    }));
}

export function _resetContributorCache() {
  _index = null;
  _firstByChip = null;
}

export function catalogGapCount() {
  const live = rows().map((r) => r.chip);
  const wanted = flattenCatalog().filter((w) => {
    const key = normChipKey(w.name);
    return !live.some((l) => {
      const lk = normChipKey(l);
      return lk === key || lk.includes(key) || key.includes(lk);
    });
  });
  return wanted.length;
}
