// hardware-catalog.js — accelerator catalog for Submit wizard & Wanted hardware.
// Curated from public vendor specs (NVIDIA, AMD, Intel, Huawei, Apple, Google,
// Moore Threads, domestic Chinese GPUs, inference ASICs). Names reflect 2025–26
// product lines; memory figures are typical HBM/VRAM configs for benchmarking.

/** Vendor metadata + default runner hint (may be overridden per chip). */
export const VENDORS = [
  { id: "NVIDIA", label: "NVIDIA", runner: "nvidia_vllm_47f5d58e" },
  { id: "AMD", label: "AMD", runner: "amd_vllm_rocm_6c18cd8f" },
  { id: "Huawei", label: "Huawei Ascend", runner: "ascend_vllm_ascend_d4aa9fda" },
  { id: "Intel", label: "Intel", runner: "nvidia_vllm_47f5d58e" },
  { id: "Apple", label: "Apple Silicon", runner: "apple_mlx_lm_9546b8b5" },
  { id: "Google", label: "Google TPU", runner: "google_vllm_tpu_68cc9ffa" },
  { id: "Moore Threads", label: "Moore Threads", runner: "moorethreads_vllm_musa_f2f6f965" },
  { id: "Qualcomm", label: "Qualcomm", runner: "nvidia_vllm_47f5d58e" },
  { id: "SambaNova", label: "SambaNova", runner: "nvidia_vllm_47f5d58e" },
  { id: "Graphcore", label: "Graphcore", runner: "nvidia_vllm_47f5d58e" },
  { id: "Hygon", label: "Hygon", runner: "amd_vllm_rocm_6c18cd8f" },
  { id: "Iluvatar", label: "Iluvatar", runner: "nvidia_vllm_47f5d58e" },
  { id: "Cambricon", label: "Cambricon", runner: "nvidia_vllm_47f5d58e" },
  { id: "Enflame", label: "Enflame", runner: "nvidia_vllm_47f5d58e" },
  { id: "Other", label: "Other / emerging", runner: "nvidia_vllm_47f5d58e" },
];

/**
 * @typedef {Object} ChipEntry
 * @property {string} name
 * @property {number} [memoryGb]
 * @property {"datacenter"|"workstation"|"consumer"|"cloud"|"edge"} tier
 * @property {string} [badge] — editorial tag for Wanted page
 * @property {string[]} [suites] — recommended starter suites
 */

/** @type {Record<string, ChipEntry[]>} */
export const CHIPS_BY_VENDOR = {
  NVIDIA: [
    { name: "B200", memoryGb: 192, tier: "datacenter", badge: "Blackwell flagship", suites: ["suite_B", "suite_G"] },
    { name: "GB200 NVL72", memoryGb: 192, tier: "datacenter", badge: "Rack-scale Blackwell", suites: ["suite_B", "suite_G"] },
    { name: "H200", memoryGb: 141, tier: "datacenter", badge: "Long-context HBM", suites: ["suite_A", "suite_D"] },
    { name: "H100 80GB", memoryGb: 80, tier: "datacenter", badge: "Industry standard", suites: ["suite_A", "suite_B"] },
    { name: "H20 96GB", memoryGb: 96, tier: "datacenter", badge: "China export tier", suites: ["suite_A", "suite_B"] },
    { name: "H800 80GB", memoryGb: 80, tier: "datacenter", suites: ["suite_A", "suite_B"] },
    { name: "A100 80GB", memoryGb: 80, tier: "datacenter", suites: ["suite_A", "suite_B"] },
    { name: "A100 40GB", memoryGb: 40, tier: "datacenter", suites: ["suite_A"] },
    { name: "A800 80GB", memoryGb: 80, tier: "datacenter", suites: ["suite_A", "suite_H"] },
    { name: "L40S", memoryGb: 48, tier: "datacenter", badge: "Cost-efficient inference", suites: ["suite_A", "suite_F"] },
    { name: "L4", memoryGb: 24, tier: "datacenter", badge: "Edge inference", suites: ["suite_F"] },
    { name: "A10", memoryGb: 24, tier: "datacenter", suites: ["suite_F"] },
    { name: "T4", memoryGb: 16, tier: "datacenter", suites: ["suite_F"] },
    { name: "RTX PRO 6000 Blackwell", memoryGb: 96, tier: "workstation", badge: "Pro workstation", suites: ["suite_A", "suite_F"] },
    { name: "RTX 6000 Ada", memoryGb: 48, tier: "workstation", suites: ["suite_A", "suite_F"] },
    { name: "RTX 5090", memoryGb: 32, tier: "consumer", badge: "Blackwell consumer", suites: ["suite_F", "suite_A"] },
    { name: "RTX 5080", memoryGb: 16, tier: "consumer", badge: "New consumer GPU", suites: ["suite_F"] },
    { name: "RTX 5070 Ti", memoryGb: 16, tier: "consumer", suites: ["suite_F"] },
    { name: "RTX 4090", memoryGb: 24, tier: "consumer", suites: ["suite_F"] },
    { name: "RTX 4080 Super", memoryGb: 16, tier: "consumer", suites: ["suite_F"] },
    { name: "RTX 3090", memoryGb: 24, tier: "consumer", suites: ["suite_F"] },
    { name: "DGX Spark", memoryGb: 128, tier: "edge", badge: "Desk-side dev kit", suites: ["suite_F"] },
  ],
  AMD: [
    { name: "MI355X", memoryGb: 288, tier: "datacenter", badge: "CDNA 4 flagship", suites: ["suite_B", "suite_G"] },
    { name: "MI350X", memoryGb: 288, tier: "datacenter", badge: "CDNA 4", suites: ["suite_B", "suite_E"] },
    { name: "MI325X", memoryGb: 256, tier: "datacenter", badge: "Datacenter gap", suites: ["suite_B", "suite_E"] },
    { name: "MI300X", memoryGb: 192, tier: "datacenter", suites: ["suite_A", "suite_B"] },
    { name: "MI300A", memoryGb: 128, tier: "datacenter", suites: ["suite_A"] },
    { name: "MI250X", memoryGb: 128, tier: "datacenter", suites: ["suite_B"] },
    { name: "MI210", memoryGb: 64, tier: "datacenter", suites: ["suite_A"] },
    { name: "Instinct MI400 (roadmap)", memoryGb: 432, tier: "datacenter", badge: "2026 roadmap", suites: ["suite_B"] },
    { name: "RX 7900 XTX", memoryGb: 24, tier: "consumer", suites: ["suite_F"] },
    { name: "RX 9070 XT", memoryGb: 16, tier: "consumer", badge: "RDNA 4", suites: ["suite_F"] },
  ],
  Huawei: [
    { name: "Ascend 910C", memoryGb: 96, tier: "datacenter", badge: "Multi-die flagship", suites: ["suite_B", "suite_G"] },
    { name: "Ascend 910B", memoryGb: 64, tier: "datacenter", suites: ["suite_A", "suite_B"] },
    { name: "Ascend 910", memoryGb: 32, tier: "datacenter", suites: ["suite_A"] },
    { name: "Ascend 310P", memoryGb: 8, tier: "edge", suites: ["suite_F"] },
    { name: "Ascend 950PR (roadmap)", memoryGb: 128, tier: "datacenter", badge: "2026 roadmap", suites: ["suite_B"] },
  ],
  Intel: [
    { name: "Gaudi 3", memoryGb: 128, tier: "datacenter", badge: "Emerging accelerator", suites: ["suite_A", "suite_B"] },
    { name: "Gaudi 2", memoryGb: 96, tier: "datacenter", suites: ["suite_A"] },
    { name: "Max 1550", memoryGb: 128, tier: "datacenter", badge: "Ponte Vecchio", suites: ["suite_B"] },
    { name: "Arc Pro B50", memoryGb: 24, tier: "workstation", suites: ["suite_F"] },
  ],
  Apple: [
    { name: "M4 Ultra", memoryGb: 192, tier: "consumer", badge: "Consumer / edge", suites: ["suite_F"] },
    { name: "M4 Max", memoryGb: 128, tier: "consumer", suites: ["suite_F"] },
    { name: "M3 Ultra", memoryGb: 192, tier: "consumer", suites: ["suite_F"] },
    { name: "M3 Max", memoryGb: 128, tier: "consumer", suites: ["suite_F"] },
    { name: "M2 Ultra", memoryGb: 192, tier: "consumer", suites: ["suite_F"] },
    { name: "M2 Max", memoryGb: 96, tier: "consumer", suites: ["suite_F"] },
  ],
  Google: [
    { name: "TPU v6e Trillium", memoryGb: 32, tier: "cloud", badge: "Cloud TPU", suites: ["suite_A", "suite_B"] },
    { name: "TPU v5e", memoryGb: 16, tier: "cloud", suites: ["suite_A"] },
    { name: "TPU v5p", memoryGb: 95, tier: "cloud", suites: ["suite_B"] },
    { name: "TPU v4", memoryGb: 32, tier: "cloud", suites: ["suite_A"] },
  ],
  "Moore Threads": [
    { name: "MTT S4000", memoryGb: 48, tier: "datacenter", suites: ["suite_A", "suite_B"] },
    { name: "MTT S80", memoryGb: 16, tier: "consumer", suites: ["suite_F"] },
    { name: "MTT S3000", memoryGb: 32, tier: "datacenter", suites: ["suite_A"] },
  ],
  Qualcomm: [
    { name: "Cloud AI 100 Ultra", memoryGb: 64, tier: "datacenter", badge: "Inference ASIC", suites: ["suite_A"] },
    { name: "Cloud AI 100", memoryGb: 32, tier: "datacenter", suites: ["suite_A"] },
  ],
  SambaNova: [
    { name: "SN40L RDU", memoryGb: 64, tier: "datacenter", badge: "Enterprise AI", suites: ["suite_B"] },
    { name: "SN30", memoryGb: 32, tier: "datacenter", suites: ["suite_A"] },
  ],
  Graphcore: [
    { name: "C600 IPU", memoryGb: 64, tier: "datacenter", badge: "Novel architecture", suites: ["suite_A"] },
    { name: "Bow-2000", memoryGb: 36, tier: "datacenter", suites: ["suite_A"] },
  ],
  Hygon: [
    { name: "DCU Z100", memoryGb: 32, tier: "datacenter", badge: "Domestic GPU", suites: ["suite_A", "suite_B"] },
    { name: "DCU K100", memoryGb: 64, tier: "datacenter", badge: "Domestic GPU", suites: ["suite_A"] },
  ],
  Iluvatar: [
    { name: "BI-V150", memoryGb: 32, tier: "datacenter", badge: "Domestic GPU", suites: ["suite_A"] },
    { name: "BI-V100", memoryGb: 32, tier: "datacenter", suites: ["suite_A"] },
  ],
  Cambricon: [
    { name: "MLU590", memoryGb: 64, tier: "datacenter", badge: "Domestic NPU", suites: ["suite_A", "suite_B"] },
    { name: "MLU370", memoryGb: 24, tier: "datacenter", suites: ["suite_A"] },
  ],
  Enflame: [
    { name: "i20", memoryGb: 32, tier: "datacenter", badge: "Domestic AI chip", suites: ["suite_A"] },
    { name: "T20", memoryGb: 64, tier: "datacenter", suites: ["suite_B"] },
  ],
  Other: [
    { name: "Custom accelerator", tier: "datacenter", suites: ["suite_A"] },
  ],
};

const TIER_LABELS = {
  datacenter: "Datacenter",
  workstation: "Workstation",
  consumer: "Consumer / edge",
  cloud: "Cloud",
  edge: "Edge",
};

/** Normalize chip name for fuzzy matching against leaderboard labels. */
export function normChipKey(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/\([^)]*\)/g, "")
    .replace(/[^a-z0-9]/g, "");
}

/** True if catalog chip appears covered by any live leaderboard chip label. */
export function chipIsCovered(catalogChip, liveChipKeys) {
  const key = normChipKey(catalogChip);
  if (!key) return false;
  for (const live of liveChipKeys) {
    if (live === key || live.includes(key) || key.includes(live)) return true;
  }
  return false;
}

export function chipsForVendor(vendorId) {
  return CHIPS_BY_VENDOR[vendorId] || CHIPS_BY_VENDOR.Other;
}

export function vendorById(id) {
  return VENDORS.find((v) => v.id === id) || VENDORS.find((v) => v.id === "Other");
}

export function tierLabel(tier) {
  return TIER_LABELS[tier] || tier || "Other";
}

/** Flat list with vendor id attached — for Wanted page & search. */
export function flattenCatalog() {
  const out = [];
  for (const v of VENDORS) {
    if (v.id === "Other") continue;
    for (const chip of chipsForVendor(v.id)) {
      if (chip.name === "Custom accelerator") continue;
      out.push({
        vendor: v.id,
        vendorLabel: v.label,
        runner: v.runner,
        ...chip,
      });
    }
  }
  return out;
}

/** Catalog entries not yet represented on the live leaderboard. */
export function wantedFromCatalog(liveChipLabels) {
  const liveKeys = (liveChipLabels || []).map(normChipKey);
  return flattenCatalog()
    .filter((w) => !chipIsCovered(w.name, liveKeys))
    .sort((a, b) => {
      const tierOrder = { datacenter: 0, cloud: 1, workstation: 2, consumer: 3, edge: 4 };
      const ta = tierOrder[a.tier] ?? 5;
      const tb = tierOrder[b.tier] ?? 5;
      if (ta !== tb) return ta - tb;
      return a.vendor.localeCompare(b.vendor) || a.name.localeCompare(b.name);
    });
}

/** Build grouped <option> HTML for a vendor's chips (Submit wizard). */
export function chipOptionsHtml(vendorId, selectedName) {
  const chips = chipsForVendor(vendorId);
  const groups = new Map();
  for (const c of chips) {
    const tier = c.tier || "datacenter";
    if (!groups.has(tier)) groups.set(tier, []);
    groups.get(tier).push(c);
  }
  const order = ["datacenter", "cloud", "workstation", "consumer", "edge"];
  let html = "";
  for (const tier of order) {
    const list = groups.get(tier);
    if (!list?.length) continue;
    html += `<optgroup label="${tierLabel(tier)}">`;
    for (const c of list) {
      const mem = c.memoryGb ? ` · ${c.memoryGb} GB` : "";
      const sel = c.name === selectedName ? " selected" : "";
      html += `<option value="${c.name.replace(/"/g, "&quot;")}"${sel}>${c.name}${mem}</option>`;
    }
    html += "</optgroup>";
  }
  return html;
}

export function recommendedSuites(chip, chipCount = 1) {
  if (chip?.suites?.length) return chip.suites;
  if (Number(chipCount) > 1) return ["suite_B", "suite_E"];
  if (chip?.tier === "consumer" || chip?.tier === "edge") return ["suite_F", "suite_A"];
  return ["suite_A", "suite_F"];
}
