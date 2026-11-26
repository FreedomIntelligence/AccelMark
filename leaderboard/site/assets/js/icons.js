// icons.js — Ultra-clean 2–4 stroke monochrome Lucide-style inline SVGs.
// Zero external CDN calls; razor-sharp at 14–20px in both light and dark modes.

const SVG_OPEN = (size, cls = "") =>
  `<svg class="ic${cls ? " " + cls : ""}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">`;

export const SUITE_ICONS = {
  suite_A: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', // zap
  suite_B: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>', // layers
  suite_C: '<path d="M3 17h4v-4h4V9h4V5h6"/>', // crisp 3-step quantization stair
  suite_D: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/>', // file-text
  suite_E: '<rect x="3" y="3" width="7" height="7" rx="1.2"/><rect x="14" y="3" width="7" height="7" rx="1.2"/><rect x="3" y="14" width="7" height="7" rx="1.2"/><rect x="14" y="14" width="7" height="7" rx="1.2"/>', // layout-grid
  suite_F: '<rect x="3" y="4" width="18" height="12" rx="2"/><line x1="2" y1="20" x2="22" y2="20"/>', // laptop
  suite_G: '<circle cx="6" cy="12" r="2.2"/><circle cx="18" cy="6" r="2.2"/><circle cx="18" cy="18" r="2.2"/><path d="M8.2 11 15.8 7"/><path d="M8.2 13 15.8 17" opacity="0.45"/>', // sparse branch
  suite_H: '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="8" y1="12" x2="16" y2="12"/>', // scan window
};

export const UI_ICONS = {
  search: '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
  moon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
  shieldCheck: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>',
  lock: '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
  flask: '<path d="M9 3h6"/><path d="M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3"/>',
  hash: '<line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/>',
  checkCircle: '<circle cx="12" cy="12" r="9"/><polyline points="9 12 11 14 15 10"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  arrowRight: '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
  arrowUpRight: '<line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/>',
  table: '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="9" x2="9" y2="21"/>',
  scatter: '<line x1="3" y1="3" x2="3" y2="21"/><line x1="3" y1="21" x2="21" y2="21"/><circle cx="9" cy="14" r="1.5" fill="currentColor"/><circle cx="13" cy="9" r="1.5" fill="currentColor"/><circle cx="18" cy="12" r="1.5" fill="currentColor"/>',
  info: '<circle cx="12" cy="12" r="9"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
  x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
  gitPullRequest: '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><line x1="6" y1="9" x2="6" y2="21"/>',
  terminal: '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
};

export function suiteIcon(suiteId, size = 16, cls = "") {
  const inner = SUITE_ICONS[suiteId] || SUITE_ICONS.suite_A;
  return `${SVG_OPEN(size, cls)}${inner}</svg>`;
}

export function icon(name, size = 16, cls = "") {
  const inner = UI_ICONS[name] || "";
  return `${SVG_OPEN(size, cls)}${inner}</svg>`;
}

/** Gen-1 brand mark SVG using currentColor + Cobalt Blue droplet. */
export function brandLogoSvg(size = 24) {
  return `<svg class="brand-mark" width="${size}" height="${size}" viewBox="0 0 64 64" fill="none" aria-hidden="true">
    <polygon points="30.03,10.5 59.0,10.5 54.55,16.71 33.05,16.71 13.57,53.41 5.0,53.41" fill="currentColor"/>
    <polygon points="34.81,19.57 37.92,19.57 59.0,53.41 50.27,53.41 43.63,41.66 21.8,41.66 39.26,34.01 33.13,22.68" fill="currentColor"/>
    <path d="M 31.58 26.2 C 31.58 26.2 34.14 29.73 34.14 31.37 A 2.56 2.56 0 1 1 29.02 31.37 C 29.02 29.73 31.58 26.2 31.58 26.2 Z" fill="var(--accent)"/>
  </svg>`;
}
