// dom_stub.mjs — Tiny DOM + Chart.js stand-in for headless node tests.
//
// modal.js's Visualize tab renderer touches a small DOM surface
// (createElement / appendChild / classList / innerHTML / textContent /
// dataset / style + getComputedStyle on the documentElement) plus the
// global Chart constructor.  Spinning up jsdom or happy-dom for that is
// overkill; this file ships the minimum implementation required to make
// `_test.renderViz(panel, row)` reach every per-suite branch without
// throwing.  When the real DOM grows another usage in modal.js, add it
// here — the failing test will tell you exactly which method.

class FakeEl {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this._attrs = {};
    this._cls = new Set();
    this._innerHTML = "";
    this._textContent = "";
    this.style = {};
    this.dataset = {};
    this.classList = {
      add:      (c) => this._cls.add(c),
      remove:   (c) => this._cls.delete(c),
      toggle:   (c, v) => (v ? this._cls.add(c) : this._cls.delete(c)),
      contains: (c) => this._cls.has(c),
    };
  }
  set className(v) {
    this._cls = new Set(String(v).split(/\s+/).filter(Boolean));
  }
  get className() { return [...this._cls].join(" "); }
  // Mirror DOM `el.id` ↔ `el.getAttribute("id")` so callers that use
  // either form (data.js's injectVendorStyles uses the property form)
  // both update the same backing slot the head's appendChild looks at.
  set id(v) { this._attrs.id = String(v); }
  get id() { return this._attrs.id || ""; }
  set innerHTML(v) {
    this._innerHTML = String(v == null ? "" : v);
    if (this._innerHTML === "") this.children = [];
  }
  get innerHTML() { return this._innerHTML; }
  set textContent(v) { this._textContent = String(v); }
  get textContent() { return this._textContent; }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this._attrs[k] = v; }
  removeAttribute(k) { delete this._attrs[k]; }
  getAttribute(k)  { return this._attrs[k] ?? null; }

  // Recursively flatten children — handy for assertions that care
  // about how many cards / canvases the renderer produced regardless
  // of nesting depth.
  descendants() {
    const out = [];
    const walk = (el) => {
      for (const c of el.children) {
        out.push(c);
        if (c instanceof FakeEl) walk(c);
      }
    };
    walk(this);
    return out;
  }
}

// installDom() wires fake document + window onto globalThis and returns
// helpers a test uses to inspect chart creation.  Each call resets
// counters so test cases stay independent.
export function installDom() {
  const created = [];

  function FakeChart(canvas, config) {
    this.canvas = canvas;
    this.config = config;
    this.destroyed = false;
    this.destroy = () => { this.destroyed = true; };
    created.push(this);
  }

  // `_byId` lets injectVendorStyles in data.js mount its singleton
  // <style> tag and re-find it on subsequent re-runs.  Tests that
  // don't care about styles can ignore it.
  const _byId = new Map();
  const head = new FakeEl("head");
  const document = {
    createElement: (tag) => new FakeEl(tag),
    documentElement: new FakeEl("html"),
    head,
    body: new FakeEl("body"),
    addEventListener: () => {},
    querySelectorAll: () => [],
    activeElement: null,
    getElementById: (id) => _byId.get(id) || null,
  };
  // Patch head.appendChild to keep an id index so getElementById works.
  const _origHeadAppend = head.appendChild.bind(head);
  head.appendChild = (el) => {
    if (el && el._attrs && el._attrs.id) _byId.set(el._attrs.id, el);
    if (el && el.tagName === "STYLE" && el._attrs?.id) _byId.set(el._attrs.id, el);
    return _origHeadAppend(el);
  };

  const window = {
    Chart: FakeChart,
    addEventListener: () => {},
  };

  globalThis.document = document;
  globalThis.window = window;
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
  globalThis.history = { replaceState: () => {} };
  globalThis.location = { hash: "", pathname: "/", search: "" };
  globalThis.setTimeout = globalThis.setTimeout || ((fn) => fn());

  return {
    FakeEl,
    document,
    window,
    chartsCreated: () => created.length,
    charts: () => created.slice(),
    reset: () => { created.length = 0; },
    makePanel: () => new FakeEl("section"),
  };
}
