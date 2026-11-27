/*
 * Shared by every chapter page: DOM and number helpers, color scales, the tooltip, the SVG
 * charts, tab navigation with browser history, the theme and language switches and the
 * chapter path under the top bar. Loaded after i18n.js and before the page's own scripts.
 *
 * A page defines TABS = { key: { tab, panel, hash } } in its own script; navigation reads it
 * once the page has loaded (DOMContentLoaded), so the order of the scripts does not matter.
 */

// =====================================================================
// Helpers
// =====================================================================
import { TABS, hooks } from "./page.ts";
import { wireSwitches } from "./switches.ts";
import { CHAPTERS } from "./chapters.ts";
import { versionLine } from "./version.ts";
import { I18N, L, applyStaticTranslations, t } from "./i18n.js";

export const $ = (selector, root = document) => root.querySelector(selector);
export const SVG_NS = "http://www.w3.org/2000/svg";

/** Creates an HTML element. Text always goes through textContent. */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

export function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value != null) node.setAttribute(key, value);
  return node;
}

export function svgText(x, y, content, cls, anchor = "start") {
  const node = svg("text", { x, y, class: cls, "text-anchor": anchor });
  node.textContent = content;
  return node;
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const cssList = (name) => cssVar(name).split(",").map((s) => s.trim()).filter(Boolean);

// Numbers follow the language: 0,001 and %99,29 in Turkish, 0.001 and 99.29% in English
export const locale = () => (I18N.lang === "en" ? "en-US" : "tr-TR");
export const formats = {};
export function fmtNum(value, digits) {
  const key = `${I18N.lang}:${digits}`;
  formats[key] ??= new Intl.NumberFormat(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return formats[key].format(value);
}
export const pctText = (text) => (I18N.lang === "en" ? `${text}%` : `%${text}`);
export const fmtPct = (value, digits = 2) => pctText(fmtNum(value, digits));
export const fmtInt = (value) => fmtNum(value, 0);
export const fmtTick = (value) => new Intl.NumberFormat(locale(), { maximumSignificantDigits: 2 }).format(value);
/** Probability label: a tiny value shows as "<0.1%" instead of 0.0%. */
export const fmtProb = (p) => (p < 0.001 ? `<${fmtPct(0.1, 1)}` : fmtPct(p * 100, 1));

export function fmtDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return "—";
  seconds = Math.max(0, Math.round(seconds));
  const hrs = Math.floor(seconds / 3600);
  const min = Math.floor((seconds % 3600) / 60);
  const sec = String(seconds % 60).padStart(2, "0");
  return hrs ? `${hrs}:${String(min).padStart(2, "0")}:${sec}` : `${min}:${sec}`;
}

// =====================================================================
// Color scales & pixel painting
// =====================================================================
export const LUTS = {};

export function hexToRgb(hex) {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** Interpolates a list of hex stops into a 256-entry RGB lookup table. */
export function buildLut(stops) {
  const rgb = stops.map(hexToRgb);
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1);
    const k = Math.min(Math.floor(t), rgb.length - 2);
    const f = t - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = rgb[k][c] + (rgb[k + 1][c] - rgb[k][c]) * f;
  }
  return lut;
}

export function readLuts() {
  LUTS.seq = buildLut(cssList("--seq-stops"));   // activations: 0 -> peak
  LUTS.div = buildLut(cssList("--div-stops"));   // filter weights: negative <- 0 -> positive
  LUTS.gray = buildLut(["#000000", "#ffffff"]);  // MNIST images as the model sees them
  LUTS.ramp = buildLut(cssList("--surface-ramp")); // neutral background (Theory: loss surface)

  // Dropout view (Theory): 0-253 is the sequential scale, 254-255 marks a dropped value
  const dropped = hexToRgb(cssVar("--drop-cell"));
  LUTS.drop = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const source = Math.round((i * 255) / 253) * 3;
    for (let c = 0; c < 3; c++) LUTS.drop[i * 3 + c] = i < 254 ? LUTS.seq[source + c] : dropped[c];
  }
}

export function lutColor(name, value) {
  const lut = LUTS[name];
  const j = clamp(Math.round(value), 0, 255) * 3;
  return `rgb(${lut[j]}, ${lut[j + 1]}, ${lut[j + 2]})`;
}

/** Paints a w x h grid of values (0-255, or -1..1 when signed) onto a canvas. */
export function paint(canvas, values, w, h, lutName, signed = false) {
  canvas._paint = [values, w, h, lutName, signed];
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  const lut = LUTS[lutName];
  for (let i = 0; i < w * h; i++) {
    const v = signed ? (values[i] + 1) * 127.5 : values[i];
    const j = clamp(Math.round(v), 0, 255) * 3;
    image.data[i * 4] = lut[j];
    image.data[i * 4 + 1] = lut[j + 1];
    image.data[i * 4 + 2] = lut[j + 2];
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

export function repaintAll() {
  for (const canvas of document.querySelectorAll("canvas")) {
    if (canvas._repaint) canvas._repaint();            // canvases with extra drawing on top
    else if (canvas._paint) paint(canvas, ...canvas._paint);
  }
}

// =====================================================================
// Tooltip
// =====================================================================
export const tip = $("#tooltip");
export let hoverHost = null;
/** Marks the chart under the pointer (other modules cannot assign hoverHost, an imported binding). */
export const setHoverHost = (host) => { hoverHost = host; };

/** rows: [{ value, label, color? }] — the value leads, the label follows. */
export function showTip(x, y, title, rows = []) {
  tip.replaceChildren();
  if (title) tip.append(h("div", { class: "tip-title", text: title }));
  for (const row of rows) {
    const line = h("div", { class: "tip-row" });
    if (row.color) line.append(h("span", { class: "tip-key", style: `background:${row.color}` }));
    line.append(h("strong", { text: row.value }), h("span", { text: row.label }));
    tip.append(line);
  }
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  let left = x + 14;
  let top = y + 14;
  if (left + box.width > innerWidth - 8) left = x - 14 - box.width;
  if (top + box.height > innerHeight - 8) top = y - 14 - box.height;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

export function hideTip() {
  tip.hidden = true;
}

// Charts re-render while training; drop a stale hover once the pointer leaves the chart
document.addEventListener("pointermove", (event) => {
  if (hoverHost && !hoverHost.contains(event.target)) {
    hoverHost._hover = null;
    hoverHost = null;
    hideTip();
  }
}, { passive: true });

export function attachGridTooltip(container, describe) {
  container.addEventListener("pointermove", (event) => {
    const target = event.target.closest("canvas, [data-i]");
    const info = target && container.contains(target) ? describe(target, event) : null;
    if (info) showTip(event.clientX, event.clientY, info.title, info.rows);
    else hideTip();
  });
  container.addEventListener("pointerleave", hideTip);
}

// =====================================================================
// Charts (hand-made SVG)
// =====================================================================
export function niceStep(range, count) {
  const raw = range / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * power;
}

export function linearTicks(min, max, step, format) {
  const ticks = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-6; v += step) {
    ticks.push({ v: +v.toFixed(10), label: format(+v.toFixed(10)) });
  }
  return ticks;
}

export const logFloor = (v) => { const b = 10 ** Math.floor(Math.log10(v)); return v >= 3 * b ? 3 * b : b; };
export const logCeil = (v) => { const b = 10 ** Math.floor(Math.log10(v)); return v <= b ? b : v <= 3 * b ? 3 * b : 10 * b; };

export function logTicks(min, max) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    for (const m of [1, 3]) {
      const v = +(m * 10 ** e).toPrecision(3);
      if (v >= min * 0.999 && v <= max * 1.001) ticks.push({ v, label: fmtTick(v) });
    }
  }
  return ticks;
}

/** Debiased exponential moving average (the same smoothing TensorBoard uses). */
export function smooth(values, weight = 0.95) {
  let last = 0;
  return values.map((v, i) => {
    last = last * weight + (1 - weight) * v;
    return last / (1 - weight ** (i + 1));
  });
}

/**
 * Line chart with gridlines, a crosshair and a shared tooltip.
 * All series share point indexes; x values start at 1 and step by 1.
 */
export function lineChart(host, o) {
  host._render = () => lineChart(host, o);
  if (!host.clientWidth) { host._w = 0; return; }   // hidden: the resize observer redraws it once shown
  host._w = host.clientWidth;
  const W = host.clientWidth;
  const H = +host.dataset.height || 220;
  const m = { l: 48, r: 16, t: 10, b: 26 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const xSpan = (o.x.max - o.x.min) || 1;
  const X = (v) => m.l + ((v - o.x.min) / xSpan) * iw;
  const f = o.y.log ? (v) => Math.log10(Math.max(v, o.y.min)) : (v) => v;
  const fMin = f(o.y.min);
  const fSpan = (f(o.y.max) - fMin) || 1;
  const Y = (v) => m.t + ih - ((clamp(f(v), fMin, fMin + fSpan) - fMin) / fSpan) * ih;

  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label });
  for (const t of o.y.ticks) {
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(t.v), y2: Y(t.v), class: "gridline" }));
    root.append(svgText(m.l - 8, Y(t.v) + 4, t.label, "tick", "end"));
  }
  for (const v of o.x.grid || []) {
    root.append(svg("line", { x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih, class: "gridline" }));
  }
  for (const t of o.x.ticks) root.append(svgText(X(t.v), H - 6, t.label, "tick", "middle"));
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "baseline" }));

  const count = o.series[0].points.length;
  if (!count) root.append(svgText(m.l + iw / 2, m.t + ih / 2, o.empty, "empty-chart", "middle"));

  for (const s of o.series) {
    if (!s.points.length) continue;
    let d = "";
    s.points.forEach((p, i) => { d += `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`; });
    root.append(svg("path", { d, class: "series-line", style: `stroke:${s.color};stroke-width:${s.width || 2}px` }));
    if (s.markers) {
      for (const p of s.points) root.append(svg("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, class: "marker", style: `fill:${s.color}` }));
    }
  }

  host.replaceChildren(root);
  if (!count) return;

  // Hover layer: the crosshair snaps to the nearest point; one tooltip lists every series
  const cross = svg("line", { y1: m.t, y2: m.t + ih, class: "crosshair", visibility: "hidden" });
  const dots = o.series.filter((s) => s.hoverDot !== false)
    .map((s) => [s, svg("circle", { r: 4, class: "marker", style: `fill:${s.color}`, visibility: "hidden" })]);
  const hit = svg("rect", { x: m.l - 6, y: m.t, width: iw + 12, height: ih, class: "hit" });
  root.append(cross, ...dots.map((d) => d[1]), hit);

  const show = (clientX, clientY) => {
    const box = root.getBoundingClientRect();
    const xValue = o.x.min + (((clientX - box.left) * (W / box.width) - m.l) / iw) * xSpan;
    const i = clamp(Math.round(xValue) - 1, 0, count - 1);
    const xPos = X(o.series[0].points[i][0]);
    cross.setAttribute("x1", xPos);
    cross.setAttribute("x2", xPos);
    cross.setAttribute("visibility", "visible");
    for (const [s, dot] of dots) {
      const p = s.points[i];
      dot.setAttribute("cx", X(p[0]));
      dot.setAttribute("cy", Y(p[1]));
      dot.setAttribute("visibility", "visible");
    }
    const t = o.tooltip(i);
    showTip(clientX, clientY, t.title, t.rows);
    host._hover = [clientX, clientY];
    hoverHost = host;
  };
  hit.addEventListener("pointermove", (e) => show(e.clientX, e.clientY));
  hit.addEventListener("pointerleave", () => {
    host._hover = null;
    cross.setAttribute("visibility", "hidden");
    for (const [, dot] of dots) dot.setAttribute("visibility", "hidden");
    hideTip();
  });
  if (host._hover) show(...host._hover);
}

/** Path for a horizontal bar: square at the baseline, 4px rounded data-end. */
export function barPath(x, y, w, hgt, r = 4) {
  r = Math.min(r, w, hgt / 2);
  return `M${x},${y}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${y + hgt - r}A${r},${r} 0 0 1 ${x + w - r},${y + hgt}H${x}Z`;
}

export function probabilityChart(host, probabilities, prediction) {
  host._render = () => probabilityChart(host, probabilities, prediction);
  if (!host.clientWidth) { host._w = 0; return; }   // hidden: the resize observer redraws it once shown
  host._w = host.clientWidth;
  const W = host.clientWidth;
  const rowH = 24;
  const thick = 14;
  const m = { l: 24, r: 64, t: 2, b: 2 };
  const H = m.t + m.b + rowH * 10;
  const iw = W - m.l - m.r;
  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "list", "aria-label": t("a.probs") });

  probabilities.forEach((p, digit) => {
    const y = m.t + digit * rowH;
    const top = digit === prediction;
    const row = svg("g", { class: "bar-row", role: "listitem", tabindex: 0, "aria-label": `${t("digit")} ${digit}: ${fmtPct(p * 100, 1)}` });
    row.append(svgText(m.l - 12, y + rowH / 2 + 4, String(digit), "bar-label", "middle"));
    const w = p * iw;
    if (w >= 0.5) {
      row.append(svg("path", {
        d: barPath(m.l, y + (rowH - thick) / 2, w, thick),
        class: "bar",
        style: `fill:${top ? "var(--series-1)" : "var(--bar-muted)"}`,
      }));
    }
    const label = fmtProb(p);
    const value = svgText(m.l + w + 6, y + rowH / 2 + 4, label, "bar-value");
    if (top) value.style.fill = "var(--text)";
    row.append(value, svg("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }));
    const tipFor = (e) => showTip(e.clientX, e.clientY, `${t("digit")} ${digit}`, [
      { value: fmtPct(p * 100, 2), label: t("probability"), color: top ? "var(--series-1)" : "var(--bar-muted)" },
    ]);
    row.addEventListener("pointermove", tipFor);
    row.addEventListener("pointerleave", hideTip);
    root.append(row);
  });
  host.replaceChildren(root);
}

/**
 * Calls back in the next frame after `target` changes size. A redraw inside a ResizeObserver
 * callback can change the layout again, which Safari reports as a page error.
 */
export function watchSize(target, callback) {
  let frame = 0;
  const observer = new ResizeObserver(() => {
    frame ||= requestAnimationFrame(() => {
      frame = 0;
      callback();
    });
  });
  observer.observe(target);
  return {
    disconnect() {
      cancelAnimationFrame(frame);
      frame = 0;
      observer.disconnect();
    },
  };
}

const resizedHosts = new Set();
let resizeFrame = 0;
export const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) resizedHosts.add(entry.target);
  resizeFrame ||= requestAnimationFrame(() => {
    resizeFrame = 0;
    const hosts = [...resizedHosts];
    resizedHosts.clear();
    for (const host of hosts) {
      if (!host.isConnected) {   // a demo rebuilt after a language change: let the old chart go
        resizeObserver.unobserve(host);
        continue;
      }
      if (host._render && host.clientWidth && host.clientWidth !== host._w) host._render();
    }
  });
});
document.querySelectorAll(".chart").forEach((chart) => resizeObserver.observe(chart));

// =====================================================================
// Tabs and navigation
// ---------------------------------------------------------------------
// Switching tabs and jumping to a section are real browser-history steps, so Back / Forward
// return to the previous place at the same scroll position. The address shows where you are
// (#teori/t-bn), so a reload or a shared link opens the same spot.
// =====================================================================
history.scrollRestoration = "manual";   // restored here, once the tab's content is visible
export const nav = { tab: null, section: null, scroll: {} };  // current tab and section, last scroll position in each tab
export const firstTab = () => Object.keys(TABS)[0];
export const glossaryOpenNow = () => Boolean(hooks.glossaryIsOpen?.());

export function parseHash(hash) {
  let text = hash || "";
  try {
    text = decodeURIComponent(text);
  } catch {
    text = "";   // a malformed address opens the first tab
  }
  const [base, section] = text.split("/");
  const tab = Object.keys(TABS).find((key) => TABS[key].hash === base) || firstTab();
  return { tab, section: section || null };
}

export const hashFor = (tab, section) => TABS[tab].hash + (section ? `/${section}` : "");

/** Shows a tab without touching the history. */
export function showTab(name) {
  if (!TABS[name]) name = firstTab();
  if (nav.tab && nav.tab !== name) nav.scroll[nav.tab] = window.scrollY;
  nav.tab = name;
  document.body.dataset.tab = name;   // CSS can show things only on some tabs
  for (const [key, tab] of Object.entries(TABS)) {
    const active = key === name;
    $(tab.tab).setAttribute("aria-selected", String(active));
    $(tab.tab).tabIndex = active ? 0 : -1;   // arrow keys move between tabs (see below)
    $(tab.panel).hidden = !active;
  }
  hideTip();
  document.dispatchEvent(new CustomEvent("tab-shown", { detail: name }));
}

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
/** The scroll behavior to use: smooth only if the reader has not asked for less motion. */
export const scrollMode = (smooth) => (smooth && !reducedMotion.matches ? "smooth" : "instant");

/** Scrolls to a position or a section as soon as it exists (some tabs render lazily). */
export function scrollWhenReady({ y = null, section = null, smooth = false, flash = false, focus = false }) {
  const started = performance.now();
  const attempt = () => {
    const waited = performance.now() - started;
    if (section) {
      const target = document.getElementById(section);
      if (target && target.getClientRects().length) {
        target.scrollIntoView({ behavior: scrollMode(smooth), block: "start" });
        if (focus) {   // keyboard users continue from the place they jumped to
          if (!target.hasAttribute("tabindex")) target.tabIndex = -1;
          target.classList.add("nav-target");
          target.focus({ preventScroll: true });
        }
        if (flash) {   // show where the jump landed
          target.classList.remove("nav-flash");
          void target.getBoundingClientRect();
          target.classList.add("nav-flash");
          setTimeout(() => target.classList.remove("nav-flash"), 1800);
        }
        return;
      }
    } else {
      const room = document.documentElement.scrollHeight - window.innerHeight;
      if (room >= y - 2 || waited > 2500) {
        window.scrollTo({ top: y, behavior: "instant" });
        return;
      }
    }
    if (waited < 4000) requestAnimationFrame(attempt);
  };
  attempt();
}

/** Goes to a tab and, optionally, a section in it, as a new history step. */
export function navigate(tab, section = null, { replace = false } = {}) {
  if (!TABS[tab] || (tab === nav.tab && !section)) return;
  if (tab === nav.tab && section === nav.section) {   // already here: back to its start, no new history step
    scrollWhenReady({ section, smooth: true });
    return;
  }
  history.replaceState({ ...history.state, tab: nav.tab, scroll: window.scrollY }, "");   // where Back returns
  const entry = { tab, section, scroll: null, glossary: glossaryOpenNow() };   // an open glossary panel stays open
  if (replace) history.replaceState(entry, "", hashFor(tab, section));
  else history.pushState(entry, "", hashFor(tab, section));
  const sameTab = tab === nav.tab;
  nav.section = section;
  showTab(tab);
  if (section) scrollWhenReady({ section, smooth: sameTab, flash: !sameTab, focus: !glossaryOpenNow() });
  else scrollWhenReady({ y: nav.scroll[tab] ?? 0 });   // back where you were reading in that tab
}

/** The name the other scripts use to switch tabs. */
export const selectTab = (name) => navigate(name);

window.addEventListener("popstate", (event) => {
  const state = event.state || { ...parseHash(location.hash), scroll: null };
  const section = state.section ?? null;
  const panelOnly = state.tab === nav.tab && section === nav.section && Boolean(state.glossary) !== glossaryOpenNow();
  nav.section = section;
  hooks.syncGlossary?.(state);   // the glossary panel opens or closes with the history entry
  if (panelOnly) return;   // only the panel changed: the page stays where it is
  showTab(state.tab);
  if (state.scroll != null) scrollWhenReady({ y: state.scroll });
  else if (state.section) scrollWhenReady({ section: state.section });
  else scrollWhenReady({ y: 0 });
});

// Phones: the top bar slides away while scrolling down (reading) and comes back on the way up
export const phoneWidth = matchMedia("(max-width: 720px)");
export let lastScrollY = window.scrollY;
window.addEventListener("scroll", () => {
  const y = window.scrollY;
  const bar = $(".topbar").offsetHeight;
  if (!phoneWidth.matches || y <= bar) document.body.classList.remove("topbar-away");
  else if (y > lastScrollY + 6) document.body.classList.add("topbar-away");
  else if (y < lastScrollY - 6) document.body.classList.remove("topbar-away");
  lastScrollY = y;
}, { passive: true });

// Keep the current history entry's scroll position up to date, so Forward also lands right
export let scrollSaveTimer = 0;
window.addEventListener("scroll", () => {
  clearTimeout(scrollSaveTimer);
  scrollSaveTimer = setTimeout(() => {
    if (nav.tab) history.replaceState({ ...history.state, tab: nav.tab, scroll: window.scrollY }, "");
  }, 250);
}, { passive: true });

// Sticky bars sit right under the top bar, whose height changes with the window width
new ResizeObserver(() => document.documentElement.style.setProperty("--topbar-h", `${$(".topbar").offsetHeight}px`)).observe($(".topbar"));

// Any element with data-go="tab:section" is a link to that place (data-chapter: on another page)
document.addEventListener("click", (event) => {
  const link = event.target.closest?.("[data-go]");
  if (!link) return;
  event.preventDefault();
  const [tab, section] = link.dataset.go.split(":");
  goToChapterPlace({ chapter: link.dataset.chapter, tab, section: section || null });
  // The link was in the tab just left: keep the keyboard on the page
  if (!section && TABS[tab] && !link.getClientRects().length) $(TABS[tab].tab).focus();
});

export function wireTabs() {
  for (const [key, tab] of Object.entries(TABS)) $(tab.tab).addEventListener("click", () => navigate(key));
  // Tab list keyboard support: ← → Home End move between tabs
  $(".tabs").addEventListener("keydown", (event) => {
    const keys = Object.keys(TABS);
    const i = keys.indexOf(nav.tab);
    const next = { ArrowRight: keys[(i + 1) % keys.length], ArrowLeft: keys[(i - 1 + keys.length) % keys.length],
      Home: keys[0], End: keys[keys.length - 1] }[event.key];
    if (!next) return;
    event.preventDefault();
    navigate(next, null, { replace: true });   // moving along the tabs is not a history step each
    $(TABS[next].tab).focus();
  });
}

// Open the tab (and section) in the address. Waits for the other scripts, so they hear it.
document.addEventListener("DOMContentLoaded", () => {
  wireTabs();
  const { tab, section } = parseHash(location.hash);
  // A reload returns to where the reader was; a link or a first visit opens its section
  const saved = history.state?.tab === tab ? history.state.scroll : null;
  history.replaceState({ tab, section, scroll: saved ?? null }, "", hashFor(tab, section));
  nav.section = section;
  showTab(tab);
  if (saved != null) scrollWhenReady({ y: saved });
  else if (section) scrollWhenReady({ section });
});


// =====================================================================
// Chapters: the learning path from the perceptron to today's networks
// =====================================================================
/** Every chapter page has <body data-chapter="id">. A chapter without a url is still to come (url: the chapter's page). */
export { CHAPTERS };   // chapters.ts: generated by scripts/build.mjs from src/chapters/*/chapter.json
export const CHAPTER = document.body.dataset.chapter || "cnn";

/** Tab addresses are the same on every chapter page. */
export const TAB_HASHES = { train: "#egitim", infer: "#cikarim", detect: "#tespit", play: "#deneme", theory: "#teori", math: "#matematik", code: "#kodlama", practice: "#alistirma" };
export const TAB_NAMES = {
  train: { tr: "Eğitim", en: "Training" }, infer: { tr: "Çıkarım", en: "Inference" }, detect: { tr: "Tespit", en: "Detection" }, play: { tr: "Deneme alanı", en: "Playground" },
  theory: { tr: "Teori", en: "Theory" }, math: { tr: "Matematik", en: "Math" }, code: { tr: "Kodlama", en: "Coding" },
  practice: { tr: "Alıştırma", en: "Practice" },
};
export const chapterOf = (id) => CHAPTERS.find((c) => c.id === id);
/**
 * Section titles ({ id: { tr, en } }) that links can show before, or without, rendering the section.
 * The perceptron chapter's titles live here (its pages read them from here), so the CNN page can
 * name them too; lesson tabs add their own sections when they start.
 */
export const PLACE_TITLES = {
  // Chapter 1: Theory, Math and Coding sections of perceptron.html
  "p-intro": { tr: "Perceptron: öğrenen ilk yapay nöron", en: "The perceptron: the first artificial neuron that learns" },
  "p-neuron": { tr: "Bir perceptron nasıl karar verir?", en: "How does a perceptron decide?" },
  "p-boundary": { tr: "Geometri: karar sınırı", en: "Geometry: the decision boundary" },
  "p-rule": { tr: "Öğrenme kuralı", en: "The learning rule" },
  "p-converge": { tr: "Ne zaman öğrenebilir? Doğrusal ayrılabilirlik", en: "When can it learn? Linear separability" },
  "p-xor": { tr: "Sınır: XOR problemi", en: "The limit: the XOR problem" },
  "pm-notation": { tr: "Gösterim ve ileri hesap", en: "Notation and the forward computation" },
  "pm-geometry": { tr: "Karar sınırının geometrisi", en: "The geometry of the decision boundary" },
  "pm-rule": { tr: "Öğrenme kuralı ve neden işe yaradığı", en: "The learning rule and why it works" },
  "pm-gd": { tr: "Gradyan inişi olarak perceptron", en: "The perceptron as gradient descent" },
  "pm-converge": { tr: "Yakınsama teoremi", en: "The convergence theorem" },
  "pm-xor": { tr: "XOR neden imkânsız?", en: "Why XOR is impossible" },
  "pc-plan": { tr: "Ne yazacağız?", en: "What will we write?" },
  "pc-data": { tr: "Ayarlar ve veri", en: "Settings and data" },
  "pc-step": { tr: "Basamak fonksiyonu", en: "The step function" },
  "pc-class": { tr: "Perceptron sınıfı: başlatma ve tahmin", en: "The Perceptron class: setup and prediction" },
  "pc-fit": { tr: "Öğrenme kuralı: fit", en: "The learning rule: fit" },
  "pc-run": { tr: "Çalıştırmak ve sonucu okumak", en: "Running it and reading the result" },
  "pc-torch": { tr: "Aynısı PyTorch ile", en: "The same in PyTorch" },
  "pc-exercises": { tr: "Alıştırmalar", en: "Exercises" },
  // Chapter 4: Theory and Math sections of yolo.html (the glossary links to them from every page)
  "y-intro": { tr: "Nesne tespiti ve segmentasyon: YOLO nasıl çalışır?", en: "Object detection and segmentation: how does YOLO work?" },
  "y-tasks": { tr: "Ne ve nerede? Dört görev", en: "What and where? Four tasks" },
  "y-iou": { tr: "Kutu ve IoU", en: "Boxes and IoU" },
  "y-grid": { tr: "Tek bakışta: üç ölçekte ızgara", en: "One look: a grid at three scales" },
  "y-nms": { tr: "Çok fazla aday: güven eşiği ve NMS", en: "Too many candidates: the confidence threshold and NMS" },
  "y-masks": { tr: "Maskeler: prototipler ve katsayılar", en: "Masks: prototypes and coefficients" },
  "y-train": { tr: "Nasıl eğitilir?", en: "How is it trained?" },
  "y-metrics": { tr: "Ne kadar iyi? Kesinlik, duyarlılık, mAP", en: "How good? Precision, recall, mAP" },
  "y-read": { tr: "Mühendis sonucu nasıl okur?", en: "How does an engineer read the result?" },
  "ym-notation": { tr: "Gösterim ve boyutlar", en: "Notation and sizes" },
  "ym-boxes": { tr: "Kutunun kodlanması", en: "How a box is encoded" },
  "ym-iou": { tr: "IoU ve IoU kayıpları", en: "IoU and the IoU losses" },
  "ym-cls": { tr: "Sınıf skoru: sigmoid ve ikili çapraz entropi", en: "The class score: sigmoid and binary cross-entropy" },
  "ym-mask": { tr: "Maske: prototiplerin doğrusal birleşimi", en: "The mask: a linear combination of prototypes" },
  "ym-nms": { tr: "NMS: algoritma ve maliyeti", en: "NMS: the algorithm and its cost" },
  "ym-ap": { tr: "Kesinlik, duyarlılık ve AP", en: "Precision, recall and AP" },
  "ym-area": { tr: "Alan ve hatası", en: "Area and its error" },
  // Chapter 2: Theory, Math and Coding sections of mlp.html
  "ml-intro": { tr: "Çok katmanlı ağ: gizli katmanlar ve geri yayılım", en: "The multilayer network: hidden layers and backpropagation" },
  "ml-act": { tr: "Basamaktan yumuşak eşiğe: aktivasyonlar", en: "From the step to a soft threshold: activations" },
  "ml-hidden": { tr: "Gizli katman: veriyi yeniden çizmek", en: "The hidden layer: redrawing the data" },
  "ml-loss": { tr: "Hatayı ölçmek: kayıp ve gradyan inişi", en: "Measuring the error: the loss and gradient descent" },
  "ml-backprop": { tr: "Geri yayılım: hatayı geriye dağıtmak", en: "Backpropagation: sending the error back" },
  "ml-capacity": { tr: "Kaç nöron yeter? Evrensel yaklaşım", en: "How many neurons? Universal approximation" },
  "ml-general": { tr: "Ezberlemek mi, genellemek mi? Eğitim, doğrulama, test", en: "Memorizing or generalizing? Training, validation, test" },
  "ml-images": { tr: "Görüntülerde MLP ve sınırları", en: "The MLP on images, and its limits" },
  "mm-forward": { tr: "Katmanlar ve ileri hesap", en: "Layers and the forward pass" },
  "mm-act": { tr: "Aktivasyonlar ve türevleri", en: "Activations and their derivatives" },
  "mm-loss": { tr: "Çapraz entropi ve çıktıdaki gradyan", en: "Cross-entropy and the gradient at the output" },
  "mm-backprop": { tr: "Geri yayılımın dört denklemi", en: "The four equations of backpropagation" },
  "mm-train": { tr: "Eğitim: mini-batch ve başlangıç ağırlıkları", en: "Training: mini-batches and initial weights" },
  "mm-universal": { tr: "Evrensel yaklaşım teoremi", en: "The universal approximation theorem" },
  "mc-plan": { tr: "Ne yazacağız?", en: "What will we write?" },
  "mc-data": { tr: "Ayarlar ve veri", en: "Settings and data" },
  "mc-act": { tr: "Aktivasyon ve türevi", en: "The activation and its derivative" },
  "mc-class": { tr: "MLP sınıfı: ağırlıklar ve ileri hesap", en: "The MLP class: weights and the forward pass" },
  "mc-backward": { tr: "Geri yayılım: backward", en: "Backpropagation: backward" },
  "mc-fit": { tr: "Eğitim döngüsü: fit", en: "The training loop: fit" },
  "mc-check": { tr: "Gradyanı kontrol etmek", en: "Checking the gradient" },
  "mc-run": { tr: "Çalıştırmak ve sonucu okumak", en: "Running it and reading the result" },
  "mc-torch": { tr: "Aynısı PyTorch ile", en: "The same in PyTorch" },
  "mc-exercises": { tr: "Alıştırmalar", en: "Exercises" },
};

/** Goes to a place on this page, or opens the other chapter's page at that place. */
export function goToChapterPlace({ chapter, tab, section = null }) {
  if (!chapter || chapter === CHAPTER) {
    navigate(tab, section);
    return;
  }
  const target = chapterOf(chapter);
  if (target?.url) location.href = target.url + (TAB_HASHES[tab] || "") + (section ? `/${section}` : "");
}

/**
 * The learning path lives behind the page title (top left): the title is a button that opens a
 * list of the chapters. It is a plain disclosure with links (not an ARIA menu): Tab moves through
 * the chapters, Esc or a click elsewhere closes it.
 */
export function renderChapters() {
  const button = $("#chapter-btn");
  const menu = $("#chapter-menu");
  if (!button || !menu) return;
  const pathName = L({ tr: "Öğrenme yolu", en: "Learning path" });
  const list = h("ol", { class: "chm-list" });
  CHAPTERS.forEach((c, k) => {
    const current = c.id === CHAPTER;
    const meta = current ? `${c.year} · ${L({ tr: "şu an buradasınız", en: "you are here" })}`
      : c.url ? c.year : `${c.year} · ${L({ tr: "yakında", en: "coming soon" })}`;
    const inner = [
      h("span", { class: "chm-num", text: String(k + 1) }),
      h("span", { class: "chm-text" },
        h("span", { class: "chm-title", text: L(c.title) }),
        h("span", { class: "chm-sub", text: L(c.sub) }),
        h("span", { class: "chm-meta", text: meta })),
    ];
    const item = c.url && !current
      ? h("a", { class: "chm-item", href: c.url }, ...inner)
      : h("span", { class: `chm-item${current ? " current" : " soon"}`, "aria-current": current ? "page" : null }, ...inner);
    list.append(h("li", {}, item));
  });
  menu.setAttribute("aria-label", pathName);
  const home = h("a", { class: "chm-home", href: "index.html" }, L({ tr: "← Ana sayfa", en: "← Home page" }));
  menu.replaceChildren(h("div", { class: "chm-head" }, h("span", { text: pathName }), home), list, versionLine("chm-version"));
  const here = CHAPTERS.findIndex((c) => c.id === CHAPTER) + 1;
  button.title = `${pathName} · ${L({ tr: "Bölüm", en: "Chapter" })} ${here}/${CHAPTERS.length}`;
}

export function setChapterMenu(open, { focusButton = false } = {}) {
  const button = $("#chapter-btn");
  const menu = $("#chapter-menu");
  if (!button || !menu || open === !menu.hidden) return;
  if (open) {   // just under the top bar, whatever its height at this width
    const bar = $(".topbar").getBoundingClientRect();
    menu.style.top = `${Math.round(bar.bottom - menu.parentElement.getBoundingClientRect().top + 6)}px`;
    menu.style.maxHeight = `${Math.max(140, Math.round(window.innerHeight - bar.bottom - 12))}px`;
  }
  menu.hidden = !open;
  button.setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("chapter-menu-open", open);
  if (focusButton) button.focus();
}

document.addEventListener("click", (event) => {
  const menu = $("#chapter-menu");
  if (!menu) return;
  if (event.target.closest("#chapter-btn")) setChapterMenu(menu.hidden);
  else if (!event.target.closest("#chapter-menu")) setChapterMenu(false);
});
// Capture phase, so Esc closes only the menu and not the glossary behind it
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || $("#chapter-menu")?.hidden !== false) return;
  event.stopImmediatePropagation();
  setChapterMenu(false, { focusButton: true });
}, true);
// Tab past the last chapter closes it (a click elsewhere is handled above; the window getting
// focus back is not a reason to close it)
$(".brand")?.addEventListener("focusout", (event) => {
  if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget)) setChapterMenu(false);
});
document.addEventListener("lang-change", renderChapters);
renderChapters();

/**
 * The top bar takes one row on a wide screen. The widths in style.css at which it drops the
 * subtitle, then the model badge, then puts the tabs on a row of their own suit the usual fonts;
 * with a wider font, longer badge text or the other language it would wrap earlier and leave the
 * badges alone on a second row. So above 1040px the same steps are also taken here (classes
 * fit-1 … fit-3), one at a time, until the bar fits.
 */
const FIT_STEPS = ["fit-1", "fit-2", "fit-3"];
const stacked = window.matchMedia("(max-width: 1040px)");
function fitTopbar() {
  const bar = $(".topbar");
  if (!bar) return;
  bar.classList.remove(...FIT_STEPS);
  if (stacked.matches) return;   // the stylesheet's own two-row layout
  const wraps = () => {
    const rows = [...bar.children].filter((el) => el.offsetParent !== null).map((el) => el.getBoundingClientRect().top);
    return Math.max(...rows) - Math.min(...rows) > 20;
  };
  for (const step of FIT_STEPS) {
    if (!wraps()) break;
    bar.classList.add(step);
  }
}
if ($(".topbar")) {
  watchSize($(".topbar"), fitTopbar);
  if ($(".badges")) watchSize($(".badges"), fitTopbar);   // badge text that changes (the CNN page's device)
  stacked.addEventListener("change", fitTopbar);
  document.addEventListener("lang-change", fitTopbar);
  document.fonts?.ready.then(fitTopbar);
  fitTopbar();
}


// Theme and language switches (switches.ts). Canvases keep their pixels, so repaint them with
// the new theme's color scales.
wireSwitches();
document.addEventListener("theme-change", () => {
  readLuts();
  repaintAll();
});

applyStaticTranslations();
readLuts();
