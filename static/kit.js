"use strict";
/*
 * Building blocks shared by the chapter pages (perceptron.js, mlp.js): number formatting for
 * TeX, seeded random numbers, the input plane, a function graph, sliders, segmented buttons
 * and the Theory section card. Everything is on one object, KIT, so the page scripts can
 * take what they need inside their own scope:
 *
 *   const { createPlane, slider, nf } = KIT;
 *
 * Canvas views register a redraw in KIT.redrawers (called on theme change) and their resize
 * observers in KIT.observers (dropped when a page rebuilds its panels).
 * Uses core.js (h, $, cssVar, hexToRgb, fmtNum, resizeObserver), i18n.js (L), glossary.js
 * (termify) and math.js (texNum), all at run time.
 */
const KIT = (() => {
const R = String.raw;

/** Mulberry32: a small seeded random generator, so "new data" can be repeated. */
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (rand) => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

// Numbers
const nf = (v, d = 2) => fmtNum(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d);   // never "-0,00"
const tx = (v, d = 2) => texNum(nf(v, d));                                  // a number inside TeX
const txs = (v, d = 2) => (v < 0 ? `-\\,${tx(-v, d)}` : `+\\,${tx(v, d)}`);  // "+ 0,5" / "− 0,5"
const txp = (v, d = 2) => (v < 0 ? `(${tx(v, d)})` : tx(v, d));             // negative in parentheses
const T = (value) => termify(L(value));

/** Static HTML with glossary links. */
function html(tag, attrs, content) {
  const node = h(tag, attrs);
  node.innerHTML = T(content);
  return node;
}
const formula = (tex) => h("div", { class: "formula", text: `\\(\\displaystyle ${tex}\\)` });

// =====================================================================
// Colors and redrawing
// =====================================================================
const COLORS = {};
function readColors() {
  Object.assign(COLORS, {
    one: cssVar("--series-3"), zero: cssVar("--critical"), line: cssVar("--series-2"), lineB: cssVar("--series-1"),
    text: cssVar("--text"), text2: cssVar("--text-2"), muted: cssVar("--muted"), grid: cssVar("--grid"),
    axis: cssVar("--axis"), surface: cssVar("--surface"), accent: cssVar("--accent"), page: cssVar("--page"),
  });
}
readColors();

const redrawers = new Set();      // views that draw on canvases: redrawn when the theme changes
const observers = [];            // resize observers of the current panels (dropped on rebuild)

/** Keeps a canvas's pixel size in step with its CSS size; calls render(size) when it changes. */
function fitCanvas(canvas, render) {
  const view = { w: 0, h: 0 };
  const resize = () => {
    const w = canvas.clientWidth;
    const hgt = canvas.clientHeight;
    if (!w || !hgt) return;
    view.w = w;
    view.h = hgt;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(hgt * dpr);
    render();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  observers.push(observer);
  redrawers.add(() => render());
  return view;
}

// =====================================================================
// The input plane
// =====================================================================
/**
 * A square canvas showing a 2-D plane: shaded regions, lines, the normal vector w and the data
 * points (green circles = class 1, red crosses = class 0).
 *
 * plane.draw(scene). Scene fields, all optional:
 *   classify(x, y) → 0 | 1 | null   flat shading of the two regions
 *   prob(x, y) → p in [0, 1]        shading that fades out near p = 0.5; contour: true also draws
 *                                   the line p = 0.5
 *   lines: [{ w, b, color, ghost, label }]   lines w·x + b = 0
 *   normal: { w, b, length }        the arrow w on the line
 *   points: [{ x, y, current, wrong, label }];  radius: point size in px;  coords: label the points
 */
function createPlane(host, { domain = [-0.5, 1.5, -0.5, 1.5], ticks = [0, 1], label, axes = ["x₁", "x₂"] } = {}) {
  const canvas = h("canvas", { role: "img", "aria-label": label || "" });
  const wrap = h("div", { class: "pc-plane" }, canvas);
  host.append(wrap);
  const ctx = canvas.getContext("2d");
  const shade = document.createElement("canvas");
  const plane = { wrap, canvas, domain, scene: {}, size: 0 };

  const [x0, x1, y0, y1] = domain;
  plane.toPx = (x, y) => [((x - x0) / (x1 - x0)) * plane.size, plane.size - ((y - y0) / (y1 - y0)) * plane.size];
  plane.toData = (px, py) => [x0 + (px / plane.size) * (x1 - x0), y0 + ((plane.size - py) / plane.size) * (y1 - y0)];
  plane.unit = () => plane.size / (x1 - x0);   // pixels per data unit

  /** Points where the line w·x + b = 0 crosses the domain square (null if it misses it). */
  plane.clipLine = (w, b) => {
    const pts = [];
    if (Math.abs(w[1]) > 1e-12) {
      for (const x of [x0, x1]) { const y = -(w[0] * x + b) / w[1]; if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]); }
    }
    if (Math.abs(w[0]) > 1e-12) {
      for (const y of [y0, y1]) { const x = -(w[1] * y + b) / w[0]; if (x >= x0 - 1e-9 && x <= x1 + 1e-9) pts.push([x, y]); }
    }
    if (pts.length < 2) return null;
    let best = [pts[0], pts[1]];
    let far = 0;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
        if (d > far) { far = d; best = [pts[i], pts[j]]; }
      }
    }
    return far > 1e-9 ? best : null;
  };

  /** The boundary point nearest the middle of the plane, and the unit normal. */
  plane.foot = (w, b) => {
    const n2 = w[0] ** 2 + w[1] ** 2;
    if (n2 < 1e-12) return null;
    const c = [(x0 + x1) / 2, (y0 + y1) / 2];
    const k = (w[0] * c[0] + w[1] * c[1] + b) / n2;
    const n = Math.sqrt(n2);
    return { p: [c[0] - k * w[0], c[1] - k * w[1]], u: [w[0] / n, w[1] / n] };
  };

  /** Grid of samples: value(x, y) at the centres of n × n cells. */
  function sampleGrid(n, value) {
    const out = new Float64Array(n * n);
    for (let r = 0; r < n; r++) {
      const y = y1 - ((r + 0.5) / n) * (y1 - y0);
      for (let c = 0; c < n; c++) out[r * n + c] = value(x0 + ((c + 0.5) / n) * (x1 - x0), y);
    }
    return out;
  }

  function drawShade(classify, prob) {
    const n = Math.max(24, Math.round(plane.size / 3));
    shade.width = n;
    shade.height = n;
    const sctx = shade.getContext("2d");
    const img = sctx.createImageData(n, n);
    const one = hexToRgb(COLORS.one);
    const zero = hexToRgb(COLORS.zero);
    const values = sampleGrid(n, prob || classify);
    for (let k = 0; k < n * n; k++) {
      const v = values[k];
      const i = k * 4;
      if (v == null || Number.isNaN(v)) continue;
      const isOne = v > 0.5;
      const rgb = isOne ? one : zero;
      img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2];
      // flat: fixed strength; prob: stronger the surer the model is
      const strength = prob ? Math.min(1, Math.abs(v - 0.5) * 2) ** 0.7 : 1;
      img.data[i + 3] = Math.round((prob ? (isOne ? 54 : 30) : (isOne ? 40 : 18)) * strength);
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(shade, 0, 0, plane.size, plane.size);
  }

  /** The line value(x, y) = level, by marching squares on a grid of about 4 px. */
  function drawContour(value, level) {
    const n = Math.max(24, Math.round(plane.size / 4));
    const v = new Float64Array((n + 1) * (n + 1));
    for (let r = 0; r <= n; r++) {
      for (let c = 0; c <= n; c++) v[r * (n + 1) + c] = value(x0 + (c / n) * (x1 - x0), y1 - (r / n) * (y1 - y0)) - level;
    }
    const cell = plane.size / n;
    const at = (r, c) => v[r * (n + 1) + c];
    const cross = (a, b) => a / (a - b);   // where the line crosses between two corners (0..1)
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const tl = at(r, c), tr = at(r, c + 1), br = at(r + 1, c + 1), bl = at(r + 1, c);
        const pts = [];
        if ((tl > 0) !== (tr > 0)) pts.push([c + cross(tl, tr), r]);
        if ((tr > 0) !== (br > 0)) pts.push([c + 1, r + cross(tr, br)]);
        if ((bl > 0) !== (br > 0)) pts.push([c + cross(bl, br), r + 1]);
        if ((tl > 0) !== (bl > 0)) pts.push([c, r + cross(tl, bl)]);
        for (let k = 0; k + 1 < pts.length; k += 2) {
          ctx.moveTo(pts[k][0] * cell, pts[k][1] * cell);
          ctx.lineTo(pts[k + 1][0] * cell, pts[k + 1][1] * cell);
        }
      }
    }
    ctx.stroke();
  }

  function arrow(from, to, color, width) {
    const [ax, ay] = plane.toPx(...from);
    const [bx, by] = plane.toPx(...to);
    const ang = Math.atan2(by - ay, bx - ax);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx - 7 * Math.cos(ang), by - 7 * Math.sin(ang));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - 11 * Math.cos(ang - 0.42), by - 11 * Math.sin(ang - 0.42));
    ctx.lineTo(bx - 11 * Math.cos(ang + 0.42), by - 11 * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
  }

  function render() {
    if (!plane.size) return;
    const s = plane.scene;
    const dpr = canvas.width / plane.size;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, plane.size, plane.size);
    ctx.fillStyle = COLORS.surface;
    ctx.fillRect(0, 0, plane.size, plane.size);
    if (s.classify || s.prob) drawShade(s.classify, s.prob);

    // Grid every 0.5 (every 1 on wide planes), axes through 0
    const stepGrid = (x1 - x0) > 3 ? 1 : 0.5;
    ctx.lineWidth = 1;
    for (let v = Math.ceil(x0 / stepGrid) * stepGrid; v <= x1 + 1e-9; v += stepGrid) {
      const [px] = plane.toPx(v, 0);
      ctx.strokeStyle = Math.abs(v) < 1e-9 ? COLORS.axis : COLORS.grid;
      ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, 0); ctx.lineTo(Math.round(px) + 0.5, plane.size); ctx.stroke();
    }
    for (let v = Math.ceil(y0 / stepGrid) * stepGrid; v <= y1 + 1e-9; v += stepGrid) {
      const [, py] = plane.toPx(0, v);
      ctx.strokeStyle = Math.abs(v) < 1e-9 ? COLORS.axis : COLORS.grid;
      ctx.beginPath(); ctx.moveTo(0, Math.round(py) + 0.5); ctx.lineTo(plane.size, Math.round(py) + 0.5); ctx.stroke();
    }
    if (s.prob && s.contour) drawContour(s.prob, 0.5);   // above the grid
    ctx.font = "11.5px system-ui, sans-serif";
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (const v of ticks) {
      const [px] = plane.toPx(v, 0);
      ctx.fillText(nf(v, Number.isInteger(v) ? 0 : 1), px, plane.size - 4);
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const v of ticks) {
      const [, py] = plane.toPx(0, v);
      ctx.fillText(nf(v, Number.isInteger(v) ? 0 : 1), 5, py);
    }
    ctx.font = "italic 13px Cambria, 'Times New Roman', serif";
    ctx.fillStyle = COLORS.text2;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(axes[0], plane.size - 6, plane.size - 18);
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(axes[1], 22, 6);

    // Lines: ghosts first (dashed), then the others
    for (const line of s.lines || []) {
      const seg = plane.clipLine(line.w, line.b);
      if (!seg) continue;
      const [a, b] = seg.map((p) => plane.toPx(...p));
      ctx.strokeStyle = line.ghost ? (line.color || COLORS.muted) : (line.color || COLORS.line);
      ctx.lineWidth = line.ghost ? 1.6 : 2.6;
      ctx.setLineDash(line.ghost ? [6, 5] : []);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.setLineDash([]);
      if (line.label) {
        const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        ctx.font = "600 12px system-ui, sans-serif";
        ctx.fillStyle = line.color || COLORS.line;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(line.label, mid[0] + 6, mid[1] - 4);
      }
    }

    // The normal vector w, from the boundary towards the class-1 side
    if (s.normal) {
      const f = plane.foot(s.normal.w, s.normal.b);
      if (f) {
        const len = s.normal.length ?? 0.38;
        const tip = [f.p[0] + f.u[0] * len, f.p[1] + f.u[1] * len];
        arrow(f.p, tip, COLORS.text, 2);
        const [tx_, ty_] = plane.toPx(tip[0] + f.u[0] * 0.08, tip[1] + f.u[1] * 0.08);
        ctx.font = "italic 600 14px Cambria, 'Times New Roman', serif";
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("w", tx_, ty_);
        plane.handles = { foot: f.p, tip };
      } else plane.handles = null;
    }

    // Points
    const r = s.radius ?? Math.max(5, Math.min(9, plane.size / 48));
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    for (const p of s.points || []) {
      const [px, py] = plane.toPx(p.x[0], p.x[1]);
      if (p.current) {
        ctx.strokeStyle = COLORS.accent;
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(px, py, r + 7, 0, Math.PI * 2); ctx.stroke();
      } else if (p.wrong) {
        ctx.strokeStyle = COLORS.zero;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.arc(px, py, r + 5, 0, Math.PI * 2); ctx.stroke();
        ctx.setLineDash([]);
      }
      if (p.y) {
        ctx.fillStyle = COLORS.one;
        ctx.strokeStyle = COLORS.surface;
        ctx.lineWidth = r > 4 ? 1.5 : 1;
        ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      } else {
        const k = r * 0.85;
        ctx.strokeStyle = COLORS.surface;
        ctx.lineWidth = r > 4 ? 5 : 3.5;
        ctx.beginPath(); ctx.moveTo(px - k, py - k); ctx.lineTo(px + k, py + k); ctx.moveTo(px + k, py - k); ctx.lineTo(px - k, py + k); ctx.stroke();
        ctx.strokeStyle = COLORS.zero;
        ctx.lineWidth = r > 4 ? 2.6 : 1.8;
        ctx.beginPath(); ctx.moveTo(px - k, py - k); ctx.lineTo(px + k, py + k); ctx.moveTo(px + k, py - k); ctx.lineTo(px - k, py + k); ctx.stroke();
      }
      if (s.coords) {
        const text = p.label ?? `(${p.x[0]}, ${p.x[1]})`;
        ctx.font = "12px system-ui, sans-serif";
        const width = ctx.measureText(text).width;
        let right = p.x[0] > cx;
        if (right && px + 10 + width > plane.size - 4) right = false;   // keep the label inside the plane
        else if (!right && px - 10 - width < 4) right = true;
        const up = p.x[1] > cy;
        ctx.fillStyle = COLORS.text2;
        ctx.textAlign = right ? "left" : "right";
        ctx.textBaseline = up ? "bottom" : "top";
        ctx.fillText(text, px + (right ? 10 : -10), py + (up ? -9 : 9));
      }
    }
    plane.afterDraw?.();
  }

  plane.draw = (scene) => {
    if (scene) plane.scene = scene;
    render();
  };
  const resize = () => {
    const size = canvas.clientWidth;
    if (!size) return;
    plane.size = size;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    render();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  observers.push(observer);
  redrawers.add(() => render());
  return plane;
}

// =====================================================================
// A function graph
// =====================================================================
/**
 * A canvas with x and y axes for curves y = f(x) and points.
 * createGraph(host, { x: [min, max], y: [min, max], xTicks, yTicks, label, xName, yName, ratio })
 * graph.draw(scene). Scene fields, all optional:
 *   curves: [{ f, color, width, dash, fill }]   f(x) → y (null: a gap); fill: area to y = 0
 *   points: [{ x, y, color, r, ring }]
 *   vlines: [{ x, color, dash }];  segments: [{ from, to, color, width, dash }]
 *   labels: [{ x, y, text, color, align, base }]
 */
function createGraph(host, { x, y, xTicks = [], yTicks = [], label = "", xName = "", yName = "", ratio = 1.6 } = {}) {
  const canvas = h("canvas", { role: "img", "aria-label": label });
  const wrap = h("div", { class: "kit-graph", style: `aspect-ratio: ${ratio}` }, canvas);
  host.append(wrap);
  const ctx = canvas.getContext("2d");
  const graph = { canvas, wrap, scene: {}, x, y, xTicks, yTicks };   // x, y and the ticks may change between draws
  const m = { l: 38, r: 12, t: 12, b: 24 };
  graph.toPx = (vx, vy) => [m.l + ((vx - graph.x[0]) / (graph.x[1] - graph.x[0])) * (view.w - m.l - m.r),
    m.t + (1 - (vy - graph.y[0]) / (graph.y[1] - graph.y[0])) * (view.h - m.t - m.b)];
  graph.toData = (px) => graph.x[0] + ((px - m.l) / (view.w - m.l - m.r)) * (graph.x[1] - graph.x[0]);

  function render() {
    if (!view.w) return;
    const s = graph.scene;
    const dpr = canvas.width / view.w;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, view.w, view.h);
    const [left, top] = [m.l, m.t];
    const right = view.w - m.r;
    const bottom = view.h - m.b;
    ctx.save();
    ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    // grid and axes
    ctx.lineWidth = 1;
    for (const t of graph.xTicks) {
      const [px] = graph.toPx(t, 0);
      ctx.strokeStyle = Math.abs(t) < 1e-12 ? COLORS.axis : COLORS.grid;
      ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, top); ctx.lineTo(Math.round(px) + 0.5, bottom); ctx.stroke();
    }
    for (const t of graph.yTicks) {
      const [, py] = graph.toPx(0, t);
      ctx.strokeStyle = Math.abs(t) < 1e-12 ? COLORS.axis : COLORS.grid;
      ctx.beginPath(); ctx.moveTo(left, Math.round(py) + 0.5); ctx.lineTo(right, Math.round(py) + 0.5); ctx.stroke();
    }
    const steps = Math.max(60, Math.round(right - left));
    for (const c of s.curves || []) {
      const pts = [];
      for (let k = 0; k <= steps; k++) {
        const vx = graph.x[0] + (k / steps) * (graph.x[1] - graph.x[0]);
        const vy = c.f(vx);
        pts.push(vy == null || !Number.isFinite(vy) ? null : graph.toPx(vx, vy));
      }
      if (c.fill) {
        const [, zero] = graph.toPx(0, 0);
        ctx.fillStyle = c.fill;
        ctx.beginPath();
        let open = false;
        let lastX = 0;
        for (const p of pts) {
          if (!p) { if (open) { ctx.lineTo(lastX, zero); ctx.closePath(); } open = false; continue; }
          if (!open) { ctx.moveTo(p[0], zero); open = true; }
          ctx.lineTo(p[0], p[1]);
          lastX = p[0];
        }
        if (open) { ctx.lineTo(lastX, zero); ctx.closePath(); }
        ctx.fill();
      }
      ctx.strokeStyle = c.color || COLORS.lineB;
      ctx.lineWidth = c.width || 2.4;
      ctx.lineJoin = "round";
      ctx.setLineDash(c.dash ? [6, 5] : []);
      ctx.beginPath();
      let pen = false;
      for (const p of pts) {
        if (!p) { pen = false; continue; }
        if (pen) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]);
        pen = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const seg of s.segments || []) {
      const a = graph.toPx(...seg.from);
      const b = graph.toPx(...seg.to);
      ctx.strokeStyle = seg.color || COLORS.text;
      ctx.lineWidth = seg.width || 1.6;
      ctx.setLineDash(seg.dash ? [5, 4] : []);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const v of s.vlines || []) {
      const [px] = graph.toPx(v.x, 0);
      ctx.strokeStyle = v.color || COLORS.muted;
      ctx.lineWidth = 1.4;
      ctx.setLineDash(v.dash === false ? [] : [4, 4]);
      ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const p of s.points || []) {
      const [px, py] = graph.toPx(p.x, p.y);
      const r = p.r || 5;
      if (p.ring) {
        ctx.strokeStyle = p.color || COLORS.accent;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(px, py, r + 4, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.fillStyle = p.color || COLORS.accent;
      ctx.strokeStyle = COLORS.surface;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    for (const t of s.labels || []) {
      const [px, py] = graph.toPx(t.x, t.y);
      ctx.font = t.font || "600 12px system-ui, sans-serif";
      ctx.fillStyle = t.color || COLORS.text2;
      ctx.textAlign = t.align || "left";
      ctx.textBaseline = t.base || "bottom";
      ctx.fillText(t.text, px, py);
    }
    ctx.restore();
    // tick labels and names, outside the clip
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of graph.xTicks) ctx.fillText(nf(t, Number.isInteger(t) ? 0 : 1), graph.toPx(t, 0)[0], bottom + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of graph.yTicks) ctx.fillText(nf(t, Number.isInteger(t) ? 0 : (Math.abs(t * 10 - Math.round(t * 10)) < 1e-9 ? 1 : 2)), left - 6, graph.toPx(0, t)[1]);
    ctx.font = "italic 13px Cambria, 'Times New Roman', serif";
    ctx.fillStyle = COLORS.text2;
    if (xName) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(xName, right - 4, bottom - 4); }
    if (yName) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(yName, left + 6, top + 2); }
  }
  const view = fitCanvas(canvas, render);
  graph.draw = (scene) => {
    if (scene) graph.scene = scene;
    render();
  };
  return graph;
}

// =====================================================================
// Controls and cards
// =====================================================================
function slider(label, { min, max, step, value, onInput, fmt = (v) => nf(v) }) {
  const input = h("input", { type: "range", min, max, step, value });
  const out = h("output", { text: fmt(value) });
  input.addEventListener("input", () => {
    out.textContent = fmt(Number(input.value));
    onInput(Number(input.value));
  });
  const node = h("label", {}, h("span", { class: "pc-slider-name" }, label), input, out);
  node.set = (v) => {
    input.value = v;
    out.textContent = fmt(v);
  };
  return node;
}

function segmented(options, current, onPick, aria) {
  const box = h("div", { class: "seg", role: "group", "aria-label": aria || null });
  for (const [value, label] of options) {
    const b = h("button", { type: "button", "aria-pressed": String(value === current), "data-value": value, text: label });
    b.addEventListener("click", () => {
      for (const other of box.children) other.setAttribute("aria-pressed", String(other === b));
      onPick(value);
    });
    box.append(b);
  }
  box.set = (value) => { for (const b of box.children) b.setAttribute("aria-pressed", String(b.dataset.value === value)); };
  return box;
}

function chartHost(height) {
  const host = h("div", { class: "chart", "data-height": height });
  resizeObserver.observe(host);
  return host;
}

/** A Theory section: number, title, a "the math" link, the text on the left and the demo on the right. */
function sectionCard(id, num, title, paragraphs, demo, mathSection = null) {
  const head = h("div", { class: "t-head" }, h("span", { class: "t-num", text: String(num) }), h("h2", { text: L(title) }));
  if (mathSection) head.append(h("a", { href: "#", class: "m-theory-link", "data-go": `math:${mathSection}`, text: L({ tr: "∑ Matematiği →", en: "∑ The math →" }) }));
  const body = h("div", { class: "t-text" }, head, ...paragraphs);
  return h("article", { class: "card t-section", id }, body, demo);
}

/** The card at the end of a chapter's Theory tab that leads to the next chapter. */
function nextChapterCard() {
  const k = CHAPTERS.findIndex((c) => c.id === CHAPTER);
  const next = CHAPTERS[k + 1];
  if (!next) return null;
  const inner = [h("span", { text: L({ tr: "Sıradaki bölüm:", en: "Next chapter:" }) }),
    h("strong", { text: `${k + 2} · ${L(next.title)}` })];
  return next.url
    // a learner who finishes a chapter starts the next one with its theory, not its training tab
    ? h("a", { class: "pc-next link", href: next.url + TAB_HASHES.theory }, ...inner, h("span", { class: "pc-next-go", text: "→" }))
    : h("div", { class: "pc-next" }, ...inner, h("span", { class: "ch-soon", text: L({ tr: "yakında", en: "coming soon" }) }));
}

return {
  R, seeded, gauss, nf, tx, txs, txp, T, html, formula, COLORS, readColors, redrawers, observers, fitCanvas,
  createPlane, createGraph, slider, segmented, chartHost, sectionCard, nextChapterCard,
};
})();
