/*
 * Theory tab: short explanations of the CNN building blocks, each with a live demo that
 * runs on a real MNIST test image and, when a trained model exists, on its own weights.
 * The maths runs in the browser; the server only provides Conv1 / BatchNorm1 weights,
 * test images and the model's logits. Uses the helpers defined in app.js.
 */
import { api, ensureCanvases, infer, modelView } from "./app.js";
import { $, LUTS, clamp, setHoverHost, cssVar, fmtInt, fmtNum, fmtPct, fmtTick, h, hideTip, lineChart, linearTicks, lutColor, navigate, niceStep, paint, probabilityChart, showTip, svg, svgText, watchSize } from "../../shared/core.js";
import { I18N, L, t } from "../../shared/i18n.js";
import { texNum } from "../../shared/math.js";
import { termify } from "../../shared/glossary/terms.js";


export const SIZE = 28;
export const T = { started: false, loading: false, params: null, version: null, batch: [], image: null, filter: 0 };

// =====================================================================
// Math (mirrors what PyTorch does in SimpleCNN)
// =====================================================================

/** 2D cross-correlation (what nn.Conv2d computes), single channel. */
export function conv2d(x, H, W, kernel, k, bias = 0, pad = 0, stride = 1) {
  const oh = Math.floor((H - k + 2 * pad) / stride) + 1;
  const ow = Math.floor((W - k + 2 * pad) / stride) + 1;
  const out = new Float32Array(oh * ow);
  for (let r = 0; r < oh; r++) {
    for (let c = 0; c < ow; c++) {
      let sum = bias;
      for (let m = 0; m < k; m++) {
        const y = r * stride + m - pad;
        if (y < 0 || y >= H) continue;
        for (let n = 0; n < k; n++) {
          const xx = c * stride + n - pad;
          if (xx >= 0 && xx < W) sum += x[y * W + xx] * kernel[m * k + n];
        }
      }
      out[r * ow + c] = sum;
    }
  }
  return { out, oh, ow };
}

/** BatchNorm in evaluation mode: uses the running statistics collected during training. */
export function batchNormEval(z, channel) {
  const bn = T.params.bn1;
  const scale = bn.gamma[channel] / Math.sqrt(bn.var[channel] + bn.eps);
  return z.map((v) => (v - bn.mean[channel]) * scale + bn.beta[channel]);
}

/** 2x2 pooling with stride 2. */
export function pool2(x, rows, cols, mode) {
  const oh = Math.floor(rows / 2);
  const ow = Math.floor(cols / 2);
  const out = new Float32Array(oh * ow);
  for (let r = 0; r < oh; r++) {
    for (let c = 0; c < ow; c++) {
      let best = -Infinity;
      let sum = 0;
      for (let m = 0; m < 2; m++) {
        for (let n = 0; n < 2; n++) {
          const v = x[(2 * r + m) * cols + 2 * c + n];
          sum += v;
          best = Math.max(best, v);
        }
      }
      out[r * ow + c] = mode === "max" ? best : sum / 4;
    }
  }
  return { out, oh, ow };
}

export const relu = (a) => a.map((v) => Math.max(0, v));
export const maxAbs = (a) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m || 1; };
export const maxOf = (a) => { let m = 0; for (const v of a) m = Math.max(m, v); return m || 1; };
export const meanOf = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
export const stdOf = (a, mu = meanOf(a)) => { let s = 0; for (const v of a) s += (v - mu) ** 2; return Math.sqrt(s / a.length); };
export const toSigned = (a, scale) => Array.from(a, (v) => clamp(v / scale, -1, 1));
export const toUnsigned = (a, scale) => Array.from(a, (v) => clamp(Math.round((v / scale) * 255), 0, 255));
export const toFloat = (pixels) => Float32Array.from(pixels, (p) => p / 255);

export function percentile(sorted, q) {
  return sorted[clamp(Math.round(q * (sorted.length - 1)), 0, sorted.length - 1)];
}

/** What ReLU sees in SimpleCNN for the selected filter: BatchNorm(Conv1(x)), 24x24. */
export function featureMap() {
  if (T.params) {
    const { out, oh, ow } = conv2d(T.image.x, SIZE, SIZE, T.params.conv1.weight[T.filter], 5, T.params.conv1.bias[T.filter]);
    return { z: batchNormEval(out, T.filter), rows: oh, cols: ow, title: t("th.bnout", { n: T.filter + 1 }) };
  }
  const { out, oh, ow } = conv2d(T.image.x, SIZE, SIZE, KERNELS.sobelX.values, 3);
  return { z: out, rows: oh, cols: ow, title: t("th.sobelout") };
}

// =====================================================================
// Small view helpers
// =====================================================================
export function showPixels(canvas, values, cols, rows, lut, cell, signedValues = false) {
  paint(canvas, values, cols, rows, lut, signedValues);
  canvas.style.width = `${cols * cell}px`;
  canvas.style.aspectRatio = `${cols} / ${rows}`;
}

/** Positions an overlay box over a pixel grid, in percent so it follows a scaled canvas. */
export function placeBox(box, x, y, w, hgt, cols, rows) {
  Object.assign(box.style, {
    left: `${(x / cols) * 100}%`, top: `${(y / rows) * 100}%`,
    width: `${(w / cols) * 100}%`, height: `${(hgt / rows) * 100}%`,
  });
  box.hidden = false;
}

export function cellFromEvent(event, canvas, cols, rows) {
  const box = canvas.getBoundingClientRect();
  return [
    clamp(Math.floor(((event.clientY - box.top) / box.height) * rows), 0, rows - 1),
    clamp(Math.floor(((event.clientX - box.left) / box.width) * cols), 0, cols - 1),
  ];
}

/** Text color that stays readable on a color-scale cell. */
export function inkFor(lutName, index) {
  const lut = LUTS[lutName];
  const j = clamp(Math.round(index), 0, 255) * 3;
  // WCAG relative luminance: above 0.18, dark text has the higher contrast
  const linear = (c) => (c / 255 <= 0.04045 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4);
  return 0.2126 * linear(lut[j]) + 0.7152 * linear(lut[j + 1]) + 0.0722 * linear(lut[j + 2]) > 0.18 ? "#0b0b0b" : "#ffffff";
}

export function colorCell(cell, value, scale, lutName, signedValues) {
  const index = signedValues ? (clamp(value / scale, -1, 1) + 1) * 127.5 : clamp(value / scale, 0, 1) * 255;
  cell.style.background = lutColor(lutName, index);
  cell.style.color = inkFor(lutName, index);
}

export const fmtValue = (v) => fmtNum(v, Number.isInteger(v) ? 0 : 2);

export function matrix(values, k, lutName, scale, signedValues) {
  const grid = h("div", { class: `matrix k${k}`, style: `grid-template-columns: repeat(${k}, auto)` });
  for (const v of values) {
    const cell = h("span", { text: fmtValue(v) });
    colorCell(cell, v, scale, lutName, signedValues);
    grid.append(cell);
  }
  return grid;
}

export function segmented(host, options, current, onChange) {
  host.replaceChildren(...options.map(([value, label]) => {
    const button = h("button", { type: "button", "aria-pressed": String(value === current), text: label });
    button.addEventListener("click", () => {
      for (const b of host.children) b.setAttribute("aria-pressed", String(b === button));
      onChange(value);
    });
    return button;
  }));
}

export function setPressed(host, index) {
  [...host.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)));
}

/** A function curve with a crosshair; o.marker draws a highlighted point on the curve. */
export function curvePlot(host, o) {
  host._render = () => curvePlot(host, o);
  if (!host.clientWidth) { host._w = 0; return; }   // hidden: the resize observer redraws it once shown
  host._w = host.clientWidth;
  const W = host.clientWidth;
  const H = +host.dataset.height || 180;
  const m = { l: 40, r: 14, t: 10, b: 24 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const X = (v) => m.l + ((v - o.x[0]) / (o.x[1] - o.x[0])) * iw;
  const Y = (v) => m.t + ih - ((clamp(v, o.y[0], o.y[1]) - o.y[0]) / (o.y[1] - o.y[0])) * ih;

  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label });
  for (const t of o.yTicks) {
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(t.v), y2: Y(t.v), class: "gridline" }));
    root.append(svgText(m.l - 8, Y(t.v) + 4, t.label, "tick", "end"));
  }
  for (const t of o.xTicks) root.append(svgText(X(t.v), H - 6, t.label, "tick", "middle"));
  const zeroY = o.y[0] < 0 && o.y[1] > 0 ? Y(0) : m.t + ih;
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: zeroY, y2: zeroY, class: "baseline" }));
  if (o.x[0] < 0 && o.x[1] > 0) root.append(svg("line", { x1: X(0), x2: X(0), y1: m.t, y2: m.t + ih, class: "baseline" }));

  let d = "";
  for (let i = 0; i <= 240; i++) {
    const x = o.x[0] + (i / 240) * (o.x[1] - o.x[0]);
    d += `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(o.fn(x)).toFixed(1)}`;
  }
  root.append(svg("path", { d, class: "series-line", style: "stroke:var(--series-1);stroke-width:2px" }));
  if (o.marker != null) {
    root.append(svg("circle", { cx: X(o.marker), cy: Y(o.fn(o.marker)), r: 5, class: "marker", style: "fill:var(--series-2)" }));
  }

  const cross = svg("line", { y1: m.t, y2: m.t + ih, class: "crosshair", visibility: "hidden" });
  const dot = svg("circle", { r: 4, class: "marker", style: "fill:var(--series-1)", visibility: "hidden" });
  const hit = svg("rect", { x: m.l, y: m.t, width: iw, height: ih, class: "hit" });
  root.append(cross, dot, hit);
  host.replaceChildren(root);

  const show = (clientX, clientY) => {
    const box = root.getBoundingClientRect();
    const x = clamp(o.x[0] + (((clientX - box.left) * (W / box.width) - m.l) / iw) * (o.x[1] - o.x[0]), o.x[0], o.x[1]);
    cross.setAttribute("x1", X(x));
    cross.setAttribute("x2", X(x));
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(x));
    dot.setAttribute("cy", Y(o.fn(x)));
    dot.setAttribute("visibility", "visible");
    const t = o.tooltip(x);
    showTip(clientX, clientY, t.title, t.rows);
    host._hover = [clientX, clientY];
    setHoverHost(host);
  };
  hit.addEventListener("pointermove", (e) => show(e.clientX, e.clientY));
  hit.addEventListener("pointerleave", () => {
    host._hover = null;
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
    hideTip();
  });
  if (host._hover) show(...host._hover);
}

/** Path for a column: 4px rounded top (data end), square at the baseline. */
export function columnPath(x, y, w, hgt, r = 4) {
  r = Math.min(r, w / 2, hgt);
  return `M${x},${y + hgt}V${y + r}A${r},${r} 0 0 1 ${x + r},${y}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${y + hgt}Z`;
}

/** Histogram on a shared x range; a bar far taller than the rest is cut and labelled. */
export function histogram(host, values, lo, hi, mean, opts = {}) {
  host._render = () => histogram(host, values, lo, hi, mean, opts);
  if (!host.clientWidth) { host._w = 0; return; }   // hidden: the resize observer redraws it once shown
  host._w = host.clientWidth;
  const W = host.clientWidth;
  const H = +host.dataset.height || 150;
  const bins = 32;
  const m = { l: 6, r: 6, t: 18, b: 22 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[clamp(Math.floor(((v - lo) / (hi - lo)) * bins), 0, bins - 1)]++;
  const sorted = [...counts].sort((a, b) => b - a);
  let top = sorted[0] || 1;
  if (sorted[1] > 0 && sorted[0] > 2.5 * sorted[1]) top = sorted[1] * 1.6;  // keep the rest readable
  const X = (v) => m.l + ((v - lo) / (hi - lo)) * iw;
  const bw = iw / bins;

  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("th.hist.aria") });
  const span = (a, b, cls) => {   // a shaded x range behind the bars
    const x1 = X(clamp(a, lo, hi));
    const x2 = X(clamp(b, lo, hi));
    if (x2 > x1) root.append(svg("rect", { x: x1, y: m.t, width: x2 - x1, height: ih, class: cls }));
  };
  if (opts.band) span(opts.band[0], opts.band[1], "hist-band");
  if (opts.zero) span(lo, 0, "hist-cut");
  counts.forEach((count, i) => {
    const x = m.l + i * bw;
    const barH = Math.min(count / top, 1) * ih;
    const from = lo + (i / bins) * (hi - lo);
    const to = lo + ((i + 1) / bins) * (hi - lo);
    const group = svg("g");
    if (barH > 0.5) group.append(svg("path", { d: columnPath(x + 0.5, m.t + ih - barH, Math.max(bw - 1, 1), barH, 2), class: "bar-col" }));
    if (count > top) group.append(svgText(x + bw / 2, m.t - 5, `↑ ${fmtInt(count)}`, "cap-label", "middle"));
    group.append(svg("rect", { x, y: m.t, width: bw, height: ih, fill: "transparent" }));
    group.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, `[${fmtNum(from, 2)}, ${fmtNum(to, 2)})`, [
      { value: fmtInt(count), label: t("th.values"), color: "var(--series-1)" },
    ]));
    group.addEventListener("pointerleave", hideTip);
    root.append(group);
  });
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "baseline" }));
  for (const t of linearTicks(lo, hi, niceStep(hi - lo, 4), (v) => fmtTick(v))) {
    root.append(svgText(X(t.v), H - 6, t.label, "tick", "middle"));
  }
  if (mean >= lo && mean <= hi) {
    root.append(svg("line", { x1: X(mean), x2: X(mean), y1: m.t, y2: m.t + ih, class: "mean-line" }));
    root.append(svgText(X(mean) + 4, m.t + 10, "μ", "mean-label"));
  }
  if (opts.zero && lo < 0 && hi > 0) root.append(svgText(X(lo) + 4, m.t + 10, "ReLU → 0", "cut-label"));
  if (opts.mark != null && opts.mark >= lo && opts.mark <= hi) {   // the value being traced
    root.append(svg("line", { x1: X(opts.mark), x2: X(opts.mark), y1: m.t + 4, y2: m.t + ih, class: "hist-mark" }));
    root.append(svg("circle", { cx: X(opts.mark), cy: m.t + 4, r: 4, class: "hist-mark-dot" }));
  }
  host.replaceChildren(root);
}

// =====================================================================
// 1 · Convolution
// =====================================================================
export const KERNELS = {
  model: { label: { tr: "Modelin filtresi (5×5)", en: "The model's filter (5×5)" } },
  sobelX: { label: { tr: "Dikey kenar · Sobel (3×3)", en: "Vertical edge · Sobel (3×3)" }, k: 3, values: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  sobelY: { label: { tr: "Yatay kenar · Sobel (3×3)", en: "Horizontal edge · Sobel (3×3)" }, k: 3, values: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  blur: { label: { tr: "Bulanıklaştırma (3×3)", en: "Blur (3×3)" }, k: 3, values: new Array(9).fill(1 / 9) },
  sharpen: { label: { tr: "Keskinleştirme (3×3)", en: "Sharpen (3×3)" }, k: 3, values: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
  custom: { label: { tr: "Özel · değerleri düzenleyin (3×3)", en: "Custom · edit the values (3×3)" }, k: 3, values: [0, 1, 0, 1, -4, 1, 0, 1, 0] },
};
export const conv = { kernel: "sobelX", picked: false, pad: 0, stride: 1, data: null, reveal: Infinity, anim: 0, pos: null };
export const CONV_CELL = 8;

export function convKernel() {
  if (conv.kernel === "model" && T.params) {
    return { k: 5, values: T.params.conv1.weight[T.filter], bias: T.params.conv1.bias[T.filter] };
  }
  const preset = KERNELS[conv.kernel] || KERNELS.sobelX;
  return { k: preset.k, values: preset.values, bias: 0 };
}

export function fillKernelOptions() {
  const select = $("#t-conv-kernel-select");
  const value = select.value;
  select.replaceChildren(...Object.entries(KERNELS).map(([key, k]) => h("option", { value: key, text: L(k.label) })));
  if (value) select.value = value;
  const modelOption = select.querySelector('option[value="model"]');
  if (modelOption) modelOption.disabled = !T.params;
}

export function setupConv() {
  const select = $("#t-conv-kernel-select");
  fillKernelOptions();
  select.value = conv.kernel;
  select.addEventListener("change", () => {
    conv.kernel = select.value;
    conv.picked = true;
    stopConvAnim();
    updateConv();
  });
  $("#t-conv-pad").addEventListener("change", (e) => { conv.pad = Number(e.target.value); stopConvAnim(); updateConv(); });
  $("#t-conv-stride").addEventListener("change", (e) => { conv.stride = Number(e.target.value); stopConvAnim(); updateConv(); });
  $("#t-conv-play").addEventListener("click", playConv);
  const out = $("#t-conv-out");
  out.addEventListener("pointermove", (e) => {
    if (!conv.data) return;
    stopConvAnim();
    const [r, c] = cellFromEvent(e, out, conv.data.ow, conv.data.oh);
    setConvPos(r * conv.data.ow + c);
  });
}

export function updateConv() {
  if (!T.image) return;
  const ker = convKernel();
  const { out, oh, ow } = conv2d(T.image.x, SIZE, SIZE, ker.values, ker.k, ker.bias, conv.pad, conv.stride);
  conv.data = { ker, out, oh, ow, scale: maxAbs(out) };

  // Input with its zero padding
  const P = conv.pad;
  const Wp = SIZE + 2 * P;
  const padded = new Array(Wp * Wp).fill(0);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) padded[(y + P) * Wp + x + P] = T.image.pixels[y * SIZE + x];
  showPixels($("#t-conv-in"), padded, Wp, Wp, "gray", CONV_CELL);
  if (P) placeBox($("#t-conv-orig"), P, P, SIZE, SIZE, Wp, Wp);
  else $("#t-conv-orig").hidden = true;
  $("#t-conv-in-cap").textContent = P ? t("th.conv.inpad", { p: P }) : t("th.conv.in");

  renderKernel(ker);
  $("#t-conv-out-cap").textContent = t("th.conv.out", { s: `${oh}×${ow}` });
  $("#t-conv-size").textContent = ` \\(\\displaystyle = \\left\\lfloor \\frac{28 - ${ker.k} + 2\\cdot ${P}}{${conv.stride}} \\right\\rfloor + 1 = ${oh}\\)`;
  paintConvOut();

  // Start on the strongest response, so the calculation shows something interesting
  if (conv.pos == null || conv.pos >= oh * ow) {
    let best = 0;
    for (let i = 1; i < out.length; i++) if (Math.abs(out[i]) > Math.abs(out[best])) best = i;
    conv.pos = best;
  }
  setConvPos(conv.pos);
}

export function renderKernel(ker) {
  const host = $("#t-conv-kernel");
  const editable = conv.kernel === "custom";
  const scale = maxAbs(ker.values);
  const key = `${conv.kernel}|${T.filter}|${T.version}|${I18N.lang}`;
  if (editable && host._key === key) {
    // Recolor only, so the input being typed in keeps its focus
    [...host.children].forEach((input, i) => colorCell(input, ker.values[i], scale, "div", true));
    return;
  }
  host._key = key;
  host.className = `kernel-grid k${ker.k}`;
  host.style.gridTemplateColumns = `repeat(${ker.k}, auto)`;
  host.replaceChildren(...Array.from(ker.values, (v, i) => {
    let cell;
    if (editable) {
      cell = h("input", { type: "number", step: "0.5", value: String(v), "aria-label": t("th.conv.weight", { r: Math.floor(i / 3) + 1, c: (i % 3) + 1 }) });
      cell.addEventListener("input", () => {
        KERNELS.custom.values[i] = Number(cell.value) || 0;
        stopConvAnim();
        updateConv();
      });
    } else {
      cell = h("span", { text: fmtValue(v) });
    }
    colorCell(cell, v, scale, "div", true);
    return cell;
  }));
  $("#t-conv-kernel-cap").textContent = ker.k === 5
    ? t("th.conv.kcap5", { n: T.filter + 1, b: fmtNum(ker.bias, 3) })
    : editable ? t("th.conv.kcapedit") : t("th.conv.kcap3");
}

export function paintConvOut() {
  const canvas = $("#t-conv-out");
  const draw = () => {
    const { out, oh, ow, scale } = conv.data;
    paint(canvas, toSigned(out, scale), ow, oh, "div", true);
    const shown = Math.min(conv.reveal, oh * ow);
    if (shown < oh * ow) {   // while sliding: hide what has not been computed yet
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = cssVar("--surface-2");
      const r = Math.floor(shown / ow);
      const c = shown % ow;
      ctx.fillRect(c, r, ow - c, 1);
      ctx.fillRect(0, r + 1, ow, oh - r - 1);
    }
  };
  canvas._repaint = draw;
  draw();
  canvas.style.width = `${conv.data.ow * CONV_CELL}px`;
  canvas.style.aspectRatio = `${conv.data.ow} / ${conv.data.oh}`;
}

export function setConvPos(index) {
  const { ker, oh, ow } = conv.data;
  index = clamp(index, 0, oh * ow - 1);
  conv.pos = index;
  const r = Math.floor(index / ow);
  const c = index % ow;
  const s = conv.stride;
  const P = conv.pad;
  const Wp = SIZE + 2 * P;
  placeBox($("#t-conv-win"), c * s, r * s, ker.k, ker.k, Wp, Wp);
  placeBox($("#t-conv-cell"), c, r, 1, 1, ow, oh);

  // The pixels under the kernel (zero where it hangs over the padding)
  const patch = [];
  let sum = 0;
  for (let m = 0; m < ker.k; m++) {
    for (let n = 0; n < ker.k; n++) {
      const y = r * s + m - P;
      const x = c * s + n - P;
      const v = y >= 0 && y < SIZE && x >= 0 && x < SIZE ? T.image.x[y * SIZE + x] : 0;
      patch.push(v);
      sum += v * ker.values[m * ker.k + n];
    }
  }
  $("#t-conv-calc").replaceChildren(
    h("div", { class: "calc-block" }, h("span", { class: "calc-cap", text: t("th.conv.patch") }), matrix(patch, ker.k, "gray", 1, false)),
    h("span", { class: "op", text: "⊙" }),
    h("div", { class: "calc-block" }, h("span", { class: "calc-cap", text: t("th.conv.kernel") }), matrix(ker.values, ker.k, "div", maxAbs(ker.values), true)),
    h("div", { class: "calc-result" },
      h("span", { class: "calc-cap", text: t("th.outpos", { r: r + 1, c: c + 1 }) }),
      h("span", { text: t("th.conv.sum", { v: fmtNum(sum, 3) }) }),
      ker.bias ? h("span", { text: `+ bias ${fmtNum(ker.bias, 3)}` }) : null,
      h("strong", { text: `= ${fmtNum(sum + ker.bias, 3)}` })),
  );
}

export function playConv() {
  if (conv.anim) {
    stopConvAnim();
    return;
  }
  const total = conv.data.oh * conv.data.ow;
  const perFrame = Math.max(1, Math.ceil(total / 300));   // about 5 seconds
  conv.reveal = 0;
  $("#t-conv-play").textContent = t("th.stop");
  const frame = () => {
    conv.reveal = Math.min(total, conv.reveal + perFrame);
    paintConvOut();
    setConvPos(conv.reveal - 1);
    if (conv.reveal < total) conv.anim = requestAnimationFrame(frame);
    else stopConvAnim();
  };
  conv.anim = requestAnimationFrame(frame);
}

export function stopConvAnim() {
  if (conv.anim) cancelAnimationFrame(conv.anim);
  conv.anim = 0;
  const wasSliding = conv.reveal !== Infinity;
  conv.reveal = Infinity;
  $("#t-conv-play").textContent = t("th.slide");
  if (wasSliding && conv.data) paintConvOut();
}

// =====================================================================
// 2 · Batch normalization, step by step
// Everything is computed here, on the selected Conv1 filter's real output for a batch of
// test images, exactly as BatchNorm2d does in training mode (biased variance, ε from the model).
// =====================================================================
export const BN_MOMENTUM = 0.1;   // PyTorch's default; SimpleCNN does not change it
export const tn = (value, digits) => texNum(fmtNum(value, digits));   // a number inside a TeX formula (math.js)

export const BN_TEXT = {
  lead: {
    tr: "Batch Normalization (BatchNorm), bir katmanın çıktısını sonraki katmana vermeden önce <b>her [[channel|kanalı]] ayrı ayrı</b> standart bir ölçeğe getirir: önce değerleri ortalaması 0, standart sapması 1 olacak şekilde kaydırıp ölçekler, sonra öğrenilen iki sayıyla (γ ve β) ağın ihtiyaç duyduğu ölçeğe taşır. Aşağıda bunu adım adım, seçili filtrenin gerçek test görüntülerindeki çıktısı üzerinde izleyebilirsiniz.",
    en: "Batch normalization (BatchNorm) brings <b>each [[channel|channel]] separately</b> to a standard scale before passing a layer's output on: it first shifts and scales the values to mean 0 and standard deviation 1, then moves them to whatever scale the network needs with two learned numbers (γ and β). Below you can follow it step by step on the selected filter's real output for test images.",
  },
  "flow.x": { tr: "Conv1'in çıktısı: filtre tepkileri", en: "Conv1's output: filter responses" },
  "flow.stats": { tr: "kanalın batch ortalaması ve std. sapması", en: "the channel's batch mean and std. dev." },
  "flow.norm": { tr: "normalleştir", en: "normalize" },
  "flow.affine": { tr: "öğrenilen ölçek ve kayma", en: "learned scale and shift" },
  "flow.relu": { tr: "sonraki işlem", en: "next operation" },

  "s1.h": { tr: "Sorun: her kanal farklı bir ölçekte", en: "The problem: every channel has its own scale" },
  "s1.p": {
    tr: "<p><b>BatchNorm neyi normalleştirir?</b> SimpleCNN'de sıra <b>Conv1 → BatchNorm → ReLU</b>'dur; yani BatchNorm'un girdisi \\(x\\), Conv1'in çıktısıdır. Conv1'in bir [[kernel|filtresi]] 28×28 görüntünün üzerinde kaydırılır ve 24×24 konumun her birinde <b>tek bir sayı</b> üretir: 5×5 filtre ile altındaki 5×5 pikselin çarpımlarının toplamı, artı bias. Bu sayı, filtrenin aradığı desenin o konumda ne kadar güçlü bulunduğunu söyler: büyük pozitif değer \"desen burada var\", negatif değer \"burada tersi var\" demektir. Bu 576 sayı bir [[feature-map|öznitelik haritası]] oluşturur; 2. adımdaki kırmızı-mavi haritalar tam olarak bunlardır. Hesabın kendisi 1. kısımda (Evrişim) adım adım gösteriliyor.</p><p>Conv1'in 32 filtresinin her biri farklı bir desene bakar ve çıktıları farklı aralıklarda olur: kiminin değerleri dar bir aralıkta toplanır, kimininki geniş bir aralığa yayılır; kiminin ortalaması pozitif, kiminin negatiftir. Üstelik ağırlıklar her adımda değiştiği için bu aralıklar eğitim boyunca kayar; sonraki katman sürekli kayan girdilere uyum sağlamak zorunda kalır. BatchNorm'u öneren makale (Ioffe ve Szegedy, 2015) bu kaymayı asıl sorun olarak gördü; sonraki çalışmalar yararın daha çok kayıp yüzeyini düzgünleştirmesinden geldiğini gösterdi. Pratikteki sonuç ise açık: BatchNorm'lu ağlar daha büyük [[learning-rate|öğrenme oranlarıyla]], daha hızlı ve kararlı eğitilir.</p><p>BatchNorm her kanalı aynı başlangıç ölçeğine çeker. Grafikte her çubuk, bir kanalın bu batch'teki değer aralığıdır (ortalama ± 1 standart sapma). Mavi bant, normalleştirmeden sonra <b>bütün kanalların</b> geleceği aralıktır: dar kanallar genişler, geniş olanlar daralır, kaymış olanlar ortalanır.</p>",
    en: "<p><b>What does BatchNorm normalize?</b> In SimpleCNN the order is <b>Conv1 → BatchNorm → ReLU</b>, so BatchNorm's input \\(x\\) is Conv1's output. One Conv1 [[kernel|filter]] slides over the 28×28 image and produces <b>a single number</b> at each of the 24×24 positions: the sum of the products of the 5×5 filter and the 5×5 pixels under it, plus a bias. That number says how strongly the pattern the filter looks for is present at that position: a large positive value means \"the pattern is here\", a negative one \"its opposite is here\". These 576 numbers form a [[feature-map|feature map]]; the red and blue maps in step 2 are exactly these. Section 1 (Convolution) shows the calculation step by step.</p><p>Each of Conv1's 32 filters looks for a different pattern, and their outputs fall in different ranges: some channels' values stay in a narrow range, others spread widely; some have a positive mean, others a negative one. And since the weights change at every step, these ranges drift during training; the next layer has to keep adapting to inputs that keep moving. The paper that introduced BatchNorm (Ioffe and Szegedy, 2015) saw this drift as the main problem; later work showed that the benefit comes more from making the loss surface smoother. The practical result is clear either way: networks with BatchNorm train faster and more stably, with larger [[learning-rate|learning rates]].</p><p>BatchNorm brings every channel to the same starting scale. In the chart, each bar is one channel's range in this batch (mean ± 1 standard deviation). The blue band is where <b>every channel</b> ends up after normalization: narrow channels are widened, wide ones narrowed, shifted ones centered.</p>",
  },
  "s1.cap": { tr: "Bir çubuğa tıklayarak o filtreyi seçin; sonraki adımlar seçili filtre için hesaplanır.", en: "Click a bar to select that filter; the next steps are computed for the selected filter." },

  "s2.h": { tr: "Hangi değerler birlikte normalleştirilir?", en: "Which values are normalized together?" },
  "s2.p": {
    tr: "<p>En çok karıştırılan nokta burası. Conv1'in çıktısı 4 boyutlu bir [[tensor|tensördür]]: <b>(görüntü, kanal, satır, sütun)</b>. BatchNorm her <b>kanal</b> için ayrı çalışır ve o kanalın batch'teki <b>bütün görüntülerdeki bütün konumlarını</b> tek bir havuzda toplar.</p><p>Yani seçili filtre için sağdaki haritaların bütün piksellerinden <b>tek bir μ ve tek bir σ</b> hesaplanır ve hepsine aynı işlem uygulanır. 32 kanal için toplam 32 μ ve 32 σ vardır. Tek bir görüntünün ya da tek bir pikselin kendi istatistiği yoktur.</p><p class=\"bn-note\">Tam bağlı katmanlardan sonra kullanılan <code>BatchNorm1d</code>'de satır ve sütun yoktur: her nöron için yalnızca batch boyunca hesaplanır.</p>",
    en: "<p>This is the part most often misunderstood. Conv1's output is a 4-dimensional [[tensor|tensor]]: <b>(image, channel, row, column)</b>. BatchNorm works on each <b>channel</b> separately and pools <b>every position in every image</b> of the batch for that channel.</p><p>So for the selected filter, <b>one μ and one σ</b> are computed from all the pixels of the maps on the right, and the same operation is applied to all of them. For 32 channels there are 32 μ and 32 σ in total. A single image, or a single pixel, has no statistics of its own.</p><p class=\"bn-note\">In <code>BatchNorm1d</code>, used after fully connected layers, there are no rows and columns: each neuron is normalized over the batch only.</p>",
  },
  "s2.cap": { tr: "Bir haritanın üzerine gelin ya da dokunun: o pikselin değerini sonraki adımlarda izleyin.", en: "Hover over or tap a map: follow that pixel's value through the next steps." },
  "axes.image": { tr: "görüntü", en: "images" },
  "axes.channel": { tr: "kanal", en: "channels" },
  "axes.row": { tr: "satır", en: "rows" },
  "axes.col": { tr: "sütun", en: "columns" },
  "axes.pooled": { tr: "birlikte", en: "pooled" },
  "axes.separate": { tr: "ayrı ayrı", en: "separately" },

  "s3.h": { tr: "Kanalın ortalamasını ve standart sapmasını hesapla", en: "Compute the channel's mean and standard deviation" },
  "s3.p": {
    tr: "<p>Havuzdaki \\(m\\) değerin ortalaması \\(\\mu_B\\) ve varyansı \\(\\sigma_B^2\\) hesaplanır (B: batch):</p><div class=\"formula small\">\\(\\displaystyle \\mu_B = \\frac{1}{m}\\sum_{i=1}^{m} x_i \\qquad \\sigma_B^2 = \\frac{1}{m}\\sum_{i=1}^{m} (x_i - \\mu_B)^2\\)</div><p>Bunlar <b>bu batch'in</b> istatistikleridir; başka bir batch'te biraz farklı çıkarlar. Yukarıdaki <b>Yeni batch</b> düğmesiyle deneyin.</p>",
    en: "<p>The mean \\(\\mu_B\\) and variance \\(\\sigma_B^2\\) of the \\(m\\) pooled values are computed (B: batch):</p><div class=\"formula small\">\\(\\displaystyle \\mu_B = \\frac{1}{m}\\sum_{i=1}^{m} x_i \\qquad \\sigma_B^2 = \\frac{1}{m}\\sum_{i=1}^{m} (x_i - \\mu_B)^2\\)</div><p>These are <b>this batch's</b> statistics; another batch gives slightly different ones. Try the <b>New batch</b> button above.</p>",
  },
  h1: { tr: "BatchNorm'un girdisi x (Conv1'in çıktısı) · gölgeli alan μ ± σ", en: "BatchNorm's input x (Conv1's output) · shaded area μ ± σ" },

  "s4.h": { tr: "Normalleştir: ortalamayı çıkar, standart sapmaya böl", en: "Normalize: subtract the mean, divide by the standard deviation" },
  "s4.p": {
    tr: "<div class=\"formula small\">\\(\\displaystyle \\hat{x} = \\frac{x - \\mu_B}{\\sqrt{\\sigma_B^2 + \\varepsilon}}\\)</div><p>Her değerden kanal ortalaması çıkarılır (dağılım 0'a kayar) ve standart sapmaya bölünür (genişliği 1 olur: σ'sı 1'den küçük olan bir kanal genişler, büyük olan daralır). Dağılımın <b>şekli değişmez</b>, yalnızca yeri ve genişliği değişir: üstteki ve alttaki histogramlar aynı eksende çizildi.</p><p>ε çok küçük bir sayıdır (0,00001); σ sıfıra çok yakın olduğunda sıfıra bölmeyi önler.</p><p class=\"bn-note\">İlginç bir sonuç: evrişimin bias'ı her değere aynı sabiti ekler ve ortalama çıkarılırken bu sabit de silinir. Bu yüzden BatchNorm'dan önce gelen katmanlarda bias çoğu zaman kapatılır (<code>bias=False</code>).</p>",
    en: "<div class=\"formula small\">\\(\\displaystyle \\hat{x} = \\frac{x - \\mu_B}{\\sqrt{\\sigma_B^2 + \\varepsilon}}\\)</div><p>The channel mean is subtracted from every value (the distribution moves to 0) and the result is divided by the standard deviation (its width becomes 1: a channel with σ below 1 gets wider, one above 1 narrower). The <b>shape</b> of the distribution <b>does not change</b>, only its position and width: the histograms above and below use the same axis.</p><p>ε is a tiny number (0.00001) that prevents dividing by zero when σ is very close to zero.</p><p class=\"bn-note\">A neat consequence: the convolution's bias adds the same constant to every value, and subtracting the mean removes that constant too. That is why layers right before a BatchNorm often have their bias turned off (<code>bias=False</code>).</p>",
  },
  h2: { tr: "Normalleştirilmiş x̂ · gölgeli alan −1 … 1", en: "Normalized x̂ · shaded area −1 … 1" },

  "s5.h": { tr: "Ölçekle ve kaydır: γ ve β", en: "Scale and shift: γ and β" },
  "s5.p": {
    tr: "<div class=\"formula small\">\\(\\displaystyle y = \\gamma\\, \\hat{x} + \\beta\\)</div><p>Her kanalı zorla ortalama 0, standart sapma 1'de tutmak ağı kısıtlayabilir: örneğin bir kanalın [[relu|ReLU]]'dan önce çoğunlukla negatif kalması, yani yalnızca güçlü desenlerde \"yanması\" daha yararlı olabilir. Bu yüzden her kanalın öğrenilen iki parametresi vardır: <b>γ</b> (ölçek) dağılımı genişletir ya da daraltır, <b>β</b> (kayma) sağa ya da sola kaydırır.</p><p>γ ve β da ağırlıklar gibi [[backprop|geri yayılımla]] öğrenilir. Ağ isterse γ = σ ve β = μ seçerek normalleştirmeyi tamamen geri alabilir; yani BatchNorm ağdan bir şey eksiltmez, yalnızca öğrenmeyi kolaylaştırır.</p><p>Kaydırıcılarla deneyin; <b>Modelin γ, β değerleri</b> düğmesi eğitimin bulduğu değerlere döner.</p>",
    en: "<div class=\"formula small\">\\(\\displaystyle y = \\gamma\\, \\hat{x} + \\beta\\)</div><p>Forcing every channel to mean 0 and standard deviation 1 could restrict the network: it may be more useful for a channel to stay mostly negative before [[relu|ReLU]], so that it only \"fires\" on strong patterns. That is why every channel has two learned parameters: <b>γ</b> (scale) widens or narrows the distribution, <b>β</b> (shift) moves it left or right.</p><p>γ and β are learned with [[backprop|backpropagation]] just like the weights. If it helps, the network can choose γ = σ and β = μ and undo the normalization completely; BatchNorm takes nothing away from the network, it only makes learning easier.</p><p>Try the sliders; the <b>Model's γ and β</b> button goes back to the values training found.</p>",
  },
  h3: { tr: "BatchNorm çıktısı y · sağdaki kısım ReLU'dan geçer", en: "BatchNorm output y · the part right of 0 passes ReLU" },

  "s6.h": { tr: "Eğitimde ve çıkarımda farklı çalışır", en: "It works differently in training and at inference" },
  "s6.p": {
    tr: "<p>Yukarıdaki \\(\\mu_B\\) ve \\(\\sigma_B\\) bu batch'ten hesaplandı; eğitimde ([[train-eval|<code>model.train()</code>]]) tam olarak böyle çalışır. Ama [[inference|çıkarımda]] tek bir görüntü gelebilir ve bir görüntünün sonucunun yanındaki başka görüntülere bağlı olması istenmez.</p><p>Bu yüzden BatchNorm, eğitimin her adımında batch istatistiklerinin [[moving-average|hareketli ortalamasını]] günceller ve çıkarımda (<code>model.eval()</code>) bunları kullanır. Bu ortalamalar öğrenilmez, yalnızca biriktirilir ve [[checkpoint|checkpoint]]'e γ ve β ile birlikte kaydedilir.</p>",
    en: "<p>The \\(\\mu_B\\) and \\(\\sigma_B\\) above were computed from this batch; that is exactly how it works in training ([[train-eval|<code>model.train()</code>]]). But at [[inference|inference]] a single image may come in, and an image's result should not depend on the other images next to it.</p><p>That is why, at every training step, BatchNorm updates a [[moving-average|moving average]] of the batch statistics, and uses those at inference (<code>model.eval()</code>). They are not learned, only accumulated, and they are saved in the [[checkpoint|checkpoint]] together with γ and β.</p>",
  },

  model: {
    tr: "<b>SimpleCNN'de:</b> Her evrişimden sonra, ReLU'dan önce: <code>BatchNorm2d(32)</code> ve <code>BatchNorm2d(64)</code>. Her biri kanal başına bir γ ve bir β öğrenir (toplam 2 × (32 + 64) = 192 parametre) ve kanal başına bir hareketli ortalama ile bir hareketli varyans saklar.",
    en: "<b>In SimpleCNN:</b> after every convolution, before ReLU: <code>BatchNorm2d(32)</code> and <code>BatchNorm2d(64)</code>. Each learns one γ and one β per channel (2 × (32 + 64) = 192 parameters in total) and keeps one running mean and one running variance per channel.",
  },
};

export const bnState = {
  gamma: 1, beta: 0, z: null, zSorted: null, xh: null, xhSorted: null, mu: 0, sd: 1,
  pick: null, channels: null, channelsKey: null, frame: 0,
};

export function fillBnText() {
  for (const el of document.querySelectorAll("#t-bn [data-bn]")) {
    el.innerHTML = termify(L(BN_TEXT[el.dataset.bn]));   // static, author-written markup
  }
}

export function setupBN() {
  fillBnText();
  const gamma = $("#t-bn-gamma");
  const beta = $("#t-bn-beta");
  gamma.addEventListener("input", () => { bnState.gamma = Number(gamma.value); updateBN(); });
  beta.addEventListener("input", () => { bnState.beta = Number(beta.value); updateBN(); });
  $("#t-bn-reset").addEventListener("click", () => { resetBnParams(); updateBN(); });
  $("#t-bn-batch").addEventListener("click", async () => {
    await loadBatch(false);
    computeBnBatch();
    updateBN();
  });
  // Pick a pixel on the maps (hover, or tap on touch screens)
  const maps = $("#t-bn-maps");
  const pickFrom = (event) => {
    const canvas = event.target.closest?.("canvas");
    if (!canvas || !bnState.z) return;
    const [r, c] = cellFromEvent(event, canvas, 24, 24);
    const pick = { img: Number(canvas.dataset.i), r, c };
    if (bnState.pick && pick.img === bnState.pick.img && r === bnState.pick.r && c === bnState.pick.c) return;
    bnState.pick = pick;
    cancelAnimationFrame(bnState.frame);
    bnState.frame = requestAnimationFrame(updateBN);
  };
  maps.addEventListener("pointermove", pickFrom);
  maps.addEventListener("pointerdown", pickFrom);
  $("#t-bn-channels").addEventListener("click", (event) => {
    const hit = event.target.closest?.("[data-channel]");
    if (hit) selectFilter(Number(hit.dataset.channel));
  });
}

/** Mean and standard deviation of every Conv1 channel over the batch (step 1). */
export function channelStats() {
  const { weight, bias } = T.params.conv1;
  return weight.map((w, ch) => {
    let sum = 0;
    let sq = 0;
    let n = 0;
    for (const sample of T.batch) {
      for (const v of conv2d(sample.x, SIZE, SIZE, w, 5, bias[ch]).out) {
        sum += v;
        sq += v * v;
        n++;
      }
    }
    const mu = sum / n;
    return { mu, sd: Math.sqrt(Math.max(sq / n - mu * mu, 0)) };
  });
}

/** Conv1 output of the selected filter for every image in the batch: N x 24 x 24 values. */
export function computeBnBatch() {
  if (!T.params || !T.batch.length) return;
  const key = `${T.version}|${T.batch.map((s) => s.index).join(",")}`;
  if (bnState.channelsKey !== key) {
    bnState.channels = channelStats();
    bnState.channelsKey = key;
  }
  const w = T.params.conv1.weight[T.filter];
  const b = T.params.conv1.bias[T.filter];
  const z = new Float32Array(T.batch.length * 576);
  T.batch.forEach((sample, i) => z.set(conv2d(sample.x, SIZE, SIZE, w, 5, b).out, i * 576));
  const mu = meanOf(z);
  const sd = stdOf(z, mu);
  const denom = Math.sqrt(sd * sd + T.params.bn1.eps);
  Object.assign(bnState, {
    z, mu, sd, denom, filter: T.filter,
    zSorted: Float32Array.from(z).sort(),
    xh: z.map((v) => (v - mu) / denom),
  });
  bnState.xhSorted = Float32Array.from(bnState.xh).sort();
  // Start by following the strongest response in the first image
  let best = 0;
  for (let i = 1; i < 576; i++) if (z[i] > z[best]) best = i;
  bnState.pick = { img: 0, r: Math.floor(best / 24), c: best % 24 };
  renderBnMaps();
}

export function resetBnParams() {
  if (!T.params) return;
  bnState.gamma = T.params.bn1.gamma[T.filter];
  bnState.beta = T.params.bn1.beta[T.filter];
}

/** Step 2: the selected channel's map for every image in the batch, on one color scale. */
export function renderBnMaps() {
  const host = $("#t-bn-maps");
  const { z } = bnState;
  const n = T.batch.length;
  if (host.querySelectorAll("canvas").length !== n) {
    host.replaceChildren(...Array.from({ length: n }, (_, i) =>
      h("div", { class: "bn-map" }, h("canvas", { class: "px", "data-i": i }))));
    host.append(h("div", { class: "cell-box bn-pick", hidden: true }));
  }
  const scale = maxAbs(z);
  host.querySelectorAll("canvas").forEach((canvas, i) => {
    paint(canvas, toSigned(z.subarray(i * 576, (i + 1) * 576), scale), 24, 24, "div", true);
  });
}

/** Step 1: every channel's range (μ ± σ) before BatchNorm, and the band they all share after it. */
export function renderBnChannels(host) {
  host._render = () => renderBnChannels(host);
  if (!host.clientWidth) { host._w = 0; return; }
  host._w = host.clientWidth;
  const stats = bnState.channels;
  const W = host.clientWidth;
  const H = +host.dataset.height || 220;
  const m = { l: 40, r: 10, t: 12, b: 26 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  let lo = -1.3;
  let hi = 1.3;
  for (const s of stats) {
    lo = Math.min(lo, s.mu - s.sd);
    hi = Math.max(hi, s.mu + s.sd);
  }
  const padY = (hi - lo) * 0.06;
  lo -= padY;
  hi += padY;
  const Y = (v) => m.t + ((hi - v) / (hi - lo)) * ih;
  const slot = iw / stats.length;
  const X = (i) => m.l + (i + 0.5) * slot;

  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": L({ tr: "Kanal başına değer aralıkları", en: "Value range per channel" }) });
  for (const tick of linearTicks(lo, hi, niceStep(hi - lo, 5), (v) => fmtTick(v))) {
    root.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(tick.v), y2: Y(tick.v), class: "gridline" }));
    root.append(svgText(m.l - 6, Y(tick.v) + 4, tick.label, "tick", "end"));
  }
  root.append(svg("rect", { x: m.l, y: Y(1), width: iw, height: Y(-1) - Y(1), class: "bn-band" }));
  root.append(svgText(W - m.r - 4, Y(1) - 5, L({ tr: "BatchNorm sonrası: hepsi μ = 0, σ = 1", en: "after BatchNorm: all at μ = 0, σ = 1" }), "bn-band-label", "end"));
  root.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "baseline" }));
  stats.forEach((s, i) => {
    const selected = i === T.filter;
    const group = svg("g", { class: `bn-ch${selected ? " selected" : ""}`, "data-channel": i });
    group.append(svg("rect", { x: m.l + i * slot, y: m.t, width: slot, height: ih, fill: "transparent" }));
    group.append(svg("line", { x1: X(i), x2: X(i), y1: Y(s.mu + s.sd), y2: Y(s.mu - s.sd), class: "bn-range" }));
    group.append(svg("circle", { cx: X(i), cy: Y(s.mu), r: selected ? 4.5 : 3, class: "bn-mu" }));
    group.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, t("th.filter.n", { n: i + 1 }), [
      { value: fmtNum(s.mu, 3), label: "μ", color: "var(--series-2)" },
      { value: fmtNum(s.sd, 3), label: "σ", color: "var(--series-2)" },
    ]));
    group.addEventListener("pointerleave", hideTip);
    root.append(group);
    if (i % 4 === 0 || selected) root.append(svgText(X(i), H - 8, String(i + 1), selected ? "tick bn-tick-selected" : "tick", "middle"));
  });
  host.replaceChildren(root);
}

/** One line of a worked calculation: label, then the expression, then the result. */
export function bnCalc(host, rows) {
  host.replaceChildren(...rows.map(([label, expr, result]) => h("div", { class: "bn-calc-row" },
    h("span", { class: "calc-cap", text: label }),
    h("span", { class: "bn-expr", text: expr }),
    result != null ? h("strong", { text: result }) : null)));
}

export function updateBN() {
  if (!T.params || !bnState.z) return;
  const { z, xh, zSorted, xhSorted, mu, sd, gamma, beta, pick } = bnState;
  const n = T.batch.length;
  const eps = T.params.bn1.eps;
  $("#t-bn-gamma").value = gamma;
  $("#t-bn-beta").value = beta;
  $("#t-bn-gamma-v").textContent = fmtNum(gamma, 2);
  $("#t-bn-beta-v").textContent = fmtNum(beta, 2);
  $("#t-bn-sub").textContent = L({
    tr: `Filtre #${T.filter + 1} · batch: ${n} test görüntüsü`,
    en: `Filter #${T.filter + 1} · batch: ${n} test images`,
  });

  // Step 1
  renderBnChannels($("#t-bn-channels"));
  const mus = bnState.channels.map((c) => c.mu);
  const sds = bnState.channels.map((c) => c.sd);
  const range = (a, d) => `${fmtNum(Math.min(...a), d)} … ${fmtNum(Math.max(...a), d)}`;
  bnCalc($("#t-bn-chstats"), [
    [L({ tr: "BatchNorm öncesi", en: "Before BatchNorm" }),
      L({ tr: `ortalamalar ${range(mus, 2)}, std. sapmalar ${range(sds, 2)}`, en: `means ${range(mus, 2)}, std. devs ${range(sds, 2)}` }),
      L({ tr: `(en geniş kanal en darın ${fmtNum(Math.max(...sds) / Math.max(Math.min(...sds), 1e-9), 1)} katı)`,
        en: `(the widest channel is ${fmtNum(Math.max(...sds) / Math.max(Math.min(...sds), 1e-9), 1)}× the narrowest)` })],
    [L({ tr: "Normalleştirme sonrası", en: "After normalizing" }),
      L({ tr: "her kanalın ortalaması 0, std. sapması 1", en: "every channel has mean 0 and std. dev. 1" }), null],
  ]);

  // Step 2: which axes are pooled
  const axis = (size, name, pooled) => h("span", { class: `bn-axis${pooled ? "" : " separate"}` },
    h("b", { text: String(size) }), h("span", { text: L(BN_TEXT[`axes.${name}`]) }),
    h("small", { text: L(BN_TEXT[pooled ? "axes.pooled" : "axes.separate"]) }));
  $("#t-bn-axes").replaceChildren(
    h("span", { class: "bn-paren", text: "(" }), axis(n, "image", true), axis(32, "channel", false), axis(24, "row", true), axis(24, "col", true),
    h("span", { class: "bn-paren", text: ")" }),
    h("span", { class: "bn-axes-sum", text: L({
      tr: `→ filtre #${T.filter + 1} için ${n} × 24 × 24 = ${fmtInt(z.length)} değer → 1 μ, 1 σ`,
      en: `→ for filter #${T.filter + 1}: ${n} × 24 × 24 = ${fmtInt(z.length)} values → 1 μ, 1 σ`,
    }) }));
  const index = pick.img * 576 + pick.r * 24 + pick.c;
  const box = $("#t-bn-maps .bn-pick");
  const cell = $(`#t-bn-maps canvas[data-i="${pick.img}"]`)?.parentElement;
  if (cell) {
    cell.append(box);
    placeBox(box, pick.c, pick.r, 1, 1, 24, 24);
  }
  for (const map of document.querySelectorAll("#t-bn-maps .bn-map")) map.classList.toggle("picked", map === cell);

  // Steps 3 to 5 share one x range, so the shift and the squeeze are visible
  const x = z[index];
  const xn = xh[index];
  const y = gamma * xn + beta;
  const ys = xh.map((v) => gamma * v + beta);
  const ends = [percentile(zSorted, 0.002), percentile(zSorted, 0.998), percentile(xhSorted, 0.002), percentile(xhSorted, 0.998)];
  ends.push(gamma * ends[2] + beta, gamma * ends[3] + beta);
  let lo = Math.min(...ends);
  let hi = Math.max(...ends);
  const pad = (hi - lo) * 0.04 || 1;
  lo -= pad;
  hi += pad;
  const yMean = gamma * meanOf(xh) + beta;
  histogram($("#t-bn-h1"), z, lo, hi, mu, { band: [mu - sd, mu + sd], mark: x });
  histogram($("#t-bn-h2"), xh, lo, hi, meanOf(xh), { band: [-1, 1], mark: xn });
  histogram($("#t-bn-h3"), ys, lo, hi, yMean, { mark: y, zero: true });

  const where = L({ tr: `görüntü ${pick.img + 1}, satır ${pick.r + 1}, sütun ${pick.c + 1}`, en: `image ${pick.img + 1}, row ${pick.r + 1}, column ${pick.c + 1}` });
  $("#t-bn-stats").replaceChildren(
    h("span", { text: `\\(m = ${texNum(fmtInt(z.length))}\\)` }),
    h("span", { text: `\\(\\mu_B = ${tn(mu, 4)}\\)` }),
    h("span", { text: `\\(\\sigma_B = ${tn(sd, 4)}\\)` }));
  bnCalc($("#t-bn-c1"), [
    [L({ tr: "İzlenen değer", en: "Traced value" }), `\\(x = ${tn(x, 4)}\\)`, null],
    ["", where, null],
  ]);
  bnCalc($("#t-bn-c2"), [
    [L({ tr: "İzlenen değer", en: "Traced value" }),
      `\\(\\hat{x} = \\dfrac{${tn(x, 4)} - ${tn(mu, 4)}}{\\sqrt{${tn(sd, 4)}^2 + ${tn(eps, 5)}}} = \\mathbf{${tn(xn, 3)}}\\)`, null],
    [L({ tr: "Bütün kanal", en: "Whole channel" }),
      L({ tr: `ortalama ${fmtNum(meanOf(xh), 3)}, std. sapma ${fmtNum(stdOf(xh), 3)}`, en: `mean ${fmtNum(meanOf(xh), 3)}, std. dev. ${fmtNum(stdOf(xh), 3)}` }), null],
  ]);
  const active = ys.reduce((count, v) => count + (v > 0), 0) / ys.length;
  bnCalc($("#t-bn-c3"), [
    [L({ tr: "İzlenen değer", en: "Traced value" }),
      `\\(y = ${tn(gamma, 2)} \\cdot ${tn(xn, 3)} + (${tn(beta, 2)}) = \\mathbf{${tn(y, 3)}}\\)`, null],
    ["ReLU", `\\(\\max(0,\\ ${tn(y, 3)}) = \\mathbf{${tn(Math.max(0, y), 3)}}\\)`, null],
    [L({ tr: "Bütün kanal", en: "Whole channel" }),
      L({ tr: `ortalama ${fmtNum(yMean, 3)}, std. sapma ${fmtNum(stdOf(ys, yMean), 3)} · ReLU'dan geçen (y > 0): ${fmtPct(active * 100, 1)}`,
        en: `mean ${fmtNum(yMean, 3)}, std. dev. ${fmtNum(stdOf(ys, yMean), 3)} · passing ReLU (y > 0): ${fmtPct(active * 100, 1)}` }), null],
  ]);

  // Step 6: training vs inference
  const bn = T.params.bn1;
  const runMu = bn.mean[T.filter];
  const runSd = Math.sqrt(bn.var[T.filter]);
  const yEval = gamma * ((x - runMu) / Math.sqrt(bn.var[T.filter] + eps)) + beta;
  const initial = modelView.which === "initial";
  const modeCard = (cls, title, code, lines) => h("div", { class: `bn-mode ${cls}` },
    h("div", { class: "bn-mode-head" }, h("strong", { text: title }), h("code", { text: code })),
    ...lines.map((line) => (typeof line === "string" ? h("p", { text: line }) : line)));
  const stat = (label, value) => h("div", { class: "bn-mode-stat" }, h("span", { text: label }), h("strong", { text: value }));
  $("#t-bn-modes").replaceChildren(
    modeCard("train", L({ tr: "Eğitimde", en: "In training" }), "model.train()", [
      stat(L({ tr: "kullanılan μ, σ", en: "μ, σ used" }), `${fmtNum(mu, 4)}, ${fmtNum(sd, 4)}`),
      L({ tr: "Bu batch'ten hesaplanır; her batch'te biraz değişir.", en: "Computed from this batch; changes a little with every batch." }),
      h("div", { class: "formula small", text: `\\(\\mu_{\\text{run}} \\leftarrow ${tn(1 - BN_MOMENTUM, 1)}\\, \\mu_{\\text{run}} + ${tn(BN_MOMENTUM, 1)}\\, \\mu_B\\)` }),
      L({ tr: "Her adımda hareketli ortalama da güncellenir (aynısı varyans için).", en: "At every step the running average is updated too (the same for the variance)." }),
      stat(L({ tr: "izlenen değer → y", en: "traced value → y" }), fmtNum(y, 3)),
    ]),
    modeCard("eval", L({ tr: "Çıkarımda", en: "At inference" }), "model.eval()", [
      stat(L({ tr: "kullanılan μ, σ", en: "μ, σ used" }), `${fmtNum(runMu, 4)}, ${fmtNum(runSd, 4)}`),
      initial
        ? L({ tr: "Eğitim öncesi model: hareketli ortalamalar henüz başlangıç değerinde (μ = 0, σ = 1).", en: "Model before training: the running averages still have their initial values (μ = 0, σ = 1)." })
        : L({ tr: "Eğitim boyunca biriktirilen hareketli ortalamalar; sabittir, batch'e bağlı değildir.", en: "The running averages collected during training; fixed, independent of the batch." }),
      L({ tr: "Görüntüler tek tek de işlenebilir ve her görüntünün sonucu yalnızca kendisine bağlıdır.", en: "Images can be processed one by one, and each image's result depends only on itself." }),
      stat(L({ tr: "izlenen değer → y", en: "traced value → y" }), fmtNum(yEval, 3)),
    ]));
}

// =====================================================================
// 3 · Activation functions
// =====================================================================
export const ACTIVATIONS = {
  relu: {
    label: "ReLU", f: (x) => Math.max(0, x), y: [-1, 4],
    formula: "\\(\\displaystyle \\mathrm{ReLU}(z) = \\max(0,\\, z)\\)",
    desc: { tr: "Negatif değerleri 0 yapar, pozitifleri olduğu gibi geçirir. Haritada mavi (negatif) bölgelerin tamamen kaybolduğuna dikkat edin.",
      en: "Sets negative values to 0 and passes positive values unchanged. Notice how the blue (negative) regions disappear completely from the map." },
  },
  leaky: {
    label: "Leaky ReLU", f: (x) => (x > 0 ? x : 0.01 * x), y: [-1, 4],
    formula: {
      tr: "\\(\\displaystyle \\mathrm{LReLU}(z) = \\begin{cases} z & z > 0 \\text{ ise} \\\\ 0{,}01\\, z & \\text{aksi hâlde} \\end{cases}\\)",
      en: "\\(\\displaystyle \\mathrm{LReLU}(z) = \\begin{cases} z & \\text{if } z > 0 \\\\ 0.01\\, z & \\text{otherwise} \\end{cases}\\)",
    },
    desc: { tr: "ReLU gibidir ama negatif tarafta küçük bir eğim (0,01) bırakır; böylece negatif girişlerde de [[gradient|gradyan]] tamamen sıfırlanmaz.",
      en: "Like ReLU, but keeps a small slope (0.01) on the negative side, so the [[gradient|gradient]] does not vanish entirely for negative inputs." },
  },
  sigmoid: {
    label: "Sigmoid", f: (x) => 1 / (1 + Math.exp(-x)), y: [-0.25, 1.25],
    formula: "\\(\\displaystyle \\sigma(z) = \\frac{1}{1 + e^{-z}}\\)",
    desc: { tr: "Her değeri 0 ile 1 arasına sıkıştırır. Eğimi en fazla 0,25'tir ve |z| büyüdükçe sıfıra yaklaşır; derin ağlarda [[vanishing-gradient|kaybolan gradyana]] yol açabilir.",
      en: "Squeezes every value into the range 0 to 1. Its slope is at most 0.25 and approaches zero as |z| grows, which can cause [[vanishing-gradient|vanishing gradients]] in deep networks." },
  },
  tanh: {
    label: "Tanh", f: Math.tanh, y: [-1.25, 1.25],
    formula: "\\(\\displaystyle \\tanh(z) = \\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}\\)",
    desc: { tr: "Değerleri −1 ile 1 arasına sıkıştırır ve 0 etrafında simetriktir. Sigmoid gibi uç değerlerde doyuma ulaşır.",
      en: "Squeezes values into the range −1 to 1 and is symmetric around 0. Like the sigmoid, it saturates at extreme values." },
  },
};
export const act = { kind: "relu", data: null };

export function setupActivation() {
  segmented($("#t-act-seg"), Object.entries(ACTIVATIONS).map(([key, a]) => [key, a.label]), act.kind, (kind) => {
    act.kind = kind;
    updateActivation();
  });
  for (const id of ["#t-act-before", "#t-act-after"]) {
    const canvas = $(id);
    canvas.addEventListener("pointermove", (e) => {
      if (!act.data) return;
      const [r, c] = cellFromEvent(e, canvas, act.data.cols, act.data.rows);
      const i = r * act.data.cols + c;
      showTip(e.clientX, e.clientY, t("th.pos", { r: r + 1, c: c + 1 }), [
        { value: fmtNum(act.data.z[i], 3), label: t("th.act.before") },
        { value: fmtNum(act.data.a[i], 3), label: t("th.act.after", { f: ACTIVATIONS[act.kind].label }) },
      ]);
    });
    canvas.addEventListener("pointerleave", hideTip);
  }
}

export function updateActivation() {
  const fn = ACTIVATIONS[act.kind];
  $("#t-act-formula").innerHTML = L(fn.formula);   // static, author-written markup
  $("#t-act-desc").innerHTML = termify(L(fn.desc));   // static, author-written text; [[key|text]] → glossary links
  curvePlot($("#t-act-plot"), {
    label: t("th.act.plot", { f: fn.label }),
    x: [-4, 4], y: fn.y, fn: fn.f,
    xTicks: [-4, -2, 0, 2, 4].map((v) => ({ v, label: fmtTick(v) })),
    yTicks: linearTicks(fn.y[0], fn.y[1], niceStep(fn.y[1] - fn.y[0], 4), fmtTick),
    tooltip: (x) => ({ title: `z = ${fmtNum(x, 2)}`, rows: [{ value: fmtNum(fn.f(x), 3), label: `${fn.label}(z)`, color: "var(--series-1)" }] }),
  });
  if (!T.image) return;

  const { z, rows, cols, title } = featureMap();
  const a = z.map(fn.f);
  const scale = Math.max(maxAbs(z), maxAbs(a));
  act.data = { z, a, rows, cols };
  showPixels($("#t-act-before"), toSigned(z, scale), cols, rows, "div", 6, true);
  showPixels($("#t-act-after"), toSigned(a, scale), cols, rows, "div", 6, true);
  $("#t-act-before-cap").textContent = title;
  $("#t-act-after-cap").textContent = t("th.act.aftercap", { f: fn.label });

  const negatives = z.reduce((n, v) => n + (v < 0), 0) / z.length;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of a) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  $("#t-act-stats").replaceChildren(...[
    h("span", {}, t("th.act.neg"), " ", h("strong", { text: fmtPct(negatives * 100, 0) })),
    act.kind === "relu" ? h("span", { text: t("th.act.relu0") }) : null,
    h("span", {}, t("th.act.range"), " ", h("strong", { text: `[${fmtNum(lo, 2)}; ${fmtNum(hi, 2)}]` })),
  ].filter(Boolean));
}

// =====================================================================
// 4 · Pooling
// =====================================================================
export const pool = { mode: "max", data: null, pos: null };
export const POOL_CELL = 9;

export function setupPool() {
  segmented($("#t-pool-seg"), [["max", "Max pooling"], ["avg", "Average pooling"]], pool.mode, (mode) => {
    pool.mode = mode;
    updatePool();
  });
  const out = $("#t-pool-out");
  out.addEventListener("pointermove", (e) => {
    if (!pool.data) return;
    const [r, c] = cellFromEvent(e, out, pool.data.ow, pool.data.oh);
    setPoolPos(r * pool.data.ow + c);
  });
}

export function updatePool() {
  if (!T.image) return;
  const { z, rows, cols } = featureMap();
  const x = relu(z);   // SimpleCNN pools the ReLU output
  const { out, oh, ow } = pool2(x, rows, cols, pool.mode);
  const scale = maxOf(x);
  pool.data = { x, rows, cols, out, oh, ow };
  showPixels($("#t-pool-in"), toUnsigned(x, scale), cols, rows, "seq", POOL_CELL);
  showPixels($("#t-pool-out"), toUnsigned(out, scale), ow, oh, "seq", POOL_CELL);
  $("#t-pool-in-cap").textContent = t("th.pool.incap", { s: `${rows}×${cols}` });
  $("#t-pool-out-cap").textContent = t("th.pool.outcap", { m: pool.mode === "max" ? "Max" : "Average", s: `${oh}×${ow}` });
  if (pool.pos == null || pool.pos >= out.length) {
    let best = 0;
    for (let i = 1; i < out.length; i++) if (out[i] > out[best]) best = i;
    pool.pos = best;
  }
  setPoolPos(pool.pos);
}

export function setPoolPos(index) {
  const d = pool.data;
  pool.pos = index;
  const r = Math.floor(index / d.ow);
  const c = index % d.ow;
  placeBox($("#t-pool-win"), 2 * c, 2 * r, 2, 2, d.cols, d.rows);
  placeBox($("#t-pool-cell"), c, r, 1, 1, d.ow, d.oh);
  const values = [0, 1].flatMap((m) => [0, 1].map((n) => d.x[(2 * r + m) * d.cols + 2 * c + n]));
  const result = d.out[index];
  const scale = maxOf(d.x);
  $("#t-pool-calc").replaceChildren(
    h("div", { class: "calc-block" }, h("span", { class: "calc-cap", text: t("th.pool.region") }), matrix(values, 2, "seq", scale, false)),
    h("span", { class: "op", text: "→" }),
    h("div", { class: "calc-result" },
      h("span", { class: "calc-cap", text: t("th.outpos", { r: r + 1, c: c + 1 }) }),
      h("span", { text: `${pool.mode === "max" ? "max" : t("th.pool.mean")}(${values.map((v) => fmtNum(v, 2)).join("; ")})` }),
      h("strong", { text: `= ${fmtNum(result, 2)}` })),
  );
}

// =====================================================================
// 5 · Dropout
// =====================================================================
export const drop = { p: 0.45, mode: "train", maps: null, noise: null, shown: null };

export function setupDropout() {
  const slider = $("#t-drop-p");
  slider.addEventListener("input", () => { drop.p = Number(slider.value); updateDropout(); });
  segmented($("#t-drop-seg"), [["train", t("th.drop.train")], ["eval", t("th.drop.eval")]], drop.mode, (mode) => {
    drop.mode = mode;
    updateDropout();
  });
  $("#t-drop-new").addEventListener("click", () => { drop.noise = null; updateDropout(); });
}

/** MaxPool(ReLU(BatchNorm(Conv1(x)))) for all 32 filters: exactly what Dropout1 receives. */
export function pool1Maps() {
  return T.params.conv1.weight.map((w, c) => {
    const { out, oh, ow } = conv2d(T.image.x, SIZE, SIZE, w, 5, T.params.conv1.bias[c]);
    return pool2(relu(batchNormEval(out, c)), oh, ow, "max").out;
  });
}

export function updateDropout(newInput = false) {
  if (!T.params || !T.image) return;
  if (newInput || !drop.maps) {
    drop.maps = pool1Maps();
    drop.shown = drop.maps.map((m, c) => [meanOf(m), c]).sort((a, b) => b[0] - a[0]).slice(0, 8).map(([, c]) => c).sort((a, b) => a - b);
    drop.noise = null;
  }
  // One random number per value: raising p drops a growing superset of the same values
  if (!drop.noise) drop.noise = drop.maps.map((m) => Float32Array.from(m, () => Math.random()));
  const train = drop.mode === "train";
  const p = drop.p;
  const keepScale = 1 / (1 - p);
  const isDropped = (c, i) => train && drop.noise[c][i] < p;
  const after = drop.maps.map((m, c) => m.map((v, i) => (!train ? v : isDropped(c, i) ? 0 : v * keepScale)));

  let top = 0;
  for (const c of drop.shown) top = Math.max(top, maxOf(drop.maps[c]), maxOf(after[c]));
  const level = (v) => clamp(Math.round((v / top) * 253), 0, 253);
  const render = (host, source, marked) => {
    const canvases = ensureCanvases(host, drop.shown.length);
    drop.shown.forEach((c, k) => {
      const values = Array.from(source[c], (v, i) => (marked && isDropped(c, i) ? 255 : level(v)));
      paint(canvases[k], values, 12, 12, "drop");
      canvases[k].title = t("th.map", { n: c + 1 });
    });
  };
  render($("#t-drop-before"), drop.maps, false);
  render($("#t-drop-after"), after, true);

  let total = 0;
  let dropped = 0;
  let sumBefore = 0;
  let sumMasked = 0;
  let sumAfter = 0;
  drop.maps.forEach((m, c) => m.forEach((v, i) => {
    total++;
    sumBefore += v;
    sumAfter += after[c][i];
    if (isDropped(c, i)) dropped++;
    else sumMasked += v;
  }));
  $("#t-drop-p-v").textContent = fmtNum(p, 2);
  $("#t-drop-stats").replaceChildren(...[
    h("span", {}, t("th.drop.zeroed"), " ", h("strong", { text: `${fmtPct((dropped / total) * 100, 1)} (${fmtInt(dropped)}/${fmtInt(total)})` })),
    h("span", {}, t("th.drop.meanbefore"), " ", h("strong", { text: fmtNum(sumBefore / total, 3) })),
    train ? h("span", {}, t("th.drop.masked"), " ", h("strong", { text: fmtNum(sumMasked / total, 3) })) : null,
    h("span", {}, train ? t("th.drop.scaled", { s: fmtNum(keepScale, 2) }) : t("th.drop.inference"), " ", h("strong", { text: fmtNum(sumAfter / total, 3) })),
  ].filter(Boolean));
}

// =====================================================================
// 6 · Softmax and cross-entropy
// =====================================================================
export const sm = { logits: new Array(10).fill(0), base: null, label: 0 };

export function setupSoftmax() {
  $("#t-sm-logits").replaceChildren(...Array.from({ length: 10 }, (_, digit) => {
    const input = h("input", { type: "range", min: "-30", max: "30", step: "0.1", value: "0", "aria-label": t("th.sm.slider", { d: digit }) });
    input.addEventListener("input", () => { sm.logits[digit] = Number(input.value); updateSoftmax(); });
    return h("label", { class: "logit-row" }, h("span", { class: "logit-d", text: String(digit) }), input, h("output", { class: "logit-v" }));
  }));
  segmented($("#t-ce-label"), Array.from({ length: 10 }, (_, d) => [String(d), String(d)]), "0", (value) => {
    sm.label = Number(value);
    updateSoftmax();
  });
  $("#t-sm-reset").addEventListener("click", () => {
    if (sm.base) sm.logits = [...sm.base];
    syncLogitSliders();
    updateSoftmax();
  });
  $("#t-sm-zero").addEventListener("click", () => {
    sm.logits.fill(0);
    syncLogitSliders();
    updateSoftmax();
  });
}

export function resetSoftmax() {
  sm.base = T.image?.logits ? T.image.logits.map((v) => Math.round(v * 10) / 10) : null;
  sm.logits = sm.base ? [...sm.base] : [2, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  sm.label = T.image?.label ?? T.image?.prediction ?? 0;
  setPressed($("#t-ce-label"), sm.label);
  $("#t-sm-reset").disabled = !sm.base;
  $("#t-sm-sub").textContent = sm.base
    ? t("th.sm.subreal")
    : t("th.sm.subdemo");
  syncLogitSliders();
  updateSoftmax();
}

export function syncLogitSliders() {
  [...$("#t-sm-logits").children].forEach((row, d) => { row.querySelector("input").value = sm.logits[d]; });
}

export function updateSoftmax() {
  const top = Math.max(...sm.logits);
  const exps = sm.logits.map((z) => Math.exp(z - top));   // shifted for numerical stability
  const total = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / total);
  const best = probs.indexOf(Math.max(...probs));
  [...$("#t-sm-logits").children].forEach((row, d) => { row.querySelector("output").textContent = fmtNum(sm.logits[d], 1); });
  probabilityChart($("#t-sm-probs"), probs, best);

  const p = probs[sm.label];
  const loss = top + Math.log(total) - sm.logits[sm.label];   // -log p, exact even when p rounds to 0
  $("#t-ce-value").textContent = fmtNum(loss, 3);
  $("#t-ce-live").textContent = ` \\(= -\\log(${texNum(fmtNum(p, 4))}) = ${texNum(fmtNum(loss, 3))}\\)`;
  curvePlot($("#t-ce-plot"), {
    label: t("th.ce.plot"),
    x: [0.001, 1], y: [0, 7], fn: (q) => -Math.log(q), marker: clamp(p, 0.001, 1),
    xTicks: [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v: Math.max(v, 0.001), label: fmtPct(v * 100, 0) })),
    yTicks: [0, 2, 4, 6].map((v) => ({ v, label: String(v) })),
    tooltip: (q) => ({ title: t("th.ce.tip", { v: fmtPct(q * 100, 1) }), rows: [{ value: fmtNum(-Math.log(q), 3), label: t("th.loss"), color: "var(--series-1)" }] }),
  });
}

// =====================================================================
// 7 · Gradient descent: SGD vs Momentum vs Adam on a 3D loss surface
// The surface is drawn on a 2D canvas (shaded quads, back to front) with its contour map
// on the floor below; drag to turn it, click a point to start from there.
// =====================================================================
export const SURFACES = {
  valley: {
    name: { tr: "Dar vadi", en: "Narrow valley" },
    f: (a, b) => (a * a) / 20 + b * b,
    grad: (a, b) => [a / 10, 2 * b],
    start: [-8.5, 3.6], lr: 0.3,
    tex: String.raw`L = \theta_1^2/20 + \theta_2^2`,
  },
  saddle: {
    name: { tr: "Eyer noktası", en: "Saddle point" },
    f: (a, b) => (a * a) / 20 - (b * b) / 2 + b ** 4 / 40,
    grad: (a, b) => [a / 10, -b + b ** 3 / 10],
    start: [-8.5, 0.002], lr: 0.1,
    tex: String.raw`L = \theta_1^2/20 - \theta_2^2/2 + \theta_2^4/40`,
    saddle: [0, 0],
  },
  wells: {
    name: { tr: "İki çukur", en: "Two valleys" },
    f: (a, b) => { const u = a / 6; return 3 * (u * u - 1) ** 2 - 0.6 * u + 0.4 * b * b; },
    grad: (a, b) => { const u = a / 6; return [2 * u * (u * u - 1) - 0.1, 0.8 * b]; },
    start: [-9.5, 2.5], lr: 0.25,
    tex: String.raw`L = 3\bigl((\theta_1/6)^2 - 1\bigr)^2 - 0{,}1\,\theta_1 + 0{,}4\,\theta_2^2`,
    texEn: String.raw`L = 3\bigl((\theta_1/6)^2 - 1\bigr)^2 - 0.1\,\theta_1 + 0.4\,\theta_2^2`,
  },
};
export const OPT_RANGE = { x: [-10, 10], y: [-5, 5] };
export const OPTIMIZERS = [
  { key: "sgd", label: "SGD", color: "var(--series-1)" },
  { key: "momentum", label: "Momentum", color: "var(--series-2)" },
  { key: "adam", label: "Adam", color: "var(--series-3)" },
];
export const VIEWS = {
  "3d": { az: -0.55, el: 0.95, label: { tr: "3B", en: "3D" } },
  top: { az: 0, el: Math.PI / 2 - 1e-4, label: { tr: "Üstten", en: "From above" } },
  side: { az: 0, el: 0.06, label: { tr: "θ₁ boyunca kesit", en: "Along θ₁" } },
};
export const opt = {
  surface: "valley", lr: 0.3, start: [-8.5, 3.6], steps: 100, data: null, shown: 100, anim: 0,
  hidden: new Set(), view: { ...VIEWS["3d"] }, viewName: "3d", mesh: null, floor: null,
};

export const OPT_TEXT = {
  methods: [
    {
      key: "sgd",
      sub: { tr: "Stokastik gradyan inişi", en: "Stochastic gradient descent" },
      tex: String.raw`\theta_t = \theta_{t-1} - \eta\, g_t`,
      idea: {
        tr: "Yalnızca o anki eğime bakar: sisli bir dağda ayağının altındaki eğime göre adım atan bir yürüyüşçü gibi. Adım gradyanla orantılıdır: dik yerde büyük, düz yerde küçük.",
        en: "It only looks at the current slope, like a hiker in fog stepping by the slope under their feet. The step is proportional to the gradient: big on steep ground, small on flat ground.",
      },
      pros: [
        { tr: "Basittir, ek bellek gerektirmez; iyi ayarlandığında hâlâ çok güçlüdür.", en: "Simple, needs no extra memory; still very strong when tuned well." },
      ],
      cons: [
        { tr: "Dar bir vadide duvardan duvara zikzak çizer, vadinin tabanı boyunca sürünür.", en: "Zigzags from wall to wall in a narrow valley and crawls along its floor." },
        { tr: "Gradyanın sıfıra yaklaştığı eyer noktalarında uzun süre bekler.", en: "Waits a long time at saddle points, where the gradient is nearly zero." },
        { tr: "Uygun η'yi bulmak zordur: biraz büyük olunca ıraksar.", en: "The right η is hard to find: a little too large and it diverges." },
      ],
      look: {
        tr: "<b>Animasyonda:</b> dar vadide mavi yol önce dik θ₂ yönünde hızla iner, sonra düz θ₁ yönünde sürünür ve 100 adımda öbür ikisinin gerisinde kalır. η'yi 0,5'in üzerine çıkarın: θ₂ yönünde duvardan duvara salınmaya başlar; 1'in üzerinde salınım her adımda büyür ve yol ıraksar. Yani dik yön η'nin üst sınırını koyar, düz yön de o η ile ne kadar yavaş gidileceğini belirler.",
        en: "<b>In the animation:</b> in the narrow valley the blue path first drops quickly along the steep θ₂ direction, then crawls along the flat θ₁ direction and after 100 steps is behind the other two. Raise η above 0.5 and it starts bouncing from wall to wall along θ₂; above 1 the bounce grows at every step and the path diverges. So the steep direction caps η, and the flat direction then sets how slowly that η gets anywhere.",
      },
    },
    {
      key: "momentum",
      sub: { tr: "Ağır bir top", en: "A heavy ball" },
      tex: String.raw`\mathbf{v}_t = \mu\,\mathbf{v}_{t-1} + g_t \qquad \theta_t = \theta_{t-1} - \eta\,\mathbf{v}_t`,
      idea: {
        tr: String.raw`Yokuş aşağı yuvarlanan ağır bir top gibi hız biriktirir (burada \(\mu = 0{,}9\)). Hız, geçmiş gradyanların üstel ağırlıklı toplamıdır: hep aynı yönü gösteren bileşenler toplanıp büyür (en fazla \(1/(1-\mu) = 10\) kat), yön değiştirenler birbirini götürür.`,
        en: String.raw`Like a heavy ball rolling downhill it builds up velocity (here \(\mu = 0.9\)). The velocity is an exponentially weighted sum of past gradients: components that keep pointing the same way add up (up to \(1/(1-\mu) = 10\) times), those that flip cancel out.`,
      },
      pros: [
        { tr: "Vadinin tabanı gibi tutarlı yönlerde hızlanır, zikzakları söndürür.", en: "Speeds up in consistent directions such as a valley floor and damps zigzags." },
        { tr: "Biriktirdiği hızla sığ çukurlardan ve eyer noktalarından kurtulabilir.", en: "Its velocity can carry it out of shallow dips and away from saddle points." },
      ],
      cons: [
        { tr: "Hızı yüzünden minimumu geçer, birkaç kez salınarak durur.", en: "Its speed makes it overshoot the minimum and settle only after a few swings." },
        { tr: "η ile μ birlikte ayarlanmalıdır.", en: "η and μ have to be tuned together." },
      ],
      look: {
        tr: "<b>Animasyonda:</b> iki çukur yüzeyinde turuncu top tepeyi aşıp derin çukura ulaşır; SGD ve Adam ilk çukurda kalır.",
        en: "<b>In the animation:</b> on the two-valley surface the orange ball rolls over the hill into the deeper valley; SGD and Adam stay in the first one.",
      },
    },
    {
      key: "adam",
      sub: { tr: "Parametre başına uyarlanan adım", en: "A step adapted per parameter" },
      tex: String.raw`\theta_t = \theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}`,
      idea: {
        tr: String.raw`Gradyanın hareketli ortalamasını (\(\hat m\)), gradyan karelerinin hareketli ortalamasının köküne (\(\sqrt{\hat v}\)) böler. Böylece her parametrenin adımı kabaca η olur: gradyanı büyük olan yönde adım küçülür, küçük olan yönde büyür.`,
        en: String.raw`It divides the moving average of the gradient (\(\hat m\)) by the root of the moving average of its square (\(\sqrt{\hat v}\)). So every parameter's step is roughly η: smaller where the gradient is large, larger where it is small.`,
      },
      pros: [
        { tr: "Dar vadide iki yönde de aynı hızla ilerler; ölçekleri çok farklı parametreler için tek bir η yeter.", en: "Moves equally fast in both directions of a narrow valley; one η suits parameters of very different scales." },
        { tr: "Gradyanın çok küçük olduğu düz bölgelerde ve eyer noktalarında takılmaz.", en: "Does not get stuck on flat ground or at saddle points, where the gradient is tiny." },
        { tr: "Derin ağlarda en yaygın seçimdir; SimpleCNN de onu kullanır.", en: "The most common choice for deep networks; SimpleCNN uses it too." },
      ],
      cons: [
        { tr: "Adımı hep yaklaşık η olduğu için minimumun yakınında küçük salınımlar yapar.", en: "Its step stays near η, so it jitters a little close to the minimum." },
        { tr: "Momentum kadar hız biriktirmediğinden sığ bir çukurdan kaçamayabilir.", en: "It builds up less speed than Momentum, so it may not escape a shallow dip." },
      ],
      look: {
        tr: "<b>Animasyonda:</b> dar vadide yeşil yol neredeyse düz bir çizgidir; eyer noktasında ilk kaçan odur.",
        en: "<b>In the animation:</b> in the narrow valley the green path is almost a straight line; at the saddle point it is the first to escape.",
      },
    },
  ],
  notes: {
    valley: {
      tr: "θ₂ yönünde θ₁'e göre 20 kat daha dik, uzun ve dar bir vadi. Derin ağların kayıp yüzeyleri de böyle \"kötü koşulludur\": bazı yönlerde çok dik, bazılarında neredeyse düz. SGD'de tek bir η iki yöne birden uymaz: dik yönde salınmaması için η küçük olmalı, o zaman da düz yönde sürünür. Momentum vadinin tabanında hız kazanır; Adam iki yönü dengeler.",
      en: "A long, narrow valley, 20 times steeper along θ₂ than along θ₁. The loss surfaces of deep networks are \"ill-conditioned\" like this too: very steep in some directions, almost flat in others. For SGD no single η suits both directions: η has to be small so it does not oscillate in the steep direction, and then it crawls in the flat one. Momentum picks up speed along the valley floor; Adam balances the two directions.",
    },
    saddle: {
      tr: "Ortada bir eyer noktası var: θ₁ yönünde çukur, θ₂ yönünde tepe. Gerçek minimumlar θ₂ = ±3,16'dadır. Çok boyutlu ağlarda eyer noktaları yerel minimumlardan çok daha yaygındır. Sırttan başlayan SGD, gradyan neredeyse sıfır olduğu için eyerde uzun süre bekler; Adam küçük gradyanı büyüterek ilk kaçan olur.",
      en: "There is a saddle point in the middle: a dip along θ₁, a hill along θ₂. The real minima are at θ₂ = ±3.16. In high-dimensional networks saddle points are far more common than local minima. Starting on the ridge, SGD waits at the saddle for a long time because the gradient is almost zero; Adam scales the tiny gradient up and escapes first.",
    },
    wells: {
      tr: "Solda sığ bir yerel minimum, sağda daha derin bir global minimum, arada bir tepe. Gradyan yalnızca yakın çevreyi görür: SGD ilk bulduğu çukurda kalır, Momentum biriktirdiği hızla tepeyi aşar. η'yi değiştirip sonucun nasıl değiştiğine bakın; gerçek eğitimde de başlangıç noktası ve öğrenme oranı hangi çukura düşüleceğini etkiler.",
      en: "A shallow local minimum on the left, a deeper global minimum on the right, a hill in between. The gradient only sees the neighborhood: SGD stays in the first valley it finds, Momentum's speed carries it over the hill. Change η and see how the outcome changes; in real training too, the starting point and the learning rate decide which valley you end up in.",
    },
  },
};

export const surf = () => SURFACES[opt.surface];

/** The surface's lowest value (found once by a grid search and gradient descent), and its minima. */
export function surfaceInfo(s) {
  if (s.info) return s.info;
  let best = Infinity;
  const seeds = [];
  for (let i = 0; i <= 200; i++) {
    for (let j = 0; j <= 100; j++) {
      const a = OPT_RANGE.x[0] + (i / 200) * 20;
      const b = OPT_RANGE.y[0] + (j / 100) * 10;
      const v = s.f(a, b);
      best = Math.min(best, v);
      const g = s.grad(a, b);
      if (Math.hypot(...g) < 0.08 && v < s.f(0, 0) + 3) seeds.push([a, b]);
    }
  }
  const minima = [];
  for (const seed of seeds) {
    let p = seed;
    for (let k = 0; k < 3000; k++) { const g = s.grad(...p); p = [p[0] - 0.05 * g[0], p[1] - 0.05 * g[1]]; }
    const isMin = Math.hypot(...s.grad(...p)) < 1e-6 && s.f(p[0] + 0.01, p[1]) > s.f(...p) && s.f(p[0], p[1] + 0.01) > s.f(...p);
    if (isMin && !minima.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 0.1)) minima.push(p);
  }
  let fmax = -Infinity;
  for (let i = 0; i <= 40; i++) for (let j = 0; j <= 20; j++) fmax = Math.max(fmax, s.f(OPT_RANGE.x[0] + i / 2, OPT_RANGE.y[0] + j / 2));
  const fmin = Math.min(best, ...minima.map((p) => s.f(...p)));
  s.info = { fmin, fmax, minima, global: minima.reduce((g, p) => (s.f(...p) < s.f(...g) ? p : g), minima[0]) };
  return s.info;
}

export function runOptimizers() {
  const s = surf();
  const lr = opt.lr;
  return OPTIMIZERS.map((o) => {
    let p = [...opt.start];
    let v = [0, 0];
    let m = [0, 0];
    let q = [0, 0];
    let diverged = false;
    const path = [[...p]];
    const grads = [s.grad(...p)];
    const losses = [s.f(...p)];
    for (let t = 1; t <= opt.steps; t++) {
      if (!diverged) {
        const g = s.grad(...p);
        let next;
        if (o.key === "sgd") {
          next = [p[0] - lr * g[0], p[1] - lr * g[1]];
        } else if (o.key === "momentum") {
          v = [0.9 * v[0] + g[0], 0.9 * v[1] + g[1]];
          next = [p[0] - lr * v[0], p[1] - lr * v[1]];
        } else {
          m = [0.9 * m[0] + 0.1 * g[0], 0.9 * m[1] + 0.1 * g[1]];
          q = [0.999 * q[0] + 0.001 * g[0] ** 2, 0.999 * q[1] + 0.001 * g[1] ** 2];
          const mHat = m.map((x) => x / (1 - 0.9 ** t));
          const qHat = q.map((x) => x / (1 - 0.999 ** t));
          next = [p[0] - (lr * mHat[0]) / (Math.sqrt(qHat[0]) + 1e-8), p[1] - (lr * mHat[1]) / (Math.sqrt(qHat[1]) + 1e-8)];
        }
        if (next.every((x) => Number.isFinite(x) && Math.abs(x) < 1e3)) p = next;
        else diverged = true;
      }
      path.push([...p]);
      grads.push(s.grad(...p));
      losses.push(s.f(...p));
    }
    return { ...o, path, grads, losses, diverged };
  });
}

// ---------------------------------------------------------------------
// Texts and controls
// ---------------------------------------------------------------------
export function fillOptText() {
  const methods = $("#t-opt-methods");
  methods.replaceChildren(...OPT_TEXT.methods.map((mth) => {
    const o = OPTIMIZERS.find((x) => x.key === mth.key);
    const card = h("div", { class: "opt-method", style: `--c:${o.color}` });
    card.innerHTML = termify(`
      <div class="opt-method-head"><span class="opt-method-key"></span><strong>${o.label}</strong><span class="opt-method-sub">${L(mth.sub)}</span>
        <a href="#" class="opt-method-math" data-go="math:m-opt">${L({ tr: "∑ Matematiği", en: "∑ The math" })}</a></div>
      <div class="formula small">\\(\\displaystyle ${mth.tex}\\)</div>
      <p>${L(mth.idea)}</p>
      <ul class="opt-pc">${mth.pros.map((x) => `<li class="pro">${L(x)}</li>`).join("")}${mth.cons.map((x) => `<li class="con">${L(x)}</li>`).join("")}</ul>
      <p class="opt-look">${L(mth.look)}</p>`);   // static, author-written markup
    return card;
  }));

  const surfaces = $("#t-opt-surfaces");
  surfaces.replaceChildren(h("span", { class: "opt-ctl-label", text: L({ tr: "Kayıp yüzeyi", en: "Loss surface" }) }),
    seg3(Object.entries(SURFACES).map(([k, s]) => [k, L(s.name)]), opt.surface, (k) => selectSurface(k)));
  const views = $("#t-opt-views");
  views.replaceChildren(seg3(Object.entries(VIEWS).map(([k, v]) => [k, L(v.label)]), opt.viewName, (k) => {
    opt.viewName = k;
    opt.view = { ...VIEWS[k] };
    fillOptText();
    drawSurface3d();
  }));
  const s = surf();
  $("#t-opt-note").innerHTML = termify(`<span class="opt-note-formula">\\(${I18N.lang === "en" && s.texEn ? s.texEn : s.tex}\\)</span> ${L(OPT_TEXT.notes[opt.surface])}`);
  $("#t-opt-hint").textContent = L({ tr: "Sürükleyerek döndürün · bir noktaya tıklayıp oradan başlatın", en: "Drag to turn · click a point to start there" });
  $("#t-opt-canvas").setAttribute("aria-label", L({ tr: "Kayıp yüzeyinin 3B görünümü ve üç yöntemin yolu; ok tuşlarıyla döndürülür", en: "3D view of the loss surface and the three methods' paths; arrow keys turn it" }));
  $("#t-opt-step-label").textContent = L({ tr: "Adım", en: "Step" });
  $("#t-opt-loss-title").innerHTML = L({ tr: "Adım başına kayıp farkı \\(L - L_{\\mathrm{min}}\\) (logaritmik eksen)", en: "Loss gap per step, \\(L - L_{\\mathrm{min}}\\) (logarithmic axis)" });
}

/** A segmented control (aria-pressed buttons). */
export function seg3(options, current, onPick) {
  const box = h("div", { class: "seg" });
  for (const [value, label] of options) {
    const b = h("button", { type: "button", "aria-pressed": String(value === current), text: label });
    b.addEventListener("click", () => onPick(value));
    box.append(b);
  }
  return box;
}

export function selectSurface(key) {
  opt.surface = key;
  const s = surf();
  opt.start = [...s.start];
  opt.lr = s.lr;
  $("#t-opt-lr").value = Math.log10(opt.lr);
  stopOptAnim();
  paintSurface();
  fillOptText();
  recomputeOpt();
}

export function setupOptimization() {
  const slider = $("#t-opt-lr");
  slider.value = Math.log10(opt.lr);
  slider.addEventListener("input", () => {
    opt.lr = 10 ** Number(slider.value);
    stopOptAnim();
    recomputeOpt();
  });
  const step = $("#t-opt-step");
  step.max = String(opt.steps);
  step.addEventListener("input", () => {
    stopOptAnim();
    opt.shown = Number(step.value);
    drawOpt();
  });
  $("#t-opt-play").addEventListener("click", () => (opt.anim ? stopOptAnim() : playOpt()));
  setupSurfaceInput();
  watchSize($("#t-opt-stage"), () => drawSurface3d());
  fillOptText();
  paintSurface();
  recomputeOpt();
}

// ---------------------------------------------------------------------
// 3D drawing
// ---------------------------------------------------------------------
export const OPT_Z = { floor: -0.2, top: 0.62 };
/** Parameter space → world: θ₁ ∈ [-10, 10] → x ∈ [-1, 1], θ₂ ∈ [-5, 5] → y ∈ [-0.5, 0.5], loss → z. */
export const toWorld = (a, b, f) => [a / 10, b / 10, heightOf(f)];
export function heightOf(f) {
  const { fmin, fmax } = surfaceInfo(surf());
  return (Math.min(f, fmax) - fmin) / (fmax - fmin) * OPT_Z.top;
}

/** Orthographic camera: turn by the azimuth, tilt by the elevation, fit the box into the canvas. */
export function camera(width, height) {
  const { az, el } = opt.view;
  const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
  const raw = (x, y, z) => {
    const xr = x * ca - y * sa;
    const yr = x * sa + y * ca;
    return [xr, -(z * ce + yr * se), yr * ce - z * se];
  };
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const x of [-1, 1]) for (const y of [-0.5, 0.5]) for (const z of [OPT_Z.floor, OPT_Z.top]) {
    const [px, py] = raw(x, y, z);
    x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  const pad = 26;
  const scale = Math.min((width - 2 * pad) / Math.max(x1 - x0, 2.1), (height - 2 * pad) / Math.max(y1 - y0, 0.3));
  const ox = width / 2 - ((x0 + x1) / 2) * scale;
  const oy = height / 2 - ((y0 + y1) / 2) * scale;
  return (x, y, z) => {
    const [px, py, depth] = raw(x, y, z);
    return [ox + px * scale, oy + py * scale, depth];
  };
}

/** Rebuilds what depends on the surface and the theme: the mesh heights and the floor contour image. */
export function paintSurface() {
  const s = surf();
  const info = surfaceInfo(s);
  const NX = 46, NY = 24;
  const verts = [];
  for (let j = 0; j <= NY; j++) {
    for (let i = 0; i <= NX; i++) {
      const a = OPT_RANGE.x[0] + (i / NX) * 20;
      const b = OPT_RANGE.y[0] + (j / NY) * 10;
      const f = s.f(a, b);
      verts.push({ a, b, f, w: toWorld(a, b, f) });
    }
  }
  opt.mesh = { NX, NY, verts };
  // Floor: equal-loss bands seen from above (same look as the old flat view)
  const cols = 240, rows = 120;
  const values = new Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = OPT_RANGE.x[0] + ((c + 0.5) / cols) * 20;
      const b = OPT_RANGE.y[1] - ((r + 0.5) / rows) * 10;
      const level = Math.sqrt(Math.max(0, Math.min(1, (s.f(a, b) - info.fmin) / (info.fmax - info.fmin))));
      values[r * cols + c] = Math.round((Math.min(Math.floor(level * 14), 13) / 13) * 255);
    }
  }
  opt.floor ??= document.createElement("canvas");
  paint(opt.floor, values, cols, rows, "ramp");
  opt.floor._paint = null;   // not a page canvas: theme changes repaint it through paintSurface()
}


export function drawSurface3d() {
  const canvas = $("#t-opt-canvas");
  const stage = $("#t-opt-stage");
  if (!opt.mesh || !opt.data || !stage.clientWidth) return;
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const P = camera(W, H);
  opt.project = P;
  const s = surf();
  const info = surfaceInfo(s);
  const z0 = OPT_Z.floor;
  const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const text = cssVar("--text-2");

  // Floor with the contour image, mapped by an affine transform (orthographic view of a plane)
  const [ax, ay] = P(-1, 0.5, z0), [bx, by] = P(1, 0.5, z0), [cx, cy] = P(-1, -0.5, z0), [dx, dy] = P(1, -0.5, z0);
  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.setTransform(dpr * (bx - ax) / opt.floor.width, dpr * (by - ay) / opt.floor.width, dpr * (cx - ax) / opt.floor.height, dpr * (cy - ay) / opt.floor.height, dpr * ax, dpr * ay);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(opt.floor, 0, 0);
  ctx.restore();
  ctx.strokeStyle = cssVar("--axis");
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(dx, dy); ctx.lineTo(cx, cy); ctx.closePath(); ctx.stroke();

  // Shadows of the paths on the floor
  const shown = opt.data.filter((o) => !opt.hidden.has(o.key));
  const colors = Object.fromEntries(OPTIMIZERS.map((o) => [o.key, cssVar(o.color.slice(4, -1))]));
  for (const o of shown) {
    ctx.strokeStyle = colors[o.key];
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    o.path.slice(0, opt.shown + 1).forEach(([a, b], k) => {
      const [px, py] = P(clamp(a, -10, 10) / 10, clamp(b, -5, 5) / 10, z0);
      if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
  // Minima on the floor
  for (const m of info.minima) {
    const [px, py] = P(m[0] / 10, m[1] / 10, z0);
    ctx.strokeStyle = cssVar("--text");
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px - 5, py); ctx.lineTo(px + 5, py); ctx.moveTo(px, py - 5); ctx.lineTo(px, py + 5); ctx.stroke();
  }

  // Paths, balls and start point live on the surface. Everything is drawn back to front in one
  // list (painter's algorithm), so a valley wall in front hides the part of a path behind it.
  const lift = 0.012;
  const onSurface = (a, b) => P(clamp(a, -10, 10) / 10, clamp(b, -5, 5) / 10, heightOf(s.f(clamp(a, -10, 10), clamp(b, -5, 5))) + lift);
  // A step is a straight line in parameter space; on the curved surface it is split into short
  // pieces, so the path hugs the surface instead of cutting through the air
  const along = (path) => {
    const pts = [onSurface(...path[0])];
    for (let k = 1; k < path.length; k++) {
      const [a0, b0] = path[k - 1];
      const [a1, b1] = path[k];
      const pieces = Math.min(24, Math.max(1, Math.ceil(Math.hypot(a1 - a0, b1 - b0) / 0.35)));
      for (let q = 1; q <= pieces; q++) pts.push(onSurface(a0 + ((a1 - a0) * q) / pieces, b0 + ((b1 - b0) * q) / pieces));
    }
    return pts;
  };
  const items = [];

  // Drop lines from each ball to the floor: under the surface, so drawn before it
  for (const o of shown) {
    const [a, b] = o.path[opt.shown];
    const [px, py] = onSurface(a, b);
    const [fx, fy] = P(clamp(a, -10, 10) / 10, clamp(b, -5, 5) / 10, z0);
    ctx.strokeStyle = colors[o.key];
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(fx, fy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // The surface: quads colored by height and lit from the upper left
  const { NX, NY, verts } = opt.mesh;
  const proj = verts.map((v) => P(...v.w));
  const lut = LUTS.seq;
  const light = [-0.45, 0.35, 0.82];
  const ln = Math.hypot(...light);
  const dxw = 2 / NX, dyw = 1 / NY;
  for (let j = 0; j < NY; j++) {
    for (let i = 0; i < NX; i++) {
      const k = j * (NX + 1) + i;
      const ids = [k, k + 1, k + NX + 2, k + NX + 1];
      const depth = (proj[ids[0]][2] + proj[ids[1]][2] + proj[ids[2]][2] + proj[ids[3]][2]) / 4;
      const z = ids.map((q) => verts[q].w[2]);
      const dzdx = ((z[1] - z[0]) + (z[2] - z[3])) / 2;
      const dzdy = ((z[3] - z[0]) + (z[2] - z[1])) / 2;
      const n = [-dzdx * dyw, -dzdy * dxw, dxw * dyw];
      const lambert = Math.max(0, (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / (Math.hypot(...n) * ln));
      const shade = 0.55 + 0.5 * lambert;
      const c = Math.round(clamp(0.18 + ((z[0] + z[1] + z[2] + z[3]) / 4 / OPT_Z.top) * 0.78, 0, 1) * 255) * 3;
      const r = Math.min(255, lut[c] * shade), g = Math.min(255, lut[c + 1] * shade), b = Math.min(255, lut[c + 2] * shade);
      items.push({ depth, draw: () => {
        ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
        ctx.strokeStyle = `rgba(${(r * 0.72) | 0},${(g * 0.72) | 0},${(b * 0.72) | 0},0.9)`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ids.forEach((id, q) => (q ? ctx.lineTo(proj[id][0], proj[id][1]) : ctx.moveTo(proj[id][0], proj[id][1])));
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } });
    }
  }

  // Path pieces, a little in front of the surface under them, with a halo so they stand out
  const halo = dark ? "rgba(0,0,0,0.55)" : "rgba(255,255,255,0.85)";
  for (const o of shown) {
    const pts = along(o.path.slice(0, opt.shown + 1));
    for (let k = 1; k < pts.length; k++) {
      const p0 = pts[k - 1], p1 = pts[k];
      // The halo goes a little further back than the colored line, so neighbouring pieces'
      // halos are all drawn before their colors and the line stays unbroken
      const d = (p0[2] + p1[2]) / 2;
      for (const [width, color, bias] of [[4.5, halo, 0.03], [2.2, colors[o.key], 0.08]]) {
        items.push({ depth: d - bias, draw: () => {
          ctx.lineCap = "round";
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
          ctx.lineCap = "butt";
        } });
      }
    }
  }
  // Start point and the balls at the current step
  const [sx, sy, sd] = onSurface(...opt.start);
  items.push({ depth: sd - 0.1, draw: () => {
    ctx.fillStyle = cssVar("--surface");
    ctx.strokeStyle = cssVar("--text");
    ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  } });
  for (const o of shown) {
    const [px, py, pd] = onSurface(...o.path[opt.shown]);
    items.push({ depth: pd - 0.12, draw: () => {
      const grad = ctx.createRadialGradient(px - 2, py - 2, 1, px, py, 7);
      grad.addColorStop(0, "#ffffff");
      grad.addColorStop(0.35, colors[o.key]);
      grad.addColorStop(1, colors[o.key]);
      ctx.fillStyle = grad;
      ctx.strokeStyle = dark ? "#000" : "#fff";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, 6.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    } });
  }
  items.sort((p, q) => q.depth - p.depth);
  for (const item of items) item.draw();

  // Axis labels on the floor edges and the loss axis at the back corner
  ctx.fillStyle = text;
  ctx.font = "italic 13px Georgia, 'Times New Roman', serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const label = (x, y, z, str, dx = 0, dy = 0) => { const [px, py] = P(x, y, z); ctx.fillText(str, px + dx, py + dy); };
  const front = P(0, -0.5, z0)[2] < P(0, 0.5, z0)[2] ? -0.5 : 0.5;   // the floor edge nearer the viewer
  const side = P(-1, 0, z0)[2] < P(1, 0, z0)[2] ? -1 : 1;
  if (opt.viewName !== "side") label(0, front * 1.16, z0, "θ₁", 0, 6);
  label(side * 1.08, 0, z0, "θ₂", side * 6, 0);
  const back = [-side, -front];
  const [lx0, ly0] = P(back[0], back[1], z0);
  const [lx1, ly1] = P(back[0], back[1], OPT_Z.top);
  if (opt.viewName !== "top") {
    ctx.strokeStyle = cssVar("--axis");
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(lx0, ly0); ctx.lineTo(lx1, ly1); ctx.stroke();
    ctx.fillText("L", lx1, ly1 - 9);
  }
}

// ---------------------------------------------------------------------
// Turning the surface and picking a start point
// ---------------------------------------------------------------------
export function setupSurfaceInput() {
  const canvas = $("#t-opt-canvas");
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, az: opt.view.az, el: opt.view.el, moved: false };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (!drag.moved) return;
    opt.view.az = drag.az + dx * 0.008;
    opt.view.el = clamp(drag.el + dy * 0.006, 0.05, Math.PI / 2 - 1e-4);
    markCustomView();
    drawSurface3d();
  });
  canvas.addEventListener("pointerup", (e) => {
    if (drag && !drag.moved) pickStart(e);
    drag = null;
  });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  canvas.addEventListener("keydown", (e) => {
    const turn = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[e.key];
    if (!turn) return;
    e.preventDefault();
    opt.view.az += turn[0];
    opt.view.el = clamp(opt.view.el + turn[1], 0.05, Math.PI / 2 - 1e-4);
    markCustomView();
    drawSurface3d();
  });
}

export function markCustomView() {
  if (opt.viewName === null) return;
  opt.viewName = null;
  for (const b of $("#t-opt-views").querySelectorAll("button")) b.setAttribute("aria-pressed", "false");
}

/** The surface point under the cursor: among nearby projected samples, the one nearest the viewer. */
export function pickStart(e) {
  const P = opt.project;
  if (!P) return;
  const box = $("#t-opt-canvas").getBoundingClientRect();
  const mx = e.clientX - box.left;
  const my = e.clientY - box.top;
  const s = surf();
  let best = null;
  for (let i = 0; i <= 100; i++) {
    for (let j = 0; j <= 50; j++) {
      const a = -10 + i / 5;
      const b = -5 + j / 5;
      const [px, py, depth] = P(a / 10, b / 10, heightOf(s.f(a, b)));
      const d = Math.hypot(px - mx, py - my);
      if (d < 9 && (!best || depth < best.depth - 0.02 || (Math.abs(depth - best.depth) <= 0.02 && d < best.d))) best = { a, b, depth, d };
    }
  }
  if (!best) return;
  opt.start = [best.a, best.b];
  stopOptAnim();
  recomputeOpt();
  playOpt();
}

// ---------------------------------------------------------------------
// Everything else around the picture
// ---------------------------------------------------------------------
export function recomputeOpt() {
  opt.data = runOptimizers();
  opt.shown = opt.steps;
  $("#t-opt-lr-v").textContent = `η = ${fmtNum(opt.lr, opt.lr < 0.1 ? 3 : 2)}`;
  drawOpt();
}

export const toSup = (text) => text.replace(/[-0-9]/g, (ch) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(ch)]);
export function fmtLoss(v) {
  if (!(v > 1e-12)) return "≈ 0";
  if (v >= 100) return fmtNum(v, 0);
  if (v >= 0.001) return fmtNum(v, 4);
  const e = Math.floor(Math.log10(v));
  return `${fmtNum(v / 10 ** e, 1)}×10${toSup(String(e))}`;
}
export const vec = (p, d = 2) => `(${fmtNum(p[0], d)}; ${fmtNum(p[1], d)})`;

export function drawOpt() {
  const n = opt.shown;
  const info = surfaceInfo(surf());
  $("#t-opt-step").value = String(n);
  $("#t-opt-step-v").textContent = `${n} / ${opt.steps}`;
  drawSurface3d();

  // Legend: each method can be hidden; the text tells where it ended up
  const where = (o) => {
    const p = o.path[n];
    const minimum = info.minima.find((m) => Math.hypot(m[0] - p[0], m[1] - p[1]) < 0.6);
    if (o.diverged && n === opt.steps) return L({ tr: "ıraksadı", en: "diverged" });
    if (!minimum) return "";
    const isGlobal = minimum === info.global;
    return info.minima.length > 1 && !isGlobal ? L({ tr: "yerel minimumda", en: "in the local minimum" }) : L({ tr: "minimumda", en: "at the minimum" });
  };
  $("#t-opt-legend").replaceChildren(...opt.data.map((o) => {
    const b = h("button", { type: "button", class: "legend-item opt-toggle", "aria-pressed": String(!opt.hidden.has(o.key)) },
      h("span", { class: "key-line", style: `background:${o.color}` }),
      `${o.label} · ${t("th.opt.after", { n })} `,
      h("strong", { text: fmtLoss(o.losses[n] - info.fmin) }),
      where(o) ? ` · ${where(o)}` : "");
    b.title = L({ tr: "Göster / gizle", en: "Show / hide" });
    b.addEventListener("click", () => {
      if (opt.hidden.has(o.key)) opt.hidden.delete(o.key); else opt.hidden.add(o.key);
      drawOpt();
    });
    return b;
  }));

  // What each method sees and does at this step
  const head = [L({ tr: "Yöntem", en: "Method" }), "θ", "∇L(θ)", L({ tr: "Sonraki adım Δθ", en: "Next step Δθ" }), "|Δθ|", "L"];
  const rows = opt.data.filter((o) => !opt.hidden.has(o.key)).map((o) => {
    const p = o.path[n];
    const next = o.path[Math.min(n + 1, opt.steps)];
    const d = [next[0] - p[0], next[1] - p[1]];
    return h("tr", {},
      h("td", {}, h("span", { class: "key-line", style: `background:${o.color}` }), ` ${o.label}`),
      h("td", { text: vec(p) }), h("td", { text: vec(o.grads[n], 3) }),
      h("td", { text: n < opt.steps ? vec(d, 3) : "—" }), h("td", { text: n < opt.steps ? fmtNum(Math.hypot(...d), 3) : "—" }),
      h("td", { text: fmtNum(o.losses[n], 3) }));
  });
  $("#t-opt-table").replaceChildren(h("thead", {}, h("tr", {}, ...head.map((c) => h("th", { text: c })))), h("tbody", {}, ...rows));

  // Loss gap per step on a log axis (x = step + 1 so the shared chart helper can index it)
  const series = opt.data.filter((o) => !opt.hidden.has(o.key))
    .map((o) => ({ name: o.label, color: o.color, points: o.losses.slice(0, n + 1).map((v, i) => [i + 1, Math.max(v - info.fmin, 1e-6)]) }));
  const all = series.flatMap((s) => s.points.map((p) => p[1]));
  if (!all.length) all.push(1);
  const yMin = 10 ** Math.floor(Math.log10(Math.max(Math.min(...all), 1e-6)));
  const yMax = 10 ** Math.ceil(Math.log10(Math.max(...all)));
  const yTicks = [];
  const eMin = Math.round(Math.log10(yMin));
  const eMax = Math.round(Math.log10(yMax));
  const every = eMax - eMin > 5 ? 2 : 1;   // keep the axis readable over many decades
  for (let e = eMax; e >= eMin; e -= every) {
    yTicks.push({ v: 10 ** e, label: e >= -2 ? fmtTick(10 ** e) : `10${toSup(String(e))}` });
  }
  const xTicks = [];
  for (let st = 0; st <= opt.steps; st += 20) xTicks.push({ v: st + 1, label: String(st) });
  lineChart($("#t-opt-loss"), {
    label: t("th.opt.label"),
    empty: "",
    x: { min: 1, max: opt.steps + 1, ticks: xTicks },
    y: { log: true, min: yMin, max: yMax, ticks: yTicks },
    series,
    tooltip: (i) => ({ title: t("th.opt.step", { i }), rows: opt.data.map((o) => ({ value: fmtLoss(o.losses[i] - info.fmin), label: o.label, color: o.color })) }),
  });
}

export function playOpt() {
  stopOptAnim();
  opt.shown = 0;
  $("#t-opt-play").textContent = t("th.stop");
  let last = 0;
  const frame = (time) => {
    if (time - last > 70) {
      last = time;
      opt.shown++;
      drawOpt();
    }
    if (opt.shown < opt.steps) opt.anim = requestAnimationFrame(frame);
    else stopOptAnim();
  };
  opt.anim = requestAnimationFrame(frame);
}

export function stopOptAnim() {
  if (opt.anim) cancelAnimationFrame(opt.anim);
  opt.anim = 0;
  $("#t-opt-play").textContent = t("th.play");
}

// =====================================================================
// 0 · Architecture schematic
// Stacks are drawn to scale (height = spatial size, depth = channels); the front face of each
// stack shows the selected image's real output at that layer.
// =====================================================================
export const SCHEMATIC = {
  cy: 112,    // vertical center of the forward path
  cell: 4,    // drawn pixels per feature-map pixel
  stages: [
    { key: "input", name: { tr: "Giriş", en: "Input" }, shape: "1×28×28", size: 28, depth: 1, offset: 0, goto: "t-conv" },
    { key: "conv1", name: "Conv1", shape: "32×24×24", size: 24, depth: 6, offset: 6, goto: "t-conv" },
    { key: "pool1", name: "MaxPool1", shape: "32×12×12", size: 12, depth: 6, offset: 5, goto: "t-pool" },
    { key: "conv2", name: "Conv2", shape: "64×8×8", size: 8, depth: 9, offset: 4, goto: "t-conv" },
    { key: "pool2", name: "MaxPool2", shape: "64×4×4", size: 4, depth: 9, offset: 3, goto: "t-pool" },
    { key: "flat", name: { tr: "Vektör", en: "Vector" }, shape: "1024", column: [10, 140], goto: "t-softmax" },
    { key: "fc1", name: "FC1", shape: "512", column: [10, 80], goto: "t-softmax" },
    { key: "out", name: { tr: "Çıkış", en: "Output" }, shape: { tr: "10 sınıf", en: "10 classes" }, goto: "t-softmax" },
  ],
  // The operations on each arrow, top to bottom, with the section that explains them
  ops: [
    { width: 92, items: [["Conv 5×5", "t-conv"], ["BatchNorm", "t-bn"], ["ReLU", "t-act"]] },
    { width: 92, items: [["MaxPool 2×2", "t-pool"], ["Dropout", "t-drop", { tr: "p = 0,45 · yalnızca eğitimde", en: "p = 0.45 · training only" }]] },
    { width: 92, items: [["Conv 5×5", "t-conv"], ["BatchNorm", "t-bn"], ["ReLU", "t-act"]] },
    { width: 92, items: [["MaxPool 2×2", "t-pool"], ["Dropout", "t-drop", { tr: "p = 0,35 · yalnızca eğitimde", en: "p = 0.35 · training only" }]] },
    { width: 66, items: [[{ tr: "Düzleştir", en: "Flatten" }, "t-softmax"]] },
    { width: 66, items: [["FC", "t-softmax"], ["ReLU", "t-act"]] },
    { width: 72, items: [["FC", "t-softmax"], ["Softmax", "t-softmax"]] },
  ],
};

/** Paints values into an off-screen canvas and returns it as an image URL for SVG. */
export function pixelsURL(face) {
  const canvas = document.createElement("canvas");
  paint(canvas, face.values, face.w, face.h, face.lut);
  return canvas.toDataURL();
}

/** Jumps to a section of this tab as a history step (Back returns to the diagram). */
export const jumpToTheory = (id) => navigate("theory", id);

export function makeLink(node, target, label) {
  node.setAttribute("tabindex", "0");
  node.setAttribute("role", "link");
  node.setAttribute("aria-label", label);
  node.addEventListener("click", () => jumpToTheory(target));
  node.addEventListener("keydown", (e) => { if (e.key === "Enter") jumpToTheory(target); });
}

/** The real output of every stage for the selected image (what the front faces show). */
export function schematicFaces() {
  const faces = { input: { values: T.image.pixels, w: 28, h: 28, lut: "gray", tip: t("th.sch.selected") } };
  const { z, rows, cols } = featureMap();
  const a = relu(z);
  const scale = maxOf(a);
  const filterName = T.params ? t("th.sch.filter", { n: T.filter + 1 }) : t("th.sch.sobel");
  faces.conv1 = { values: toUnsigned(a, scale), w: cols, h: rows, lut: "seq", tip: `Conv1 · ${filterName}` };
  const pooled = pool2(a, rows, cols, "max");
  faces.pool1 = { values: toUnsigned(pooled.out, scale), w: pooled.ow, h: pooled.oh, lut: "seq", tip: `MaxPool1 · ${filterName}` };

  const result = T.image.result;
  if (result) {
    const conv2 = result.layers.find((l) => l.key === "conv2");
    const p2 = result.layers.find((l) => l.key === "pool2");
    const k = conv2.peaks.indexOf(Math.max(...conv2.peaks));   // the map that responds most
    faces.conv2 = { values: conv2.maps[k], w: 8, h: 8, lut: "seq", tip: `Conv2 · ${t("th.sch.strongest", { n: k + 1 })}` };
    faces.pool2 = { values: p2.maps[k], w: 4, h: 4, lut: "seq", tip: `MaxPool2 · ${t("th.sch.map", { n: k + 1 })}` };
    // The 1024-vector: undo the per-map display scaling with each map's peak, then one shared scale
    const flat = p2.maps.flatMap((m, c) => m.map((v) => (v / 255) * p2.peaks[c]));
    faces.flat = { values: toUnsigned(flat, maxOf(flat)), w: 8, h: 128, lut: "seq", tip: t("th.sch.flat") };
    faces.fc1 = { values: result.fc1.values, w: 8, h: 64, lut: "seq", tip: `FC1 · ${t("th.sch.fc1", { n: result.fc1.active })}` };
    faces.probabilities = result.probabilities;
    faces.prediction = result.prediction;
  }
  return faces;
}

export function renderSchematic() {
  const root = $("#t-schematic");
  if (!root || !T.image) return;
  const { cy, cell, stages, ops } = SCHEMATIC;
  const faces = schematicFaces();

  const marker = svg("marker", { id: "sch-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
  marker.append(svg("path", { d: "M0,0 L10,5 L0,10 z", class: "sch-head" }));
  const defs = svg("defs");
  defs.append(marker);
  const parts = [defs];
  const centers = {};
  let x = 18;

  stages.forEach((stage, i) => {
    const group = svg("g", { class: "sch-stage" });
    makeLink(group, stage.goto, `${L(stage.name)} ${L(stage.shape)}`);
    const face = faces[stage.key];
    let width;
    let faceBox;

    if (stage.size) {
      // Stack: back slabs first, then the front face with the real output
      const side = stage.size * cell;
      for (let k = stage.depth - 1; k >= 1; k--) {
        group.append(svg("rect", { x: x + k * stage.offset, y: cy - side / 2 - k * stage.offset, width: side, height: side, class: "sch-slab" }));
      }
      faceBox = { x, y: cy - side / 2, w: side, h: side };
      width = side + stage.offset * (stage.depth - 1);
    } else if (stage.column) {
      const [cw, ch] = stage.column;
      faceBox = { x, y: cy - ch / 2, w: cw, h: ch };
      width = cw;
    } else {
      // Output: one bar per class, the predicted class highlighted
      width = 84;
      const probs = faces.probabilities;
      for (let d = 0; d < 10; d++) {
        const y = cy - 55 + d * 11;
        group.append(svgText(x + 4, y + 8.5, String(d), "sch-digit", "middle"));
        group.append(svg("rect", { x: x + 12, y: y + 2, width: 66, height: 7, rx: 2, class: "sch-track" }));
        if (probs && probs[d] > 0.002) {
          group.append(svg("rect", { x: x + 12, y: y + 2, width: Math.max(2, probs[d] * 66), height: 7, rx: 2, class: d === faces.prediction ? "sch-bar top" : "sch-bar" }));
        }
      }
    }

    if (faceBox) {
      if (face) {
        const image = svg("image", { x: faceBox.x, y: faceBox.y, width: faceBox.w, height: faceBox.h, preserveAspectRatio: "none", class: "sch-face" });
        image.setAttribute("href", pixelsURL(face));
        image.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, face.tip, [{ value: `${face.w}×${face.h}`, label: t("th.sch.real") }]));
        image.addEventListener("pointerleave", hideTip);
        group.append(image);
      } else {
        group.append(svg("rect", { x: faceBox.x, y: faceBox.y, width: faceBox.w, height: faceBox.h, class: "sch-slab" }));
      }
      group.append(svg("rect", { x: faceBox.x, y: faceBox.y, width: faceBox.w, height: faceBox.h, class: "sch-frame" }));
    }

    // Name and shape on a shared baseline under every stage
    const mid = x + width / 2;
    centers[stage.key] = mid;
    const name = stage.key === "out" && faces.probabilities ? t("th.sch.pred", { d: faces.prediction }) : L(stage.name);
    group.append(svgText(mid, cy + 82, name, "sch-name", "middle"), svgText(mid, cy + 97, L(stage.shape), "sch-shape", "middle"));
    parts.push(group);

    // Arrow to the next stage, with its operations stacked above it
    const op = ops[i];
    if (op) {
      const from = x + width + 6;
      const to = x + width + op.width - 6;
      parts.push(svg("line", { x1: from, x2: to, y1: cy, y2: cy, class: "sch-arrow", "marker-end": "url(#sch-arrow)" }));
      op.items.forEach(([labelText, target, noteText], j) => {
        const label = L(labelText);
        const note = L(noteText);
        const text = svgText((from + to) / 2, cy - 9 - (op.items.length - 1 - j) * 14, label, "sch-op", "middle");
        makeLink(text, target, t("th.sch.goto", { l: label }));
        if (note) {
          text.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, label, [{ value: note, label: "" }]));
          text.addEventListener("pointerleave", hideTip);
        }
        parts.push(text);
      });
      x = x + width + op.width;
    }
  });

  // Training only: the loss flows back through the network and Adam updates every weight
  const backY = cy + 122;
  parts.push(svg("line", { x1: centers.out, x2: centers.input, y1: backY, y2: backY, class: "sch-arrow sch-back", "marker-end": "url(#sch-arrow)" }));
  const back = svgText((centers.out + centers.input) / 2, backY + 16, t("th.sch.back"), "sch-back-label", "middle");
  makeLink(back, "t-opt", t("th.sch.backlink"));
  parts.push(back);

  root.replaceChildren(...parts);
}

// =====================================================================
// Shared input image & filter, loading, lifecycle
// =====================================================================
let paramsRequest = 0;

/** Loads the selected model's weights; false if a newer request took over while this one waited. */
export async function loadParams() {
  const request = ++paramsRequest;
  try {
    const params = await api(`/api/theory/params?model=${modelView.which}`);
    if (request !== paramsRequest) return false;
    T.params = params.ready ? params : null;
    T.modelVersion = params.ready ? params.version : null;
    // Caches (BatchNorm batch, kernel grid) are keyed by this, so switching models refreshes them
    T.version = params.ready ? `${modelView.which}-${params.version}` : null;
  } catch {
    if (request !== paramsRequest) return false;
    T.params = null;
  }
  $("#t-filters-title").textContent = t(modelView.which === "initial" ? "th.filters.initial" : "th.filters.trained");
  const modelOption = $("#t-conv-kernel-select").querySelector('option[value="model"]');
  modelOption.disabled = !T.params;
  if (T.params && !conv.picked) conv.kernel = "model";
  if (!T.params && conv.kernel === "model") conv.kernel = "sobelX";
  $("#t-conv-kernel-select").value = conv.kernel;
  $("#t-filter-group").hidden = !T.params;
  $("#t-filters").hidden = !T.params;
  for (const demo of document.querySelectorAll("#panel-theory [data-needs-model]")) demo.classList.toggle("no-model", !T.params);
  renderFilterStrip();
  return true;
}

export async function loadBatch(selectFirst = true) {
  try {
    const { samples } = await api("/api/test-samples?n=48");
    T.batch = samples.map((s) => ({ ...s, x: toFloat(s.pixels) }));
  } catch (error) {
    $("#t-image-label").textContent = error.message;
    return;
  }
  if (selectFirst) await selectTestImage(T.batch[0]);
}

export async function selectTestImage(sample) {
  T.image = { x: sample.x, pixels: sample.pixels, label: sample.label, testIndex: sample.index, logits: null, prediction: null };
  await fetchLogits();
  refreshAll();
}

/** The selected model's output for the current image (scores and every layer, for the schematic). */
export async function fetchLogits() {
  if (!T.params || !T.image) return;
  const image = T.image;
  const which = modelView.which;
  const current = () => T.image === image && modelView.which === which;
  try {
    const body = image.testIndex != null ? { test_index: image.testIndex } : { pixels: Array.from(image.pixels) };
    body.model = which;
    const result = await api("/api/predict", body);
    if (!current()) return;
    image.logits = result.logits;
    image.prediction = result.prediction;
    image.result = result;   // every layer's output, used by the schematic
  } catch {
    if (!current()) return;
    image.logits = null;
    image.result = null;
  }
}

/** Describes the image the demos run on, in the current language. */
export function imageTitle() {
  const img = T.image;
  if (img.testIndex != null) return t("th.img.test", { i: img.testIndex, l: img.label });
  return img.label != null ? t("th.img.infertest", { l: img.label }) : t("th.img.drawing");
}

export function renderToolbar() {
  if (T.image) {
    paint($("#t-thumb"), T.image.pixels, 28, 28, "gray");
    $("#t-image-label").textContent = imageTitle();
  }
  $("#t-use-infer").disabled = !infer.result;
  if (T.params) {
    paint($("#t-filter-thumb"), toSigned(T.params.conv1.weight[T.filter], maxAbs(T.params.conv1.weight[T.filter])), 5, 5, "div", true);
    $("#t-filter-label").textContent = `#${T.filter + 1}`;
  }
}

export function renderFilterStrip() {
  const strip = $("#t-filter-strip");
  if (!T.params) {
    strip.replaceChildren();
    return;
  }
  const canvases = ensureCanvases(strip, 32);
  T.params.conv1.weight.forEach((w, i) => {
    paint(canvases[i], toSigned(w, maxAbs(w)), 5, 5, "div", true);
    canvases[i].classList.toggle("selected", i === T.filter);
    canvases[i].title = t("th.filter.n", { n: i + 1 });
  });
}

export function selectFilter(index) {
  if (!T.params) return;
  T.filter = (index + 32) % 32;
  renderToolbar();
  renderFilterStrip();
  if (conv.kernel === "model") {
    stopConvAnim();
    conv.pos = null;
    updateConv();
  }
  pool.pos = null;
  updateActivation();
  updatePool();
  computeBnBatch();
  resetBnParams();
  updateBN();
  renderSchematic();
}

/** Recomputes every image-dependent demo (after a new image, filter set or model). */
export function refreshAll() {
  if (!T.image) return;
  stopConvAnim();
  conv.pos = null;
  pool.pos = null;
  renderToolbar();
  renderSchematic();
  updateConv();
  updateActivation();
  updatePool();
  resetSoftmax();
  updateDropout(true);
  if (T.params && T.batch.length) {
    if (bnState.filter !== T.filter || bnState.version !== T.version || !bnState.z) {
      computeBnBatch();
      resetBnParams();
      bnState.version = T.version;
    }
    updateBN();
  }
}

export function setupToolbar() {
  $("#t-next-image").addEventListener("click", () => {
    if (!T.batch.length) return;
    const others = T.batch.filter((s) => s.index !== T.image?.testIndex);
    selectTestImage(others[Math.floor(Math.random() * others.length)]);
  });
  $("#t-use-infer").addEventListener("click", async () => {
    const r = infer.result;
    if (!r) return;
    T.image = {
      x: toFloat(r.input), pixels: r.input, label: r.label, prediction: null, testIndex: null, logits: null, result: null,
      fromInference: true,
    };
    await fetchLogits();   // recomputed with the model selected here
    refreshAll();
  });
  $("#t-filter-prev").addEventListener("click", () => selectFilter(T.filter - 1));
  $("#t-filter-next").addEventListener("click", () => selectFilter(T.filter + 1));
  $("#t-filter-strip").addEventListener("click", (e) => {
    const canvas = e.target.closest("canvas");
    if (canvas) selectFilter(Number(canvas.dataset.i));
  });
}

export async function startTheory() {
  if (T.started) return;
  T.started = true;
  setupToolbar();
  setupConv();
  setupBN();
  setupActivation();
  setupPool();
  setupDropout();
  setupSoftmax();
  setupOptimization();
  updateActivation();
  T.loading = true;
  await loadParams();
  await loadBatch();
  T.loading = false;
}

document.addEventListener("tab-shown", (event) => {
  if (event.detail !== "theory") {
    if (conv.anim) stopConvAnim();
    if (opt.anim) stopOptAnim();
    return;
  }
  if (!T.started) startTheory();
  else renderToolbar();
});

// A newer model (a finished epoch or a new training run) refreshes the demos
export async function reloadModel() {
  T.loading = true;
  if (!(await loadParams())) return;   // a newer switch is under way and finishes the job
  const which = modelView.which;
  if (T.image) await fetchLogits();
  if (which !== modelView.which) return;
  T.loading = false;
  bnState.z = null;
  refreshAll();
}

document.addEventListener("model-status", (event) => {
  const model = event.detail;
  if (!T.started || T.loading || model.version === T.modelVersion) return;
  if (!model.ready && modelView.which === "trained") return;
  reloadModel();
});

// The model switch (shared with the Inference tab): recompute every demo with the other weights
document.addEventListener("model-view-change", () => {
  if (T.started) reloadModel();
});

// Language change: rebuild the controls created here, then redraw every demo
document.addEventListener("lang-change", () => {
  if (!T.started) return;
  fillKernelOptions();
  fillBnText();
  fillOptText();
  const dropButtons = $("#t-drop-seg").children;
  if (dropButtons.length === 2) {
    dropButtons[0].textContent = t("th.drop.train");
    dropButtons[1].textContent = t("th.drop.eval");
  }
  for (const [i, row] of [...$("#t-sm-logits").children].entries()) {
    row.querySelector("input").setAttribute("aria-label", t("th.sm.slider", { d: i }));
  }
  $("#t-conv-play").textContent = conv.anim ? t("th.stop") : t("th.slide");
  $("#t-opt-play").textContent = opt.anim ? t("th.stop") : t("th.play");
  $("#t-filters-title").textContent = t(modelView.which === "initial" ? "th.filters.initial" : "th.filters.trained");
  renderFilterStrip();
  refreshAll();
  drawOpt();
});

document.addEventListener("theme-change", () => {
  if (!T.started) return;
  paintSurface();
  if (bnState.z) renderBnMaps();   // the maps use the theme's color scale
  refreshAll();
  drawOpt();
});
