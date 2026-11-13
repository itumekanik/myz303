"use strict";
/*
 * Theory tab: short explanations of the CNN building blocks, each with a live demo that
 * runs on a real MNIST test image and, when a trained model exists, on its own weights.
 * The maths runs in the browser; the server only provides Conv1 / BatchNorm1 weights,
 * test images and the model's logits. Uses the helpers defined in app.js.
 */

const SIZE = 28;
const T = { started: false, loading: false, params: null, version: null, batch: [], image: null, filter: 0 };

// =====================================================================
// Math (mirrors what PyTorch does in SimpleCNN)
// =====================================================================

/** 2D cross-correlation (what nn.Conv2d computes), single channel. */
function conv2d(x, H, W, kernel, k, bias = 0, pad = 0, stride = 1) {
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
function batchNormEval(z, channel) {
  const bn = T.params.bn1;
  const scale = bn.gamma[channel] / Math.sqrt(bn.var[channel] + bn.eps);
  return z.map((v) => (v - bn.mean[channel]) * scale + bn.beta[channel]);
}

/** 2x2 pooling with stride 2. */
function pool2(x, rows, cols, mode) {
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

const relu = (a) => a.map((v) => Math.max(0, v));
const maxAbs = (a) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m || 1; };
const maxOf = (a) => { let m = 0; for (const v of a) m = Math.max(m, v); return m || 1; };
const meanOf = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
const stdOf = (a, mu = meanOf(a)) => { let s = 0; for (const v of a) s += (v - mu) ** 2; return Math.sqrt(s / a.length); };
const toSigned = (a, scale) => Array.from(a, (v) => clamp(v / scale, -1, 1));
const toUnsigned = (a, scale) => Array.from(a, (v) => clamp(Math.round((v / scale) * 255), 0, 255));
const toFloat = (pixels) => Float32Array.from(pixels, (p) => p / 255);

function percentile(sorted, q) {
  return sorted[clamp(Math.round(q * (sorted.length - 1)), 0, sorted.length - 1)];
}

/** What ReLU sees in SimpleCNN for the selected filter: BatchNorm(Conv1(x)), 24x24. */
function featureMap() {
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
function showPixels(canvas, values, cols, rows, lut, cell, signedValues = false) {
  paint(canvas, values, cols, rows, lut, signedValues);
  canvas.style.width = `${cols * cell}px`;
  canvas.style.aspectRatio = `${cols} / ${rows}`;
}

/** Positions an overlay box over a pixel grid, in percent so it follows a scaled canvas. */
function placeBox(box, x, y, w, hgt, cols, rows) {
  Object.assign(box.style, {
    left: `${(x / cols) * 100}%`, top: `${(y / rows) * 100}%`,
    width: `${(w / cols) * 100}%`, height: `${(hgt / rows) * 100}%`,
  });
  box.hidden = false;
}

function cellFromEvent(event, canvas, cols, rows) {
  const box = canvas.getBoundingClientRect();
  return [
    clamp(Math.floor(((event.clientY - box.top) / box.height) * rows), 0, rows - 1),
    clamp(Math.floor(((event.clientX - box.left) / box.width) * cols), 0, cols - 1),
  ];
}

/** Text color that stays readable on a color-scale cell. */
function inkFor(lutName, index) {
  const lut = LUTS[lutName];
  const j = clamp(Math.round(index), 0, 255) * 3;
  return 0.299 * lut[j] + 0.587 * lut[j + 1] + 0.114 * lut[j + 2] > 150 ? "#0b0b0b" : "#ffffff";
}

function colorCell(cell, value, scale, lutName, signedValues) {
  const index = signedValues ? (clamp(value / scale, -1, 1) + 1) * 127.5 : clamp(value / scale, 0, 1) * 255;
  cell.style.background = lutColor(lutName, index);
  cell.style.color = inkFor(lutName, index);
}

const fmtValue = (v) => fmtNum(v, Number.isInteger(v) ? 0 : 2);

function matrix(values, k, lutName, scale, signedValues) {
  const grid = h("div", { class: `matrix k${k}`, style: `grid-template-columns: repeat(${k}, auto)` });
  for (const v of values) {
    const cell = h("span", { text: fmtValue(v) });
    colorCell(cell, v, scale, lutName, signedValues);
    grid.append(cell);
  }
  return grid;
}

function segmented(host, options, current, onChange) {
  host.replaceChildren(...options.map(([value, label]) => {
    const button = h("button", { type: "button", "aria-pressed": String(value === current), text: label });
    button.addEventListener("click", () => {
      for (const b of host.children) b.setAttribute("aria-pressed", String(b === button));
      onChange(value);
    });
    return button;
  }));
}

function setPressed(host, index) {
  [...host.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)));
}

/** A function curve with a crosshair; o.marker draws a highlighted point on the curve. */
function curvePlot(host, o) {
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
    hoverHost = host;
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
function columnPath(x, y, w, hgt, r = 4) {
  r = Math.min(r, w / 2, hgt);
  return `M${x},${y + hgt}V${y + r}A${r},${r} 0 0 1 ${x + r},${y}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${y + hgt}Z`;
}

/** Histogram on a shared x range; a bar far taller than the rest is cut and labelled. */
function histogram(host, values, lo, hi, mean, opts = {}) {
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
const KERNELS = {
  model: { label: { tr: "Modelin filtresi (5×5)", en: "The model's filter (5×5)" } },
  sobelX: { label: { tr: "Dikey kenar · Sobel (3×3)", en: "Vertical edge · Sobel (3×3)" }, k: 3, values: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  sobelY: { label: { tr: "Yatay kenar · Sobel (3×3)", en: "Horizontal edge · Sobel (3×3)" }, k: 3, values: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  blur: { label: { tr: "Bulanıklaştırma (3×3)", en: "Blur (3×3)" }, k: 3, values: new Array(9).fill(1 / 9) },
  sharpen: { label: { tr: "Keskinleştirme (3×3)", en: "Sharpen (3×3)" }, k: 3, values: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
  custom: { label: { tr: "Özel · değerleri düzenleyin (3×3)", en: "Custom · edit the values (3×3)" }, k: 3, values: [0, 1, 0, 1, -4, 1, 0, 1, 0] },
};
const conv = { kernel: "sobelX", picked: false, pad: 0, stride: 1, data: null, reveal: Infinity, anim: 0, pos: null };
const CONV_CELL = 8;

function convKernel() {
  if (conv.kernel === "model" && T.params) {
    return { k: 5, values: T.params.conv1.weight[T.filter], bias: T.params.conv1.bias[T.filter] };
  }
  const preset = KERNELS[conv.kernel] || KERNELS.sobelX;
  return { k: preset.k, values: preset.values, bias: 0 };
}

function fillKernelOptions() {
  const select = $("#t-conv-kernel-select");
  const value = select.value;
  select.replaceChildren(...Object.entries(KERNELS).map(([key, k]) => h("option", { value: key, text: L(k.label) })));
  if (value) select.value = value;
  const modelOption = select.querySelector('option[value="model"]');
  if (modelOption) modelOption.disabled = !T.params;
}

function setupConv() {
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

function updateConv() {
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
  $("#t-conv-size").textContent = ` = ⌊(28 − ${ker.k} + 2·${P}) / ${conv.stride}⌋ + 1 = ${oh}`;
  paintConvOut();

  // Start on the strongest response, so the calculation shows something interesting
  if (conv.pos == null || conv.pos >= oh * ow) {
    let best = 0;
    for (let i = 1; i < out.length; i++) if (Math.abs(out[i]) > Math.abs(out[best])) best = i;
    conv.pos = best;
  }
  setConvPos(conv.pos);
}

function renderKernel(ker) {
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

function paintConvOut() {
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

function setConvPos(index) {
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

function playConv() {
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

function stopConvAnim() {
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
const BN_MOMENTUM = 0.1;   // PyTorch's default; SimpleCNN does not change it

const BN_TEXT = {
  lead: {
    tr: "Batch Normalization (BatchNorm), bir katmanın çıktısını sonraki katmana vermeden önce <b>her [[channel|kanalı]] ayrı ayrı</b> standart bir ölçeğe getirir: önce değerleri ortalaması 0, standart sapması 1 olacak şekilde kaydırıp ölçekler, sonra öğrenilen iki sayıyla (γ ve β) ağın ihtiyaç duyduğu ölçeğe taşır. Aşağıda bunu adım adım, seçili filtrenin gerçek test görüntülerindeki çıktısı üzerinde izleyebilirsiniz.",
    en: "Batch normalization (BatchNorm) brings <b>each [[channel|channel]] separately</b> to a standard scale before passing a layer's output on: it first shifts and scales the values to mean 0 and standard deviation 1, then moves them to whatever scale the network needs with two learned numbers (γ and β). Below you can follow it step by step on the selected filter's real output for test images.",
  },
  "flow.x": { tr: "evrişim çıktısı", en: "convolution output" },
  "flow.stats": { tr: "kanalın batch ortalaması ve std. sapması", en: "the channel's batch mean and std. dev." },
  "flow.norm": { tr: "normalleştir", en: "normalize" },
  "flow.affine": { tr: "öğrenilen ölçek ve kayma", en: "learned scale and shift" },
  "flow.relu": { tr: "sonraki işlem", en: "next operation" },

  "s1.h": { tr: "Sorun: her kanal farklı bir ölçekte", en: "The problem: every channel has its own scale" },
  "s1.p": {
    tr: "<p>Conv1'in 32 filtresinin her biri farklı bir desene bakar ve çıktıları farklı aralıklarda olur: kiminin değerleri dar bir aralıkta toplanır, kimininki geniş bir aralığa yayılır; kiminin ortalaması pozitif, kiminin negatiftir. Üstelik ağırlıklar her adımda değiştiği için bu aralıklar eğitim boyunca kayar; sonraki katman sürekli kayan girdilere uyum sağlamak zorunda kalır. Eğitim yavaşlar ve [[learning-rate|öğrenme oranını]] küçük tutmak gerekir.</p><p>BatchNorm her kanalı aynı başlangıç ölçeğine çeker. Grafikte her çubuk, bir kanalın bu batch'teki değer aralığıdır (ortalama ± 1 standart sapma). Mavi bant, normalleştirmeden sonra <b>bütün kanalların</b> geleceği aralıktır: dar kanallar genişler, geniş olanlar daralır, kaymış olanlar ortalanır.</p>",
    en: "<p>Each of Conv1's 32 filters looks for a different pattern, and their outputs fall in different ranges: some channels' values stay in a narrow range, others spread widely; some have a positive mean, others a negative one. And since the weights change at every step, these ranges drift during training; the next layer has to keep adapting to inputs that keep moving. Training slows down and the [[learning-rate|learning rate]] has to stay small.</p><p>BatchNorm brings every channel to the same starting scale. In the chart, each bar is one channel's range in this batch (mean ± 1 standard deviation). The blue band is where <b>every channel</b> ends up after normalization: narrow channels are widened, wide ones narrowed, shifted ones centered.</p>",
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
    tr: "<p>Havuzdaki <i>m</i> değerin ortalaması μ<sub>B</sub> ve varyansı σ<sub>B</sub><sup>2</sup> hesaplanır (B: batch):</p><div class=\"formula small\">μ<sub>B</sub> = (1/m) Σ x<sub>i</sub> &nbsp;&nbsp; σ<sub>B</sub><sup>2</sup> = (1/m) Σ (x<sub>i</sub> − μ<sub>B</sub>)<sup>2</sup></div><p>Bunlar <b>bu batch'in</b> istatistikleridir; başka bir batch'te biraz farklı çıkarlar. Yukarıdaki <b>Yeni batch</b> düğmesiyle deneyin.</p>",
    en: "<p>The mean μ<sub>B</sub> and variance σ<sub>B</sub><sup>2</sup> of the <i>m</i> pooled values are computed (B: batch):</p><div class=\"formula small\">μ<sub>B</sub> = (1/m) Σ x<sub>i</sub> &nbsp;&nbsp; σ<sub>B</sub><sup>2</sup> = (1/m) Σ (x<sub>i</sub> − μ<sub>B</sub>)<sup>2</sup></div><p>These are <b>this batch's</b> statistics; another batch gives slightly different ones. Try the <b>New batch</b> button above.</p>",
  },
  h1: { tr: "Evrişim çıktısı x · gölgeli alan μ ± σ", en: "Convolution output x · shaded area μ ± σ" },

  "s4.h": { tr: "Normalleştir: ortalamayı çıkar, standart sapmaya böl", en: "Normalize: subtract the mean, divide by the standard deviation" },
  "s4.p": {
    tr: "<div class=\"formula small\">x̂ = (x − μ<sub>B</sub>) / √(σ<sub>B</sub><sup>2</sup> + ε)</div><p>Her değerden kanal ortalaması çıkarılır (dağılım 0'a kayar) ve standart sapmaya bölünür (genişliği 1 olur: σ'sı 1'den küçük olan bir kanal genişler, büyük olan daralır). Dağılımın <b>şekli değişmez</b>, yalnızca yeri ve genişliği değişir: üstteki ve alttaki histogramlar aynı eksende çizildi.</p><p>ε çok küçük bir sayıdır (0,00001); σ sıfıra çok yakın olduğunda sıfıra bölmeyi önler.</p><p class=\"bn-note\">İlginç bir sonuç: evrişimin bias'ı her değere aynı sabiti ekler ve ortalama çıkarılırken bu sabit de silinir. Bu yüzden BatchNorm'dan önce gelen katmanlarda bias çoğu zaman kapatılır (<code>bias=False</code>).</p>",
    en: "<div class=\"formula small\">x̂ = (x − μ<sub>B</sub>) / √(σ<sub>B</sub><sup>2</sup> + ε)</div><p>The channel mean is subtracted from every value (the distribution moves to 0) and the result is divided by the standard deviation (its width becomes 1: a channel with σ below 1 gets wider, one above 1 narrower). The <b>shape</b> of the distribution <b>does not change</b>, only its position and width: the histograms above and below use the same axis.</p><p>ε is a tiny number (0.00001) that prevents dividing by zero when σ is very close to zero.</p><p class=\"bn-note\">A neat consequence: the convolution's bias adds the same constant to every value, and subtracting the mean removes that constant too. That is why layers right before a BatchNorm often have their bias turned off (<code>bias=False</code>).</p>",
  },
  h2: { tr: "Normalleştirilmiş x̂ · gölgeli alan −1 … 1", en: "Normalized x̂ · shaded area −1 … 1" },

  "s5.h": { tr: "Ölçekle ve kaydır: γ ve β", en: "Scale and shift: γ and β" },
  "s5.p": {
    tr: "<div class=\"formula small\">y = γ · x̂ + β</div><p>Her kanalı zorla ortalama 0, standart sapma 1'de tutmak ağı kısıtlayabilir: örneğin bir kanalın [[relu|ReLU]]'dan önce çoğunlukla negatif kalması, yani yalnızca güçlü desenlerde \"yanması\" daha yararlı olabilir. Bu yüzden her kanalın öğrenilen iki parametresi vardır: <b>γ</b> (ölçek) dağılımı genişletir ya da daraltır, <b>β</b> (kayma) sağa ya da sola kaydırır.</p><p>γ ve β da ağırlıklar gibi [[backprop|geri yayılımla]] öğrenilir. Ağ isterse γ = σ ve β = μ seçerek normalleştirmeyi tamamen geri alabilir; yani BatchNorm ağdan bir şey eksiltmez, yalnızca öğrenmeyi kolaylaştırır.</p><p>Kaydırıcılarla deneyin; <b>Modelin γ, β değerleri</b> düğmesi eğitimin bulduğu değerlere döner.</p>",
    en: "<div class=\"formula small\">y = γ · x̂ + β</div><p>Forcing every channel to mean 0 and standard deviation 1 could restrict the network: it may be more useful for a channel to stay mostly negative before [[relu|ReLU]], so that it only \"fires\" on strong patterns. That is why every channel has two learned parameters: <b>γ</b> (scale) widens or narrows the distribution, <b>β</b> (shift) moves it left or right.</p><p>γ and β are learned with [[backprop|backpropagation]] just like the weights. If it helps, the network can choose γ = σ and β = μ and undo the normalization completely; BatchNorm takes nothing away from the network, it only makes learning easier.</p><p>Try the sliders; the <b>Model's γ and β</b> button goes back to the values training found.</p>",
  },
  h3: { tr: "BatchNorm çıktısı y · sağdaki kısım ReLU'dan geçer", en: "BatchNorm output y · the part right of 0 passes ReLU" },

  "s6.h": { tr: "Eğitimde ve çıkarımda farklı çalışır", en: "It works differently in training and at inference" },
  "s6.p": {
    tr: "<p>Yukarıdaki μ<sub>B</sub> ve σ<sub>B</sub> bu batch'ten hesaplandı; eğitimde ([[train-eval|<code>model.train()</code>]]) tam olarak böyle çalışır. Ama [[inference|çıkarımda]] tek bir görüntü gelebilir ve bir görüntünün sonucunun yanındaki başka görüntülere bağlı olması istenmez.</p><p>Bu yüzden BatchNorm, eğitimin her adımında batch istatistiklerinin [[moving-average|hareketli ortalamasını]] günceller ve çıkarımda (<code>model.eval()</code>) bunları kullanır. Bu ortalamalar öğrenilmez, yalnızca biriktirilir ve [[checkpoint|checkpoint]]'e γ ve β ile birlikte kaydedilir.</p>",
    en: "<p>The μ<sub>B</sub> and σ<sub>B</sub> above were computed from this batch; that is exactly how it works in training ([[train-eval|<code>model.train()</code>]]). But at [[inference|inference]] a single image may come in, and an image's result should not depend on the other images next to it.</p><p>That is why, at every training step, BatchNorm updates a [[moving-average|moving average]] of the batch statistics, and uses those at inference (<code>model.eval()</code>). They are not learned, only accumulated, and they are saved in the [[checkpoint|checkpoint]] together with γ and β.</p>",
  },

  model: {
    tr: "<b>SimpleCNN'de:</b> Her evrişimden sonra, ReLU'dan önce: <code>BatchNorm2d(32)</code> ve <code>BatchNorm2d(64)</code>. Her biri kanal başına bir γ ve bir β öğrenir (toplam 2 × (32 + 64) = 192 parametre) ve kanal başına bir hareketli ortalama ile bir hareketli varyans saklar.",
    en: "<b>In SimpleCNN:</b> after every convolution, before ReLU: <code>BatchNorm2d(32)</code> and <code>BatchNorm2d(64)</code>. Each learns one γ and one β per channel (2 × (32 + 64) = 192 parameters in total) and keeps one running mean and one running variance per channel.",
  },
};

const bnState = {
  gamma: 1, beta: 0, z: null, zSorted: null, xh: null, xhSorted: null, mu: 0, sd: 1,
  pick: null, channels: null, channelsKey: null, frame: 0,
};

function fillBnText() {
  for (const el of document.querySelectorAll("#t-bn [data-bn]")) {
    el.innerHTML = termify(L(BN_TEXT[el.dataset.bn]));   // static, author-written markup
  }
}

function setupBN() {
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
function channelStats() {
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
function computeBnBatch() {
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

function resetBnParams() {
  if (!T.params) return;
  bnState.gamma = T.params.bn1.gamma[T.filter];
  bnState.beta = T.params.bn1.beta[T.filter];
}

/** Step 2: the selected channel's map for every image in the batch, on one color scale. */
function renderBnMaps() {
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
function renderBnChannels(host) {
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
function bnCalc(host, rows) {
  host.replaceChildren(...rows.map(([label, expr, result]) => h("div", { class: "bn-calc-row" },
    h("span", { class: "calc-cap", text: label }),
    h("span", { class: "bn-expr", text: expr }),
    result != null ? h("strong", { text: result }) : null)));
}

function updateBN() {
  if (!T.params || !bnState.z) return;
  const { z, xh, zSorted, xhSorted, mu, sd, denom, gamma, beta, pick } = bnState;
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
    h("span", {}, "m = ", h("strong", { text: fmtInt(z.length) })),
    h("span", {}, "μ", h("sub", { text: "B" }), " = ", h("strong", { text: fmtNum(mu, 4) })),
    h("span", {}, "σ", h("sub", { text: "B" }), " = ", h("strong", { text: fmtNum(sd, 4) })));
  bnCalc($("#t-bn-c1"), [
    [L({ tr: "İzlenen değer", en: "Traced value" }), `x = ${fmtNum(x, 4)}`, null],
    ["", where, null],
  ]);
  bnCalc($("#t-bn-c2"), [
    [L({ tr: "İzlenen değer", en: "Traced value" }),
      `x̂ = (${fmtNum(x, 4)} − ${fmtNum(mu, 4)}) / √(${fmtNum(sd, 4)}² + ${fmtNum(eps, 5)})`, `= ${fmtNum(xn, 3)}`],
    [L({ tr: "Bütün kanal", en: "Whole channel" }),
      L({ tr: `ortalama ${fmtNum(meanOf(xh), 3)}, std. sapma ${fmtNum(stdOf(xh), 3)}`, en: `mean ${fmtNum(meanOf(xh), 3)}, std. dev. ${fmtNum(stdOf(xh), 3)}` }), null],
  ]);
  const active = ys.reduce((count, v) => count + (v > 0), 0) / ys.length;
  bnCalc($("#t-bn-c3"), [
    [L({ tr: "İzlenen değer", en: "Traced value" }),
      `y = ${fmtNum(gamma, 2)} · ${fmtNum(xn, 3)} + (${fmtNum(beta, 2)})`, `= ${fmtNum(y, 3)}`],
    ["ReLU", `max(0, ${fmtNum(y, 3)})`, `= ${fmtNum(Math.max(0, y), 3)}`],
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
      h("div", { class: "formula small" }, "μ", h("sub", { text: "run" }), ` ← ${fmtNum(1 - BN_MOMENTUM, 1)} · μ`, h("sub", { text: "run" }),
        ` + ${fmtNum(BN_MOMENTUM, 1)} · μ`, h("sub", { text: "B" })),
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
const ACTIVATIONS = {
  relu: {
    label: "ReLU", f: (x) => Math.max(0, x), y: [-1, 4],
    formula: "ReLU(<i>x</i>) = max(0, <i>x</i>)",
    desc: { tr: "Negatif değerleri 0 yapar, pozitifleri olduğu gibi geçirir. Haritada mavi (negatif) bölgelerin tamamen kaybolduğuna dikkat edin.",
      en: "Sets negative values to 0 and passes positive values unchanged. Notice how the blue (negative) regions disappear completely from the map." },
  },
  leaky: {
    label: "Leaky ReLU", f: (x) => (x > 0 ? x : 0.01 * x), y: [-1, 4],
    formula: {
      tr: "LReLU(<i>x</i>) = <i>x</i> (<i>x</i> &gt; 0 ise), 0,01·<i>x</i> (aksi halde)",
      en: "LReLU(<i>x</i>) = <i>x</i> (if <i>x</i> &gt; 0), 0.01·<i>x</i> (otherwise)",
    },
    desc: { tr: "ReLU gibidir ama negatif tarafta küçük bir eğim (0,01) bırakır; böylece negatif girişlerde de [[gradient|gradyan]] tamamen sıfırlanmaz.",
      en: "Like ReLU, but keeps a small slope (0.01) on the negative side, so the [[gradient|gradient]] does not vanish entirely for negative inputs." },
  },
  sigmoid: {
    label: "Sigmoid", f: (x) => 1 / (1 + Math.exp(-x)), y: [-0.25, 1.25],
    formula: "σ(<i>x</i>) = 1 / (1 + e<sup>−<i>x</i></sup>)",
    desc: { tr: "Her değeri 0 ile 1 arasına sıkıştırır. |x| büyüdükçe eğimi sıfıra yaklaşır; derin ağlarda [[gradient|gradyanın]] sönümlenmesine yol açabilir.",
      en: "Squeezes every value into the range 0 to 1. Its slope approaches zero as |x| grows, which can make [[gradient|gradients]] vanish in deep networks." },
  },
  tanh: {
    label: "Tanh", f: Math.tanh, y: [-1.25, 1.25],
    formula: "tanh(<i>x</i>) = (e<sup><i>x</i></sup> − e<sup>−<i>x</i></sup>) / (e<sup><i>x</i></sup> + e<sup>−<i>x</i></sup>)",
    desc: { tr: "Değerleri −1 ile 1 arasına sıkıştırır ve 0 etrafında simetriktir. Sigmoid gibi uç değerlerde doyuma ulaşır.",
      en: "Squeezes values into the range −1 to 1 and is symmetric around 0. Like the sigmoid, it saturates at extreme values." },
  },
};
const act = { kind: "relu", data: null };

function setupActivation() {
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

function updateActivation() {
  const fn = ACTIVATIONS[act.kind];
  $("#t-act-formula").innerHTML = L(fn.formula);   // static, author-written markup
  $("#t-act-desc").innerHTML = termify(L(fn.desc));   // static, author-written text; [[key|text]] → glossary links
  curvePlot($("#t-act-plot"), {
    label: t("th.act.plot", { f: fn.label }),
    x: [-4, 4], y: fn.y, fn: fn.f,
    xTicks: [-4, -2, 0, 2, 4].map((v) => ({ v, label: fmtTick(v) })),
    yTicks: linearTicks(fn.y[0], fn.y[1], niceStep(fn.y[1] - fn.y[0], 4), fmtTick),
    tooltip: (x) => ({ title: `x = ${fmtNum(x, 2)}`, rows: [{ value: fmtNum(fn.f(x), 3), label: `${fn.label}(x)`, color: "var(--series-1)" }] }),
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
const pool = { mode: "max", data: null, pos: null };
const POOL_CELL = 9;

function setupPool() {
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

function updatePool() {
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

function setPoolPos(index) {
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
const drop = { p: 0.45, mode: "train", maps: null, noise: null, shown: null };

function setupDropout() {
  const slider = $("#t-drop-p");
  slider.addEventListener("input", () => { drop.p = Number(slider.value); updateDropout(); });
  segmented($("#t-drop-seg"), [["train", t("th.drop.train")], ["eval", t("th.drop.eval")]], drop.mode, (mode) => {
    drop.mode = mode;
    updateDropout();
  });
  $("#t-drop-new").addEventListener("click", () => { drop.noise = null; updateDropout(); });
}

/** MaxPool(ReLU(BatchNorm(Conv1(x)))) for all 32 filters: exactly what Dropout1 receives. */
function pool1Maps() {
  return T.params.conv1.weight.map((w, c) => {
    const { out, oh, ow } = conv2d(T.image.x, SIZE, SIZE, w, 5, T.params.conv1.bias[c]);
    return pool2(relu(batchNormEval(out, c)), oh, ow, "max").out;
  });
}

function updateDropout(newInput = false) {
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
const sm = { logits: new Array(10).fill(0), base: null, label: 0 };

function setupSoftmax() {
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

function resetSoftmax() {
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

function syncLogitSliders() {
  [...$("#t-sm-logits").children].forEach((row, d) => { row.querySelector("input").value = sm.logits[d]; });
}

function updateSoftmax() {
  const top = Math.max(...sm.logits);
  const exps = sm.logits.map((z) => Math.exp(z - top));   // shifted for numerical stability
  const total = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / total);
  const best = probs.indexOf(Math.max(...probs));
  [...$("#t-sm-logits").children].forEach((row, d) => { row.querySelector("output").textContent = fmtNum(sm.logits[d], 1); });
  probabilityChart($("#t-sm-probs"), probs, best);

  const p = probs[sm.label];
  const loss = -Math.log(Math.max(p, 1e-12));
  $("#t-ce-value").textContent = fmtNum(loss, 3);
  $("#t-ce-live").textContent = ` = −log(${fmtNum(p, 4)}) = ${fmtNum(loss, 3)}`;
  curvePlot($("#t-ce-plot"), {
    label: t("th.ce.plot"),
    x: [0.001, 1], y: [0, 7], fn: (q) => -Math.log(q), marker: clamp(p, 0.001, 1),
    xTicks: [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v: Math.max(v, 0.001), label: fmtPct(v * 100, 0) })),
    yTicks: [0, 2, 4, 6].map((v) => ({ v, label: String(v) })),
    tooltip: (q) => ({ title: t("th.ce.tip", { v: fmtPct(q * 100, 1) }), rows: [{ value: fmtNum(-Math.log(q), 3), label: t("th.loss"), color: "var(--series-1)" }] }),
  });
}

// =====================================================================
// 7 · Gradient descent: SGD vs Momentum vs Adam
// =====================================================================
const SURFACE = { f: (a, b) => (a * a) / 20 + b * b, grad: (a, b) => [a / 10, 2 * b], x: [-10, 10], y: [-5, 5] };
const OPTIMIZERS = [
  { key: "sgd", label: "SGD", color: "var(--series-1)" },
  { key: "momentum", label: "Momentum", color: "var(--series-2)" },
  { key: "adam", label: "Adam", color: "var(--series-3)" },
];
const opt = { lr: 0.3, start: [-8.5, 3.6], steps: 80, data: null, shown: 80, anim: 0 };

function runOptimizers() {
  const lr = opt.lr;
  return OPTIMIZERS.map((o) => {
    let p = [...opt.start];
    let v = [0, 0];
    let m = [0, 0];
    let s = [0, 0];
    let diverged = false;
    const path = [[...p]];
    const losses = [SURFACE.f(...p)];
    for (let t = 1; t <= opt.steps; t++) {
      if (!diverged) {
        const g = SURFACE.grad(...p);
        let next;
        if (o.key === "sgd") {
          next = [p[0] - lr * g[0], p[1] - lr * g[1]];
        } else if (o.key === "momentum") {
          v = [0.9 * v[0] + g[0], 0.9 * v[1] + g[1]];
          next = [p[0] - lr * v[0], p[1] - lr * v[1]];
        } else {
          m = [0.9 * m[0] + 0.1 * g[0], 0.9 * m[1] + 0.1 * g[1]];
          s = [0.999 * s[0] + 0.001 * g[0] ** 2, 0.999 * s[1] + 0.001 * g[1] ** 2];
          const mHat = m.map((q) => q / (1 - 0.9 ** t));
          const sHat = s.map((q) => q / (1 - 0.999 ** t));
          next = [p[0] - (lr * mHat[0]) / (Math.sqrt(sHat[0]) + 1e-8), p[1] - (lr * mHat[1]) / (Math.sqrt(sHat[1]) + 1e-8)];
        }
        if (next.every((q) => Number.isFinite(q) && Math.abs(q) < 1e3)) p = next;
        else diverged = true;
      }
      path.push([...p]);
      losses.push(SURFACE.f(...p));
    }
    return { ...o, path, losses, diverged };
  });
}

function setupOptimization() {
  const slider = $("#t-opt-lr");
  slider.value = Math.log10(opt.lr);
  slider.addEventListener("input", () => {
    opt.lr = 10 ** Number(slider.value);
    stopOptAnim();
    recomputeOpt();
  });
  $("#t-opt-play").addEventListener("click", () => (opt.anim ? stopOptAnim() : playOpt()));
  $("#t-opt-stage").addEventListener("click", (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - box.left) / box.width;
    const fy = (e.clientY - box.top) / box.height;
    opt.start = [SURFACE.x[0] + fx * (SURFACE.x[1] - SURFACE.x[0]), SURFACE.y[1] - fy * (SURFACE.y[1] - SURFACE.y[0])];
    stopOptAnim();
    recomputeOpt();
    playOpt();
  });
  paintSurface();
  recomputeOpt();
}

/** Neutral banded background: each band is a range of equal loss. */
function paintSurface() {
  const cols = 240;
  const rows = 120;
  const fMax = SURFACE.f(SURFACE.x[1], SURFACE.y[1]);
  const values = new Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = SURFACE.x[0] + ((c + 0.5) / cols) * (SURFACE.x[1] - SURFACE.x[0]);
      const b = SURFACE.y[1] - ((r + 0.5) / rows) * (SURFACE.y[1] - SURFACE.y[0]);
      const level = Math.sqrt(SURFACE.f(a, b) / fMax);
      values[r * cols + c] = Math.round((Math.min(Math.floor(level * 12), 11) / 11) * 255);
    }
  }
  paint($("#t-opt-surface"), values, cols, rows, "ramp");
}

function recomputeOpt() {
  opt.data = runOptimizers();
  opt.shown = opt.steps;
  $("#t-opt-lr-v").textContent = `η = ${fmtNum(opt.lr, opt.lr < 0.1 ? 3 : 2)}`;
  drawOpt();
}

const toSup = (text) => text.replace(/[-0-9]/g, (ch) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(ch)]);
function fmtLoss(v) {
  if (!(v > 1e-12)) return "≈ 0";
  if (v >= 100) return fmtNum(v, 0);
  if (v >= 0.001) return fmtNum(v, 4);
  const e = Math.floor(Math.log10(v));
  return `${fmtNum(v / 10 ** e, 1)}×10${toSup(String(e))}`;
}

function drawOpt() {
  const n = opt.shown;
  const X = (a) => ((a - SURFACE.x[0]) / (SURFACE.x[1] - SURFACE.x[0])) * 200;
  const Y = (b) => ((SURFACE.y[1] - b) / (SURFACE.y[1] - SURFACE.y[0])) * 100;
  const parts = [svg("path", { d: `M${X(0) - 3},${Y(0)}h6M${X(0)},${Y(0) - 3}v6`, class: "opt-min" })];
  for (const o of opt.data) {
    const points = o.path.slice(0, n + 1).map(([a, b]) => `${X(a).toFixed(2)},${Y(b).toFixed(2)}`).join(" ");
    parts.push(svg("polyline", { points, class: "opt-path", style: `stroke:${o.color}` }));
  }
  parts.push(svg("circle", { cx: X(opt.start[0]), cy: Y(opt.start[1]), r: 2.2, class: "opt-start" }));
  for (const o of opt.data) {
    const [a, b] = o.path[n];
    parts.push(svg("circle", { cx: X(a), cy: Y(b), r: 2.4, class: "opt-dot", style: `fill:${o.color}` }));
  }
  $("#t-opt-svg").replaceChildren(...parts);

  $("#t-opt-legend").replaceChildren(...opt.data.map((o) => h("span", { class: "legend-item" },
    h("span", { class: "key-line", style: `background:${o.color}` }),
    `${o.label} · ${t("th.opt.after", { n })} `,
    h("strong", { text: fmtLoss(o.losses[n]) }),
    o.diverged ? ` · ${t("th.opt.diverged")}` : "")));

  // Loss per step on a log axis (x = step + 1 so the shared chart helper can index it)
  const series = opt.data.map((o) => ({ name: o.label, color: o.color, points: o.losses.slice(0, n + 1).map((v, i) => [i + 1, Math.max(v, 1e-6)]) }));
  const all = series.flatMap((s) => s.points.map((p) => p[1]));
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
  for (let s = 0; s <= opt.steps; s += 20) xTicks.push({ v: s + 1, label: String(s) });
  lineChart($("#t-opt-loss"), {
    label: t("th.opt.label"),
    empty: "",
    x: { min: 1, max: opt.steps + 1, ticks: xTicks },
    y: { log: true, min: yMin, max: yMax, ticks: yTicks },
    series,
    tooltip: (i) => ({ title: t("th.opt.step", { i }), rows: opt.data.map((o) => ({ value: fmtLoss(o.losses[i]), label: o.label, color: o.color })) }),
  });
}

function playOpt() {
  stopOptAnim();
  opt.shown = 0;
  $("#t-opt-play").textContent = t("th.stop");
  let last = 0;
  const frame = (time) => {
    if (time - last > 45) {
      last = time;
      opt.shown++;
      drawOpt();
    }
    if (opt.shown < opt.steps) opt.anim = requestAnimationFrame(frame);
    else stopOptAnim();
  };
  opt.anim = requestAnimationFrame(frame);
}

function stopOptAnim() {
  if (opt.anim) cancelAnimationFrame(opt.anim);
  opt.anim = 0;
  $("#t-opt-play").textContent = t("th.play");
}

// =====================================================================
// 0 · Architecture schematic
// Stacks are drawn to scale (height = spatial size, depth = channels); the front face of each
// stack shows the selected image's real output at that layer.
// =====================================================================
const SCHEMATIC = {
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
function pixelsURL(face) {
  const canvas = document.createElement("canvas");
  paint(canvas, face.values, face.w, face.h, face.lut);
  return canvas.toDataURL();
}

/** Jumps to a section of this tab as a history step (Back returns to the diagram). */
const jumpToTheory = (id) => navigate("theory", id);

function makeLink(node, target, label) {
  node.setAttribute("tabindex", "0");
  node.setAttribute("role", "link");
  node.setAttribute("aria-label", label);
  node.addEventListener("click", () => jumpToTheory(target));
  node.addEventListener("keydown", (e) => { if (e.key === "Enter") jumpToTheory(target); });
}

/** The real output of every stage for the selected image (what the front faces show). */
function schematicFaces() {
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

function renderSchematic() {
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
async function loadParams() {
  try {
    const params = await api(`/api/theory/params?model=${modelView.which}`);
    if (params.ready && params.model !== modelView.which) showOutdatedServer();
    T.params = params.ready ? params : null;
    T.modelVersion = params.ready ? params.version : null;
    // Caches (BatchNorm batch, kernel grid) are keyed by this, so switching models refreshes them
    T.version = params.ready ? `${modelView.which}-${params.version}` : null;
  } catch {
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
}

async function loadBatch(selectFirst = true) {
  try {
    const { samples } = await api("/api/test-samples?n=48");
    T.batch = samples.map((s) => ({ ...s, x: toFloat(s.pixels) }));
  } catch (error) {
    $("#t-image-label").textContent = error.message;
    return;
  }
  if (selectFirst) await selectTestImage(T.batch[0]);
}

async function selectTestImage(sample) {
  T.image = { x: sample.x, pixels: sample.pixels, label: sample.label, testIndex: sample.index, logits: null, prediction: null };
  await fetchLogits();
  refreshAll();
}

/** The selected model's output for the current image (scores and every layer, for the schematic). */
async function fetchLogits() {
  if (!T.params || !T.image) return;
  try {
    const body = T.image.testIndex != null ? { test_index: T.image.testIndex } : { pixels: Array.from(T.image.pixels) };
    body.model = modelView.which;
    const result = await api("/api/predict", body);
    T.image.logits = result.logits;
    T.image.prediction = result.prediction;
    T.image.result = result;   // every layer's output, used by the schematic
  } catch {
    T.image.logits = null;
    T.image.result = null;
  }
}

/** Describes the image the demos run on, in the current language. */
function imageTitle() {
  const img = T.image;
  if (img.testIndex != null) return t("th.img.test", { i: img.testIndex, l: img.label });
  return img.label != null ? t("th.img.infertest", { l: img.label }) : t("th.img.drawing");
}

function renderToolbar() {
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

function renderFilterStrip() {
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

function selectFilter(index) {
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
function refreshAll() {
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

function setupToolbar() {
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
  // The toolbar sticks right under the top bar, whose height changes with the window width
  const topbar = $(".topbar");
  new ResizeObserver(() => document.documentElement.style.setProperty("--topbar-h", `${topbar.offsetHeight}px`)).observe(topbar);
}

async function startTheory() {
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
  if (event.detail !== "theory") return;
  if (!T.started) startTheory();
  else renderToolbar();
});

// A newer model (a finished epoch or a new training run) refreshes the demos
async function reloadModel() {
  T.loading = true;
  await loadParams();
  if (T.image) await fetchLogits();
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
