"use strict";
/*
 * SimpleCNN dashboard front-end.
 * Polls /api/status for training progress and renders it, and sends drawings or
 * test images to /api/predict to show what every layer of the network produces.
 */

// =====================================================================
// Helpers
// =====================================================================
const $ = (selector, root = document) => root.querySelector(selector);
const SVG_NS = "http://www.w3.org/2000/svg";

/** Creates an HTML element. Text always goes through textContent. */
function h(tag, props = {}, ...children) {
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

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value != null) node.setAttribute(key, value);
  return node;
}

function svgText(x, y, content, cls, anchor = "start") {
  const node = svg("text", { x, y, class: cls, "text-anchor": anchor });
  node.textContent = content;
  return node;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const cssList = (name) => cssVar(name).split(",").map((s) => s.trim()).filter(Boolean);

// Numbers follow the language: 0,001 and %99,29 in Turkish, 0.001 and 99.29% in English
const locale = () => (I18N.lang === "en" ? "en-US" : "tr-TR");
const formats = {};
function fmtNum(value, digits) {
  const key = `${I18N.lang}:${digits}`;
  formats[key] ??= new Intl.NumberFormat(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return formats[key].format(value);
}
const pctText = (text) => (I18N.lang === "en" ? `${text}%` : `%${text}`);
const fmtPct = (value, digits = 2) => pctText(fmtNum(value, digits));
const fmtInt = (value) => fmtNum(value, 0);
const fmtTick = (value) => new Intl.NumberFormat(locale(), { maximumSignificantDigits: 2 }).format(value);
/** Probability label: a tiny value shows as "<0.1%" instead of 0.0%. */
const fmtProb = (p) => (p < 0.001 ? `<${fmtPct(0.1, 1)}` : fmtPct(p * 100, 1));

function fmtDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return "—";
  seconds = Math.max(0, Math.round(seconds));
  const hrs = Math.floor(seconds / 3600);
  const min = Math.floor((seconds % 3600) / 60);
  const sec = String(seconds % 60).padStart(2, "0");
  return hrs ? `${hrs}:${String(min).padStart(2, "0")}:${sec}` : `${min}:${sec}`;
}

async function api(path, body) {
  const options = body === undefined
    ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(localizeServerMessage(data.error) || `HTTP ${response.status}`);
  return data;
}

/** The server answers in Turkish; known messages are shown in the page's language. */
const SERVER_MESSAGES = [
  [/^Önce bir rakam çizin\.$/, "srv.draw"],
  [/^Henüz eğitilmiş bir model yok\.$/, "srv.nomodel"],
  [/^Eğitim zaten sürüyor\.$/, "srv.running"],
  [/^Geçersiz model\.$/, "srv.badmodel"],
  [/^Görüntü gönderilmedi\.$/, "srv.noimage"],
  [/^Geçersiz test görüntüsü\.$/, "srv.badtest"],
  [/^28×28 = 784 piksel bekleniyor\.$/, "srv.pixels"],
  [/^İstek çok büyük\.$/, "srv.toolarge"],
  [/^Geçersiz JSON\.$/, "srv.json"],
  [/^Bulunamadı\.$/, "srv.notfound"],
  [/^'(?<name>\w+)' (?<lo>\S+) ile (?<hi>\S+) arasında olmalı\.$/, "srv.range"],
  [/^'(?<name>\w+)' geçerli bir sayı değil\.$/, "srv.nan"],
];

function localizeServerMessage(message) {
  if (!message) return message;
  for (const [pattern, key] of SERVER_MESSAGES) {
    const match = message.match(pattern);
    if (match) return t(key, match.groups);
  }
  return message;
}

// =====================================================================
// Color scales & pixel painting
// =====================================================================
const LUTS = {};

function hexToRgb(hex) {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** Interpolates a list of hex stops into a 256-entry RGB lookup table. */
function buildLut(stops) {
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

function readLuts() {
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

function lutColor(name, value) {
  const lut = LUTS[name];
  const j = clamp(Math.round(value), 0, 255) * 3;
  return `rgb(${lut[j]}, ${lut[j + 1]}, ${lut[j + 2]})`;
}

/** Paints a w x h grid of values (0-255, or -1..1 when signed) onto a canvas. */
function paint(canvas, values, w, h, lutName, signed = false) {
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

function repaintAll() {
  for (const canvas of document.querySelectorAll("canvas")) {
    if (canvas._repaint) canvas._repaint();            // canvases with extra drawing on top
    else if (canvas._paint) paint(canvas, ...canvas._paint);
  }
}

// =====================================================================
// Tooltip
// =====================================================================
const tip = $("#tooltip");
let hoverHost = null;

/** rows: [{ value, label, color? }] — the value leads, the label follows. */
function showTip(x, y, title, rows = []) {
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

function hideTip() {
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

function attachGridTooltip(container, describe) {
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
function niceStep(range, count) {
  const raw = range / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * power;
}

function linearTicks(min, max, step, format) {
  const ticks = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-6; v += step) {
    ticks.push({ v: +v.toFixed(10), label: format(+v.toFixed(10)) });
  }
  return ticks;
}

const logFloor = (v) => { const b = 10 ** Math.floor(Math.log10(v)); return v >= 3 * b ? 3 * b : b; };
const logCeil = (v) => { const b = 10 ** Math.floor(Math.log10(v)); return v <= b ? b : v <= 3 * b ? 3 * b : 10 * b; };

function logTicks(min, max) {
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
function smooth(values, weight = 0.95) {
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
function lineChart(host, o) {
  host._render = () => lineChart(host, o);
  if (!host.clientWidth) return;              // hidden tab: draw when it becomes visible
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
function barPath(x, y, w, hgt, r = 4) {
  r = Math.min(r, w, hgt / 2);
  return `M${x},${y}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${y + hgt - r}A${r},${r} 0 0 1 ${x + w - r},${y + hgt}H${x}Z`;
}

function probabilityChart(host, probabilities, prediction) {
  host._render = () => probabilityChart(host, probabilities, prediction);
  if (!host.clientWidth) return;
  host._w = host.clientWidth;
  const W = host.clientWidth;
  const rowH = 24;
  const thick = 14;
  const m = { l: 24, r: 64, t: 2, b: 2 };
  const H = m.t + m.b + rowH * 10;
  const iw = W - m.l - m.r;
  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("a.probs") });

  probabilities.forEach((p, digit) => {
    const y = m.t + digit * rowH;
    const top = digit === prediction;
    const row = svg("g", { class: "bar-row", tabindex: 0, "aria-label": `${t("digit")} ${digit}: ${fmtPct(p * 100, 1)}` });
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

const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const host = entry.target;
    if (host._render && host.clientWidth && host.clientWidth !== host._w) host._render();
  }
});
document.querySelectorAll(".chart").forEach((chart) => resizeObserver.observe(chart));

// =====================================================================
// Training tab
// =====================================================================
const STAGES = ["loading", "training", "validating", "testing", "saving"];
const stateLabel = (state) => t(`state.${state}`);   // idle, running, done, stopped, error
const stageLabel = (stage) => t(`busy.${stage}`);    // what a running job is doing
const form = $("#train-form");
const seen = {};
let lastStatus = null;

function resetSeen(runId) {
  Object.assign(seen, { run: runId, charts: "", history: "", log: -1, samplesRun: null, samplesKey: undefined });
}

function renderStatus(st) {
  if (st.run_id !== seen.run) resetSeen(st.run_id);
  const epochs = st.config?.epochs ?? 0;
  const losses = st.batch_losses;
  const ema = smooth(losses);
  const last = st.history[st.history.length - 1];

  // Header badges
  $("#device-badge").textContent = st.device;
  renderModelBadge(st.model);

  // Form: opens with the SimpleCNN.py defaults; only a run in progress shows its own settings
  if (!seen.formFilled) {
    const values = st.running ? st.config : st.defaults;
    for (const key of ["epochs", "lr", "batch_size", "seed"]) form.elements[key].value = values[key];
    seen.formFilled = true;
  }
  for (const input of form.querySelectorAll("input")) input.disabled = st.running;
  $("#btn-start").disabled = st.running;
  $("#btn-start").textContent = t(st.running ? "btn.running" : st.run_id ? "btn.retrain" : "btn.start");
  $("#btn-stop").disabled = !st.running;

  // Stage tracker
  const pill = $("#state-pill");
  pill.textContent = stateLabel(st.state);
  pill.className = `state-pill ${st.state}`;
  const current = st.stage === "done" ? STAGES.length : STAGES.indexOf(st.stage);
  const hints = {
    loading: st.dataset_sizes
      ? `${fmtInt(st.dataset_sizes.train)} / ${fmtInt(st.dataset_sizes.validation)} / ${fmtInt(st.dataset_sizes.test)}`
      : t("hint.loading"),
    training: st.epoch ? `Epoch ${st.epoch}/${epochs} · batch ${st.batch}/${st.batches}` : t("hint.training"),
    validating: last ? t("hint.lastval", { v: fmtPct(last.val_acc) }) : t("hint.validating"),
    testing: st.test ? t("hint.acc", { v: fmtPct(st.test.acc) }) : t("hint.testing"),
    saving: "checkpoints/simple_cnn.pt",
  };
  for (const li of $("#stepper").children) {
    const i = STAGES.indexOf(li.dataset.stage);
    let cls = "pending";
    if (st.state === "done") cls = "done";
    else if (current >= 0 && i < current) cls = "done";
    else if (i === current) cls = st.state === "running" ? "active pulse" : "halted";
    li.className = cls;
    li.querySelector("[data-hint]").textContent = hints[li.dataset.stage];
  }

  // Progress bars
  const batches = st.batches || 0;
  const inEpoch = st.stage === "training" ? (batches ? st.batch / batches : 0) : (st.epoch ? 1 : 0);
  let total = epochs && batches ? ((st.epoch - 1) * batches + (st.stage === "training" ? st.batch : batches)) / (epochs * batches) : 0;
  if (st.state === "done") total = 1;
  total = clamp(total, 0, 1);
  const eta = st.running && total > 0.02 ? (st.elapsed * (1 - total)) / total : null;
  $("#epoch-progress").style.width = `${inEpoch * 100}%`;
  $("#epoch-progress-label").textContent = st.epoch ? `Epoch ${st.epoch}/${epochs}` : "Epoch";
  $("#epoch-progress-value").textContent = batches ? `${st.stage === "training" ? st.batch : batches}/${batches} batch` : "—";
  $("#total-progress").style.width = `${total * 100}%`;
  $("#total-progress-value").textContent = st.run_id ? `${fmtPct(total * 100, 0)}${eta != null ? ` · ${t("eta", { t: fmtDuration(eta) })}` : ""}` : "—";
  $("#epoch-meter").setAttribute("aria-valuenow", Math.round(inEpoch * 100));
  $("#total-meter").setAttribute("aria-valuenow", Math.round(total * 100));
  $("#train-loop").classList.toggle("running", st.running && st.stage === "training");
  $("#step-count").textContent = t("steps.total", { n: fmtInt(st.step) });

  // Stat tiles
  $("#tile-epoch").textContent = st.epoch ? `${st.epoch} / ${epochs}` : "—";
  $("#tile-epoch-sub").textContent = st.running ? stageLabel(st.stage) : st.run_id ? stateLabel(st.state).toLocaleLowerCase(locale()) : t("tile.notstarted");
  $("#tile-loss").textContent = ema.length ? fmtNum(ema[ema.length - 1], 4) : "—";
  $("#tile-val").textContent = last ? fmtPct(last.val_acc) : "—";
  if (st.history.length) {
    const best = st.history.reduce((a, b) => (b.val_acc > a.val_acc ? b : a));
    $("#tile-val-sub").textContent = t("tile.best", { v: fmtPct(best.val_acc), e: best.epoch });
  } else {
    $("#tile-val-sub").textContent = t("tile.everyepoch");
  }
  $("#tile-test").textContent = st.test ? fmtPct(st.test.acc) : "—";
  $("#tile-test-sub").textContent = st.test ? t("tile.loss", { v: fmtNum(st.test.loss, 4) }) : t("tile.atend");
  $("#tile-time").textContent = st.elapsed != null ? fmtDuration(st.elapsed) : "—";
  $("#tile-time-sub").textContent = eta != null ? t("eta", { t: fmtDuration(eta) }) : st.state === "done" ? t("tile.totaltime") : " ";

  renderRunBanner(st);
  renderCharts(st, ema);
  renderFilters(st);
  renderSamples(st);
  renderHistory(st);
  renderLog(st);
  renderModelState(st);
}

function renderModelBadge(model) {
  const badge = $("#model-badge");
  badge.classList.toggle("ready", model.ready);
  let text = t("model.none");
  if (model.ready) {
    text = modelSourceText(model);
    if (model.meta?.test_acc != null) text += ` · test ${fmtPct(model.meta.test_acc)}`;
  }
  $("#model-badge-text").textContent = text;
  badge.title = model.meta?.trained_at
    ? t("model.trainedat", { d: new Date(model.meta.trained_at).toLocaleString(locale()), n: model.meta.epochs })
    : t("model.used");
}

function renderRunBanner(st) {
  const banner = $("#run-banner");
  const key = `${st.run_id}|${st.state}|${I18N.lang}`;
  if (seen.banner === key) return;
  seen.banner = key;
  banner.className = "banner inline";
  banner.replaceChildren();
  if (st.state === "done" && st.test) {
    banner.classList.add("good");
    const go = h("button", { type: "button", class: "btn small", text: t("run.try") });
    go.addEventListener("click", () => selectTab("infer"));
    banner.append(
      h("span", { class: "icon", text: "✓" }),
      h("span", { text: t("run.done", { v: fmtPct(st.test.acc) }) }),
      go,
    );
    go.style.marginLeft = "auto";
  } else if (st.state === "stopped" || st.state === "error") {
    banner.classList.add(st.state);
    banner.append(h("span", { class: "icon", text: "!" }), h("span", { text: st.state === "stopped" ? t("run.stopped") : st.message }));
    if (st.state === "stopped" && st.history.length) {
      banner.append(h("span", { text: " " + t("run.stoppedmodel", { e: st.history.length }) }));
    }
  }
  banner.hidden = !banner.childElementCount;
}

function renderCharts(st, ema) {
  const key = `${st.run_id}|${st.batch_losses.length}|${st.history.length}|${st.batches}|${st.config?.epochs}`;
  if (key === seen.charts) return;
  seen.charts = key;
  const epochs = st.config?.epochs || 10;
  const batches = st.batches;

  // --- Batch loss (log scale), x axis in steps, labelled by epoch
  const raw = st.batch_losses.map((v, i) => [i + 1, v]);
  const avg = ema.map((v, i) => [i + 1, v]);
  const values = st.batch_losses.filter((v) => v > 0);
  const yMin = values.length ? logFloor(Math.min(...values)) : 0.01;
  const yMax = values.length ? logCeil(Math.max(...values)) : 3;
  const every = Math.ceil(epochs / 12);
  const xTicks = [];
  const xGrid = [];
  if (batches) {
    for (let e = 1; e <= epochs; e++) {
      if ((e - 1) % every === 0) xTicks.push({ v: (e - 0.5) * batches, label: String(e) });
      if (e < epochs) xGrid.push(e * batches + 0.5);
    }
  }
  lineChart($("#chart-batch"), {
    label: t("chart.batch.label"),
    empty: t("chart.batch.empty"),
    x: { min: 0.5, max: batches ? epochs * batches + 0.5 : 1, ticks: xTicks, grid: xGrid },
    y: { log: true, min: yMin, max: yMax, ticks: logTicks(yMin, yMax) },
    series: [
      { name: t("raw"), points: raw, color: "var(--raw-line)", width: 1, hoverDot: false },
      { name: t("avg"), points: avg, color: "var(--series-1)", width: 2 },
    ],
    tooltip: (i) => ({
      title: t("chart.batch.tip", { s: fmtInt(i + 1), e: Math.floor(i / batches) + 1, b: (i % batches) + 1 }),
      rows: [
        { value: fmtNum(avg[i][1], 4), label: t("avg").toLocaleLowerCase(locale()), color: "var(--series-1)" },
        { value: fmtNum(raw[i][1], 4), label: t("raw").toLocaleLowerCase(locale()), color: "var(--raw-line)" },
      ],
    }),
  });

  // --- Epoch-level loss and accuracy
  const history = st.history;
  const epochTicks = [];
  const tickEvery = Math.ceil(epochs / 12);
  for (let e = 1; e <= epochs; e++) if ((e - 1) % tickEvery === 0 || e === epochs) epochTicks.push({ v: e, label: String(e) });
  const xEpoch = { min: epochs > 1 ? 0.6 : 0.5, max: epochs > 1 ? epochs + 0.4 : 1.5, ticks: epochTicks };
  const pointsOf = (key) => history.map((row) => [row.epoch, row[key]]);
  const epochTooltip = (keyTrain, keyVal, format) => (i) => ({
    title: `Epoch ${history[i].epoch}`,
    rows: [
      { value: format(history[i][keyTrain]), label: t("series.train").toLocaleLowerCase(locale()), color: "var(--series-1)" },
      { value: format(history[i][keyVal]), label: t("series.val").toLocaleLowerCase(locale()), color: "var(--series-2)" },
    ],
  });

  const lossMax = history.length ? Math.max(...history.flatMap((r) => [r.train_loss, r.val_loss])) : 0.3;
  const lossStep = niceStep(lossMax, 4);
  const lossTop = Math.ceil(lossMax / lossStep) * lossStep;
  lineChart($("#chart-loss"), {
    label: t("chart.loss.label"),
    empty: t("chart.firstepoch"),
    x: xEpoch,
    y: { min: 0, max: lossTop, ticks: linearTicks(0, lossTop, lossStep, (v) => fmtTick(v)) },
    series: [
      { name: t("series.train"), points: pointsOf("train_loss"), color: "var(--series-1)", markers: true },
      { name: t("series.val"), points: pointsOf("val_loss"), color: "var(--series-2)", markers: true },
    ],
    tooltip: epochTooltip("train_loss", "val_loss", (v) => fmtNum(v, 4)),
  });

  const accMin = history.length ? Math.min(...history.flatMap((r) => [r.train_acc, r.val_acc])) : 90;
  const accStep = niceStep(100 - accMin + 0.5, 4);
  const accBottom = Math.max(0, Math.floor((accMin - 0.25) / accStep) * accStep);
  lineChart($("#chart-acc"), {
    label: t("chart.acc.label"),
    empty: t("chart.firstepoch"),
    x: xEpoch,
    y: { min: accBottom, max: 100, ticks: linearTicks(accBottom, 100, accStep, (v) => pctText(fmtTick(v))) },
    series: [
      { name: t("series.train"), points: pointsOf("train_acc"), color: "var(--series-1)", markers: true },
      { name: t("series.val"), points: pointsOf("val_acc"), color: "var(--series-2)", markers: true },
    ],
    tooltip: epochTooltip("train_acc", "val_acc", (v) => fmtPct(v)),
  });
}

function renderFilters(st) {
  const grid = $("#filters");
  if (!st.filters) {
    grid.replaceChildren();
    return;
  }
  if (grid.childElementCount !== st.filters.length) {
    grid.replaceChildren(...st.filters.map((_, i) => h("canvas", { class: "px", "data-i": i })));
  }
  st.filters.forEach((weights, i) => paint(grid.children[i], weights, 5, 5, "div", true));
}

function renderSamples(st) {
  const grid = $("#samples");
  const sub = $("#samples-sub");
  if (!st.samples) {
    grid.replaceChildren();
    sub.textContent = t("samples.empty");
    return;
  }
  if (seen.samplesRun !== st.run_id) {
    seen.samplesRun = st.run_id;
    seen.samplesKey = undefined;
    grid.replaceChildren(...st.samples.images.map((pixels, i) => {
      const canvas = h("canvas", { class: "px" });
      paint(canvas, pixels, 28, 28, "gray");
      return h("figure", { class: "sample", "data-i": i }, canvas, h("figcaption"));
    }));
  }
  const sp = st.sample_predictions;
  const key = sp ? sp.epoch : null;
  if (key === seen.samplesKey) return;
  seen.samplesKey = key;
  if (!sp) return;
  [...grid.children].forEach((fig, i) => {
    const label = st.samples.labels[i];
    const pred = sp.predictions[i];
    const ok = pred === label;
    fig.className = `sample ${ok ? "ok" : "bad"}`;
    fig._info = { label, pred, confidence: sp.confidence[i] };
    const caption = fig.querySelector("figcaption");
    caption.replaceChildren(h("span", { class: "mark", text: ok ? "✓" : "✗" }), ` ${pred}${ok ? "" : ` (${label})`}`);
  });
  sub.textContent = t("samples.sub", {
    when: sp.epoch === 0 ? t("samples.before") : t("samples.after", { e: sp.epoch }),
    c: sp.correct, n: sp.predictions.length,
  });
}

function renderHistory(st) {
  const key = `${st.run_id}|${st.history.length}`;
  if (key === seen.history) return;
  seen.history = key;
  const body = $("#history-table tbody");
  if (!st.history.length) {
    body.replaceChildren(h("tr", { class: "empty" }, h("td", { colspan: 6, text: t("nodata") })));
    return;
  }
  const best = st.history.reduce((a, b) => (b.val_acc > a.val_acc ? b : a));
  body.replaceChildren(...st.history.map((r) => h("tr", { class: r === best ? "best" : null },
    h("td", { text: String(r.epoch) }),
    h("td", { text: fmtNum(r.train_loss, 4) }),
    h("td", { text: fmtPct(r.train_acc) }),
    h("td", { text: fmtNum(r.val_loss, 4) }),
    h("td", { text: fmtPct(r.val_acc) }),
    h("td", { text: fmtDuration(r.time) }),
  )));
}

function renderLog(st) {
  if (st.log.length === seen.log) return;
  seen.log = st.log.length;
  const log = $("#log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  log.textContent = st.log.length ? st.log.join("\n") : t("log.empty");
  if (atBottom) log.scrollTop = log.scrollHeight;
}

attachGridTooltip($("#filters"), (canvas) => ({
  title: t("filter.n", { n: +canvas.dataset.i + 1 }),
  rows: [{ value: "5×5", label: t("filter.tip") }],
}));
attachGridTooltip($("#samples"), (node) => {
  const fig = node.closest("figure");
  if (!fig?._info) return null;
  const { label, pred, confidence } = fig._info;
  return {
    title: t("valimg.n", { n: +fig.dataset.i + 1 }),
    rows: [
      { value: String(pred), label: t("valimg.pred", { v: fmtPct(confidence * 100, 1) }) },
      { value: String(label), label: t("truelabel") },
    ],
  };
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const msg = $("#form-msg");
  msg.textContent = "";
  const config = {
    epochs: Number(form.elements.epochs.value),
    lr: Number(form.elements.lr.value),
    batch_size: Number(form.elements.batch_size.value),
    seed: Number(form.elements.seed.value),
  };
  try {
    await api("/api/train", config);
    poll();
  } catch (error) {
    msg.textContent = error.message;
  }
});

$("#btn-stop").addEventListener("click", async () => {
  try {
    await api("/api/stop", {});
  } finally {
    poll();
  }
});

// =====================================================================
// Which model the Inference and Theory tabs show
// "trained" or "initial": the random weights training starts from, to show what training changes
// =====================================================================
const modelView = { which: "trained" };
const REQUIRED_SERVER_VERSION = 3;   // matches SERVER_VERSION in dashboard.py

/** The page files are newer than the running server (it was started before an update). */
function showOutdatedServer() {
  $("#outdated-banner").hidden = false;
}
const modelLabel = (which) => (which ? t(`model.${which}`) : "");

/** The server names the model in Turkish ("Kayıtlı model", "Epoch 3/10 sonrası…"); show it in the page's language. */
function modelSourceText(model) {
  const source = model.source || "";
  if (source === "Kayıtlı model") return t("src.saved");
  if (source === "Yeni eğitilen model") return t("src.new");
  const epoch = source.match(/^Epoch (\d+)\/(\d+)/);
  return epoch ? t("src.epoch", { e: epoch[1], n: epoch[2] }) : source;
}

/** The initial model always exists; the trained one only after a training run or a saved checkpoint. */
const modelUsable = () => modelView.which === "initial" || Boolean(lastStatus?.model.ready);

function setModelView(which) {
  if (which === modelView.which) return;
  modelView.which = which;
  for (const b of document.querySelectorAll(".model-switch button")) {
    b.setAttribute("aria-pressed", String(b.dataset.model === which));
  }
  if (lastStatus) renderModelState(lastStatus);
  document.dispatchEvent(new CustomEvent("model-view-change", { detail: which }));
}

function describeModels(model) {
  const text = modelView.which === "initial"
    ? t("model.initial.desc", { s: model.initial_seed ?? "—" })
    : model.ready
      ? `${modelSourceText(model)}${model.meta?.test_acc != null ? ` · ${t("model.testacc", { v: fmtPct(model.meta.test_acc) })}` : ""}`
      : t("model.notrained");
  for (const el of document.querySelectorAll("[data-model-desc]")) el.textContent = text;
}

document.addEventListener("click", (event) => {
  const button = event.target.closest?.(".model-switch button");
  if (button) setModelView(button.dataset.model);
});

// Switching the model re-runs the current input, so both versions can be compared directly
document.addEventListener("model-view-change", () => {
  if (!$("#panel-infer").hidden) syncInference();
  if (infer.last && modelUsable()) requestPredict(infer.last);
});

// =====================================================================
// Inference tab
// =====================================================================
const LAYERS = [
  { key: "input", title: { tr: "Giriş", en: "Input" }, shape: "1 × 28 × 28",
    desc: { tr: "Gri tonlu görüntü; her piksel 0 ile 1 arasında.", en: "A grayscale image; every pixel is between 0 and 1." } },
  { key: "conv1", short: "Conv1", title: "Conv1 → BatchNorm → ReLU", cell: 48,
    desc: { tr: "32 adet 5×5 filtre görüntüyü tarar; her filtre bir öznitelik haritası üretir (kenar, köşe, çizgi gibi).",
      en: "32 filters of 5×5 scan the image; each filter produces one feature map (edges, corners, strokes…)." } },
  { key: "pool1", short: "MaxPool 1", title: "MaxPool 2×2", cell: 36,
    desc: { tr: "Her 2×2 bölgenin en büyük değeri alınır, boyut yarıya iner. Dropout yalnızca eğitimde çalışır.",
      en: "The largest value of every 2×2 region is kept, halving the size. Dropout only runs during training." } },
  { key: "conv2", short: "Conv2", title: "Conv2 → BatchNorm → ReLU", cell: 32,
    desc: { tr: "64 filtre, önceki 32 haritayı birleştirerek daha karmaşık desenler arar.",
      en: "64 filters combine the previous 32 maps to look for more complex patterns." } },
  { key: "pool2", short: "MaxPool 2", title: "MaxPool 2×2 → Flatten", cell: 26,
    desc: { tr: "64 × 4 × 4 = 1024 değer tek bir vektöre düzleştirilip tam bağlı katmana verilir.",
      en: "64 × 4 × 4 = 1024 values are flattened into a single vector for the fully connected layer." } },
  { key: "fc1", title: "FC1 (1024 → 512) → ReLU", shape: "512",
    desc: { tr: "Tam bağlı katman. 512 nöron, 32 × 16 ızgara olarak gösteriliyor.",
      en: "A fully connected layer. Its 512 neurons are shown as a 32 × 16 grid." } },
  { key: "out", title: "FC2 (512 → 10) → Softmax", shape: "10",
    desc: { tr: "Her rakam için bir skor; Softmax bu skorları toplamı 1 olan olasılıklara çevirir.",
      en: "One score per digit; softmax turns the scores into probabilities that sum to 1." } },
];
const layerRows = {};
const pad = $("#draw");
const padCtx = pad.getContext("2d");
const infer = { busy: false, queued: null, last: null, lastSent: 0, inked: false, modelVersion: null, samplesRequested: false, result: null };

function buildLayerRows() {
  const host = $("#layers");
  for (const layer of LAYERS) {
    const shape = h("span", { class: "shape", text: layer.shape || "" });
    const note = h("p", { text: L(layer.desc) });
    const body = h("div", { class: "maps" }, h("p", { class: "placeholder", text: t("layers.placeholder") }));
    if (layer.cell) body.style.setProperty("--cell", `${layer.cell}px`);
    host.append(h("div", { class: "layer-row" }, h("div", { class: "layer-meta" }, h("h3", { text: L(layer.title) }), shape, note), body));
    layerRows[layer.key] = { body, shape };
    if (layer.cell) {
      attachGridTooltip(body, (canvas) => ({
        title: `${layer.short} · ${t("map.n", { n: +canvas.dataset.i + 1 })}`,
        rows: [{ value: fmtNum(body._peaks?.[+canvas.dataset.i] ?? 0, 2), label: t("map.peak") }],
      }));
    }
  }
  attachGridTooltip(layerRows.fc1.body, (canvas, event) => {
    if (!canvas._paint) return null;
    const box = canvas.getBoundingClientRect();
    const col = clamp(Math.floor(((event.clientX - box.left) / box.width) * 32), 0, 31);
    const row = clamp(Math.floor(((event.clientY - box.top) / box.height) * 16), 0, 15);
    const index = row * 32 + col;
    return { title: `FC1 · ${t("neuron.n", { n: index + 1 })}`, rows: [{ value: fmtPct((canvas._paint[0][index] / 255) * 100, 0), label: t("neuron.rel") }] };
  });
  attachGridTooltip(layerRows.input.body, () => ({ title: t("input.image"), rows: [{ value: "28×28", label: t("pixels") }] }));
}

function ensureCanvases(container, count, cls) {
  if (container.querySelectorAll("canvas").length !== count) {
    container.replaceChildren(...Array.from({ length: count }, (_, i) => h("canvas", { class: `px ${cls || ""}`.trim(), "data-i": i })));
  }
  return container.querySelectorAll("canvas");
}

function renderLayers(result) {
  ensureCanvases(layerRows.input.body, 1, "input").forEach((c) => paint(c, result.input, 28, 28, "gray"));

  for (const layer of result.layers) {
    const [channels, height, width] = layer.shape;
    const row = layerRows[layer.key];
    row.shape.textContent = `${channels} × ${height} × ${width}`;
    row.body._peaks = layer.peaks;
    ensureCanvases(row.body, channels).forEach((c, i) => paint(c, layer.maps[i], width, height, "seq"));
  }

  const fc = layerRows.fc1;
  ensureCanvases(fc.body, 1, "wide").forEach((c) => paint(c, result.fc1.values, 32, 16, "seq"));
  fc.shape.textContent = `512 · ${t("neurons.active", { n: result.fc1.active })}`;

  const out = layerRows.out.body;
  out.replaceChildren(h("div", { class: "out-cells" }, ...result.probabilities.map((p, digit) => {
    const cell = h("div", { class: `out-cell${digit === result.prediction ? " top" : ""}`, "data-i": digit },
      h("span", { class: "out-swatch", style: `background:${lutColor("seq", p * 255)}` }),
      h("span", { class: "out-digit", text: String(digit) }),
      h("span", { class: "out-pct", text: fmtProb(p) }));
    return cell;
  })));
}

function renderPrediction(result, input) {
  infer.result = result;
  infer.shownInput = input;   // to redraw in another language
  const confidence = result.probabilities[result.prediction];
  $("#pred-digit").textContent = String(result.prediction);
  $("#pred-conf").textContent = t("pred.conf", { v: fmtPct(confidence * 100, 1), d: result.prediction });
  const source = $("#pred-source");
  if (input.kind === "test") {
    const ok = result.prediction === result.label;
    source.replaceChildren(
      `${t("pred.test", { i: input.index, l: result.label })} · `,
      h("span", { class: ok ? "ok" : "bad", text: ok ? t("pred.ok") : t("pred.bad") }),
    );
  } else {
    source.textContent = t("pred.drawing");
  }
  const tag = $("#pred-model");
  tag.hidden = false;
  tag.textContent = modelLabel(result.model);
  tag.classList.toggle("initial", result.model === "initial");
  paint($("#input-view"), result.input, 28, 28, "gray");
  probabilityChart($("#prob-chart"), result.probabilities, result.prediction);
  renderLayers(result);
}

function resetPrediction(message = t("pred.prompt")) {
  infer.shownInput = null;
  $("#pred-digit").textContent = "–";
  $("#pred-conf").textContent = message;
  $("#pred-source").textContent = "";
}

async function sendPredict(input) {
  infer.busy = true;
  infer.lastSent = performance.now();
  try {
    const body = input.kind === "draw" ? { image: pad.toDataURL("image/png") } : { test_index: input.index };
    body.model = modelView.which;
    const result = await api("/api/predict", body);
    if (result.model !== body.model) showOutdatedServer();   // an old server ignores the model choice
    renderPrediction(result, input);
  } catch (error) {
    resetPrediction(error.message);
  } finally {
    infer.busy = false;
    if (infer.queued) {
      const next = infer.queued;
      infer.queued = null;
      sendPredict(next);
    }
  }
}

/** Sends a prediction request; while one is in flight only the latest input is kept. */
function requestPredict(input, throttled = false) {
  infer.last = input;
  if (input.kind === "draw" && !infer.inked) return;
  if (throttled && performance.now() - infer.lastSent < 120) return;
  if (infer.busy) {
    infer.queued = input;
    return;
  }
  sendPredict(input);
}

function clearPad() {
  padCtx.fillStyle = "#000";
  padCtx.fillRect(0, 0, pad.width, pad.height);
  infer.inked = false;
}

function padPoint(event) {
  const box = pad.getBoundingClientRect();
  return [((event.clientX - box.left) * pad.width) / box.width, ((event.clientY - box.top) * pad.height) / box.height];
}

function strokeTo(from, to) {
  padCtx.strokeStyle = "#fff";
  padCtx.lineWidth = 20;
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.beginPath();
  padCtx.moveTo(...from);
  padCtx.lineTo(...to);
  padCtx.stroke();
  infer.inked = true;
}

let drawing = false;
let lastPoint = null;
pad.addEventListener("pointerdown", (event) => {
  if (!modelUsable()) return;
  pad.setPointerCapture(event.pointerId);
  if (infer.last?.kind === "test") clearPad();  // switching from a test image to drawing
  for (const node of $("#test-samples").children) node.classList.remove("selected");
  drawing = true;
  lastPoint = padPoint(event);
  strokeTo(lastPoint, lastPoint);
  infer.last = { kind: "draw" };
});
pad.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const point = padPoint(event);
  strokeTo(lastPoint, point);
  lastPoint = point;
  requestPredict({ kind: "draw" }, true);
});
const endStroke = () => {
  if (!drawing) return;
  drawing = false;
  requestPredict({ kind: "draw" });
};
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

$("#btn-clear").addEventListener("click", () => {
  clearPad();
  infer.last = { kind: "draw" };
  for (const node of $("#test-samples").children) node.classList.remove("selected");
  resetPrediction();
});

async function loadTestSamples(autoPick) {
  const grid = $("#test-samples");
  try {
    const { samples } = await api("/api/test-samples?n=12");
    grid.replaceChildren(...samples.map((sample) => {
      const canvas = h("canvas", { class: "px" });
      paint(canvas, sample.pixels, 28, 28, "gray");
      const button = h("button", { type: "button", class: "sample", "data-label": sample.label, "aria-label": t("sample.aria", { l: sample.label }) },
        canvas, h("span", { class: "caption", text: t("sample.label", { l: sample.label }) }));
      button.addEventListener("click", () => pickSample(sample, button));
      return button;
    }));
    if (autoPick && samples.length) pickSample(samples[0], grid.firstElementChild);
  } catch (error) {
    grid.replaceChildren(h("p", { class: "placeholder", text: error.message }));
  }
}

function pickSample(sample, button) {
  for (const node of $("#test-samples").children) node.classList.toggle("selected", node === button);
  clearPad();
  requestPredict({ kind: "test", index: sample.index, label: sample.label });
}

$("#btn-more").addEventListener("click", () => loadTestSamples(false));

/** Loads examples on first visit and re-runs the last input when a newer model arrives. */
function syncInference() {
  const model = lastStatus?.model;
  if (!modelUsable() || $("#panel-infer").hidden) return;
  if (!infer.samplesRequested) {
    infer.samplesRequested = true;
    loadTestSamples(!infer.last);
  }
  if (model.version !== infer.modelVersion) {
    const changed = infer.modelVersion !== null;
    infer.modelVersion = model.version;
    if (changed && infer.last) requestPredict(infer.last);
  }
}

function renderModelState(st) {
  $("#no-model").hidden = st.model.ready;
  $("#panel-infer").classList.toggle("dim", !modelUsable());
  describeModels(st.model);
  syncInference();
  document.dispatchEvent(new CustomEvent("model-status", { detail: st.model }));
}

// =====================================================================
// Tabs, polling, start-up
// =====================================================================
const TABS = {
  train: { tab: "#tab-train", panel: "#panel-train", hash: "#egitim" },
  infer: { tab: "#tab-infer", panel: "#panel-infer", hash: "#cikarim" },
  theory: { tab: "#tab-theory", panel: "#panel-theory", hash: "#teori" },
  code: { tab: "#tab-code", panel: "#panel-code", hash: "#kodlama" },
};

function selectTab(name) {
  document.body.dataset.tab = name;   // CSS shows the floating model switch only where it applies
  for (const [key, tab] of Object.entries(TABS)) {
    $(tab.tab).setAttribute("aria-selected", String(key === name));
    $(tab.panel).hidden = key !== name;
  }
  history.replaceState(null, "", TABS[name].hash);
  hideTip();
  if (name === "infer") syncInference();
  document.dispatchEvent(new CustomEvent("tab-shown", { detail: name }));
}

for (const [key, tab] of Object.entries(TABS)) $(tab.tab).addEventListener("click", () => selectTab(key));

let pollTimer = null;
let polling = false;
let pollAgain = false;

async function poll() {
  if (polling) {
    pollAgain = true;
    return;
  }
  polling = true;
  clearTimeout(pollTimer);
  let delay = 2000;
  try {
    const status = await api("/api/status");
    $("#conn-banner").hidden = true;
    if ((status.server_version ?? 0) < REQUIRED_SERVER_VERSION) showOutdatedServer();
    lastStatus = status;
    renderStatus(status);
    delay = status.running ? 400 : 2000;
  } catch (error) {
    $("#conn-banner").hidden = false;
  } finally {
    polling = false;
    if (pollAgain) {
      pollAgain = false;
      delay = 0;
    }
    pollTimer = setTimeout(poll, delay);
  }
}

// =====================================================================
// Zoom: click a map, filter or image to see it enlarged
// =====================================================================
const zoom = { dialog: $("#zoom"), canvas: $("#zoom-canvas"), source: null, peak: null };

/** Canvases whose click already does something else are not zoomable. */
function isZoomable(canvas) {
  return canvas._paint && canvas !== zoom.canvas
    && !canvas.closest(".filter-strip, #test-samples, .theory-bar, .draw-card");
}

function zoomInfo(canvas) {
  const i = Number(canvas.dataset.i ?? 0);
  const row = canvas.closest(".layer-row");
  if (row) {
    const count = row.querySelectorAll("canvas").length;
    return { title: row.querySelector("h3").textContent + (count > 1 ? ` · ${t("map.n", { n: i + 1 })}` : ""), sub: row.querySelector(".shape").textContent };
  }
  if (canvas.closest("#filters")) return { title: t("filter.n", { n: i + 1 }), sub: t("zoom.filter.sub") };
  const sample = canvas.closest("#samples figure");
  if (sample?._info) {
    const { label, pred, confidence } = sample._info;
    return { title: t("valimg.n", { n: Number(sample.dataset.i) + 1 }), sub: t("zoom.sample.sub", { p: pred, v: fmtPct(confidence * 100, 1), l: label }) };
  }
  const dropRow = canvas.closest(".drop-row");
  if (dropRow) return { title: `Dropout · ${canvas.title}`, sub: `${dropRow.querySelector(".drop-label").textContent} · 12×12` };
  if (canvas.id === "input-view") return { title: t("zoom.input"), sub: t("zoom.input.sub") };
  const caption = canvas.closest("figure")?.querySelector("figcaption");
  return { title: caption?.textContent || t("zoom.title"), sub: "" };
}

const zoomLegend = (lut) => ({
  div: [t("negative"), "diverging", t("positive")],
  seq: ["0", "sequential", t("highest")],
  drop: ["0", "sequential", t("zoom.droplegend")],
})[lut];

function openZoom(canvas) {
  const [values, w, hgt, lut, signedValues] = canvas._paint;
  zoom.source = canvas;
  const maps = canvas.closest(".maps");
  zoom.peak = maps?._peaks ? maps._peaks[Number(canvas.dataset.i)] : null;   // real activation scale, if known
  paint(zoom.canvas, values, w, hgt, lut, signedValues);
  const side = Math.min(480, window.innerWidth * 0.8, window.innerHeight * 0.6);
  const scale = side / Math.max(w, hgt);
  zoom.canvas.style.width = `${Math.round(w * scale)}px`;
  zoom.canvas.style.height = `${Math.round(hgt * scale)}px`;
  // Fixed width while open, so the changing readout text never resizes (and re-centers) the dialog
  zoom.dialog.style.width = `${Math.max(Math.round(w * scale), 400) + 40}px`;

  const info = zoomInfo(canvas);
  $("#zoom-title").textContent = info.title;
  $("#zoom-sub").textContent = [info.sub, `${w}×${hgt} ${t("pixels")}`].filter(Boolean).join(" · ");
  const legend = zoomLegend(lut);
  $("#zoom-legend").replaceChildren(...(legend
    ? [h("span", { text: legend[0] }), h("span", { class: `scale-bar ${legend[1]}` }), h("span", { text: legend[2] })]
    : [h("span", { text: t("zoom.gray") })]));
  $("#zoom-readout").textContent = t("zoom.hover");
  hideTip();
  zoom.dialog.showModal();
}

function zoomReadout(event) {
  const [values, w, hgt, lut, signedValues] = zoom.canvas._paint;
  const box = zoom.canvas.getBoundingClientRect();
  const c = clamp(Math.floor(((event.clientX - box.left) / box.width) * w), 0, w - 1);
  const r = clamp(Math.floor(((event.clientY - box.top) / box.height) * hgt), 0, hgt - 1);
  const v = values[r * w + c];
  let text;
  if (lut === "gray") text = t("zoom.pixel", { v: fmtNum(v / 255, 2) });
  else if (signedValues) text = t("zoom.scaled", { v: fmtNum(v, 2) });
  else if (lut === "drop" && v >= 254) text = t("zoom.dropped");
  else if (zoom.peak != null) text = t("zoom.act", { v: fmtNum((v / 255) * zoom.peak, 3) });
  else text = t("zoom.relative", { v: fmtPct((v / (lut === "drop" ? 253 : 255)) * 100, 0) });
  $("#zoom-readout").textContent = `${t("zoom.rc", { r: r + 1, c: c + 1 })} · ${text}`;
}

document.addEventListener("click", (event) => {
  const canvas = event.target.closest?.("canvas");
  if (canvas && isZoomable(canvas)) openZoom(canvas);
});
zoom.canvas.addEventListener("pointermove", zoomReadout);
$("#zoom-close").addEventListener("click", () => zoom.dialog.close());
zoom.dialog.addEventListener("click", (event) => { if (event.target === zoom.dialog) zoom.dialog.close(); });   // backdrop
// Live views (filters while training, inference after a new epoch) keep the zoomed copy current
setInterval(() => {
  if (zoom.dialog.open && zoom.source?._paint && zoom.source._paint[0] !== zoom.canvas._paint?.[0]) {
    paint(zoom.canvas, ...zoom.source._paint);
  }
}, 500);

// Theme: "light", "dark" or "system" (follows the OS). The choice is saved in this browser.
const systemDark = matchMedia("(prefers-color-scheme: dark)");
const themeChoice = () => document.documentElement.dataset.theme || "system";

function markThemeButtons() {
  for (const b of document.querySelectorAll("#theme-switch button")) {
    b.setAttribute("aria-pressed", String(b.dataset.themeChoice === themeChoice()));
  }
}

function applyTheme(choice) {
  if (choice === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
  try {
    if (choice === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", choice);
  } catch { /* storage blocked: the choice lasts until the page is reloaded */ }
  markThemeButtons();
  document.dispatchEvent(new CustomEvent("theme-change"));
}

for (const b of document.querySelectorAll("#theme-switch button")) {
  b.addEventListener("click", () => applyTheme(b.dataset.themeChoice));
}
systemDark.addEventListener("change", () => {
  if (themeChoice() === "system") document.dispatchEvent(new CustomEvent("theme-change"));
});
// Canvases keep their pixels, so repaint them with the new theme's color scales
document.addEventListener("theme-change", () => {
  readLuts();
  repaintAll();
  if (infer.result) renderLayers(infer.result);
});
markThemeButtons();

// Language switch: static text swaps in i18n.js; everything built here is redrawn
for (const b of document.querySelectorAll("#lang-switch button")) b.addEventListener("click", () => setLang(b.dataset.lang));
document.addEventListener("lang-change", () => {
  if (zoom.dialog.open) zoom.dialog.close();
  hideTip();
  Object.assign(seen, { charts: "", history: "", log: -1, samplesKey: undefined, banner: undefined });
  if (lastStatus) renderStatus(lastStatus);
  $("#layers").replaceChildren();
  buildLayerRows();
  if (infer.result && infer.shownInput) renderPrediction(infer.result, infer.shownInput);
  else resetPrediction();
  for (const b of $("#test-samples").querySelectorAll("button.sample")) {
    b.setAttribute("aria-label", t("sample.aria", { l: b.dataset.label }));
    b.querySelector(".caption").textContent = t("sample.label", { l: b.dataset.label });
  }
  $("#form-msg").textContent = "";
});

applyStaticTranslations();
readLuts();
buildLayerRows();
clearPad();
resetPrediction();
poll();
// Wait for theory.js (loaded after this file) so it hears which tab opens first
document.addEventListener("DOMContentLoaded", () => {
  selectTab(Object.keys(TABS).find((key) => TABS[key].hash === location.hash) || "train");
});
