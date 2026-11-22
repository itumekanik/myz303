/*
 * SimpleCNN dashboard (chapter 3): polls /api/status for training progress and renders it,
 * and sends drawings or test images to /api/predict to show what every layer produces.
 * The shared helpers (h, $, charts, navigation, theme) are in core.js.
 */

/**
 * The dashboard's server API (/api/status, /api/predict, …): sent to the Python server when
 * dashboard.py serves the page (training with PyTorch), else answered in the browser by
 * engines/cnn/backend.ts with the same JSON (training with TensorFlow.js).
 */
import { localApi } from "../../engines/cnn/backend.ts";
import { onServer } from "../../shared/engine.ts";
import { I18N, L, t } from "../../shared/i18n.js";
import { $, attachGridTooltip, clamp, fmtDuration, fmtInt, fmtNum, fmtPct, fmtProb, fmtTick, h, hideTip, lineChart, linearTicks, locale, logCeil, logFloor, logTicks, lutColor, niceStep, paint, pctText, probabilityChart, selectTab, smooth } from "../../shared/core.js";
import { TABS } from "../../shared/page.ts";

export async function api(path, body) {
  if (onServer) return serverApi(path, body);
  try {
    return await localApi(path, body);
  } catch (error) {
    throw new Error(localizeServerMessage(error.message) || error.message, { cause: error });
  }
}

async function serverApi(path, body) {
  const options = body === undefined
    ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(localizeServerMessage(data.error) || data.error || `HTTP ${response.status}`);
  return data;
}

/** The server answers in Turkish; known messages are shown in the page's language. */
export const SERVER_MESSAGES = [
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
  [/^Geçersiz istek\.$/, "srv.badrequest"],
  [/^GPU yanıt vermeyi bıraktı\.$/, "srv.gpulost"],
  [/^'(?<name>\w+)' (?<lo>\S+) ile (?<hi>\S+) arasında olmalı\.$/, "srv.range"],
  [/^'(?<name>\w+)' geçerli bir sayı değil\.$/, "srv.nan"],
];

export function localizeServerMessage(message) {
  if (!message) return message;
  for (const [pattern, key] of SERVER_MESSAGES) {
    const match = message.match(pattern);
    if (match) return t(key, match.groups);
  }
  return message;
}

// =====================================================================
// Training tab
// =====================================================================
export const STAGES = ["loading", "training", "validating", "testing", "saving"];
export const stateLabel = (state) => t(`state.${state}`);   // idle, running, done, stopped, error
export const stageLabel = (stage) => t(`busy.${stage}`);    // what a running job is doing
export const form = $("#train-form");
const editedFields = new Set();
form.addEventListener("input", (event) => editedFields.add(event.target.name));
export const seen = {};
export let lastStatus = null;

export function resetSeen(runId) {
  Object.assign(seen, { run: runId, charts: "", history: "", log: -1, samplesRun: null, samplesKey: undefined });
}

export function renderStatus(st) {
  if (st.run_id !== seen.run) resetSeen(st.run_id);
  const epochs = st.config?.epochs ?? 0;
  const losses = st.batch_losses;
  const ema = smooth(losses);
  const last = st.history[st.history.length - 1];

  // Header badges
  // "GPU · NVIDIA GeForce RTX 2060": narrow top bars show only "GPU", the full name is in the tooltip
  const [deviceKind, ...deviceName] = String(st.device).split(" · ");
  $("#device-badge").replaceChildren(h("span", { class: "badge-short", text: deviceKind }),
    deviceName.length ? h("span", { class: "badge-long", text: ` · ${deviceName.join(" · ")}` }) : "");
  $("#device-badge").title = `${L({ tr: "Eğitimin çalıştığı cihaz", en: "Device training runs on" })}: ${st.device}`;
  renderModelBadge(st.model);

  // Form: opens with the SimpleCNN.py defaults; only a run in progress shows its own settings.
  // A value the reader typed before this first status arrived stays.
  if (!seen.formFilled) {
    const values = st.running ? st.config : st.defaults;
    for (const key of ["epochs", "lr", "batch_size", "seed"]) {
      if (st.running || !editedFields.has(key)) form.elements[key].value = values[key];
    }
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
    saving: t(onServer ? "hint.saving.server" : "hint.saving"),
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

export function renderModelBadge(model) {
  const badge = $("#model-badge");
  badge.classList.toggle("ready", model.ready);
  // Narrow top bars keep only the accuracy; the model's source is in the long part
  if (model.ready && model.meta?.test_acc != null) {
    $("#model-badge-text").replaceChildren(h("span", { class: "badge-long", text: `${modelSourceText(model)} · ` }), `test ${fmtPct(model.meta.test_acc)}`);
  } else {
    $("#model-badge-text").textContent = model.ready ? modelSourceText(model) : t("model.none");
  }
  badge.title = model.meta?.trained_at
    ? t("model.trainedat", { d: new Date(model.meta.trained_at).toLocaleString(locale()), n: model.meta.epochs })
    : t("model.used");
}

export function renderRunBanner(st) {
  const banner = $("#run-banner");
  const key = `${st.run_id}|${st.state}|${I18N.lang}|${st.restored?.trained_at}|${st.model?.ready}`;
  if (seen.banner === key) return;
  seen.banner = key;
  banner.className = "banner inline";
  banner.replaceChildren();
  if (st.state === "done" && st.test) {
    banner.classList.add("good");
    const go = h("button", { type: "button", class: "btn small", text: t("run.try") });
    go.addEventListener("click", () => selectTab("infer"));
    const when = st.restored && new Date(st.restored.trained_at).toLocaleString(locale(), { dateStyle: "medium", timeStyle: "short" });
    // In the browser: the model that came with the site was trained with PyTorch; a run in this
    // browser replaces it until the reader goes back to it. On dashboard.py: the last run, saved in
    // checkpoints/, which the next run replaces.
    const fromBrowser = st.restored?.origin === "this browser";
    const v = fmtPct(st.test.acc);
    const text = onServer ? t(st.restored ? "run.restored" : "run.done.server", { d: when, v })
      : !st.restored ? t("run.done", { v })
        : fromBrowser ? t("run.restored.browser", { d: when, v })
          : t("run.restored.pytorch", { d: when, v, dev: st.model?.meta?.device ?? "GPU" });
    banner.append(h("span", { class: "icon", text: "✓" }), h("span", { text }), go);
    go.style.marginLeft = "auto";
    if (!onServer && (fromBrowser || !st.restored)) {
      const back = h("button", { type: "button", class: "btn small", text: t("run.reset") });
      back.addEventListener("click", async () => {
        back.disabled = true;
        try {
          await api("/api/reset", {});
        } catch (error) {
          back.disabled = false;
          back.title = error.message;
        }
        poll();
      });
      banner.append(back);
    }
  } else if (!st.run_id && !st.running && st.model?.ready) {
    // A saved model but no saved run: trained before runs were kept, so there is nothing to show yet
    banner.classList.add("info");
    banner.append(h("span", { class: "icon", text: "i" }), h("span", { text: t("run.nohistory") }));
  } else if (st.state === "stopped" || st.state === "error") {
    banner.classList.add(st.state);
    banner.append(h("span", { class: "icon", text: "!" }), h("span", { text: st.state === "stopped" ? t("run.stopped") : localizeServerMessage(st.message) }));
    if (st.state === "stopped" && st.history.length) {
      banner.append(h("span", { text: " " + t("run.stoppedmodel", { e: st.history.length }) }));
    }
  }
  banner.hidden = !banner.childElementCount;
}

export function renderCharts(st, ema) {
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

export function renderFilters(st) {
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

export function renderSamples(st) {
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

export function renderHistory(st) {
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

export function renderLog(st) {
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
export const modelView = { which: "trained" };

export const modelLabel = (which) => (which ? t(`model.${which}`) : "");

/** The server names the model in Turkish ("Kayıtlı model", "Epoch 3/10 sonrası…"); show it in the page's language. */
export function modelSourceText(model) {
  const source = model.source || "";
  if (source === "Kayıtlı model") return t("src.saved");
  if (source === "Yeni eğitilen model") return t("src.new");
  const partial = source.match(/^Epoch (\d+)\/(\d+) sonrası \(eğitim tamamlanmadı\)$/);
  if (partial) return t("src.partial", { e: partial[1], n: partial[2] });
  const epoch = source.match(/^Epoch (\d+)\/(\d+)/);
  return epoch ? t("src.epoch", { e: epoch[1], n: epoch[2] }) : source;
}

/** The initial model always exists; the trained one only after a training run or a saved checkpoint. */
export const modelUsable = () => modelView.which === "initial" || Boolean(lastStatus?.model.ready);

export function setModelView(which) {
  if (which === modelView.which) return;
  modelView.which = which;
  for (const b of document.querySelectorAll(".model-switch button")) {
    b.setAttribute("aria-pressed", String(b.dataset.model === which));
  }
  if (lastStatus) renderModelState(lastStatus);
  document.dispatchEvent(new CustomEvent("model-view-change", { detail: which }));
}

export function describeModels(model) {
  const text = modelView.which === "initial"
    ? t("model.initial.desc", { s: model.initial_seed ?? "—" })
    : model.ready
      ? `${modelSourceText(model)}${model.meta?.test_acc != null ? ` · ${t("model.testacc", { v: fmtPct(model.meta.test_acc) })}` : ""}`
      : t("model.notrained");
  for (const el of document.querySelectorAll("[data-model-desc]")) el.textContent = text;
  const float = document.querySelector(".model-float");
  if (float) float.title = `${L({ tr: "Gösterilen model", en: "Model shown" })}: ${text}`;
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
export const LAYERS = [
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
export const layerRows = {};
export const pad = $("#draw");
export const padCtx = pad.getContext("2d");
export const infer = { busy: false, queued: null, last: null, lastSent: 0, inked: false, modelVersion: null, samplesRequested: false, result: null, cleared: 0 };

export function buildLayerRows() {
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

export function ensureCanvases(container, count, cls) {
  if (container.querySelectorAll("canvas").length !== count) {
    container.replaceChildren(...Array.from({ length: count }, (_, i) => h("canvas", { class: `px ${cls || ""}`.trim(), "data-i": i })));
  }
  return container.querySelectorAll("canvas");
}

export function renderLayers(result) {
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

export function renderPrediction(result, input) {
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

export function resetPrediction(message = t("pred.prompt")) {
  infer.shownInput = null;
  $("#pred-digit").textContent = "–";
  $("#pred-conf").textContent = message;
  $("#pred-source").textContent = "";
}

export async function sendPredict(input) {
  infer.busy = true;
  infer.lastSent = performance.now();
  const cleared = infer.cleared; // a reply to a drawing erased in the meantime is ignored
  try {
    const body = input.kind === "draw" ? { image: pad.toDataURL("image/png") } : { test_index: input.index };
    body.model = modelView.which;
    const result = await api("/api/predict", body);
    if (cleared === infer.cleared) renderPrediction(result, input);
  } catch (error) {
    if (cleared === infer.cleared) resetPrediction(error.message);
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
export function requestPredict(input, throttled = false) {
  infer.last = input;
  if (input.kind === "draw" && !infer.inked) return;
  if (throttled && performance.now() - infer.lastSent < 120) return;
  if (infer.busy) {
    infer.queued = input;
    return;
  }
  sendPredict(input);
}

export function clearPad() {
  padCtx.fillStyle = "#000";
  padCtx.fillRect(0, 0, pad.width, pad.height);
  infer.inked = false;
}

export function padPoint(event) {
  const box = pad.getBoundingClientRect();
  return [((event.clientX - box.left) * pad.width) / box.width, ((event.clientY - box.top) * pad.height) / box.height];
}

export function strokeTo(from, to) {
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

export let drawing = false;
export let lastPoint = null;
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
export const endStroke = () => {
  if (!drawing) return;
  drawing = false;
  requestPredict({ kind: "draw" });
};
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

$("#btn-clear").addEventListener("click", () => {
  clearPad();
  infer.cleared += 1;
  infer.queued = null;
  infer.last = { kind: "draw" };
  for (const node of $("#test-samples").children) node.classList.remove("selected");
  resetPrediction();
});

export async function loadTestSamples(autoPick) {
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

export function pickSample(sample, button) {
  for (const node of $("#test-samples").children) node.classList.toggle("selected", node === button);
  clearPad();
  requestPredict({ kind: "test", index: sample.index, label: sample.label });
}

$("#btn-more").addEventListener("click", () => loadTestSamples(false));

/** Loads examples on first visit and re-runs the last input when a newer model arrives. */
export function syncInference() {
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

export function renderModelState(st) {
  $("#no-model").hidden = st.model.ready;
  $("#panel-infer").classList.toggle("dim", !modelUsable());
  describeModels(st.model);
  syncInference();
  document.dispatchEvent(new CustomEvent("model-status", { detail: st.model }));
}

// =====================================================================
// Tabs, polling, start-up
// =====================================================================
Object.assign(TABS, {
  train: { tab: "#tab-train", panel: "#panel-train", hash: "#egitim" },
  infer: { tab: "#tab-infer", panel: "#panel-infer", hash: "#cikarim" },
  theory: { tab: "#tab-theory", panel: "#panel-theory", hash: "#teori" },
  math: { tab: "#tab-math", panel: "#panel-math", hash: "#matematik" },
  code: { tab: "#tab-code", panel: "#panel-code", hash: "#kodlama" },
  practice: { tab: "#tab-practice", panel: "#panel-practice", hash: "#alistirma" },
});

// The CNN page's own reactions to a tab change
document.addEventListener("tab-shown", (event) => {
  if (zoom.dialog.open) zoom.dialog.close();
  if (event.detail === "infer") syncInference();
});

export let pollTimer = null;
export let polling = false;
export let pollAgain = false;

export async function poll() {
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
export const zoom = { dialog: $("#zoom"), canvas: $("#zoom-canvas"), source: null, peak: null };

/** Canvases whose click already does something else are not zoomable. */
export function isZoomable(canvas) {
  return canvas._paint && canvas !== zoom.canvas
    && !canvas.closest(".filter-strip, #test-samples, .theory-bar, .draw-card, .bn-maps");   // BatchNorm maps: a click picks a pixel
}

export function zoomInfo(canvas) {
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

export const zoomLegend = (lut) => ({
  div: [t("negative"), "diverging", t("positive")],
  seq: ["0", "sequential", t("highest")],
  drop: ["0", "sequential", t("zoom.droplegend")],
})[lut];

export function openZoom(canvas) {
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

export function zoomReadout(event) {
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

document.addEventListener("theme-change", () => {
  if (infer.result) renderLayers(infer.result);
});

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

buildLayerRows();
clearPad();
resetPrediction();
poll();
