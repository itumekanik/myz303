/**
 * The live demos of the YOLO Theory tab, one per section. They work on two validation tiles:
 * Shanghai_img209 (buildings, a river, four sports fields) and Paris_img637 (51 small houses).
 *
 * The demos that need the network show a button first: the model and its runtime are about 12 MB, so they are loaded
 * only when the reader asks. Once one demo has run the model, the others start by themselves.
 */
import { fmtNum, fmtPct, h, paint } from "../../shared/core.js";
import type { Text } from "../../shared/chapters.ts";
import { unpack } from "../../engines/data.ts";
import { aboveConfidence, anchorOf, boxIou, candidate, STRIDES, type Box, type Candidate } from "../../engines/yolo/detect.ts";
import type { Rgb } from "../../engines/yolo/image.ts";
import { areaSquareMetres, instanceMask, maskLogits, type InstanceMask } from "../../engines/yolo/masks.ts";
import { averagePrecision, matchPredictions, precisionRecall } from "../../engines/yolo/metrics.ts";
import { loadModel, objectsOf, runSegmentation, type InstanceObject, type SegmentationOutput } from "../../engines/yolo/model.ts";
import { maskIou } from "../../engines/yolo/raster.ts";
import {
  MATCH_RGB,
  SEG_CLASSES,
  SEG_RGB,
  SEM_CLASSES,
  SEM_RGB,
  T,
  TileView,
  classOverlay,
  css,
  decodeImage,
  loadSamples,
  maskOverlay,
  truthsOf,
  type Sample,
  type Truth,
} from "./page.ts";
import { epochViewer, loadRecord, type ShotChoice } from "./training.ts";

export const MAIN_TILE = "Shanghai_img209";
export const DENSE_TILE = "Paris_img637";

const nf = (v: number, d = 2) => fmtNum(v, d) as string;
const pct = (v: number, d = 0) => fmtPct(100 * v, d) as string;
/** A number inside TeX: a comma (Turkish decimal, English thousands) needs braces or TeX adds a space after it. */
const texNumber = (v: number, d = 2) => nf(v, d).replace(/,/g, "{,}");
const NC = SEG_CLASSES.length;

// ---------------------------------------------------------------------
// Tiles and model runs, shared by the demos
// ---------------------------------------------------------------------

interface Tile {
  sample: Sample;
  image: HTMLImageElement;
  rgb: Rgb;
  trueClasses: Uint8Array;
}

interface Run extends Tile {
  seg: SegmentationOutput;
  truths: Truth[];
}

const tiles = new Map<string, Promise<Tile>>();
const runs = new Map<string, Promise<Run>>();

function tileOf(name: string): Promise<Tile> {
  let t = tiles.get(name);
  if (!t) {
    t = (async () => {
      const set = await loadSamples();
      const sample = set.samples.find((s) => s.name === name);
      if (!sample) throw new Error(`no sample ${name}`);
      const { image, rgb } = await decodeImage(`data:image/jpeg;base64,${sample.jpeg}`);
      return { sample, image, rgb, trueClasses: await unpack(sample.landCover) };
    })();
    tiles.set(name, t);
  }
  return t;
}

function runOn(name: string): Promise<Run> {
  let r = runs.get(name);
  if (!r) {
    r = (async () => {
      const [tile, model] = await Promise.all([tileOf(name), loadModel("seg")]);
      const seg = await runSegmentation(model, tile.rgb);
      return { ...tile, seg, truths: truthsOf(tile.sample, seg.letterbox) };
    })();
    r.catch(() => runs.delete(name));
    runs.set(name, r);
  }
  return r;
}

let modelWanted = false;
const waiting = new Set<() => void>();

/** Called when the Theory tab is rebuilt: the old demos' start buttons are gone. */
export function resetDemos(): void {
  waiting.clear();
}

/**
 * A demo that needs the network: a button until the reader asks (or another demo already did), then
 * `ready(run)` fills `body`.
 */
function needsModel(name: string, body: HTMLElement, ready: (run: Run) => void): HTMLElement {
  const gate = h("div", { class: "yo-gate" });
  const start = () => {
    waiting.delete(start);
    gate.replaceChildren(
      h("p", { class: "sub", role: "status", text: T({ tr: "Model yükleniyor ve çalışıyor…", en: "Loading and running the model…" }) }),
    );
    runOn(name).then(
      (run) => {
        gate.remove();
        body.hidden = false;
        ready(run);
      },
      (e: Error) => gate.replaceChildren(h("p", { class: "banner", role: "alert", text: e.message })),
    );
  };
  body.hidden = true;
  if (modelWanted) {
    queueMicrotask(start);
  } else {
    const button = h("button", {
      type: "button",
      class: "btn primary small",
      text: T({ tr: "Gerçek modeli çalıştır", en: "Run the real model" }),
    });
    button.addEventListener("click", () => {
      modelWanted = true;
      for (const other of [...waiting]) other();
    });
    waiting.add(start);
    gate.append(
      button,
      h("p", {
        class: "sub",
        text: T({
          tr: "Bölümün eğitilmiş modeli bu tarayıcıda çalışır (bir kez ~12 MB yüklenir, sonra bütün demolar kullanır).",
          en: "The chapter's trained model runs in this browser (about 12 MB, loaded once and then used by every demo).",
        }),
      }),
    );
  }
  return h("div", { class: "yo-gated" }, gate, body);
}

function segmented<V extends string | number>(options: [V, string][], current: V, pick: (v: V) => void, aria: string): HTMLElement {
  const box = h("div", { class: "seg", role: "group", "aria-label": aria });
  for (const [value, label] of options) {
    const b = h("button", { type: "button", "aria-pressed": String(value === current), text: label });
    b.addEventListener("click", () => {
      for (const other of box.children) other.setAttribute("aria-pressed", String(other === b));
      pick(value);
    });
    box.append(b);
  }
  return box;
}

export function range(
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  format: (v: number) => string,
  onInput: (v: number) => void,
) {
  const input = h("input", { type: "range", min, max, step, value }) as HTMLInputElement;
  const out = h("output", { text: format(value) });
  input.addEventListener("input", () => {
    out.textContent = format(Number(input.value));
    onInput(Number(input.value));
  });
  const node = h("label", { class: "yo-slider" }, h("span", { text: label }), input, out) as HTMLElement & { set: (v: number) => void };
  node.set = (v: number) => {
    input.value = String(v);
    out.textContent = format(v);
  };
  return node;
}

const demo = (...children: (Node | null)[]) => h("div", { class: "t-demo yo-demo" }, ...children);

/** "Next object": choosing an object without a pointer (the demos also take a click on the tile). */
function nextObject(count: () => number, current: () => number, choose: (k: number) => void): HTMLElement {
  const b = h("button", { type: "button", class: "btn small", text: T({ tr: "Sonraki nesne →", en: "Next object →" }) });
  b.addEventListener("click", () => count() && choose((current() + 1) % count()));
  return b;
}
const caption = (text = "") => h("p", { class: "sub yo-caption", "aria-live": "polite", text });

/** A distinct colour for the k-th object. */
function hue(k: number): [number, number, number] {
  const hh = (k * 137.508) % 360;
  const f = (n: number) => {
    const kk = (n + hh / 30) % 12;
    return Math.round(255 * (0.55 - 0.45 * Math.max(-1, Math.min(kk - 3, 9 - kk, 1))));
  };
  return [f(0), f(8), f(4)];
}

const boundsOf = (xy: readonly number[]): Box => {
  const xs = xy.filter((_, i) => i % 2 === 0);
  const ys = xy.filter((_, i) => i % 2 === 1);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
};

// ---------------------------------------------------------------------
// 1. Four tasks on one tile (true labels, no model)
// ---------------------------------------------------------------------

type Task = "classify" | "detect" | "semantic" | "instance";

export function tasksDemo(): HTMLElement {
  const view = new TileView(T({ tr: "Aynı kare, dört görev", en: "The same tile, four tasks" }));
  const chips = h("div", { class: "yo-chips" });
  const note = caption();
  let task: Task = "detect";
  let tile: Tile | null = null;
  const draw = () => {
    if (!tile) return;
    const { sample: s, image, trueClasses } = tile;
    const polys = s.polygons.map((p) => ({ cls: p.cls, xy: p.xy.map((v, i) => v * (i % 2 ? s.height : s.width)) }));
    chips.replaceChildren();
    if (task === "classify") {
      view.show(image, s.width, s.height, []);
      const counts = new Array<number>(SEM_CLASSES.length).fill(0);
      let valid = 0;
      for (const c of trueClasses) {
        if (c === 255) continue;
        counts[c]++;
        valid++;
      }
      const present: [Text, boolean][] = [
        [SEG_CLASSES[0], polys.some((p) => p.cls === 0)],
        [SEG_CLASSES[1], polys.some((p) => p.cls === 1)],
        ...SEM_CLASSES.slice(2).map((name, k): [Text, boolean] => [name, counts[k + 2] > 0.005 * valid]),
      ];
      chips.replaceChildren(
        ...present.map(([name, yes]) => h("span", { class: `yo-chip${yes ? " on" : ""}`, text: `${yes ? "✓" : "✗"} ${T(name)}` })),
      );
      note.textContent = T({
        tr: "Sınıflandırma tek bir cevap verir: karede ne var. Nerede olduğunu, kaç tane olduğunu söylemez.",
        en: "Classification gives one answer: what is in the tile. It does not say where, or how many.",
      });
    } else if (task === "detect") {
      view.show(image, s.width, s.height, [{ boxes: polys.map((p) => ({ box: boundsOf(p.xy), rgb: SEG_RGB[p.cls] })) }]);
      const n = (k: number) => polys.filter((p) => p.cls === k).length;
      note.textContent = T({
        tr: `Tespit: her nesneye bir kutu ve bir sınıf. ${n(0)} bina, ${n(1)} spor sahası; şekilleri yok, yalnızca dikdörtgenler.`,
        en: `Detection: a box and a class for every object. ${n(0)} buildings, ${n(1)} sports fields; no shapes, only rectangles.`,
      });
    } else if (task === "semantic") {
      view.show(image, s.width, s.height, [{ canvas: classOverlay(trueClasses, s.width, s.height) }]);
      chips.replaceChildren(
        ...SEM_CLASSES.slice(1).map((name, k) =>
          h("span", { class: "yo-key" }, h("i", { style: `background: ${css(SEM_RGB[k + 1])}` }), T(name)),
        ),
      );
      note.textContent = T({
        tr: "Anlamsal segmentasyon: her piksele bir sınıf. Yol, su, yeşil alan gibi sayılamayan yüzeyler için doğru araç; ama yan yana iki bina tek bir leke olur.",
        en: "Semantic segmentation: a class for every pixel. The right tool for surfaces you cannot count, like roads, water or grass; but two buildings side by side become one patch.",
      });
    } else {
      view.show(image, s.width, s.height, [{ polygons: polys.map((p, k) => ({ xy: p.xy, rgb: hue(k), width: 1.5, fill: 0.45 })) }]);
      note.textContent = T({
        tr: `Örnek segmentasyonu: her nesneye kendi maskesi. ${polys.length} nesnenin her biri ayrı renkte; bitişik binalar da ayrı sayılır.`,
        en: `Instance segmentation: every object gets its own mask. Each of the ${polys.length} objects has its own colour; buildings that touch are still counted apart.`,
      });
    }
  };
  const pick = segmented<Task>(
    [
      ["classify", T({ tr: "Sınıflandırma", en: "Classification" })],
      ["detect", T({ tr: "Tespit", en: "Detection" })],
      ["semantic", T({ tr: "Anlamsal segm.", en: "Semantic segm." })],
      ["instance", T({ tr: "Örnek segm.", en: "Instance segm." })],
    ],
    task,
    (v) => {
      task = v;
      draw();
    },
    T({ tr: "Görev", en: "Task" }),
  );
  void tileOf(MAIN_TILE).then((t) => {
    tile = t;
    draw();
  });
  return demo(
    pick,
    view.canvas,
    chips,
    note,
    h("p", {
      class: "sub",
      text: T({
        tr: "Doğru etiketler (SpaceNet ve OpenStreetMap), model değil.",
        en: "True labels (SpaceNet and OpenStreetMap), not the model.",
      }),
    }),
  );
}

// ---------------------------------------------------------------------
// 2. Boxes and IoU: drag the prediction over the true box
// ---------------------------------------------------------------------

export function iouDemo(): HTMLElement {
  const view = new TileView(T({ tr: "Doğru kutu ve sürüklenebilir tahmin kutusu", en: "The true box and a prediction box you can drag" }));
  view.canvas.classList.add("yo-drag");
  const readout = h("div", { class: "yo-readout", "aria-live": "polite" });
  let tile: Tile | null = null;
  let truth: Box = [0, 0, 1, 1];
  let pred: Box = [0, 0, 1, 1];
  const sliders: (HTMLElement & { set: (v: number) => void })[] = [];
  const draw = () => {
    if (!tile) return;
    const s = tile.sample;
    const iou = boxIou(truth, pred);
    const w = Math.max(0, Math.min(truth[2], pred[2]) - Math.max(truth[0], pred[0]));
    const hh = Math.max(0, Math.min(truth[3], pred[3]) - Math.max(truth[1], pred[1]));
    const inter: Box = [
      Math.max(truth[0], pred[0]),
      Math.max(truth[1], pred[1]),
      Math.max(truth[0], pred[0]) + w,
      Math.max(truth[1], pred[1]) + hh,
    ];
    const good = iou >= 0.5;
    view.show(tile.image, s.width, s.height, [
      {
        boxes: [
          { box: truth, rgb: MATCH_RGB.missed, label: T({ tr: "doğru", en: "true" }), dashed: true },
          { box: pred, rgb: good ? MATCH_RGB.hit : MATCH_RGB.wrong, label: T({ tr: "tahmin", en: "prediction" }) },
        ],
        paint: (ctx, sc, dpr) => {
          if (w && hh) {
            ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
            ctx.fillRect(inter[0] * sc, inter[1] * sc, w * sc, hh * sc);
          }
          // the corner that resizes the prediction, also when the boxes do not overlap
          ctx.fillStyle = "#fff";
          ctx.fillRect(pred[2] * sc - 5 * dpr, pred[3] * sc - 5 * dpr, 10 * dpr, 10 * dpr);
        },
      },
    ]);
    const m2 = s.metresPerPixel[0] * s.metresPerPixel[1];
    const area = (b: Box) => (b[2] - b[0]) * (b[3] - b[1]);
    const union = area(truth) + area(pred) - w * hh;
    const fmtBox = (b: Box) => `(${b.map((v) => nf(v, 0)).join("; ")})`;
    const centre = (b: Box) => `(${[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]].map((v) => nf(v, 0)).join("; ")})`;
    readout.replaceChildren(
      h("div", {
        class: "formula small",
        text: `\\(\\mathrm{IoU} = \\dfrac{\\text{${T({ tr: "kesişim", en: "intersection" })}}}{\\text{${T({ tr: "birleşim", en: "union" })}}} = \\dfrac{${texNumber(w * hh * m2, 0)}\\ \\mathrm{m^2}}{${texNumber(union * m2, 0)}\\ \\mathrm{m^2}} = ${texNumber(iou, 2)}\\)`,
      }),
      h("p", {
        class: `pc-msg ${good ? "good" : "bad"}`,
        text: good
          ? T({ tr: "IoU ≥ 0,5: bu tahmin doğru tespit sayılır.", en: "IoU ≥ 0.5: this prediction counts as a correct detection." })
          : T({
              tr: "IoU < 0,5: bu tahmin yanlış tespit sayılır ve gerçek nesne kaçırılmış olur.",
              en: "IoU < 0.5: this prediction counts as wrong, and the true object as missed.",
            }),
      }),
      h(
        "p",
        { class: "sub" },
        T({ tr: "Tahmin, köşe yazılışıyla (x₁; y₁; x₂; y₂) = ", en: "The prediction in corner form (x₁; y₁; x₂; y₂) = " }),
        h("code", { text: fmtBox(pred) }),
        T({ tr: ", merkez yazılışıyla (cx; cy; w; h) = ", en: ", in centre form (cx; cy; w; h) = " }),
        h("code", { text: centre(pred) }),
        T({ tr: " piksel.", en: " pixels." }),
      ),
    );
  };
  const sync = () => {
    const [cx, cy, w, hh] = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2, pred[2] - pred[0], pred[3] - pred[1]];
    [cx, cy, w, hh].forEach((v, k) => sliders[k]?.set(Math.round(v)));
  };
  const setFrom = (cx: number, cy: number, w: number, hh: number) => {
    pred = [cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2];
    draw();
  };
  const current = () => [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2, pred[2] - pred[0], pred[3] - pred[1]];
  const labels: Text[] = [
    { tr: "merkez x", en: "centre x" },
    { tr: "merkez y", en: "centre y" },
    { tr: "genişlik", en: "width" },
    { tr: "yükseklik", en: "height" },
  ];
  const controls = h("div", { class: "yo-slider-grid" });
  // dragging: inside the box moves it, the corner square resizes it
  let drag: { mode: "move" | "size"; x: number; y: number; box: Box } | null = null;
  view.canvas.addEventListener("pointerdown", (e) => {
    const p = view.toTile(e);
    const near = Math.abs(p.x - pred[2]) < 12 && Math.abs(p.y - pred[3]) < 12;
    const inside = p.x > pred[0] && p.x < pred[2] && p.y > pred[1] && p.y < pred[3];
    if (!near && !inside) return;
    drag = { mode: near ? "size" : "move", x: p.x, y: p.y, box: [...pred] as Box };
    view.canvas.setPointerCapture(e.pointerId);
  });
  view.canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const p = view.toTile(e);
    const dx = p.x - drag.x;
    const dy = p.y - drag.y;
    const b = drag.box;
    pred =
      drag.mode === "move"
        ? [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
        : [b[0], b[1], Math.max(b[0] + 8, b[2] + dx), Math.max(b[1] + 8, b[3] + dy)];
    sync();
    draw();
  });
  const end = () => (drag = null);
  view.canvas.addEventListener("pointerup", end);
  view.canvas.addEventListener("pointercancel", end);
  void tileOf(MAIN_TILE).then((t) => {
    tile = t;
    const s = t.sample;
    const fields = s.polygons.filter((p) => p.cls === 1).map((p) => boundsOf(p.xy.map((v, i) => v * (i % 2 ? s.height : s.width))));
    truth = fields.sort((a, b) => (b[2] - b[0]) * (b[3] - b[1]) - (a[2] - a[0]) * (a[3] - a[1]))[0] ?? [200, 200, 400, 400];
    const tw = truth[2] - truth[0];
    const th = truth[3] - truth[1];
    pred = [truth[0] + 0.25 * tw, truth[1] + 0.15 * th, truth[2] + 0.25 * tw, truth[3] + 0.15 * th];
    labels.forEach((label, k) => {
      const max = k < 2 ? (k ? s.height : s.width) : Math.max(s.width, s.height);
      const slider = range(
        T(label),
        k < 2 ? 0 : 10,
        max,
        1,
        Math.round(current()[k]),
        (v) => nf(v, 0),
        (v) => {
          const c = current();
          c[k] = v;
          setFrom(c[0], c[1], c[2], c[3]);
        },
      );
      sliders.push(slider);
      controls.append(slider);
    });
    draw();
  });
  return demo(
    h("p", {
      class: "sub",
      text: T({
        tr: "Kesikli turuncu kutu doğru etiket (en büyük futbol sahası). Tahmin kutusunu sürükleyin; sağ alt köşedeki kareden boyutunu değiştirin.",
        en: "The dashed orange box is the true label (the largest football field). Drag the prediction box; resize it from the square at its bottom-right corner.",
      }),
    }),
    view.canvas,
    readout,
    controls,
  );
}

// ---------------------------------------------------------------------
// 3. One look: three grids of candidates
// ---------------------------------------------------------------------

export function gridDemo(): HTMLElement {
  const view = new TileView(T({ tr: "Kare ve seçilen ölçeğin ızgarası", en: "The tile and the grid of the chosen scale" }));
  const info = h("div", { class: "yo-readout", "aria-live": "polite" });
  const counts = caption();
  let level = 0;
  let cell = -1;
  let run: Run | null = null;
  const levelStart = (l: number) => STRIDES.slice(0, l).reduce((n, s) => n + (640 / s) ** 2, 0);
  /** Candidates above 0.25 per scale and class. */
  const aboveByLevel = () =>
    [0, 1, 2].map((l) => {
      const n = [0, 0];
      for (let j = levelStart(l); j < levelStart(l + 1); j++) {
        const c = candidate(run!.seg.head, NC, j);
        if (c.score > 0.25) n[c.cls]++;
      }
      return n;
    });
  const bestCell = (l: number) => {
    const head = run!.seg.head;
    let best = levelStart(l);
    let bestScore = -1;
    for (let j = levelStart(l); j < levelStart(l + 1); j++) {
      const c = candidate(head, NC, j);
      if (c.score > bestScore) {
        bestScore = c.score;
        best = j;
      }
    }
    return best;
  };
  const draw = () => {
    if (!run) return;
    const { seg, sample: s, rgb } = run;
    const lb = seg.letterbox;
    const stride = STRIDES[level];
    const grid = 640 / stride;
    const start = levelStart(level);
    const scores = new Float32Array(grid * grid);
    for (let k = 0; k < grid * grid; k++) scores[k] = candidate(seg.head, NC, start + k).score;
    const c = cell >= 0 ? candidate(seg.head, NC, cell) : null;
    const box = c
      ? ([
          (c.box[0] - lb.left) / lb.gain,
          (c.box[1] - lb.top) / lb.gain,
          (c.box[2] - lb.left) / lb.gain,
          (c.box[3] - lb.top) / lb.gain,
        ] as Box)
      : null;
    view.show(run.image, rgb.width, rgb.height, [
      {
        paint: (ctx, sc, dpr) => {
          const size = (stride / lb.gain) * sc; // one cell in canvas pixels
          const x0 = (-lb.left / lb.gain) * sc;
          const y0 = (-lb.top / lb.gain) * sc;
          for (let k = 0; k < grid * grid; k++) {
            const v = scores[k];
            if (v < 0.05) continue;
            ctx.fillStyle = `rgba(255, 196, 0, ${Math.min(0.9, 0.25 + v)})`;
            ctx.fillRect(x0 + (k % grid) * size, y0 + Math.floor(k / grid) * size, size, size);
          }
          ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
          ctx.lineWidth = dpr;
          ctx.beginPath();
          for (let g = 0; g <= grid; g++) {
            ctx.moveTo(x0 + g * size, y0);
            ctx.lineTo(x0 + g * size, y0 + grid * size);
            ctx.moveTo(x0, y0 + g * size);
            ctx.lineTo(x0 + grid * size, y0 + g * size);
          }
          ctx.stroke();
          if (c) {
            const a = anchorOf(cell);
            ctx.strokeStyle = "#00e5ff";
            ctx.lineWidth = 2.5 * dpr;
            ctx.strokeRect(x0 + a.gx * size, y0 + a.gy * size, size, size);
            if (box) {
              ctx.setLineDash([5 * dpr, 3 * dpr]);
              ctx.strokeRect(box[0] * sc, box[1] * sc, (box[2] - box[0]) * sc, (box[3] - box[1]) * sc);
              ctx.setLineDash([]);
            }
          }
        },
      },
    ]);
    const per = aboveByLevel();
    const line = (l: number, lang: "tr" | "en") =>
      `${640 / STRIDES[l]}×${640 / STRIDES[l]}: ${per[l][0]} ${lang === "tr" ? "bina" : "buildings"}, ${per[l][1]} ${lang === "tr" ? "saha" : "fields"}`;
    counts.textContent = T({
      tr: `Skoru 0,25'i geçen adaylar, ölçek ölçek: ${[0, 1, 2].map((l) => line(l, "tr")).join(" · ")}.${per[0][1] ? "" : " İnce ızgarada hiç saha yok: sahalar kaba ızgaralarda bulunur."}`,
      en: `Candidates scoring above 0.25, scale by scale: ${[0, 1, 2].map((l) => line(l, "en")).join(" · ")}.${per[0][1] ? "" : " No field on the fine grid: fields are found on the coarse ones."}`,
    });
    if (c && box) {
      const a = anchorOf(cell);
      const scoresText = SEG_CLASSES.map((name, k) => `${T(name)} ${nf(seg.head.data[(4 + k) * seg.head.count + cell], 2)}`).join(", ");
      const mpp = s.metresPerPixel[0];
      info.replaceChildren(
        h("p", {
          text: T({
            tr: `Hücre (${a.gx}; ${a.gy}), ${a.grid}×${a.grid} ızgarada, aday no ${cell}. Hücre ${stride} model pikseli ≈ ${nf((stride / lb.gain) * mpp, 1)} m.`,
            en: `Cell (${a.gx}; ${a.gy}) of the ${a.grid}×${a.grid} grid, candidate ${cell}. A cell is ${stride} model pixels ≈ ${nf((stride / lb.gain) * mpp, 1)} m.`,
          }),
        }),
        h("p", {
          text: T({
            tr: `Önerdiği kutu (kesikli): ${nf(box[2] - box[0], 0)} × ${nf(box[3] - box[1], 0)} piksel. Skorlar: ${scoresText}. Bunlara 32 maske katsayısı eklenince 4 + 2 + 32 = 38 sayı.`,
            en: `The box it proposes (dashed): ${nf(box[2] - box[0], 0)} × ${nf(box[3] - box[1], 0)} pixels. Scores: ${scoresText}. With 32 mask coefficients that makes 4 + 2 + 32 = 38 numbers.`,
          }),
        }),
      );
    }
  };
  const pick = segmented<number>(
    [
      [0, "80 × 80 · stride 8"],
      [1, "40 × 40 · stride 16"],
      [2, "20 × 20 · stride 32"],
    ],
    level,
    (v) => {
      level = v;
      cell = bestCell(level);
      draw();
    },
    T({ tr: "Ölçek", en: "Scale" }),
  );
  const select = (e: PointerEvent) => {
    if (!run) return;
    const lb = run.seg.letterbox;
    const p = view.toTile(e);
    const stride = STRIDES[level];
    const grid = 640 / stride;
    const gx = Math.floor((p.x * lb.gain + lb.left) / stride);
    const gy = Math.floor((p.y * lb.gain + lb.top) / stride);
    if (gx < 0 || gy < 0 || gx >= grid || gy >= grid) return;
    cell = levelStart(level) + gy * grid + gx;
    draw();
  };
  view.canvas.addEventListener("pointerdown", select);
  view.canvas.addEventListener("pointermove", (e) => e.buttons && select(e));
  // the keyboard moves the chosen cell with the arrow keys
  view.canvas.tabIndex = 0;
  view.canvas.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!run || !step || cell < 0) return;
    e.preventDefault();
    const a = anchorOf(cell);
    const gx = Math.min(a.grid - 1, Math.max(0, a.gx + step[0]));
    const gy = Math.min(a.grid - 1, Math.max(0, a.gy + step[1]));
    cell = levelStart(a.level) + gy * a.grid + gx;
    draw();
  });
  const body = h(
    "div",
    { class: "yo-demo-body" },
    pick,
    view.canvas,
    h("p", {
      class: "sub",
      text: T({
        tr: "Bir hücreye tıklayın (ya da kareye odaklanıp ok tuşlarını kullanın). Sarı ne kadar koyuysa o hücrenin en yüksek sınıf skoru o kadar büyük.",
        en: "Click a cell (or focus the tile and use the arrow keys). The deeper the yellow, the higher that cell's best class score.",
      }),
    }),
    info,
    counts,
  );
  return demo(
    needsModel(MAIN_TILE, body, (r) => {
      run = r;
      // open on the scale where most candidates pass the threshold
      const totals = aboveByLevel().map((n) => n[0] + n[1]);
      (pick.children[totals.indexOf(Math.max(...totals))] as HTMLButtonElement).click();
    }),
  );
}

// ---------------------------------------------------------------------
// 4. Too many candidates: threshold and NMS, step by step
// ---------------------------------------------------------------------

interface NmsStep {
  kept: Candidate;
  removed: Candidate[];
}

function nmsSteps(sorted: Candidate[], iou: number): NmsStep[] {
  const done = new Set<Candidate>();
  const steps: NmsStep[] = [];
  for (const c of sorted) {
    if (done.has(c)) continue;
    done.add(c);
    const removed = sorted.filter((o) => !done.has(o) && o.cls === c.cls && boxIou(o.box, c.box) > iou);
    for (const o of removed) done.add(o);
    steps.push({ kept: c, removed });
  }
  return steps;
}

export function nmsDemo(): HTMLElement {
  const view = new TileView(T({ tr: "Adaylar ve NMS", en: "Candidates and NMS" }));
  const status = h("p", { class: "yo-big", "aria-live": "polite" });
  let run: Run | null = null;
  let conf = 0.25;
  let iou = 0.7;
  let above: Candidate[] = [];
  let steps: NmsStep[] = [];
  let at = 0;
  const recompute = () => {
    if (!run) return;
    above = aboveConfidence(run.seg.head, NC, conf);
    steps = nmsSteps(above, iou);
    at = Math.min(at, steps.length);
  };
  const toImage = (b: readonly number[]): Box => {
    const lb = run!.seg.letterbox;
    return [(b[0] - lb.left) / lb.gain, (b[1] - lb.top) / lb.gain, (b[2] - lb.left) / lb.gain, (b[3] - lb.top) / lb.gain];
  };
  const draw = () => {
    if (!run) return;
    const kept = steps.slice(0, at).map((s) => s.kept);
    const removed = steps.slice(0, at).flatMap((s) => s.removed);
    const decided = new Set([...kept, ...removed]);
    const waitingBoxes = above.filter((c) => !decided.has(c));
    const last = at ? steps[at - 1] : null;
    view.show(run.image, run.rgb.width, run.rgb.height, [
      {
        boxes: [
          ...waitingBoxes.map((c) => ({ box: toImage(c.box), rgb: [200, 200, 200] })),
          ...(last ? last.removed.map((c) => ({ box: toImage(c.box), rgb: MATCH_RGB.wrong, dashed: true })) : []),
          ...kept.map((c) => ({
            box: toImage(c.box),
            rgb: c === last?.kept ? [0, 229, 255] : MATCH_RGB.hit,
            label: c === last?.kept ? nf(c.score, 2) : undefined,
          })),
        ],
      },
    ]);
    const base = T({
      tr: `Skoru ${nf(conf, 2)}'i geçen ${above.length} aday.`,
      en: `${above.length} candidates score above ${nf(conf, 2)}.`,
    });
    status.textContent = last
      ? `${base} ${T({
          tr: `Adım ${at}/${steps.length}: kalanların en yükseği (${T(SEG_CLASSES[last.kept.cls])}, ${nf(last.kept.score, 2)}) tutuldu; onunla IoU'su ${nf(iou, 2)}'yi aşan ${last.removed.length} aday silindi. Şimdiye kadar ${at} nesne.`,
          en: `Step ${at}/${steps.length}: the best of those left (${T(SEG_CLASSES[last.kept.cls])}, ${nf(last.kept.score, 2)}) is kept; ${last.removed.length} candidates overlapping it by IoU above ${nf(iou, 2)} are removed. ${at} objects so far.`,
        })}`
      : `${base} ${T({ tr: `NMS sonunda ${steps.length} nesne kalacak. "Bir adım"a basın.`, en: `NMS will leave ${steps.length} objects. Press "One step".` })}`;
  };
  const button = (label: string, fn: () => void) => {
    const b = h("button", { type: "button", class: "btn small", text: label });
    b.addEventListener("click", () => {
      fn();
      draw();
    });
    return b;
  };
  const body = h(
    "div",
    { class: "yo-demo-body" },
    h(
      "div",
      { class: "yo-slider-grid" },
      range(
        T({ tr: "Güven eşiği", en: "Confidence threshold" }),
        0.05,
        0.9,
        0.05,
        conf,
        (v) => nf(v, 2),
        (v) => {
          conf = v;
          at = 0;
          recompute();
          draw();
        },
      ),
      range(
        T({ tr: "NMS IoU eşiği", en: "NMS IoU threshold" }),
        0.3,
        0.95,
        0.05,
        iou,
        (v) => nf(v, 2),
        (v) => {
          iou = v;
          at = 0;
          recompute();
          draw();
        },
      ),
    ),
    h(
      "div",
      { class: "t-controls" },
      button(T({ tr: "Baştan", en: "Restart" }), () => (at = 0)),
      button(T({ tr: "Bir adım", en: "One step" }), () => (at = Math.min(steps.length, at + 1))),
      button(T({ tr: "Sona git", en: "To the end" }), () => (at = steps.length)),
    ),
    view.canvas,
    h(
      "div",
      { class: "yo-legend" },
      h(
        "span",
        { class: "yo-key" },
        h("i", { style: "border: 2px solid rgb(200, 200, 200)" }),
        T({ tr: "sırada bekleyen", en: "waiting" }),
      ),
      h("span", { class: "yo-key" }, h("i", { style: `background: ${css(MATCH_RGB.hit)}` }), T({ tr: "tutulan", en: "kept" })),
      h(
        "span",
        { class: "yo-key" },
        h("i", { style: `border: 2px dashed ${css(MATCH_RGB.wrong)}` }),
        T({ tr: "bu adımda silinen", en: "removed in this step" }),
      ),
    ),
    status,
  );
  return demo(
    needsModel(MAIN_TILE, body, (r) => {
      run = r;
      recompute();
      draw();
    }),
  );
}

// ---------------------------------------------------------------------
// 5. Masks: 32 prototypes and their coefficients
// ---------------------------------------------------------------------

export function masksDemo(): HTMLElement {
  const view = new TileView(T({ tr: "Seçilen nesnenin maskesi", en: "The mask of the chosen object" }));
  const protoGrid = h("div", { class: "yo-protos" });
  const mix = h("canvas", { class: "px yo-mix", width: 160, height: 160 }) as HTMLCanvasElement;
  const readout = h("p", { class: "yo-big", "aria-live": "polite" });
  const sliderHost = h("div");
  let run: Run | null = null;
  let objects: InstanceObject[] = [];
  let chosen = 0;
  let coeffs = new Float32Array(32);
  let proto = 0;
  let protoCells: HTMLElement[] = [];
  const draw = () => {
    if (!run) return;
    const o = objects[chosen];
    const { seg, sample: s } = run;
    const lb = seg.letterbox;
    const mask = instanceMask(coeffs, seg.protos, o.box, lb.size);
    view.show(run.image, run.rgb.width, run.rgb.height, [
      {
        canvas: maskOverlay(lb.size, [{ mask, rgb: SEG_RGB[o.cls] }], 0.55),
        letterbox: lb,
        boxes: objects.map((x, k) => ({ box: x.imageBox, rgb: k === chosen ? [0, 229, 255] : [255, 255, 255], dashed: k !== chosen })),
      },
    ]);
    const logits = maskLogits(coeffs, seg.protos);
    let max = 1e-6;
    for (const v of logits) max = Math.max(max, Math.abs(v));
    paint(
      mix,
      Array.from(logits, (v) => v / max),
      seg.protos.width,
      seg.protos.height,
      "div",
      true,
    );
    const area = areaSquareMetres(mask.area, lb.gain, s.metresPerPixel);
    readout.textContent = T({
      tr: `${T(SEG_CLASSES[o.cls])}, skor ${nf(o.score, 2)}: maske ${nf(mask.area, 0)} model pikseli ≈ ${nf(area, 0)} m².`,
      en: `${T(SEG_CLASSES[o.cls])}, score ${nf(o.score, 2)}: the mask is ${nf(mask.area, 0)} model pixels ≈ ${nf(area, 0)} m².`,
    });
    protoCells.forEach((cellNode, k) => {
      cellNode.setAttribute("aria-pressed", String(k === proto));
      cellNode.setAttribute(
        "aria-label",
        T({ tr: `Prototip ${k + 1}, katsayı ${nf(coeffs[k], 2)}`, en: `Prototype ${k + 1}, coefficient ${nf(coeffs[k], 2)}` }),
      );
      (cellNode.querySelector("output") as HTMLElement).textContent = nf(coeffs[k], 2);
    });
  };
  const choose = (k: number) => {
    chosen = k;
    coeffs = Float32Array.from(objects[k].coefficients);
    buildSlider();
    draw();
  };
  const buildSlider = () => {
    const original = objects[chosen].coefficients;
    let r = 1;
    for (const v of original) r = Math.max(r, 2 * Math.abs(v));
    const reset = h("button", { type: "button", class: "btn small", text: T({ tr: "Katsayıları sıfırla", en: "Reset the coefficients" }) });
    reset.addEventListener("click", () => choose(chosen));
    sliderHost.replaceChildren(
      h(
        "div",
        { class: "yo-slider-grid" },
        range(
          `c${proto + 1}`,
          -Number(r.toFixed(1)),
          Number(r.toFixed(1)),
          0.01,
          Number(coeffs[proto].toFixed(2)),
          (v) => nf(v, 2),
          (v) => {
            coeffs[proto] = v;
            draw();
          },
        ),
        reset,
      ),
    );
  };
  view.canvas.addEventListener("pointerdown", (e) => {
    if (!run) return;
    const p = view.toTile(e);
    const lb = run.seg.letterbox;
    const mx = p.x * lb.gain + lb.left;
    const my = p.y * lb.gain + lb.top;
    // the smallest object whose mask covers the point, else whose box does
    const hit = (m: InstanceMask) => {
      const x = Math.floor(mx) - m.x;
      const y = Math.floor(my) - m.y;
      return x >= 0 && y >= 0 && x < m.width && y < m.height && m.data[y * m.width + x] === 1;
    };
    const order = objects.map((_, k) => k).sort((a, b) => objects[a].mask.area - objects[b].mask.area);
    const k =
      order.find((i) => hit(objects[i].mask)) ??
      order.find(
        (i) =>
          p.x >= objects[i].imageBox[0] && p.x <= objects[i].imageBox[2] && p.y >= objects[i].imageBox[1] && p.y <= objects[i].imageBox[3],
      );
    if (k !== undefined) choose(k);
  });
  const body = h(
    "div",
    { class: "yo-demo-body" },
    h(
      "div",
      { class: "t-controls" },
      h("p", {
        class: "sub",
        text: T({
          tr: "Bir nesneye tıklayın; sonra bir prototip seçip katsayısını değiştirin.",
          en: "Click an object; then pick a prototype and change its coefficient.",
        }),
      }),
      nextObject(
        () => objects.length,
        () => chosen,
        choose,
      ),
    ),
    view.canvas,
    readout,
    h("div", {
      class: "mini-title",
      text: T({
        tr: "32 prototip (bütün nesneler için aynı) ve bu nesnenin katsayıları",
        en: "The 32 prototypes (the same for every object) and this object's coefficients",
      }),
    }),
    protoGrid,
    sliderHost,
    h(
      "div",
      { class: "yo-mix-row" },
      mix,
      h("p", {
        class: "sub",
        text: T({
          tr: "Doğrusal birleşim Σ cₖ·Pₖ, 160 × 160 (kırmızı artı, mavi eksi). Kutunun içinde artı olan pikseller, σ'dan sonra 0,5'i geçer: maske onlardır.",
          en: "The linear combination Σ cₖ·Pₖ, 160 × 160 (red positive, blue negative). The pixels inside the box where it is positive pass 0.5 after σ: they are the mask.",
        }),
      }),
    ),
  );
  return demo(
    needsModel(MAIN_TILE, body, (r) => {
      run = r;
      objects = objectsOf(r.seg, NC, r.rgb, { conf: 0.25, iou: 0.7 });
      const { protos } = r.seg;
      const n = protos.width * protos.height;
      protoCells = Array.from({ length: protos.k }, (_, k) => {
        const plane = protos.data.subarray(k * n, (k + 1) * n);
        let max = 1e-6;
        for (const v of plane) max = Math.max(max, Math.abs(v));
        const c = h("canvas", { class: "px", width: protos.width, height: protos.height }) as HTMLCanvasElement;
        paint(
          c,
          Array.from(plane, (v) => v / max),
          protos.width,
          protos.height,
          "div",
          true,
        );
        const b = h("button", { type: "button", class: "yo-proto", title: `P${k + 1}` }, c, h("output"));
        b.addEventListener("click", () => {
          proto = k;
          buildSlider();
          draw();
        });
        return b;
      });
      protoGrid.replaceChildren(...protoCells);
      const fields = objects.map((o, k) => [o, k] as const).filter(([o]) => o.cls === 1);
      const start = fields.length ? fields.sort((a, b) => b[0].mask.area - a[0].mask.area)[0][1] : 0;
      if (objects.length) choose(start);
    }),
  );
}

// ---------------------------------------------------------------------
// 6. Training: the same tile, epoch by epoch (a small copy of the Training tab)
// ---------------------------------------------------------------------

const theoryShot: ShotChoice = { run: "seg-pretrained", shot: 0 };

export function epochDemo(): HTMLElement {
  const box = demo(h("p", { class: "sub", text: T({ tr: "Yükleniyor…", en: "Loading…" }) }));
  void loadRecord().then(({ record, tile }) => {
    if (!tile) return;
    const v = epochViewer(record, tile, theoryShot, ["seg-pretrained", "seg-scratch"]);
    box.replaceChildren(
      v.controls,
      v.canvas,
      v.note,
      h("p", {
        class: "sub",
        text: T({
          tr: "Eğitimde hiç görülmeyen bir doğrulama karesi, eğitim sırasında kaydedilen ağırlıklarla. Bütün eğriler Eğitim sekmesinde.",
          en: "A validation tile never seen in training, predicted with weights saved along the way. All the curves are in the Training tab.",
        }),
      }),
    );
  });
  return box;
}

// ---------------------------------------------------------------------
// 7. How good? Precision, recall and the PR curve
// ---------------------------------------------------------------------

interface Scored {
  object: InstanceObject;
  correct: boolean;
}

export function metricsDemo(): HTMLElement {
  const view = new TileView(T({ tr: "Doğru, yanlış ve kaçan binalar", en: "Correct, wrong and missed buildings" }));
  const chart = h("div", { class: "yo-pr" });
  const readout = h("p", { class: "yo-big", "aria-live": "polite" });
  let run: Run | null = null;
  let conf = 0.25;
  let matchIou = 0.5;
  let all: InstanceObject[] = [];
  let scored: Scored[] = [];
  let curve = { precision: [] as number[], recall: [] as number[] };
  let ap = 0;
  let truths: Truth[] = [];
  const recompute = () => {
    if (!run) return;
    truths = run.truths.filter((t) => t.cls === 0);
    const preds = all.filter((o) => o.cls === 0);
    const m = matchPredictions(preds, truths, (p, t) => maskIou(preds[p].mask, truths[t].mask), matchIou);
    scored = preds.map((o, i) => ({ object: o, correct: m.matches[i].truth >= 0 })).sort((a, b) => b.object.score - a.object.score);
    curve = precisionRecall(
      scored.map((s) => s.object.score),
      scored.map((s) => s.correct),
      truths.length,
    );
    ap = averagePrecision(curve.recall, curve.precision);
  };
  const draw = () => {
    if (!run) return;
    const shown = scored.filter((s) => s.object.score > conf);
    const n = shown.length;
    const tp = shown.filter((s) => s.correct).length;
    const precision = n ? tp / n : 1;
    const recall = truths.length ? tp / truths.length : 0;
    // which true buildings are found at this threshold
    const kept = shown.map((s) => s.object);
    const found = matchPredictions(kept, truths, (p, t) => maskIou(kept[p].mask, truths[t].mask), matchIou);
    const lb = run.seg.letterbox;
    view.show(run.image, run.rgb.width, run.rgb.height, [
      {
        canvas: maskOverlay(
          lb.size,
          shown.map((s) => ({ mask: s.object.mask, rgb: s.correct ? MATCH_RGB.hit : MATCH_RGB.wrong })),
          0.5,
        ),
        letterbox: lb,
        polygons: found.missed.map((k) => ({ xy: truths[k].xy, rgb: MATCH_RGB.missed, width: 2, dashed: true })),
      },
    ]);
    // the PR curve: the envelope's area is the AP
    const W = 260;
    const H = 200;
    const P = 30;
    const X = (r: number) => P + r * (W - P - 8);
    const Y = (p: number) => H - P + -p * (H - P - 8);
    const env = [...curve.precision];
    for (let i = env.length - 2; i >= 0; i--) env[i] = Math.max(env[i], env[i + 1]);
    const line = curve.recall.map((r, i) => `${i ? "L" : "M"}${X(r).toFixed(1)},${Y(curve.precision[i]).toFixed(1)}`).join("");
    let area = `M${X(0)},${Y(0)}`;
    curve.recall.forEach(
      (r, i) =>
        (area += `L${X(i ? curve.recall[i - 1] : 0).toFixed(1)},${Y(env[i]).toFixed(1)}L${X(r).toFixed(1)},${Y(env[i]).toFixed(1)}`),
    );
    area += `L${X(curve.recall.at(-1) ?? 0)},${Y(0)}Z`;
    chart.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${T({ tr: "Kesinlik-duyarlılık eğrisi", en: "Precision-recall curve" })}">
      <path d="${area}" class="yo-pr-area"/>
      <line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(0)}" class="baseline"/><line x1="${X(0)}" y1="${Y(0)}" x2="${X(0)}" y2="${Y(1)}" class="baseline"/>
      <path d="${line}" class="yo-pr-line"/>
      <circle cx="${X(recall)}" cy="${Y(precision)}" r="6" class="yo-pr-point"/>
      <text x="${X(1)}" y="${H - 8}" text-anchor="end" class="tick">${T({ tr: "duyarlılık", en: "recall" })} →</text>
      <text x="${X(0) + 4}" y="${Y(1) + 10}" class="tick">${T({ tr: "kesinlik", en: "precision" })} ↑</text>
      <text x="${X(0) - 4}" y="${Y(0) + 4}" text-anchor="end" class="tick">0</text><text x="${X(0) - 4}" y="${Y(1) + 4}" text-anchor="end" class="tick">1</text>
      <text x="${X(1)}" y="${Y(0) + 14}" text-anchor="middle" class="tick">1</text>
    </svg>`;
    readout.textContent = T({
      tr: `Eşik ${nf(conf, 2)}: ${n} tespit, ${tp} doğru, ${truths.length - found.matches.filter((m) => m.truth >= 0).length} kaçan. Kesinlik ${pct(precision)}, duyarlılık ${pct(recall)}. Bu karede bina AP${Math.round(matchIou * 100)} = ${nf(ap, 2)} (taralı alan).`,
      en: `Threshold ${nf(conf, 2)}: ${n} detections, ${tp} correct, ${truths.length - found.matches.filter((m) => m.truth >= 0).length} missed. Precision ${pct(precision)}, recall ${pct(recall)}. On this tile, building AP${Math.round(matchIou * 100)} = ${nf(ap, 2)} (the shaded area).`,
    });
  };
  const body = h(
    "div",
    { class: "yo-demo-body" },
    h(
      "div",
      { class: "yo-slider-grid" },
      range(
        T({ tr: "Güven eşiği", en: "Confidence threshold" }),
        0.05,
        0.95,
        0.05,
        conf,
        (v) => nf(v, 2),
        (v) => {
          conf = v;
          draw();
        },
      ),
      segmented<number>(
        [
          [0.5, "IoU ≥ 0,50"],
          [0.75, "IoU ≥ 0,75"],
        ].map(([v, l]) => [v as number, T({ tr: l as string, en: (l as string).replace(",", ".") })]),
        matchIou,
        (v) => {
          matchIou = v;
          recompute();
          draw();
        },
        T({ tr: "Doğru sayılma eşiği", en: "Threshold to count as correct" }),
      ),
    ),
    h("div", { class: "yo-pr-row" }, view.canvas, chart),
    readout,
    h(
      "div",
      { class: "yo-legend" },
      h("span", { class: "yo-key" }, h("i", { style: `background: ${css(MATCH_RGB.hit)}` }), T({ tr: "doğru", en: "correct" })),
      h("span", { class: "yo-key" }, h("i", { style: `background: ${css(MATCH_RGB.wrong)}` }), T({ tr: "yanlış", en: "wrong" })),
      h("span", { class: "yo-key" }, h("i", { style: `border: 2px dashed ${css(MATCH_RGB.missed)}` }), T({ tr: "kaçan", en: "missed" })),
    ),
  );
  return demo(
    needsModel(DENSE_TILE, body, (r) => {
      run = r;
      // like validation: keep almost every candidate, the curve is drawn over all of them
      all = objectsOf(r.seg, NC, r.rgb, { conf: 0.01, iou: 0.7 });
      recompute();
      draw();
    }),
  );
}

// ---------------------------------------------------------------------
// 8. Reading the result: how much one mask pixel changes the area
// ---------------------------------------------------------------------

/** The mask grown (r > 0) or shrunk (r < 0) by r pixels, with a round brush, kept inside the size × size input. */
function grow(m: InstanceMask, r: number, size: number): InstanceMask {
  if (!r) return m;
  const R = Math.abs(r);
  const pad = r > 0 ? R : 0;
  const w = m.width + 2 * pad;
  const hh = m.height + 2 * pad;
  const at = (x: number, y: number) => x >= 0 && y >= 0 && x < m.width && y < m.height && m.data[y * m.width + x] === 1;
  const data = new Uint8Array(w * hh);
  let area = 0;
  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < w; x++) {
      const sx = x - pad;
      const sy = y - pad;
      const gx = m.x + sx;
      const gy = m.y + sy;
      if (gx < 0 || gy < 0 || gx >= size || gy >= size) continue; // outside the input: no pixel to grow into
      let on = r < 0;
      for (let dy = -R; dy <= R && on === r < 0; dy++) {
        for (let dx = -R; dx <= R; dx++) {
          if (dx * dx + dy * dy > R * R) continue;
          const v = at(sx + dx, sy + dy);
          if (r > 0 && v) {
            on = true;
            break;
          }
          if (r < 0 && !v) {
            on = false;
            break;
          }
        }
      }
      if (on) {
        data[y * w + x] = 1;
        area++;
      }
    }
  }
  return { x: m.x - pad, y: m.y - pad, width: w, height: hh, data, area };
}

export function areaDemo(): HTMLElement {
  const view = new TileView(T({ tr: "Bir binanın maskesi ve alanı", en: "A building's mask and its area" }));
  const readout = h("div", { class: "yo-readout", "aria-live": "polite" });
  let run: Run | null = null;
  let objects: InstanceObject[] = [];
  let chosen = 0;
  let shift = 0;
  const PROTO = 4; // one prototype pixel = 4 model pixels
  const draw = () => {
    if (!run || !objects.length) return;
    const { seg, sample: s } = run;
    const lb = seg.letterbox;
    const o = objects[chosen];
    const moved = grow(o.mask, shift * PROTO, lb.size);
    const a0 = areaSquareMetres(o.mask.area, lb.gain, s.metresPerPixel);
    const a1 = areaSquareMetres(moved.area, lb.gain, s.metresPerPixel);
    const ew = (PROTO / lb.gain) * s.metresPerPixel[0]; // one prototype pixel on the ground, east-west
    const ns = (PROTO / lb.gain) * s.metresPerPixel[1]; // and north-south
    const change = shift && a0 ? ` (${a1 >= a0 ? "+" : "−"}${pct(Math.abs(a1 / a0 - 1))})` : "";
    view.show(run.image, run.rgb.width, run.rgb.height, [
      {
        canvas: maskOverlay(lb.size, [{ mask: moved, rgb: shift ? MATCH_RGB.missed : SEG_RGB[0] }], 0.55),
        letterbox: lb,
        boxes: objects.map((x, k) => ({ box: x.imageBox, rgb: k === chosen ? [0, 229, 255] : [255, 255, 255], dashed: k !== chosen })),
      },
    ]);
    readout.replaceChildren(
      h("p", {
        class: "yo-big",
        text: T({
          tr: shift
            ? `Model: ${nf(a0, 0)} m². Sınır her yandan ${Math.abs(shift)} prototip pikseli ${shift > 0 ? "dışarı" : "içeri"} kayınca: ${nf(a1, 0)} m²${change}.`
            : `Model: ${nf(a0, 0)} m². Sınırı kaydırmak için kaydırıcıyı kullanın.`,
          en: shift
            ? `Model: ${nf(a0, 0)} m². With the outline moved ${Math.abs(shift)} prototype pixel${Math.abs(shift) > 1 ? "s" : ""} ${shift > 0 ? "out" : "in"} on every side: ${nf(a1, 0)} m²${change}.`
            : `Model: ${nf(a0, 0)} m². Move the outline with the slider.`,
        }),
      }),
      h("p", {
        class: "sub",
        text: T({
          tr: `Maske 160 × 160 prototiplerden gelir: bir prototip pikseli 4 model pikseli, bu karede doğu-batı ${nf(ew, 2)} m, kuzey-güney ${nf(ns, 2)} m. Küçük bir binada bu kadarlık bir kayma alanı büyük oranda değiştirir; büyük bir binada daha az.`,
          en: `The mask comes from 160 × 160 prototypes: one prototype pixel is 4 model pixels, on this tile ${nf(ew, 2)} m east-west and ${nf(ns, 2)} m north-south. On a small building a shift this size changes the area a lot; on a large one, less.`,
        }),
      }),
    );
  };
  view.canvas.addEventListener("pointerdown", (e) => {
    if (!run) return;
    const p = view.toTile(e);
    const order = objects.map((_, k) => k).sort((a, b) => objects[a].mask.area - objects[b].mask.area);
    const k = order.find(
      (i) =>
        p.x >= objects[i].imageBox[0] && p.x <= objects[i].imageBox[2] && p.y >= objects[i].imageBox[1] && p.y <= objects[i].imageBox[3],
    );
    if (k !== undefined) {
      chosen = k;
      draw();
    }
  });
  const body = h(
    "div",
    { class: "yo-demo-body" },
    h(
      "div",
      { class: "t-controls" },
      h("p", {
        class: "sub",
        text: T({ tr: "Bir binaya tıklayın, sonra sınırını kaydırın.", en: "Click a building, then move its outline." }),
      }),
      nextObject(
        () => objects.length,
        () => chosen,
        (k) => {
          chosen = k;
          draw();
        },
      ),
    ),
    view.canvas,
    h(
      "div",
      { class: "yo-slider-grid" },
      range(
        T({ tr: "Sınırı kaydır (prototip pikseli)", en: "Move the outline (prototype pixels)" }),
        -2,
        2,
        1,
        0,
        (v) => (v > 0 ? `+${v}` : String(v)),
        (v) => {
          shift = v;
          draw();
        },
      ),
    ),
    readout,
  );
  return demo(
    needsModel(DENSE_TILE, body, (r) => {
      run = r;
      objects = objectsOf(r.seg, NC, r.rgb, { conf: 0.25, iou: 0.7 }).filter((o) => o.cls === 0);
      // start with a small house: the effect is the lesson
      const order = objects.map((_, k) => k).sort((a, b) => objects[a].mask.area - objects[b].mask.area);
      chosen = order[Math.floor(order.length / 4)] ?? 0;
      draw();
    }),
  );
}
