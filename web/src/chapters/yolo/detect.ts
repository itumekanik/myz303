/**
 * The Detection tab: both trained models on a validation tile (or the reader's own image), in the
 * browser. The network runs once per image; moving the confidence or NMS slider only repeats the
 * post-processing (engines/yolo/detect.ts), so the reader sees what each step removes.
 *
 * With a validation tile the true outlines are known: a prediction that overlaps a true object of
 * its class by mask IoU ≥ 0.5 is correct (green), the others are wrong (red), true objects nobody
 * found are missed (orange). Areas are mask pixels × the ground size of a pixel.
 */
import { $, fmtNum, fmtPct, h } from "../../shared/core.js";
import { unpack } from "../../engines/data.ts";
import { aboveConfidence, candidateCount, coefficients, nms, type Candidate } from "../../engines/yolo/detect.ts";
import { toImageBox } from "../../engines/yolo/image.ts";
import { areaSquareMetres, instanceMask, type InstanceMask } from "../../engines/yolo/masks.ts";
import { matchPredictions } from "../../engines/yolo/metrics.ts";
import { loadModel, runLandCover, runSegmentation, type LandCoverResult, type SegmentationOutput } from "../../engines/yolo/model.ts";
import { maskIou, unionArea } from "../../engines/yolo/raster.ts";
import {
  CITIES,
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
  type Layer,
  type Sample,
  type SampleSet,
  type Truth,
  truthsOf,
} from "./page.ts";
import type { Rgb } from "../../engines/yolo/image.ts";
import type { Text } from "../../shared/chapters.ts";

const MATCH_IOU = 0.5;
/** The tile shown first: buildings, a river and four sports fields the model finds. */
const FIRST = "Shanghai_img209";
const DEFAULT_MPP = 0.3;

interface Shown extends Candidate {
  mask: InstanceMask;
  imageBox: [number, number, number, number];
  /** index of the matched true object, -1 if none (only with a validation tile) */
  truth: number;
}

/** What the page knows about the image on screen. */
interface Current {
  /** the file name of an uploaded image (a sample is named by its city and tile) */
  label: string;
  image: HTMLImageElement;
  rgb: Rgb;
  sample: Sample | null;
  metresPerPixel: [number, number];
  instances?: SegmentationOutput;
  land?: LandCoverResult;
  truths?: Truth[];
  trueClasses?: Uint8Array;
}

const state = {
  set: null as SampleSet | null,
  pick: 0,
  current: null as Current | null,
  conf: 0.25,
  iou: 0.7,
  view: "objects" as "objects" | "land",
  compare: false,
  boxes: true,
  status: "" as string,
  error: "" as string,
  busy: false,
  ms: { network: 0, land: 0 },
};

let started = false;
let view: TileView | null = null;

const root = () => $("#yd-root") as HTMLElement;
const nf = (v: number, d = 0) => fmtNum(v, d) as string;
const pct = (v: number, d = 1) => fmtPct(100 * v, d) as string;

// ---------------------------------------------------------------------
// Running the models
// ---------------------------------------------------------------------

async function showSample(k: number): Promise<void> {
  const set = state.set!;
  const s = set.samples[k];
  state.pick = k;
  const { image, rgb } = await decodeImage(`data:image/jpeg;base64,${s.jpeg}`);
  const truthClasses = await unpack(s.landCover);
  await analyse({ label: s.tile, image, rgb, sample: s, metresPerPixel: s.metresPerPixel, trueClasses: truthClasses });
}

async function showUpload(file: File): Promise<void> {
  const url = URL.createObjectURL(file);
  try {
    const { image, rgb } = await decodeImage(url);
    state.pick = -1;
    state.compare = false;
    await analyse({ label: file.name, image, rgb, sample: null, metresPerPixel: [DEFAULT_MPP, DEFAULT_MPP] });
  } catch {
    state.error = T({ tr: "Bu dosya bir görüntü olarak açılamadı.", en: "This file could not be opened as an image." });
    render();
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function analyse(current: Current): Promise<void> {
  state.current = current;
  state.busy = true;
  state.error = "";
  state.status = T({ tr: "Modeller yükleniyor (ilk seferde ~10 MB)…", en: "Loading the models (~10 MB the first time)…" });
  render();
  try {
    const [seg, sem] = await Promise.all([loadModel("seg"), loadModel("sem")]);
    state.status = T({ tr: "Model çalışıyor…", en: "Running the model…" });
    render();
    await new Promise((r) => setTimeout(r, 0)); // let the status paint before the CPU is busy
    const instances = await runSegmentation(seg, current.rgb);
    const land = await runLandCover(sem, current.rgb);
    if (state.current !== current) return; // another image was picked meanwhile
    current.instances = instances;
    current.land = land;
    state.ms = { network: instances.ms.network, land: land.ms.network };
    if (current.sample) current.truths = truthsOf(current.sample, instances.letterbox);
  } catch (e) {
    state.error = `${T({ tr: "Model çalıştırılamadı:", en: "The model could not run:" })} ${(e as Error).message}`;
  } finally {
    state.busy = false;
    state.status = "";
    render();
  }
}

/** Confidence filter, NMS and masks again, from the network output kept in memory. */
function postprocess(c: Current): { above: number; shown: Shown[]; missed: number[] } {
  const r = c.instances!;
  const classes = SEG_CLASSES.length;
  const above = aboveConfidence(r.head, classes, state.conf);
  const kept = nms(above, state.iou);
  const shown: Shown[] = kept.map((k) => ({
    ...k,
    mask: instanceMask(coefficients(r.head, classes, k.index), r.protos, k.box, r.letterbox.size),
    imageBox: toImageBox(k.box, r.letterbox, c.rgb.width, c.rgb.height),
    truth: -1,
  }));
  let missed: number[] = [];
  if (c.truths) {
    const truths = c.truths;
    const m = matchPredictions(shown, truths, (p, t) => maskIou(shown[p].mask, truths[t].mask), MATCH_IOU);
    m.matches.forEach((x, i) => (shown[i].truth = x.truth));
    missed = m.missed;
  }
  return { above: above.length, shown, missed };
}

// ---------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------

function picker(): HTMLElement {
  const set = state.set!;
  const thumbs = set.samples.map((s, k) =>
    h(
      "button",
      { type: "button", class: "yo-thumb", "aria-pressed": String(k === state.pick), title: `${T(CITIES[s.city])} · ${s.tile}` },
      h("img", { src: `data:image/jpeg;base64,${s.jpeg}`, alt: "", loading: "lazy" }),
      h("span", { text: T(CITIES[s.city]) }),
    ),
  );
  thumbs.forEach((b, k) => b.addEventListener("click", () => !state.busy && void showSample(k)));
  const input = h("input", { type: "file", accept: "image/*", class: "yo-file", id: "yd-file" }) as HTMLInputElement;
  input.addEventListener("change", () => input.files?.[0] && void showUpload(input.files[0]));
  const upload = h(
    "label",
    { class: "btn small yo-upload", for: "yd-file" },
    T({ tr: "Kendi görüntünüzü yükleyin", en: "Upload your own image" }),
  );
  return h(
    "div",
    { class: "card yo-picker" },
    h(
      "div",
      { class: "card-head" },
      h(
        "div",
        {},
        h("h2", { text: T({ tr: "Bir kare seçin", en: "Pick a tile" }) }),
        h("p", {
          class: "sub",
          text: T({
            tr: "Doğrulama setinden sekiz kare, her şehirden iki tane: modeller bunları eğitimde hiç görmedi. Bir kare yaklaşık 195 m × 195 m.",
            en: "Eight validation tiles, two per city: the models never saw them in training. A tile is about 195 m × 195 m.",
          }),
        }),
      ),
      h("div", { class: "yo-upload-box" }, input, upload),
    ),
    h("div", { class: "yo-thumbs" }, ...thumbs),
    h("p", {
      class: "sub yo-note",
      text: T({
        tr: "Kendi görüntünüz bilgisayarınızdan çıkmaz; model tarayıcıda çalışır. Model 0,3 m/piksel çözünürlükte, tepeden çekilmiş uydu görüntüleriyle eğitildi; başka ölçekte ya da açıda sonuç kötüleşir.",
        en: "Your image never leaves your computer; the model runs in the browser. The model was trained on top-down satellite images at 0.3 m per pixel; at another scale or angle it does worse.",
      }),
    }),
  );
}

function controls(): HTMLElement {
  const slider = (label: string, min: number, max: number, step: number, value: number, set: (v: number) => void) => {
    const input = h("input", { type: "range", min, max, step, value }) as HTMLInputElement;
    const out = h("output", { text: nf(value, 2) });
    input.addEventListener("input", () => {
      set(Number(input.value));
      out.textContent = nf(Number(input.value), 2);
      refresh();
    });
    return h("label", { class: "yo-slider" }, h("span", { text: label }), input, out);
  };
  const check = (label: string, value: boolean, set: (v: boolean) => void, disabled = false) => {
    const input = h("input", { type: "checkbox", checked: value, disabled }) as HTMLInputElement;
    input.addEventListener("change", () => {
      set(input.checked);
      refresh();
    });
    return h("label", { class: "yo-check" }, input, h("span", { text: label }));
  };
  const objects = state.view === "objects";
  return h(
    "div",
    { class: "card yo-controls" },
    h("h2", { text: T({ tr: "Ayarlar", en: "Settings" }) }),
    objects ? slider(T({ tr: "Güven eşiği", en: "Confidence threshold" }), 0.05, 0.95, 0.05, state.conf, (v) => (state.conf = v)) : null,
    objects ? slider(T({ tr: "NMS IoU eşiği", en: "NMS IoU threshold" }), 0.3, 0.95, 0.05, state.iou, (v) => (state.iou = v)) : null,
    objects ? check(T({ tr: "Kutuları göster", en: "Show the boxes" }), state.boxes, (v) => (state.boxes = v)) : null,
    check(
      objects
        ? T({ tr: "Doğru etiketlerle karşılaştır", en: "Compare with the true labels" })
        : T({ tr: "Doğru haritayı göster", en: "Show the true map" }),
      state.compare,
      (v) => (state.compare = v),
      !state.current?.sample,
    ),
    h("div", { class: "yo-pipeline", id: "yd-pipeline" }),
  );
}

function render(): void {
  const host = root();
  if (!state.set) return;
  const c = state.current;
  const seg = h(
    "div",
    { class: "seg", role: "group", "aria-label": T({ tr: "Gösterilen model", en: "Model shown" }) },
    ...(
      [
        ["objects", T({ tr: "Nesneler", en: "Objects" })],
        ["land", T({ tr: "Arazi örtüsü", en: "Land cover" })],
      ] as const
    ).map(([value, label]) => {
      const b = h("button", { type: "button", "aria-pressed": String(state.view === value), text: label });
      b.addEventListener("click", () => {
        state.view = value;
        render();
      });
      return b;
    }),
  );
  view = new TileView(T({ tr: "Seçilen kare ve modelin bulduğu nesneler", en: "The chosen tile and what the model found" }));
  const status = state.error
    ? h("p", { class: "banner yo-status", role: "alert", text: state.error })
    : state.status
      ? h("p", { class: "yo-status", role: "status", text: state.status })
      : null;
  host.replaceChildren(
    h(
      "div",
      { class: "yo-detect" },
      picker(),
      h(
        "div",
        { class: "yo-main" },
        h(
          "div",
          { class: "card yo-view-card" },
          h(
            "div",
            { class: "card-head" },
            h("h2", { text: c ? (c.sample ? `${T(CITIES[c.sample.city])} · ${c.sample.tile}` : c.label) : "" }),
            seg,
          ),
          status,
          view.canvas,
          h("div", { class: "yo-legend", id: "yd-legend" }),
          h("p", {
            class: "sub",
            text: T({
              tr: "Görüntü ve bina sınırları: SpaceNet 2 (CC BY-SA 4.0). Spor sahası, yol, yeşil alan, su ve otopark etiketleri: © OpenStreetMap katkıcıları (ODbL).",
              en: "Imagery and building outlines: SpaceNet 2 (CC BY-SA 4.0). Sports field, road, green, water and parking labels: © OpenStreetMap contributors (ODbL).",
            }),
          }),
        ),
        h("div", { class: "yo-side" }, controls(), h("div", { class: "card yo-results", id: "yd-results" })),
      ),
    ),
  );
  refresh();
}

/** Redraws the tile and the numbers for the current settings, without rebuilding the page. */
function refresh(): void {
  const c = state.current;
  if (!c || !view) return;
  const legend = $("#yd-legend") as HTMLElement;
  const results = $("#yd-results") as HTMLElement;
  const pipeline = $("#yd-pipeline") as HTMLElement | null;
  if (!c.instances || !c.land) {
    view.show(c.image, c.rgb.width, c.rgb.height, []);
    legend.replaceChildren();
    results.replaceChildren(h("p", { class: "sub", text: state.busy ? state.status : "" }));
    return;
  }
  if (state.view === "objects") showObjects(c, legend, results, pipeline);
  else showLand(c, legend, results);
}

const swatch = (rgb: readonly number[], label: string, outline = false) =>
  h("span", { class: "yo-key" }, h("i", { style: outline ? `border: 2px solid ${css(rgb)}` : `background: ${css(rgb)}` }), label);

function showObjects(c: Current, legend: HTMLElement, results: HTMLElement, pipeline: HTMLElement | null): void {
  const { above, shown, missed } = postprocess(c);
  const lb = c.instances!.letterbox;
  const compare = state.compare && !!c.truths;
  const colour = (s: Shown) => (compare ? (s.truth >= 0 ? MATCH_RGB.hit : MATCH_RGB.wrong) : SEG_RGB[s.cls]);
  const layer: Layer = {
    canvas: maskOverlay(
      lb.size,
      shown.map((s) => ({ mask: s.mask, rgb: colour(s) })),
      0.45,
    ),
    letterbox: lb,
    boxes: state.boxes ? shown.map((s) => ({ box: s.imageBox, rgb: colour(s), label: nf(s.score, 2) })) : [],
    polygons: compare ? missed.map((k) => ({ xy: c.truths![k].xy, rgb: MATCH_RGB.missed, width: 2, dashed: true })) : [],
  };
  view!.show(c.image, c.rgb.width, c.rgb.height, [layer]);

  legend.replaceChildren(
    ...(compare
      ? [
          swatch(MATCH_RGB.hit, T({ tr: "doğru tespit", en: "correct" })),
          swatch(MATCH_RGB.wrong, T({ tr: "yanlış tespit", en: "wrong" })),
          swatch(MATCH_RGB.missed, T({ tr: "kaçan (doğru sınır)", en: "missed (true outline)" }), true),
        ]
      : SEG_CLASSES.map((name, k) => swatch(SEG_RGB[k], T(name)))),
  );

  if (pipeline) {
    pipeline.replaceChildren(
      h(
        "div",
        {},
        h("strong", { text: nf(candidateCount(lb.size)) }),
        T({ tr: " aday (her ızgara hücresinde bir)", en: " candidates (one per grid cell)" }),
      ),
      h(
        "div",
        {},
        h("strong", { text: nf(above) }),
        T({ tr: ` güven eşiğini (${nf(state.conf, 2)}) geçti`, en: ` passed the confidence threshold (${nf(state.conf, 2)})` }),
      ),
      h("div", {}, h("strong", { text: nf(shown.length) }), T({ tr: " nesne NMS'ten sonra kaldı", en: " objects left after NMS" })),
    );
  }

  // numbers per class
  const mpp = c.metresPerPixel;
  const groundArea = c.rgb.width * c.rgb.height * mpp[0] * mpp[1];
  // one column per class, one row per number (the side column is narrow)
  const perClass = SEG_CLASSES.map((_, k) => {
    const mine = shown.filter((s) => s.cls === k);
    const area = areaSquareMetres(
      unionArea(
        mine.map((s) => s.mask),
        lb.size,
        lb.size,
      ),
      lb.gain,
      mpp,
    );
    const truths = c.truths?.filter((t) => t.cls === k) ?? [];
    const trueArea = areaSquareMetres(
      unionArea(
        truths.map((t) => t.mask),
        lb.size,
        lb.size,
      ),
      lb.gain,
      mpp,
    );
    const hits = mine.filter((s) => s.truth >= 0).length;
    return { found: mine.length, area, truths: truths.length, trueArea, hits };
  });
  const line = (label: Text, cell: (p: (typeof perClass)[number]) => string) =>
    h("tr", {}, h("th", { scope: "row", text: T(label) }), ...perClass.map((p) => h("td", { text: cell(p) })));
  const rows: HTMLElement[] = [
    line({ tr: "bulunan", en: "found" }, (p) => nf(p.found)),
    line({ tr: "alan", en: "area" }, (p) => `${nf(p.area)} m²`),
  ];
  if (c.truths) {
    rows.push(
      line({ tr: "gerçekte", en: "true" }, (p) => nf(p.truths)),
      line({ tr: "gerçek alan", en: "true area" }, (p) => `${nf(p.trueArea)} m²`),
      line({ tr: "doğru tespit", en: "correct" }, (p) => nf(p.hits)),
      line({ tr: "kesinlik", en: "precision" }, (p) => (p.found ? pct(p.hits / p.found, 0) : "–")),
      line({ tr: "duyarlılık", en: "recall" }, (p) => (p.truths ? pct(p.hits / p.truths, 0) : "–")),
    );
  }
  const head = ["", ...SEG_CLASSES.map((name) => T(name))];
  const buildings = shown.filter((s) => s.cls === 0);
  const builtArea = areaSquareMetres(
    unionArea(
      buildings.map((s) => s.mask),
      lb.size,
      lb.size,
    ),
    lb.gain,
    mpp,
  );
  results.replaceChildren(
    h("h2", { text: T({ tr: "Sonuç", en: "Result" }) }),
    h(
      "div",
      { class: "yo-table-wrap" },
      h(
        "table",
        { class: "yo-table" },
        h("thead", {}, h("tr", {}, ...head.map((t) => h("th", { scope: "col", text: t })))),
        h("tbody", {}, ...rows),
      ),
    ),
    h(
      "p",
      { class: "yo-big" },
      T({ tr: "Yapılaşma oranı (bina tabanı / kare alanı): ", en: "Built-up ratio (building footprint / tile area): " }),
      h("strong", { text: pct(builtArea / groundArea) }),
    ),
    h("p", {
      class: "sub",
      text: c.sample
        ? T({
            tr: `Alan = maskedeki piksel × bir pikselin zemindeki alanı (${nf(mpp[0], 2)} m × ${nf(mpp[1], 2)} m). Çatı görünür, taban görünmez: bu bir çatı izdüşümü alanıdır. Doğru sayılmak için maske, aynı sınıftan bir gerçek nesneyle en az %${MATCH_IOU * 100} örtüşmeli (maske IoU).`,
            en: `Area = mask pixels × the ground area of one pixel (${nf(mpp[0], 2)} m × ${nf(mpp[1], 2)} m). The roof is visible, the footprint is not: this is the roof's projected area. To count as correct, a mask must overlap a true object of its class by at least ${MATCH_IOU * 100}% (mask IoU).`,
          })
        : T({
            tr: `Alan, piksel başına ${nf(DEFAULT_MPP, 1)} m varsayımıyla hesaplandı; görüntünüzün çözünürlüğü farklıysa alan da farklıdır.`,
            en: `Areas assume ${nf(DEFAULT_MPP, 1)} m per pixel; if your image has another resolution, the areas differ.`,
          }),
    }),
    h("p", {
      class: "sub",
      text: T({
        tr: `Ağ bu bilgisayarda ${nf(state.ms.network)} ms'de çalıştı (tarayıcıda, sunucusuz).`,
        en: `The network ran in ${nf(state.ms.network)} ms on this computer (in the browser, no server).`,
      }),
    }),
  );
}

function showLand(c: Current, legend: HTMLElement, results: HTMLElement): void {
  const land = c.land!;
  const lb = land.letterbox;
  const truth = state.compare && c.trueClasses;
  const layer: Layer = truth
    ? { canvas: classOverlay(c.trueClasses!, c.rgb.width, c.rgb.height) }
    : { canvas: classOverlay(land.classes, lb.size, lb.size), letterbox: lb };
  view!.show(c.image, c.rgb.width, c.rgb.height, [layer]);
  legend.replaceChildren(...SEM_CLASSES.slice(1).map((name, k) => swatch(SEM_RGB[k + 1], T(name))));

  const total = land.counts.reduce((a, b) => a + b, 0);
  const rows: HTMLElement[] = [];
  let compare: { share: number[]; iou: number[]; accuracy: number } | null = null;
  if (c.trueClasses) compare = compareLand(c, land);
  SEM_CLASSES.forEach((name, k) => {
    const cells: (string | HTMLElement)[] = [swatch(SEM_RGB[k], T(name)), pct(land.counts[k] / total)];
    if (compare) cells.push(pct(compare.share[k]), Number.isNaN(compare.iou[k]) ? "–" : nf(compare.iou[k], 2));
    rows.push(h("tr", {}, ...cells.map((v, i) => h(i ? "td" : "th", { scope: i ? null : "row" }, v))));
  });
  const head = ["", T({ tr: "model", en: "model" }), ...(compare ? [T({ tr: "gerçek", en: "true" }), T({ tr: "IoU", en: "IoU" })] : [])];
  results.replaceChildren(
    h("h2", { text: T({ tr: "Karenin ne kadarı ne?", en: "How much of the tile is what?" }) }),
    h(
      "div",
      { class: "yo-table-wrap" },
      h(
        "table",
        { class: "yo-table" },
        h("thead", {}, h("tr", {}, ...head.map((t) => h("th", { scope: "col", text: t })))),
        h("tbody", {}, ...rows),
      ),
    ),
    compare
      ? h(
          "p",
          { class: "yo-big" },
          T({ tr: "Doğru sınıflanan piksel: ", en: "Pixels classified correctly: " }),
          h("strong", { text: pct(compare.accuracy) }),
        )
      : null,
    h("p", {
      class: "sub",
      text: T({
        tr: `Her piksel tek bir sınıfa atanır; bu yüzden binalar burada tek tek sayılmaz. Gerçek harita OpenStreetMap'ten üretildi ve kusursuz değil: eksik çizilmiş otoparklar, görüntüden yeni yollar. Ağ ${nf(state.ms.land)} ms'de çalıştı.`,
        en: `Every pixel gets one class, so buildings are not counted one by one here. The true map was made from OpenStreetMap and is not perfect: car parks that were never drawn, roads newer than the image. The network ran in ${nf(state.ms.land)} ms.`,
      }),
    }),
  );
}

/** Class shares of the true map and the IoU of every class, on the tile's own pixels. */
function compareLand(c: Current, land: LandCoverResult): { share: number[]; iou: number[]; accuracy: number } {
  const lb = land.letterbox;
  const truth = c.trueClasses!;
  const n = SEM_CLASSES.length;
  const inter = new Array<number>(n).fill(0);
  const predicted = new Array<number>(n).fill(0);
  const actual = new Array<number>(n).fill(0);
  let valid = 0;
  let correct = 0;
  const { width: w, height: hgt } = c.rgb;
  for (let y = 0; y < hgt; y++) {
    const my = lb.top + Math.min(lb.height - 1, Math.floor((y + 0.5) * lb.gain));
    for (let x = 0; x < w; x++) {
      const t = truth[y * w + x];
      if (t === 255) continue; // no image data
      const mx = lb.left + Math.min(lb.width - 1, Math.floor((x + 0.5) * lb.gain));
      const p = land.classes[my * lb.size + mx];
      valid++;
      actual[t]++;
      predicted[p]++;
      if (p === t) {
        inter[t]++;
        correct++;
      }
    }
  }
  return {
    share: actual.map((a) => a / valid),
    iou: inter.map((i, k) => (actual[k] + predicted[k] - i ? i / (actual[k] + predicted[k] - i) : Number.NaN)),
    accuracy: correct / valid,
  };
}

async function start(): Promise<void> {
  if (started) return;
  started = true;
  try {
    state.set = await loadSamples();
  } catch (e) {
    root().replaceChildren(h("p", { class: "banner", role: "alert", text: (e as Error).message }));
    return;
  }
  render();
  await showSample(
    Math.max(
      0,
      state.set.samples.findIndex((s) => s.name === FIRST),
    ),
  );
}

document.addEventListener("tab-shown", (e) => {
  if ((e as CustomEvent).detail === "detect") void start();
});
document.addEventListener("lang-change", () => {
  if (state.set) render();
});
