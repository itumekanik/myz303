/**
 * The Training tab: the record of how the chapter's models were trained (data/yolo-run.js, written
 * by tools/export_yolo_assets.py from Ultralytics' results.csv and the saved epoch weights).
 * Training YOLO takes a GPU for an hour and 2 GB of data, so the page shows our run instead of
 * starting one; the Coding tab shows how to repeat it.
 */
import { $, fmtDuration, fmtNum, fmtPct, h, lineChart, linearTicks, niceStep, resizeObserver } from "../../shared/core.js";
import { loadData, unpack, type Packed } from "../../engines/data.ts";
import { fillPolygon } from "../../engines/yolo/raster.ts";
import {
  CITIES,
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
} from "./page.ts";
import type { Text } from "../../shared/chapters.ts";

export type RunName = "seg-pretrained" | "seg-scratch" | "sem-pretrained";

interface Run {
  epochs: number;
  seconds: number;
  curves: Record<string, number[]>;
  final: Record<string, number | number[] | null>;
}

interface SegShot {
  epoch: number;
  objects: { cls: number; score: number; box: number[]; xy: number[] }[];
}
interface SemShot {
  epoch: number;
  classes: Packed;
  width: number;
  height: number;
}

export interface RunRecord {
  device: string;
  settings: { imgsz: number; batch: number; epochs: number; flipud: number; degrees: number; seed: number };
  images: { train: number; val: number };
  runs: Record<RunName, Run>;
  snapshotTile: string;
  snapshots: Record<RunName, (SegShot | SemShot)[]>;
}

const RUN_NAMES: Record<RunName, Text> = {
  "seg-pretrained": { tr: "Önceden eğitilmiş (COCO'dan başlayan)", en: "Pretrained (starting from COCO)" },
  "seg-scratch": { tr: "Sıfırdan (rastgele ağırlıklar)", en: "From scratch (random weights)" },
  "sem-pretrained": { tr: "Arazi örtüsü modeli", en: "Land-cover model" },
};

const state = {
  record: null as RunRecord | null,
  tile: null as Tile | null,
  loss: "box" as "box" | "seg" | "cls",
  run: "seg-pretrained" as RunName,
  shot: 6,
};
let started = false;

const nf = (v: number, d = 2) => fmtNum(v, d) as string;
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--critical)"];

function chart(height = 220): HTMLElement {
  const host = h("div", { class: "chart", "data-height": height }) as HTMLElement;
  resizeObserver.observe(host);
  return host;
}

/**
 * A line chart over the epochs; `series` are [label, values, colour]. Without `yMin` the axis starts
 * just below the lowest value, so a loss that falls from 1.9 to 1.6 does not look flat.
 */
function curves(
  host: HTMLElement,
  label: string,
  series: [string, number[], string][],
  format: (v: number) => string,
  yMin?: number,
): void {
  const all = series.flatMap(([, v]) => v);
  const n = Math.max(...series.map(([, v]) => v.length));
  const low = Math.min(...all);
  const hi = Math.max(...all);
  let step = niceStep(hi - (yMin ?? low), 4);
  const lo = yMin ?? Math.floor(low / step) * step;
  step = niceStep(hi - lo, 4);
  const top = Math.ceil(hi / step) * step;
  lineChart(host, {
    label,
    empty: "",
    x: { min: 1, max: n, ticks: [1, 20, 40, 60, 80, 100].filter((v) => v <= n).map((v) => ({ v, label: String(v) })) },
    y: { min: lo, max: top, ticks: linearTicks(lo, top, step, format) },
    series: series.map(([name, values, color]) => ({ name, points: values.map((v, i) => [i + 1, v]), color })),
    tooltip: (i: number) => ({
      title: `${T({ tr: "Epoch", en: "Epoch" })} ${i + 1}`,
      rows: series.map(([name, values, color]) => ({ value: format(values[i]), label: name, color })),
    }),
  });
}

const card = (title: string, sub: string | null, ...body: (Node | null)[]) =>
  h(
    "div",
    { class: "card yo-card" },
    h("div", { class: "card-head" }, h("div", {}, h("h2", { text: title }), sub ? h("p", { class: "sub", text: sub }) : null)),
    ...body,
  );

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

function intro(r: RunRecord): HTMLElement {
  const run = r.runs;
  const facts: [Text, string][] = [
    [
      { tr: "Veri", en: "Data" },
      T({
        tr: `${r.images.train} eğitim, ${r.images.val} doğrulama karesi; Las Vegas, Paris, Şanghay, Hartum`,
        en: `${r.images.train} training and ${r.images.val} validation tiles; Las Vegas, Paris, Shanghai, Khartoum`,
      }),
    ],
    [
      { tr: "Modeller", en: "Models" },
      T({
        tr: "YOLO26n-seg (bina, spor sahası) ve YOLO26n-sem (6 arazi sınıfı)",
        en: "YOLO26n-seg (building, sports field) and YOLO26n-sem (6 land-cover classes)",
      }),
    ],
    [
      { tr: "Ayarlar", en: "Settings" },
      T({
        tr: `${r.settings.epochs} epoch, görüntü ${r.settings.imgsz} px, batch ${r.settings.batch}; dikey çevirme ve 90°'ye kadar döndürme (kuş bakışında "yukarı" yok)`,
        en: `${r.settings.epochs} epochs, ${r.settings.imgsz} px images, batch ${r.settings.batch}; vertical flips and rotations up to 90° (a view from above has no "up")`,
      }),
    ],
    [{ tr: "Donanım", en: "Hardware" }, `${r.device} (6 GB)`],
    [
      { tr: "Süre", en: "Time" },
      T({
        tr: `${fmtDuration(run["seg-pretrained"].seconds)} (bina), ${fmtDuration(run["sem-pretrained"].seconds)} (arazi örtüsü), ${fmtDuration(run["seg-scratch"].seconds)} (sıfırdan)`,
        en: `${fmtDuration(run["seg-pretrained"].seconds)} (buildings), ${fmtDuration(run["sem-pretrained"].seconds)} (land cover), ${fmtDuration(run["seg-scratch"].seconds)} (from scratch)`,
      }),
    ],
  ];
  return h(
    "div",
    { class: "card yo-card" },
    h("h2", { text: T({ tr: "Bu modeller nasıl eğitildi?", en: "How were these models trained?" }) }),
    h("dl", { class: "yo-facts" }, ...facts.flatMap(([k, v]) => [h("dt", { text: T(k) }), h("dd", { text: v })])),
    h("p", {
      class: "sub",
      text: T({
        tr: "Bu sekme bizim eğitimimizin kaydıdır; tarayıcıda yeni bir eğitim başlatılmaz. YOLO'yu eğitmek bir ekran kartında yaklaşık bir saat ve 2 GB'a yakın veri indirmek ister; CNN bölümündeki küçük ağ gibi tarayıcıda eğitmek pratik değil. Aynı eğitimi Colab'da ya da kendi bilgisayarınızda tekrarlamak Kodlama sekmesinde anlatılacak.",
        en: "This tab is the record of our training; no new training starts in the browser. Training YOLO takes about an hour on a graphics card and close to 2 GB of downloaded data; unlike the small network of the CNN chapter, training it in a browser is not practical. How to repeat the same training on Colab or your own computer will be in the Coding tab.",
      }),
    }),
  );
}

function pretraining(r: RunRecord): HTMLElement {
  const host = chart(240);
  const p = r.runs["seg-pretrained"];
  const s = r.runs["seg-scratch"];
  const at = (run: Run, e: number) => run.curves["metrics/mAP50(M)"][e - 1];
  const rows = SEG_CLASSES.map((name, k) =>
    h(
      "tr",
      {},
      h("th", { scope: "row", text: T(name) }),
      h("td", { text: nf((p.final.maskAP50 as number[])[k]) }),
      h("td", { text: nf((s.final.maskAP50 as number[])[k]) }),
    ),
  );
  rows.push(
    h(
      "tr",
      { class: "yo-total" },
      h("th", { scope: "row", text: T({ tr: "ortalama (mAP50)", en: "mean (mAP50)" }) }),
      h("td", { text: nf(p.final.maskMAP50 as number) }),
      h("td", { text: nf(s.final.maskMAP50 as number) }),
    ),
  );
  const out = card(
    T({ tr: "Önceden eğitmenin faydası", en: "What pretraining is worth" }),
    T({
      tr: "Aynı veri, aynı ayarlar; tek fark başlangıç. Biri COCO veri setinde (günlük fotoğraflar) eğitilmiş ağırlıklarla başladı, öteki rastgele ağırlıklarla. Eğri: doğrulama setinde maske mAP50.",
      en: "Same data, same settings; only the start differs. One started from weights trained on COCO (everyday photos), the other from random weights. Curve: mask mAP50 on the validation set.",
    }),
    host,
    h(
      "div",
      { class: "yo-legend" },
      ...[0, 1].map((k) =>
        h("span", { class: "yo-key" }, h("i", { style: `background: ${SERIES[k]}` }), T(RUN_NAMES[k ? "seg-scratch" : "seg-pretrained"])),
      ),
    ),
    h("p", {
      class: "yo-big",
      text: T({
        tr: `10. epoch'ta: ${nf(at(p, 10))} ile ${nf(at(s, 10))}. Sıfırdan başlayan model 100. epoch'ta (${nf(at(s, 100))}) bile önceden eğitilmiş modelin 10. epoch'una yetişemedi.`,
        en: `At epoch 10: ${nf(at(p, 10))} against ${nf(at(s, 10))}. Even at epoch 100 (${nf(at(s, 100))}) the model that started from scratch had not caught up with the pretrained one at epoch 10.`,
      }),
    }),
    h(
      "div",
      { class: "yo-table-wrap" },
      h(
        "table",
        { class: "yo-table" },
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            h("th", { scope: "col", text: T({ tr: "maske AP50 (en iyi epoch)", en: "mask AP50 (best epoch)" }) }),
            h("th", { scope: "col", text: T({ tr: "önceden eğitilmiş", en: "pretrained" }) }),
            h("th", { scope: "col", text: T({ tr: "sıfırdan", en: "from scratch" }) }),
          ),
        ),
        h("tbody", {}, ...rows),
      ),
    ),
    h("p", {
      class: "sub",
      text: T({
        tr: "Spor sahaları binalardan çok daha zayıf: örnek az (eğitimde 529 saha, 10.974 bina) ve etiketleri OpenStreetMap'ten geliyor, yani gürültülü.",
        en: "Sports fields are much weaker than buildings: there are few of them (529 fields against 10,974 buildings in training) and their labels come from OpenStreetMap, so they are noisy.",
      }),
    }),
  );
  curves(
    host,
    T({ tr: "Maske mAP50, önceden eğitilmiş ve sıfırdan", en: "Mask mAP50, pretrained and from scratch" }),
    [
      [T(RUN_NAMES["seg-pretrained"]), p.curves["metrics/mAP50(M)"], SERIES[0]],
      [T(RUN_NAMES["seg-scratch"]), s.curves["metrics/mAP50(M)"], SERIES[1]],
    ],
    (v) => nf(v),
    0,
  );
  return out;
}

const LOSSES: Record<"box" | "seg" | "cls", { name: Text; about: Text }> = {
  box: {
    name: { tr: "Kutu", en: "Box" },
    about: { tr: "Kutunun yeri ve boyu ne kadar yanlış (IoU'ya dayalı).", en: "How wrong the box's place and size are (based on IoU)." },
  },
  seg: {
    name: { tr: "Maske", en: "Mask" },
    about: {
      tr: "Maskenin pikselleri ne kadar yanlış (ikili çapraz entropi).",
      en: "How wrong the mask's pixels are (binary cross-entropy).",
    },
  },
  cls: {
    name: { tr: "Sınıf", en: "Class" },
    about: {
      tr: "Sınıf skorları ne kadar yanlış: bina mı, saha mı, hiçbiri mi.",
      en: "How wrong the class scores are: building, field or nothing.",
    },
  },
};

function losses(r: RunRecord): HTMLElement {
  const host = chart(220);
  const about = h("p", { class: "sub" });
  const draw = () => {
    const c = r.runs["seg-pretrained"].curves;
    about.textContent = T(LOSSES[state.loss].about);
    curves(
      host,
      T({ tr: "Kayıp, eğitim ve doğrulama", en: "Loss, training and validation" }),
      [
        [T({ tr: "eğitim", en: "training" }), c[`train/${state.loss}_loss`], SERIES[0]],
        [T({ tr: "doğrulama", en: "validation" }), c[`val/${state.loss}_loss`], SERIES[1]],
      ],
      (v) => nf(v),
    );
  };
  const pick = segmented(
    (Object.keys(LOSSES) as ("box" | "seg" | "cls")[]).map((k) => [k, T(LOSSES[k].name)]),
    state.loss,
    (v) => {
      state.loss = v;
      draw();
    },
    T({ tr: "Kayıp terimi", en: "Loss term" }),
  );
  const out = card(
    T({ tr: "Kayıp: üç terim", en: "Loss: three terms" }),
    T({
      tr: "YOLO'nun kaybı birkaç terimin toplamıdır; her biri ayrı bir hatayı ölçer. Önceden eğitilmiş modelin eğrileri.",
      en: "YOLO's loss is a sum of terms, each measuring a different mistake. Curves of the pretrained model.",
    }),
    pick,
    about,
    host,
  );
  draw();
  return out;
}

function metrics(r: RunRecord): HTMLElement {
  const host = chart(220);
  const c = r.runs["seg-pretrained"].curves;
  const out = card(
    T({ tr: "Doğrulamada ölçümler", en: "Scores on the validation set" }),
    T({
      tr: "Her epoch'un sonunda 131 doğrulama karesinde ölçüldü (maske). mAP50-95, IoU eşiğini 0,50'den 0,95'e kadar artırarak ortalama alır; sınırları tam tutturmayı da ödüllendirir.",
      en: "Measured on the 131 validation tiles after every epoch (masks). mAP50-95 averages over IoU thresholds from 0.50 to 0.95, so it also rewards outlines that fit exactly.",
    }),
    host,
  );
  curves(
    host,
    T({ tr: "Kesinlik, duyarlılık, mAP", en: "Precision, recall, mAP" }),
    [
      [T({ tr: "kesinlik", en: "precision" }), c["metrics/precision(M)"], SERIES[0]],
      [T({ tr: "duyarlılık", en: "recall" }), c["metrics/recall(M)"], SERIES[1]],
      ["mAP50", c["metrics/mAP50(M)"], SERIES[2]],
      ["mAP50-95", c["metrics/mAP50-95(M)"], SERIES[3]],
    ],
    (v) => nf(v),
    0,
  );
  return out;
}

function landCover(r: RunRecord): HTMLElement {
  const host = chart(200);
  const run = r.runs["sem-pretrained"];
  const iou = (run.final.iou as number[] | null) ?? [];
  const bars = SEM_CLASSES.map((name, k) =>
    h(
      "div",
      { class: "yo-bar" },
      h("span", { text: T(name) }),
      h(
        "span",
        { class: "yo-bar-track" },
        h("i", { style: `width: ${100 * (iou[k] ?? 0)}%; background: ${css(SEM_RGB[k] ?? [128, 128, 128])}` }),
      ),
      h("output", { text: iou[k] === undefined ? "–" : nf(iou[k]) }),
    ),
  );
  const out = card(
    T({ tr: "Arazi örtüsü modeli", en: "The land-cover model" }),
    T({
      tr: `Her piksele bir sınıf veren ikinci model. En iyi epoch'ta mIoU ${nf(run.final.mIoU as number)}, doğru sınıflanan piksel ${fmtPct(100 * (run.final.pixelAccuracy as number), 0)}.`,
      en: `The second model, which gives every pixel a class. At its best epoch mIoU ${nf(run.final.mIoU as number)}, pixels classified correctly ${fmtPct(100 * (run.final.pixelAccuracy as number), 0)}.`,
    }),
    host,
    h("h3", { text: T({ tr: "Sınıf başına IoU", en: "IoU per class" }) }),
    h("div", { class: "yo-bars" }, ...bars),
    h("p", {
      class: "sub",
      text: T({
        tr: "Bina en iyisi: sınırları SpaceNet'te elle çizildi. Otopark en zayıfı: OpenStreetMap'te çoğu eksik ya da bir parkın içinde kalmış. Etiketin kalitesi modelin tavanıdır.",
        en: "Buildings do best: their outlines were drawn by hand in SpaceNet. Parking does worst: in OpenStreetMap many are missing or sit inside a park. The quality of the labels is the model's ceiling.",
      }),
    }),
  );
  curves(
    host,
    T({ tr: "mIoU ve piksel doğruluğu", en: "mIoU and pixel accuracy" }),
    [
      ["mIoU", run.curves["metrics/mIoU"], SERIES[0]],
      [T({ tr: "piksel doğruluğu", en: "pixel accuracy" }), run.curves["metrics/pixel_acc"], SERIES[1]],
    ],
    (v) => nf(v),
    0,
  );
  return out;
}

export interface Tile {
  image: HTMLImageElement;
  width: number;
  height: number;
  city: string;
}

/** Which run and which saved epoch a viewer shows. */
export interface ShotChoice {
  run: RunName;
  shot: number;
}

/**
 * The record's validation tile predicted with the weights saved after a chosen epoch: the Training tab
 * shows it, and Theory section 6 shows a smaller copy with its own choice.
 */
export function epochViewer(
  r: RunRecord,
  tile: Tile,
  choice: ShotChoice,
  runs: RunName[],
): { controls: HTMLElement; canvas: HTMLCanvasElement; note: HTMLElement } {
  const view = new TileView(
    T({ tr: "Aynı doğrulama karesi, seçilen epoch'taki tahmin", en: "The same validation tile, predicted at the chosen epoch" }),
  );
  const note = h("p", { class: "yo-big", "aria-live": "polite" });
  const redraw = async () => {
    const { image, width, height } = tile;
    const shot = r.snapshots[choice.run][choice.shot];
    if ("objects" in shot) {
      const masks = shot.objects.map((o) => ({ mask: fillPolygon(o.xy, width, height), rgb: SEG_RGB[o.cls] }));
      view.show(image, width, height, [
        {
          canvas: maskOverlay(Math.max(width, height), masks, 0.45),
          boxes: shot.objects.map((o) => ({ box: o.box, rgb: SEG_RGB[o.cls] })),
        },
      ]);
      const counts = SEG_CLASSES.map((name, k) => `${shot.objects.filter((o) => o.cls === k).length} ${T(name)}`).join(", ");
      note.textContent = T({ tr: `${shot.epoch}. epoch: ${counts}.`, en: `Epoch ${shot.epoch}: ${counts}.` });
    } else {
      const classes = await unpack(shot.classes);
      view.show(image, width, height, [{ canvas: classOverlay(classes, shot.width, shot.height) }]);
      note.textContent = T({ tr: `${shot.epoch}. epoch.`, en: `Epoch ${shot.epoch}.` });
    }
  };
  const runPick = segmented(
    runs.map((k) => [k, T(RUN_NAMES[k])]),
    choice.run,
    (v) => {
      choice.run = v;
      void redraw();
    },
    T({ tr: "Model", en: "Model" }),
  );
  const shots = segmented(
    r.snapshots["seg-pretrained"].map((s, i) => [i, String(s.epoch)]),
    choice.shot,
    (v) => {
      choice.shot = v;
      void redraw();
    },
    T({ tr: "Epoch", en: "Epoch" }),
  );
  shots.classList.add("digits");
  const controls = h(
    "div",
    { class: "yo-shot-controls" },
    runPick,
    h("div", { class: "yo-epochs" }, h("span", { text: T({ tr: "Epoch:", en: "Epoch:" }) }), shots),
  );
  void redraw();
  return { controls, canvas: view.canvas, note };
}

/** The training record and its validation tile (loaded once). */
export async function loadRecord(): Promise<{ record: RunRecord; tile: Tile | null }> {
  const [record, samples] = await Promise.all([loadData<RunRecord>("yolo-run"), loadSamples()]);
  const s = samples.samples.find((x) => x.name === record.snapshotTile);
  if (!s) return { record, tile: null };
  const { image } = await decodeImage(`data:image/jpeg;base64,${s.jpeg}`);
  return { record, tile: { image, width: s.width, height: s.height, city: s.city } };
}

function epochs(r: RunRecord): HTMLElement {
  const city = state.tile ? T(CITIES[state.tile.city]) : "";
  const parts = state.tile ? epochViewer(r, state.tile, state, Object.keys(RUN_NAMES) as RunName[]) : null;
  return card(
    T({ tr: "Aynı kare, epoch epoch", en: "The same tile, epoch by epoch" }),
    T({
      tr: `${city} · ${r.snapshotTile.split("_")[1]}: eğitimde hiç görülmeyen bir doğrulama karesi. Eğitim sırasında kaydedilen ağırlıklarla tahmin edildi (güven eşiği 0,25).`,
      en: `${city} · ${r.snapshotTile.split("_")[1]}: a validation tile never seen in training. Predicted with the weights saved during training (confidence threshold 0.25).`,
    }),
    parts?.controls ?? null,
    parts?.canvas ?? null,
    parts?.note ?? null,
  );
}

function render(): void {
  const r = state.record;
  if (!r) return;
  ($("#yt-root") as HTMLElement).replaceChildren(
    h("div", { class: "yo-training" }, intro(r), epochs(r), pretraining(r), losses(r), metrics(r), landCover(r)),
  );
}

async function start(): Promise<void> {
  if (started) return;
  started = true;
  try {
    const { record, tile } = await loadRecord();
    state.record = record;
    state.tile = tile;
  } catch (e) {
    ($("#yt-root") as HTMLElement).replaceChildren(h("p", { class: "banner", role: "alert", text: (e as Error).message }));
    return;
  }
  render();
}

document.addEventListener("tab-shown", (e) => {
  if ((e as CustomEvent).detail === "train") void start();
});
document.addEventListener("lang-change", () => {
  if (state.record) render();
});
