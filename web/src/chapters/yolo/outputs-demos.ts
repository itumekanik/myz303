/**
 * The demos of the Outputs tab (outputs.ts). Two of them need no data: the damage survey (a 2 × 2
 * confusion matrix with a threshold) and the toy tile (detection's 3 × 3 matrix, matched the way
 * Ultralytics does it). The others read data/yolo-outputs.js, written by tools/export_yolo_assets.py
 * from runs/…/seg-pretrained: the files, the console, results.csv, label statistics, sample
 * batches, the confusion matrix at thresholds 0.001 and 0.01 … 0.95 with example crops, the P, R, F1
 * and PR curves, and the land-cover model's pixel matrix.
 */
import { fmtInt, fmtNum, h, lineChart, linearTicks, niceStep, resizeObserver } from "../../shared/core.js";
import { KIT } from "../../shared/kit.js";
import { loadData, unpack, type Packed } from "../../engines/data.ts";
import type { Text } from "../../shared/chapters.ts";
import { SEG_CLASSES, SEM_CLASSES, T } from "./page.ts";
import { range } from "./theory-demos.ts";

const { segmented } = KIT as {
  segmented: <V extends string>(
    options: [V, string][],
    current: V,
    pick: (v: V) => void,
    aria: string,
  ) => HTMLElement & { set: (v: V) => void };
};

// ---------------------------------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------------------------------

interface Example {
  tile: string;
  jpeg: string;
  size: [number, number];
  pred: number[] | null;
  truth: number[] | null;
  score: number | null;
}

interface CurveSet {
  confidence: number[];
  precision: number[][];
  recall: number[][];
  f1: number[][];
  prRecall: number[];
  prPrecision: number[][];
  bestF1: { confidence: number; f1: number };
  table: { precision: number[]; recall: number[]; ap50: number[]; ap50_95: number[] };
}

export interface Outputs {
  files: { path: string; bytes: number }[];
  console: {
    start: string[];
    header: string;
    progress: string;
    epochs: { epoch: number; train: string; valHeader: string; val: string }[];
    end: string[];
    final: string[];
  } | null;
  results: Record<string, number[]>;
  labels: { counts: number[]; boxes: Packed };
  batches: Record<"first" | "last" | "valLabels" | "valPred", { name: string; jpeg: string }>;
  matrix: {
    names: string[];
    confs: number[];
    counts: number[][];
    images: number[];
    tiles: number;
    exampleConf: number;
    examples: Record<string, { count: number; items: Example[] }>;
  };
  curves: Record<"B" | "M", CurveSet>;
  semMatrix: { names: string[]; counts: number[][] };
}

let loading: Promise<Outputs> | null = null;
let wanted: ((v: Outputs) => void)[] = [];
/** The data, once the tab is opened (about 1.5 MB); demos built earlier wait for it. */
export const outputsData = (): Promise<Outputs> => loading ?? new Promise<Outputs>((resolve) => wanted.push(resolve));
export function loadOutputs(): Promise<Outputs> {
  if (!loading) {
    loading = loadData<Outputs>("yolo-outputs");
    const waiting = wanted;
    wanted = [];
    void loading.then((d) => waiting.forEach((f) => f(d)));
  }
  return loading;
}

// ---------------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------------

const nf = (v: number, d = 2) => fmtNum(v, d) as string;
const int = (v: number) => fmtInt(v) as string;
const pct = (v: number, d = 0) => T({ tr: `%${nf(100 * v, d)}`, en: `${nf(100 * v, d)}%` });
const ratio = (a: number, b: number) => (b ? a / b : 0);
const f1Of = (p: number, r: number) => (p + r ? (2 * p * r) / (p + r) : 0);
const demo = (...children: (Node | null)[]) => h("div", { class: "t-demo yo-demo" }, ...children);
const caption = (text: Text) => h("p", { class: "sub yo-caption", text: T(text) });
const live = () => h("p", { class: "yo-big", "aria-live": "polite" });

/** A demo that shows "loading" until the tab's data is there. */
function withData(build: (d: Outputs) => (Node | null)[]): HTMLElement {
  const box = demo(h("p", { class: "placeholder", text: T({ tr: "Çıktılar yükleniyor…", en: "Loading the outputs…" }) }));
  void outputsData()
    .then((d) => box.replaceChildren(...build(d)))
    .catch((e: Error) => box.replaceChildren(h("p", { class: "banner", role: "alert", text: e.message })));
  return box;
}

function chart(height = 220): HTMLElement {
  const host = h("div", { class: "chart", "data-height": height }) as HTMLElement;
  resizeObserver.observe(host);
  return host;
}

const sizeText = (bytes: number) => (bytes >= 1e6 ? `${nf(bytes / 1e6, 1)} MB` : bytes >= 1e3 ? `${nf(bytes / 1e3, 0)} kB` : `${bytes} B`);

/** The colour of a matrix cell: the accent, stronger for a larger share (at most 55%, so the text stays readable). */
function heat(cell: HTMLElement, share: number): void {
  const p = Math.round(Math.min(1, Math.max(0, share)) * 55);
  cell.style.background = p ? `color-mix(in srgb, var(--accent) ${p}%, var(--surface))` : "";
}

// ---------------------------------------------------------------------------------------------------
// 1. The folder
// ---------------------------------------------------------------------------------------------------

type When = "start" | "epoch" | "mosaic" | "end";
const WHEN: Record<When, Text> = {
  start: { tr: "Eğitim başlarken", en: "When training starts" },
  epoch: { tr: "Her epoch'un sonunda", en: "After every epoch" },
  mosaic: { tr: "Son 10 epoch başlarken (mozaik kapanınca)", en: "When the last 10 epochs start (mosaic off)" },
  end: { tr: "Eğitim bitince (best.pt ile son doğrulama)", en: "When training ends (a last validation with best.pt)" },
};
const KINDS: { re: RegExp; when: When; what: Text; section: string | null }[] = [
  {
    re: /^args\.yaml$/,
    when: "start",
    what: { tr: "Eğitimin bütün ayarları: aynı eğitimi tekrarlamak için", en: "Every setting of the run: to repeat the same training" },
    section: "yr-console",
  },
  {
    re: /^labels\.jpg$/,
    when: "start",
    what: { tr: "Etiketlerin özeti: sınıf sayıları, konumlar, boyutlar", en: "A summary of the labels: class counts, positions, sizes" },
    section: "yr-labels",
  },
  {
    re: /^train_batch[0-2]\.jpg$/,
    when: "start",
    what: { tr: "Modelin gördüğü ilk üç yığın (veri artırmayla)", en: "The first three batches the model sees (augmented)" },
    section: "yr-batches",
  },
  {
    re: /^results\.csv$/,
    when: "epoch",
    what: { tr: "Her epoch'un kayıpları ve ölçümleri, bir satır", en: "Every epoch's losses and scores, one row each" },
    section: "yr-results",
  },
  {
    re: /^weights\/last\.pt$/,
    when: "epoch",
    what: {
      tr: "Son epoch'un ağırlıkları (yarıda kalan eğitim buradan sürer)",
      en: "The last epoch's weights (an interrupted run resumes from here)",
    },
    section: "yr-results",
  },
  {
    re: /^weights\/best\.pt$/,
    when: "epoch",
    what: {
      tr: "Doğrulamada en iyi epoch'un ağırlıkları: kullanılacak model",
      en: "The weights of the best epoch on validation: the model to use",
    },
    section: "yr-results",
  },
  {
    re: /^weights\/epoch\d+\.pt$/,
    when: "epoch",
    what: { tr: "Ara kayıtlar (save_period ayarıyla)", en: "Snapshots along the way (the save_period setting)" },
    section: null,
  },
  {
    re: /^train_batch\d{3,}\.jpg$/,
    when: "mosaic",
    what: { tr: "Mozaiksiz yığınlar: son epoch'larda görüntüler bütün", en: "Batches without mosaic: whole tiles in the last epochs" },
    section: "yr-batches",
  },
  { re: /^results\.png$/, when: "end", what: { tr: "results.csv'nin grafikleri", en: "results.csv as charts" }, section: "yr-results" },
  {
    re: /^confusion_matrix(_normalized)?\.png$/,
    when: "end",
    what: { tr: "Karışıklık matrisi: sayılar ve sütuna göre oranlar", en: "The confusion matrix: counts and shares per column" },
    section: "yr-cm-real",
  },
  {
    re: /^(Box|Mask)(F1|PR|P|R)_curve\.png$/,
    when: "end",
    what: {
      tr: "P, R, F1 ve PR eğrileri; kutular (Box) ve maskeler (Mask) için",
      en: "The P, R, F1 and PR curves, for boxes (Box) and masks (Mask)",
    },
    section: "yr-curves",
  },
  {
    re: /^val_batch\d+_(labels|pred)\.jpg$/,
    when: "end",
    what: {
      tr: "Doğrulama karelerinde doğru etiketler ve tahminler yan yana",
      en: "True labels and predictions on validation tiles, side by side",
    },
    section: "yr-batches",
  },
];

/** The Outputs tab's section numbers, for the links of the folder. */
const SECTION_NUMBER: Record<string, number> = {
  "yr-console": 2,
  "yr-results": 3,
  "yr-labels": 4,
  "yr-batches": 5,
  "yr-cm-real": 8,
  "yr-curves": 10,
};

export function folderDemo(): HTMLElement {
  return withData((d) => {
    const groups = (Object.keys(WHEN) as When[]).map((when) => {
      const rows = KINDS.filter((k) => k.when === when).flatMap((k) => {
        const files = d.files.filter((f) => k.re.test(f.path));
        if (!files.length) return [];
        const bytes = files.reduce((s, f) => s + f.bytes, 0);
        const name = files.length > 2 ? `${files[0].path} … (${files.length})` : files.map((f) => f.path).join(", ");
        return [
          h(
            "li",
            { class: "yr-file" },
            h("code", { text: name }),
            h("span", { class: "yr-size", text: sizeText(bytes) }),
            h("span", { class: "yr-what", text: T(k.what) }),
            k.section
              ? h("a", {
                  href: "#",
                  "data-go": `outputs:${k.section}`,
                  class: "yr-go",
                  text: T({ tr: `${SECTION_NUMBER[k.section]}. kısım →`, en: `section ${SECTION_NUMBER[k.section]} →` }),
                })
              : null,
          ),
        ];
      });
      return h("div", { class: "yr-when" }, h("h3", { text: T(WHEN[when]) }), h("ul", { class: "yr-files" }, ...rows));
    });
    const total = d.files.reduce((s, f) => s + f.bytes, 0);
    return [
      h("div", { class: "yr-tree" }, h("div", { class: "yr-root", text: "runs/yolo/seg-pretrained/" }), ...groups),
      caption({
        tr: `Bizim klasörde ${d.files.length} dosya, toplam ${sizeText(total)}; çoğu ara ağırlık kayıtları. ONNX dosyaları burada sayılmadı: onları eğitimden sonra "export" komutu yazar.`,
        en: `${d.files.length} files in our folder, ${sizeText(total)} in all; most of it is weight snapshots. The ONNX files are left out: the "export" command writes them after training.`,
      }),
    ];
  });
}

// ---------------------------------------------------------------------------------------------------
// 2. The console
// ---------------------------------------------------------------------------------------------------

const TRAIN_COLUMNS: [string, Text][] = [
  [
    "Epoch",
    {
      tr: "Kaçıncı epoch / toplam. Bir epoch, 528 eğitim karesinin hepsinin bir kez görülmesidir.",
      en: "Which epoch / how many. One epoch shows every one of the 528 training tiles once.",
    },
  ],
  [
    "GPU_mem",
    {
      tr: "Ekran kartında kullanılan bellek. 6 GB'lık kartta 3–5 GB: batch'i büyütmeye fazla yer yok.",
      en: "Memory used on the graphics card. 3–5 GB on a 6 GB card: not much room for a larger batch.",
    },
  ],
  [
    "box_loss",
    {
      tr: "Kutu kaybı (CIoU): kutular gerçek kutulara ne kadar uymuyor. Bu ve sonraki kayıplar, epoch içindeki yığınların ortalamasıdır; küçülmeli.",
      en: "The box loss (CIoU): how badly the boxes fit the true boxes. This and the following losses are averages over the epoch's batches; they should shrink.",
    },
  ],
  [
    "seg_loss",
    {
      tr: "Maske kaybı: her pikselin maske içinde mi dışında mı olduğuna dair ikili çapraz entropi.",
      en: "The mask loss: binary cross-entropy on whether each pixel is inside the mask.",
    },
  ],
  [
    "cls_loss",
    {
      tr: "Sınıf kaybı: bina mı, spor sahası mı, hiçbiri mi. İlk epoch'larda hızla düşer.",
      en: "The class loss: building, sports field or neither. It falls fast in the first epochs.",
    },
  ],
  [
    "l1_loss",
    {
      tr: "Kutu kenarlarına olan dört uzaklık üzerindeki L1 kaybı. YOLO26'da önceki sürümlerdeki DFL'nin yerini aldı; değeri küçük ama ağırlığı (1,5) var.",
      en: "The L1 loss on the four distances to the box edges. In YOLO26 it replaces the DFL of earlier versions; its value is small but it has a weight (1.5).",
    },
  ],
  [
    "sem_loss",
    {
      tr: "Yardımcı bir kayıp: eğitim sırasında modelin ucuna takılan anlamsal segmentasyon başının kaybı (maskelerden kurulur). Yalnızca eğitimde vardır; doğrulamada 0 yazılır.",
      en: "An auxiliary loss: that of a semantic segmentation head attached to the model during training (built from the masks). It exists only in training; validation writes 0.",
    },
  ],
  [
    "Instances",
    {
      tr: "Epoch'un son yığınındaki nesne sayısı. Kareden kareye değişir; bir şey ölçmez.",
      en: "The number of objects in the epoch's last batch. It changes from tile to tile; it measures nothing.",
    },
  ],
  ["Size", { tr: "Girişin boyu: 640 × 640.", en: "The input size: 640 × 640." }],
  [
    "progress",
    {
      tr: 'İlerleme: 132 yığın (528 kare / batch 4), saniyede kaç yığın (it/s) ve geçen süre. Epoch sürerken "geçen<kalan" yazar.',
      en: 'Progress: 132 batches (528 tiles / batch 4), batches per second (it/s) and the time taken. While the epoch runs it shows "elapsed<remaining".',
    },
  ],
];
const VAL_COLUMNS: [string, Text][] = [
  [
    "Class",
    {
      tr: '"all": iki sınıfın ortalaması. Eğitim bitince her sınıf için ayrı bir satır da yazılır.',
      en: '"all": the mean of the two classes. When training ends, every class gets its own row too.',
    },
  ],
  ["Images", { tr: "Doğrulama karesi sayısı (131).", en: "The number of validation tiles (131)." }],
  [
    "Instances",
    {
      tr: "Doğrulamadaki gerçek nesne sayısı: 2329 bina + 172 spor sahası = 2501.",
      en: "The true objects in validation: 2329 buildings + 172 sports fields = 2501.",
    },
  ],
  [
    "Box P",
    {
      tr: "Kutuların kesinliği. Dikkat: Ultralytics P ile R'yi, ortalama F1'in en yüksek olduğu güven eşiğinde yazar (bizde ≈ 0,26), 0,25'te değil.",
      en: "Precision of the boxes. Note: Ultralytics writes P and R at the confidence threshold where the mean F1 is highest (≈ 0.26 for us), not at 0.25.",
    },
  ],
  ["Box R", { tr: "Kutuların duyarlılığı, aynı eşikte.", en: "Recall of the boxes, at the same threshold." }],
  [
    "Box mAP50",
    {
      tr: "Kutu IoU'su 0,5 eşiğiyle mAP: eşikten bağımsız, bütün PR eğrisinin altındaki alan.",
      en: "mAP with a box IoU threshold of 0.5: independent of the threshold, the area under the whole PR curve.",
    },
  ],
  [
    "Box mAP50-95",
    {
      tr: "0,50'den 0,95'e on IoU eşiğinin ortalaması: sınırları tam tutturmayı da ister.",
      en: "The mean over ten IoU thresholds from 0.50 to 0.95: it also asks for outlines that fit exactly.",
    },
  ],
  [
    "Mask P",
    {
      tr: "Aynı ölçümler maskelerle (IoU maskelerden hesaplanır). Bina sayımı için asıl bakılacak sütunlar bunlar.",
      en: "The same scores with masks (IoU computed from masks). For counting buildings, these are the columns to read.",
    },
  ],
  ["Mask R", { tr: "Maskelerin duyarlılığı.", en: "Recall of the masks." }],
  [
    "Mask mAP50",
    {
      tr: "Maske mAP50: bölümde bina 0,69, spor sahası 0,30, ortalama 0,50.",
      en: "Mask mAP50: building 0.69, sports field 0.30, mean 0.50 in this chapter.",
    },
  ],
  [
    "Mask mAP50-95",
    {
      tr: "Maske mAP50-95. best.pt seçilirken bu ve kutu mAP50-95'in toplamına bakılır.",
      en: "Mask mAP50-95. best.pt is picked by the sum of this and the box mAP50-95.",
    },
  ],
];

function terminalTable(columns: [string, Text][], cells: string[], current: { col: string }, explain: (col: string) => void): HTMLElement {
  const head = h(
    "tr",
    {},
    ...columns.map(([name]) => {
      const b = h("button", { type: "button", "aria-pressed": String(name === current.col), text: name === "progress" ? "…" : name });
      b.setAttribute("aria-label", name === "progress" ? T({ tr: "ilerleme", en: "progress" }) : name);
      b.addEventListener("click", () => explain(name));
      return h("th", { scope: "col" }, b);
    }),
  );
  return h(
    "div",
    { class: "yr-term-wrap" },
    h(
      "table",
      { class: "yr-term" },
      h("thead", {}, head),
      h("tbody", {}, h("tr", {}, ...cells.map((c, i) => h("td", { class: columns[i]?.[0] === current.col ? "on" : "", text: c })))),
    ),
  );
}

export function consoleDemo(): HTMLElement {
  return withData((d) => {
    const c = d.console;
    if (!c) return [caption({ tr: "Bu eğitimin konsol çıktısı kaydedilmemiş.", en: "This run's console output was not saved." })];
    let shown = c.epochs.length - 1;
    const current = { col: "box_loss" };
    const host = h("div", { class: "yr-console" });
    const note = live();
    const draw = () => {
      const e = c.epochs[shown];
      const [before, after] = e.train.split(": ");
      const trainCells = [...before.trim().split(/\s+/), after.replace(/\s*━+|\s*─+|╸/g, "").trim()];
      const valCells = e.val.trim().split(/\s+/);
      const explain = (col: string) => {
        current.col = col;
        draw();
      };
      host.replaceChildren(
        h("p", { class: "yr-term-label", text: T({ tr: "Eğitim satırı (epoch sonunda)", en: "Training line (end of the epoch)" }) }),
        terminalTable(TRAIN_COLUMNS, trainCells, current, explain),
        h("p", { class: "yr-term-label", text: T({ tr: "Doğrulama satırı (131 kare)", en: "Validation line (131 tiles)" }) }),
        terminalTable(VAL_COLUMNS, valCells, current, explain),
      );
      const text = [...TRAIN_COLUMNS, ...VAL_COLUMNS].find(([n]) => n === current.col)?.[1];
      note.textContent = text ? `${current.col === "progress" ? T({ tr: "İlerleme", en: "Progress" }) : current.col}: ${T(text)}` : "";
    };
    const pick = segmented(
      c.epochs.map((e, i) => [String(i), `${e.epoch}`]),
      String(shown),
      (v) => {
        shown = Number(v);
        draw();
      },
      T({ tr: "Epoch", en: "Epoch" }),
    );
    pick.classList.add("digits");
    draw();
    return [
      h("div", { class: "yo-epochs" }, h("span", { text: T({ tr: "Epoch:", en: "Epoch:" }) }), pick),
      host,
      note,
      h(
        "details",
        { class: "yr-more" },
        h("summary", { text: T({ tr: "Başlangıç ve bitiş satırları", en: "The first and last lines" }) }),
        h("pre", { class: "yr-pre", text: [...c.start, "…", c.header, c.progress, "…", ...c.end, ...c.final].join("\n") }),
      ),
      caption({ tr: "Sütun adına tıklayınca ne anlama geldiği yazılır.", en: "Click a column name to see what it means." }),
    ];
  });
}

// ---------------------------------------------------------------------------------------------------
// 3. results.csv
// ---------------------------------------------------------------------------------------------------

/** best.pt: the epoch with the highest fitness = mAP50-95(B) + mAP50-95(M). */
export function bestEpoch(r: Record<string, number[]>): { epoch: number; fitness: number[] } {
  const fitness = r["metrics/mAP50-95(B)"].map((v, i) => v + r["metrics/mAP50-95(M)"][i]);
  return { epoch: fitness.indexOf(Math.max(...fitness)) + 1, fitness };
}

const PANELS: { key: string; name: Text; series: [Text | string, string][] }[] = [
  {
    key: "box",
    name: { tr: "Kutu kaybı", en: "Box loss" },
    series: [
      [{ tr: "eğitim", en: "training" }, "train/box_loss"],
      [{ tr: "doğrulama", en: "validation" }, "val/box_loss"],
    ],
  },
  {
    key: "seg",
    name: { tr: "Maske kaybı", en: "Mask loss" },
    series: [
      [{ tr: "eğitim", en: "training" }, "train/seg_loss"],
      [{ tr: "doğrulama", en: "validation" }, "val/seg_loss"],
    ],
  },
  {
    key: "cls",
    name: { tr: "Sınıf kaybı", en: "Class loss" },
    series: [
      [{ tr: "eğitim", en: "training" }, "train/cls_loss"],
      [{ tr: "doğrulama", en: "validation" }, "val/cls_loss"],
    ],
  },
  {
    key: "sem",
    name: { tr: "Yardımcı kayıp (sem)", en: "Auxiliary loss (sem)" },
    series: [[{ tr: "eğitim", en: "training" }, "train/sem_loss"]],
  },
  {
    key: "map",
    name: { tr: "mAP ve uygunluk", en: "mAP and fitness" },
    series: [
      ["mAP50 (M)", "metrics/mAP50(M)"],
      ["mAP50-95 (B)", "metrics/mAP50-95(B)"],
      ["mAP50-95 (M)", "metrics/mAP50-95(M)"],
      [{ tr: "uygunluk", en: "fitness" }, "fitness"],
    ],
  },
  {
    key: "pr",
    name: { tr: "Kesinlik, duyarlılık", en: "Precision, recall" },
    series: [
      ["P (M)", "metrics/precision(M)"],
      ["R (M)", "metrics/recall(M)"],
    ],
  },
  { key: "lr", name: { tr: "Öğrenme oranı", en: "Learning rate" }, series: [["lr", "lr/pg0"]] },
];
const PANEL_NOTES: Record<string, Text> = {
  box: {
    tr: "İki eğri birlikte iniyor: model ezberlemiyor, öğreniyor. Doğrulama kaybı eğitiminkinden düşük; çünkü eğitim görüntüleri döndürülüp mozaiklenerek zorlaştırılıyor. 91. epoch'ta mozaik kapanınca eğitim eğrisi bir an sıçrar.",
    en: "Both curves fall together: the model learns rather than memorises. The validation loss is below the training loss because training images are made harder with rotations and mosaics. When the mosaic stops at epoch 91 the training curve jumps for a moment.",
  },
  seg: {
    tr: "Maske kaybı 100. epoch'ta hâlâ iniyor: daha uzun eğitim maskeleri biraz daha iyileştirebilirdi.",
    en: "The mask loss is still falling at epoch 100: training longer could improve the masks a little more.",
  },
  cls: {
    tr: "Sınıf kaybı ilk birkaç epoch'ta çok hızlı düşer: COCO'dan gelen ağırlıklar \"nesne var mı\" sorusunu zaten biliyor, yalnızca yeni iki sınıfı öğreniyor.",
    en: "The class loss falls very fast in the first few epochs: the weights from COCO already know whether there is an object; they only learn the two new classes.",
  },
  sem: {
    tr: "Yalnızca eğitimde hesaplanır. 91. epoch'tan sonra belirgin düşer: mozaik kalkınca görüntüler bütün ve daha kolaydır.",
    en: "Computed only in training. It drops clearly after epoch 91: without the mosaic the tiles are whole and easier.",
  },
  map: {
    tr: "Uygunluk (fitness) = mAP50-95(B) + mAP50-95(M). best.pt, bu toplamın en yüksek olduğu epoch'un ağırlıklarıdır. Eğri dalgalı: tek bir epoch'a bakıp karar vermeyin.",
    en: "Fitness = mAP50-95(B) + mAP50-95(M). best.pt holds the weights of the epoch where this sum is highest. The curve is noisy: do not judge from a single epoch.",
  },
  pr: {
    tr: "Her epoch'ta, o epoch'un en iyi F1 eşiğinde ölçülür; eşik epoch'tan epoch'a değiştiği için zikzak yapar. İlk epoch'taki yüksek kesinlik yanıltıcıdır: model çok az şey bulur, bulduğunun çoğu doğrudur.",
    en: "Measured every epoch at that epoch's best F1 threshold; since the threshold moves between epochs, the curves zigzag. The high precision of the first epoch misleads: the model finds very little, and most of it is right.",
  },
  lr: {
    tr: "Önce ısınma (warmup): ilk 3 epoch'ta oran yükselir, yeni baş rastgele başlarken büyük adımlar ağı bozmasın diye. Sonra doğrusal olarak sıfıra yaklaşır: sona doğru küçük adımlarla ince ayar.",
    en: "First a warmup: the rate rises during the first 3 epochs, so big steps do not spoil the network while the new head starts from random. Then it falls linearly towards zero: fine-tuning with small steps near the end.",
  },
};

export function resultsDemo(): HTMLElement {
  return withData((d) => {
    const r: Record<string, number[]> = { ...d.results };
    const best = bestEpoch(d.results);
    r.fitness = best.fitness;
    const n = r["train/box_loss"].length;
    let panel = PANELS[0];
    const host = chart(220);
    const note = live();
    const legend = h("div", { class: "yo-legend" });
    const colors = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--critical)"];
    const draw = () => {
      const series = panel.series.map(([name, key], i) => ({
        name: typeof name === "string" ? name : T(name),
        values: r[key],
        color: colors[i],
      }));
      const all = series.flatMap((s) => s.values);
      const low = Math.min(...all);
      const hi = Math.max(...all);
      const small = hi < 0.01;
      const format = (v: number) => (small ? nf(v * 1000, 1) : nf(v, hi < 1 ? 2 : 1));
      let step = niceStep(hi - low, 4);
      const lo = Math.floor(low / step) * step;
      step = niceStep(hi - lo, 4);
      const top = Math.ceil(hi / step) * step;
      lineChart(host, {
        label: T(panel.name),
        empty: "",
        x: {
          min: 1,
          max: n,
          ticks: [1, 20, 40, 60, 80, 100].filter((v) => v <= n).map((v) => ({ v, label: String(v) })),
          grid: [best.epoch],
        },
        y: { min: lo, max: top, ticks: linearTicks(lo, top, step, format) },
        series: series.map((s) => ({ name: s.name, points: s.values.map((v, i) => [i + 1, v]), color: s.color })),
        tooltip: (i: number) => ({
          title: `${T({ tr: "Epoch", en: "Epoch" })} ${i + 1}${i + 1 === best.epoch ? " · best.pt" : ""}`,
          rows: series.map((s) => ({ value: small ? nf(s.values[i], 5) : nf(s.values[i], 3), label: s.name, color: s.color })),
        }),
      });
      legend.replaceChildren(
        ...series.map((s) => h("span", { class: "yo-key" }, h("i", { style: `background: ${s.color}` }), h("span", { text: s.name }))),
      );
      note.textContent = T(PANEL_NOTES[panel.key]) + (small ? T({ tr: " (Eksen: binde.)", en: " (Axis: per thousand.)" }) : "");
    };
    const pick = segmented(
      PANELS.map((p) => [p.key, T(p.name)]),
      panel.key,
      (v) => {
        panel = PANELS.find((p) => p.key === v) ?? PANELS[0];
        draw();
      },
      T({ tr: "Gösterilen sütunlar", en: "Columns shown" }),
    );
    queueMicrotask(draw);
    return [
      pick,
      host,
      legend,
      h("p", {
        class: "sub yo-caption",
        text: T({
          tr: `Dikey çizgi: best.pt (epoch ${best.epoch}, uygunluk ${nf(best.fitness[best.epoch - 1], 3)}). last.pt: epoch ${n} (${nf(best.fitness[n - 1], 3)}).`,
          en: `Vertical line: best.pt (epoch ${best.epoch}, fitness ${nf(best.fitness[best.epoch - 1], 3)}). last.pt: epoch ${n} (${nf(best.fitness[n - 1], 3)}).`,
        }),
      }),
      note,
    ];
  });
}

// ---------------------------------------------------------------------------------------------------
// 4. labels.jpg
// ---------------------------------------------------------------------------------------------------

export function labelsDemo(): HTMLElement {
  return withData((d) => {
    const counts = d.labels.counts;
    const max = Math.max(...counts);
    const bars = h(
      "div",
      { class: "yo-bars" },
      ...counts.map((c, k) =>
        h(
          "div",
          { class: "yo-bar" },
          h("span", { text: T(SEG_CLASSES[k]) }),
          h("span", { class: "yo-bar-track" }, h("i", { style: `width: ${(100 * c) / max}%; background: var(--series-${k + 1})` })),
          h("output", { text: int(c) }),
        ),
      ),
    );
    const canvas = h("canvas", { class: "yr-scatter", width: 560, height: 560, role: "img" }) as HTMLCanvasElement;
    let mode: "size" | "place" = "size";
    let boxes: Uint8Array | null = null;
    const note = live();
    const draw = () => {
      if (!boxes || !canvas.isConnected) return;
      const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
      const style = getComputedStyle(canvas);
      const W = canvas.width;
      const P = 56;
      const S = W - P - 36;
      ctx.clearRect(0, 0, W, W);
      ctx.fillStyle = style.getPropertyValue("--surface-2");
      ctx.fillRect(P, 16, S, S);
      ctx.strokeStyle = style.getPropertyValue("--grid");
      ctx.fillStyle = style.getPropertyValue("--text-2");
      ctx.font = "22px system-ui, sans-serif";
      ctx.textAlign = "center";
      for (let k = 0; k <= 5; k++) {
        const t = k / 5;
        const scale = mode === "size" ? 0.5 : 1; // sizes: most boxes are small, so the axes end at half a tile
        ctx.beginPath();
        ctx.moveTo(P + t * S, 16);
        ctx.lineTo(P + t * S, 16 + S);
        ctx.moveTo(P, 16 + S - t * S);
        ctx.lineTo(P + S, 16 + S - t * S);
        ctx.stroke();
        ctx.fillText(nf(t * scale, 1), P + t * S, W - 4);
        ctx.textAlign = "right";
        ctx.fillText(nf(t * scale, 1), P - 8, 16 + S - t * S + 7);
        ctx.textAlign = "center";
      }
      const colors = [style.getPropertyValue("--series-1"), style.getPropertyValue("--series-2")];
      for (const cls of [0, 1]) {
        ctx.fillStyle = colors[cls];
        ctx.globalAlpha = cls ? 0.8 : 0.25;
        for (let i = 0; i < boxes.length; i += 5) {
          if (boxes[i] !== cls) continue;
          const [x, y] = mode === "size" ? [boxes[i + 3] / 255 / 0.5, boxes[i + 4] / 255 / 0.5] : [boxes[i + 1] / 255, boxes[i + 2] / 255];
          if (x > 1 || y > 1) continue;
          ctx.fillRect(P + x * S - 2.5, 16 + S - y * S - 2.5, 5, 5);
        }
      }
      ctx.globalAlpha = 1;
      canvas.setAttribute(
        "aria-label",
        mode === "size"
          ? T({
              tr: "Eğitim kutularının genişliği ve yüksekliği (karenin oranı olarak)",
              en: "Width and height of the training boxes (as a share of the tile)",
            })
          : T({ tr: "Eğitim kutularının merkezleri (karenin oranı olarak)", en: "Centres of the training boxes (as a share of the tile)" }),
      );
      note.textContent =
        mode === "size"
          ? T({
              tr: "Yatay eksen genişlik, dikey eksen yükseklik (1 = karenin tamamı; eksen 0,5'te biter). Binaların yarısından fazlasının uzun kenarı karenin onda birinden kısadır (≈ 20 m); spor sahaları iki kat büyüktür ve çok daha dağınıktır.",
              en: "Width across, height up (1 = the whole tile; the axes end at 0.5). More than half the buildings have a longer side under a tenth of the tile (≈ 20 m); sports fields are twice as large and far more varied.",
            })
          : T({
              tr: "Kutu merkezleri karenin her yerine yayılmış: kareler şehirden rastgele kesildiği için nesneler bir yerde toplanmıyor. Kenarlardaki yoğunluk, karenin sınırında kesilen binalardır.",
              en: "Box centres spread over the whole tile: tiles were cut from the city at random, so objects do not crowd in one place. The extra density at the edges is buildings cut by the tile border.",
            });
    };
    void unpack(d.labels.boxes).then((b) => {
      boxes = b;
      draw();
    });
    document.addEventListener("theme-change", draw);
    const pick = segmented(
      [
        ["size", T({ tr: "Boyutlar", en: "Sizes" })],
        ["place", T({ tr: "Konumlar", en: "Positions" })],
      ],
      mode,
      (v) => {
        mode = v;
        draw();
      },
      T({ tr: "Gösterilen", en: "Shown" }),
    );
    return [
      h("h3", { class: "yr-mini", text: T({ tr: "Eğitimdeki nesneler", en: "Objects in training" }) }),
      bars,
      caption({
        tr: `Bir spor sahasına ${nf(counts[0] / counts[1], 0)} bina düşüyor.`,
        en: `${nf(counts[0] / counts[1], 0)} buildings for every sports field.`,
      }),
      pick,
      canvas,
      h(
        "div",
        { class: "yo-legend" },
        ...SEG_CLASSES.map((c, k) =>
          h("span", { class: "yo-key" }, h("i", { style: `background: var(--series-${k + 1})` }), h("span", { text: T(c) })),
        ),
      ),
      note,
    ];
  });
}

// ---------------------------------------------------------------------------------------------------
// 5. Batches
// ---------------------------------------------------------------------------------------------------

const BATCHES: { key: keyof Outputs["batches"]; name: Text; about: Text }[] = [
  {
    key: "first",
    name: { tr: "İlk yığın", en: "First batch" },
    about: {
      tr: "Resimde yığının dört görüntüsü var (batch 4). Her biri bir mozaiktir: dört kare birleştirilmiş, sonra rastgele döndürülüp çevrilmiş ve kırpılmış. Etiketler de aynı dönüşümle taşındı (çerçeveler; üstlerindeki sayı sınıf numarası: 0 bina, 1 spor sahası). Gri bölgeler görüntünün dışı. Etiketlerin doğru taşınıp taşınmadığını görmenin ilk yeri bu resimdir.",
      en: "The picture holds the batch's four images (batch 4). Each is a mosaic: four tiles joined, then rotated, flipped and cropped at random. The labels moved with them (the outlines; the number on them is the class: 0 building, 1 sports field). Grey areas are outside the image. This picture is the first place to check that the labels moved correctly.",
    },
  },
  {
    key: "last",
    name: { tr: "Mozaiksiz yığın", en: "Batch without mosaic" },
    about: {
      tr: "Son 10 epoch'ta (close_mosaic = 10) mozaik kapanır: model, kullanılacağı gibi bütün kareleri görür. Döndürme ve çevirme sürer. Dosya adındaki 11880 yığın numarasıdır: 90 epoch × 132 yığın.",
      en: "In the last 10 epochs (close_mosaic = 10) the mosaic stops: the model sees whole tiles, the way it will be used. Rotations and flips go on. The 11880 in the file name is the batch number: 90 epochs × 132 batches.",
    },
  },
  {
    key: "valLabels",
    name: { tr: "Doğrulama: etiketler", en: "Validation: labels" },
    about: {
      tr: "Doğrulama karelerinin doğru etiketleri. Bir sonraki seçenekle aynı kareler: ikisinin arasında gidip gelerek kaçan ve fazladan bulunan nesneleri arayın.",
      en: "The true labels of validation tiles. The same tiles as the next option: switch between the two to look for missed and extra objects.",
    },
  },
  {
    key: "valPred",
    name: { tr: "Doğrulama: tahminler", en: "Validation: predictions" },
    about: {
      tr: "best.pt'nin tahminleri, skoruyla. Ultralytics bu resme yalnızca skoru 0,25'i geçenleri çizer; matrisin ve eğrilerin hesabı ise 0,001'den başlar.",
      en: "best.pt's predictions with their scores. Ultralytics draws only those above 0.25 in this picture; the matrix and the curves start from 0.001.",
    },
  },
];

export function batchesDemo(): HTMLElement {
  return withData((d) => {
    let key: keyof Outputs["batches"] = "first";
    const img = h("img", { class: "yr-batch", alt: "", decoding: "async" }) as HTMLImageElement;
    const name = h("code", { class: "yr-batch-name" });
    const note = live();
    const draw = () => {
      const b = BATCHES.find((x) => x.key === key) ?? BATCHES[0];
      img.src = `data:image/jpeg;base64,${d.batches[key].jpeg}`;
      img.alt = `${d.batches[key].name}: ${T(b.name)}`;
      name.textContent = d.batches[key].name;
      note.textContent = T(b.about);
    };
    const pick = segmented(
      BATCHES.map((b) => [b.key, T(b.name)]),
      key,
      (v) => {
        key = v;
        draw();
      },
      T({ tr: "Görüntü", en: "Image" }),
    );
    draw();
    return [pick, name, img, note];
  });
}

// ---------------------------------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------------------------------

interface MatrixView {
  table: HTMLElement;
  cells: HTMLButtonElement[][];
}

/**
 * A confusion matrix as Ultralytics draws it: rows are the prediction, columns the truth. Cells are
 * buttons; `label(r, c)` names a cell for screen readers.
 */
function matrixTable(names: string[], pick: (r: number, c: number) => void, label: (r: number, c: number) => string): MatrixView {
  const cells: HTMLButtonElement[][] = [];
  const body = names.map((rowName, r) => {
    cells.push([]);
    return h(
      "tr",
      {},
      h("th", { scope: "row", text: rowName }),
      ...names.map((_, c) => {
        const b = h("button", { type: "button", class: "yr-cell" + (r === c ? " diag" : "") }) as HTMLButtonElement;
        b.addEventListener("click", () => pick(r, c));
        b.setAttribute("aria-label", label(r, c));
        cells[r].push(b);
        return h("td", {}, b);
      }),
    );
  });
  const table = h(
    "div",
    { class: "yr-matrix-wrap" },
    h("div", { class: "yr-axis-top", text: T({ tr: "gerçek →", en: "true →" }) }),
    h(
      "div",
      { class: "yr-matrix-row" },
      h("div", { class: "yr-axis-side", text: T({ tr: "tahmin →", en: "predicted →" }) }),
      h(
        "table",
        { class: "yr-matrix" },
        h("thead", {}, h("tr", {}, h("td", {}), ...names.map((c) => h("th", { scope: "col", text: c })))),
        h("tbody", {}, ...body),
      ),
    ),
  );
  return { table, cells };
}

// ---------------------------------------------------------------------------------------------------
// 6. The damage survey (a 2 × 2 matrix)
// ---------------------------------------------------------------------------------------------------

/** 100 buildings after an earthquake: 10 badly damaged. A model gives each a damage score. */
const DAMAGED = [0.95, 0.9, 0.86, 0.8, 0.72, 0.66, 0.58, 0.45, 0.33, 0.2];
const SOUND = (() => {
  const high = [0.81, 0.7, 0.64, 0.6, 0.55, 0.52];
  const out = [...high];
  let s = 7;
  while (out.length < 90) {
    s = (s * 16807) % 2147483647;
    out.push(Math.round(((s / 2147483647) * 0.48 + 0.01) * 100) / 100); // 0.01 … 0.49
  }
  return out;
})();
/** The buildings in a fixed shuffled order on the 10 × 10 grid. */
const SURVEY = (() => {
  const all = [...DAMAGED.map((score) => ({ score, damaged: true })), ...SOUND.map((score) => ({ score, damaged: false }))];
  let s = 11;
  for (let i = all.length - 1; i > 0; i--) {
    s = (s * 16807) % 2147483647;
    const j = s % (i + 1);
    [all[i], all[j]] = [all[j], all[i]];
  }
  return all;
})();

/** TP, FP, FN, TN of the survey at a threshold (a score ≥ the threshold is called "damaged"). */
export function surveyCounts(threshold: number): { tp: number; fp: number; fn: number; tn: number } {
  const c = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (const b of SURVEY) {
    const flagged = b.score >= threshold;
    if (b.damaged) c[flagged ? "tp" : "fn"]++;
    else c[flagged ? "fp" : "tn"]++;
  }
  return c;
}

const OUTCOME: Record<"tp" | "fp" | "fn" | "tn", Text> = {
  tp: { tr: "TP: hasarlı, işaretlendi", en: "TP: damaged, flagged" },
  fp: { tr: "FP: sağlam, boşuna işaretlendi", en: "FP: sound, flagged for nothing" },
  fn: { tr: "FN: hasarlı, gözden kaçtı", en: "FN: damaged, missed" },
  tn: { tr: "TN: sağlam, işaretlenmedi", en: "TN: sound, not flagged" },
};

export function surveyDemo(): HTMLElement {
  let threshold = 0.5;
  const grid = h("div", { class: "yr-survey", role: "img" });
  const squares = SURVEY.map(() => h("i", {}));
  grid.append(...squares);
  const names = [T({ tr: "hasarlı", en: "damaged" }), T({ tr: "sağlam", en: "sound" })];
  let focus: "tp" | "fp" | "fn" | "tn" | null = null;
  const KEYS: ("tp" | "fp" | "fn" | "tn")[][] = [
    ["tp", "fp"],
    ["fn", "tn"],
  ];
  const view = matrixTable(
    names,
    (r, c) => {
      focus = focus === KEYS[r][c] ? null : KEYS[r][c];
      draw();
    },
    (r, c) => T(OUTCOME[KEYS[r][c]]),
  );
  const stats = h("dl", { class: "yr-stats" });
  const note = live();
  const draw = () => {
    const c = surveyCounts(threshold);
    SURVEY.forEach((b, i) => {
      const flagged = b.score >= threshold;
      const kind = b.damaged ? (flagged ? "tp" : "fn") : flagged ? "fp" : "tn";
      squares[i].className = `yr-${kind}${focus && focus !== kind ? " dim" : ""}`;
    });
    grid.setAttribute(
      "aria-label",
      T({
        tr: `100 bina: ${c.tp} TP, ${c.fp} FP, ${c.fn} FN, ${c.tn} TN`,
        en: `100 buildings: ${c.tp} TP, ${c.fp} FP, ${c.fn} FN, ${c.tn} TN`,
      }),
    );
    const values = [
      [c.tp, c.fp],
      [c.fn, c.tn],
    ];
    view.cells.forEach((row, r) =>
      row.forEach((b, k) => {
        const key = KEYS[r][k];
        b.replaceChildren(h("span", { class: "yr-cell-key", text: key.toUpperCase() }), h("strong", { text: String(values[r][k]) }));
        heat(b, values[r][k] / 90);
        b.setAttribute("aria-pressed", String(focus === key));
      }),
    );
    const precision = ratio(c.tp, c.tp + c.fp);
    const recall = ratio(c.tp, c.tp + c.fn);
    const rows: [Text | string, string, string][] = [
      [{ tr: "Doğruluk", en: "Accuracy" }, "(TP + TN) / 100", pct((c.tp + c.tn) / 100)],
      [{ tr: "Kesinlik", en: "Precision" }, "TP / (TP + FP)", c.tp + c.fp ? nf(precision) : "–"],
      [{ tr: "Duyarlılık", en: "Recall" }, "TP / (TP + FN)", nf(recall)],
      [{ tr: "Özgüllük", en: "Specificity" }, "TN / (TN + FP)", nf(ratio(c.tn, c.tn + c.fp))],
      ["F1", "2PR / (P + R)", nf(f1Of(precision, recall))],
    ];
    stats.replaceChildren(
      ...rows.flatMap(([k, f, v]) => [
        h("dt", { text: `${typeof k === "string" ? k : T(k)}` }),
        h("dd", {}, h("span", { class: "yr-formula", text: f }), h("strong", { text: v })),
      ]),
    );
    note.textContent =
      c.tp + c.fp === 0
        ? T({
            tr: `Hiçbir bina işaretlenmedi: doğruluk yine ${pct((c.tp + c.tn) / 100)}! Ama 10 hasarlı binanın hiçbiri bulunmadı (duyarlılık 0). Dengesiz veride doğruluk yanıltır.`,
            en: `No building is flagged: accuracy is still ${pct((c.tp + c.tn) / 100)}! But none of the 10 damaged buildings is found (recall 0). On imbalanced data, accuracy misleads.`,
          })
        : T({
            tr: `Eşik ${nf(threshold)}: ${c.tp + c.fp} bina ekip gönderilmek üzere işaretlendi; ${c.fp} tanesi boşuna (FP), ${c.fn} hasarlı bina gözden kaçtı (FN).`,
            en: `Threshold ${nf(threshold)}: ${c.tp + c.fp} buildings are flagged for an inspection team; ${c.fp} of them for nothing (FP), ${c.fn} damaged buildings are missed (FN).`,
          });
  };
  const slider = range(
    T({ tr: "Hasar skoru eşiği", en: "Damage score threshold" }),
    0.05,
    1,
    0.05,
    threshold,
    (v) => nf(v),
    (v) => {
      threshold = v;
      draw();
    },
  );
  const none = h("button", { type: "button", class: "btn", text: T({ tr: 'Hepsine "sağlam" de', en: 'Call them all "sound"' }) });
  none.addEventListener("click", () => {
    threshold = 1;
    slider.set(1);
    draw();
  });
  draw();
  return demo(
    h("div", { class: "yr-survey-row" }, grid, h("div", {}, view.table, stats)),
    h(
      "div",
      { class: "yo-legend" },
      ...(["tp", "fp", "fn", "tn"] as const).map((k) =>
        h("span", { class: "yo-key" }, h("i", { class: `yr-${k}` }), h("span", { text: T(OUTCOME[k]) })),
      ),
    ),
    h("div", { class: "yo-slider-grid" }, slider, none),
    note,
    caption({
      tr: "Sayılar uydurmadır: 100 binadan 10'u ağır hasarlı. Matristeki bir hücreye tıklayınca o binalar vurgulanır.",
      en: "The numbers are made up: 10 of 100 buildings are badly damaged. Click a cell of the matrix to highlight its buildings.",
    }),
  );
}

// ---------------------------------------------------------------------------------------------------
// 7. The toy tile (detection: a 3 × 3 matrix, no TN)
// ---------------------------------------------------------------------------------------------------

type Box = [number, number, number, number];
interface ToyTruth {
  id: string;
  cls: number;
  box: Box;
}
interface ToyPred {
  id: string;
  cls: number;
  score: number;
  box: Box;
  about: Text;
}
const TOY_TRUTHS: ToyTruth[] = [
  { id: "B1", cls: 0, box: [20, 20, 70, 60] },
  { id: "B2", cls: 0, box: [90, 20, 140, 70] },
  { id: "B3", cls: 0, box: [170, 20, 230, 60] },
  { id: "B4", cls: 0, box: [250, 20, 290, 70] },
  { id: "B5", cls: 0, box: [20, 90, 60, 140] },
  { id: "B6", cls: 0, box: [80, 90, 130, 130] },
  { id: "B7", cls: 0, box: [150, 90, 200, 140] },
  { id: "B8", cls: 0, box: [220, 90, 280, 130] },
  { id: "S1", cls: 1, box: [20, 185, 140, 285] },
  { id: "S2", cls: 1, box: [170, 185, 280, 285] },
];
const TOY_PREDS: ToyPred[] = [
  { id: "p1", cls: 0, score: 0.92, box: [22, 22, 70, 62], about: { tr: "B1'i iyi buldu.", en: "Finds B1 well." } },
  { id: "p2", cls: 0, score: 0.85, box: [92, 18, 141, 68], about: { tr: "B2'yi buldu.", en: "Finds B2." } },
  {
    id: "p3",
    cls: 0,
    score: 0.45,
    box: [95, 25, 145, 72],
    about: { tr: "B2'ye ikinci kutu: B2'yi zaten p2 aldı, bu FP.", en: "A second box on B2: p2 already took B2, so this is an FP." },
  },
  { id: "p4", cls: 0, score: 0.78, box: [172, 22, 228, 62], about: { tr: "B3'ü buldu.", en: "Finds B3." } },
  {
    id: "p5",
    cls: 0,
    score: 0.3,
    box: [250, 24, 288, 70],
    about: {
      tr: "B4'ü buldu ama skoru düşük: eşik 0,3'ü geçince kaybolur.",
      en: "Finds B4 with a low score: it disappears once the threshold passes 0.3.",
    },
  },
  {
    id: "p6",
    cls: 0,
    score: 0.66,
    box: [95, 98, 150, 140],
    about: {
      tr: "B6'ya kaymış bir kutu: IoU 0,35 < 0,5. Hem FP (yanlış kutu) hem FN (B6 bulunmadı).",
      en: "A box shifted off B6: IoU 0.35 < 0.5. Both an FP (a wrong box) and an FN (B6 not found).",
    },
  },
  { id: "p7", cls: 0, score: 0.55, box: [148, 92, 198, 142], about: { tr: "B7'yi buldu.", en: "Finds B7." } },
  {
    id: "p8",
    cls: 0,
    score: 0.18,
    box: [222, 90, 280, 132],
    about: {
      tr: "B8'i buldu ama skoru çok düşük: ancak eşik 0,18'in altındaysa sayılır.",
      en: "Finds B8 with a very low score: it counts only below a threshold of 0.18.",
    },
  },
  { id: "p9", cls: 1, score: 0.71, box: [24, 187, 140, 281], about: { tr: "S1 sahasını buldu.", en: "Finds field S1." } },
  {
    id: "p10",
    cls: 0,
    score: 0.48,
    box: [172, 189, 276, 285],
    about: {
      tr: 'S2 sahasının yerini buldu ama "bina" dedi: yanlış sınıf. Matriste köşegenin dışına düşer.',
      en: "Finds where field S2 is but calls it a building: the wrong class. It lands off the diagonal.",
    },
  },
  {
    id: "p11",
    cls: 0,
    score: 0.62,
    box: [232, 147, 262, 172],
    about: {
      tr: 'Boş zeminde bir "bina" (bir konteyner): arka plandan FP.',
      en: 'A "building" on empty ground (a container): an FP from the background.',
    },
  },
  {
    id: "p12",
    cls: 1,
    score: 0.35,
    box: [75, 150, 135, 176],
    about: {
      tr: "Bir otoparkı spor sahası sandı: arka plandan FP.",
      en: "Takes a car park for a sports field: an FP from the background.",
    },
  },
];

const toyIou = (a: Box, b: Box) => {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const hh = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * hh;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
};

/**
 * Ultralytics' ConfusionMatrix.process_batch on the toy tile: detections above the threshold, highest
 * score first, each claims the free truth it overlaps most if that IoU is above 0.5, whatever the
 * classes. Returns the matrix (rows predicted, columns true, last = background) and which boxes fell
 * in which cell.
 */
export function toyMatrix(threshold: number): { counts: number[][]; members: string[][][] } {
  const preds = TOY_PREDS.filter((p) => p.score > threshold);
  const iou = TOY_TRUTHS.map((t) => preds.map((p) => toyIou(t.box, p.box)));
  const match = TOY_TRUTHS.map(() => -1);
  const candidates = preds.map((_, j) => j).filter((j) => iou.some((row) => row[j] > 0.5));
  candidates.sort((a, b) => preds[b].score - preds[a].score);
  for (const j of candidates) {
    let g = 0;
    for (let k = 1; k < TOY_TRUTHS.length; k++) if (iou[k][j] > iou[g][j]) g = k;
    if (iou[g][j] > 0.5) {
      match[g] = j;
      iou[g] = iou[g].map(() => 0);
    }
  }
  const counts = [0, 1, 2].map(() => [0, 0, 0]);
  const members = [0, 1, 2].map(() => [[], [], []] as string[][]);
  TOY_TRUTHS.forEach((t, g) => {
    const j = match[g];
    const row = j >= 0 ? preds[j].cls : 2;
    counts[row][t.cls]++;
    members[row][t.cls].push(t.id, ...(j >= 0 ? [preds[j].id] : []));
  });
  const used = new Set(match.filter((j) => j >= 0));
  preds.forEach((p, j) => {
    if (used.has(j)) return;
    counts[p.cls][2]++;
    members[p.cls][2].push(p.id);
  });
  return { counts, members };
}

const TOY_CELL: Text[][] = [
  [
    { tr: "Bina dendi, gerçekten bina: TP (bina).", en: "Called a building, really a building: TP (building)." },
    {
      tr: "Bina dendi, gerçekte spor sahası: yanlış sınıf. Bina için FP, saha için FN.",
      en: "Called a building, really a sports field: the wrong class. An FP for buildings, an FN for fields.",
    },
    {
      tr: "Bina dendi, orada hiçbir şey yok (ya da nesne zaten bulunmuştu): FP.",
      en: "Called a building where there is nothing (or the object was already found): FP.",
    },
  ],
  [
    { tr: "Saha dendi, gerçekte bina: yanlış sınıf.", en: "Called a field, really a building: the wrong class." },
    { tr: "Saha dendi, gerçekten saha: TP (spor sahası).", en: "Called a field, really a field: TP (sports field)." },
    { tr: "Saha dendi, orada saha yok: FP.", en: "Called a field where there is none: FP." },
  ],
  [
    { tr: "Gerçek bina, hiçbir kutu bulmadı: FN (kaçan).", en: "A true building no box found: FN (missed)." },
    { tr: "Gerçek saha, hiçbir kutu bulmadı: FN.", en: "A true field no box found: FN." },
    {
      tr: 'Arka plan × arka plan: "orada bir şey yok ve bir şey denmedi". Sayılamaz: bir karede kaç tane "boş yer" vardır? Tespitte TN yoktur.',
      en: 'Background × background: "nothing there and nothing said". It cannot be counted: how many "empty places" does a tile have? Detection has no TN.',
    },
  ],
];

export function toyDemo(): HTMLElement {
  let threshold = 0.25;
  let picked: [number, number] | null = [0, 2];
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 300 300");
  svg.setAttribute("class", "yr-toy");
  svg.setAttribute("role", "img");
  const names = [T(SEG_CLASSES[0]), T(SEG_CLASSES[1]), T({ tr: "arka plan", en: "background" })];
  const view = matrixTable(
    names,
    (r, c) => {
      picked = picked && picked[0] === r && picked[1] === c ? null : [r, c];
      draw();
    },
    (r, c) => `${T({ tr: "tahmin", en: "predicted" })}: ${names[r]}, ${T({ tr: "gerçek", en: "true" })}: ${names[c]}`,
  );
  const note = live();
  const list = h("ul", { class: "yr-toy-list" });
  const draw = () => {
    const m = toyMatrix(threshold);
    const on = new Set(picked ? m.members[picked[0]][picked[1]] : []);
    svg.replaceChildren();
    const add = (tag: string, attrs: Record<string, string | number>, text?: string) => {
      const el = document.createElementNS(svgNS, tag);
      for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
      if (text) el.textContent = text;
      svg.append(el);
      return el;
    };
    add("rect", { x: 0, y: 0, width: 300, height: 300, class: "yr-toy-ground" });
    for (const t of TOY_TRUTHS) {
      const [x1, y1, x2, y2] = t.box;
      add("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: `yr-toy-truth c${t.cls}${on.has(t.id) ? " on" : ""}` });
      add("text", { x: x1 + 3, y: y2 - 4, class: "yr-toy-id" }, t.id);
    }
    for (const p of TOY_PREDS) {
      if (p.score <= threshold) continue;
      const [x1, y1, x2, y2] = p.box;
      add("rect", {
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
        class: `yr-toy-pred c${p.cls}${on.has(p.id) ? " on" : ""}${picked && !on.has(p.id) ? " dim" : ""}`,
      });
      add("text", { x: x2 - 2, y: y1 + 10, "text-anchor": "end", class: `yr-toy-score c${p.cls}` }, `${p.id} ${nf(p.score)}`);
    }
    svg.setAttribute(
      "aria-label",
      T({
        tr: "Oyuncak kare: 8 bina, 2 spor sahası ve modelin kutuları",
        en: "Toy tile: 8 buildings, 2 sports fields and the model's boxes",
      }),
    );
    view.cells.forEach((row, r) =>
      row.forEach((b, c) => {
        b.replaceChildren(h("strong", { text: r === 2 && c === 2 ? "–" : String(m.counts[r][c]) }));
        heat(b, m.counts[r][c] / 8);
        b.setAttribute("aria-pressed", String(!!picked && picked[0] === r && picked[1] === c));
      }),
    );
    const tp = m.counts[0][0];
    const buildingP = ratio(tp, m.counts[0][0] + m.counts[0][1] + m.counts[0][2]);
    const buildingR = ratio(tp, m.counts[0][0] + m.counts[1][0] + m.counts[2][0]);
    note.textContent =
      (picked ? `${T(TOY_CELL[picked[0]][picked[1]])} ` : "") +
      T({
        tr: `Bina: kesinlik ${tp}/${m.counts[0].reduce((a, b) => a + b, 0)} = ${nf(buildingP)}, duyarlılık ${tp}/8 = ${nf(buildingR)}.`,
        en: `Building: precision ${tp}/${m.counts[0].reduce((a, b) => a + b, 0)} = ${nf(buildingP)}, recall ${tp}/8 = ${nf(buildingR)}.`,
      });
    const ids = picked ? m.members[picked[0]][picked[1]].filter((id) => id.startsWith("p")) : [];
    list.replaceChildren(
      ...TOY_PREDS.filter((p) => ids.includes(p.id)).map((p) =>
        h("li", {}, h("strong", { text: `${p.id} (${nf(p.score)}): ` }), T(p.about)),
      ),
    );
  };
  const slider = range(
    T({ tr: "Güven eşiği", en: "Confidence threshold" }),
    0.05,
    0.95,
    0.05,
    threshold,
    (v) => nf(v),
    (v) => {
      threshold = v;
      draw();
    },
  );
  draw();
  return demo(
    h("div", { class: "yr-toy-row" }, svg, view.table),
    h(
      "div",
      { class: "yo-legend" },
      h(
        "span",
        { class: "yo-key" },
        h("i", { class: "yr-key-truth" }),
        h("span", { text: T({ tr: "gerçek nesne (kesikli)", en: "true object (dashed)" }) }),
      ),
      h(
        "span",
        { class: "yo-key" },
        h("i", { class: "yr-key-pred c0" }),
        h("span", { text: T({ tr: '"bina" tahmini', en: '"building" prediction' }) }),
      ),
      h(
        "span",
        { class: "yo-key" },
        h("i", { class: "yr-key-pred c1" }),
        h("span", { text: T({ tr: '"spor sahası" tahmini', en: '"sports field" prediction' }) }),
      ),
    ),
    slider,
    note,
    list,
    caption({
      tr: "Bir hücreye tıklayınca o hücreye düşen kutular vurgulanır ve açıklanır.",
      en: "Click a cell to highlight and explain the boxes that fall in it.",
    }),
  );
}

// ---------------------------------------------------------------------------------------------------
// 8. Our model's matrix
// ---------------------------------------------------------------------------------------------------

function exampleCard(e: Example, cell: [number, number]): HTMLElement {
  const [w, hgt] = e.size;
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${hgt}`);
  svg.setAttribute("aria-hidden", "true");
  const rect = (b: number[], cls: string) => {
    const r = document.createElementNS(svgNS, "rect");
    r.setAttribute("x", String(b[0]));
    r.setAttribute("y", String(b[1]));
    r.setAttribute("width", String(b[2] - b[0]));
    r.setAttribute("height", String(b[3] - b[1]));
    r.setAttribute("class", cls);
    svg.append(r);
  };
  if (e.truth) rect(e.truth, "yr-ex-truth");
  if (e.pred) rect(e.pred, `yr-ex-pred c${cell[0]}`);
  const city = e.tile.split("_")[0];
  return h(
    "figure",
    { class: "yr-ex" },
    h("div", { class: "yr-ex-img" }, h("img", { src: `data:image/jpeg;base64,${e.jpeg}`, alt: "", width: w, height: hgt }), svg),
    h("figcaption", { text: `${city}${e.score !== null ? ` · ${nf(e.score)}` : ""}` }),
  );
}

export function realMatrixDemo(): HTMLElement {
  return withData((d) => {
    const m = d.matrix;
    const names = [T(SEG_CLASSES[0]), T(SEG_CLASSES[1]), T({ tr: "arka plan", en: "background" })];
    let k = 0; // index into m.confs: 0.001 first, as Ultralytics draws it
    let normalized = false;
    let picked: [number, number] = [0, 2];
    const view = matrixTable(
      names,
      (r, c) => {
        if (r === 2 && c === 2) return;
        picked = [r, c];
        draw();
      },
      (r, c) => `${T({ tr: "tahmin", en: "predicted" })}: ${names[r]}, ${T({ tr: "gerçek", en: "true" })}: ${names[c]}`,
    );
    const stats = h("table", { class: "yo-table yr-class-table" });
    const note = live();
    const gallery = h("div", { class: "yr-gallery" });
    const galleryNote = h("p", { class: "sub yo-caption" });
    const draw = () => {
      const counts = m.counts[k];
      const at = (r: number, c: number) => counts[r * 3 + c];
      const colSum = (c: number) => at(0, c) + at(1, c) + at(2, c);
      view.cells.forEach((row, r) =>
        row.forEach((b, c) => {
          const v = at(r, c);
          const share = normalized ? ratio(v, colSum(c)) : 0;
          const empty = r === 2 && c === 2;
          b.replaceChildren(h("strong", { text: empty ? "–" : normalized ? (c === 2 ? "–" : nf(share)) : int(v) }));
          heat(b, empty ? 0 : normalized ? (c === 2 ? 0 : share) : Math.log10(1 + v) / Math.log10(1 + 25000));
          b.disabled = empty;
          b.setAttribute("aria-pressed", String(picked[0] === r && picked[1] === c));
        }),
      );
      const head = h(
        "tr",
        {},
        ...[T({ tr: "sınıf", en: "class" }), T({ tr: "kesinlik", en: "precision" }), T({ tr: "duyarlılık", en: "recall" }), "F1"].map((t) =>
          h("th", { scope: "col", text: t }),
        ),
      );
      stats.replaceChildren(
        h("thead", {}, head),
        h(
          "tbody",
          {},
          ...[0, 1].map((c) => {
            const p = ratio(at(c, c), at(c, 0) + at(c, 1) + at(c, 2));
            const r = ratio(at(c, c), colSum(c));
            return h(
              "tr",
              {},
              h("th", { scope: "row", text: names[c] }),
              h("td", { text: nf(p) }),
              h("td", { text: nf(r) }),
              h("td", { text: nf(f1Of(p, r)) }),
            );
          }),
        ),
      );
      const conf = m.confs[k];
      note.textContent =
        k === 0
          ? T({
              tr: `Eşik 0,001 (Ultralytics'in doğrulama eşiği, confusion_matrix.png bu eşikle çizilir): bina satırında ${int(at(0, 2))} FP. Skoru 0,001'i geçen hemen her kutu sayılıyor. Kaydırıcıyı 0,25'e getirin.`,
              en: `Threshold 0.001 (Ultralytics' validation threshold; confusion_matrix.png is drawn at it): ${int(at(0, 2))} FPs in the building row. Nearly every box scoring over 0.001 counts. Move the slider to 0.25.`,
            })
          : T({
              tr: `Eşik ${nf(conf)}: ${int(at(0, 0))} bina bulundu, ${int(at(2, 0))} bina kaçtı (FN), ${int(at(0, 2))} kutu boşa (FP). Spor sahası: ${int(colSum(1))} örnek, doğru bulunan ${int(at(1, 1))}.`,
              en: `Threshold ${nf(conf)}: ${int(at(0, 0))} buildings found, ${int(at(2, 0))} missed (FN), ${int(at(0, 2))} boxes for nothing (FP). ${int(at(1, 1))} of ${int(colSum(1))} sports fields found.`,
            });
      const cell = m.examples[`${picked[0]},${picked[1]}`];
      gallery.replaceChildren(...(cell?.items ?? []).map((e) => exampleCard(e, picked)));
      galleryNote.textContent = cell
        ? T({
            tr: `Seçilen hücre (tahmin: ${names[picked[0]]}, gerçek: ${names[picked[1]]}): ${int(cell.count)} örnek (eşik ${nf(m.exampleConf)}); rastgele ${cell.items.length} tanesi, ayrı karelerden. Kesikli çerçeve gerçek kutu, düz çerçeve tahmin ve skoru.`,
            en: `Chosen cell (predicted: ${names[picked[0]]}, true: ${names[picked[1]]}): ${int(cell.count)} cases at threshold ${nf(m.exampleConf)}; ${cell.items.length} of them at random, from different tiles. Dashed: the true box; solid: the prediction and its score.`,
          })
        : T({ tr: "Bu hücrede eşik 0,25'te örnek yok.", en: "This cell has no cases at threshold 0.25." });
    };
    const slider = range(
      T({ tr: "Güven eşiği", en: "Confidence threshold" }),
      0,
      m.confs.length - 1,
      1,
      k,
      (i) => nf(m.confs[i], i ? 2 : 3),
      (i) => {
        k = i;
        draw();
      },
    );
    const norm = segmented(
      [
        ["counts", T({ tr: "Sayılar", en: "Counts" })],
        ["norm", T({ tr: "Sütuna göre oran", en: "Share of each column" })],
      ],
      "counts",
      (v) => {
        normalized = v === "norm";
        draw();
      },
      T({ tr: "Gösterim", en: "Display" }),
    );
    draw();
    return [
      h("div", { class: "yo-slider-grid" }, slider, norm),
      view.table,
      note,
      h("div", { class: "yo-table-wrap" }, stats),
      h("h3", { class: "yr-mini", text: T({ tr: "Hücredeki örnekler", en: "Cases in the cell" }) }),
      galleryNote,
      gallery,
    ];
  });
}

// ---------------------------------------------------------------------------------------------------
// 9. The land-cover matrix (pixels)
// ---------------------------------------------------------------------------------------------------

export function semMatrixDemo(): HTMLElement {
  return withData((d) => {
    const counts = d.semMatrix.counts;
    const names = SEM_CLASSES.map((c) => T(c));
    let mode: "counts" | "col" | "row" = "col";
    let picked: [number, number] = [3, 4];
    const view = matrixTable(
      names,
      (r, c) => {
        picked = [r, c];
        draw();
      },
      (r, c) => `${T({ tr: "tahmin", en: "predicted" })}: ${names[r]}, ${T({ tr: "gerçek", en: "true" })}: ${names[c]}`,
    );
    const row = (r: number) => counts[r].reduce((a, b) => a + b, 0);
    const col = (c: number) => counts.reduce((a, rr) => a + rr[c], 0);
    const total = counts.flat().reduce((a, b) => a + b, 0);
    const note = live();
    const four = h("div", { class: "yr-four" });
    const draw = () => {
      view.cells.forEach((cells, r) =>
        cells.forEach((b, c) => {
          const v = counts[r][c];
          const share = mode === "col" ? ratio(v, col(c)) : mode === "row" ? ratio(v, row(r)) : ratio(v, total);
          b.replaceChildren(h("strong", { text: mode === "counts" ? nf(v / 1e6, 1) : share < 0.005 ? "" : nf(share) }));
          heat(b, mode === "counts" ? Math.sqrt(share / 0.4) : share);
          b.setAttribute("aria-pressed", String(picked[0] === r && picked[1] === c));
        }),
      );
      const [r, c] = picked;
      const v = counts[r][c];
      note.textContent =
        r === c
          ? T({
              tr: `Duyarlılık ${pct(ratio(v, col(c)))}: gerçekte ${names[c]} olan pikseller içinde doğru bulunanların payı. Kesinlik ${pct(ratio(v, row(r)))}: "${names[r]}" denen pikseller içinde gerçekten ${names[r]} olanların payı.`,
              en: `${pct(ratio(v, col(c)))} of the pixels that are really ${names[c]} were found (recall); ${pct(ratio(v, row(r)))} of the pixels called "${names[r]}" really are (precision).`,
            })
          : T({
              tr: `Gerçekte ${names[c]} olan pikseller içinde "${names[r]}" denenlerin payı: ${pct(ratio(v, col(c)))} (${nf(v / 1e6, 2)} milyon piksel).`,
              en: `${pct(ratio(v, col(c)))} of the pixels that are really ${names[c]} were called "${names[r]}" (${nf(v / 1e6, 2)} million pixels).`,
            });
      // one class against the rest: the 2 × 2 table, with a TN this time
      const k = c;
      const tp = counts[k][k];
      const fp = row(k) - tp;
      const fn = col(k) - tp;
      const tn = total - tp - fp - fn;
      four.replaceChildren(
        h("p", {
          class: "yr-mini",
          text: T({
            tr: `"${names[k]}" sınıfı, öbürlerine karşı (milyon piksel):`,
            en: `The class "${names[k]}" against the rest (million pixels):`,
          }),
        }),
        h(
          "dl",
          { class: "yr-stats" },
          ...(
            [
              ["TP", tp],
              ["FP", fp],
              ["FN", fn],
              ["TN", tn],
            ] as [string, number][]
          ).flatMap(([key, value]) => [h("dt", { text: key }), h("dd", {}, h("strong", { text: nf(value / 1e6, 2) }))]),
          h("dt", { text: "IoU" }),
          h("dd", {}, h("span", { class: "yr-formula", text: "TP / (TP + FP + FN)" }), h("strong", { text: nf(ratio(tp, tp + fp + fn)) })),
        ),
      );
    };
    const pick = segmented(
      [
        ["col", T({ tr: "Sütuna göre (duyarlılık)", en: "By column (recall)" })],
        ["row", T({ tr: "Satıra göre (kesinlik)", en: "By row (precision)" })],
        ["counts", T({ tr: "Milyon piksel", en: "Million pixels" })],
      ],
      mode,
      (v) => {
        mode = v;
        draw();
      },
      T({ tr: "Gösterim", en: "Display" }),
    );
    draw();
    return [
      pick,
      view.table,
      note,
      four,
      caption({
        tr: `131 doğrulama karesinin ${nf(total / 1e6, 1)} milyon pikseli. Bir hücreye tıklayın; altta o sütunun sınıfı için TP, FP, FN ve TN yazılır.`,
        en: `${nf(total / 1e6, 1)} million pixels of the 131 validation tiles. Click a cell; below are TP, FP, FN and TN for the class of its column.`,
      }),
    ];
  });
}

// ---------------------------------------------------------------------------------------------------
// 10. P, R, F1 and PR curves
// ---------------------------------------------------------------------------------------------------

export function curvesDemo(): HTMLElement {
  return withData((d) => {
    let key: "B" | "M" = "M";
    let cls = 0;
    const host = chart(220);
    const pr = h("div", { class: "yo-pr yr-pr" });
    const note = live();
    const colors = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
    const draw = () => {
      const c = d.curves[key];
      const n = c.confidence.length;
      const series: [string, number[], string][] = [
        [T({ tr: "kesinlik", en: "precision" }), c.precision[cls], colors[0]],
        [T({ tr: "duyarlılık", en: "recall" }), c.recall[cls], colors[1]],
        ["F1", c.f1[cls], colors[2]],
      ];
      const f1 = c.f1[cls];
      const best = f1.indexOf(Math.max(...f1));
      lineChart(host, {
        label: T({ tr: "Güven eşiğine göre kesinlik, duyarlılık ve F1", en: "Precision, recall and F1 by confidence threshold" }),
        empty: "",
        x: { min: 1, max: n, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => ({ v: 1 + v * (n - 1), label: nf(v, 1) })), grid: [best + 1] },
        y: { min: 0, max: 1, ticks: linearTicks(0, 1, 0.25, (v: number) => nf(v)) },
        series: series.map(([name, values, color]) => ({ name, points: values.map((v, i) => [i + 1, v]), color })),
        tooltip: (i: number) => ({
          title: `${T({ tr: "eşik", en: "threshold" })} ${nf(c.confidence[i])}`,
          rows: series.map(([name, values, color]) => ({ value: nf(values[i]), label: name, color })),
        }),
      });
      // the PR curve, drawn by hand (recall is not evenly spaced)
      const W = 260;
      const H = 200;
      const P = 30;
      const X = (r: number) => P + r * (W - P - 8);
      const Y = (p: number) => H - P - p * (H - P - 8);
      const line = c.prRecall.map((r, i) => `${i ? "L" : "M"}${X(r).toFixed(1)},${Y(c.prPrecision[cls][i]).toFixed(1)}`).join("");
      pr.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${T({ tr: "Kesinlik-duyarlılık eğrisi", en: "Precision-recall curve" })}">
        <line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(0)}" class="baseline"/><line x1="${X(0)}" y1="${Y(0)}" x2="${X(0)}" y2="${Y(1)}" class="baseline"/>
        <path d="${line} L${X(c.prRecall.at(-1) ?? 1)},${Y(0)} L${X(0)},${Y(0)} Z" class="yo-pr-area"/>
        <path d="${line}" class="yo-pr-line"/>
        <circle cx="${X(c.recall[cls][best])}" cy="${Y(c.precision[cls][best])}" r="6" class="yo-pr-point"/>
        <text x="${X(1)}" y="${H - 8}" text-anchor="end" class="tick">${T({ tr: "duyarlılık", en: "recall" })} →</text>
        <text x="${X(0) + 4}" y="${Y(1) + 10}" class="tick">${T({ tr: "kesinlik", en: "precision" })} ↑</text>
        <text x="${X(0) - 4}" y="${Y(0) + 4}" text-anchor="end" class="tick">0</text><text x="${X(0) - 4}" y="${Y(1) + 4}" text-anchor="end" class="tick">1</text>
        <text x="${X(1)}" y="${Y(0) + 14}" text-anchor="middle" class="tick">1</text>
      </svg>`;
      note.textContent = T({
        tr: `${T(SEG_CLASSES[cls])}, ${key === "M" ? "maskeler" : "kutular"}: F1'in en yüksek olduğu eşik ${nf(c.confidence[best])} (F1 = ${nf(f1[best])}); bu eşikte kesinlik ${nf(c.precision[cls][best])}, duyarlılık ${nf(c.recall[cls][best])}. AP50 = ${nf(c.table.ap50[cls])}: PR eğrisinin altındaki alan (nokta, en iyi F1 eşiği).`,
        en: `${T(SEG_CLASSES[cls])}, ${key === "M" ? "masks" : "boxes"}: the highest F1 (${nf(f1[best])}) is at threshold ${nf(c.confidence[best])}; there precision is ${nf(c.precision[cls][best])} and recall ${nf(c.recall[cls][best])}. AP50 = ${nf(c.table.ap50[cls])}: the area under the PR curve (the dot: the best F1 threshold).`,
      });
    };
    const pickKey = segmented(
      [
        ["M", T({ tr: "Maske (Mask…png)", en: "Mask (Mask…png)" })],
        ["B", T({ tr: "Kutu (Box…png)", en: "Box (Box…png)" })],
      ],
      key,
      (v) => {
        key = v;
        draw();
      },
      T({ tr: "Kutu ya da maske", en: "Box or mask" }),
    );
    const pickCls = segmented(
      SEG_CLASSES.map((c, k) => [String(k), T(c)]),
      String(cls),
      (v) => {
        cls = Number(v);
        draw();
      },
      T({ tr: "Sınıf", en: "Class" }),
    );
    queueMicrotask(draw);
    return [
      h("div", { class: "yr-picks" }, pickKey, pickCls),
      host,
      h(
        "div",
        { class: "yo-legend" },
        ...[T({ tr: "kesinlik", en: "precision" }), T({ tr: "duyarlılık", en: "recall" }), "F1"].map((t, i) =>
          h("span", { class: "yo-key" }, h("i", { style: `background: ${colors[i]}` }), h("span", { text: t })),
        ),
      ),
      pr,
      note,
    ];
  });
}
