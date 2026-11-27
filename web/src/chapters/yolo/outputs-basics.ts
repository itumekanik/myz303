/**
 * Two demos of the Outputs tab that need no YOLO at all: the confusion matrix of a three-class
 * classifier (cat, dog, mouse: 30 photos) read one class against the rest, and AP and mAP worked out
 * by hand on ranked lists. The same arithmetic then carries over to detection.
 */
import { fmtNum, h } from "../../shared/core.js";
import { KIT } from "../../shared/kit.js";
import type { Text } from "../../shared/chapters.ts";
import { averagePrecision, precisionRecall } from "../../engines/yolo/metrics.ts";
import { T } from "./page.ts";

const { segmented } = KIT as {
  segmented: <V extends string>(
    options: [V, string][],
    current: V,
    pick: (v: V) => void,
    aria: string,
  ) => HTMLElement & { set: (v: V) => void };
};

const nf = (v: number, d = 2) => fmtNum(v, d) as string;
const ratio = (a: number, b: number) => (b ? a / b : 0);
const f1Of = (p: number, r: number) => (p + r ? (2 * p * r) / (p + r) : 0);
const demo = (...children: (Node | null)[]) => h("div", { class: "t-demo yo-demo" }, ...children);
const caption = (text: Text) => h("p", { class: "sub yo-caption", text: T(text) });

export const PETS: { name: Text; icon: string }[] = [
  { name: { tr: "kedi", en: "cat" }, icon: "🐱" },
  { name: { tr: "köpek", en: "dog" }, icon: "🐶" },
  { name: { tr: "fare", en: "mouse" }, icon: "🐭" },
];

/** Confusion matrices of 30 photos: rows are the prediction, columns the truth (as Ultralytics draws it). */
export const PET_SCENARIOS: { key: string; name: Text; matrix: number[][] }[] = [
  {
    key: "base",
    name: { tr: "Örnek", en: "Example" },
    matrix: [
      [7, 2, 1],
      [2, 9, 0],
      [1, 1, 7],
    ],
  },
  {
    key: "rare",
    name: { tr: "Fare az", en: "Few mice" },
    matrix: [
      [13, 1, 1],
      [1, 13, 1],
      [0, 0, 0],
    ],
  },
  {
    key: "dog",
    name: { tr: 'Hep "köpek"', en: 'Always "dog"' },
    matrix: [
      [0, 0, 0],
      [10, 12, 8],
      [0, 0, 0],
    ],
  },
];

/** One class against the rest: TP its diagonal cell, FP the rest of its row, FN the rest of its column, TN the others. */
export function oneVsRest(m: number[][], k: number): { tp: number; fp: number; fn: number; tn: number } {
  const total = m.flat().reduce((a, b) => a + b, 0);
  const tp = m[k][k];
  const fp = m[k].reduce((a, b) => a + b, 0) - tp;
  const fn = m.reduce((a, row) => a + row[k], 0) - tp;
  return { tp, fp, fn, tn: total - tp - fp - fn };
}

/** Per class precision, recall, F1 and support; then the macro, weighted and micro averages. */
export function petScores(m: number[][]) {
  const total = m.flat().reduce((a, b) => a + b, 0);
  const rows = m.map((_, k) => {
    const c = oneVsRest(m, k);
    const p = ratio(c.tp, c.tp + c.fp);
    const r = ratio(c.tp, c.tp + c.fn);
    return { p, r, f1: f1Of(p, r), support: c.tp + c.fn, predicted: c.tp + c.fp };
  });
  const mean = (f: (x: (typeof rows)[number]) => number) => rows.reduce((a, x) => a + f(x), 0) / rows.length;
  const weighted = (f: (x: (typeof rows)[number]) => number) => rows.reduce((a, x) => a + f(x) * x.support, 0) / total;
  const correct = m.reduce((a, row, k) => a + row[k], 0);
  return {
    rows,
    total,
    accuracy: correct / total,
    macro: { p: mean((x) => x.p), r: mean((x) => x.r), f1: mean((x) => x.f1) },
    weighted: { p: weighted((x) => x.p), r: weighted((x) => x.r), f1: weighted((x) => x.f1) },
  };
}

type Role = "tp" | "fp" | "fn" | "tn";
const role = (r: number, c: number, k: number): Role => (r === k && c === k ? "tp" : r === k ? "fp" : c === k ? "fn" : "tn");
const ROLE_TEXT: Record<Role, Text> = {
  tp: { tr: "TP: doğru bulundu", en: "TP: found" },
  fp: { tr: "FP: yanlış alarm", en: "FP: false alarm" },
  fn: { tr: "FN: kaçtı", en: "FN: missed" },
  tn: { tr: "TN: doğru reddedildi", en: "TN: rightly rejected" },
};

export function petsDemo(): HTMLElement {
  let scenario = PET_SCENARIOS[0];
  let k = 0;
  let picked: [number, number] | null = null;
  const cards = h("div", { class: "yr-pets" });
  const table = h("table", { class: "yr-matrix yr-pet-matrix" });
  const work = h("div", { class: "yr-work", "aria-live": "polite" });
  const scores = h("table", { class: "yo-table yr-class-table" });
  const note = h("p", { class: "yo-big", "aria-live": "polite" });
  const name = (i: number) => `${PETS[i].icon} ${T(PETS[i].name)}`;
  const draw = () => {
    const m = scenario.matrix;
    // the 30 photos, grouped by what they really are
    cards.replaceChildren(
      ...PETS.map((_, c) => {
        const items: HTMLElement[] = [];
        m.forEach((row, r) => {
          for (let n = 0; n < row[c]; n++) {
            const kind = role(r, c, k);
            const dim = picked && (picked[0] !== r || picked[1] !== c);
            items.push(
              h(
                "span",
                {
                  class: `yr-pet yr-role-${kind}${dim ? " dim" : ""}`,
                  role: "img",
                  "aria-label": `${T({ tr: "gerçek", en: "true" })}: ${T(PETS[c].name)}, ${T({ tr: "tahmin", en: "predicted" })}: ${T(PETS[r].name)} (${T(ROLE_TEXT[kind])})`,
                },
                h("span", { class: "yr-pet-true", text: PETS[c].icon }),
                h("span", { class: "yr-pet-pred", text: PETS[r].icon }),
              ),
            );
          }
        });
        const count = m.reduce((a, row) => a + row[c], 0);
        return h(
          "div",
          { class: "yr-pet-group" },
          h("div", { class: "yr-pet-head", text: `${T({ tr: "Gerçekte", en: "Really" })} ${name(c)} (${count})` }),
          h("div", { class: "yr-pet-row" }, ...items),
        );
      }),
    );
    // the matrix, coloured for the chosen class
    const head = h("tr", {}, h("td", {}), ...PETS.map((_, c) => h("th", { scope: "col", text: name(c) })));
    const body = m.map((row, r) =>
      h(
        "tr",
        {},
        h("th", { scope: "row", text: name(r) }),
        ...row.map((v, c) => {
          const kind = role(r, c, k);
          const b = h(
            "button",
            {
              type: "button",
              class: `yr-cell yr-role-${kind}`,
              "aria-pressed": String(!!picked && picked[0] === r && picked[1] === c),
              "aria-label": `${T({ tr: "tahmin", en: "predicted" })}: ${T(PETS[r].name)}, ${T({ tr: "gerçek", en: "true" })}: ${T(PETS[c].name)}: ${v} (${kind.toUpperCase()})`,
            },
            h("span", { class: "yr-cell-key", text: kind.toUpperCase() }),
            h("strong", { text: String(v) }),
          );
          b.addEventListener("click", () => {
            picked = picked && picked[0] === r && picked[1] === c ? null : [r, c];
            draw();
          });
          return h("td", {}, b);
        }),
      ),
    );
    table.replaceChildren(h("thead", {}, head), h("tbody", {}, ...body));
    // the hand computation for the chosen class
    const c = oneVsRest(m, k);
    const rowRest = m[k].filter((_, j) => j !== k);
    const colRest = m.filter((_, j) => j !== k).map((row) => row[k]);
    const others = m.flatMap((row, r) => row.filter((_, j) => r !== k && j !== k));
    const p = ratio(c.tp, c.tp + c.fp);
    const r = ratio(c.tp, c.tp + c.fn);
    const who = T(PETS[k].name);
    const line = (label: string, sum: number[], value: number, why: Text) =>
      h("li", {}, h("strong", { text: `${label} = ${sum.length > 1 ? `${sum.join(" + ")} = ` : ""}${value}` }), ` ${T(why)}`);
    work.replaceChildren(
      h("p", { class: "yr-mini", text: T({ tr: `"${who}" sınıfı, öbürlerine karşı:`, en: `The class "${who}" against the rest:` }) }),
      h(
        "ul",
        {},
        line("TP", [c.tp], c.tp, { tr: "(köşegendeki hücre)", en: "(the diagonal cell)" }),
        line("FP", rowRest, c.fp, {
          tr: `(satırın geri kalanı: "${who}" dendi ama değil)`,
          en: `(the rest of the row: called "${who}" but is not)`,
        }),
        line("FN", colRest, c.fn, {
          tr: `(sütunun geri kalanı: gerçekte ${who}, başka bir şey dendi)`,
          en: `(the rest of the column: really ${who}, called something else)`,
        }),
        line("TN", others, c.tn, { tr: "(öteki dört hücre)", en: "(the other four cells)" }),
      ),
      h("p", {
        class: "yr-calc",
        text:
          `P = ${c.tp} / (${c.tp} + ${c.fp}) = ${c.tp + c.fp ? nf(p) : T({ tr: "tanımsız (0/0)", en: "undefined (0/0)" })};  ` +
          `R = ${c.tp} / (${c.tp} + ${c.fn}) = ${nf(r)};  F1 = 2·P·R / (P + R) = ${nf(f1Of(p, r))}`,
      }),
    );
    // all classes, and the three averages
    const s = petScores(m);
    const cols = [
      T({ tr: "sınıf", en: "class" }),
      T({ tr: "kesinlik", en: "precision" }),
      T({ tr: "duyarlılık", en: "recall" }),
      "F1",
      T({ tr: "örnek", en: "support" }),
    ];
    const tr = (label: string, v: { p: number; r: number; f1: number }, support: string, cls = "") =>
      h(
        "tr",
        { class: cls },
        h("th", { scope: "row", text: label }),
        h("td", { text: nf(v.p) }),
        h("td", { text: nf(v.r) }),
        h("td", { text: nf(v.f1) }),
        h("td", { text: support }),
      );
    scores.replaceChildren(
      h("thead", {}, h("tr", {}, ...cols.map((t) => h("th", { scope: "col", text: t })))),
      h(
        "tbody",
        {},
        ...s.rows.map((x, i) => tr(name(i), x, String(x.support), i === k ? "yr-on" : "")),
        tr(T({ tr: "makro ortalama", en: "macro average" }), s.macro, String(s.total), "yo-total"),
        tr(T({ tr: "ağırlıklı ortalama", en: "weighted average" }), s.weighted, String(s.total)),
        tr(
          T({ tr: "mikro ortalama (= doğruluk)", en: "micro average (= accuracy)" }),
          { p: s.accuracy, r: s.accuracy, f1: s.accuracy },
          String(s.total),
        ),
      ),
    );
    note.textContent =
      scenario.key === "base"
        ? T({
            tr: `Doğruluk ${nf(s.accuracy)}, makro F1 ${nf(s.macro.f1)}: sınıflar dengeli olduğu için iki ortalama birbirine yakın.`,
            en: `Accuracy ${nf(s.accuracy)}, macro F1 ${nf(s.macro.f1)}: the classes are balanced, so the two averages are close.`,
          })
        : scenario.key === "rare"
          ? T({
              tr: `Doğruluk ${nf(s.accuracy)} ama makro F1 ${nf(s.macro.f1)}: model hiç "fare" demiyor (fare kesinliği tanımsız, 0 sayıldı; duyarlılığı 0). Mikro ortalama bunu saklar, makro ortalama gösterir.`,
              en: `Accuracy ${nf(s.accuracy)} but macro F1 ${nf(s.macro.f1)}: the model never says "mouse" (the mouse precision is undefined, counted as 0; its recall is 0). The micro average hides this; the macro average shows it.`,
            })
          : T({
              tr: `Her şeye "köpek": köpeğin duyarlılığı 1 ama kesinliği ${nf(s.rows[1].p)}; doğruluk ${nf(s.accuracy)}. Yüksek duyarlılık tek başına iyi model demek değildir.`,
              en: `"Dog" for everything: the dog's recall is 1 but its precision ${nf(s.rows[1].p)}; accuracy ${nf(s.accuracy)}. A high recall alone does not make a good model.`,
            });
  };
  const pickScenario = segmented(
    PET_SCENARIOS.map((x) => [x.key, T(x.name)]),
    scenario.key,
    (v) => {
      scenario = PET_SCENARIOS.find((x) => x.key === v) ?? PET_SCENARIOS[0];
      picked = null;
      draw();
    },
    T({ tr: "Senaryo", en: "Scenario" }),
  );
  const pickClass = segmented(
    PETS.map((_, i) => [String(i), name(i)]),
    "0",
    (v) => {
      k = Number(v);
      draw();
    },
    T({ tr: "Seçilen sınıf", en: "Chosen class" }),
  );
  draw();
  return demo(
    h("div", { class: "yr-picks" }, pickScenario, pickClass),
    cards,
    h(
      "div",
      { class: "yo-legend" },
      ...(["tp", "fp", "fn", "tn"] as Role[]).map((x) =>
        h("span", { class: "yo-key" }, h("i", { class: `yr-role-${x}` }), h("span", { text: T(ROLE_TEXT[x]) })),
      ),
    ),
    h(
      "div",
      { class: "yr-matrix-wrap" },
      h("div", { class: "yr-axis-top", text: T({ tr: "gerçek →", en: "true →" }) }),
      h("div", { class: "yr-matrix-row" }, h("div", { class: "yr-axis-side", text: T({ tr: "tahmin →", en: "predicted →" }) }), table),
    ),
    work,
    h("div", { class: "yo-table-wrap" }, scores),
    note,
    caption({
      tr: "Her kartta büyük simge gerçek sınıf, küçük simge tahmin. Renkler seçilen sınıfa göredir; bir hücreye tıklayınca o hücrenin fotoğrafları vurgulanır.",
      en: "On every card the large icon is the true class, the small one the prediction. The colours follow the chosen class; click a cell to highlight its photos.",
    }),
  );
}

// ---------------------------------------------------------------------------------------------------
// AP and mAP by hand
// ---------------------------------------------------------------------------------------------------

/** For each class: eight photos ranked by the model's score for that class, which are right, and how
 * many of the class's photos never made it into the list. */
export const RANKED: { hits: boolean[]; missing: number }[] = [
  { hits: [true, true, false, true, false, false, true, false], missing: 0 },
  { hits: [true, false, true, true, false, true, false, false], missing: 0 },
  { hits: [false, true, true, false, false, true, false, false], missing: 1 },
];
export const RANK_SCORES = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];

/** AP by hand: the mean of the precision at the rank of every right item; a positive never listed adds 0. */
export function handAp(hits: readonly boolean[], missing: number): { ap: number; terms: number[]; positives: number } {
  let tp = 0;
  const terms: number[] = [];
  hits.forEach((ok, i) => {
    if (!ok) return;
    tp++;
    terms.push(tp / (i + 1));
  });
  const positives = tp + missing;
  return { ap: positives ? terms.reduce((a, b) => a + b, 0) / positives : 0, terms, positives };
}

/** The same list read Ultralytics' and COCO's way: the precision envelope at 101 recall levels. */
export function cocoAp(hits: readonly boolean[], missing: number): number {
  const positives = hits.filter(Boolean).length + missing;
  const { precision, recall } = precisionRecall(RANK_SCORES.slice(0, hits.length), [...hits], positives);
  return averagePrecision(recall, precision);
}

export function apDemo(): HTMLElement {
  const lists = RANKED.map((x) => ({ hits: [...x.hits], missing: x.missing }));
  let k = 0;
  const table = h("table", { class: "yo-table yr-rank" });
  const chart = h("div", { class: "yo-pr yr-pr" });
  const work = h("div", { class: "yr-work", "aria-live": "polite" });
  const map = h("table", { class: "yo-table yr-class-table" });
  const name = (i: number) => `${PETS[i].icon} ${T(PETS[i].name)}`;
  const draw = () => {
    const list = lists[k];
    const { ap, terms, positives } = handAp(list.hits, list.missing);
    let tp = 0;
    const points: [number, number][] = [];
    const rows = list.hits.map((ok, i) => {
      if (ok) tp++;
      const p = tp / (i + 1);
      const r = ratio(tp, positives);
      points.push([r, p]);
      const toggle = h("button", {
        type: "button",
        class: `yr-hit${ok ? " on" : ""}`,
        "aria-pressed": String(ok),
        "aria-label": T({ tr: `${i + 1}. sıra doğru mu`, en: `Is rank ${i + 1} right` }),
        text: ok ? "✓" : "✗",
      });
      toggle.addEventListener("click", () => {
        list.hits[i] = !list.hits[i];
        draw();
      });
      return h(
        "tr",
        { class: ok ? "yr-on" : "" },
        h("th", { scope: "row", text: String(i + 1) }),
        h("td", { text: nf(RANK_SCORES[i]) }),
        h("td", {}, toggle),
        h("td", { text: `${tp}/${i + 1} = ${nf(p)}` }),
        h("td", { text: `${tp}/${positives} = ${nf(r)}` }),
      );
    });
    const head = [T({ tr: "sıra", en: "rank" }), T({ tr: "skor", en: "score" }), T({ tr: "doğru mu?", en: "right?" }), "P", "R"];
    table.replaceChildren(h("thead", {}, h("tr", {}, ...head.map((t) => h("th", { scope: "col", text: t })))), h("tbody", {}, ...rows));
    // PR curve and its envelope
    const W = 260;
    const H = 200;
    const P = 30;
    const X = (r: number) => P + r * (W - P - 8);
    const Y = (p: number) => H - P - p * (H - P - 8);
    const env = points.map(([, p]) => p);
    for (let i = env.length - 2; i >= 0; i--) env[i] = Math.max(env[i], env[i + 1]);
    let area = `M${X(0)},${Y(0)}L${X(0)},${Y(env[0] ?? 0)}`;
    points.forEach(
      ([r], i) =>
        (area += `L${X(r).toFixed(1)},${Y(env[i]).toFixed(1)}${i + 1 < points.length ? `L${X(r).toFixed(1)},${Y(env[i + 1]).toFixed(1)}` : ""}`),
    );
    area += `L${X(points.at(-1)?.[0] ?? 0)},${Y(0)}Z`;
    const line = points.map(([r, p], i) => `${i ? "L" : "M"}${X(r).toFixed(1)},${Y(p).toFixed(1)}`).join("");
    const dots = points
      .map(
        ([r, p], i) =>
          `<circle cx="${X(r).toFixed(1)}" cy="${Y(p).toFixed(1)}" r="4" class="${list.hits[i] ? "yr-dot-hit" : "yr-dot-miss"}"/>`,
      )
      .join("");
    chart.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${T({ tr: "Kesinlik-duyarlılık eğrisi ve zarfı", en: "Precision-recall curve and its envelope" })}">
      <path d="${area}" class="yo-pr-area"/>
      <line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(0)}" class="baseline"/><line x1="${X(0)}" y1="${Y(0)}" x2="${X(0)}" y2="${Y(1)}" class="baseline"/>
      <path d="${line}" class="yo-pr-line"/>${dots}
      <text x="${X(1)}" y="${H - 8}" text-anchor="end" class="tick">${T({ tr: "duyarlılık", en: "recall" })} →</text>
      <text x="${X(0) + 6}" y="${Y(0) - 6}" class="tick">${T({ tr: "kesinlik", en: "precision" })} ↑</text>
      <text x="${X(0) - 4}" y="${Y(0) + 4}" text-anchor="end" class="tick">0</text><text x="${X(0) - 4}" y="${Y(1) + 4}" text-anchor="end" class="tick">1</text>
      <text x="${X(1)}" y="${Y(0) + 14}" text-anchor="middle" class="tick">1</text>
    </svg>`;
    const sum = [...terms.map((t) => nf(t)), ...Array.from({ length: list.missing }, () => "0")];
    work.replaceChildren(
      h("p", {
        class: "yr-calc",
        text: `AP(${T(PETS[k].name)}) = (${sum.join(" + ") || "0"}) / ${positives} = ${nf(ap)}`,
      }),
      h("p", {
        class: "sub yo-caption",
        text: T({
          tr: `Her ✓ satırındaki kesinlik toplanır, ${T(PETS[k].name)} fotoğraflarının sayısına (${positives}) bölünür${list.missing ? `; listeye hiç girmeyen ${list.missing} fotoğraf 0 olarak sayılır` : ""}. Ultralytics ve COCO'nun yöntemiyle (zarf, 101 duyarlılık düzeyi): ${nf(cocoAp(list.hits, list.missing))}.`,
          en: `The precision on every ✓ row is added up and divided by the number of ${T(PETS[k].name)} photos (${positives})${list.missing ? `; the ${list.missing} photo never listed counts as 0` : ""}. Ultralytics' and COCO's way (the envelope at 101 recall levels): ${nf(cocoAp(list.hits, list.missing))}.`,
        }),
      }),
    );
    const aps = lists.map((x) => handAp(x.hits, x.missing).ap);
    const mean = aps.reduce((a, b) => a + b, 0) / aps.length;
    map.replaceChildren(
      h("thead", {}, h("tr", {}, h("th", { scope: "col", text: T({ tr: "sınıf", en: "class" }) }), h("th", { scope: "col", text: "AP" }))),
      h(
        "tbody",
        {},
        ...aps.map((v, i) =>
          h("tr", { class: i === k ? "yr-on" : "" }, h("th", { scope: "row", text: name(i) }), h("td", { text: nf(v) })),
        ),
        h(
          "tr",
          { class: "yo-total" },
          h("th", { scope: "row", text: `mAP = (${aps.map((v) => nf(v)).join(" + ")}) / 3` }),
          h("td", { text: nf(mean) }),
        ),
      ),
    );
  };
  const pick = segmented(
    PETS.map((_, i) => [String(i), name(i)]),
    "0",
    (v) => {
      k = Number(v);
      draw();
    },
    T({ tr: "Sınıf", en: "Class" }),
  );
  draw();
  return demo(
    pick,
    h("div", { class: "yr-ap-row" }, h("div", { class: "yo-table-wrap" }, table), chart),
    work,
    h("div", { class: "yo-table-wrap" }, map),
    caption({
      tr: "✓ ya da ✗ düğmesine basarak bir fotoğrafın doğru olup olmadığını değiştirin: tablo, eğri, AP ve mAP yeniden hesaplanır. Yeşil noktalar ✓, gri noktalar ✗; taralı alan zarfın altı.",
      en: "Press ✓ or ✗ to change whether a photo is right: the table, the curve, the AP and the mAP are computed again. Green dots are ✓, grey dots ✗; the shaded area lies under the envelope.",
    }),
  );
}
