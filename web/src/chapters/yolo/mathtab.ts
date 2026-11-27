/**
 * The YOLO Math tab: eight sections rendered by lesson.js (createLesson), with derivations revealed one
 * line at a time and widgets that compute with the chapter's engine (engines/yolo/).
 *
 * The formulas follow the installed Ultralytics code: boxes decoded from four distances to the cell
 * centre (tal.py: make_anchors, dist2bbox), the CIoU box loss (metrics.py: bbox_iou), binary
 * cross-entropy against IoU-scaled class targets (tal.py), the mask loss averaged inside the box
 * (loss.py: single_mask_loss) and COCO's 101-point AP (metrics.py: compute_ap). The numbers about our
 * model were measured with scripts/math_numbers.py in the training folder.
 */
import { fmtNum, h } from "../../shared/core.js";
import { L } from "../../shared/i18n.js";
import { createLesson } from "../../shared/lesson.js";
import { KIT } from "../../shared/kit.js";
import type { Text } from "../../shared/chapters.ts";
import { averagePrecision, iouVariants, precisionRecall } from "../../engines/yolo/metrics.ts";

const R = String.raw;
const nf = (v: number, d = 2) => fmtNum(v, d) as string;
/** A number inside TeX: the Turkish decimal comma needs braces. */
const tx = (v: number, d = 2) => nf(v, d).replace(/,/g, "{,}");
const T = (t: Text) => L(t) as string;
const eq = (tex: string) => h("div", { class: "m-eq", text: `\\(\\displaystyle ${tex}\\)` });
const { slider } = KIT as {
  slider: (
    label: string,
    o: { min: number; max: number; step: number; value: number; onInput: (v: number) => void; fmt?: (v: number) => string },
  ) => HTMLElement;
};

type State = Record<string, Record<string, number | number[] | boolean[]>>;

// =====================================================================
// Widgets
// =====================================================================

const svgNode = (markup: string) => {
  const box = h("div", { class: "ym-svg" });
  box.innerHTML = markup;
  return box;
};

const WIDGETS: Record<string, (state: State) => HTMLElement> = {
  /** Input size → grids, number of candidates, output shapes. */
  sizes(state) {
    const s = (state.sizes ??= { size: 640 });
    const out = h("div", { class: "m-widget-out" });
    const draw = () => {
      const S = s.size as number;
      const g = [8, 16, 32].map((st) => S / st);
      const N = g.reduce((a, b) => a + b * b, 0);
      out.replaceChildren(
        eq(R`${g.map((v) => `${v}^2`).join(" + ")} = ${g.map((v) => v * v).join(" + ")} = ${N} = \tfrac{21}{1024}\cdot ${S}^2`),
        eq(
          R`\text{${T({ tr: "çıktı", en: "output" })}}: 1 \times 38 \times ${N}, \qquad \text{${T({ tr: "prototipler", en: "prototypes" })}}: 1 \times 32 \times ${S / 4} \times ${S / 4}`,
        ),
        h("p", {
          class: "m-why",
          text: T({
            tr: `${S} piksellik girişte ${N} aday var; giriş boyu iki katına çıkınca aday sayısı dört katına çıkar, çünkü sayı S² ile büyür.`,
            en: `A ${S}-pixel input gives ${N} candidates; doubling the input size quadruples the number, because it grows with S².`,
          }),
        }),
      );
    };
    const box = h(
      "div",
      { class: "m-widget" },
      h(
        "div",
        { class: "t-controls" },
        slider(T({ tr: "giriş S", en: "input S" }), {
          min: 320,
          max: 1280,
          step: 32,
          value: s.size as number,
          fmt: (v) => String(v),
          onInput: (v) => ((s.size = v), draw()),
        }),
      ),
      out,
    );
    draw();
    return box;
  },

  /** A cell, its anchor point and four distances → the box. */
  decode(state) {
    const s = (state.decode ??= { l: 2.992, t: 4.945, r: 2.555, b: 3.789 });
    const out = h("div", { class: "m-widget-out" });
    const stride = 16;
    const [gx, gy] = [37, 24];
    const draw = () => {
      const ax = (gx + 0.5) * stride;
      const ay = (gy + 0.5) * stride;
      const [l, t, r, b] = [s.l, s.t, s.r, s.b] as number[];
      const x1 = ax - l * stride;
      const y1 = ay - t * stride;
      const x2 = ax + r * stride;
      const y2 = ay + b * stride;
      // a 12 × 12-cell window around the cell, in pixels
      const ox = (gx - 5) * stride;
      const oy = (gy - 6) * stride;
      const k = 300 / (12 * stride);
      const X = (v: number) => ((v - ox) * k).toFixed(1);
      const Y = (v: number) => ((v - oy) * k).toFixed(1);
      let grid = "";
      for (let i = 0; i <= 12; i++)
        grid += `<line x1="${i * 25}" y1="0" x2="${i * 25}" y2="300" class="ym-grid"/><line x1="0" y1="${i * 25}" x2="300" y2="${i * 25}" class="ym-grid"/>`;
      out.replaceChildren(
        svgNode(`<svg viewBox="0 0 300 300" role="img" aria-label="${T({ tr: "Hücre, çapa noktası ve kutu", en: "Cell, anchor point and box" })}">${grid}
          <rect x="${X(gx * stride)}" y="${Y(gy * stride)}" width="25" height="25" class="ym-cell"/>
          <rect x="${X(x1)}" y="${Y(y1)}" width="${((x2 - x1) * k).toFixed(1)}" height="${((y2 - y1) * k).toFixed(1)}" class="ym-box"/>
          <line x1="${X(ax)}" y1="${Y(ay)}" x2="${X(x1)}" y2="${Y(ay)}" class="ym-arrow"/><line x1="${X(ax)}" y1="${Y(ay)}" x2="${X(x2)}" y2="${Y(ay)}" class="ym-arrow"/>
          <line x1="${X(ax)}" y1="${Y(ay)}" x2="${X(ax)}" y2="${Y(y1)}" class="ym-arrow"/><line x1="${X(ax)}" y1="${Y(ay)}" x2="${X(ax)}" y2="${Y(y2)}" class="ym-arrow"/>
          <circle cx="${X(ax)}" cy="${Y(ay)}" r="4" class="ym-anchor"/></svg>`),
        eq(R`a = \big((${gx} + ${tx(0.5, 1)})\cdot ${stride},\ (${gy} + ${tx(0.5, 1)})\cdot ${stride}\big) = (${ax},\ ${ay})`),
        eq(
          R`(x_1, y_1, x_2, y_2) = (${ax} - ${tx(l)}\cdot ${stride},\ ${ay} - ${tx(t)}\cdot ${stride},\ ${ax} + ${tx(r)}\cdot ${stride},\ ${ay} + ${tx(b)}\cdot ${stride}) = (${tx(x1, 1)},\ ${tx(y1, 1)},\ ${tx(x2, 1)},\ ${tx(y2, 1)})`,
        ),
      );
    };
    const box = h(
      "div",
      { class: "m-widget" },
      h(
        "div",
        { class: "t-controls" },
        ...(["l", "t", "r", "b"] as const).map((key) =>
          slider(key, {
            min: 0,
            max: 8,
            step: 0.05,
            value: s[key] as number,
            fmt: (v) => nf(v, 2),
            onInput: (v) => ((s[key] = v), draw()),
          }),
        ),
      ),
      out,
    );
    draw();
    return box;
  },

  /** IoU, GIoU, DIoU and CIoU as the prediction slides past the true box. */
  ious(state) {
    const s = (state.ious ??= { dx: 0.4 });
    const out = h("div", { class: "m-widget-out" });
    const truth = [0, 0, 10, 10];
    const at = (dx: number) => iouVariants(truth, [dx * 10, 2, dx * 10 + 12, 10]);
    const draw = () => {
      const dx = s.dx as number;
      const v = at(dx);
      // the curves over dx from -2.5 to 2.5
      const xs = Array.from({ length: 201 }, (_, i) => -2.5 + i * 0.025);
      const W = 320;
      const H = 170;
      const X = (d: number) => (((d + 2.5) / 5) * (W - 40) + 30).toFixed(1);
      const Y = (y: number) => (H - 20 - ((y + 1) / 2) * (H - 30)).toFixed(1);
      const path = (k: "iou" | "giou" | "ciou") => xs.map((d, i) => `${i ? "L" : "M"}${X(d)},${Y(at(d)[k])}`).join("");
      out.replaceChildren(
        svgNode(`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="IoU, GIoU, CIoU">
          <line x1="30" x2="${W - 10}" y1="${Y(0)}" y2="${Y(0)}" class="baseline"/>
          <text x="26" y="${Y(1) + 4}" text-anchor="end" class="tick">1</text><text x="26" y="${Y(0) + 4}" text-anchor="end" class="tick">0</text><text x="26" y="${Y(-1) + 4}" text-anchor="end" class="tick">−1</text>
          <path d="${path("iou")}" class="ym-c1"/><path d="${path("giou")}" class="ym-c2"/><path d="${path("ciou")}" class="ym-c3"/>
          <line x1="${X(dx)}" x2="${X(dx)}" y1="10" y2="${H - 20}" class="crosshair"/>
          <text x="${W - 10}" y="${H - 4}" text-anchor="end" class="tick">${T({ tr: "kayma (kutu genişliği)", en: "shift (box widths)" })} →</text></svg>`),
        h(
          "div",
          { class: "yo-legend" },
          ...(
            [
              ["IoU", "ym-k1"],
              ["GIoU", "ym-k2"],
              ["CIoU", "ym-k3"],
            ] as const
          ).map(([name, cls]) => h("span", { class: "yo-key" }, h("i", { class: cls }), name)),
        ),
        eq(
          R`\mathrm{IoU} = ${tx(v.iou, 3)}, \quad \mathrm{GIoU} = ${tx(v.giou, 3)}, \quad \mathrm{DIoU} = ${tx(v.diou, 3)}, \quad \mathrm{CIoU} = ${tx(v.ciou, 3)}, \quad L_{\text{box}} = 1 - \mathrm{CIoU} = ${tx(1 - v.ciou, 3)}`,
        ),
        h("p", {
          class: "m-why",
          text:
            v.iou === 0
              ? T({
                  tr: "Kutular kesişmiyor: IoU düz 0, eğimi yok. GIoU ve CIoU ise kutu yaklaştıkça artmaya devam ediyor, yani gradyan kutuyu doğru yöne itiyor.",
                  en: "The boxes do not overlap: IoU is flat at 0, with no slope. GIoU and CIoU keep rising as the box comes closer, so the gradient still pushes it the right way.",
                })
              : T({
                  tr: "Kutular kesişiyor: dördü de kayma azaldıkça artar. Tam üst üste gelince IoU 1'e, kayıp 0'a yaklaşır (kutuların boyları farklı olduğu için tam 1 olmaz).",
                  en: "The boxes overlap: all four grow as the shift shrinks. Right on top of each other IoU nears 1 and the loss 0 (not exactly, the boxes differ in size).",
                }),
        }),
      );
    };
    const box = h(
      "div",
      { class: "m-widget" },
      h(
        "div",
        { class: "t-controls" },
        slider(T({ tr: "kayma", en: "shift" }), {
          min: -2.5,
          max: 2.5,
          step: 0.05,
          value: s.dx as number,
          fmt: (v) => nf(v, 2),
          onInput: (v) => ((s.dx = v), draw()),
        }),
      ),
      out,
    );
    draw();
    return box;
  },

  /** Sigmoid, binary cross-entropy and its gradient p − y. */
  bce(state) {
    const s = (state.bce ??= { z: 1, y: 0.8 });
    const out = h("div", { class: "m-widget-out" });
    const draw = () => {
      const z = s.z as number;
      const y = s.y as number;
      const p = 1 / (1 + Math.exp(-z));
      const loss = -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      out.replaceChildren(
        eq(
          R`p = \sigma(${tx(z)}) = ${tx(p, 3)}, \qquad L = -\big[${tx(y)}\ln ${tx(p, 3)} + ${tx(1 - y)}\ln ${tx(1 - p, 3)}\big] = ${tx(loss, 3)}`,
        ),
        eq(R`\frac{\partial L}{\partial z} = p - y = ${tx(p, 3)} - ${tx(y)} = ${tx(p - y, 3)}`),
        h("p", {
          class: "m-why",
          text: T({
            tr:
              Math.abs(p - y) < 0.02
                ? "Skor hedefe ulaştı: gradyan neredeyse 0, öğrenme durur."
                : p > y
                  ? "Skor hedefin üstünde: gradyan artı, z küçültülecek."
                  : "Skor hedefin altında: gradyan eksi, z büyütülecek.",
            en:
              Math.abs(p - y) < 0.02
                ? "The score reached its target: the gradient is almost 0, learning stops."
                : p > y
                  ? "The score is above its target: the gradient is positive, z will be lowered."
                  : "The score is below its target: the gradient is negative, z will be raised.",
          }),
        }),
      );
    };
    const box = h(
      "div",
      { class: "m-widget" },
      h(
        "div",
        { class: "t-controls" },
        slider("z", { min: -6, max: 6, step: 0.1, value: s.z as number, fmt: (v) => nf(v, 1), onInput: (v) => ((s.z = v), draw()) }),
        slider(T({ tr: "hedef y", en: "target y" }), {
          min: 0,
          max: 1,
          step: 0.05,
          value: s.y as number,
          fmt: (v) => nf(v, 2),
          onInput: (v) => ((s.y = v), draw()),
        }),
      ),
      out,
    );
    draw();
    return box;
  },

  /** Gradient descent on four mask coefficients until σ(Σ c_k P_k) draws a disc. */
  maskfit(state) {
    const n = 24;
    const protos: Float32Array[] = [new Float32Array(n * n), new Float32Array(n * n), new Float32Array(n * n), new Float32Array(n * n)];
    const target = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = ((j + 0.5) / n) * 2 - 1;
        const y = ((i + 0.5) / n) * 2 - 1;
        const k = i * n + j;
        protos[0][k] = x;
        protos[1][k] = y;
        protos[2][k] = 1;
        protos[3][k] = x * x + y * y;
        target[k] = (x - 0.2) ** 2 + (y + 0.1) ** 2 < 0.3 ? 1 : 0;
      }
    }
    const s = (state.maskfit ??= { c: [0, 0, 0, 0], steps: 0 });
    const out = h("div", { class: "m-widget-out" });
    const lr = 4;
    const lossAndGrad = (c: number[]) => {
      const g = [0, 0, 0, 0];
      let loss = 0;
      const m = new Float32Array(n * n);
      for (let k = 0; k < n * n; k++) {
        const z = c[0] * protos[0][k] + c[1] * protos[1][k] + c[2] * protos[2][k] + c[3] * protos[3][k];
        const p = 1 / (1 + Math.exp(-z));
        m[k] = p;
        loss -= target[k] * Math.log(p + 1e-9) + (1 - target[k]) * Math.log(1 - p + 1e-9);
        for (let q = 0; q < 4; q++) g[q] += (p - target[k]) * protos[q][k];
      }
      return { loss: loss / (n * n), grad: g.map((v) => v / (n * n)), m };
    };
    const paintMask = (canvas: HTMLCanvasElement, values: Float32Array, outline: boolean) => {
      const ctx = canvas.getContext("2d")!;
      const img = ctx.createImageData(n, n);
      for (let k = 0; k < n * n; k++) {
        const v = Math.round(255 * values[k]);
        const edge = outline && target[k] === 1;
        img.data.set(edge ? [Math.min(255, v + 40), v, Math.max(0, v - 60), 255] : [v, v, v, 255], 4 * k);
      }
      ctx.putImageData(img, 0, 0);
    };
    const tgt = h("canvas", { class: "ym-mask", width: n, height: n }) as HTMLCanvasElement;
    const cur = h("canvas", { class: "ym-mask", width: n, height: n }) as HTMLCanvasElement;
    paintMask(tgt, target, false);
    const draw = () => {
      const c = s.c as number[];
      const { loss, grad, m } = lossAndGrad(c);
      paintMask(cur, m, true);
      out.replaceChildren(
        h(
          "div",
          { class: "ym-masks" },
          h("figure", {}, tgt, h("figcaption", { text: T({ tr: "hedef maske y", en: "target mask y" }) })),
          h("figure", {}, cur, h("figcaption", { text: T({ tr: `σ(Σ cₖPₖ), ${s.steps} adım`, en: `σ(Σ cₖPₖ), ${s.steps} steps` }) })),
        ),
        eq(R`c = (${c.map((v) => tx(v)).join(R`,\ `)}), \qquad L = ${tx(loss, 4)}`),
        eq(R`\nabla_c L = (${grad.map((v) => tx(v, 4)).join(R`,\ `)})`),
      );
    };
    const step = (k: number) => {
      for (let i = 0; i < k; i++) {
        const { grad } = lossAndGrad(s.c as number[]);
        s.c = (s.c as number[]).map((v, q) => v - lr * grad[q]);
        s.steps = (s.steps as number) + 1;
      }
      draw();
    };
    const button = (label: string, fn: () => void) => {
      const b = h("button", { type: "button", class: "btn small", text: label });
      b.addEventListener("click", fn);
      return b;
    };
    const box = h(
      "div",
      { class: "m-widget" },
      h(
        "div",
        { class: "t-controls" },
        button(T({ tr: "1 adım", en: "1 step" }), () => step(1)),
        button(T({ tr: "50 adım", en: "50 steps" }), () => step(50)),
        button(T({ tr: "Baştan", en: "Start over" }), () => {
          s.c = [0, 0, 0, 0];
          s.steps = 0;
          draw();
        }),
      ),
      out,
      h("p", {
        class: "m-why",
        text: T({
          tr: "Dört prototip: P₁ = x, P₂ = y, P₃ = 1, P₄ = x² + y². Gradyan inişi (öğrenme oranı 4) yalnızca dört katsayıyı değiştirir; daire, P₄'ün eksi ve P₃'ün artı katsayısıyla çizilir, x ve y katsayıları onu yerine kaydırır.",
          en: "Four prototypes: P₁ = x, P₂ = y, P₃ = 1, P₄ = x² + y². Gradient descent (learning rate 4) changes only the four coefficients; the disc is drawn by a negative coefficient on P₄ and a positive one on P₃, while those of x and y move it into place.",
        }),
      }),
    );
    draw();
    return box;
  },

  /** A hand-made PR curve: detections sorted by score, each correct or not. */
  pr(state) {
    const s = (state.pr ??= { truths: 5, correct: [true, true, false, true, false, true, false, false] });
    const scores = [0.95, 0.91, 0.85, 0.8, 0.7, 0.62, 0.5, 0.4];
    const out = h("div", { class: "m-widget-out" });
    const draw = () => {
      const correct = s.correct as boolean[];
      const truths = s.truths as number;
      const tp = correct.filter(Boolean).length;
      if (truths < tp) s.truths = tp;
      const { precision, recall } = precisionRecall(scores, correct, s.truths as number);
      const ap = averagePrecision(recall, precision);
      const env = [...precision];
      for (let i = env.length - 2; i >= 0; i--) env[i] = Math.max(env[i], env[i + 1]);
      const rows = scores.map((sc, i) => {
        const toggle = h("button", {
          type: "button",
          class: `btn small ym-tp${correct[i] ? " on" : ""}`,
          text: correct[i] ? "TP" : "FP",
          "aria-pressed": String(correct[i]),
        });
        toggle.addEventListener("click", () => {
          correct[i] = !correct[i];
          draw();
        });
        return h(
          "tr",
          {},
          h("td", { text: String(i + 1) }),
          h("td", { text: nf(sc, 2) }),
          h("td", {}, toggle),
          h("td", { text: nf(precision[i], 3) }),
          h("td", { text: nf(recall[i], 3) }),
          h("td", { text: nf(env[i], 3) }),
        );
      });
      const W = 220;
      const H = 160;
      const X = (r: number) => (24 + r * (W - 34)).toFixed(1);
      const Y = (p: number) => (H - 18 - p * (H - 28)).toFixed(1);
      let area = `M${X(0)},${Y(0)}`;
      recall.forEach((r, i) => (area += `L${X(i ? recall[i - 1] : 0)},${Y(env[i])}L${X(r)},${Y(env[i])}`));
      area += `L${X(recall.at(-1) ?? 0)},${Y(0)}Z`;
      const line = recall.map((r, i) => `${i ? "L" : "M"}${X(r)},${Y(precision[i])}`).join("");
      out.replaceChildren(
        h(
          "div",
          { class: "ym-pr" },
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
                  ...["#", T({ tr: "skor", en: "score" }), "", "P", "R", T({ tr: "zarf", en: "envelope" })].map((t) =>
                    h("th", { scope: "col", text: t }),
                  ),
                ),
              ),
              h("tbody", {}, ...rows),
            ),
          ),
          svgNode(`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="PR"><path d="${area}" class="yo-pr-area"/><path d="${line}" class="ym-c1"/>
            <line x1="${X(0)}" x2="${X(1)}" y1="${Y(0)}" y2="${Y(0)}" class="baseline"/><line x1="${X(0)}" x2="${X(0)}" y1="${Y(0)}" y2="${Y(1)}" class="baseline"/>
            <text x="${X(1)}" y="${H - 2}" text-anchor="end" class="tick">R →</text><text x="${Number(X(0)) + 4}" y="${Number(Y(1)) + 8}" class="tick">P ↑</text></svg>`),
        ),
        eq(R`\mathrm{AP} = \frac{1}{101}\sum_{k=0}^{100} \hat P\Big(R \ge \tfrac{k}{100}\Big) = ${tx(ap, 4)}`),
      );
    };
    const box = h(
      "div",
      { class: "m-widget" },
      h(
        "div",
        { class: "t-controls" },
        slider(T({ tr: "gerçek nesne sayısı", en: "true objects" }), {
          min: 4,
          max: 10,
          step: 1,
          value: s.truths as number,
          fmt: (v) => String(v),
          onInput: (v) => ((s.truths = v), draw()),
        }),
      ),
      out,
      h("p", {
        class: "m-why",
        text: T({
          tr: "TP/FP düğmelerine basarak bir tespitin doğru ya da yanlış olduğunu değiştirin.",
          en: "Press a TP/FP button to make a detection correct or wrong.",
        }),
      }),
    );
    draw();
    return box;
  },

  /** Relative area error of a square building when its outline moves by δ on every side. */
  area(state) {
    const s = (state.area ??= { a: 10, d: 1.22 });
    const out = h("div", { class: "m-widget-out" });
    const draw = () => {
      const a = s.a as number;
      const d = s.d as number;
      const small = ((a - 2 * d) / a) ** 2 - 1;
      const big = ((a + 2 * d) / a) ** 2 - 1;
      const pct = (v: number) => `${v >= 0 ? "+" : "-"}${tx(Math.abs(100 * v), 0)}\\%`;
      out.replaceChildren(
        eq(
          R`\frac{(${tx(a, 0)} - 2\cdot ${tx(d)})^2}{${tx(a, 0)}^2} - 1 = ${pct(small)}, \qquad \frac{(${tx(a, 0)} + 2\cdot ${tx(d)})^2}{${tx(a, 0)}^2} - 1 = ${pct(big)}`,
        ),
        eq(R`\approx \pm\frac{4\delta}{a} = \pm ${tx((400 * d) / a, 0)}\%`),
      );
    };
    const box = h(
      "div",
      { class: "m-widget" },
      h(
        "div",
        { class: "t-controls" },
        slider(T({ tr: "kenar a (m)", en: "side a (m)" }), {
          min: 5,
          max: 100,
          step: 1,
          value: s.a as number,
          fmt: (v) => String(v),
          onInput: (v) => ((s.a = v), draw()),
        }),
        slider(T({ tr: "kayma δ (m)", en: "shift δ (m)" }), {
          min: 0,
          max: 2.5,
          step: 0.01,
          value: s.d as number,
          fmt: (v) => nf(v, 2),
          onInput: (v) => ((s.d = v), draw()),
        }),
      ),
      out,
    );
    draw();
    return box;
  },
};

// =====================================================================
// Sections
// =====================================================================

const MATH = [
  {
    id: "ym-notation",
    title: { tr: "Gösterim ve boyutlar", en: "Notation and sizes" },
    blocks: [
      {
        p: {
          tr: R`Girişi \(S \times S\) piksellik bir görüntü olarak yazalım (bu bölümde \(S = 640\)). Gövde onu \(s \in \{8, 16, 32\}\) adımlarıyla (stride) küçültür; her ölçekte \(G_s = S/s\) satır ve sütunluk bir ızgara olur. Her hücre bir adaydır.`,
          en: R`Write the input as an \(S \times S\)-pixel image (in this chapter \(S = 640\)). The backbone shrinks it with strides \(s \in \{8, 16, 32\}\); at every scale there is a grid of \(G_s = S/s\) rows and columns. Every cell is a candidate.`,
        },
      },
      {
        table: {
          head: [
            { tr: "Sembol", en: "Symbol" },
            { tr: "Değer", en: "Value" },
            { tr: "Anlamı", en: "Meaning" },
          ],
          rows: [
            [R`\(S\)`, "640", { tr: "giriş boyu (letterbox'tan sonra)", en: "input size (after the letterbox)" }],
            [R`\(s,\ G_s\)`, R`\(8, 16, 32;\ 80, 40, 20\)`, { tr: "stride ve ızgara boyu", en: "stride and grid size" }],
            [R`\(N\)`, "8400", { tr: "aday sayısı", en: "number of candidates" }],
            [R`\(C\)`, "2", { tr: "sınıf sayısı (bina, spor sahası)", en: "number of classes (building, sports field)" }],
            [R`\(K\)`, "32", { tr: "maske prototipi sayısı", en: "number of mask prototypes" }],
            [
              R`\(4 + C + K\)`,
              "38",
              {
                tr: "bir adayın sayıları: kutu, sınıf skorları, maske katsayıları",
                en: "a candidate's numbers: box, class scores, mask coefficients",
              },
            ],
            [R`\(P_k\)`, R`\(160 \times 160\)`, { tr: "prototip haritaları, \\(S/4\\) boyunda", en: "prototype maps, \\(S/4\\) in size" }],
          ],
        },
      },
      {
        derive: {
          title: { tr: "Aday sayısı neden 8400?", en: "Why 8400 candidates?" },
          steps: [
            {
              tex: R`N = \Big(\frac{S}{8}\Big)^2 + \Big(\frac{S}{16}\Big)^2 + \Big(\frac{S}{32}\Big)^2`,
              why: { tr: "Üç ızgaranın hücreleri toplanır.", en: "The cells of the three grids add up." },
            },
            {
              tex: R`= \frac{S^2}{64}\Big(1 + \frac14 + \frac1{16}\Big) = \frac{S^2}{64}\cdot\frac{21}{16} = \frac{21\,S^2}{1024}`,
              why: {
                tr: "Her ölçek bir öncekinin dörtte biri kadar hücre verir: geometrik bir toplam.",
                en: "Each scale has a quarter of the cells of the one before: a geometric sum.",
              },
            },
            {
              tex: R`S = 640:\quad N = \frac{21\cdot 409\,600}{1024} = 8400`,
              why: {
                tr: "Sayının dörtte üçünden fazlası (6400) en ince ızgaradan gelir.",
                en: "More than three quarters (6400) come from the finest grid.",
              },
            },
          ],
        },
      },
      { widget: "sizes" },
      {
        p: {
          tr: R`Ağın ağırlıkları: YOLO26n-seg'de 3.083.302 [[parameter|parametre]] var, SimpleCNN'in 582.218'inin yaklaşık 5,3 katı; YOLO26n-sem'de 1.631.212. Çoğu gövdenin ve boynun evrişim filtrelerindedir; Bölüm 3'teki parametre sayımı burada da aynen geçerlidir.`,
          en: R`The network's weights: YOLO26n-seg has 3,083,302 [[parameter|parameters]], about 5.3 times SimpleCNN's 582,218; YOLO26n-sem has 1,631,212. Most sit in the convolution filters of the backbone and the neck; the parameter count of chapter 3 works the same way here.`,
        },
      },
      { links: [{ tab: "theory", section: "y-grid" }] },
    ],
  },
  {
    id: "ym-boxes",
    title: { tr: "Kutunun kodlanması", en: "How a box is encoded" },
    blocks: [
      {
        p: {
          tr: R`Her hücrenin bir <b>çapa noktası</b> vardır: hücrenin merkezi, piksel cinsinden \(a = \big((g_x + \tfrac12)\,s,\ (g_y + \tfrac12)\,s\big)\). YOLO26 bu noktadan kutunun dört kenarına uzaklığı tahmin eder: \(l, t, r, b\) (sol, üst, sağ, alt), stride cinsinden. Önceden tanımlı çapa kutuları yoktur ("çapasız").`,
          en: R`Every cell has an <b>anchor point</b>: the centre of the cell, in pixels \(a = \big((g_x + \tfrac12)\,s,\ (g_y + \tfrac12)\,s\big)\). YOLO26 predicts the distances from that point to the four sides of the box: \(l, t, r, b\) (left, top, right, bottom), in strides. There are no predefined anchor boxes ("anchor-free").`,
        },
      },
      { eq: R`x_1 = a_x - l\,s, \qquad y_1 = a_y - t\,s, \qquad x_2 = a_x + r\,s, \qquad y_2 = a_y + b\,s` },
      {
        p: {
          tr: R`Merkez yazılışına geçmek kolaydır: \(c_x = (x_1 + x_2)/2\), \(w = x_2 - x_1 = (l + r)\,s\). Gerçek bir örnek, Şanghay karesinin 40 × 40 ızgarasında en yüksek skorlu aday (hücre (37, 24), bina skoru 0,81): uzaklıklar \(l = 2{,}99,\ t = 4{,}95,\ r = 2{,}56,\ b = 3{,}79\).`,
          en: R`Getting to the centre form is easy: \(c_x = (x_1 + x_2)/2\), \(w = x_2 - x_1 = (l + r)\,s\). A real example, the highest-scoring candidate on the 40 × 40 grid of the Shanghai tile (cell (37, 24), building score 0.81): distances \(l = 2.99,\ t = 4.95,\ r = 2.56,\ b = 3.79\).`,
        },
      },
      { widget: "decode" },
      {
        p: {
          tr: R`Neden stride ile ölçeklenir? Aynı \(l = 3\) değeri ince ızgarada 24, kaba ızgarada 96 pikseldir. Böylece her ölçek kendi büyüklüğündeki nesneler için benzer sayılar üretir ve ağın çıktıları hep küçük kalır. YOLO26'da bu dört sayı doğrudan tahmin edilir; YOLOv8'deki DFL (uzaklığın bir olasılık dağılımı olarak tahmini) kaldırılmıştır.`,
          en: R`Why scale by the stride? The same \(l = 3\) is 24 pixels on the fine grid and 96 on the coarse one. So each scale produces similar numbers for objects of its own size and the network's outputs stay small. In YOLO26 the four numbers are predicted directly; YOLOv8's DFL (predicting the distance as a probability distribution) is gone.`,
        },
      },
      {
        links: [
          { tab: "theory", section: "y-grid" },
          { tab: "theory", section: "y-iou" },
        ],
      },
    ],
  },
  {
    id: "ym-iou",
    title: { tr: "IoU ve IoU kayıpları", en: "IoU and the IoU losses" },
    blocks: [
      {
        p: {
          tr: R`İki kutu \(A, B\) için kesişim genişliği \(\max\big(0, \min(x_2^A, x_2^B) - \max(x_1^A, x_1^B)\big)\), yüksekliği de benzerdir. [[iou|IoU]] = kesişim / birleşim. Kutu kaybı olarak \(1 - \mathrm{IoU}\) kullanılabilir, ama bir sorunu var:`,
          en: R`For two boxes \(A, B\) the width of the intersection is \(\max\big(0, \min(x_2^A, x_2^B) - \max(x_1^A, x_1^B)\big)\), and its height likewise. [[iou|IoU]] = intersection / union. \(1 - \mathrm{IoU}\) could serve as the box loss, but it has a problem:`,
        },
      },
      {
        derive: {
          title: { tr: "Kutular kesişmeyince IoU öğretmez", en: "IoU teaches nothing when the boxes do not overlap" },
          steps: [
            {
              tex: R`|A \cap B| = 0 \;\Rightarrow\; \mathrm{IoU} = 0`,
              why: {
                tr: "Tahmin kutusu gerçek kutudan uzaktaysa kesişim yoktur.",
                en: "If the prediction is away from the true box there is no intersection.",
              },
            },
            {
              tex: R`\frac{\partial\,\mathrm{IoU}}{\partial x_1} = 0, \quad \dots`,
              why: {
                tr: "Kutuyu biraz kaydırmak IoU'yu değiştirmez: 0 kalır. Gradyan sıfırdır, kayıp kutuya hangi yöne gideceğini söyleyemez.",
                en: "Moving the box a little does not change IoU: it stays 0. The gradient is zero, and the loss cannot tell the box which way to go.",
              },
            },
            {
              tex: R`\mathrm{GIoU} = \mathrm{IoU} - \frac{|C| - |A \cup B|}{|C|}`,
              why: {
                tr: "GIoU, iki kutuyu içine alan en küçük kutuyu \\(C\\) ekler: kutular uzaklaştıkça \\(C\\)'nin boş kısmı büyür, GIoU 0'ın altına iner. Artık her konumda bir eğim vardır; değeri −1 ile 1 arasındadır.",
                en: "GIoU adds the smallest box \\(C\\) enclosing both: as the boxes move apart, the empty part of \\(C\\) grows and GIoU drops below 0. Now there is a slope everywhere; its value lies between −1 and 1.",
              },
            },
            {
              tex: R`\mathrm{CIoU} = \mathrm{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v, \qquad v = \frac{4}{\pi^2}\Big(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\Big)^2`,
              why: {
                tr: "CIoU merkezler arasındaki uzaklığı \\(\\rho\\), \\(C\\)'nin köşegeni \\(c\\)'ye bölerek cezalandırır ve en-boy oranı farkını \\(v\\) ile ekler (\\(\\alpha = v / (1 - \\mathrm{IoU} + v)\\)). Ultralytics'in kutu kaybı \\(1 - \\mathrm{CIoU}\\)'dur; YOLO26 buna dört uzaklık üzerinde bir L1 terimi de ekler.",
                en: "CIoU penalizes the distance \\(\\rho\\) between the centres, divided by the diagonal \\(c\\) of \\(C\\), and adds the difference in aspect ratio through \\(v\\) (\\(\\alpha = v / (1 - \\mathrm{IoU} + v)\\)). Ultralytics' box loss is \\(1 - \\mathrm{CIoU}\\); YOLO26 adds an L1 term on the four distances.",
              },
            },
          ],
        },
      },
      { widget: "ious" },
      {
        links: [
          { tab: "theory", section: "y-iou" },
          { tab: "theory", section: "y-train" },
        ],
      },
    ],
  },
  {
    id: "ym-cls",
    title: { tr: "Sınıf skoru: sigmoid ve ikili çapraz entropi", en: "The class score: sigmoid and binary cross-entropy" },
    blocks: [
      {
        p: {
          tr: R`SimpleCNN on rakama [[softmax|softmax]] uyguluyordu: olasılıklar toplamı 1'di, çünkü her görüntü tam olarak bir rakamdır. YOLO'nun bir adayı ise hiçbir nesneye ait olmayabilir (boş zemin) ya da ileride birden fazla sınıfa birden uyabilir. Bu yüzden her sınıfın skoru ayrı bir [[sigmoid|sigmoid]]'dir: \(p_c = \sigma(z_c)\), toplamları 1 olmak zorunda değildir.`,
          en: R`SimpleCNN applied a [[softmax|softmax]] over ten digits: the probabilities summed to 1, because every image is exactly one digit. A YOLO candidate may belong to no object at all (empty ground), or might fit more than one class. So every class score is its own [[sigmoid|sigmoid]]: \(p_c = \sigma(z_c)\), and they need not sum to 1.`,
        },
      },
      {
        derive: {
          title: { tr: "İkili çapraz entropinin gradyanı", en: "The gradient of binary cross-entropy" },
          steps: [
            {
              tex: R`L = -\big[\,y \ln p + (1 - y)\ln(1 - p)\,\big], \qquad p = \sigma(z)`,
              why: { tr: "Her sınıf için ayrı bir evet/hayır kaybı.", en: "A separate yes/no loss for every class." },
            },
            {
              tex: R`\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1 - y}{1 - p} = \frac{p - y}{p(1 - p)}`,
              why: { tr: "Logaritmaların türevi.", en: "The derivatives of the logarithms." },
            },
            {
              tex: R`\frac{\partial L}{\partial z} = \frac{p - y}{p(1 - p)}\cdot p(1 - p) = p - y`,
              why: {
                tr: "Zincir kuralı ve \\(\\sigma' = \\sigma(1 - \\sigma)\\): Bölüm 2'deki sonucun aynısı. Hata, tahmin ile hedefin farkıdır.",
                en: "The chain rule and \\(\\sigma' = \\sigma(1 - \\sigma)\\): the same result as in chapter 2. The error is the difference between prediction and target.",
              },
            },
          ],
        },
      },
      {
        p: {
          tr: R`Hedef \(y\) her zaman 1 değildir. [[label-assignment|Etiket atamasında]] bir nesneye atanan adayın sınıf hedefi, kutusunun ne kadar iyi oturduğuna göre ölçeklenir: nesnenin en iyi adayı, o nesneyle ulaşılan en yüksek IoU'yu hedef alır; ötekiler orantılı olarak daha azını. Atanmayan adayların hedefi 0'dır. Böylece skor "burada bir bina var" ile birlikte "kutum iyi oturuyor" bilgisini de taşır.`,
          en: R`The target \(y\) is not always 1. In [[label-assignment|label assignment]] the class target of a candidate assigned to an object is scaled by how well its box fits: the object's best candidate aims at the highest IoU reached for that object, the others proportionally less. Unassigned candidates aim at 0. So the score carries "there is a building here" together with "my box fits well".`,
        },
      },
      { widget: "bce" },
      {
        links: [
          { tab: "theory", section: "y-train" },
          { chapter: "mlp", tab: "math", section: "mm-loss" },
        ],
      },
    ],
  },
  {
    id: "ym-mask",
    title: { tr: "Maske: prototiplerin doğrusal birleşimi", en: "The mask: a linear combination of prototypes" },
    blocks: [
      {
        p: {
          tr: R`Bir nesnenin maskesi her piksel \((i, j)\) için \(z_{ij} = \sum_{k=1}^{K} c_k\, P_{k,ij}\) ve \(m_{ij} = \sigma(z_{ij})\) ile hesaplanır; kutunun dışındaki pikseller atılır. Eğitimde maske kaybı, kutu içindeki \(A\) pikselin ikili çapraz entropisinin ortalamasıdır (Ultralytics, maskeleri 160 × 160 çözünürlükte karşılaştırır):`,
          en: R`An object's mask is computed for every pixel \((i, j)\) as \(z_{ij} = \sum_{k=1}^{K} c_k\, P_{k,ij}\) and \(m_{ij} = \sigma(z_{ij})\); pixels outside the box are dropped. In training, the mask loss is the mean binary cross-entropy over the \(A\) pixels inside the box (Ultralytics compares masks at 160 × 160):`,
        },
      },
      {
        eq: {
          tr: R`L_{\text{mask}} = \frac{1}{A}\sum_{(i,j)\,\in\,\text{kutu}} -\big[\,y_{ij}\ln m_{ij} + (1 - y_{ij})\ln(1 - m_{ij})\,\big]`,
          en: R`L_{\text{mask}} = \frac{1}{A}\sum_{(i,j)\,\in\,\text{box}} -\big[\,y_{ij}\ln m_{ij} + (1 - y_{ij})\ln(1 - m_{ij})\,\big]`,
        },
      },
      {
        derive: {
          title: { tr: "Kaybın katsayılara göre türevi", en: "The loss's derivative with respect to the coefficients" },
          steps: [
            {
              tex: R`\frac{\partial L_{\text{mask}}}{\partial z_{ij}} = \frac{1}{A}\,(m_{ij} - y_{ij})`,
              why: { tr: "Önceki kısmın sonucu, her piksel için.", en: "The result of the previous section, for every pixel." },
            },
            {
              tex: R`\frac{\partial z_{ij}}{\partial c_k} = P_{k,ij}`,
              why: { tr: "\\(z\\), katsayılarda doğrusaldır.", en: "\\(z\\) is linear in the coefficients." },
            },
            {
              tex: R`\frac{\partial L_{\text{mask}}}{\partial c_k} = \frac{1}{A}\sum_{(i,j)} (m_{ij} - y_{ij})\,P_{k,ij}`,
              why: {
                tr: "Bir katsayının gradyanı, hatanın o prototiple ilişkisidir: prototip hatanın olduğu yerlerde büyükse katsayısı çok değişir. Aynı hata prototiplerin kendisine de geri yayılır; ağ hem prototipleri hem katsayıları birlikte öğrenir.",
                en: "A coefficient's gradient is how the error lines up with that prototype: where the prototype is large where the errors are, its coefficient changes a lot. The same error also flows back into the prototypes; the network learns prototypes and coefficients together.",
              },
            },
          ],
        },
      },
      { widget: "maskfit" },
      { links: [{ tab: "theory", section: "y-masks" }] },
    ],
  },
  {
    id: "ym-nms",
    title: { tr: "NMS: algoritma ve maliyeti", en: "NMS: the algorithm and its cost" },
    blocks: [
      {
        list: [
          {
            tr: R`Skoru eşiğin (\(\tau_{\text{conf}}\), çoğunlukla 0,25) üstündeki \(n\) adayı skora göre sırala: \(O(n \log n)\).`,
            en: R`Sort the \(n\) candidates scoring above the threshold (\(\tau_{\text{conf}}\), mostly 0.25) by score: \(O(n \log n)\).`,
          },
          { tr: R`Listenin başındaki adayı tut.`, en: R`Keep the candidate at the head of the list.` },
          {
            tr: R`Aynı sınıftan olup onunla IoU'su \(\tau_{\text{IoU}}\)'yu aşan adayları sil.`,
            en: R`Remove the candidates of the same class that overlap it by an IoU above \(\tau_{\text{IoU}}\).`,
          },
          {
            tr: R`Liste boşalana kadar 2'ye dön. En kötü durumda her tutulan aday kalanların hepsiyle karşılaştırılır: \(O(n^2)\).`,
            en: R`Go back to 2 until the list is empty. In the worst case every kept candidate is compared with all the rest: \(O(n^2)\).`,
          },
        ],
        ordered: true,
      },
      {
        p: {
          tr: R`\(\tau_{\text{IoU}}\) bir denge kurar: küçük bir eşik birbirine dokunan iki gerçek binayı tek binaya indirebilir, büyük bir eşik aynı binanın kopyalarını bırakır. Şanghay karesinde (güven eşiği 0,25, 107 aday) ölçtük:`,
          en: R`\(\tau_{\text{IoU}}\) strikes a balance: a small threshold can merge two real buildings that touch into one, a large one leaves copies of the same building. Measured on the Shanghai tile (confidence threshold 0.25, 107 candidates):`,
        },
      },
      {
        table: {
          head: [
            R`\(\tau_{\text{IoU}}\)`,
            { tr: "kalan nesne", en: "objects left" },
            { tr: "bina", en: "buildings" },
            { tr: "spor sahası", en: "sports fields" },
          ],
          rows: [
            ["0,3", "14", "10", "4"],
            ["0,5", "14", "10", "4"],
            ["0,7", "15", "11", "4"],
            ["0,9", "19", "15", "4"],
          ].map((r) => r.map((c) => ({ tr: c, en: c.replace(",", ".") }))),
        },
      },
      {
        p: {
          tr: R`NMS'siz tahminin fikri: eğitimde bir nesneye yalnızca tek aday atanırsa (bire bir atama), ağ öteki adaylara düşük skor vermeyi öğrenir; çıkarımda en yüksek skorlu \(k\) aday doğrudan alınır, karşılaştırma gerekmez. YOLO26 bu başı bire-çok başla birlikte eğitir; bizim modelde NMS'siz baş maske mAP50'de 0,47, NMS'li çıktı 0,50 verdi.`,
          en: R`The idea of prediction without NMS: if training assigns a single candidate to each object (one-to-one assignment), the network learns to give the other candidates low scores; at inference the top \(k\) candidates are taken directly, with no comparisons. YOLO26 trains this head together with a one-to-many head; on our model the head without NMS gave a mask mAP50 of 0.47, the output with NMS 0.50.`,
        },
      },
      { links: [{ tab: "theory", section: "y-nms" }, { tab: "detect" }] },
    ],
  },
  {
    id: "ym-ap",
    title: { tr: "Kesinlik, duyarlılık ve AP", en: "Precision, recall and AP" },
    blocks: [
      { eq: R`P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN} = \frac{TP}{n_{\text{gt}}}` },
      {
        p: {
          tr: R`Tespitler skora göre sıralanır ve listede aşağı inildikçe her satırda \(P\) ve \(R\) yeniden hesaplanır: bu, güven eşiğini adım adım düşürmek demektir. \(R\) hiç azalmaz; \(P\) inip çıkabilir. COCO (ve Ultralytics) önce \(P\)'nin <b>zarfını</b> alır, yani her noktada sağındaki en yüksek kesinliği, sonra onu 101 duyarlılık düzeyinde okur:`,
          en: R`The detections are sorted by score and \(P\) and \(R\) are recomputed at every row going down the list: this is lowering the confidence threshold step by step. \(R\) never falls; \(P\) can go up and down. COCO (and Ultralytics) first takes the <b>envelope</b> of \(P\), the highest precision to the right of every point, and then reads it at 101 recall levels:`,
        },
      },
      { eq: R`\hat P(r) = \max_{r' \ge r} P(r'), \qquad \mathrm{AP} = \frac{1}{101}\sum_{k=0}^{100} \hat P\Big(\tfrac{k}{100}\Big)` },
      {
        p: {
          tr: R`Aşağıdaki örnekte 5 gerçek nesne ve 8 tespit var; başlangıçta 4 tespit doğru, en yüksek duyarlılık 0,8. Zarf 0 ile 0,4 arasında 1, 0,6'ya kadar 0,75, 0,8'e kadar 0,67'dir; 0,8'in üstünde tespit olmadığı için 0'dır. AP = (41 · 1 + 20 · 0,75 + 20 · 0,667) / 101 ≈ 0,69.`,
          en: R`The example below has 5 true objects and 8 detections; at the start 4 of them are correct, and the highest recall is 0.8. The envelope is 1 from 0 to 0.4, 0.75 up to 0.6, 0.67 up to 0.8, and 0 above 0.8 because no detection reaches it. AP = (41 · 1 + 20 · 0.75 + 20 · 0.667) / 101 ≈ 0.69.`,
        },
      },
      { widget: "pr" },
      {
        p: {
          tr: R`mAP, sınıfların AP'lerinin ortalamasıdır. mAP50'de doğru sayılmak için IoU ≥ 0,5 yeter; mAP50-95 aynı hesabı \(\tau = 0{,}50, 0{,}55, \dots, 0{,}95\) için tekrarlayıp ortalar. Bizim modelde bina maske AP50 0,69, AP50-95 0,35; spor sahası 0,30 ve 0,15. İki sayı arasındaki büyük fark, sınırların yaklaşık ama tam değil oturduğunu söyler.`,
          en: R`mAP is the mean of the classes' APs. For mAP50 an IoU ≥ 0.5 is enough to count as correct; mAP50-95 repeats the computation for \(\tau = 0.50, 0.55, \dots, 0.95\) and averages. On our model the building mask AP50 is 0.69 and AP50-95 0.35; for sports fields 0.30 and 0.15. The large gap between the two says the outlines fit roughly but not exactly.`,
        },
      },
      { links: [{ tab: "theory", section: "y-metrics" }, { tab: "train" }] },
    ],
  },
  {
    id: "ym-area",
    title: { tr: "Alan ve hatası", en: "Area and its error" },
    blocks: [
      {
        p: {
          tr: R`Bir pikselin zemindeki boyutu <b>GSD</b>'dir (yer örnekleme aralığı): \(g_x\) doğu-batı, \(g_y\) kuzey-güney. Maske model girişinde \(n\) piksel kaplıyorsa ve letterbox kazancı \(\gamma\) ise (650 piksellik karede \(\gamma = 640/650\)), zemindeki alan:`,
          en: R`The ground size of a pixel is the <b>GSD</b> (ground sampling distance): \(g_x\) east-west, \(g_y\) north-south. If a mask covers \(n\) pixels of the model input and the letterbox gain is \(\gamma\) (\(\gamma = 640/650\) for a 650-pixel tile), its area on the ground is:`,
        },
      },
      {
        eq: {
          tr: R`A = \frac{n}{\gamma^2}\, g_x\, g_y, \qquad \text{bir prototip pikseli: } \frac{4}{\gamma}\, g = 4\cdot\frac{650}{640}\cdot 0{,}3\ \mathrm{m} \approx 1{,}22\ \mathrm{m}`,
          en: R`A = \frac{n}{\gamma^2}\, g_x\, g_y, \qquad \text{one prototype pixel: } \frac{4}{\gamma}\, g = 4\cdot\frac{650}{640}\cdot 0.3\ \mathrm{m} \approx 1.22\ \mathrm{m}`,
        },
      },
      {
        derive: {
          title: { tr: "Sınırdaki bir kaymanın alana etkisi", en: "What a shift of the outline does to the area" },
          steps: [
            {
              tex: R`A_\pm = (a \pm 2\delta)^2`,
              why: {
                tr: "Kenarı \\(a\\) olan kare bir bina; sınır her yandan \\(\\delta\\) kadar dışarı (+) ya da içeri (−) kayıyor.",
                en: "A square building of side \\(a\\); the outline moves out (+) or in (−) by \\(\\delta\\) on every side.",
              },
            },
            {
              tex: R`\frac{A_\pm}{a^2} - 1 = \pm\frac{4\delta}{a} + \frac{4\delta^2}{a^2}`,
              why: { tr: "Açın ve \\(a^2\\)'ye bölün.", en: "Expand and divide by \\(a^2\\)." },
            },
            {
              tex: R`\approx \pm\frac{4\delta}{a}`,
              why: {
                tr: "Bağıl hata kenar uzunluğuyla ters orantılıdır: 10 m'lik bir evde \\(\\delta = 1{,}22\\) m kayma alanı −%43 / +%55 değiştirir, 100 m'lik bir depoda yalnızca yaklaşık ±%5.",
                en: "The relative error is inversely proportional to the side: on a 10 m house a shift of \\(\\delta = 1.22\\) m changes the area by −43% / +55%, on a 100 m warehouse only by about ±5%.",
              },
            },
          ],
        },
      },
      { widget: "area" },
      {
        p: {
          tr: R`Maskeyi 160'tan 640'a nasıl büyüttüğümüz de ölçülebilir: doğrulama setindeki 1.571 bina tespitinde, gerçek sınırla ortalama maske IoU'su prototip piksellerini olduğu gibi büyütünce (en yakın komşu) 0,742, logitleri bilinear büyütünce (Ultralytics'in ve bu sayfanın yaptığı) 0,751 çıktı. Kazanç küçüktür; ayrıntının asıl sınırı prototiplerin 160 × 160 çözünürlüğüdür.`,
          en: R`How the mask is enlarged from 160 to 640 can be measured too: over the 1,571 building detections of the validation set, the mean mask IoU with the true outline was 0.742 when the prototype pixels were simply enlarged (nearest neighbour) and 0.751 when the logits were enlarged bilinearly (what Ultralytics and this page do). The gain is small; the real limit to the detail is the 160 × 160 resolution of the prototypes.`,
        },
      },
      { links: [{ tab: "theory", section: "y-read" }, { tab: "detect" }] },
    ],
  },
];

createLesson({ tab: "math", main: "#ym-main", toc: "#ym-toc", sections: MATH, widgets: WIDGETS, kicker: { tr: "Kısım", en: "Section" } });
