/**
 * Measuring detections against the true objects: matching at an IoU threshold, precision, recall and
 * average precision computed the way Ultralytics does (COCO: the precision envelope read at 101 recall levels).
 */

/** What one prediction turned out to be after matching. */
export interface Match {
  /** index of the matched true object, or -1 (a false positive) */
  truth: number;
  iou: number;
}

/**
 * Greedy matching, highest score first: each prediction takes the free true object of the same class it
 * overlaps most, if that overlap reaches `threshold`. `iou(p, t)` measures prediction p against truth t.
 */
export function matchPredictions(
  predictions: readonly { cls: number; score: number }[],
  truths: readonly { cls: number }[],
  iou: (p: number, t: number) => number,
  threshold = 0.5,
): { matches: Match[]; missed: number[] } {
  const order = predictions.map((_, i) => i).sort((a, b) => predictions[b].score - predictions[a].score);
  const taken = new Uint8Array(truths.length);
  const matches: Match[] = predictions.map(() => ({ truth: -1, iou: 0 }));
  for (const p of order) {
    let best = -1;
    let bestIou = threshold;
    truths.forEach((t, k) => {
      if (taken[k] || t.cls !== predictions[p].cls) return;
      const v = iou(p, k);
      if (v >= bestIou) {
        bestIou = v;
        best = k;
      }
    });
    if (best >= 0) {
      taken[best] = 1;
      matches[p] = { truth: best, iou: bestIou };
    }
  }
  const missed = truths.map((_, k) => k).filter((k) => !taken[k]);
  return { matches, missed };
}

/** Precision and recall after each prediction, highest score first. */
export function precisionRecall(
  scores: readonly number[],
  correct: readonly boolean[],
  truths: number,
): { precision: number[]; recall: number[] } {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const precision: number[] = [];
  const recall: number[] = [];
  let tp = 0;
  order.forEach((i, n) => {
    if (correct[i]) tp++;
    precision.push(tp / (n + 1));
    recall.push(truths ? tp / truths : 0);
  });
  return { precision, recall };
}

/** Ultralytics' compute_ap: the precision envelope, read at recall 0, 0.01, …, 1 and averaged. */
export function averagePrecision(recall: readonly number[], precision: readonly number[]): number {
  const n = recall.length;
  const mpre = [1, ...precision, 0, 0];
  for (let i = mpre.length - 2; i >= 0; i--) mpre[i] = Math.max(mpre[i], mpre[i + 1]);
  let sum = 0;
  for (let k = 0; k <= 100; k++) {
    const x = k * 0.01; // numpy.linspace(0, 1, 101) gives k * 0.01 (0.35000000000000003, not 0.35)
    let lo = 0; // first recall >= x (numpy.searchsorted, side="left")
    let hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (recall[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    sum += mpre[1 + lo];
  }
  return sum / 101;
}

/**
 * IoU and the losses built on it, as Ultralytics' bbox_iou computes them (boxes as x1, y1, x2, y2):
 * GIoU subtracts the empty part of the smallest enclosing box, DIoU the centre distance over that box's
 * diagonal, CIoU also an aspect-ratio term. The box loss is 1 − CIoU.
 */
export function iouVariants(
  a: readonly number[],
  b: readonly number[],
  eps = 1e-7,
): { iou: number; giou: number; diou: number; ciou: number } {
  const w1 = a[2] - a[0];
  const h1 = a[3] - a[1] + eps;
  const w2 = b[2] - b[0];
  const h2 = b[3] - b[1] + eps;
  const inter = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = w1 * h1 + w2 * h2 - inter + eps;
  const iou = inter / union;
  const cw = Math.max(a[2], b[2]) - Math.min(a[0], b[0]);
  const ch = Math.max(a[3], b[3]) - Math.min(a[1], b[1]);
  const c2 = cw * cw + ch * ch + eps;
  const rho2 = ((b[0] + b[2] - a[0] - a[2]) ** 2 + (b[1] + b[3] - a[1] - a[3]) ** 2) / 4;
  const v = (4 / Math.PI ** 2) * (Math.atan(w2 / h2) - Math.atan(w1 / h1)) ** 2;
  const alpha = v / (1 - iou + v + eps);
  const cArea = cw * ch + eps;
  return { iou, giou: iou - (cArea - union) / cArea, diou: iou - rho2 / c2, ciou: iou - (rho2 / c2 + v * alpha) };
}
