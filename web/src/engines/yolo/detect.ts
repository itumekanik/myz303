/**
 * From the network's raw output to a list of objects, as Ultralytics' non_max_suppression does it.
 *
 * For a 640 × 640 input the head gives 8400 candidates: one per cell of three grids, 80 × 80 (stride 8,
 * small objects), 40 × 40 (stride 16) and 20 × 20 (stride 32, large objects). Each candidate is one
 * column of 4 + classes + 32 numbers: box centre x, y, width, height (in input pixels), one score per
 * class (already through a sigmoid) and 32 mask coefficients.
 */

export type Box = [number, number, number, number]; // x1, y1, x2, y2

export interface Candidate {
  /** column of the head output, 0…8399 */
  index: number;
  cls: number;
  score: number;
  box: Box;
}

export interface Head {
  /** [1, 4 + classes + coefficients, candidates] */
  data: Float32Array;
  rows: number;
  count: number;
}

export const STRIDES = [8, 16, 32];

/** Which grid cell a candidate belongs to: level 0 is the finest grid. */
export function anchorOf(index: number, size = 640): { level: number; stride: number; grid: number; gx: number; gy: number } {
  let start = 0;
  for (let level = 0; level < STRIDES.length; level++) {
    const grid = size / STRIDES[level];
    if (index < start + grid * grid) {
      const k = index - start;
      return { level, stride: STRIDES[level], grid, gx: k % grid, gy: Math.floor(k / grid) };
    }
    start += grid * grid;
  }
  throw new RangeError(`candidate ${index} is outside the ${start} grid cells`);
}

/** Number of candidates for a square input: (s/8)² + (s/16)² + (s/32)². */
export const candidateCount = (size = 640): number => STRIDES.reduce((n, s) => n + (size / s) ** 2, 0);

export function boxIou(p: readonly number[], q: readonly number[]): number {
  const w = Math.max(0, Math.min(p[2], q[2]) - Math.max(p[0], q[0]));
  const h = Math.max(0, Math.min(p[3], q[3]) - Math.max(p[1], q[1]));
  const inter = w * h;
  const union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** One candidate: its best class and the box in corner form. */
export function candidate(head: Head, classes: number, index: number): Candidate {
  const { data: d, count: n } = head;
  let score = -Infinity;
  let cls = 0;
  for (let c = 0; c < classes; c++) {
    const s = d[(4 + c) * n + index];
    if (s > score) {
      score = s;
      cls = c;
    }
  }
  const cx = d[index];
  const cy = d[n + index];
  const w = d[2 * n + index];
  const h = d[3 * n + index];
  return { index, cls, score, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] };
}

/** Every candidate whose best class score is above the confidence threshold, highest score first. */
export function aboveConfidence(head: Head, classes: number, conf: number): Candidate[] {
  const { data: d, count: n } = head;
  const out: Candidate[] = [];
  for (let j = 0; j < n; j++) {
    let best = -Infinity;
    for (let c = 0; c < classes; c++) best = Math.max(best, d[(4 + c) * n + j]);
    if (best > conf) out.push(candidate(head, classes, j));
  }
  return out.sort((a, b) => b.score - a.score);
}

/**
 * Non-maximum suppression per class: walk from the highest score down and drop a box when it overlaps an
 * already kept box of the same class by more than `iou`.
 */
export function nms<T extends Candidate>(sorted: readonly T[], iou: number, maxDetections = 300): T[] {
  const kept: T[] = [];
  for (const c of sorted) {
    if (kept.length === maxDetections) break;
    if (kept.every((k) => k.cls !== c.cls || boxIou(k.box, c.box) <= iou)) kept.push(c);
  }
  return kept;
}

/** The 32 mask coefficients of one candidate. */
export function coefficients(head: Head, classes: number, index: number): Float32Array {
  const nm = head.rows - 4 - classes;
  const out = new Float32Array(nm);
  for (let k = 0; k < nm; k++) out[k] = head.data[(4 + classes + k) * head.count + index];
  return out;
}

export interface Detection extends Candidate {
  coefficients: Float32Array;
}

/** Confidence filter, then NMS, then the mask coefficients of what is left. */
export function detect(head: Head, classes: number, { conf = 0.25, iou = 0.7 } = {}): Detection[] {
  return nms(aboveConfidence(head, classes, conf), iou).map((c) => ({ ...c, coefficients: coefficients(head, classes, c.index) }));
}
