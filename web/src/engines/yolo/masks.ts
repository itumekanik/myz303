/**
 * Instance masks as Ultralytics builds them (ops.process_mask with upsample=True):
 *
 *   logits = coefficients · prototypes          32 maps of 160 × 160 → one 160 × 160 map
 *   enlarge the map to 640 × 640 (bilinear, PyTorch's F.interpolate with align_corners=False)
 *   mask   = logits > 0  (that is, sigmoid > 0.5), kept only inside the detection box
 *
 * Only the pixels inside the box are computed: the same result, a fraction of the work.
 */
import type { Box } from "./detect.ts";

export interface Prototypes {
  /** [1, k, h, w] */
  data: Float32Array;
  k: number;
  height: number;
  width: number;
}

/** A mask in model-input pixels, stored only over its box. */
export interface InstanceMask {
  x: number;
  y: number;
  width: number;
  height: number;
  /** 1 inside the object, row after row */
  data: Uint8Array;
  /** number of 1s */
  area: number;
}

/** The mask's logit map at prototype resolution (for the lesson's "linear combination" picture). */
export function maskLogits(coefficients: Float32Array, protos: Prototypes): Float32Array {
  const { data: P, k, width, height } = protos;
  const n = width * height;
  const out = new Float32Array(n);
  for (let c = 0; c < k; c++) {
    const w = coefficients[c];
    const plane = c * n;
    for (let i = 0; i < n; i++) out[i] += w * P[plane + i];
  }
  return out;
}

/** Bilinear source positions of PyTorch's F.interpolate(align_corners=False), float32 like PyTorch on the CPU. */
function sampling(outSize: number, inSize: number, from: number, to: number) {
  const scale = Math.fround(inSize / outSize);
  const n = Math.max(0, to - from);
  const i0 = new Int32Array(n);
  const i1 = new Int32Array(n);
  const l1 = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let src = Math.fround(Math.fround(scale * Math.fround(from + k + 0.5)) - 0.5);
    if (src < 0) src = 0;
    const a = Math.floor(src);
    i0[k] = a;
    i1[k] = a < inSize - 1 ? a + 1 : a;
    l1[k] = Math.fround(src - a);
  }
  return { i0, i1, l1 };
}

/** The mask of one detection; `box` in model-input pixels. */
export function instanceMask(coefficients: Float32Array, protos: Prototypes, box: Box, size = 640): InstanceMask {
  // crop_mask keeps the columns x1 <= c < x2 and the rows y1 <= r < y2
  const x0 = Math.min(Math.max(Math.ceil(box[0]), 0), size);
  const x1 = Math.min(Math.max(Math.ceil(box[2]), 0), size);
  const y0 = Math.min(Math.max(Math.ceil(box[1]), 0), size);
  const y1 = Math.min(Math.max(Math.ceil(box[3]), 0), size);
  const width = Math.max(0, x1 - x0);
  const height = Math.max(0, y1 - y0);
  const data = new Uint8Array(width * height);
  if (!width || !height) return { x: x0, y: y0, width, height, data, area: 0 };

  const X = sampling(size, protos.width, x0, x1);
  const Y = sampling(size, protos.height, y0, y1);
  // logits only on the prototype cells the box touches
  const px0 = X.i0[0];
  const px1 = X.i1[width - 1];
  const py0 = Y.i0[0];
  const py1 = Y.i1[height - 1];
  const lw = px1 - px0 + 1;
  const lh = py1 - py0 + 1;
  const L = new Float32Array(lw * lh);
  const { data: P, k, width: pw, height: ph } = protos;
  for (let c = 0; c < k; c++) {
    const w = coefficients[c];
    const plane = c * pw * ph;
    for (let y = 0; y < lh; y++) {
      const src = plane + (py0 + y) * pw + px0;
      const dst = y * lw;
      for (let x = 0; x < lw; x++) L[dst + x] += w * P[src + x];
    }
  }
  let area = 0;
  for (let y = 0; y < height; y++) {
    const r0 = (Y.i0[y] - py0) * lw;
    const r1 = (Y.i1[y] - py0) * lw;
    const b = Y.l1[y];
    const a = 1 - b;
    for (let x = 0; x < width; x++) {
      const c0 = X.i0[x] - px0;
      const c1 = X.i1[x] - px0;
      const d = X.l1[x];
      const v = a * ((1 - d) * L[r0 + c0] + d * L[r0 + c1]) + b * ((1 - d) * L[r1 + c0] + d * L[r1 + c1]);
      if (v > 0) {
        data[y * width + x] = 1;
        area++;
      }
    }
  }
  return { x: x0, y: y0, width, height, data, area };
}

/** Pixels inside the letterboxed image that a mask covers, converted to square metres of ground. */
export function areaSquareMetres(pixels: number, gain: number, metresPerPixel: readonly [number, number]): number {
  return (pixels / (gain * gain)) * metresPerPixel[0] * metresPerPixel[1];
}
