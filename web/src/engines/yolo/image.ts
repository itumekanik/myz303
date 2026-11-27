/**
 * Preparing an image for YOLO exactly as Ultralytics does in Python, so the page finds the same objects:
 * letterbox (keep the aspect ratio, pad with grey 114 to 640 × 640), OpenCV's bilinear resize, RGB in 0…1.
 *
 * A canvas could resize too, but browsers smooth differently: in Chromium and Firefox half of the pixels
 * came out different from OpenCV (up to 20 grey levels), enough to change the building count. This code
 * matches cv2.resize(..., INTER_LINEAR) byte for byte (tests/unit/yolo-image.test.ts).
 */

/** An 8-bit RGB image, row after row, 3 bytes per pixel. */
export interface Rgb {
  width: number;
  height: number;
  data: Uint8Array;
}

/** Python's round(): halves go to the even neighbour (round(2.5) = 2), unlike Math.round. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r) + 0; // + 0 turns -0 into 0
}

const COEF_BITS = 11; // OpenCV's INTER_RESIZE_COEF_BITS
const ONE = 1 << COEF_BITS;

/**
 * Source index and the two fixed-point weights for every output position (half-pixel centres, float32 like
 * OpenCV). At the edges OpenCV treats the two directions differently, and the result shows it: columns
 * past the edge get the weights 1 and 0; rows past the edge keep their weights and read the edge row twice.
 */
function linearWeights(dst: number, src: number, clampWeights: boolean) {
  const scale = src / dst;
  const index = new Int32Array(dst);
  const w0 = new Int32Array(dst);
  const w1 = new Int32Array(dst);
  for (let d = 0; d < dst; d++) {
    let f = Math.fround((d + 0.5) * scale - 0.5);
    let s = Math.floor(f);
    f = Math.fround(f - s);
    if (clampWeights && s < 0) {
      s = 0;
      f = 0;
    }
    if (clampWeights && s >= src - 1) {
      s = src - 1;
      f = 0;
    }
    index[d] = s;
    w0[d] = roundHalfEven(Math.fround(1 - f) * ONE);
    w1[d] = roundHalfEven(f * ONE);
  }
  return { index, w0, w1 };
}

/** OpenCV turns INTER_LINEAR into a 2 × 2 average when the image shrinks to exactly half. */
function halve(image: Rgb, width: number, height: number): Rgb {
  const { width: sw, data: src } = image;
  const out = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = (2 * y * sw + 2 * x) * 3;
      const b = a + sw * 3;
      for (let c = 0; c < 3; c++) out[(y * width + x) * 3 + c] = (src[a + c] + src[a + 3 + c] + src[b + c] + src[b + 3 + c] + 2) >> 2;
    }
  }
  return { width, height, data: out };
}

/** cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) for 8-bit RGB. */
export function resizeLinear(image: Rgb, width: number, height: number): Rgb {
  const { width: sw, height: sh, data: src } = image;
  if (sw === 2 * width && sh === 2 * height) return halve(image, width, height);
  const X = linearWeights(width, sw, true);
  const Y = linearWeights(height, sh, false);
  // horizontal pass of one source row, values × 2048; two rows are needed at a time
  const rows = new Map<number, Int32Array>();
  const row = (y: number): Int32Array => {
    let r = rows.get(y);
    if (r) return r;
    r = new Int32Array(width * 3);
    for (let x = 0; x < width; x++) {
      const s0 = (y * sw + X.index[x]) * 3;
      const s1 = X.index[x] + 1 < sw ? s0 + 3 : s0;
      for (let c = 0; c < 3; c++) r[3 * x + c] = src[s0 + c] * X.w0[x] + src[s1 + c] * X.w1[x];
    }
    rows.set(y, r);
    return r;
  };
  const out = new Uint8Array(width * height * 3);
  const clampRow = (y: number) => Math.min(Math.max(y, 0), sh - 1);
  for (let y = 0; y < height; y++) {
    const y0 = clampRow(Y.index[y]);
    const r0 = row(y0);
    const r1 = row(clampRow(Y.index[y] + 1));
    const b0 = Y.w0[y];
    const b1 = Y.w1[y];
    const o = y * width * 3;
    for (let i = 0; i < width * 3; i++) {
      // OpenCV's vertical pass: (((b0 * (S0 >> 4)) >> 16) + ((b1 * (S1 >> 4)) >> 16) + 2) >> 2
      const v = (((b0 * (r0[i] >> 4)) >> 16) + ((b1 * (r1[i] >> 4)) >> 16) + 2) >> 2;
      out[o + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    for (const key of rows.keys()) if (key < y0) rows.delete(key);
  }
  return { width, height, data: out };
}

/** Where the image went inside the square model input. */
export interface Letterbox {
  size: number;
  /** model pixels per image pixel */
  gain: number;
  /** size of the resized image inside the square */
  width: number;
  height: number;
  /** grey rows above and columns left of it */
  top: number;
  left: number;
}

export const PAD_GREY = 114;

/** Ultralytics' LetterBox(new_shape=size, auto=False): the numbers only. */
export function letterboxParams(width: number, height: number, size = 640): Letterbox {
  const gain = Math.min(size / height, size / width);
  const w = Math.max(roundHalfEven(width * gain), 1);
  const h = Math.max(roundHalfEven(height * gain), 1);
  return { size, gain, width: w, height: h, top: roundHalfEven((size - h) / 2 - 0.1), left: roundHalfEven((size - w) / 2 - 0.1) };
}

/** The square model input: the image resized into the middle, grey around it. */
export function letterbox(image: Rgb, size = 640): { image: Rgb; box: Letterbox } {
  const box = letterboxParams(image.width, image.height, size);
  const inner = box.width === image.width && box.height === image.height ? image : resizeLinear(image, box.width, box.height);
  const data = new Uint8Array(size * size * 3).fill(PAD_GREY);
  for (let y = 0; y < box.height; y++) {
    data.set(inner.data.subarray(y * box.width * 3, (y + 1) * box.width * 3), ((y + box.top) * size + box.left) * 3);
  }
  return { image: { width: size, height: size, data }, box };
}

/** Model coordinates back to image coordinates, clipped to the image (Ultralytics' scale_boxes). */
export function toImageBox(b: readonly number[], lb: Letterbox, width: number, height: number): [number, number, number, number] {
  const clip = (v: number, max: number) => Math.min(Math.max(v, 0), max);
  return [
    clip((b[0] - lb.left) / lb.gain, width),
    clip((b[1] - lb.top) / lb.gain, height),
    clip((b[2] - lb.left) / lb.gain, width),
    clip((b[3] - lb.top) / lb.gain, height),
  ];
}

/** The network input: 1 × 3 × H × W floats, channel after channel, values 0…1. */
export function toChw(image: Rgb): Float32Array {
  const n = image.width * image.height;
  const out = new Float32Array(3 * n);
  const d = image.data;
  for (let i = 0; i < n; i++) {
    out[i] = d[3 * i] / 255;
    out[n + i] = d[3 * i + 1] / 255;
    out[2 * n + i] = d[3 * i + 2] / 255;
  }
  return out;
}
