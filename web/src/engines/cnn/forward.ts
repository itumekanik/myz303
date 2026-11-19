/**
 * SimpleCNN's forward pass in plain TypeScript, at inference (eval mode: BatchNorm uses its
 * running statistics, Dropout passes everything). One image costs about 4 million
 * multiply-adds, a few milliseconds, so the Inference and Theory tabs need no ML library.
 *
 *   Conv1 5×5 (1→32) → BatchNorm → ReLU → MaxPool 2×2      28×28 → 24×24 → 12×12
 *   Conv2 5×5 (32→64) → BatchNorm → ReLU → MaxPool 2×2     12×12 → 8×8 → 4×4
 *   Flatten (64·4·4 = 1024, PyTorch order: channel, row, column) → FC1 512 → ReLU → FC2 10
 *
 * The weights use PyTorch's names and layouts (conv: [out, in, k, k], linear: [out, in]), as
 * exported by tools/export_assets.py, so the numbers match PyTorch's (tests/unit/cnn-forward).
 */

export const PARAM_NAMES = [
  "conv1.weight",
  "conv1.bias",
  "conv2.weight",
  "conv2.bias",
  "batch1.weight",
  "batch1.bias",
  "batch1.running_mean",
  "batch1.running_var",
  "batch2.weight",
  "batch2.bias",
  "batch2.running_mean",
  "batch2.running_var",
  "fc1.weight",
  "fc1.bias",
  "fc2.weight",
  "fc2.bias",
] as const;
export type ParamName = (typeof PARAM_NAMES)[number];
export type CnnWeights = Record<ParamName, Float32Array>;

export const BN_EPS = 1e-5;
export const DROPOUT = [0.45, 0.35] as const;

export interface ForwardResult {
  /** After Conv1 → BatchNorm → ReLU: 32 maps of 24×24. */
  conv1: Float32Array;
  /** After the first max pooling: 32 × 12×12. */
  pool1: Float32Array;
  /** After Conv2 → BatchNorm → ReLU: 64 × 8×8. */
  conv2: Float32Array;
  /** After the second max pooling: 64 × 4×4 (flattened as it is for FC1). */
  pool2: Float32Array;
  /** After FC1 → ReLU: 512 values. */
  fc1: Float32Array;
  logits: Float32Array;
  probabilities: Float32Array;
}

/** "Valid" 2-D cross-correlation (what nn.Conv2d computes), every output channel, plus bias. */
function conv(x: Float32Array, cIn: number, size: number, w: Float32Array, b: Float32Array, cOut: number, k: number): Float32Array {
  const o = size - k + 1;
  const out = new Float32Array(cOut * o * o);
  for (let co = 0; co < cOut; co++) {
    const plane = co * o * o;
    out.fill(b[co], plane, plane + o * o);
    for (let ci = 0; ci < cIn; ci++) {
      const xBase = ci * size * size;
      const wBase = (co * cIn + ci) * k * k;
      for (let m = 0; m < k; m++) {
        for (let n = 0; n < k; n++) {
          const wv = w[wBase + m * k + n];
          for (let r = 0; r < o; r++) {
            const xRow = xBase + (r + m) * size + n;
            const outRow = plane + r * o;
            for (let c = 0; c < o; c++) out[outRow + c] += wv * x[xRow + c];
          }
        }
      }
    }
  }
  return out;
}

/** BatchNorm with the running statistics, then ReLU, in place. */
function batchNormRelu(
  x: Float32Array,
  channels: number,
  gamma: Float32Array,
  beta: Float32Array,
  mean: Float32Array,
  variance: Float32Array,
): void {
  const per = x.length / channels;
  for (let c = 0; c < channels; c++) {
    const scale = gamma[c] / Math.sqrt(variance[c] + BN_EPS);
    const shift = beta[c] - mean[c] * scale;
    for (let i = c * per; i < (c + 1) * per; i++) {
      const v = x[i] * scale + shift;
      x[i] = v > 0 ? v : 0;
    }
  }
}

/** 2×2 max pooling with stride 2. */
function maxPool(x: Float32Array, channels: number, size: number): Float32Array {
  const o = size >> 1;
  const out = new Float32Array(channels * o * o);
  for (let c = 0; c < channels; c++) {
    for (let r = 0; r < o; r++) {
      for (let q = 0; q < o; q++) {
        const i = c * size * size + 2 * r * size + 2 * q;
        out[c * o * o + r * o + q] = Math.max(x[i], x[i + 1], x[i + size], x[i + size + 1]);
      }
    }
  }
  return out;
}

/** y = W x + b for W of shape [out, in]. */
function linear(x: Float32Array, w: Float32Array, b: Float32Array): Float32Array {
  const nOut = b.length;
  const nIn = x.length;
  const out = new Float32Array(nOut);
  for (let j = 0; j < nOut; j++) {
    let s = b[j];
    const row = j * nIn;
    for (let i = 0; i < nIn; i++) s += w[row + i] * x[i];
    out[j] = s;
  }
  return out;
}

export function softmax(logits: Float32Array): Float32Array {
  const peak = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - peak));
  const sum = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / sum);
}

/** Runs SimpleCNN on one 28×28 image with values in [0, 1]. */
export function forward(w: CnnWeights, image: Float32Array): ForwardResult {
  const conv1 = conv(image, 1, 28, w["conv1.weight"], w["conv1.bias"], 32, 5);
  batchNormRelu(conv1, 32, w["batch1.weight"], w["batch1.bias"], w["batch1.running_mean"], w["batch1.running_var"]);
  const pool1 = maxPool(conv1, 32, 24);
  const conv2 = conv(pool1, 32, 12, w["conv2.weight"], w["conv2.bias"], 64, 5);
  batchNormRelu(conv2, 64, w["batch2.weight"], w["batch2.bias"], w["batch2.running_mean"], w["batch2.running_var"]);
  const pool2 = maxPool(conv2, 64, 8);
  const fc1 = linear(pool2, w["fc1.weight"], w["fc1.bias"]);
  for (let i = 0; i < fc1.length; i++) if (fc1[i] < 0) fc1[i] = 0;
  const logits = linear(fc1, w["fc2.weight"], w["fc2.bias"]);
  return { conv1, pool1, conv2, pool2, fc1, logits, probabilities: softmax(logits) };
}
