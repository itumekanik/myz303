/**
 * A SimpleCNN training run in the browser, on TensorFlow.js: the counterpart of dashboard.py's
 * TrainingSession. It goes through the same stages (loading → training ⇄ validating → testing →
 * saving → done), records the same fields (batch losses, epoch history, Conv1 filters, predictions
 * on 16 fixed validation images, a log) and snapshot() returns them as /api/status does.
 *
 * Same as SimpleCNN.py: the validation set (the same 12,000 images, data/mnist-train.js), the
 * initial weights' distribution (PyTorch's default, ±1/√fan_in), Adam (β 0.9 / 0.999, ε 1e-8),
 * cross-entropy, BatchNorm (ε 1e-5, momentum 0.1), Dropout 0.45 / 0.35. Not the same: the random
 * numbers (another generator), so a run here does not repeat PyTorch's run digit for digit.
 */
import { ApiError } from "./errors.ts";
import { loadData, mnist, unpack, type ImageSet, type Packed } from "./data.ts";
import { DROPOUT, PARAM_NAMES, type CnnWeights, type ParamName } from "./forward.ts";
import { saveModel } from "./store.ts";
import { backendLost, loadTf, type Tf } from "./tf.ts";
import type * as TF from "@tensorflow/tfjs";

export const DEFAULTS = { epochs: 10, lr: 1e-3, batch_size: 256, seed: 42 };
const NUM_SAMPLES = 16; // validation images whose predictions are shown while training
const FILTER_EVERY = 10; // refresh the Conv1 filter snapshot every N batches
const LOG_LIMIT = 300;
const EVAL_BATCH = 1000;

export interface Config {
  epochs: number;
  lr: number;
  batch_size: number;
  seed: number;
}

/** A finished run, as dashboard.py saves it (simple_cnn_run.json) and the Training tab shows it. */
export interface RunRecord {
  trained_at: string;
  config: Config;
  epoch: number;
  batch: number;
  batches: number;
  step: number;
  dataset_sizes: { train: number; validation: number; test: number } | null;
  batch_losses: number[];
  history: Record<string, number>[];
  test: { loss: number; acc: number } | null;
  filters: number[][] | null;
  samples: { images: number[][]; labels: number[] } | null;
  sample_predictions: { epoch: number; predictions: number[]; confidence: number[]; correct: number } | null;
  log: string[];
  started_at: number | null;
  finished_at: number | null;
}

const RUN_FIELDS = [
  "config",
  "epoch",
  "batch",
  "batches",
  "step",
  "dataset_sizes",
  "batch_losses",
  "history",
  "test",
  "filters",
  "samples",
  "sample_predictions",
  "log",
  "started_at",
  "finished_at",
] as const;

/** Validates the training settings, with dashboard.py's messages. */
export function parseConfig(body: Record<string, unknown>): Config {
  const number = (key: keyof Config, integer: boolean, low: number, high: number): number => {
    const raw = body[key] ?? DEFAULTS[key];
    let value = Number(raw);
    if (raw === "" || raw === null || !Number.isFinite(value)) throw new ApiError(`'${key}' geçerli bir sayı değil.`);
    if (integer) value = Math.trunc(value);
    if (value < low || value > high) throw new ApiError(`'${key}' ${low} ile ${high} arasında olmalı.`);
    return value;
  };
  return {
    epochs: number("epochs", true, 1, 50),
    lr: number("lr", false, 1e-5, 1.0),
    batch_size: number("batch_size", true, 8, 4096),
    seed: number("seed", true, 0, 2 ** 31 - 1),
  };
}

/**
 * The GPU's name from WebGL's renderer string:
 * "ANGLE (NVIDIA, NVIDIA GeForce RTX 2060 (0x00001F15) Direct3D11 vs_5_0 ps_5_0, D3D11)" → "NVIDIA GeForce RTX 2060",
 * "ANGLE (Intel, Intel(R) UHD Graphics (0x00009BC4) Direct3D11 …)" → "Intel(R) UHD Graphics".
 */
export function gpuFromRenderer(renderer: string): string {
  const inner = renderer.match(/^ANGLE \((.*)\)$/)?.[1];
  let name = inner ? (inner.split(", ")[1] ?? inner) : renderer;
  name = name.match(/Renderer:\s*(.+)$/)?.[1] ?? name; // "ANGLE Metal Renderer: Apple M2" (macOS)
  return name
    .replace(/\s*\(0x[0-9a-f]+\).*$/i, "")
    .replace(/\s+(Direct3D\d*|OpenGL|Vulkan)\b.*$/i, "")
    .trim();
}

let gpuName: string | null | undefined;
let tfBackend: string | null = null;
/** "GPU · <name> (WebGPU)" or "CPU", like dashboard.py's device_name(). */
export function deviceName(): string {
  if (gpuName === undefined) {
    gpuName = null;
    try {
      const gl = document.createElement("canvas").getContext("webgl2") ?? document.createElement("canvas").getContext("webgl");
      if (gl) {
        const ext = gl.getExtension("WEBGL_debug_renderer_info");
        const renderer = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
        gpuName = gpuFromRenderer(renderer);
        if (/SwiftShader|llvmpipe|software/i.test(renderer)) gpuName = null; // no real GPU
      }
    } catch {
      gpuName = null;
    }
  }
  if (tfBackend === "cpu" || !gpuName) return "CPU";
  const api = tfBackend === "webgpu" ? " (WebGPU)" : tfBackend === "webgl" ? " (WebGL)" : "";
  return `GPU · ${gpuName}${api}`;
}

/** mulberry32: a small seeded random generator (the run's seed fixes the weights and the shuffling). */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** PyTorch's default initialisation: weights and biases uniform in ±1/√fan_in; BatchNorm γ 1, β 0. */
function initialWeights(rand: () => number): CnnWeights {
  const uniform = (n: number, fanIn: number) => {
    const bound = 1 / Math.sqrt(fanIn);
    return Float32Array.from({ length: n }, () => (2 * rand() - 1) * bound);
  };
  return {
    "conv1.weight": uniform(32 * 25, 25),
    "conv1.bias": uniform(32, 25),
    "conv2.weight": uniform(64 * 32 * 25, 32 * 25),
    "conv2.bias": uniform(64, 32 * 25),
    "batch1.weight": new Float32Array(32).fill(1),
    "batch1.bias": new Float32Array(32),
    "batch1.running_mean": new Float32Array(32),
    "batch1.running_var": new Float32Array(32).fill(1),
    "batch2.weight": new Float32Array(64).fill(1),
    "batch2.bias": new Float32Array(64),
    "batch2.running_mean": new Float32Array(64),
    "batch2.running_var": new Float32Array(64).fill(1),
    "fc1.weight": uniform(512 * 1024, 1024),
    "fc1.bias": uniform(512, 1024),
    "fc2.weight": uniform(10 * 512, 512),
    "fc2.bias": uniform(10, 512),
  };
}

/** SimpleCNN as a TensorFlow.js model (NHWC; flattening order handled when converting weights). */
function buildModel(tf: Tf): TF.Sequential {
  const L = tf.layers;
  return tf.sequential({
    layers: [
      L.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 5, name: "conv1" }),
      L.batchNormalization({ axis: -1, momentum: 0.9, epsilon: 1e-5, name: "batch1" }),
      L.reLU(),
      L.maxPooling2d({ poolSize: 2, strides: 2 }),
      L.dropout({ rate: DROPOUT[0] }),
      L.conv2d({ filters: 64, kernelSize: 5, name: "conv2" }),
      L.batchNormalization({ axis: -1, momentum: 0.9, epsilon: 1e-5, name: "batch2" }),
      L.reLU(),
      L.maxPooling2d({ poolSize: 2, strides: 2 }),
      L.dropout({ rate: DROPOUT[1] }),
      L.flatten(),
      L.dense({ units: 512, activation: "relu", name: "fc1" }),
      L.dense({ units: 10, name: "fc2" }),
    ],
  });
}

/** PyTorch layout → the model (conv [o,i,k,k] → [k,k,i,o]; FC1's inputs (c,h,w) → (h,w,c)). */
function setWeights(tf: Tf, model: TF.Sequential, w: CnnWeights): void {
  tf.tidy(() => {
    const t = (name: ParamName, shape: number[]) => tf.tensor(w[name], shape);
    const bn = (k: string) =>
      (["weight", "bias", "running_mean", "running_var"] as const).map((p) =>
        t(`${k}.${p}` as ParamName, [w[`${k}.weight` as ParamName].length]),
      );
    model.getLayer("conv1").setWeights([t("conv1.weight", [32, 1, 5, 5]).transpose([2, 3, 1, 0]), t("conv1.bias", [32])]);
    model.getLayer("conv2").setWeights([t("conv2.weight", [64, 32, 5, 5]).transpose([2, 3, 1, 0]), t("conv2.bias", [64])]);
    model.getLayer("batch1").setWeights(bn("batch1"));
    model.getLayer("batch2").setWeights(bn("batch2"));
    model.getLayer("fc1").setWeights([t("fc1.weight", [512, 64, 4, 4]).transpose([2, 3, 1, 0]).reshape([1024, 512]), t("fc1.bias", [512])]);
    model.getLayer("fc2").setWeights([t("fc2.weight", [10, 512]).transpose(), t("fc2.bias", [10])]);
  });
}

/** The model → PyTorch layout (the inverse of setWeights), for the forward pass, the page and saving. */
async function getWeights(tf: Tf, model: TF.Sequential): Promise<CnnWeights> {
  // Reorder on the GPU (tidy keeps only the returned tensors), then read them back asynchronously
  const tensors = tf.tidy(() => {
    const [c1w, c1b] = model.getLayer("conv1").getWeights();
    const [c2w, c2b] = model.getLayer("conv2").getWeights();
    const b1 = model.getLayer("batch1").getWeights();
    const b2 = model.getLayer("batch2").getWeights();
    const [f1w, f1b] = model.getLayer("fc1").getWeights();
    const [f2w, f2b] = model.getLayer("fc2").getWeights();
    const t: Partial<Record<ParamName, TF.Tensor>> = {
      "conv1.weight": c1w.transpose([3, 2, 0, 1]),
      "conv1.bias": c1b.clone(),
      "conv2.weight": c2w.transpose([3, 2, 0, 1]),
      "conv2.bias": c2b.clone(),
      "fc1.weight": f1w.reshape([4, 4, 64, 512]).transpose([3, 2, 0, 1]),
      "fc1.bias": f1b.clone(),
      "fc2.weight": f2w.transpose(),
      "fc2.bias": f2b.clone(),
    };
    (["weight", "bias", "running_mean", "running_var"] as const).forEach((p, i) => {
      t[`batch1.${p}`] = b1[i].clone();
      t[`batch2.${p}`] = b2[i].clone();
    });
    return t as Record<ParamName, TF.Tensor>;
  });
  const out = {} as CnnWeights;
  for (const name of PARAM_NAMES) out[name] = Float32Array.from((await tensors[name].data()) as Float32Array);
  tf.dispose(Object.values(tensors));
  return out;
}

/** The 32 Conv1 kernels (5×5), each scaled to [-1, 1] for display (dashboard.py's conv1_filters). */
function conv1Filters(w: Float32Array): number[][] {
  return Array.from({ length: 32 }, (_, k) => {
    const f = w.subarray(k * 25, (k + 1) * 25);
    const scale = Math.max(1e-8, ...Array.from(f, Math.abs));
    return Array.from(f, (v) => Math.round((v / scale) * 1000) / 1000);
  });
}

class StopTraining extends Error {}

/** The error TensorFlow.js's backends give when the GPU device or context is gone. */
const GPU_LOST = /device.*\blost\b|context.*\blost\b|DXGI_ERROR/i;

/**
 * Lets the page run between evaluation batches. Not tf.nextFrame() (requestAnimationFrame): that
 * stops in a hidden tab, and so would the run. Message events are not throttled there either.
 */
function breathe(): Promise<void> {
  return new Promise((resolve) => {
    const { port1, port2 } = new MessageChannel();
    port1.onmessage = () => {
      port1.close();
      resolve();
    };
    port2.postMessage(null);
  });
}

const now = () => Date.now() / 1000;
const r = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;
const pad = (n: number) => String(n).padStart(2, "0");
const clock = () => {
  const d = new Date();
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};
/** Local time as YYYY-MM-DDTHH:MM:SS, like Python's datetime.now().isoformat(timespec="seconds"). */
const isoNow = () => {
  const d = new Date();
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};

export interface EngineLink {
  loadState(weights: CnnWeights, source: string, meta?: Record<string, unknown> | null): void;
  setInitial(weights: CnnWeights, seed: number): void;
  /** Renames the current model (a run ended early and left the model of its last epoch). */
  relabel(source: string): void;
}

type Fields = Omit<RunRecord, "trained_at"> & {
  state: "idle" | "running" | "done" | "stopped" | "error";
  stage: string | null;
  message: string;
  restored: { trained_at: string; origin?: string } | null;
};

export class TrainingSession {
  runId = 0;
  private f!: Fields;
  private stopRequested = false;
  private running = false;

  private engine: EngineLink;

  constructor(engine: EngineLink) {
    this.engine = engine;
    this.reset(null);
  }

  private reset(config: Config | null): void {
    this.f = {
      config: config as Config,
      state: "idle",
      stage: null,
      message: "",
      epoch: 0,
      batch: 0,
      batches: 0,
      step: 0,
      dataset_sizes: null,
      batch_losses: [],
      history: [],
      test: null,
      filters: null,
      samples: null,
      sample_predictions: null,
      log: [],
      started_at: null,
      finished_at: null,
      restored: null,
    };
  }

  private log(text: string): void {
    this.f.log.push(`[${clock()}] ${text}`);
    if (this.f.log.length > LOG_LIMIT) this.f.log.splice(0, this.f.log.length - LOG_LIMIT);
  }

  /** Shows a finished run (the PyTorch run that came with the site, or this browser's last one). */
  restore(run: RunRecord, origin: "PyTorch" | "this browser"): void {
    this.reset(run.config);
    for (const key of RUN_FIELDS) (this.f as Record<string, unknown>)[key] = run[key];
    this.f.log = [...run.log];
    this.runId += 1; // a new id: the page redraws its charts and tables for this run
    this.f.state = "done";
    this.f.stage = "done";
    this.f.restored = { trained_at: run.trained_at, origin };
    this.log(
      origin === "PyTorch"
        ? `Loaded the run that trained the bundled model with PyTorch (${run.trained_at})`
        : `Loaded the last run trained in this browser (${run.trained_at})`,
    );
  }

  snapshot() {
    const f = this.f;
    const end = f.finished_at ?? now();
    return {
      run_id: this.runId,
      state: f.state,
      stage: f.stage,
      running: this.running,
      message: f.message,
      config: f.config,
      epoch: f.epoch,
      batch: f.batch,
      batches: f.batches,
      step: f.step,
      dataset_sizes: f.dataset_sizes,
      elapsed: f.started_at ? r(end - f.started_at, 1) : null,
      batch_losses: [...f.batch_losses],
      history: [...f.history],
      test: f.test,
      filters: f.filters,
      samples: f.samples,
      sample_predictions: f.sample_predictions,
      log: [...f.log],
      restored: f.restored,
    };
  }

  start(config: Config): boolean {
    if (this.running) return false;
    this.runId += 1;
    this.reset(config);
    Object.assign(this.f, { state: "running", stage: "loading", started_at: now() });
    this.stopRequested = false;
    this.running = true;
    void this.run(config).finally(() => {
      this.running = false;
      this.f.finished_at = now();
    });
    return true;
  }

  stop(): void {
    this.stopRequested = true;
  }

  private async run(config: Config): Promise<void> {
    let model: TF.Sequential | null = null;
    let optimizer: TF.Optimizer | null = null;
    let tf: Tf | null = null;
    let backend: string | null = null;
    /** A run that ended early leaves the model of its last finished epoch. */
    const unfinished = () => {
      if (this.f.history.length) this.engine.relabel(`Epoch ${this.f.history.length}/${config.epochs} sonrası (eğitim tamamlanmadı)`);
    };
    try {
      const rand = seeded(config.seed);
      const initial = initialWeights(rand);
      const loaded = await loadTf();
      tf = loaded.tf;
      backend = loaded.backend;
      tfBackend = loaded.backend;
      this.log(`Device: ${deviceName()} · TensorFlow.js ${tf.version.tfjs}, ${loaded.backend} backend`);
      if (loaded.backend === "cpu") this.log("No GPU backend is available: training on the CPU will be very slow.");
      this.log("Loading MNIST...");
      const [train, test] = await Promise.all([
        loadData<{ images: Packed; labels: Packed; validation: Packed }>("mnist-train").then(async (d) => ({
          set: await mnist(d.images, d.labels),
          validation: await unpack(d.validation),
        })),
        loadData<{ images: Packed; labels: Packed }>("mnist-test").then((d) => mnist(d.images, d.labels)),
      ]);
      const valIdx = Array.from(new Uint16Array(train.validation.buffer, train.validation.byteOffset, train.validation.byteLength / 2));
      const isVal = new Uint8Array(train.set.count);
      for (const i of valIdx) isVal[i] = 1;
      const trainIdx = Array.from({ length: train.set.count }, (_, i) => i).filter((i) => !isVal[i]);
      const testIdx = Array.from({ length: test.count }, (_, i) => i);
      const batches = Math.ceil(trainIdx.length / config.batch_size);
      const sampleIdx = valIdx.slice(0, NUM_SAMPLES);
      Object.assign(this.f, {
        batches,
        dataset_sizes: { train: trainIdx.length, validation: valIdx.length, test: test.count },
        filters: conv1Filters(initial["conv1.weight"]),
        samples: {
          images: sampleIdx.map((i) => Array.from(train.set.pixels.subarray(i * 784, (i + 1) * 784))),
          labels: sampleIdx.map((i) => train.set.labels[i]),
        },
      });
      this.log(`Data: ${trainIdx.length} train / ${valIdx.length} validation / ${test.count} test images, ${batches} batches per epoch`);

      model = buildModel(tf);
      setWeights(tf, model, initial);
      optimizer = tf.train.adam(config.lr, 0.9, 0.999, 1e-8);
      model.compile({
        optimizer,
        loss: (yt: TF.Tensor, yp: TF.Tensor) => tf!.losses.softmaxCrossEntropy(yt, yp),
        metrics: ["categoricalAccuracy"],
      });
      const batchOf = (set: ImageSet, idx: number[]) => {
        const x = new Float32Array(idx.length * 784);
        idx.forEach((i, k) => {
          for (let p = 0; p < 784; p++) x[k * 784 + p] = set.pixels[i * 784 + p] / 255;
        });
        return {
          x: tf!.tensor4d(x, [idx.length, 28, 28, 1]),
          y: tf!.tidy(() =>
            tf!
              .oneHot(
                tf!.tensor1d(
                  idx.map((i) => set.labels[i]),
                  "int32",
                ),
                10,
              )
              .toFloat(),
          ),
        };
      };
      const evaluate = async (set: ImageSet, idx: number[]) => {
        let loss = 0,
          correct = 0;
        for (let s = 0; s < idx.length; s += EVAL_BATCH) {
          if (this.stopRequested) throw new StopTraining();
          const part = idx.slice(s, s + EVAL_BATCH);
          const { x, y } = batchOf(set, part);
          const [l, c] = tf!.tidy(() => {
            const logits = model!.predict(x) as TF.Tensor;
            return [tf!.losses.softmaxCrossEntropy(y, logits).mul(part.length), logits.argMax(-1).equal(y.argMax(-1)).sum()];
          });
          loss += (await l.data())[0];
          correct += (await c.data())[0];
          tf!.dispose([x, y, l, c]);
          await breathe();
        }
        return { acc: (100 * correct) / idx.length, loss: loss / idx.length };
      };
      const predictSamples = async (epoch: number) => {
        const { x, y } = batchOf(train.set, sampleIdx);
        const probs = tf!.tidy(() => tf!.softmax(model!.predict(x) as TF.Tensor));
        const p = (await probs.array()) as number[][];
        tf!.dispose([x, y, probs]);
        const predictions = p.map((row) => row.indexOf(Math.max(...row)));
        this.f.sample_predictions = {
          epoch,
          predictions,
          confidence: p.map((row) => r(Math.max(...row), 4)),
          correct: predictions.filter((v, k) => v === this.f.samples!.labels[k]).length,
        };
      };
      await predictSamples(0);

      for (let epoch = 1; epoch <= config.epochs; epoch++) {
        Object.assign(this.f, { stage: "training", epoch, batch: 0 });
        // shuffle (Fisher-Yates with the run's generator)
        for (let i = trainIdx.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          [trainIdx[i], trainIdx[j]] = [trainIdx[j], trainIdx[i]];
        }
        let lossSum = 0,
          correct = 0;
        for (let b = 0; b < batches; b++) {
          if (this.stopRequested) throw new StopTraining();
          const idx = trainIdx.slice(b * config.batch_size, (b + 1) * config.batch_size);
          const { x, y } = batchOf(train.set, idx);
          const [loss, acc] = (await model.trainOnBatch(x, y)) as number[];
          tf.dispose([x, y]);
          lossSum += loss;
          correct += acc * idx.length;
          this.f.batch = b + 1;
          this.f.step += 1;
          this.f.batch_losses.push(r(loss, 5));
          if ((b + 1) % FILTER_EVERY === 0 || b + 1 === batches) {
            const kernel = tf.tidy(() => model!.getLayer("conv1").getWeights()[0].transpose([3, 2, 0, 1]));
            this.f.filters = conv1Filters((await kernel.data()) as Float32Array);
            kernel.dispose();
          }
        }
        const trainLoss = lossSum / batches,
          trainAcc = (100 * correct) / trainIdx.length;
        this.f.stage = "validating";
        const val = await evaluate(train.set, valIdx);
        this.f.history.push({
          epoch,
          train_loss: r(trainLoss, 5),
          train_acc: r(trainAcc, 3),
          val_loss: r(val.loss, 5),
          val_acc: r(val.acc, 3),
          time: r(now() - this.f.started_at!, 1),
        });
        this.log(
          `Epoch ${epoch}: Train Loss: ${trainLoss.toFixed(4)}, Train Accuracy: ${trainAcc.toFixed(2)}% | Validation Loss: ${val.loss.toFixed(4)}, Validation Accuracy: ${val.acc.toFixed(2)}%`,
        );
        await predictSamples(epoch);
        // From the first finished epoch on, "before training" means this run's starting point
        if (epoch === 1) this.engine.setInitial(initial, config.seed);
        this.engine.loadState(await getWeights(tf, model), `Epoch ${epoch}/${config.epochs} sonrası (eğitim sürüyor)`);
      }

      this.f.stage = "testing";
      const t = await evaluate(test, testIdx);
      this.f.test = { loss: r(t.loss, 5), acc: r(t.acc, 3) };
      this.log(`Test Loss: ${t.loss.toFixed(4)}, Test Accuracy: ${t.acc.toFixed(2)}%`);
      if (this.stopRequested) throw new StopTraining();

      this.f.stage = "saving";
      const weights = await getWeights(tf, model);
      const meta = {
        ...config,
        test_acc: r(t.acc, 3),
        test_loss: r(t.loss, 5),
        val_acc: this.f.history.at(-1)!.val_acc,
        device: deviceName(),
        trained_at: isoNow(),
      };
      this.engine.loadState(weights, "Yeni eğitilen model", meta);
      this.f.finished_at = now();
      const run = { trained_at: meta.trained_at, ...Object.fromEntries(RUN_FIELDS.map((k) => [k, this.f[k]])) } as RunRecord;
      try {
        await saveModel({ weights, initial, meta, run });
        this.log("Model saved in this browser (IndexedDB); it is loaded again next time");
      } catch (e) {
        this.log(`Could not save the model in this browser: ${(e as Error).message}`);
      }
      Object.assign(this.f, { stage: "done", state: "done", finished_at: now() });
    } catch (e) {
      unfinished();
      if (e instanceof StopTraining) {
        this.log("Training stopped by the user.");
        Object.assign(this.f, { state: "stopped", message: "Eğitim durduruldu." });
      } else {
        const err = e as Error;
        console.warn(err);
        this.log(`Error: ${err.name}: ${err.message}`);
        const gpuLost = GPU_LOST.test(`${err.name} ${err.message}`);
        if (gpuLost && backend) backendLost(backend); // the next run falls back to the next backend
        Object.assign(this.f, { state: "error", message: gpuLost ? "GPU yanıt vermeyi bıraktı." : `${err.name}: ${err.message}` });
      }
    } finally {
      try {
        optimizer?.dispose(); // compile() does not take ownership of an optimizer instance
        model?.dispose();
      } catch (e) {
        console.warn(e); // a lost GPU device cannot free its buffers
      }
    }
  }
}
