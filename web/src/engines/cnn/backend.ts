/**
 * The SimpleCNN dashboard's server API, answered in the browser.
 *
 * The CNN page (chapters/cnn/app.js, theory.js) was written for dashboard.py: it calls
 * /api/status, /api/train, /api/stop, /api/predict, /api/test-samples and /api/theory/params.
 * localApi() takes the same paths and bodies and returns the same JSON, from:
 * - the trained weights and their run (data/cnn-trained.js, cnn-run.js: trained with PyTorch),
 *   or the last run trained in this browser (saved in IndexedDB, store.ts);
 * - the plain TypeScript forward pass (forward.ts) for predictions;
 * - the MNIST test set (data/mnist-test.js) for test images;
 * - a training session on TensorFlow.js (session.ts) for /api/train.
 * Errors are thrown with dashboard.py's (Turkish) messages, which the page translates.
 */
import { loadData, mnist, type ImageSet, type Packed } from "./data.ts";
import { ApiError } from "./errors.ts";
import { preprocessDrawing } from "./drawing.ts";
import { BN_EPS, DROPOUT, forward, type CnnWeights } from "./forward.ts";
import { TrainingSession, DEFAULTS, parseConfig, deviceName, type RunRecord } from "./session.ts";
import { clearSaved, loadSaved } from "./store.ts";
import { loadModel, unpackWeights, type PackedModel } from "./weights.ts";

const SERVER_VERSION = 5; // dashboard.py's SERVER_VERSION: the same API
const MODELS = ["trained", "initial"] as const;
type Which = (typeof MODELS)[number];

export { ApiError };

interface ModelSlot {
  weights: CnnWeights | null;
  source: string | null;
  meta: Record<string, unknown> | null;
  version: number;
}

const engine = {
  trained: { weights: null, source: null, meta: null, version: 0 } as ModelSlot,
  initial: null as { weights: CnnWeights; seed: number } | null,
  initialLoading: null as Promise<void> | null,
  test: null as Promise<ImageSet> | null,
};

/** Called by the training session: a new model (after an epoch, or at the end). */
export function loadState(weights: CnnWeights, source: string, meta: Record<string, unknown> | null = null): void {
  Object.assign(engine.trained, { weights, source, meta, version: engine.trained.version + 1 });
}

/** The weights a training run starts from (the "before training" model). */
export function setInitial(weights: CnnWeights, seed: number): void {
  engine.initial = { weights, seed };
  engine.initialLoading = Promise.resolve();
}

const relabel = (source: string): void => {
  engine.trained.source = source;
};

const session = new TrainingSession({ loadState, setInitial, relabel });

let started: Promise<void> | null = null;
/** Set while /api/reset replaces the model, so a run cannot start in the middle of it. */
let resetting: Promise<void> | null = null;

/** Loads the saved model and its run: the browser's own last run if there is one, else the PyTorch run. */
function start(): Promise<void> {
  started ??= (async () => {
    const saved = await loadSaved().catch(() => null);
    // A record from an older version, or a damaged one, must not lock the reader out of the page
    if (
      saved?.weights?.["fc2.weight"] instanceof Float32Array &&
      saved.initial?.["fc2.weight"] &&
      saved.meta &&
      Array.isArray(saved.run?.log)
    ) {
      loadState(saved.weights, "Kayıtlı model", saved.meta);
      setInitial(saved.initial, Number(saved.meta.seed ?? DEFAULTS.seed));
      session.restore(saved.run, "this browser");
      return;
    }
    await loadBundled();
  })().catch((error: unknown) => {
    started = null; // e.g. a data file failed to load: the next request tries again
    throw error;
  });
  return started;
}

/** The model that comes with the site (trained with PyTorch) and its run. */
async function loadBundled(): Promise<void> {
  const [{ weights, packed }, run] = await Promise.all([loadModel("cnn-trained"), loadData<RunRecord>("cnn-run")]);
  loadState(weights, "Kayıtlı model", packed.meta ?? null);
  engine.initial = null; // cnn-initial.js: that run's starting weights, loaded when asked for
  engine.initialLoading = null;
  session.restore(run, "PyTorch");
}

async function initialModel(): Promise<{ weights: CnnWeights; seed: number }> {
  engine.initialLoading ??= loadData<PackedModel>("cnn-initial").then((p) => {
    if (!engine.initial) engine.initial = { weights: unpackWeights(p), seed: Number(p.seed ?? DEFAULTS.seed) };
  });
  await engine.initialLoading;
  return engine.initial!;
}

function testSet(): Promise<ImageSet> {
  engine.test ??= loadData<{ images: Packed; labels: Packed }>("mnist-test").then((d) => mnist(d.images, d.labels));
  return engine.test;
}

async function pick(which: Which): Promise<CnnWeights> {
  if (which === "initial") return (await initialModel()).weights;
  if (!engine.trained.weights) throw new ApiError("Henüz eğitilmiş bir model yok.");
  return engine.trained.weights;
}

const round = (v: number, digits: number) => Math.round(v * 10 ** digits) / 10 ** digits;

/** 0-1 values → 0-255 pixels. */
const toUint8 = (values: Float32Array) => Array.from(values, (v) => Math.round(Math.min(1, Math.max(0, v)) * 255));

/** A non-negative activation map scaled to 0-255 (0 stays 0, the peak becomes 255). */
function activationToUint8(values: Float32Array): number[] {
  let peak = 0;
  for (const v of values) if (v > peak) peak = v;
  if (peak <= 1e-8) return new Array(values.length).fill(0);
  return Array.from(values, (v) => Math.min(255, Math.max(0, Math.round((v / peak) * 255))));
}

function predict(weights: CnnWeights, image: Float32Array, which: Which, label: number | null) {
  const r = forward(weights, image);
  const shapes = { conv1: [32, 24, 24], pool1: [32, 12, 12], conv2: [64, 8, 8], pool2: [64, 4, 4] } as const;
  const layers = (Object.keys(shapes) as (keyof typeof shapes)[]).map((key) => {
    const [c, h, w] = shapes[key];
    const maps = [];
    const peaks = [];
    for (let k = 0; k < c; k++) {
      const map = r[key].subarray(k * h * w, (k + 1) * h * w);
      maps.push(activationToUint8(map));
      peaks.push(round(Math.max(...map), 3));
    }
    return { key, shape: [c, h, w], maps, peaks };
  });
  let best = 0;
  r.probabilities.forEach((p, k) => {
    if (p > r.probabilities[best]) best = k;
  });
  return {
    prediction: best,
    probabilities: Array.from(r.probabilities, (p) => round(p, 5)),
    logits: Array.from(r.logits, (v) => round(v, 3)),
    input: toUint8(image),
    layers,
    fc1: { values: activationToUint8(r.fc1), active: r.fc1.filter((v) => v > 0).length },
    model: which,
    label,
  };
}

function theoryParams(weights: CnnWeights, which: Which) {
  const values = (a: Float32Array) => Array.from(a, (v) => round(v, 6));
  const conv = weights["conv1.weight"];
  return {
    ready: true,
    model: which,
    version: engine.trained.version,
    conv1: {
      weight: Array.from({ length: 32 }, (_, k) => values(conv.subarray(k * 25, (k + 1) * 25))),
      bias: values(weights["conv1.bias"]),
    },
    bn1: {
      gamma: values(weights["batch1.weight"]),
      beta: values(weights["batch1.bias"]),
      mean: values(weights["batch1.running_mean"]),
      var: values(weights["batch1.running_var"]),
      eps: BN_EPS,
    },
    dropout: [...DROPOUT],
  };
}

function whichOf(value: unknown): Which {
  const which = (value ?? "trained") as Which;
  if (!MODELS.includes(which)) throw new ApiError("Geçersiz model.");
  return which;
}

/** Answers a request for one of dashboard.py's API paths (body undefined: GET, else POST). */
export async function localApi(path: string, body?: Record<string, unknown>): Promise<unknown> {
  await start();
  const url = new URL(path, "http://local");
  const route = url.pathname;
  if (body === undefined) {
    if (route === "/api/status") {
      return {
        ...session.snapshot(),
        model: {
          ready: Boolean(engine.trained.weights),
          source: engine.trained.source,
          meta: engine.trained.meta,
          version: engine.trained.version,
          initial_seed: engine.initial?.seed ?? DEFAULTS.seed,
        },
        defaults: DEFAULTS,
        device: deviceName(),
        server_version: SERVER_VERSION,
      };
    }
    if (route === "/api/theory/params") {
      const which = whichOf(url.searchParams.get("model"));
      if (which === "trained" && !engine.trained.weights) return { ready: false };
      return theoryParams(await pick(which), which);
    }
    if (route === "/api/test-samples") {
      const count = Math.max(1, Math.min(Number(url.searchParams.get("n") ?? 12) || 12, 48));
      const set = await testSet();
      const chosen = new Set<number>();
      while (chosen.size < count) chosen.add(Math.floor(Math.random() * set.count));
      return {
        samples: [...chosen].map((index) => ({
          index,
          label: set.labels[index],
          pixels: Array.from(set.pixels.subarray(index * 784, (index + 1) * 784)),
        })),
      };
    }
    throw new ApiError("Bulunamadı.");
  }
  if (route === "/api/train") {
    const config = parseConfig(body); // throws ApiError with the server's message
    await resetting?.catch(() => undefined);
    if (!session.start(config)) throw new ApiError("Eğitim zaten sürüyor.");
    return { ok: true };
  }
  if (route === "/api/reset") {
    // Not in dashboard.py: forget this browser's model and go back to the one that comes with the site
    if (session.snapshot().running) throw new ApiError("Eğitim zaten sürüyor.");
    resetting ??= (async () => {
      await clearSaved().catch(() => undefined);
      await loadBundled();
    })().finally(() => {
      resetting = null;
    });
    await resetting;
    return { ok: true };
  }
  if (route === "/api/stop") {
    session.stop();
    return { ok: true };
  }
  if (route === "/api/predict") {
    const which = whichOf(body.model);
    let image: Float32Array | null;
    let label: number | null = null;
    if ("image" in body) {
      image = await preprocessDrawing(String(body.image));
      if (!image) throw new ApiError("Önce bir rakam çizin.");
    } else if ("test_index" in body) {
      const set = await testSet();
      const index = Number(body.test_index);
      if (!Number.isInteger(index) || index < 0 || index >= set.count) throw new ApiError("Geçersiz test görüntüsü.");
      image = Float32Array.from(set.pixels.subarray(index * 784, (index + 1) * 784), (v) => v / 255);
      label = set.labels[index];
    } else if ("pixels" in body) {
      const pixels = body.pixels;
      if (!Array.isArray(pixels) || pixels.length !== 784) throw new ApiError("28×28 = 784 piksel bekleniyor.");
      image = Float32Array.from(pixels as number[], (v) => Math.min(255, Math.max(0, Number(v))) / 255);
    } else throw new ApiError("Görüntü gönderilmedi.");
    return predict(await pick(which), image, which, label);
  }
  throw new ApiError("Bulunamadı.");
}
