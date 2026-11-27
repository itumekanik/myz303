/**
 * Running the two YOLO26n models with ONNX Runtime Web (WebAssembly, on the CPU).
 *
 * Everything arrives as data files (a page opened from a file may run scripts but not fetch):
 * data/ort-wasm.js holds ONNX Runtime's WebAssembly binary (written by scripts/build.mjs from
 * node_modules), data/yolo-seg.js and data/yolo-sem.js the models (tools/export_yolo_assets.py).
 * One thread: more would need cross-origin isolation, which neither GitHub Pages nor file:// has.
 */
import * as ort from "onnxruntime-web/wasm";
import { loadData, unpack, type Packed } from "../data.ts";
import { detect, type Detection, type Head } from "./detect.ts";
import { letterbox, toChw, toImageBox, type Letterbox, type Rgb } from "./image.ts";
import { instanceMask, type InstanceMask, type Prototypes } from "./masks.ts";

export interface YoloModel {
  session: ort.InferenceSession;
  names: string[];
  size: number;
}

/** What data/yolo-seg.js and data/yolo-sem.js contain. */
export interface PackedYolo {
  names: string[];
  size: number;
  onnx: Packed;
}

let runtime: Promise<void> | undefined;

/** Hands ONNX Runtime its WebAssembly binary, once. */
export function initRuntime(wasm: () => Promise<Uint8Array>): Promise<void> {
  runtime ??= wasm().then((binary) => {
    ort.env.wasm.wasmBinary = binary;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
  });
  return runtime;
}

export async function createModel(onnx: Uint8Array, names: string[], size: number): Promise<YoloModel> {
  if (!runtime) throw new Error("initRuntime() first");
  await runtime;
  const session = await ort.InferenceSession.create(onnx, { executionProviders: ["wasm"] });
  return { session, names: [...names], size };
}

const models = new Map<string, Promise<YoloModel>>();

/** In the page: the instance ("seg") or land-cover ("sem") model, loaded once. */
export function loadModel(task: "seg" | "sem"): Promise<YoloModel> {
  let model = models.get(task);
  if (!model) {
    void initRuntime(async () => unpack(await loadData<Packed>("ort-wasm")));
    model = loadData<PackedYolo>(`yolo-${task}`).then(async (p) => createModel(await unpack(p.onnx), p.names, p.size));
    model.catch(() => models.delete(task)); // allow a retry
    models.set(task, model);
  }
  return model;
}

const now = () => performance.now();

/** Letterbox and the input tensor for an image. */
function prepare(model: YoloModel, image: Rgb): { input: ort.Tensor; box: Letterbox; ms: number } {
  const t = now();
  const { image: square, box } = letterbox(image, model.size);
  const input = new ort.Tensor("float32", toChw(square), [1, 3, model.size, model.size]);
  return { input, box, ms: now() - t };
}

export interface InstanceObject extends Detection {
  mask: InstanceMask;
  /** the box in image pixels */
  imageBox: [number, number, number, number];
}

export interface InstanceResult {
  letterbox: Letterbox;
  objects: InstanceObject[];
  /** the raw outputs, for the lesson's step-by-step pictures */
  head: Head;
  protos: Prototypes;
  ms: { prepare: number; network: number; post: number };
}

/** Buildings and sports fields: boxes, scores and masks. */
export async function runInstances(model: YoloModel, image: Rgb, options: { conf?: number; iou?: number } = {}): Promise<InstanceResult> {
  const { input, box, ms: prepareMs } = prepare(model, image);
  let t = now();
  const out = await model.session.run({ [model.session.inputNames[0]]: input });
  const network = now() - t;
  t = now();
  const [h, p] = model.session.outputNames.map((n) => out[n]);
  const head: Head = { data: h.data as Float32Array, rows: h.dims[1], count: h.dims[2] };
  const protos: Prototypes = { data: p.data as Float32Array, k: p.dims[1], height: p.dims[2], width: p.dims[3] };
  const objects = detect(head, model.names.length, options).map((d) => ({
    ...d,
    mask: instanceMask(d.coefficients, protos, d.box, model.size),
    imageBox: toImageBox(d.box, box, image.width, image.height),
  }));
  return { letterbox: box, objects, head, protos, ms: { prepare: prepareMs, network, post: now() - t } };
}

export interface LandCoverResult {
  letterbox: Letterbox;
  /** class of every model-input pixel (size × size) */
  classes: Uint8Array;
  /** pixels per class inside the image (the grey letterbox border is not counted) */
  counts: number[];
  ms: { prepare: number; network: number };
}

/** Background, building, road, green, water, parking: one class per pixel. */
export async function runLandCover(model: YoloModel, image: Rgb): Promise<LandCoverResult> {
  const { input, box, ms: prepareMs } = prepare(model, image);
  const t = now();
  const out = await model.session.run({ [model.session.inputNames[0]]: input });
  const network = now() - t;
  const classes = out[model.session.outputNames[0]].data as Uint8Array;
  const counts = new Array<number>(model.names.length).fill(0);
  for (let y = box.top; y < box.top + box.height; y++) {
    for (let x = box.left; x < box.left + box.width; x++) counts[classes[y * model.size + x]]++;
  }
  return { letterbox: box, classes, counts, ms: { prepare: prepareMs, network } };
}
