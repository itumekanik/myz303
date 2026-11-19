/** SimpleCNN weights as exported by tools/export_assets.py (PyTorch names, shapes and layouts). */
import { float32, loadData } from "./data.ts";
import { PARAM_NAMES, type CnnWeights } from "./forward.ts";

export interface PackedTensor {
  shape: number[];
  data: string;
}

/** What the data files cnn-trained.js and cnn-initial.js contain. */
export interface PackedModel {
  meta?: Record<string, unknown>;
  seed?: number;
  tensors: Record<string, PackedTensor>;
}

const SHAPES: Record<string, number[]> = {
  "conv1.weight": [32, 1, 5, 5],
  "conv2.weight": [64, 32, 5, 5],
  "fc1.weight": [512, 1024],
  "fc2.weight": [10, 512],
};

/** Decodes the tensors and checks that every parameter is there with the expected size. */
export function unpackWeights(model: PackedModel): CnnWeights {
  const out = {} as CnnWeights;
  for (const name of PARAM_NAMES) {
    const t = model.tensors[name];
    if (!t) throw new Error(`weights: ${name} is missing`);
    const values = float32(t.data);
    const expected = t.shape.reduce((a, b) => a * b, 1);
    if (values.length !== expected) throw new Error(`weights: ${name} has ${values.length} values, its shape says ${expected}`);
    if (SHAPES[name] && SHAPES[name].join() !== t.shape.join()) throw new Error(`weights: ${name} has shape ${t.shape}`);
    out[name] = values;
  }
  return out;
}

export async function loadModel(name: "cnn-trained" | "cnn-initial"): Promise<{ weights: CnnWeights; packed: PackedModel }> {
  const packed = await loadData<PackedModel>(name);
  return { weights: unpackWeights(packed), packed };
}

/** Total number of parameters (582,218 for SimpleCNN). */
export const parameterCount = (w: CnnWeights): number =>
  PARAM_NAMES.filter((n) => !n.includes("running")).reduce((s, n) => s + w[n].length, 0);
