import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";
import { fromBase64 } from "../../src/engines/cnn/data.ts";
import { forward } from "../../src/engines/cnn/forward.ts";
import { parameterCount, unpackWeights, type PackedModel } from "../../src/engines/cnn/weights.ts";

const here = dirname(fileURLToPath(import.meta.url));

/** Runs a data file the way a page does (a script that fills window.__AI_DATA__). */
function dataFile<T>(name: string): T {
  const window: { __AI_DATA__?: Record<string, unknown> } = {};
  runInNewContext(readFileSync(join(here, `../../public/data/${name}.js`), "utf8"), { window });
  return window.__AI_DATA__![name] as T;
}

interface Fixture {
  images: string;
  logits: number[][];
  sums: Record<"conv1" | "pool1" | "conv2" | "pool2" | "fc1", number[]>;
}

describe("SimpleCNN forward pass in TypeScript (engines/cnn/forward.ts)", () => {
  const weights = unpackWeights(dataFile<PackedModel>("cnn-trained"));
  const fixture = JSON.parse(readFileSync(join(here, "../fixtures/cnn-parity.json"), "utf8")) as Fixture;
  const pixels = fromBase64(fixture.images);
  const image = (k: number) => Float32Array.from(pixels.subarray(k * 784, (k + 1) * 784), (v) => v / 255);

  it("has SimpleCNN's 582,218 parameters", () => {
    assert.equal(parameterCount(weights), 582218);
  });

  it("gives PyTorch's logits on the test images", () => {
    fixture.logits.forEach((expected, k) => {
      const { logits } = forward(weights, image(k));
      const largest = Math.max(...expected.map((v, j) => Math.abs(v - logits[j])));
      assert.ok(largest < 1e-3, `image ${k}: largest difference ${largest}`);
    });
  });

  it("gives PyTorch's output at every stage (sums per image)", () => {
    fixture.logits.forEach((_, k) => {
      const result = forward(weights, image(k));
      for (const stage of ["conv1", "pool1", "conv2", "pool2", "fc1"] as const) {
        const sum = result[stage].reduce((s, v) => s + v, 0);
        const expected = fixture.sums[stage][k];
        assert.ok(Math.abs(sum - expected) < 1e-3 * Math.max(1, Math.abs(expected)), `image ${k}, ${stage}: ${sum} vs ${expected}`);
      }
    });
  });

  it("probabilities sum to 1 and the shapes are right", () => {
    const r = forward(weights, image(0));
    assert.ok(Math.abs(r.probabilities.reduce((s, v) => s + v, 0) - 1) < 1e-6);
    assert.deepEqual(
      [r.conv1.length, r.pool1.length, r.conv2.length, r.pool2.length, r.fc1.length, r.logits.length],
      [32 * 24 * 24, 32 * 12 * 12, 64 * 8 * 8, 64 * 4 * 4, 512, 10],
    );
  });

  it("rejects incomplete weights", () => {
    assert.throws(() => unpackWeights({ tensors: {} }), /conv1\.weight is missing/);
  });
});
