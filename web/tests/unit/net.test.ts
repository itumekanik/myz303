import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { NET } from "../../src/engines/net.js";

/** A seeded random generator (mulberry32), so every run sees the same weights. */
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

const XOR = [
  { x: [0, 0], y: 0 },
  { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 },
  { x: [1, 1], y: 0 },
];

describe("NET (the MLP engine of chapter 2)", () => {
  it("counts parameters as Σ n_l (n_{l-1} + 1)", () => {
    assert.equal(NET.paramCount([2, 2, 1]), 9);
    assert.equal(NET.paramCount([2, 3, 1]), 13);
    assert.equal(NET.paramCount([784, 512, 10]), 784 * 512 + 512 + 512 * 10 + 10);
  });

  for (const act of ["sigmoid", "tanh"] as const) {
    it(`backpropagation matches the numerical gradient (${act})`, () => {
      const net = NET.create([2, 3, 1], { act, out: "sigmoid", rand: seeded(7) });
      const data = [
        { x: [0.3, -0.8], y: 1 },
        { x: [-1.2, 0.4], y: 0 },
      ];
      const g = NET.gradient(net, data);
      const n = NET.numericGradient(net, data);
      let largest = 0;
      g.forEach((layer, l) => {
        layer.W.forEach((row, j) => row.forEach((v, i) => (largest = Math.max(largest, Math.abs(v - n[l].W[j][i])))));
        layer.b.forEach((v, j) => (largest = Math.max(largest, Math.abs(v - n[l].b[j]))));
      });
      assert.ok(largest < 1e-7, `largest difference ${largest}`);
    });
  }

  it("the output error of sigmoid + cross-entropy is ŷ − y", () => {
    const net = NET.create([1, 1], { out: "sigmoid", rand: seeded(1) });
    const x = [0.5];
    const yHat = NET.predict(net, x);
    const g = NET.gradient(net, [{ x, y: 1 }]);
    assert.ok(Math.abs(g[0].b[0] - (yHat - 1)) < 1e-12); // ∂L/∂b = δ = ŷ − y
  });

  it("a 2-2-1 network learns XOR", () => {
    const rand = seeded(1);
    const net = NET.create([2, 2, 1], { act: "sigmoid", out: "sigmoid", rand });
    const before = NET.lossOf(net, XOR);
    for (let epoch = 0; epoch < 4000; epoch++) NET.trainEpoch(net, XOR, { lr: 2, batch: 4, rand });
    assert.ok(NET.lossOf(net, XOR) < before / 5);
    assert.equal(NET.accuracy(net, XOR), 1);
  });
});
