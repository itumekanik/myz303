import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DEFAULTS, gpuFromRenderer, parseConfig } from "../../src/engines/cnn/session.ts";

describe("training settings (engines/cnn/session.ts)", () => {
  it("fills in SimpleCNN.py's defaults", () => {
    assert.deepEqual(parseConfig({}), DEFAULTS);
    assert.deepEqual(DEFAULTS, { epochs: 10, lr: 1e-3, batch_size: 256, seed: 42 });
  });

  it("accepts numbers given as text, and truncates whole-number settings like Python's int()", () => {
    assert.deepEqual(parseConfig({ epochs: "3", lr: "0.01", batch_size: 64.9, seed: "7" }), {
      epochs: 3,
      lr: 0.01,
      batch_size: 64,
      seed: 7,
    });
  });

  it("rejects values out of range with dashboard.py's messages", () => {
    assert.throws(() => parseConfig({ epochs: 0 }), /^Error: 'epochs' 1 ile 50 arasında olmalı\.$/);
    assert.throws(() => parseConfig({ lr: 2 }), /'lr' 0.00001 ile 1 arasında olmalı|'lr' 1e-5 ile 1 arasında olmalı/);
    assert.throws(() => parseConfig({ batch_size: "abc" }), /'batch_size' geçerli bir sayı değil\./);
  });
});

describe("GPU name from WebGL's renderer string", () => {
  it("reads ANGLE strings on Windows", () => {
    assert.equal(
      gpuFromRenderer("ANGLE (NVIDIA, NVIDIA GeForce RTX 2060 (0x00001F15) Direct3D11 vs_5_0 ps_5_0, D3D11)"),
      "NVIDIA GeForce RTX 2060",
    );
    assert.equal(
      gpuFromRenderer("ANGLE (Intel, Intel(R) UHD Graphics (0x00009BC4) Direct3D11 vs_5_0 ps_5_0, D3D11)"),
      "Intel(R) UHD Graphics",
    );
  });

  it("reads macOS strings and other strings as they are", () => {
    assert.equal(gpuFromRenderer("ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)"), "Apple M2");
    assert.equal(gpuFromRenderer("Mali-G78"), "Mali-G78");
  });
});
