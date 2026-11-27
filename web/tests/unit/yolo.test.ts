/**
 * The YOLO engine (src/engines/yolo/) against what Python gives: OpenCV's resize, Ultralytics' letterbox,
 * NMS, masks and AP, and both trained models through ONNX Runtime. The expected values are in
 * tests/fixtures/yolo-parity.json, written by tools/export_yolo_assets.py.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { before, describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";
import { gunzipSync } from "node:zlib";
import { fromBase64 } from "../../src/engines/data.ts";
import { aboveConfidence, anchorOf, boxIou, candidateCount, nms, type Candidate } from "../../src/engines/yolo/detect.ts";
import { letterbox, letterboxParams, resizeLinear, roundHalfEven, toImageBox } from "../../src/engines/yolo/image.ts";
import { areaSquareMetres } from "../../src/engines/yolo/masks.ts";
import { averagePrecision, matchPredictions, precisionRecall } from "../../src/engines/yolo/metrics.ts";
import { createModel, initRuntime, runInstances, runLandCover, type PackedYolo, type YoloModel } from "../../src/engines/yolo/model.ts";
import { fillPolygon, maskIou, unionArea } from "../../src/engines/yolo/raster.ts";

const here = dirname(fileURLToPath(import.meta.url));
const WEB = join(here, "../..");

interface Fixture {
  resize: { src: string; width: number; height: number; dstWidth: number; dstHeight: number; dst: string }[];
  letterbox: {
    params: {
      height: number;
      width: number;
      newWidth: number;
      newHeight: number;
      top: number;
      left: number;
      gain: number;
      out: number[];
    }[];
    pixels: { src: string; width: number; height: number; size: number; dst: string };
  };
  model: {
    rgb: { gzip: string };
    width: number;
    height: number;
    conf: number;
    iou: number;
    aboveConf: number;
    detections: { cls: number; score: number; box: number[]; maskArea: number }[];
    landCover: number[];
  };
  ap: { recall: number[]; precision: number[]; ap: number }[];
}

const fixture = JSON.parse(readFileSync(join(here, "../fixtures/yolo-parity.json"), "utf8")) as Fixture;

/** Runs a data file the way a page does (a script that fills window.__AI_DATA__). */
function dataFile<T>(name: string): T {
  const window: { __AI_DATA__?: Record<string, unknown> } = {};
  runInNewContext(readFileSync(join(WEB, `public/data/${name}.js`), "utf8"), { window });
  return window.__AI_DATA__![name] as T;
}

const gz = (p: { gzip: string } | string) => new Uint8Array(gunzipSync(fromBase64(typeof p === "string" ? p : p.gzip)));

describe("YOLO image preparation (engines/yolo/image.ts)", () => {
  it("rounds halves to even like Python", () => {
    assert.deepEqual([0.5, 1.5, 2.5, -0.5, -1.5, 2.4, 2.6, 3].map(roundHalfEven), [0, 2, 2, 0, -2, 2, 3, 3]);
  });

  it("resizes byte for byte like cv2.resize(INTER_LINEAR), the exact-half case included", () => {
    for (const c of fixture.resize) {
      const out = resizeLinear({ width: c.width, height: c.height, data: fromBase64(c.src) }, c.dstWidth, c.dstHeight);
      const want = fromBase64(c.dst);
      const diff = out.data.findIndex((v, i) => v !== want[i]);
      assert.equal(diff, -1, `${c.width}×${c.height} → ${c.dstWidth}×${c.dstHeight}: first difference at byte ${diff}`);
    }
  });

  it("letterboxes like Ultralytics' LetterBox(auto=False)", () => {
    for (const p of fixture.letterbox.params) {
      const lb = letterboxParams(p.width, p.height, 640);
      assert.deepEqual([lb.width, lb.height, lb.top, lb.left], [p.newWidth, p.newHeight, p.top, p.left], `${p.width}×${p.height}`);
      assert.ok(Math.abs(lb.gain - p.gain) < 1e-12);
    }
    const c = fixture.letterbox.pixels;
    const { image } = letterbox({ width: c.width, height: c.height, data: fromBase64(c.src) }, c.size);
    assert.deepEqual(image.data, fromBase64(c.dst));
  });

  it("maps model boxes back to the image and clips them", () => {
    const lb = letterboxParams(1000, 500, 640); // gain 0.64, 160 grey rows above
    assert.deepEqual(toImageBox([64, 160, 640, 480], lb, 1000, 500), [100, 0, 1000, 500]);
    assert.deepEqual(toImageBox([-5, 100, 700, 600], lb, 1000, 500), [0, 0, 1000, 500]);
  });
});

describe("YOLO detection steps (engines/yolo/detect.ts)", () => {
  it("has 8400 candidates on three grids for a 640 input", () => {
    assert.equal(candidateCount(640), 8400);
    assert.equal(candidateCount(512), 5376);
    assert.deepEqual(anchorOf(0), { level: 0, stride: 8, grid: 80, gx: 0, gy: 0 });
    assert.deepEqual(anchorOf(6399), { level: 0, stride: 8, grid: 80, gx: 79, gy: 79 });
    assert.deepEqual(anchorOf(6400), { level: 1, stride: 16, grid: 40, gx: 0, gy: 0 });
    assert.deepEqual(anchorOf(8399), { level: 2, stride: 32, grid: 20, gx: 19, gy: 19 });
    assert.throws(() => anchorOf(8400));
  });

  it("computes IoU", () => {
    assert.equal(boxIou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150);
    assert.equal(boxIou([0, 0, 10, 10], [20, 20, 30, 30]), 0);
  });

  it("suppresses only boxes of the same class that overlap more than the threshold", () => {
    const c = (cls: number, score: number, box: Candidate["box"]): Candidate => ({ index: 0, cls, score, box });
    const sorted = [c(0, 0.9, [0, 0, 10, 10]), c(0, 0.8, [1, 0, 11, 10]), c(1, 0.7, [1, 0, 11, 10]), c(0, 0.6, [5, 0, 15, 10])];
    // IoU(first, second) = 0.818 → dropped; the class-1 box stays; IoU(first, fourth) = 1/3 → kept
    assert.deepEqual(
      nms(sorted, 0.7).map((k) => k.score),
      [0.9, 0.7, 0.6],
    );
    // a box overlapping exactly at the threshold stays (torchvision drops IoU > threshold)
    assert.equal(nms([c(0, 0.9, [0, 0, 10, 10]), c(0, 0.8, [0, 0, 10, 5])], 0.5).length, 2);
    assert.equal(nms(sorted, 0.7, 2).length, 2);
  });
});

describe("YOLO metrics (engines/yolo/metrics.ts)", () => {
  it("gives Ultralytics' compute_ap", () => {
    for (const c of fixture.ap)
      assert.ok(Math.abs(averagePrecision(c.recall, c.precision) - c.ap) < 1e-12, `${averagePrecision(c.recall, c.precision)} vs ${c.ap}`);
  });

  it("matches predictions to true objects greedily, best score first", () => {
    const preds = [
      { cls: 0, score: 0.6, box: [0, 0, 10, 10] },
      { cls: 0, score: 0.9, box: [1, 0, 11, 10] },
      { cls: 1, score: 0.8, box: [20, 20, 30, 30] },
    ];
    const truths = [
      { cls: 0, box: [0, 0, 10, 10] },
      { cls: 0, box: [40, 40, 50, 50] },
    ];
    const { matches, missed } = matchPredictions(preds, truths, (p, t) => boxIou(preds[p].box, truths[t].box));
    assert.deepEqual(
      matches.map((m) => m.truth),
      [-1, 0, -1],
    ); // the 0.9 box takes the building; the 0.6 one is a duplicate; class 1 has no truth
    assert.deepEqual(missed, [1]);
    const { precision, recall } = precisionRecall(
      preds.map((p) => p.score),
      matches.map((m) => m.truth >= 0),
      truths.length,
    );
    assert.deepEqual(precision, [1, 0.5, 1 / 3]);
    assert.deepEqual(recall, [0.5, 0.5, 0.5]);
  });

  it("converts mask pixels to square metres", () => {
    // 1000 model pixels at gain 0.5 are 4000 image pixels; 0.3 m × 0.25 m each
    assert.ok(Math.abs(areaSquareMetres(1000, 0.5, [0.3, 0.25]) - 300) < 1e-9);
  });
});

describe("YOLO polygons and mask overlap (engines/yolo/raster.ts)", () => {
  it("fills the pixels whose centre lies inside a polygon", () => {
    const square = fillPolygon([2, 2, 6, 2, 6, 5, 2, 5], 10, 10); // 4 × 3 pixels
    assert.deepEqual([square.x, square.y, square.width, square.height, square.area], [2, 2, 4, 3, 12]);
    const triangle = fillPolygon([0, 0, 10, 0, 0, 10], 10, 10); // half of 10 × 10, centres strictly inside
    assert.equal(triangle.area, 45);
    const clipped = fillPolygon([-5, -5, 3, -5, 3, 3, -5, 3], 10, 10); // only the part inside the image
    assert.deepEqual([clipped.x, clipped.y, clipped.area], [0, 0, 9]);
  });

  it("measures IoU and the union of overlapping masks", () => {
    const a = fillPolygon([0, 0, 4, 0, 4, 4, 0, 4], 10, 10);
    const b = fillPolygon([2, 0, 6, 0, 6, 4, 2, 4], 10, 10);
    assert.equal(maskIou(a, b), 8 / 24);
    assert.equal(maskIou(a, a), 1);
    assert.equal(maskIou(a, fillPolygon([7, 7, 9, 7, 9, 9, 7, 9], 10, 10)), 0);
    assert.equal(unionArea([a, b], 10, 10), 24);
  });
});

describe("YOLO26n models in ONNX Runtime (engines/yolo/model.ts)", () => {
  let seg: YoloModel;
  let sem: YoloModel;
  const m = fixture.model;
  const image = { width: m.width, height: m.height, data: gz(m.rgb) };

  before(async () => {
    await initRuntime(async () => new Uint8Array(readFileSync(join(WEB, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"))));
    const load = (name: string) => {
      const p = dataFile<PackedYolo>(name);
      return createModel(gz(p.onnx as { gzip: string }), p.names, p.size);
    };
    [seg, sem] = await Promise.all([load("yolo-seg"), load("yolo-sem")]);
  });

  it("finds the objects Ultralytics finds, with the same scores, boxes and mask areas", async () => {
    const r = await runInstances(seg, image, { conf: m.conf, iou: m.iou });
    assert.deepEqual(seg.names, ["building", "sports_field"]);
    assert.equal(r.head.count, 8400);
    assert.equal(r.protos.k, 32);
    assert.equal(aboveConfidence(r.head, 2, m.conf).length, m.aboveConf);
    assert.equal(r.objects.length, m.detections.length);
    r.objects.forEach((o, i) => {
      const want = m.detections[i];
      assert.equal(o.cls, want.cls, `object ${i}: class`);
      assert.ok(Math.abs(o.score - want.score) < 1e-3, `object ${i}: score ${o.score} vs ${want.score}`);
      // float16 inside the model: between 512 and 1024 its step is 0.5 px, and the two runtimes may round differently
      o.box.forEach((v, k) => assert.ok(Math.abs(v - want.box[k]) <= 0.5, `object ${i}: box ${o.box} vs ${want.box}`));
      const tolerance = Math.max(5, 0.002 * want.maskArea); // pixels where the logit is ~0 may flip
      assert.ok(Math.abs(o.mask.area - want.maskArea) <= tolerance, `object ${i}: mask ${o.mask.area} vs ${want.maskArea}`);
    });
  });

  it("paints the land-cover classes Ultralytics paints", async () => {
    const r = await runLandCover(sem, image);
    assert.deepEqual(sem.names, ["background", "building", "road", "green", "water", "parking"]);
    const total = r.counts.reduce((a, b) => a + b, 0);
    assert.equal(total, 640 * 640);
    r.counts.forEach((n, c) => assert.ok(Math.abs(n - m.landCover[c]) <= 0.001 * total, `class ${c}: ${n} vs ${m.landCover[c]}`));
  });
});
