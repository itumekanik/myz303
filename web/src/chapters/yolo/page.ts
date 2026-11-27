/**
 * Chapter 4, YOLO: what its tabs share. The tabs, the sample tiles (data/yolo-samples.js), class
 * names and colours, and TileView: a satellite tile on a canvas with the model's masks, boxes or
 * land-cover map on top.
 *
 * The models work on the letterboxed 640 × 640 input; the tile is drawn at its own size. Model
 * pixels go back to tile pixels through the letterbox (engines/yolo/image.ts).
 */
import { TABS } from "../../shared/page.ts";
import { L } from "../../shared/i18n.js";
import { h, watchSize } from "../../shared/core.js";
import type { Text } from "../../shared/chapters.ts";
import { loadData, type Packed } from "../../engines/data.ts";
import type { Letterbox, Rgb } from "../../engines/yolo/image.ts";
import type { InstanceMask } from "../../engines/yolo/masks.ts";
import { fillPolygon } from "../../engines/yolo/raster.ts";

Object.assign(TABS, {
  theory: { tab: "#tab-theory", panel: "#panel-theory", hash: "#teori" },
  detect: { tab: "#tab-detect", panel: "#panel-detect", hash: "#tespit" },
  train: { tab: "#tab-train", panel: "#panel-train", hash: "#egitim" },
  outputs: { tab: "#tab-outputs", panel: "#panel-outputs", hash: "#ciktilar" },
  math: { tab: "#tab-math", panel: "#panel-math", hash: "#matematik" },
  code: { tab: "#tab-code", panel: "#panel-code", hash: "#kodlama" },
  practice: { tab: "#tab-practice", panel: "#panel-practice", hash: "#alistirma" },
});

export interface Sample {
  name: string;
  city: string;
  tile: string;
  width: number;
  height: number;
  /** ground size of one pixel: east-west, north-south (the tiles are in degrees, so they differ) */
  metresPerPixel: [number, number];
  jpeg: string;
  /** true objects: class and polygon (x, y pairs divided by the tile size) */
  polygons: { cls: number; xy: number[] }[];
  /** true land cover: one class per pixel, 255 where the image has no data */
  landCover: Packed;
}

export interface SampleSet {
  source: string;
  segNames: string[];
  semNames: string[];
  samples: Sample[];
}

export const loadSamples = (): Promise<SampleSet> => loadData<SampleSet>("yolo-samples");

/** A true object: its class, its outline in tile pixels and its mask in model-input pixels. */
export interface Truth {
  cls: number;
  xy: number[];
  mask: InstanceMask;
}

/** The true polygons of a tile as masks in model-input pixels (where the predicted masks live). */
export function truthsOf(s: Sample, lb: Letterbox): Truth[] {
  return s.polygons.map((p) => {
    const xy = p.xy.map((v, i) => v * (i % 2 ? s.height : s.width));
    const model = xy.map((v, i) => v * lb.gain + (i % 2 ? lb.top : lb.left));
    return { cls: p.cls, xy, mask: fillPolygon(model, lb.size, lb.size) };
  });
}

export const CITIES: Record<string, Text> = {
  Vegas: { tr: "Las Vegas", en: "Las Vegas" },
  Paris: { tr: "Paris", en: "Paris" },
  Shanghai: { tr: "Şanghay", en: "Shanghai" },
  Khartoum: { tr: "Hartum", en: "Khartoum" },
};

export const SEG_CLASSES: Text[] = [
  { tr: "bina", en: "building" },
  { tr: "spor sahası", en: "sports field" },
];
export const SEM_CLASSES: Text[] = [
  { tr: "arka plan", en: "background" },
  { tr: "bina", en: "building" },
  { tr: "yol", en: "road" },
  { tr: "yeşil alan", en: "green area" },
  { tr: "su", en: "water" },
  { tr: "otopark", en: "parking" },
];

/** Colours drawn over the imagery: the same in both themes, since the photo is the background. */
export const SEG_RGB: [number, number, number][] = [
  [255, 214, 10],
  [236, 72, 196],
];
export const SEM_RGB: [number, number, number][] = [
  [0, 0, 0],
  [232, 70, 70],
  [215, 215, 215],
  [60, 200, 80],
  [40, 140, 240],
  [250, 180, 30],
];
export const MATCH_RGB = { hit: [40, 200, 90], wrong: [235, 60, 60], missed: [255, 150, 0] } as const;
export const css = (c: readonly number[], alpha = 1): string => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;

export const T = (text: Text): string => L(text) as string;

/** Decodes a tile and reads its pixels (the browsers decode JPEG exactly as OpenCV does). */
export async function decodeImage(src: string, maxSide = Infinity): Promise<{ image: HTMLImageElement; rgb: Rgb; scale: number }> {
  const image = new Image();
  image.src = src;
  await image.decode();
  // an image larger than maxSide is read smaller; scale = original pixels per pixel read
  const scale = Math.max(1, Math.max(image.naturalWidth, image.naturalHeight) / maxSide);
  const width = Math.round(image.naturalWidth / scale);
  const height = Math.round(image.naturalHeight / scale);
  return { image, rgb: readRgb(image, width, height), scale };
}

export function readRgb(source: CanvasImageSource, width: number, height: number): Rgb {
  const canvas = h("canvas", { width, height }) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(source, 0, 0, width, height);
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    data[3 * i] = rgba[4 * i];
    data[3 * i + 1] = rgba[4 * i + 1];
    data[3 * i + 2] = rgba[4 * i + 2];
  }
  return { width, height, data };
}

/** A model-input-sized overlay: masks painted in their colours (later masks on top). */
export function maskOverlay(size: number, masks: { mask: InstanceMask; rgb: readonly number[] }[], alpha = 0.5): HTMLCanvasElement {
  const canvas = h("canvas", { width: size, height: size }) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(size, size);
  const a = Math.round(alpha * 255);
  for (const { mask: m, rgb } of masks) {
    for (let y = 0; y < m.height; y++) {
      for (let x = 0; x < m.width; x++) {
        if (!m.data[y * m.width + x]) continue;
        const o = 4 * ((m.y + y) * size + m.x + x);
        img.data[o] = rgb[0];
        img.data[o + 1] = rgb[1];
        img.data[o + 2] = rgb[2];
        img.data[o + 3] = a;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** A class map as an overlay; class 0 (background) and 255 (no data) stay clear. */
export function classOverlay(classes: Uint8Array, width: number, height: number, alpha = 0.55): HTMLCanvasElement {
  const canvas = h("canvas", { width, height }) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  const a = Math.round(alpha * 255);
  for (let i = 0; i < width * height; i++) {
    const c = classes[i];
    if (!c || c >= SEM_RGB.length) continue;
    img.data.set([SEM_RGB[c][0], SEM_RGB[c][1], SEM_RGB[c][2], a], 4 * i);
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export interface Layer {
  /** an overlay in model-input pixels (placed through the letterbox) or in tile pixels */
  canvas?: HTMLCanvasElement;
  letterbox?: Letterbox;
  /** outlines in tile pixels */
  boxes?: { box: readonly number[]; rgb: readonly number[]; label?: string; dashed?: boolean }[];
  polygons?: { xy: readonly number[]; rgb: readonly number[]; width?: number; dashed?: boolean; fill?: number }[];
  /** free drawing on top; `s` is canvas pixels per tile pixel, `dpr` the device pixel ratio */
  paint?: (ctx: CanvasRenderingContext2D, s: number, dpr: number) => void;
}

/** A tile on a square canvas that follows its CSS size, with layers drawn on top. */
export class TileView {
  readonly canvas: HTMLCanvasElement;
  private image: CanvasImageSource | null = null;
  private width = 1;
  private height = 1;
  private layers: Layer[] = [];

  constructor(label: string) {
    this.canvas = h("canvas", { class: "yo-tile", role: "img", "aria-label": label }) as HTMLCanvasElement;
    watchSize(this.canvas, () => this.draw());
  }

  show(image: CanvasImageSource, width: number, height: number, layers: Layer[]): void {
    this.image = image;
    this.width = width;
    this.height = height;
    this.layers = layers;
    this.draw();
  }

  draw(): void {
    const c = this.canvas;
    const cssWidth = c.clientWidth;
    if (!cssWidth || !this.image) return;
    const dpr = window.devicePixelRatio || 1;
    const cssHeight = (cssWidth * this.height) / this.width;
    c.style.height = `${cssHeight}px`;
    c.width = Math.round(cssWidth * dpr);
    c.height = Math.round(cssHeight * dpr);
    const ctx = c.getContext("2d")!;
    const s = c.width / this.width; // canvas pixels per tile pixel
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.image, 0, 0, c.width, c.height);
    for (const layer of this.layers) {
      if (layer.canvas) {
        ctx.imageSmoothingEnabled = false;
        const lb = layer.letterbox;
        if (lb) ctx.drawImage(layer.canvas, lb.left, lb.top, lb.width, lb.height, 0, 0, c.width, c.height);
        else ctx.drawImage(layer.canvas, 0, 0, c.width, c.height);
      }
      ctx.lineJoin = "round";
      for (const p of layer.polygons ?? []) {
        ctx.beginPath();
        for (let i = 0; i < p.xy.length; i += 2) ctx[i ? "lineTo" : "moveTo"](p.xy[i] * s, p.xy[i + 1] * s);
        ctx.closePath();
        if (p.fill) {
          ctx.fillStyle = css(p.rgb, p.fill);
          ctx.fill();
        }
        ctx.setLineDash(p.dashed ? [4 * dpr, 3 * dpr] : []);
        ctx.lineWidth = (p.width ?? 1.5) * dpr;
        ctx.strokeStyle = css(p.rgb);
        ctx.stroke();
      }
      for (const b of layer.boxes ?? []) {
        ctx.setLineDash(b.dashed ? [4 * dpr, 3 * dpr] : []);
        ctx.lineWidth = 1.5 * dpr;
        ctx.strokeStyle = css(b.rgb);
        ctx.strokeRect(b.box[0] * s, b.box[1] * s, (b.box[2] - b.box[0]) * s, (b.box[3] - b.box[1]) * s);
        if (b.label) {
          ctx.font = `${11 * dpr}px system-ui, sans-serif`;
          const w = ctx.measureText(b.label).width + 6 * dpr;
          const y = Math.max(0, b.box[1] * s - 14 * dpr);
          ctx.fillStyle = css(b.rgb, 0.9);
          ctx.fillRect(b.box[0] * s, y, w, 14 * dpr);
          ctx.fillStyle = "#000";
          ctx.fillText(b.label, b.box[0] * s + 3 * dpr, y + 11 * dpr);
        }
      }
      ctx.setLineDash([]);
      layer.paint?.(ctx, s, dpr);
    }
  }

  /** Where a pointer event is, in tile pixels. */
  toTile(event: { clientX: number; clientY: number }): { x: number; y: number } {
    const r = this.canvas.getBoundingClientRect();
    return { x: ((event.clientX - r.left) / r.width) * this.width, y: ((event.clientY - r.top) / r.height) * this.height };
  }
}
