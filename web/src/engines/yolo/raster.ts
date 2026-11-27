/**
 * Polygons to pixel masks, and how much two masks overlap: what comparing a prediction with the
 * true outline of a building needs. A pixel belongs to a polygon when its centre lies inside it.
 */
import type { InstanceMask } from "./masks.ts";

/** The pixels of a polygon (flat x, y list in pixels), stored over its bounding box inside width × height. */
export function fillPolygon(xy: readonly number[], width: number, height: number): InstanceMask {
  const n = xy.length / 2;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    minX = Math.min(minX, xy[2 * i]);
    maxX = Math.max(maxX, xy[2 * i]);
    minY = Math.min(minY, xy[2 * i + 1]);
    maxY = Math.max(maxY, xy[2 * i + 1]);
  }
  const x0 = Math.max(0, Math.floor(minX));
  const y0 = Math.max(0, Math.floor(minY));
  const x1 = Math.min(width, Math.ceil(maxX));
  const y1 = Math.min(height, Math.ceil(maxY));
  const w = Math.max(0, x1 - x0);
  const h = Math.max(0, y1 - y0);
  const data = new Uint8Array(w * h);
  let area = 0;
  const crossings: number[] = [];
  for (let y = y0; y < y1; y++) {
    const cy = y + 0.5;
    crossings.length = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const ay = xy[2 * j + 1];
      const by = xy[2 * i + 1];
      if (ay <= cy !== by <= cy) {
        const ax = xy[2 * j];
        const bx = xy[2 * i];
        crossings.push(ax + ((cy - ay) / (by - ay)) * (bx - ax));
      }
    }
    crossings.sort((a, b) => a - b);
    for (let k = 0; k + 1 < crossings.length; k += 2) {
      // pixels whose centre x + 0.5 lies between two crossings
      const from = Math.max(x0, Math.ceil(crossings[k] - 0.5));
      const to = Math.min(x1, Math.ceil(crossings[k + 1] - 0.5));
      for (let x = from; x < to; x++) {
        const i = (y - y0) * w + (x - x0);
        if (!data[i]) {
          data[i] = 1;
          area++;
        }
      }
    }
  }
  return { x: x0, y: y0, width: w, height: h, data, area };
}

/** Intersection over union of two masks in the same pixel grid. */
export function maskIou(a: InstanceMask, b: InstanceMask): number {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.width, b.x + b.width);
  const y1 = Math.min(a.y + a.height, b.y + b.height);
  let inter = 0;
  for (let y = y0; y < y1; y++) {
    const ra = (y - a.y) * a.width - a.x;
    const rb = (y - b.y) * b.width - b.x;
    for (let x = x0; x < x1; x++) if (a.data[ra + x] && b.data[rb + x]) inter++;
  }
  const union = a.area + b.area - inter;
  return union > 0 ? inter / union : 0;
}

/** Pixels covered by at least one of the masks (overlapping masks are not counted twice). */
export function unionArea(masks: readonly InstanceMask[], width: number, height: number): number {
  const seen = new Uint8Array(width * height);
  let area = 0;
  for (const m of masks) {
    for (let y = 0; y < m.height; y++) {
      for (let x = 0; x < m.width; x++) {
        if (!m.data[y * m.width + x]) continue;
        const i = (m.y + y) * width + m.x + x;
        if (!seen[i]) {
          seen[i] = 1;
          area++;
        }
      }
    }
  }
  return area;
}
