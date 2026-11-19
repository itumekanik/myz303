/**
 * Turns a drawing (white digit on black, any size) into an MNIST-style input, as
 * dashboard.py's preprocess_drawing does: MNIST digits were scaled to fit a 20×20 box and
 * centred by their centre of mass inside a 28×28 image, so the drawing gets the same treatment.
 */

/** Grey levels (0–255) of an image given as a data: URL (a data: image never taints a canvas). */
async function greyPixels(dataUrl: string): Promise<{ grey: Uint8ClampedArray; width: number; height: number }> {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const grey = new Uint8ClampedArray(canvas.width * canvas.height);
  for (let i = 0; i < grey.length; i++) grey[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
  return { grey, width: canvas.width, height: canvas.height };
}

/** The drawing as 28×28 values in [0, 1], or null when the canvas is empty. */
export async function preprocessDrawing(dataUrl: string): Promise<Float32Array | null> {
  const { grey, width, height } = await greyPixels(dataUrl);
  let x0 = width,
    y0 = height,
    x1 = -1,
    y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (grey[y * width + x] > 30) {
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
    }
  }
  if (x1 < 0) return null;

  // Crop to the digit and scale its longer side to 20 pixels
  const w = x1 - x0 + 1,
    h = y1 - y0 + 1;
  const scale = 20 / Math.max(w, h);
  const nw = Math.max(1, Math.round(w * scale)),
    nh = Math.max(1, Math.round(h * scale));
  const src = document.createElement("canvas");
  src.width = w;
  src.height = h;
  const crop = new ImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = grey[(y0 + y) * width + x0 + x];
      crop.data.set([v, v, v, 255], 4 * (y * w + x));
    }
  }
  src.getContext("2d")!.putImageData(crop, 0, 0);
  const small = document.createElement("canvas");
  small.width = nw;
  small.height = nh;
  const sctx = small.getContext("2d", { willReadFrequently: true })!;
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(src, 0, 0, nw, nh);
  const scaled = sctx.getImageData(0, 0, nw, nh).data;
  const digit = new Float32Array(nw * nh);
  for (let i = 0; i < digit.length; i++) digit[i] = scaled[4 * i] / 255;

  // Place it so that its centre of mass lands in the middle of the 28×28 image
  let total = 0,
    cy = 0,
    cx = 0;
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const v = digit[y * nw + x];
      total += v;
      cy += v * y;
      cx += v * x;
    }
  }
  const clampTo = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  const top = clampTo(Math.round(14 - cy / total), 28 - nh);
  const left = clampTo(Math.round(14 - cx / total), 28 - nw);
  const out = new Float32Array(784);
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) out[(top + y) * 28 + left + x] = Math.min(1, digit[y * nw + x]);
  return out;
}
