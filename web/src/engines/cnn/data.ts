/** The CNN chapter's data: float32 weights and the MNIST image sets. Loading itself is in ../data.ts. */
import { fromBase64, unpack, type Packed } from "../data.ts";

export { fromBase64, gunzip, loadData, unpack, type Packed } from "../data.ts";

/** Little-endian float32 values (every browser is little-endian). */
export function float32(text: string): Float32Array {
  const bytes = fromBase64(text);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** An MNIST image set: 28×28 uint8 pixels, one image after the other, and the labels. */
export interface ImageSet {
  count: number;
  pixels: Uint8Array;
  labels: Uint8Array;
}

/** Parses the original MNIST files (idx format: 16-byte header for images, 8 for labels). */
export async function mnist(packedImages: Packed, packedLabels: Packed): Promise<ImageSet> {
  const [images, labels] = await Promise.all([unpack(packedImages), unpack(packedLabels)]);
  const count = new DataView(labels.buffer, labels.byteOffset).getUint32(4);
  return { count, pixels: images.subarray(16, 16 + count * 784), labels: labels.subarray(8, 8 + count) };
}
