/**
 * Loading the data files in public/data/ (written by tools/export_assets.py).
 *
 * A page opened from a file may not fetch() other files, but it may run scripts: every data file
 * is a script that stores its content in window.__AI_DATA__[name]. Binary arrays inside are base64
 * strings, gzip-compressed where marked ({ gzip: "…" }).
 */

declare global {
  interface Window {
    __AI_DATA__?: Record<string, unknown>;
  }
}

/** A base64 string, optionally gzip-compressed. */
export type Packed = string | { gzip: string };

const loading = new Map<string, Promise<unknown>>();

/** Loads data/<name>.js once and returns what it stored (later calls share the same promise). */
export function loadData<T>(name: string): Promise<T> {
  let promise = loading.get(name);
  if (!promise) {
    promise = new Promise((resolve, reject) => {
      const ready = () => window.__AI_DATA__?.[name];
      if (ready()) {
        resolve(ready());
        return;
      }
      const script = document.createElement("script");
      script.src = `data/${name}.js`;
      script.async = true;
      script.onload = () => {
        const value = ready();
        if (value) {
          delete window.__AI_DATA__![name]; // the caller keeps what it needs; the strings can go
          resolve(value);
        } else reject(new Error(`data/${name}.js did not define "${name}"`));
      };
      script.onerror = () => {
        loading.delete(name); // allow a retry
        reject(new Error(`data/${name}.js could not be loaded`));
      };
      document.head.append(script);
    });
    loading.set(name, promise);
  }
  return promise as Promise<T>;
}

export function fromBase64(text: string): Uint8Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** The bytes of a packed array. */
export async function unpack(packed: Packed): Promise<Uint8Array> {
  return typeof packed === "string" ? fromBase64(packed) : gunzip(fromBase64(packed.gzip));
}

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
