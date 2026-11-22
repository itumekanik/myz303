/**
 * TensorFlow.js, loaded only when a training run starts (about 1.8 MB). The library files are
 * copied from node_modules to vendor/ by scripts/build.mjs and loaded with <script> tags, which
 * also works on a page opened from a file. The fastest available backend is chosen:
 * WebGPU, then WebGL, then the CPU (which is far too slow to train on, but works).
 */
import type * as TF from "@tensorflow/tfjs";

declare global {
  interface Window {
    tf?: typeof TF;
  }
}

export type Tf = typeof TF;

function script(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.onload = () => resolve();
    el.onerror = () => reject(new Error(`${src} could not be loaded`));
    document.head.append(el);
  });
}

let loading: Promise<{ tf: Tf; backend: string }> | null = null;
/** Backends whose GPU device was lost on this page: the next run uses the next one in line. */
const lost = new Set<string>();

/** The GPU device of this backend is gone (driver reset, hang): do not use it again on this page. */
export function backendLost(backend: string): void {
  lost.add(backend);
  loading = null;
}

export function loadTf(): Promise<{ tf: Tf; backend: string }> {
  loading ??= (async () => {
    if (!window.tf) await script("vendor/tf.min.js");
    const tf = window.tf!;
    if (!tf.findBackendFactory("webgpu")) {
      await script("vendor/tf-backend-webgpu.min.js").catch(() => undefined); // optional: WebGL works without it
    }
    for (const backend of ["webgpu", "webgl", "cpu"]) {
      try {
        if (lost.has(backend) || (backend === "webgpu" && !("gpu" in navigator))) continue;
        if (await tf.setBackend(backend)) {
          await tf.ready();
          return { tf, backend };
        }
      } catch {
        /* try the next backend */
      }
    }
    throw new Error("TensorFlow.js found no backend to run on");
  })();
  loading.catch(() => (loading = null));
  return loading;
}
