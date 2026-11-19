/**
 * The last model trained in this browser, kept in IndexedDB (the browser's own storage, which
 * also works for a page opened from a file), so it is there again after a reload, like
 * checkpoints/simple_cnn.pt for the Python dashboard. Only this browser on this computer sees it.
 */
import type { CnnWeights } from "./forward.ts";
import type { RunRecord } from "./session.ts";

export interface SavedModel {
  weights: CnnWeights;
  initial: CnnWeights;
  meta: Record<string, unknown>;
  run: RunRecord;
}

const DB = "ai-learning-path";
const STORE = "cnn";
const KEY = "last-run";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function request<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const req = run(db.transaction(STORE, mode).objectStore(STORE));
      req.onsuccess = () => resolve(req.result as T);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

export const saveModel = (saved: SavedModel): Promise<unknown> => request("readwrite", (s) => s.put(saved, KEY));

export async function loadSaved(): Promise<SavedModel | null> {
  if (typeof indexedDB === "undefined") return null;
  return (await request<SavedModel | undefined>("readonly", (s) => s.get(KEY))) ?? null;
}

/** Forgets the browser's model: the page goes back to the one trained with PyTorch. */
export const clearSaved = (): Promise<unknown> => request("readwrite", (s) => s.delete(KEY));
