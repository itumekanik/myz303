/**
 * Chapter 4, Practice tab: the questions of questions.ts, shown by shared/practice-view.js. The experiments
 * load the instance model and decode the Shanghai tile the first time one of them runs.
 */
import { createPractice } from "../../shared/practice-view.js";
import { loadModel } from "../../engines/yolo/model.ts";
import { decodeImage, loadSamples } from "./page.ts";
import { buildPractice, type YoloLab } from "./questions.ts";
import { MAIN_TILE } from "./theory-demos.ts";

let loading: Promise<YoloLab> | null = null;

function lab(): Promise<YoloLab> {
  loading ??= (async () => {
    const [model, set] = await Promise.all([loadModel("seg"), loadSamples()]);
    const sample = set.samples.find((s) => s.name === MAIN_TILE);
    if (!sample) throw new Error(`no sample ${MAIN_TILE}`);
    const { rgb } = await decodeImage(`data:image/jpeg;base64,${sample.jpeg}`);
    return { model, tile: rgb };
  })();
  loading.catch(() => (loading = null)); // a file that failed to load is tried again next time
  return loading;
}

createPractice({ main: "#px-main", toc: "#px-toc", summary: "#px-summary", sections: buildPractice(lab) });
