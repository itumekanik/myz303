/*
 * Chapter 3, Practice tab: the questions of questions.ts, shown by shared/practice-view.js.
 * The experiments load the trained model and the test images the first time one of them runs.
 */
import { loadData, mnist } from "../../engines/cnn/data.ts";
import { loadModel } from "../../engines/cnn/weights.ts";
import { createPractice } from "../../shared/practice-view.js";
import { buildPractice } from "./questions.ts";

let loading = null;
function lab() {
  loading ??= Promise.all([
    loadModel("cnn-trained"),
    loadData("mnist-test").then((data) => mnist(data.images, data.labels)),
  ]).then(([{ weights }, test]) => ({ weights, test }));
  loading.catch(() => { loading = null; });   // a file that failed to load is tried again next time
  return loading;
}

createPractice({ main: "#px-main", toc: "#px-toc", summary: "#px-summary", sections: buildPractice(lab) });
