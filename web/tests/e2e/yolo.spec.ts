import { readFileSync } from "node:fs";
import { expect, open, overflows, test } from "./fixtures";

/**
 * Chapter 4, YOLO. The first tile is Shanghai_img209; the numbers below are what ONNX Runtime gives in
 * Python on the same tile with the same post-processing (107 candidates above 0.25, then 11 buildings
 * and 4 sports fields), so these tests also check that every browser finds what Python finds.
 */
test.describe("chapter 4: yolo", () => {
  test("theory: eight sections and the demos without the model open at once", async ({ page, errors }) => {
    await open(page, "yolo.html");
    await expect(page.locator("#tab-theory")).toHaveAttribute("aria-selected", "true");
    for (const id of ["y-intro", "y-tasks", "y-iou", "y-grid", "y-nms", "y-masks", "y-train", "y-metrics", "y-read", "y-recap"])
      await expect(page.locator(`#${id}`)).toBeVisible();
    await expect(page.locator("#panel-theory .katex").first()).toBeVisible();
    await expect(page.locator(".katex-error")).toHaveCount(0);
    // the four tasks, from the true labels
    await page.locator("#y-tasks .seg button").nth(0).click();
    await expect(page.locator("#y-tasks .yo-chip.on")).toHaveCount(5);
    await page.locator("#y-tasks .seg button").nth(3).click();
    await expect(page.locator("#y-tasks .yo-caption")).toContainText("19 nesne");
    // IoU of the true box and the prediction box, moved with a slider
    await expect(page.locator("#y-iou .pc-msg")).toContainText("IoU < 0,5");
    // the models are not loaded until asked for
    await expect(page.locator("#y-grid .yo-gate button")).toBeVisible();
    await expect(page.locator(".gl-term[data-term=nms]").first()).toBeVisible();
    void errors;
  });

  test("theory: the real model runs once and every demo uses it", async ({ page, errors }) => {
    test.setTimeout(120_000);
    await open(page, "yolo.html");
    await page.locator("#y-nms .yo-gate button").click();
    const nms = page.locator("#y-nms .yo-big");
    await expect(nms).toContainText("107 aday", { timeout: 90_000 });
    await expect(nms).toContainText("15 nesne");
    await page.getByRole("button", { name: "Bir adım" }).click();
    await expect(nms).toContainText("Adım 1/15");
    await page.getByRole("button", { name: "Sona git" }).click();
    await expect(nms).toContainText("Adım 15/15");
    // the other demos started by themselves
    await expect(page.locator("#y-grid .yo-caption")).toContainText("40×40: 64 bina, 6 saha", { timeout: 60_000 });
    await expect(page.locator("#y-masks .yo-proto")).toHaveCount(32);
    await expect(page.locator("#y-metrics .yo-big")).toContainText("51 tespit", { timeout: 60_000 });
    await expect(page.locator("#y-metrics .yo-big")).toContainText("AP50");
    await page.locator("#y-read .yo-slider input").fill("1");
    await expect(page.locator("#y-read .yo-big")).toContainText("dışarı");
    void errors;
  });

  test("detection: both models run in the browser and find what Python finds", async ({ page, errors }) => {
    test.setTimeout(120_000);
    await open(page, "yolo.html#tespit");
    await expect(page).toHaveTitle("YOLO · Bölüm 4");
    await expect(page.locator("#tab-detect")).toHaveAttribute("aria-selected", "true");
    const pipeline = page.locator("#yd-pipeline");
    await expect(pipeline).toContainText("8.400 aday", { timeout: 90_000 });
    await expect(pipeline).toContainText("107 güven eşiğini");
    await expect(pipeline).toContainText("15 nesne");
    const found = page.locator("#yd-results tbody tr").first();
    await expect(found).toHaveText(/bulunan\s*11\s*4/);

    // a lower threshold keeps more candidates; NMS removes the duplicates
    await page.locator("#yd-root .yo-slider input").first().fill("0.05");
    await expect(pipeline).not.toContainText("107 güven eşiğini");

    // against the true labels: correct, wrong and missed
    await page.getByLabel("Doğru etiketlerle karşılaştır").check();
    await expect(page.locator("#yd-legend")).toContainText("kaçan");
    await expect(page.locator("#yd-results")).toContainText("duyarlılık");

    // the land-cover model
    await page.getByRole("button", { name: "Arazi örtüsü" }).click();
    await expect(page.locator("#yd-results")).toContainText("Doğru sınıflanan piksel");
    await expect(page.locator("#yd-results tbody tr")).toHaveCount(6);
    void errors;
  });

  test("detection: another tile, and the page in English", async ({ page, errors }) => {
    test.setTimeout(120_000);
    await open(page, "yolo.html#tespit");
    await expect(page.locator("#yd-pipeline")).toContainText("8.400", { timeout: 90_000 });
    await page.locator(".yo-thumb").first().click();
    await expect(page.locator(".yo-view-card h2")).toContainText("Las Vegas");
    await expect(page.locator(".yo-thumb").first()).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("#yd-pipeline")).toContainText("8.400", { timeout: 60_000 });
    await page.locator("#lang-switch button[data-lang=en]").click();
    await expect(page.locator("#yd-pipeline")).toContainText("8,400 candidates");
    await expect(page.locator("#yd-results h2")).toHaveText("Result");
    void errors;
  });

  test("training: the record of our runs, and the same tile epoch by epoch", async ({ page, errors }) => {
    await open(page, "yolo.html#egitim");
    await expect(page.locator("#tab-train")).toHaveAttribute("aria-selected", "true");
    await expect(page.locator(".yo-facts")).toContainText("528");
    const charts = page.locator("#yt-root .chart svg");
    await expect(charts).toHaveCount(4);
    for (let k = 0; k < 4; k++) expect(await charts.nth(k).locator("path.series-line").count()).toBeGreaterThan(0);
    await expect(page.locator("#yt-root .yo-table").first()).toContainText("0,69"); // building mask AP50, pretrained
    const note = page.locator(".yo-training .yo-big").first();
    await expect(note).toContainText("100. epoch");
    await page.locator("#yt-root").getByRole("button", { name: "1", exact: true }).click();
    await expect(note).toContainText("1. epoch");
    await page.locator("#yt-root").getByRole("button", { name: "Sıfırdan (rastgele ağırlıklar)" }).click();
    await expect(note).toContainText("0 bina, 0 spor sahası"); // random weights find nothing after one epoch
    void errors;
  });

  test("outputs: the files of the run, and the confusion matrix in four steps", async ({ page, errors }) => {
    test.setTimeout(90_000);
    await open(page, "yolo.html#ciktilar");
    await expect(page.locator("#tab-outputs")).toHaveAttribute("aria-selected", "true");
    for (const id of [
      "yr-intro",
      "yr-folder",
      "yr-console",
      "yr-results",
      "yr-labels",
      "yr-batches",
      "yr-cm-intro",
      "yr-cm-toy",
      "yr-cm-real",
      "yr-cm-sem",
      "yr-curves",
      "yr-recap",
    ])
      await expect(page.locator(`#${id}`)).toBeVisible();
    await expect(page.locator(".katex-error")).toHaveCount(0);
    // 1. the damage survey: calling everything "sound" is still 90% accurate
    const survey = page.locator("#yr-cm-intro");
    await expect(survey.locator(".yr-stats")).toContainText("%91");
    await survey.getByRole("button", { name: 'Hepsine "sağlam" de' }).click();
    await expect(survey.locator(".yr-stats")).toContainText("%90");
    await expect(survey.locator(".yo-big")).toContainText("duyarlılık 0");
    // 2. the toy tile, matched the way Ultralytics does it: 5 buildings found, 3 boxes for nothing, 3 missed
    const toy = page.locator("#yr-cm-toy");
    await expect(toy.locator(".yr-cell").nth(0)).toHaveText("5");
    await expect(toy.locator(".yr-cell").nth(2)).toHaveText("3");
    await expect(toy.locator(".yr-cell").nth(6)).toHaveText("3");
    await expect(toy.locator(".yr-cell").nth(8)).toHaveText("–");
    await expect(toy.locator(".yr-toy-list li")).toHaveCount(3);
    // 3. our matrix: at 0.001 as confusion_matrix.png, then at 0.25
    const real = page.locator("#yr-cm-real");
    await expect(real.locator(".yr-cell").nth(2)).toHaveText("24.461", { timeout: 60_000 });
    await real.locator(".yo-slider input").fill("25");
    await expect(real.locator(".yr-cell").nth(0)).toHaveText("1.520");
    await expect(real.locator(".yr-cell").nth(2)).toHaveText("687");
    await expect(real.locator(".yr-ex")).toHaveCount(6);
    await real.locator(".yr-cell").nth(6).click();
    await expect(real.locator("p.yo-caption").first()).toContainText("807 örnek");
    // 4. land cover: one class against the rest has a TN
    await expect(page.locator("#yr-cm-sem .yr-four")).toContainText("TN");
    // the console, results.csv and the curves come from the run's own files
    await expect(page.locator("#yr-console .yr-term").first()).toContainText("100/100");
    await expect(page.locator("#yr-results .yo-caption").first()).toContainText("epoch 88");
    await expect(page.locator("#yr-curves .chart svg path.series-line")).toHaveCount(3);
    // a file of the folder leads to its section
    await page.locator("#yr-folder").getByRole("link", { name: "8. kısım →" }).click();
    await expect(page.locator("#yr-cm-real")).toBeInViewport();
    void errors;
  });

  test("math: eight sections, derivations and widgets that compute", async ({ page, errors }) => {
    await open(page, "yolo.html#matematik");
    await expect(page.locator("#ym-toc li")).toHaveCount(8);
    await expect(page.locator("#ym-main .katex").first()).toBeVisible();
    await expect(page.locator(".katex-error")).toHaveCount(0);
    // the number of candidates follows the input size: 21 S² / 1024
    const sizes = page.locator("#ym-notation .m-widget");
    await expect(sizes).toContainText("8400");
    await sizes.locator("input[type=range]").fill("512");
    await expect(sizes).toContainText("5376");
    // the hand-made PR curve: AP over 101 recall levels
    const pr = page.locator("#ym-ap .m-widget");
    await expect(pr).toContainText("0,6865");
    await pr.locator(".ym-tp").nth(2).click(); // the third detection becomes correct
    await expect(pr).not.toContainText("0,6865");
    // gradient descent on the mask coefficients lowers the loss
    const fit = page.locator("#ym-mask .m-widget");
    await expect(fit).toContainText("0,6931");
    await fit.getByRole("button", { name: "50 adım" }).click();
    await expect(fit).not.toContainText("0,6931");
    void errors;
  });

  test("theory → math: a section's link opens its derivation", async ({ page, errors }) => {
    await open(page, "yolo.html");
    await page.locator("#y-iou .m-theory-link").click();
    await expect(page).toHaveURL(/#matematik\/ym-iou$/);
    await expect(page.locator("#ym-iou")).toBeVisible();
    void errors;
  });

  test("coding: two Python files, each downloadable, with the Colab way for command-line files", async ({ page, errors }) => {
    await open(page, "yolo.html#kodlama");
    await expect(page.locator("#yc-toc li")).toHaveCount(15);
    await expect(page.locator("#yc-main .code-line").first()).toBeVisible();
    await expect(page.locator("#yc-main")).not.toContainText("bu derlemede yok");
    await expect(page.locator("#yc-main")).not.toContainText("Kod parçası bulunamadı");
    await expect(page.locator("#yc-main")).toContainText("def make_datasets(");
    await expect(page.locator("#yc-main")).toContainText("def train(task");
    const boxes = page.locator(".run-box");
    await expect(boxes).toHaveCount(2);
    for (const [k, file, line] of [
      [0, "YOLO_SpaceNet_data.py", "def download_tile("],
      [1, "YOLO_SpaceNet.py", 'if __name__ == "__main__":'],
    ] as const) {
      const box = boxes.nth(k);
      const [download] = await Promise.all([page.waitForEvent("download"), box.locator(".run-download").click()]);
      expect(download.suggestedFilename()).toBe(file);
      expect(readFileSync(await download.path(), "utf8")).toContain(line);
    }
    // Colab: the file is saved with %%writefile and run with !python and its arguments
    const model = boxes.nth(1);
    await model.locator("summary").first().click();
    await expect(model).toContainText("%%writefile YOLO_SpaceNet.py");
    await expect(model).toContainText("!python YOLO_SpaceNet.py train seg");
    await model.locator("summary").nth(2).click();
    await expect(model).toContainText(".venv/bin/python YOLO_SpaceNet.py train seg");
    void errors;
  });

  test("practice: a typical wrong number gets its own feedback, and an experiment runs the real model", async ({ page, errors }) => {
    test.setTimeout(120_000);
    await open(page, "yolo.html#alistirma");
    await expect(page.locator("#px-toc li")).toHaveCount(10);
    await expect(page.locator(".px-count")).toContainText("0 / 37");
    const iou = page.locator('[data-q="i-value"]');
    const input = iou.locator("input");
    await input.fill("0,25");
    await input.press("Enter");
    await expect(iou.locator(".px-feedback")).toContainText("iki kez"); // divided by the sum of the areas
    await input.fill("0,33");
    await input.press("Enter");
    await expect(iou).toHaveAttribute("data-state", "correct");
    // NMS by hand
    const nms = page.locator('[data-q="n-count"]').locator("input");
    await nms.fill("2");
    await nms.press("Enter");
    await expect(page.locator('[data-q="n-count"]')).toHaveAttribute("data-state", "correct");
    // an experiment: without colour the model finds no sports field
    const grey = page.locator('[data-q="e-grey"]');
    await grey.locator(".px-opt").nth(0).click();
    await grey.getByRole("button", { name: "Tahmini sına" }).click();
    await expect(grey.locator(".px-run")).toContainText("gri tonlamada 0 saha", { timeout: 90_000 });
    await expect(grey).toHaveAttribute("data-state", "missed");
    await expect(page.locator(".px-count")).toContainText("Doğru: 2");
    void errors;
  });

  test("fits a phone", async ({ page, errors }) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width: 375, height: 800 });
    await open(page, "yolo.html#tespit");
    await expect(page.locator("#yd-pipeline")).toContainText("8.400", { timeout: 90_000 });
    expect(await overflows(page)).toBe(false);
    await page.locator("#tab-train").click();
    await expect(page.locator(".yo-facts")).toBeVisible();
    expect(await overflows(page)).toBe(false);
    for (const tab of ["theory", "outputs", "math", "code", "practice"]) {
      await page.locator(`#tab-${tab}`).click();
      await page.waitForTimeout(300);
      expect(await overflows(page), tab).toBe(false);
    }
    void errors;
  });
});
