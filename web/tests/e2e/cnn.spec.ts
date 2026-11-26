import type { Page } from "@playwright/test";
import { expect, open, overflows, test } from "./fixtures";

test.describe("chapter 3: cnn", () => {
  test("opens on the Theory tab, and the Training tab then draws its charts", async ({ page, errors }) => {
    await open(page, "cnn.html");
    await expect(page.locator("#tab-theory")).toHaveAttribute("aria-selected", "true");
    await expect(page.locator(".tabs [role=tab]").first()).toHaveId("tab-theory");
    await expect(page.locator("#panel-theory")).toBeVisible();
    await expect(page.locator("#panel-train")).toBeHidden();
    await page.locator("#tab-train").click();
    await expect(page).toHaveURL(/#egitim$/);
    for (const chart of ["#chart-batch", "#chart-loss", "#chart-acc"]) {
      const drawn = page.locator(`${chart} svg`);
      await expect(drawn).toBeVisible();
      expect(Number(await drawn.getAttribute("width")), chart).toBeGreaterThan(100);
      expect(await page.locator(`${chart} svg path, ${chart} svg polyline`).count(), chart).toBeGreaterThan(0);
    }
    void errors;
  });

  test("training tab shows the run that trained the bundled model", async ({ page, errors }) => {
    await open(page, "cnn.html#egitim");
    await expect(page).toHaveTitle("CNN · Bölüm 3");
    await expect(page.locator("#run-banner")).toContainText("PyTorch");
    await expect(page.locator("#model-badge-text")).toContainText("99,29");
    await expect(page.locator("#log")).toContainText("Epoch 10");
    await expect(page.locator("#state-pill")).toContainText("Tamamlandı");
    await expect(page.locator("#conn-banner")).toBeHidden();
    void errors;
  });

  test("inference: a test image and a drawing are classified in the browser", async ({ page, errors }) => {
    await open(page, "cnn.html#cikarim");
    const first = page.locator("#test-samples > *").first();
    await expect(first).toBeVisible();
    await first.click();
    await expect(page.locator("#pred-digit")).toHaveText(/^\d$/);
    await expect(page.locator("#pred-conf")).toContainText("%");
    await expect(page.locator("#layers canvas").nth(100)).toBeAttached();

    // draw a "1": a vertical stroke
    await page.locator("#btn-clear").click();
    const box = (await page.locator("#draw").boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.15);
    await page.mouse.down();
    for (let k = 1; k <= 12; k++) await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * (0.15 + 0.06 * k));
    await page.mouse.up();
    await expect(page.locator("#pred-digit")).toHaveText("1", { timeout: 10_000 });
    void errors;
  });

  test("inference: the model before training is loaded on demand", async ({ page, errors }) => {
    await open(page, "cnn.html#cikarim");
    await page.locator("#test-samples > *").first().click();
    await expect(page.locator("#pred-digit")).toHaveText(/^\d$/);
    await page.locator('.model-switch button[data-model="initial"]').click();
    await expect(page.locator("#pred-model")).toContainText("öncesi");
    await expect(page.locator("#pred-conf")).toContainText("%");
    void errors;
  });

  test("theory, math and coding tabs render; SimpleCNN.py is bundled", async ({ page, errors }) => {
    await open(page, "cnn.html#teori");
    await expect(page.locator("#t-conv")).toBeVisible();
    await expect(page.locator("#panel-theory canvas").nth(30)).toBeAttached(); // waits until the demos are drawn
    await page.locator("#tab-math").click();
    await expect(page.locator("#m-softmax")).toBeAttached();
    await expect(page.locator("#panel-math .katex").first()).toBeVisible();
    await page.locator("#tab-code").click();
    await expect(page.locator("#panel-code")).toContainText("class SimpleCNN");
    await expect(page.locator(".katex-error")).toHaveCount(0);
    void errors;
  });

  test("Theory, Math and Practice teach the smallest CNN first, then Dropout and BatchNorm", async ({ page, errors }) => {
    const ids = (selector: string) => page.locator(selector).evaluateAll((els) => els.map((el) => el.id));
    await open(page, "cnn.html#teori");
    expect(await ids("#panel-theory article.t-section")).toEqual(["t-conv", "t-act", "t-pool", "t-softmax", "t-opt", "t-drop", "t-bn"]);
    expect(await page.locator("#panel-theory article.t-section .t-num").allTextContents()).toEqual(["1", "2", "3", "4", "5", "6", "7"]);
    await expect(page.locator("#t-drop .t-bridge")).toContainText("İlk beş kısım");
    await page.locator("#tab-math").click();
    await expect(page.locator("#m-bn")).toBeAttached();
    expect(await ids("#panel-math article.m-step")).toEqual([
      "m-notation",
      "m-conv",
      "m-act",
      "m-pool",
      "m-fc",
      "m-softmax",
      "m-backprop",
      "m-opt",
      "m-drop",
      "m-bn",
    ]);
    await expect(page.locator("#m-bn")).toContainText("Özet: bu sekmede ne öğrendik?");
    await page.locator("#tab-practice").click();
    await expect(page.locator("#px-bn")).toBeAttached();
    expect(await ids("#panel-practice article.tut-step")).toEqual([
      "px-conv",
      "px-act",
      "px-pool",
      "px-softmax",
      "px-opt",
      "px-drop",
      "px-bn",
      "px-transfer",
      "px-model",
    ]);
    void errors;
  });

  test("fits a phone screen", async ({ page, errors }) => {
    await page.setViewportSize({ width: 375, height: 760 });
    await open(page, "cnn.html");
    for (const tab of ["theory", "infer", "train", "math", "code"]) {
      await page.locator(`#tab-${tab}`).click();
      await page.waitForTimeout(400);
      expect(await overflows(page), tab).toBe(false);
    }
    void errors;
  });
});

test.describe("chapter 3: the Practice tab", () => {
  const question = (page: Page, id: string) => page.locator(`[data-q="${id}"]`);
  const predict = async (page: Page, id: string, option: number) => {
    await question(page, id).locator(".px-opt").nth(option).click();
    await question(page, id).locator("button[type=submit]").click();
  };

  test("a question on the real model runs it, and the result survives a reload", async ({ page, errors }) => {
    await open(page, "cnn.html#alistirma");
    await expect(page.locator(".px-count")).toContainText("0 / 25");
    await expect(page.locator("#px-toc button")).toHaveCount(9);
    const number = question(page, "cv-size");
    await number.locator("input").fill("16");
    await number.locator("input").press("Enter");
    await expect(number).toHaveAttribute("data-state", "correct");
    await predict(page, "e-black", 1);
    const black = question(page, "e-black");
    await expect(black).toHaveAttribute("data-state", "correct", { timeout: 30_000 });
    await expect(black.locator(".px-run")).toContainText('%97,4 olasılıkla "1"');
    await page.reload();
    await expect(question(page, "e-black").locator(".px-run")).toContainText('%97,4 olasılıkla "1"');
    await expect(page.locator(".px-count")).toContainText("2 / 25");
    await expect(page.locator(".katex-error")).toHaveCount(0);
    void errors;
  });

  test("the shift experiments show what the model really does", async ({ page, errors }) => {
    test.slow();
    await open(page, "cnn.html#alistirma/px-model");
    await predict(page, "e-shift", 2);
    await expect(question(page, "e-shift")).toHaveAttribute("data-state", "correct", { timeout: 60_000 });
    await expect(question(page, "e-shift").locator(".px-run")).toContainText("40 rakamın 39 tanesini");
    await predict(page, "e-far", 0);
    await expect(question(page, "e-far")).toHaveAttribute("data-state", "missed", { timeout: 60_000 });
    await expect(question(page, "e-far").locator(".px-run")).toContainText("40 rakamın 13 tanesini");
    void errors;
  });

  test("a run in progress survives a language change, and a reset cancels it", async ({ page, errors }) => {
    test.slow();
    await open(page, "cnn.html#alistirma/px-model");
    await predict(page, "e-far", 2);
    await expect(question(page, "e-far").locator(".px-feedback")).toContainText("Model çalışıyor");
    await expect(question(page, "e-far").locator("input").first()).toBeDisabled();
    await page.locator('#lang-switch [data-lang="en"]').click();
    await expect(question(page, "e-far")).toHaveAttribute("data-state", "correct", { timeout: 60_000 });
    await expect(question(page, "e-far").locator(".px-run")).toContainText("13 of the 40");
    await predict(page, "e-shift", 2);
    await page.getByRole("button", { name: "Reset progress" }).click();
    await page.getByRole("button", { name: "Yes, reset" }).click();
    await page.waitForTimeout(4000);
    await expect(question(page, "e-shift")).toHaveAttribute("data-state", "");
    await expect(page.locator(".px-count")).toContainText("0 / 25");
    void errors;
  });

  test("1024 written with a thousands point counts, and the answer shown can be typed back", async ({ page, errors }) => {
    await open(page, "cnn.html#alistirma/px-pool");
    const flat = question(page, "pl-flat");
    await flat.locator("input").fill("1.024");
    await flat.locator("input").press("Enter");
    await expect(flat).toHaveAttribute("data-state", "correct");
    await flat.getByRole("button", { name: /Yeniden/ }).click();
    await flat.locator("input").fill("1000");
    await flat.locator("input").press("Enter");
    await flat.getByRole("button", { name: /Cevabı göster/ }).click();
    await expect(flat.locator(".px-verdict")).toHaveText("Doğru cevap: 1024");
    void errors;
  });

  test("a damaged saved model result does not break the tab", async ({ page, errors }) => {
    const damaged = { cnn: { total: 20, answers: { "e-black": { status: "correct", pick: 1, run: { answer: 1, values: [] } } } } };
    await page.addInitScript((saved) => localStorage.setItem("practice-v1", saved), JSON.stringify(damaged));
    await open(page, "cnn.html#alistirma/px-model");
    await expect(question(page, "e-black")).toHaveAttribute("data-state", "correct");
    await expect(page.locator("#px-main .px-q")).toHaveCount(25);
    await expect(page.locator(".px-count")).toContainText("1 / 25");
    void errors;
  });

  test("a model file that fails to load can be tried again", async ({ page, baseURL }) => {
    test.skip(!baseURL?.startsWith("http"), "requests can be blocked over http only");
    await page.route("**/data/mnist-test.js", (route) => route.abort());
    await open(page, "cnn.html#alistirma/px-model");
    await predict(page, "e-black", 1);
    await expect(question(page, "e-black").locator(".px-feedback")).toContainText("yüklenemedi", { timeout: 30_000 });
    await page.unroute("**/data/mnist-test.js");
    await question(page, "e-black").locator("button[type=submit]").click();
    await expect(question(page, "e-black")).toHaveAttribute("data-state", "correct", { timeout: 30_000 });
  });

  test("fits a phone, with all six tabs visible", async ({ page, errors }) => {
    await page.setViewportSize({ width: 320, height: 700 });
    await open(page, "cnn.html#alistirma");
    await expect(page.locator("#px-main .px-q").first()).toBeVisible();
    expect(await overflows(page)).toBe(false);
    expect(await page.locator(".tabs").evaluate((el) => el.scrollWidth <= el.clientWidth + 1), "all six tabs are visible").toBe(true);
    void errors;
  });
});

test.describe("chapter 3: training in the browser", () => {
  // One project is enough (TensorFlow.js loading from file:// is the harder case); slow without a GPU
  test.skip(({ baseURL }) => !baseURL?.startsWith("file:"), "runs in the file:// project only");
  test.slow();

  test("a run starts on TensorFlow.js, reports batches and can be stopped", async ({ page, errors }) => {
    await open(page, "cnn.html#egitim");
    await page.fill('input[name="epochs"]', "1");
    await page.locator("#btn-start").click();
    // only the new run's log names TensorFlow.js (the bundled PyTorch run is shown before it)
    await expect(page.locator("#log")).toContainText("TensorFlow.js", { timeout: 60_000 });
    await expect(page.locator("#step-count")).toHaveText(/: [1-9]/, { timeout: 120_000 });
    await page.locator("#btn-stop").click();
    await expect(page.locator("#state-pill")).toContainText("Durduruldu", { timeout: 120_000 });
    await expect(page.locator("#log")).toContainText("stopped by the user");
    void errors;
  });
});
