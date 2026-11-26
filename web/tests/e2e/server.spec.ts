/**
 * The pages as dashboard.py serves them: marked <html data-engine="server">, with the CNN page
 * sending its /api/* requests to the server (which trains with PyTorch) instead of answering them
 * in the browser. The server is stood in for by Playwright: it marks the pages the way
 * dashboard.py does and answers /api/status with the bundled PyTorch run.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Page } from "@playwright/test";
import { expect, open, test } from "./fixtures";

const DATA = join(import.meta.dirname, "../../dist/data");

/** The run that trained the bundled model (data/cnn-run.js), as dashboard.py's /api/status sends it. */
function status(): Record<string, unknown> {
  const script = readFileSync(join(DATA, "cnn-run.js"), "utf8");
  const json = script.slice(script.indexOf("] = ") + 4).replace(/;\s*$/, "");
  const run = JSON.parse(json) as Record<string, unknown>;
  return {
    ...run,
    run_id: 1,
    state: "done",
    stage: "done",
    running: false,
    message: "",
    elapsed: 196,
    restored: { trained_at: run.trained_at },
    model: { ready: true, source: "Kayıtlı model", meta: { test_acc: 0.9929, epochs: 10 }, version: 1, initial_seed: 42 },
    defaults: { epochs: 10, lr: 0.001, batch_size: 256, seed: 42 },
    device: "GPU · NVIDIA GeForce RTX 2060",
    server_version: 5,
  };
}

/** Serves the site as dashboard.py does; returns the bodies of the POST requests to /api/train. */
async function asDashboard(page: Page): Promise<unknown[]> {
  const trained: unknown[] = [];
  await page.route(/\.html(\?.*)?$|\/$/, async (route) => {
    const response = await route.fetch();
    const body = (await response.text()).replace("<html", '<html data-engine="server"');
    await route.fulfill({ response, body });
  });
  await page.route("**/api/status", (route) => route.fulfill({ json: status() }));
  await page.route("**/api/train", (route) => {
    trained.push(route.request().postDataJSON());
    return route.fulfill({ json: { ok: true } });
  });
  return trained;
}

test.describe("served by dashboard.py", () => {
  test.skip(({ baseURL }) => !baseURL?.startsWith("http"), "dashboard.py serves over http");

  test("the CNN page shows the server's run and texts for training with PyTorch", async ({ page, errors }) => {
    await asDashboard(page);
    await open(page, "cnn.html#egitim");
    await expect(page.locator("#run-banner")).toContainText("Son eğitim yüklendi");
    await expect(page.locator("#run-banner")).not.toContainText("Siteyle gelen");
    await expect(page.locator("#run-banner button")).toHaveCount(1); // "Try inference", no "back to the site's model"
    await expect(page.locator('[data-i18n="train.defaults"]')).toHaveText(/PyTorch ile çalışır/);
    await expect(page.locator('[data-stage="saving"] [data-hint]')).toHaveText("checkpoints/simple_cnn.pt");
    await page.locator('button[data-info="stage.saving"]').click();
    await expect(page.getByText("checkpoints/simple_cnn.pt dosyasına kaydedilir")).toBeVisible();
    await expect(page.getByText("bu tarayıcıda", { exact: false }).filter({ visible: true })).toHaveCount(0);
    void errors;
  });

  test("a training run is started on the server, not in the browser", async ({ page, errors }) => {
    const trained = await asDashboard(page);
    const tf: string[] = [];
    page.on("request", (r) => {
      if (/\/(vendor|data)\//.test(r.url())) tf.push(r.url());
    });
    await open(page, "cnn.html#egitim");
    await expect(page.locator("#state-pill")).toContainText("Tamamlandı");
    await page.fill('input[name="epochs"]', "2");
    await page.locator("#btn-start").click();
    await expect.poll(() => trained.length).toBe(1);
    expect(trained[0]).toMatchObject({ epochs: 2, batch_size: 256, seed: 42 });
    expect(tf, "TensorFlow.js and the data files are not loaded").toEqual([]);
    void errors;
  });

  test("a value typed before the first status arrives is kept", async ({ page, errors }) => {
    await asDashboard(page);
    let answer!: () => void;
    const held = new Promise<void>((resolve) => (answer = resolve));
    await page.route("**/api/status", async (route) => {
      await held;
      await route.fulfill({ json: status() });
    });
    await open(page, "cnn.html#egitim");
    await page.fill('input[name="epochs"]', "2");
    answer();
    await expect(page.locator("#state-pill")).toContainText("Tamamlandı");
    await expect(page.locator('input[name="epochs"]')).toHaveValue("2");
    await expect(page.locator('input[name="batch_size"]')).toHaveValue("256");
    void errors;
  });

  test("a stopped server is reported with how to restart it", async ({ page }) => {
    await asDashboard(page);
    await page.route("**/api/status", (route) => route.abort());
    await open(page, "cnn.html#egitim");
    await expect(page.locator("#conn-banner")).toBeVisible();
    // innerText: only the text shown (the banner holds both versions, one of them hidden)
    const shown = await page.locator("#conn-banner").innerText();
    expect(shown).toContain("Sunucuya ulaşılamıyor");
    expect(shown).not.toContain("data/");
  });

  test("the start page says the pages come from the Python dashboard", async ({ page, errors }) => {
    await asDashboard(page);
    await open(page, "index.html");
    await expect(page.locator(".lp-offline h2")).toHaveText("Python paneli");
    void errors;
  });
});
