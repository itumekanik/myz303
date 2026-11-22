import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, open, overflows, test } from "./fixtures";

const CHAPTERS = [
  {
    id: "perceptron",
    number: 1,
    title: "Perceptron",
    sections: ["p-intro", "p-neuron", "p-boundary", "p-rule", "p-converge", "p-xor"],
    math: "#pm",
    code: "#pc",
    python: "class Perceptron",
  },
  {
    id: "mlp",
    number: 2,
    title: "Çok katmanlı ağ",
    sections: ["ml-intro", "ml-act", "ml-hidden", "ml-loss", "ml-backprop", "ml-capacity", "ml-images"],
    math: "#mm",
    code: "#mc",
    python: "class MLP",
  },
];

test.describe("start page", () => {
  test("lists the learning path and links every chapter", async ({ page, errors }) => {
    await open(page, "index.html");
    await expect(page.locator("a.lp-chapter")).toHaveCount(3);
    await expect(page.locator("a.lp-chapter").first()).toHaveAttribute("href", "perceptron.html");
    await expect(page.locator("a.lp-chapter").last()).toHaveAttribute("href", "cnn.html");
    await expect(page.locator(".lp-chapter.soon")).toHaveCount(0);
    await page.locator("a.lp-chapter").first().click();
    await expect(page).toHaveTitle("Perceptron · Bölüm 1");
    void errors;
  });

  test("shows the version from CHANGELOG.md, also in the chapter menu", async ({ page, errors }) => {
    const { version } = JSON.parse(readFileSync(join(import.meta.dirname, "../../package.json"), "utf8")) as { version: string };
    await open(page, "index.html");
    const line = page.locator(".lp-version");
    await expect(line).toContainText(`Sürüm ${version} · `);
    await expect(line.locator("a")).toHaveAttribute("href", /CHANGELOG\.md$/);
    await page.locator('#lang-switch [data-lang="en"]').click();
    await expect(line).toContainText(`Version ${version} · `);
    await expect(line.locator("a")).toHaveText("What's new?");
    await page.locator('#lang-switch [data-lang="tr"]').click();
    await open(page, "perceptron.html");
    await page.locator("#chapter-btn").click();
    await expect(page.locator(".chm-version")).toContainText(`Sürüm ${version}`);
    void errors;
  });

  test("a malformed address opens the first tab", async ({ page, errors }) => {
    await open(page, "mlp.html#%E0%A4%A");
    await expect(page.locator("#tab-theory")).toHaveAttribute("aria-selected", "true");
    await expect(page.locator("body")).toHaveAttribute("data-tab", "theory");
    void errors;
  });

  test("a chapter opened in English has its glossary button in English", async ({ page, errors }) => {
    await page.addInitScript(() => localStorage.setItem("lang", "en"));
    await open(page, "perceptron.html");
    await expect(page.locator(".glossary-btn-text")).toHaveText("Glossary");
    void errors;
  });

  test("switches to English and back", async ({ page, errors }) => {
    await open(page, "index.html");
    await page.locator('#lang-switch [data-lang="en"]').click();
    await expect(page.locator(".brand-title")).toHaveText("AI learning path");
    await expect(page.locator(".lp-path h2")).toHaveText("Learning path");
    await page.locator('#lang-switch [data-lang="tr"]').click();
    await expect(page.locator(".brand-title")).toHaveText("Yapay zekâ öğrenme yolu");
    void errors;
  });
});

for (const c of CHAPTERS) {
  test.describe(`chapter ${c.number}: ${c.id}`, () => {
    test("theory: every section, typeset math, chapter title", async ({ page, errors }) => {
      await open(page, `${c.id}.html`);
      await expect(page).toHaveTitle(`${c.id === "mlp" ? "MLP" : c.title} · Bölüm ${c.number}`);
      await expect(page.locator(".brand-sub")).toHaveText(new RegExp(`^Bölüm ${c.number} · `));
      for (const id of c.sections) await expect(page.locator(`#${id}`)).toBeVisible();
      await expect(page.locator("#panel-theory .katex").first()).toBeVisible();
      await expect(page.locator(".katex-error")).toHaveCount(0);
      void errors;
    });

    test("math and coding tabs render, Python source is bundled", async ({ page, errors }) => {
      await open(page, `${c.id}.html`);
      await page.locator("#tab-math").click();
      await expect(page).toHaveURL(/#matematik$/);
      await expect(page.locator(`${c.math}-toc li`).first()).toBeVisible();
      await expect(page.locator(`${c.math}-main .katex`).first()).toBeVisible();
      await expect(page.locator(".katex-error")).toHaveCount(0);
      await page.locator("#tab-code").click();
      await expect(page.locator(`${c.code}-main .code-line`).first()).toBeVisible();
      await expect(page.locator(`${c.code}-main`)).toContainText(c.python);
      await expect(page.locator(`${c.code}-main`)).not.toContainText("bu derlemede yok");
      void errors;
    });

    test("playground tab opens", async ({ page, errors }) => {
      await open(page, `${c.id}.html#deneme`);
      await expect(page.locator("#panel-play")).toBeVisible();
      await expect(page.locator("#panel-play canvas").first()).toBeVisible();
      void errors;
    });

    test("glossary opens and its links lead to the CNN chapter", async ({ page, errors }) => {
      await open(page, `${c.id}.html`);
      await page.locator("#glossary-btn").click();
      await expect(page.locator("#glossary")).toBeVisible();
      expect(await page.locator("#glossary .g-entry").count()).toBeGreaterThan(50);
      await page.evaluate(() => document.querySelector("#g-dropout")?.scrollIntoView());
      await expect(page.locator("#g-dropout .g-place.soon")).toHaveCount(0);
      await page.locator("#g-dropout a.g-place").first().click();
      await expect(page).toHaveURL(/cnn\.html#/);
      await expect(page.locator("#t-drop, #m-drop").first()).toBeVisible();
      void errors;
    });

    test("English, and the chapter menu leads to the other chapters", async ({ page, errors }) => {
      await open(page, `${c.id}.html`);
      await page.locator('#lang-switch [data-lang="en"]').click();
      await expect(page).toHaveTitle(new RegExp(`· Chapter ${c.number}$`));
      await expect(page.locator(".brand-sub")).toHaveText(new RegExp(`^Chapter ${c.number} · `));
      await page.locator("#chapter-btn").click();
      const menu = page.locator("#chapter-menu");
      await expect(menu).toBeVisible();
      await expect(menu.locator(".chm-item")).toHaveCount(3);
      await expect(menu.locator(".chm-item.soon")).toHaveCount(0);
      const other = CHAPTERS.find((o) => o.id !== c.id)!;
      await menu.locator(`a.chm-item[href="${other.id}.html"]`).click();
      await expect(page).toHaveTitle(new RegExp(`· Chapter ${other.number}$`)); // the language is kept
      void errors;
    });

    test("fits a phone screen", async ({ page, errors }) => {
      await page.setViewportSize({ width: 375, height: 760 });
      await open(page, `${c.id}.html`);
      expect(await overflows(page)).toBe(false);
      for (const tab of ["play", "math", "code"]) {
        await page.locator(`#tab-${tab}`).click();
        await page.waitForTimeout(300);
        expect(await overflows(page), tab).toBe(false);
      }
      void errors;
    });
  });
}

test("perceptron: the learning rule demo steps through the examples", async ({ page, errors }) => {
  await open(page, "perceptron.html#teori/p-rule");
  const demo = page.locator("#p-rule");
  const before = await demo.locator(".pc-msg").textContent();
  await demo.getByRole("button", { name: "Sonraki örnek" }).click();
  await expect(demo.locator(".pc-msg")).not.toHaveText(before ?? "");
  void errors;
});

test("mlp: the hidden layer learns XOR in the browser", async ({ page, errors }) => {
  await open(page, "mlp.html#teori/ml-hidden");
  const demo = page.locator("#ml-hidden");
  await demo.getByRole("button", { name: "▶ Eğit" }).click();
  await expect(demo.locator(".pc-msg")).toContainText("Çözüldü", { timeout: 20_000 });
  void errors;
});

test("focusing a control of the sticky top bar does not move the page", async ({ page, errors }) => {
  await open(page, "perceptron.html#matematik");
  await expect(page.locator("#panel-math .katex").first()).toBeVisible();
  await page.evaluate(() => window.scrollTo({ top: 900, behavior: "instant" }));
  await page.locator("#glossary-btn").evaluate((el: HTMLElement) => el.focus());
  await page.waitForTimeout(200);
  expect(await page.evaluate(() => Math.round(window.scrollY))).toBe(900);
  void errors;
});

test("perceptron: Regenerate keeps the points placed by hand", async ({ page, errors }) => {
  await open(page, "perceptron.html#deneme");
  const plane = page.locator("#panel-play canvas.pg-canvas");
  await plane.click({ position: { x: 40, y: 40 } });
  const status = page.locator("#panel-play .pc-msg").first();
  const before = await status.innerText();
  await page.getByRole("button", { name: "Yeniden üret" }).click();
  await expect(status).toHaveText(before);
  void errors;
});

test("mlp: the loss demo does not call the step limit a minimum", async ({ page, errors }) => {
  await open(page, "mlp.html");
  const demo = page.locator("#ml-loss");
  await demo.getByRole("button", { name: "MSE" }).click();
  await demo
    .locator('input[type="range"]')
    .first()
    .evaluate((el: HTMLInputElement) => {
      el.value = "0.1";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
  await demo.getByRole("button", { name: "Bir adım" }).evaluate((el: HTMLButtonElement) => {
    for (let i = 0; i < 200; i++) el.click();
  });
  await expect(demo.locator(".pc-msg").first()).toContainText("minimuma henüz varmadı");
  void errors;
});

test("perceptron → mlp: the next-chapter card opens the MLP's theory", async ({ page, errors }) => {
  await open(page, "perceptron.html#teori/p-xor");
  await page.locator("#p-xor a.pc-next").click();
  await expect(page).toHaveURL(/mlp\.html#teori$/);
  await expect(page.locator("#tab-theory")).toHaveAttribute("aria-selected", "true");
  void errors;
});

test("mlp → cnn: the next-chapter card opens the CNN theory", async ({ page, errors }) => {
  await open(page, "mlp.html#teori/ml-images");
  await page.locator("#ml-images a.pc-next").click();
  await expect(page).toHaveURL(/cnn\.html#teori$/);
  await expect(page.locator("#tab-theory")).toHaveAttribute("aria-selected", "true");
  void errors;
});

test("mlp: a See also link opens the CNN Math tab at that section", async ({ page, errors }) => {
  await open(page, "mlp.html#matematik");
  await expect(page.locator("#mm-main .m-link-soon")).toHaveCount(0);
  await page.locator('#mm-main a[data-chapter="cnn"][data-go="math:m-softmax"]').first().click();
  await expect(page).toHaveURL(/cnn\.html#matematik\/m-softmax$/);
  await expect(page.locator("#m-softmax")).toBeVisible();
  void errors;
});
