import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Page } from "@playwright/test";
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
    const planned = page.locator(".lp-chapter.soon");
    await expect(planned).toHaveCount(3);
    await expect(planned.first()).toContainText("RNN");
    await expect(planned.last()).toContainText("Transformer");
    for (const k of [0, 1, 2]) await expect(planned.nth(k)).toContainText("yakında");
    await expect(page.locator("div.lp-chapter a, .lp-chapter.soon[href]")).toHaveCount(0); // not links
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
      await expect(menu.locator(".chm-item")).toHaveCount(6);
      await expect(menu.locator(".chm-item.soon")).toHaveCount(3);
      await expect(menu.locator(".chm-item.soon").first()).toContainText("coming soon");
      const other = CHAPTERS.find((o) => o.id !== c.id)!;
      await expect(menu.getByRole("link", { name: "Home page" })).toHaveAttribute("href", "index.html");
      await menu.locator(`a.chm-item[href="${other.id}.html"]`).click();
      await expect(page).toHaveTitle(new RegExp(`· Chapter ${other.number}$`)); // the language is kept
      void errors;
    });

    test("the house in the top bar leads back to the start page", async ({ page, errors }) => {
      await page.setViewportSize({ width: 375, height: 760 }); // also on a phone, where it takes the logo's place
      await open(page, `${c.id}.html#matematik`);
      const home = page.getByRole("link", { name: "Ana sayfa" }).first();
      await expect(home).toBeVisible();
      await home.click();
      await expect(page.locator(".lp-chapter").first()).toBeVisible();
      expect(new URL(page.url()).pathname).toMatch(/\/index\.html$/);
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

test.describe("perceptron: the Practice tab", () => {
  const question = (page: Page, id: string) => page.locator(`[data-q="${id}"]`);

  test("a wrong choice says why, a right one counts and survives a reload", async ({ page, errors }) => {
    await open(page, "perceptron.html#alistirma");
    const q = question(page, "n-zero");
    await q.getByRole("button", { name: "Kontrol et" }).click();
    await expect(q.locator(".px-feedback")).toContainText("Önce bir seçenek");
    await q.locator(".px-opt").nth(1).click();
    await q.getByRole("button", { name: "Kontrol et" }).click();
    await expect(q.locator(".px-feedback")).toContainText("Henüz değil");
    await expect(q.locator(".px-feedback")).toContainText("eşitlik bu koşula girmez");
    await expect(q.getByRole("button", { name: "Cevabı göster" })).toBeVisible();
    await q.locator(".px-opt").nth(0).click();
    await q.getByRole("button", { name: "Kontrol et" }).click();
    await expect(q.locator(".px-feedback")).toContainText("Doğru.");
    await expect(q.locator(".px-feedback")).toBeFocused();
    await expect(page.locator(".px-count")).toContainText("1 / 15");
    await page.reload();
    await expect(question(page, "n-zero")).toHaveAttribute("data-state", "correct");
    await expect(question(page, "n-zero").locator("input").first()).toBeDisabled();
    await expect(page.locator(".px-count")).toContainText("1 / 15");
    void errors;
  });

  test("a number can be typed with a comma or a dot, and a non-number is not a wrong answer", async ({ page, errors }) => {
    await open(page, "perceptron.html#alistirma");
    const q = question(page, "n-z");
    const input = q.locator("input");
    await input.fill("abc");
    await input.press("Enter");
    await expect(q.locator(".px-feedback")).toContainText("Bir sayı yazın");
    await expect(q.getByRole("button", { name: "Cevabı göster" })).toBeHidden();
    await input.fill("0,4");
    await input.press("Enter");
    await expect(q.locator(".px-feedback")).toContainText("Henüz değil");
    await input.fill("0,3");
    await input.press("Enter");
    await expect(q).toHaveAttribute("data-state", "correct");
    await page.locator('#lang-switch [data-lang="en"]').click();
    await expect(question(page, "n-z").locator(".px-feedback")).toContainText("Correct.");
    const other = question(page, "r-b").locator("input");
    await other.fill("-0.15");
    await other.press("Enter");
    await expect(question(page, "r-b")).toHaveAttribute("data-state", "correct");
    await expect(page.locator(".katex-error")).toHaveCount(0);
    void errors;
  });

  test("a prediction runs the real model, and shows what happened", async ({ page, errors }) => {
    await open(page, "perceptron.html#alistirma/px-converge");
    const q = question(page, "c-flip");
    await q.locator(".px-opt").nth(0).click();
    await q.getByRole("button", { name: "Tahmini sına" }).click();
    await expect(q).toHaveAttribute("data-state", "missed");
    await expect(q.locator(".px-run")).toContainText("200. epoch");
    await expect(q.locator(".px-opt.is-right")).toContainText("Hayır");
    await expect(q.locator(".px-opt.is-wrong")).toContainText("birkaç");
    await expect(page.locator(".px-count")).toContainText("Doğru: 0");
    await q.getByRole("button", { name: "Yeniden çöz" }).click();
    await expect(page.locator(".px-count")).toContainText("0 / 15");
    await expect(q.locator("input").first()).toBeFocused();
    void errors;
  });

  test("showing the answer finishes a question without counting it as correct", async ({ page, errors }) => {
    await open(page, "perceptron.html#alistirma");
    const q = question(page, "b-cross");
    await q.locator("input").fill("5");
    await q.getByRole("button", { name: "Kontrol et" }).click();
    await q.getByRole("button", { name: "Cevabı göster" }).click();
    await expect(q).toHaveAttribute("data-state", "revealed");
    await expect(q.locator(".px-verdict")).toHaveText("Doğru cevap: 1");
    await expect(page.locator(".px-count")).toContainText("1 / 15");
    await expect(page.locator(".px-count")).toContainText("Doğru: 0");
    void errors;
  });

  test("resetting asks first, and a finished section is marked", async ({ page, errors }) => {
    await open(page, "perceptron.html#alistirma");
    for (const [id, pick] of [
      ["n-zero", 0],
      ["n-and", 1],
    ] as const) {
      await question(page, id).locator(".px-opt").nth(pick).click();
      await question(page, id).getByRole("button", { name: "Kontrol et" }).click();
    }
    await question(page, "n-z").locator("input").fill("0.3");
    await question(page, "n-z").getByRole("button", { name: "Kontrol et" }).click();
    await expect(page.locator('#px-toc [data-step="px-neuron"]')).toHaveClass(/px-done/);
    await page.getByRole("button", { name: "İlerlemeyi sıfırla" }).click();
    await page.getByRole("button", { name: "Vazgeç" }).click();
    await expect(page.locator(".px-count")).toContainText("3 / 15");
    await page.getByRole("button", { name: "İlerlemeyi sıfırla" }).click();
    await page.getByRole("button", { name: "Evet, sıfırla" }).click();
    await expect(page.locator(".px-count")).toContainText("0 / 15");
    await expect(question(page, "n-zero")).toHaveAttribute("data-state", "");
    await expect(page.locator('#px-toc [data-step="px-neuron"]')).not.toHaveClass(/px-done/);
    void errors;
  });

  test("the start page shows the chapter's progress", async ({ page, errors }) => {
    await open(page, "perceptron.html#alistirma");
    await question(page, "n-zero").locator(".px-opt").nth(0).click();
    await question(page, "n-zero").getByRole("button", { name: "Kontrol et" }).click();
    await expect(page.locator(".px-count")).toContainText("1 / 15");
    await open(page, "index.html");
    await expect(page.locator("a.lp-chapter").first()).toContainText("Alıştırma 1/15");
    await expect(page.locator("a.lp-chapter").nth(1)).not.toContainText("Alıştırma");
    void errors;
  });

  test("fits a phone, with every tab within reach", async ({ page, errors }) => {
    await page.setViewportSize({ width: 320, height: 700 });
    await open(page, "perceptron.html#alistirma");
    const tabsFit = () => page.locator(".tabs").evaluate((el) => el.scrollWidth <= el.clientWidth + 1);
    await expect(page.locator("#px-main .px-q").first()).toBeVisible();
    expect(await overflows(page)).toBe(false);
    expect(await tabsFit(), "all five tabs are visible").toBe(true);
    await page.locator('#lang-switch [data-lang="en"]').click();
    await expect(page.locator("#px-main .px-q").first()).toBeVisible();
    expect(await overflows(page)).toBe(false);
    expect(await tabsFit(), "all five tabs are visible").toBe(true);
    await page.locator("#tab-theory").click();
    await page.locator("#tab-practice").click();
    await expect(page.locator("#panel-practice")).toBeVisible();
    void errors;
  });
});

test.describe("mlp: the Practice tab", () => {
  const question = (page: Page, id: string) => page.locator(`[data-q="${id}"]`);

  test("takes a rounded number, runs a real prediction and keeps its own progress", async ({ page, errors }) => {
    await open(page, "mlp.html#alistirma");
    await expect(page.locator(".px-count")).toContainText("0 / 19");
    await expect(page.locator("#px-toc button")).toHaveCount(6);
    const rounded = question(page, "l-bce");
    await rounded.locator("input").fill("0,8");
    await rounded.locator("input").press("Enter");
    await expect(rounded.locator(".px-feedback")).toContainText("Henüz değil");
    await rounded.locator("input").fill("0,693");
    await rounded.locator("input").press("Enter");
    await expect(rounded).toHaveAttribute("data-state", "correct");
    const predict = question(page, "bp-sym");
    await predict.locator(".px-opt").nth(1).click();
    await predict.getByRole("button", { name: "Tahmini sına" }).click();
    await expect(predict).toHaveAttribute("data-state", "correct");
    await expect(predict.locator(".px-run")).toContainText("birebir aynı");
    await expect(page.locator(".px-count")).toContainText("2 / 19");
    await expect(page.locator(".katex-error")).toHaveCount(0);
    await open(page, "perceptron.html#alistirma");
    await expect(page.locator(".px-count")).toContainText("0 / 15");
    await open(page, "index.html");
    await expect(page.locator("a.lp-chapter").nth(1)).toContainText("Alıştırma 2/19");
    await expect(page.locator("a.lp-chapter").first()).not.toContainText("Alıştırma");
    void errors;
  });

  test("every prediction runs, in English too", async ({ page, errors }) => {
    await page.addInitScript(() => localStorage.setItem("lang", "en"));
    await open(page, "mlp.html#alistirma");
    for (const id of ["h-xor", "l-mse", "bp-sym", "i-shift"]) {
      await question(page, id).locator(".px-opt").first().click();
      await question(page, id).getByRole("button", { name: "Test my prediction" }).click();
      await expect(question(page, id).locator(".px-run")).toBeVisible();
    }
    await expect(page.locator(".px-count")).toContainText("4 / 19");
    await expect(page.locator(".katex-error")).toHaveCount(0);
    void errors;
  });

  test("fits a phone, with every tab visible", async ({ page, errors }) => {
    await page.setViewportSize({ width: 320, height: 700 });
    await open(page, "mlp.html#alistirma");
    await expect(page.locator("#px-main .px-q").first()).toBeVisible();
    expect(await overflows(page)).toBe(false);
    expect(await page.locator(".tabs").evaluate((el) => el.scrollWidth <= el.clientWidth + 1), "all five tabs are visible").toBe(true);
    void errors;
  });
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
