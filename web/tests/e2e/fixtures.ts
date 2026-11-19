import { test as base, expect, type Page } from "@playwright/test";

/** Every test fails if the page logs an error or throws. */
export const test = base.extend<{ errors: string[] }>({
  errors: async ({ page }, use) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(`console: ${m.text()}`);
    });
    await use(errors);
    expect(errors, "console errors and exceptions").toEqual([]);
  },
});

export { expect };

/** Opens a page of the site (relative to dist/), works for both http:// and file://. */
export async function open(page: Page, path: string): Promise<void> {
  await page.goto(path);
  await page.waitForLoadState("load");
}

/** True when the page is wider than the window (a horizontal scroll bar). */
export const overflows = (page: Page): Promise<boolean> =>
  page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
