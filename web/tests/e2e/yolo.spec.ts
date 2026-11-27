import { expect, open, overflows, test } from "./fixtures";

/**
 * Chapter 4, YOLO. The first tile is Shanghai_img209; the numbers below are what ONNX Runtime gives in
 * Python on the same tile with the same post-processing (107 candidates above 0.25, then 11 buildings
 * and 4 sports fields), so these tests also check that every browser finds what Python finds.
 */
test.describe("chapter 4: yolo", () => {
  test("detection: both models run in the browser and find what Python finds", async ({ page, errors }) => {
    test.setTimeout(120_000);
    await open(page, "yolo.html");
    await expect(page).toHaveTitle("YOLO · Bölüm 4");
    await expect(page.locator("#tab-detect")).toHaveAttribute("aria-selected", "true");
    const pipeline = page.locator("#yd-pipeline");
    await expect(pipeline).toContainText("8.400 aday", { timeout: 90_000 });
    await expect(pipeline).toContainText("107 güven eşiğini");
    await expect(pipeline).toContainText("15 nesne");
    const found = page.locator("#yd-results tbody tr").first();
    await expect(found).toHaveText(/bulunan\s*11\s*4/);

    // a lower threshold keeps more candidates; NMS removes the duplicates
    await page.locator(".yo-slider input").first().fill("0.05");
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
    await expect(page.locator(".yo-table").first()).toContainText("0,69"); // building mask AP50, pretrained
    const note = page.locator(".yo-training .yo-big").first();
    await expect(note).toContainText("100. epoch");
    await page.getByRole("button", { name: "1", exact: true }).click();
    await expect(note).toContainText("1. epoch");
    await page.getByRole("button", { name: "Sıfırdan (rastgele ağırlıklar)" }).click();
    await expect(note).toContainText("0 bina, 0 spor sahası"); // random weights find nothing after one epoch
    void errors;
  });

  test("fits a phone", async ({ page, errors }) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width: 375, height: 800 });
    await open(page, "yolo.html");
    await expect(page.locator("#yd-pipeline")).toContainText("8.400", { timeout: 90_000 });
    expect(await overflows(page)).toBe(false);
    await page.locator("#tab-train").click();
    await expect(page.locator(".yo-facts")).toBeVisible();
    expect(await overflows(page)).toBe(false);
    void errors;
  });
});
