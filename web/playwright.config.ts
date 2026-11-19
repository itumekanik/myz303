/**
 * End-to-end tests of the built site (run `npm run build` first). Every test runs twice:
 * - "http": dist/ served over HTTP, as GitHub Pages serves it;
 * - "file": dist/ opened straight from the disk (file://), as a student who double-clicks index.html.
 * Locally the installed Google Chrome is used; on CI, Playwright's own Chromium.
 */
import { defineConfig, devices } from "@playwright/test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const CI = Boolean(process.env.CI);
const PORT = 4173;
const browser = CI ? {} : { channel: "chrome" };

export default defineConfig({
  testDir: "tests/e2e",
  fullyParallel: true,
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  reporter: CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { trace: "retain-on-failure" },
  webServer: {
    command: `node scripts/serve.mjs dist ${PORT}`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !CI,
  },
  projects: [
    { name: "http", use: { ...devices["Desktop Chrome"], ...browser, baseURL: `http://127.0.0.1:${PORT}/` } },
    { name: "file", use: { ...devices["Desktop Chrome"], ...browser, baseURL: `${pathToFileURL(resolve("dist")).href}/` } },
  ],
});
