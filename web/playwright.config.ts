/**
 * End-to-end tests of the built site (run `npm run build` first). Every browser is tried both ways:
 * - "http": dist/ served over HTTP, as GitHub Pages serves it;
 * - "file": dist/ opened straight from the disk (file://), as a student who double-clicks index.html.
 * The phone projects emulate an Android phone and an iPhone over HTTP.
 * Locally the installed Google Chrome is used; on CI, Playwright's own Chromium.
 * WebKit here is Playwright's build, close to Safari but not Safari itself.
 */
import { defineConfig, devices } from "@playwright/test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const CI = Boolean(process.env.CI);
const PORT = 4173;
const browser = CI ? {} : { channel: "chrome" };
const HTTP = `http://127.0.0.1:${PORT}/`;
const FILE = `${pathToFileURL(resolve("dist")).href}/`;

export default defineConfig({
  testDir: "tests/e2e",
  fullyParallel: true,
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  reporter: CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { trace: "retain-on-failure" },
  webServer: {
    command: `node scripts/serve.mjs dist ${PORT}`,
    url: HTTP,
    reuseExistingServer: !CI,
  },
  projects: [
    { name: "http", use: { ...devices["Desktop Chrome"], ...browser, baseURL: HTTP } },
    { name: "file", use: { ...devices["Desktop Chrome"], ...browser, baseURL: FILE } },
    { name: "firefox-http", use: { ...devices["Desktop Firefox"], baseURL: HTTP } },
    { name: "firefox-file", use: { ...devices["Desktop Firefox"], baseURL: FILE } },
    { name: "webkit-http", use: { ...devices["Desktop Safari"], baseURL: HTTP } },
    { name: "webkit-file", use: { ...devices["Desktop Safari"], baseURL: FILE } },
    { name: "android", use: { ...devices["Pixel 7"], ...browser, baseURL: HTTP } },
    { name: "iphone", use: { ...devices["iPhone 15"], baseURL: HTTP } },
  ],
});
