/**
 * Builds the static site into dist/: index.html (the start page), one <id>.html per ready chapter
 * and assets/ (one classic script per page, one stylesheet).
 *
 * Every page must also work when opened straight from the disk (file://), where browsers refuse
 * fetch(), <script type="module" src> and font files. So each page gets a single classic IIFE
 * script, the Python sources of the Coding tabs are bundled as text, and KaTeX's fonts are inlined
 * into the stylesheet. The same dist/ is what GitHub Pages serves.
 *
 *   node scripts/build.mjs            production build (minified, hashed file names)
 *   import { createBuild } ...        used by scripts/dev.mjs for the watch mode
 */
import * as esbuild from "esbuild";
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { loadChapters, runtimeList } from "./chapters.mjs";
import { readRelease } from "./release.mjs";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const SRC = join(ROOT, "src");
export const DIST = join(ROOT, "dist");

const TAB_LABELS = { train: "Eğitim", infer: "Çıkarım", theory: "Teori", play: "Deneme alanı", math: "Matematik", code: "Kodlama" };
const FAVICON =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232a78d6'/%3E%3Cg stroke='%23fff' stroke-width='2' fill='none'%3E%3Cpath d='M6 10l10 6M6 22l10-6M16 16h9'/%3E%3C/g%3E%3Ccircle cx='16' cy='16' r='4.5' fill='%23fff'/%3E%3C/svg%3E";
/**
 * In watch mode the pages reload themselves after every rebuild (esbuild's serve mode), except
 * when dashboard.py serves the same dist/ folder (it marks the page data-engine="server").
 */
const LIVE_RELOAD = `<script>if (!document.documentElement.dataset.engine) new EventSource("/esbuild").addEventListener("change", () => location.reload());</script>`;

const read = (path) => readFileSync(path, "utf8").replace(/\r\n/g, "\n");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const indent = (text, n) =>
  text
    .trimEnd()
    .split("\n")
    .map((line) => (line ? " ".repeat(n) + line : line))
    .join("\n");

/** An optional part of a chapter page (badges.html, before-main.html, after-main.html), or nothing. */
const optional = (dir, file, n) => (existsSync(join(dir, file)) ? indent(read(join(dir, file)), n) : "");

/** Fills {{name}} placeholders; an unknown placeholder is a build error. */
function fill(template, values, name) {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => {
    if (!(key in values)) throw new Error(`${name}: no value for {{${key}}}`);
    return values[key];
  });
}

/** KaTeX lists woff2, woff and ttf for every font; every browser we support reads woff2, so only it is inlined. */
const katexWoff2Only = {
  name: "katex-woff2-only",
  setup(build) {
    build.onLoad({ filter: /katex(\.min)?\.css$/ }, (args) => ({
      contents: read(args.path)
        .replace(/,url\([^)]*\.woff\) format\("woff"\)/g, "")
        .replace(/,url\([^)]*\.ttf\) format\("truetype"\)/g, ""),
      loader: "css",
      resolveDir: dirname(args.path),
    }));
  },
};

export function esbuildOptions({ dev, chapters, outdir = DIST, plugins = [] }) {
  const entryPoints = { landing: join(SRC, "app/landing.ts"), app: join(SRC, "styles/app.css") };
  for (const c of chapters) if (c.url) entryPoints[c.id] = join(c.dir, "main.js");
  return {
    absWorkingDir: ROOT,
    entryPoints,
    outdir: join(outdir, "assets"),
    entryNames: dev ? "[name]" : "[name]-[hash]",
    bundle: true,
    format: "iife",
    platform: "browser",
    target: ["es2022"],
    charset: "utf8",
    minify: !dev,
    sourcemap: dev ? "inline" : "linked",
    legalComments: "eof",
    metafile: true,
    logLevel: "warning",
    define: { __CHAPTERS__: JSON.stringify(runtimeList(chapters)), __RELEASE__: JSON.stringify(readRelease(ROOT)) },
    loader: { ".py": "text", ".woff2": "dataurl" },
    plugins: [katexWoff2Only, ...plugins],
  };
}

/** Which file in assets/ each entry point became. */
function assetNames(metafile) {
  const names = {};
  for (const [out, info] of Object.entries(metafile.outputs)) {
    if (!info.entryPoint || out.endsWith(".map")) continue;
    const name = out.split("/").pop();
    const entry = info.entryPoint.replace(/\\/g, "/");
    if (entry.endsWith("styles/app.css")) names.css = name;
    else if (entry.endsWith("app/landing.ts")) names.landing = name;
    else names[entry.split("/").at(-2)] = name; // src/chapters/<id>/main.js
  }
  return names;
}

/** Writes index.html and one page per ready chapter. */
export function writePages({ chapters, metafile, dev, outdir = DIST }) {
  const assets = assetNames(metafile);
  const styles = `<link rel="stylesheet" href="assets/${assets.css}">`;
  const script = (file) => `<script src="assets/${file}"></script>` + (dev ? `\n  ${LIVE_RELOAD}` : "");
  const shell = read(join(SRC, "app/shell.html"));
  for (const c of chapters.filter((ch) => ch.url)) {
    const p = c.page;
    const brand = p.brand ?? c.title;
    const tabs = p.tabs
      .map(
        (t, i) =>
          `      <button type="button" role="tab" id="tab-${t}" aria-controls="panel-${t}" aria-selected="${i === 0}" data-i18n="tab.${t}">${TAB_LABELS[t]}</button>`,
      )
      .join("\n");
    const html = fill(
      shell,
      {
        titleTr: esc(`${p.title.tr} · Bölüm ${c.number}`),
        titleEn: esc(`${p.title.en} · Chapter ${c.number}`),
        favicon: FAVICON,
        id: c.id,
        logo: read(join(c.dir, "logo.svg")).trim(),
        brandTitle: esc(brand.tr),
        brandTitleEn: esc(brand.en),
        brandSub: esc(`Bölüm ${c.number} · ${c.sub.tr}`),
        brandSubEn: esc(`Chapter ${c.number} · ${c.sub.en}`),
        tabs,
        panels: indent(read(join(c.dir, "panels.html")), 4),
        badges: optional(c.dir, "badges.html", 6),
        beforeMain: optional(c.dir, "before-main.html", 2),
        afterMain: optional(c.dir, "after-main.html", 2),
        styles,
        scripts: script(assets[c.id]),
      },
      `${c.id}.html`,
    );
    writeFileSync(join(outdir, c.url), html);
  }
  const landing = fill(read(join(SRC, "app/landing.html")), { favicon: FAVICON, styles, scripts: script(assets.landing) }, "index.html");
  writeFileSync(join(outdir, "index.html"), landing);
  writeFileSync(join(outdir, ".nojekyll"), ""); // GitHub Pages: serve the files as they are
}

/** Libraries a page loads only when it needs them, with <script> tags (not bundled). */
const VENDOR = {
  "tf.min.js": "node_modules/@tensorflow/tfjs/dist/tf.min.js",
  "tf-backend-webgpu.min.js": "node_modules/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.min.js",
};

/** Copies public/ (the data files of tools/export_assets.py) and the vendor libraries into the site. */
export function copyStatic(outdir = DIST) {
  if (existsSync(join(ROOT, "public"))) cpSync(join(ROOT, "public"), outdir, { recursive: true });
  mkdirSync(join(outdir, "vendor"), { recursive: true });
  for (const [name, from] of Object.entries(VENDOR)) cpSync(join(ROOT, from), join(outdir, "vendor", name));
}

export async function build({ dev = false, outdir = DIST } = {}) {
  const chapters = loadChapters(join(SRC, "chapters"));
  rmSync(outdir, { recursive: true, force: true });
  mkdirSync(outdir, { recursive: true });
  const result = await esbuild.build(esbuildOptions({ dev, chapters, outdir }));
  writePages({ chapters, metafile: result.metafile, dev, outdir });
  copyStatic(outdir);
  return { chapters, result };
}

// Run as a script: production build
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const started = performance.now();
  const { chapters, result } = await build();
  const sizes = Object.entries(result.metafile.outputs)
    .filter(([out]) => !out.endsWith(".map"))
    .map(
      ([out, info]) =>
        `  ${relative(ROOT, join(ROOT, out)).replace(/\\/g, "/").padEnd(40)} ${(info.bytes / 1024).toFixed(0).padStart(6)} KB`,
    );
  console.log(sizes.join("\n"));
  const ready = chapters.filter((c) => c.url).map((c) => c.url);
  console.log(
    `dist/: index.html, ${ready.join(", ")} (${chapters.length - ready.length} chapter(s) still to come) in ${Math.round(performance.now() - started)} ms`,
  );
}
