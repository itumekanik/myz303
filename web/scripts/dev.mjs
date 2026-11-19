/**
 * Development mode: builds dist/ without minifying, serves it at http://127.0.0.1:5173/ and
 * rebuilds on every change; the open page reloads itself.
 *
 *   npm run dev            (PORT=8080 npm run dev for another port)
 *
 * Scripts, styles and the Python sources are watched by esbuild. The page templates (*.html) and
 * logos (*.svg) are watched here. A changed chapter.json needs a restart (the chapter list is
 * read once, at start-up).
 */
import * as esbuild from "esbuild";
import { mkdirSync, rmSync, watch } from "node:fs";
import { join } from "node:path";
import { DIST, SRC, esbuildOptions, writePages } from "./build.mjs";
import { loadChapters } from "./chapters.mjs";

const PORT = Number(process.env.PORT ?? 5173);
const chapters = loadChapters(join(SRC, "chapters"));
rmSync(DIST, { recursive: true, force: true });
mkdirSync(DIST, { recursive: true });

const pages = {
  name: "pages",
  setup(build) {
    build.onEnd((result) => {
      if (!result.metafile) return; // the build failed: esbuild has printed why
      try {
        writePages({ chapters, metafile: result.metafile, dev: true });
      } catch (e) {
        console.error(e.message);
      }
    });
  },
};

const ctx = await esbuild.context(esbuildOptions({ dev: true, chapters, plugins: [pages] }));
await ctx.watch();
const { port } = await ctx.serve({ servedir: DIST, host: "127.0.0.1", port: PORT });
console.log(`Serving dist/ at http://127.0.0.1:${port}/ (Ctrl+C to stop)`);

watch(SRC, { recursive: true }, (_event, file) => {
  if (!file) return;
  if (/\.(html|svg)$/.test(file)) ctx.rebuild().catch(() => {});
  else if (file.endsWith("chapter.json")) console.log(`${file} changed: restart \`npm run dev\` to read the chapter list again.`);
});
