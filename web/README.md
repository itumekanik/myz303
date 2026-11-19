# AI learning path: the static site

The learning path of this repository (perceptron → MLP → CNN → …) as a static web site. It needs no
server and no Python: everything, including training the networks, runs in the browser.

- **Online:** published on GitHub Pages by [.github/workflows/web.yml](../.github/workflows/web.yml).
- **Offline:** download `ai-learning-path.zip` from the start page, unpack it and double-click
  `index.html`. The same files work from the disk (`file://`) as from the web.

Chapters 1 (perceptron) and 2 (MLP) are here. Chapter 3 (CNN) is listed as "coming soon" until it
moves over from the Python dashboard (`dashboard.py`).

## For students

Open the site (or `index.html` of the downloaded copy) in a current Chrome, Edge, Firefox or Safari.
Pick a chapter on the start page; the title at the top left of every chapter opens the learning path.
The language (TR/EN) and theme switches are at the top right, the glossary next to them.

## Development

Requires Node.js 22.12 or newer.

```bash
cd web
npm install
npm run dev          # http://127.0.0.1:5173/, rebuilds and reloads on every change
npm run build        # the site in dist/ (open dist/index.html, or `npm run preview`)
npm run check        # what CI runs: lint, typecheck, unit tests, build, end-to-end tests
```

| Command | What it does |
|---|---|
| `npm run lint` | ESLint and the Prettier format check |
| `npm run format` | formats the TypeScript, CSS, scripts and tests with Prettier |
| `npm run typecheck` | TypeScript (`tsc --noEmit`) |
| `npm test` | unit tests (`node:test`): the MLP engine's gradients, chapter discovery |
| `npm run test:e2e` | Playwright: every page, both served over http and opened from `file://` |
| `npm run preview` | serves `dist/` at http://127.0.0.1:4173/ |

The end-to-end tests use the installed Google Chrome locally and Playwright's Chromium on CI
(`npx playwright install chromium` once, if Chrome is not installed).

## How it works

```
src/
  app/            the page shell (shell.html), the start page (landing.html, landing.ts)
  chapters/<id>/  one folder per chapter (see below)
  engines/        net.js: a small MLP with backpropagation, SGD and Adam (types in net.d.ts)
  shared/         what every chapter page uses: i18n, core (navigation, charts, chapter menu),
                  kit (planes, graphs, controls), lesson (Math and Coding tabs), codeview,
                  glossary/, math (KaTeX), switches (theme, language), page (tabs and hooks)
  styles/         app.css = KaTeX + style.css + landing.css
scripts/          build.mjs, dev.mjs, serve.mjs, chapters.mjs (chapter discovery)
tests/            unit/ (node:test) and e2e/ (Playwright)
```

**Opening from the disk.** Browsers refuse `fetch()`, `<script type="module" src>`, workers from
files and font files on a page opened from `file://`. So `scripts/build.mjs` (esbuild) bundles every
page into one classic script (IIFE), bundles the Python sources of the Coding tabs as text, and
inlines KaTeX's fonts into the stylesheet. File names carry a content hash, so GitHub Pages never
serves a stale script. Larger data (the CNN chapter's weights and MNIST) will follow the same rule:
JavaScript files loaded with a `<script>` tag when they are needed.

**Chapters.** `src/chapters/<id>/chapter.json` describes a chapter (see
[chapter.schema.json](src/chapters/chapter.schema.json)): its place on the path (`order`), title,
year, and `status`. A `planned` chapter appears as "coming soon". A `ready` chapter also has:

- `main.js`: the page's entry point, importing its modules in start-up order;
- `panels.html`: the tab panels inside `<main>`;
- `logo.svg`: the logo in the top bar.

The build finds the folders, numbers the chapters, generates `<id>.html` from `src/app/shell.html`
and gives every page the chapter list (the chapter menu, the start page, glossary links). Adding a
folder is all it takes to add a chapter.

**Ported code.** The chapter content and the shared UI came from the Python dashboard's `static/`
folder as JavaScript modules with explicit imports. They are still JavaScript (`allowJs`, not type
checked); new code is TypeScript in strict mode. Code that runs early but needs a module loaded
later (core.js → the glossary panel) goes through `shared/page.ts` hooks, so the module graph has no
cycles.

## Licenses

KaTeX (MIT) is bundled into the stylesheet and the page scripts.
