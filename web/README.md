# AI learning path: the static site

The learning path of this repository (perceptron → MLP → CNN → …) as a static web site. It needs no
server and no Python: everything, including training the networks, runs in the browser.

- **Online:** published on GitHub Pages by [.github/workflows/web.yml](../.github/workflows/web.yml).
- **Offline:** download `ai-learning-path.zip` from the start page, unpack it and double-click
  `index.html`. The same files work from the disk (`file://`) as from the web.

All three chapters are here: 1 the perceptron, 2 the multilayer network (MLP), 3 the convolutional
network (CNN, SimpleCNN on MNIST).

## For students

Open the site (or `index.html` of the downloaded copy) in a current Chrome, Edge, Firefox or Safari.
Pick a chapter on the start page; the title at the top left of every chapter opens the learning path.
The language (TR/EN) and theme switches are at the top right, the glossary next to them.

In the CNN chapter, the Inference and Theory tabs use a model that comes with the site (trained with
`SimpleCNN.py` in PyTorch, 99.29% test accuracy). The Training tab trains SimpleCNN again in your
browser, on your GPU: about 35–40 seconds per epoch on a gaming GPU, a few minutes per epoch on a
laptop's integrated GPU (lower the number of epochs to try it quickly). On a laptop with two GPUs,
choose "High performance" for the browser in the operating system's graphics settings. The model you
train is kept in this browser until you go back to the one that comes with the site.

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
| `npm test` | unit tests (`node:test`): the MLP engine's gradients, SimpleCNN's forward pass against PyTorch, training settings, chapter discovery |
| `npm run test:e2e` | Playwright: every page, both served over http and opened from `file://` (and a short training run on TensorFlow.js) |
| `npm run preview` | serves `dist/` at http://127.0.0.1:4173/ |

The end-to-end tests use the installed Google Chrome locally and Playwright's Chromium on CI
(`npx playwright install chromium` once, if Chrome is not installed).

### The CNN chapter's data

`public/data/` holds what the CNN chapter needs from the Python side, written by
[tools/export_assets.py](tools/export_assets.py) from the dashboard's checkpoint and MNIST download:

| File | Content | Loaded |
|---|---|---|
| `cnn-trained.js` | the trained weights (float32, PyTorch layout) and the run's settings | when the page opens |
| `cnn-run.js` | that run: curves, filters, sample predictions, log | when the page opens |
| `cnn-initial.js` | the random weights that run started from ("before training") | when that model is chosen |
| `mnist-test.js` | the 10,000 test images | for test images (Inference, Theory) |
| `mnist-train.js` | the 60,000 training images, and which 12,000 are the validation set (the same split as `SimpleCNN.py`) | when a training run starts |

After training a new model with the Python dashboard, refresh them (from the repository root):

```bash
.venv/Scripts/python.exe web/tools/export_assets.py      # Windows; .venv/bin/python on macOS / Linux
```

The script also writes `tests/fixtures/cnn-parity.json`, PyTorch's outputs at every layer for a few
test images, which the unit tests compare with the TypeScript forward pass.

## How it works

```
src/
  app/            the page shell (shell.html), the start page (landing.html, landing.ts)
  chapters/<id>/  one folder per chapter (see below)
  engines/        net.js: a small MLP with backpropagation, SGD and Adam (chapters 1–2)
    cnn/          SimpleCNN in the browser (chapter 3):
                    forward.ts   the forward pass in plain TypeScript (Inference, Theory)
                    backend.ts   dashboard.py's API (/api/status, /api/predict, …) answered locally
                    session.ts   a training run on TensorFlow.js (WebGPU, else WebGL)
                    store.ts     the browser's own model (IndexedDB)
                    data.ts, weights.ts, drawing.ts, tf.ts
  shared/         what every chapter page uses: i18n, core (navigation, charts, chapter menu),
                  kit (planes, graphs, controls), lesson (Math and Coding tabs), codeview,
                  glossary/, math (KaTeX), switches (theme, language), page (tabs and hooks)
  styles/         app.css = KaTeX + style.css + landing.css
public/data/      the CNN chapter's data files (tools/export_assets.py)
scripts/          build.mjs, dev.mjs, serve.mjs, chapters.mjs (chapter discovery)
tools/            export_assets.py (Python: checkpoint and MNIST → public/data/)
tests/            unit/ (node:test), e2e/ (Playwright), fixtures/
```

**Opening from the disk.** Browsers refuse `fetch()`, `<script type="module" src>`, workers from
files and font files on a page opened from `file://`. So `scripts/build.mjs` (esbuild) bundles every
page into one classic script (IIFE), bundles the Python sources of the Coding tabs as text, and
inlines KaTeX's fonts into the stylesheet. File names carry a content hash, so GitHub Pages never
serves a stale script. Data files are scripts too (`window.__AI_DATA__[name] = …`, binary arrays as
base64, gzip-compressed where it helps) loaded with a `<script>` tag when they are needed, and
TensorFlow.js is loaded the same way (from `vendor/`) only when a training run starts.

**The CNN chapter without a server.** Its page code (`chapters/cnn/app.js`, `theory.js`) was
written for the Python dashboard's API. `engines/cnn/backend.ts` answers the same requests with the
same JSON, so the page works as before: predictions come from the TypeScript forward pass (a few
milliseconds per image, the same numbers as PyTorch to 1e-3), training from TensorFlow.js. A run in
the browser uses `SimpleCNN.py`'s settings, validation split, initialisation, optimizer and layers,
but another random generator, so it gets close to PyTorch's results rather than repeating them.

**Chapters.** `src/chapters/<id>/chapter.json` describes a chapter (see
[chapter.schema.json](src/chapters/chapter.schema.json)): its place on the path (`order`), title,
year, and `status`. A `planned` chapter appears as "coming soon". A `ready` chapter also has:

- `main.js`: the page's entry point, importing its modules in start-up order;
- `panels.html`: the tab panels inside `<main>`;
- `logo.svg`: the logo in the top bar;
- optionally `badges.html` (more badges in the top bar), `before-main.html` and `after-main.html`.

The build finds the folders, numbers the chapters, generates `<id>.html` from `src/app/shell.html`
and gives every page the chapter list (the chapter menu, the start page, glossary links). Adding a
folder is all it takes to add a chapter.

**Ported code.** The chapter content and the shared UI came from the Python dashboard's `static/`
folder as JavaScript modules with explicit imports. They are still JavaScript (`allowJs`, not type
checked); new code is TypeScript in strict mode (erasable syntax only, so Node runs the unit tests
without a build step). Code that runs early but needs a module loaded later (core.js → the glossary
panel) goes through `shared/page.ts` hooks, so the module graph has no cycles.

## Licenses

KaTeX (MIT) is bundled into the stylesheet and the page scripts. TensorFlow.js (Apache 2.0) is
copied to `vendor/`. MNIST (Yann LeCun, Corinna Cortes, Christopher Burges; CC BY-SA 3.0) is in
`public/data/` as the original files.
