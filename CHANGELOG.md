# Changelog

What changed in the AI learning path (the site at https://itumekanik.github.io/myz303/, its offline
copy and the Python dashboard). The newest release is first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/): a new chapter or feature raises the second number, a
fix the third.

The site shows the newest version here at the bottom of the start page and of the chapter menu (the
page title, top left). To release one, add a section below with the version and the date, set the
same version in `web/package.json`, and push to `main`. The build fails if the two differ.

## [1.0.0] - 2026-11-20

The first release for students: the whole learning path works from GitHub Pages, from the
downloaded copy and from the Python dashboard, all built from one source (`web/`).

### Added

- The version and the date at the bottom of the start page and of the chapter menu, with a link
  to this list of changes.
- This changelog.

### Changed

- The Python dashboard (`dashboard.py`, `start.bat`) serves the same pages as the site. In the CNN
  chapter, training on the dashboard runs in PyTorch on this computer and is saved to
  `checkpoints/simple_cnn.pt`. Everywhere else it runs in the browser. Texts that hold for only
  one of the two (the run banner, the info popups, a few notes and glossary entries) are shown
  only where they apply.
- The old addresses `/perceptron` and `/mlp` lead to `perceptron.html` and `mlp.html`.
- `start.bat` builds the pages the first time it runs, if Node.js is installed.

### Removed

- `static/`, the dashboard's own copy of the pages.

## [0.4.0] - 2026-11-19

The learning path as a static site that needs no server and no Python.

### Added

- The site at https://itumekanik.github.io/myz303/, published by GitHub Actions after every
  change. A ZIP of the site can be downloaded from the start page; unpacked, it opens by
  double-clicking `index.html`, without internet.
- A start page: what the course is and the chapters of the learning path.
- The CNN chapter in the browser: inference and the Theory tab use SimpleCNN trained in PyTorch
  (99.29% test accuracy), and the Training tab trains it again on the student's GPU with
  TensorFlow.js. The trained model is kept in the browser until the student goes back to the one
  that comes with the site.

## [0.3.0] - 2026-11-19

The dashboard becomes a course: three chapters in historical order.

### Added

- Chapter 1, the perceptron (1958): theory with interactive demos, a playground, the math
  (including Novikoff's convergence proof) and `Perceptron.py` written from scratch.
- Chapter 2, the multilayer network (1986): XOR in the hidden space, the loss surface,
  backpropagation step by step, fitting a curve, a playground with 0 to 3 hidden layers, the
  math of backpropagation and `MLP.py` with a gradient check.
- The chapter menu behind the page title, and glossary terms for the new chapters.
- Learning goals at the start of every chapter, a recap at the end, and self-check questions in
  the CNN chapter.

### Changed

- The CNN chapter builds on chapters 1 and 2 instead of starting from zero.
- Exercises run from easy to hard, with hints.

### Fixed

- A teaching review corrected several claims, among them: XOR's stuck runs sit on a flat region,
  not a local minimum; 2×2 pooling is robust to about one pixel, not shift invariant; dropout
  prevents co-adaptation. Two broken exercises and the perceptron's ×2 button were fixed.

## [0.2.0] - 2026-11-17

The CNN dashboard explains itself.

### Added

- The Math tab: the mathematics of every SimpleCNN layer, with derivations revealed step by step
  and seven interactive examples.
- A glossary of key terms with cards and a searchable side panel.
- Info popups on the Training tab, with live values from the current settings.
- BatchNorm explained in six steps on Conv1's real output.
- The optimizers (SGD, Momentum, Adam) on a 3D loss surface.
- Formulas typeset with KaTeX.
- The last training run is kept across restarts of the dashboard.

### Changed

- The pages work from 320 px phones to wide screens.
- Back and Forward work across tabs and sections, and the address keeps the section.

## [0.1.0] - 2026-11-12

The first version: a dashboard (`dashboard.py`) for SimpleCNN on MNIST.

### Added

- Training, Inference and Theory tabs, with the network's layers, filters and maps shown for a
  drawn or chosen digit.
- The Coding tab: a step-by-step tutorial on `SimpleCNN.py`.
- A switch between the model before and after training.
- Turkish and English, a light and a dark theme, and `start.bat` to start it on Windows.
