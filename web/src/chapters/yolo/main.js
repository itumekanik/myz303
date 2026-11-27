/*
 * Entry point of the yolo.html page (bundled by scripts/build.mjs into one classic script).
 * The order of the imports is the order in which the modules start.
 */
import "../../shared/core.js";
import "./page.ts";
import "./theory.ts";
import "./detect.ts";
import "./training.ts";
import "./mathtab.ts";
import "./codetab.ts";
import "./practice.ts";
import "../../shared/glossary/panel.js";
import "../../shared/math.js";
