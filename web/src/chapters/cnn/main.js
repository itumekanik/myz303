/*
 * Entry point of the cnn.html page (bundled by scripts/build.mjs into one classic script).
 * The order of the imports is the order in which the modules start, as the <script> tags did in
 * the Python dashboard's index.html. The dashboard's server is replaced by engines/cnn/backend.ts.
 */
import "../../shared/core.js";
import "./app.js";
import "./theory.js";
import "./tutorial.js";
import "./mathtab.js";
import "./practice.js";
import "../../shared/glossary/panel.js";
import "./help.js";
import "../../shared/math.js";
