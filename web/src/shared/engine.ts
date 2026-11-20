/**
 * Where the pages run their computations: in the browser, or on the Python dashboard.
 *
 * The same build serves both. dashboard.py (in the repository root) serves dist/ and marks every
 * page it sends with <html data-engine="server">; then the CNN chapter sends its /api/* requests
 * to that server, which trains with PyTorch. Without the mark (GitHub Pages, a page opened from
 * the disk, `npm run dev`) everything runs in the browser.
 *
 * Text that is true for only one of the two is marked data-engine="browser" or "server" and the
 * other one is hidden by the stylesheet (style.css), so texts need no code to switch.
 */
export type Engine = "browser" | "server";

export const ENGINE: Engine = document.documentElement.dataset.engine === "server" ? "server" : "browser";

export const onServer = ENGINE === "server";
