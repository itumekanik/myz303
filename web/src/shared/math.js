/*
 * Math typesetting with KaTeX (the katex npm package).
 *
 * Texts write formulas in LaTeX between \( and \). They are typeset once at start-up and
 * then whenever new text appears on the page (language change, popups, live demos): a
 * MutationObserver sees the added text and typesets just that part of the page.
 *
 * In Turkish, numbers use a decimal comma; texNum() wraps it as {,} so TeX does not add
 * the space it puts after a punctuation comma.
 */
import renderMathInElement from "katex/contrib/auto-render";


export const MATH_OPTIONS = {
  delimiters: [{ left: "\\(", right: "\\)", display: false }],
  throwOnError: false,   // a typo shows the source in red instead of breaking the page
  strict: "ignore",      // Turkish letters inside \text{} are fine
};

/** A formatted number, ready to go inside a TeX formula. */
export const texNum = (text) => String(text).replace(/,/g, "{,}");

export function typesetMath(root) {
  renderMathInElement(root, MATH_OPTIONS);
}

export const hasMath = (node) => (node.nodeType === Node.TEXT_NODE ? node.data : node.textContent || "").includes("\\(");

export const mathObserver = new MutationObserver((records) => {
  const roots = new Set();
  for (const record of records) {
    for (const node of record.addedNodes) {
      if (hasMath(node)) roots.add(node.nodeType === Node.TEXT_NODE ? record.target : node);
    }
  }
  for (const root of roots) {
    // Skip a root inside another one, it is typeset with it
    if (root.isConnected && ![...roots].some((other) => other !== root && other.contains(root))) typesetMath(root);
  }
});

typesetMath(document.body);
mathObserver.observe(document.body, { childList: true, subtree: true });
