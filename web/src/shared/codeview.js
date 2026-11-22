/*
 * Code display shared by the Coding tabs: a small Python syntax highlighter, an editor-like
 * code box with numbered notes and a copy button, and a terminal window.
 * Uses h() and navigate() from core.js and L() from i18n.js.
 */
// =====================================================================
// Python syntax highlighting (small tokenizer, VS Code "Dark+" colors)
// =====================================================================
import { L } from "./i18n.js";
import { h, navigate } from "./core.js";

export const PY_CONTROL = new Set(["if", "elif", "else", "for", "while", "return", "with", "try", "except", "finally",
  "raise", "import", "from", "as", "pass", "break", "continue", "yield", "in", "not", "and", "or", "is",
  "assert", "del", "global", "nonlocal", "async", "await"]);
export const PY_STORAGE = new Set(["def", "class", "lambda"]);
export const PY_CONST = new Set(["True", "False", "None"]);
export const PY_BUILTIN = new Set(["print", "range", "len", "int", "float", "str", "list", "dict", "tuple", "set",
  "super", "sum", "min", "max", "abs", "round", "type", "next", "iter", "isinstance", "enumerate", "zip", "open"]);

/** Splits one line into [text, class] tokens. `state` carries open triple-quoted strings and bracket depth. */
export function tokenizePython(text, state) {
  const out = [];
  const push = (s, cls) => { if (s) out.push([s, cls]); };
  let i = 0;
  if (state.triple) {
    const end = text.indexOf(state.triple);
    if (end === -1) return [[text, "tok-doc"]];
    push(text.slice(0, end + 3), "tok-doc");
    i = end + 3;
    state.triple = null;
  }
  let previous = "";
  while (i < text.length) {
    const rest = text.slice(i);
    let m;
    if (rest[0] === "#") { push(rest, "tok-com"); break; }
    if ((m = rest.match(/^[rRbBuUfF]{0,2}("""|''')/))) {
      const end = rest.indexOf(m[1], m[0].length);
      if (end === -1) { push(rest, "tok-doc"); state.triple = m[1]; break; }
      push(rest.slice(0, end + 3), "tok-doc");
      i += end + 3;
      continue;
    }
    if ((m = rest.match(/^([rRbBuUfF]{0,2})(["'])/))) {
      let j = m[0].length;
      while (j < rest.length && rest[j] !== m[2]) j += rest[j] === "\\" ? 2 : 1;
      const str = rest.slice(0, j + 1);
      if (/f/i.test(m[1])) {
        // f-string: the {expressions} inside keep normal colors
        for (const part of str.split(/(\{[^{}]*\})/)) {
          if (/^\{[^{}]*\}$/.test(part)) { push("{", "tok-brace"); push(part.slice(1, -1), "tok-var"); push("}", "tok-brace"); }
          else push(part, "tok-str");
        }
      } else {
        push(str, "tok-str");
      }
      i += str.length;
      continue;
    }
    if ((m = rest.match(/^\d[\d_]*(\.\d+)?([eE][+-]?\d+)?/)) && !/\w/.test(text[i - 1] || "")) {
      push(m[0], "tok-num");
      i += m[0].length;
      continue;
    }
    if ((m = rest.match(/^@[\w.]+/))) { push(m[0], "tok-dec"); i += m[0].length; continue; }
    if ((m = rest.match(/^[A-Za-z_]\w*/))) {
      const word = m[0];
      const after = rest.slice(word.length);
      let cls = "tok-var";
      if (previous === "def") cls = "tok-fn";
      else if (previous === "class") cls = "tok-cls";
      else if (PY_STORAGE.has(word) || PY_CONST.has(word) || word === "self") cls = "tok-def";
      else if (PY_CONTROL.has(word)) cls = "tok-kw";
      else if (/^[A-Z]/.test(word) && /[a-z]/.test(word)) cls = "tok-cls";        // CamelCase: a class
      else if (state.depth > 0 && /^\s*=(?!=)/.test(after)) cls = "tok-param";   // keyword argument
      else if (/^\s*\(/.test(after)) cls = PY_BUILTIN.has(word) ? "tok-bi" : "tok-fn";   // a call
      push(word, cls);
      previous = PY_STORAGE.has(word) ? word : "";
      i += word.length;
      continue;
    }
    const ch = rest[0];
    if ("([{".includes(ch)) state.depth++;
    if (")]}".includes(ch)) state.depth = Math.max(0, state.depth - 1);
    push(ch, /[=+\-*/<>%!&|^~]/.test(ch) ? "tok-op" : null);
    i++;
  }
  return out;
}

export function highlightInto(node, text, state, lang) {
  if (lang !== "python") {
    node.textContent = text;
    return;
  }
  for (const [part, cls] of tokenizePython(text, state)) {
    node.append(cls ? h("span", { class: cls, text: part }) : document.createTextNode(part));
  }
}

// =====================================================================
// Rich text: `code`, **bold**, *italic*, [[tab:section|link text]]
// =====================================================================
/** `text` may be a plain string or a { tr, en } pair (resolved with L()). */
export function rich(text) {
  text = L(text);
  const fragment = document.createDocumentFragment();
  for (const part of text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\])/)) {
    if (!part) continue;
    if (part.startsWith("`")) fragment.append(h("code", { class: "ic", text: part.slice(1, -1) }));
    else if (part.startsWith("**")) fragment.append(h("strong", { text: part.slice(2, -2) }));
    else if (part.startsWith("[[")) {
      const [target, label] = part.slice(2, -2).split("|");
      const [tab, section] = target.split(":");
      const link = h("a", { href: "#", class: "tut-link", text: label });
      link.addEventListener("click", (e) => {
        e.preventDefault();
        navigate(tab, section || null);
      });
      fragment.append(link);
    } else if (part.startsWith("*") && part.endsWith("*") && part.length > 2) fragment.append(h("em", { text: part.slice(1, -1) }));
    else fragment.append(part);
  }
  return fragment;
}


// =====================================================================
// Code editor component
// =====================================================================
/**
 * lines: [{ n, text }]; options: { file, lang, notes: [{ at, text }], fold: true (hide the first docstring),
 * caption, tall }. A note's `at` is the line's text (exact match first, then substring).
 * Note `at`/`text` and `caption` may be { tr, en } pairs; they are resolved with L() here.
 */
export function codeEditor(lines, options) {
  const lang = options.lang || "python";
  const state = { triple: null, depth: 0 };

  // Notes: find their lines and number them in reading order
  const notes = (options.notes || []).map((note) => {
    const at = L(note.at);
    let index = lines.findIndex((l) => l.text.trim() === at.trim());
    if (index === -1) index = lines.findIndex((l) => l.text.includes(at));
    if (index === -1) console.warn("Tutorial note not found:", at);
    return { ...note, index };
  }).filter((note) => note.index >= 0).sort((a, b) => a.index - b.index);
  notes.forEach((note, k) => { note.mark = String(k + 1); });

  // A docstring right after the def/class line can be folded away
  let foldStart = -1;
  let foldEnd = -1;
  if (options.fold) {
    foldStart = lines.findIndex((l, k) => k > 0 && /^\s*(r?""")/.test(l.text));
    if (foldStart > 0) {
      const opener = lines[foldStart].text.trim();
      foldEnd = opener.length > 3 && opener.endsWith('"""') ? foldStart
        : lines.findIndex((l, k) => k > foldStart && l.text.includes('"""'));
    }
    if (foldEnd < foldStart) foldStart = -1;
  }

  const body = h("div", { class: "code-body", tabindex: 0, role: "group", "aria-label": L({ tr: "Kod", en: "Code" }) });   // scrolls sideways with the keyboard too
  const rows = [];
  let foldGroup = null;
  lines.forEach((line, k) => {
    const row = h("div", { class: "code-line" });
    const gutter = h("span", { class: "code-gutter", text: line.n != null ? String(line.n) : "" });
    const src = h("span", { class: "code-src" });
    highlightInto(src, line.text || " ", state, lang);
    row.append(gutter, src);
    const note = notes.find((nt) => nt.index === k);
    if (note) {
      row.classList.add("has-note");
      row.append(h("span", { class: "code-mark", text: note.mark, "aria-hidden": "true" }));
    }
    rows[k] = row;
    if (k === foldStart) {
      // Fold bar standing in for the hidden docstring lines
      const hidden = foldEnd - foldStart + 1;
      const foldText = (open) => (open
        ? L({ tr: `"""…"""  açıklama metnini gizle`, en: `"""…"""  hide the docstring` })
        : L({
          tr: `"""…"""  açıklama metni (docstring) · ${hidden} satır · göstermek için tıklayın`,
          en: `"""…"""  docstring · ${hidden} ${hidden === 1 ? "line" : "lines"} · click to show`,
        }));
      const bar = h("button", { type: "button", class: "code-fold", "aria-expanded": "false" },
        h("span", { class: "code-gutter", text: "⋯" }),
        h("span", { class: "code-fold-text", text: foldText(false) }));
      foldGroup = h("div", { class: "code-fold-lines", hidden: true });
      bar.addEventListener("click", () => {
        const open = foldGroup.hidden;
        foldGroup.hidden = !open;
        bar.setAttribute("aria-expanded", String(open));
        bar.querySelector(".code-fold-text").textContent = foldText(open);
      });
      body.append(bar, foldGroup);
    }
    if (foldGroup && k >= foldStart && k <= foldEnd) foldGroup.append(row);
    else body.append(row);
  });

  const copyLabel = () => L({ tr: "Kopyala", en: "Copy" });
  const copy = h("button", { type: "button", class: "code-copy", text: copyLabel() });
  copy.addEventListener("click", async () => {
    const text = lines.map((l) => l.text).join("\n");
    let copied = true;
    try {
      await navigator.clipboard.writeText(text);
    } catch {   // e.g. a page opened from a file
      const area = h("textarea", {});
      area.value = text;
      document.body.append(area);
      area.select();
      copied = document.execCommand("copy");
      area.remove();
      copy.focus({ preventScroll: true });
    }
    copy.textContent = copied ? L({ tr: "Kopyalandı ✓", en: "Copied ✓" }) : L({ tr: "Kopyalanamadı", en: "Could not copy" });
    setTimeout(() => { copy.textContent = copyLabel(); }, 1500);
  });

  const numbered = lines.filter((l) => l.n != null);
  const first = numbered.length ? numbered[0].n : 0;
  const last = numbered.length ? numbered[numbered.length - 1].n : 0;
  const range = numbered.length && options.fromFile
    ? L({ tr: `satır ${first}–${last}`, en: `lines ${first}–${last}` }) : (lang === "python" ? "Python" : "");
  const head = h("div", { class: "code-head" },
    h("span", { class: "code-dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")),
    h("span", { class: "code-tab" }, h("span", { class: "code-tab-icon", text: lang === "python" ? "py" : ">_" }), L(options.file) || L({ tr: "kod", en: "code" })),
    h("span", { class: "code-meta", text: range }),
    copy);

  const figure = h("figure", { class: `code${options.tall ? " tall" : ""}` }, head, body);
  if (notes.length) {
    const list = h("ol", { class: "code-notes" });
    for (const note of notes) {
      const item = h("li", {}, h("span", { class: "code-note-mark", text: note.mark }), h("span", {}, rich(note.text)));
      // Hovering a note lights up its line
      item.addEventListener("pointerenter", () => rows[note.index].classList.add("lit"));
      item.addEventListener("pointerleave", () => rows[note.index].classList.remove("lit"));
      list.append(item);
    }
    figure.append(list);
  }
  if (options.caption) figure.append(h("figcaption", {}, rich(options.caption)));
  return figure;
}

/** A terminal window. lines: ["$ command", "output", ...]; lines starting with "$ " are commands. */
export function terminal(title, lines) {
  const body = h("div", { class: "term-body" });
  for (const line of lines) {
    if (line.startsWith("$ ")) body.append(h("div", { class: "term-cmd" }, h("span", { class: "term-prompt", text: "PS>" }), " ", line.slice(2)));
    else body.append(h("div", { class: "term-out", text: line || " " }));
  }
  return h("figure", { class: "term" },
    h("div", { class: "term-head" }, h("span", { class: "code-dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")), h("span", { text: title })),
    body);
}
