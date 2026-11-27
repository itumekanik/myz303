/*
 * A lesson tab: numbered sections with a table of contents, rendered the first time the tab
 * is shown and again on every language change (the reader stays at the same section).
 * Chapter pages use it for their Math and Coding tabs.
 *
 *   createLesson({ tab: "math", main: "#pm-main", toc: "#pm-toc", sections, widgets,
 *                  kicker: { tr: "Kısım", en: "Section" }, source: { text, name } })
 *   A chapter with two files passes sources: [{ text, name }, …] instead; source is the first one.
 *
 * Section: { id, title, blocks }. Blocks (every text may be a { tr, en } pair):
 *   { p }                     a paragraph of HTML; \( TeX \) is typeset, [[key|text]] links a term
 *   { h }                     a sub-heading
 *   { eq }                    a displayed formula (TeX without delimiters)
 *   { derive: { title, steps: [{ tex, why }] } }   revealed one line at a time
 *   { widget: name }          widgets[name](state) returns a node
 *   { model }                 a highlighted "in this chapter" note (HTML)
 *   { callout: "tip" | "warn" | "info", title, text }
 *   { table: { head, rows } } cells are HTML
 *   { list: [html, …], ordered }
 *   { links: [{ chapter?, tab, section? }, …] }
 *   { code, name, notes, caption }   a code box (codeview.js)
 *   { file: { from, to, notes, source? } }    lines cut from the source file, by the text of the first and
 *                             last line; source names one of the sources (default: the first)
 *   { term: title, lines }           a terminal window
 *   { run: { file, text, packages, … } }   how to run a Python file: download, Colab, Windows, macOS (runbox.js)
 *
 * Uses i18n.js (L), core.js (h, $, navigate, PLACE_TITLES), glossary.js (termify, placeLabel)
 * and codeview.js (codeEditor, terminal).
 */
import { L } from "./i18n.js";
import { $, CHAPTER, PLACE_TITLES, chapterOf, h, navigate, scrollMode } from "./core.js";
import { placeLabel, termify } from "./glossary/terms.js";
import { codeEditor, terminal } from "./codeview.js";
import { runBox } from "./runbox.js";

export const lessonText = (value) => (value == null || typeof value === "string" ? value || "" : L(value));

/** Author-written HTML with glossary links. */
export function lessonHTML(tag, attrs, html) {
  const node = h(tag, attrs);
  node.innerHTML = termify(lessonText(html));
  return node;
}

export const lessonEq = (tex, cls = "") => h("div", { class: `m-eq ${cls}`.trim(), text: `\\(\\displaystyle ${tex}\\)` });

export function createLesson(config) {
  const L_ = { started: false, rendered: false, observer: null, derive: {}, state: {}, sources: new Map() };
  const sourceFiles = config.sources ?? (config.source ? [config.source] : []);
  const { tab, sections, widgets = {} } = config;
  for (const s of sections) PLACE_TITLES[s.id] = s.title;   // links can name the sections before they render

  /** A derivation revealed one line at a time. */
  function derivation(block, key) {
    const steps = block.steps;
    const box = h("div", { class: "m-derive" });
    const counter = h("span", { class: "m-derive-count" });
    const list = h("ol", { class: "m-derive-steps" });
    for (const step of steps) list.append(h("li", {}, lessonEq(lessonText(step.tex)), lessonHTML("p", { class: "m-why" }, step.why)));
    const next = h("button", { type: "button", class: "btn small primary", text: L({ tr: "Sonraki adım ↓", en: "Next step ↓" }) });
    const all = h("button", { type: "button", class: "btn small", text: L({ tr: "Hepsini göster", en: "Show all" }) });
    const reset = h("button", { type: "button", class: "btn small", text: L({ tr: "Baştan", en: "Start over" }) });
    const show = (n, animate) => {
      L_.derive[key] = n;
      [...list.children].forEach((li, k) => {
        const visible = k < n;
        if (visible && li.hidden && animate) li.classList.add("m-reveal");
        li.hidden = !visible;
      });
      counter.textContent = L({ tr: `${n} / ${steps.length} adım`, en: `step ${n} of ${steps.length}` });
      next.disabled = n >= steps.length;
      all.disabled = n >= steps.length;
      reset.disabled = n <= 1;
    };
    next.addEventListener("click", () => {
      show(Math.min(steps.length, (L_.derive[key] ?? 1) + 1), true);
      list.children[L_.derive[key] - 1]?.scrollIntoView({ behavior: scrollMode(true), block: "nearest" });
      if (next.disabled) reset.focus({ preventScroll: true });   // the focused button just turned off
    });
    all.addEventListener("click", () => {
      show(steps.length, true);
      if (all.disabled) reset.focus({ preventScroll: true });
    });
    reset.addEventListener("click", () => {
      show(1, false);
      next.focus({ preventScroll: true });
    });
    box.append(
      h("div", { class: "m-derive-head" }, h("span", { class: "m-derive-kicker", text: L({ tr: "Türetme", en: "Derivation" }) }),
        h("strong", { text: L(block.title) }), counter),
      list,
      h("div", { class: "m-derive-ctl" }, next, all, reset));
    show(L_.derive[key] ?? 1, false);
    return box;
  }

  function sourceLines(spec, source) {
    const all = source.map((text, k) => ({ n: k + 1, text }));
    if (spec.whole) return all;
    const start = all.findIndex((l) => l.text.includes(spec.from));
    const end = start === -1 ? -1 : all.findIndex((l, k) => k >= start && l.text.includes(spec.to));
    return start === -1 || end === -1 ? null : all.slice(start, end + 1);
  }

  function renderBlock(block, key) {
    if (block.p) return lessonHTML("p", {}, block.p);
    if (block.h) return h("h3", { class: "m-h3", text: L(block.h) });
    if (block.eq) return lessonEq(lessonText(block.eq));
    if (block.derive) return derivation(block.derive, key);
    if (block.widget) return widgets[block.widget](L_.state);
    if (block.model) return lessonHTML("div", { class: "in-model m-model" }, block.model);
    if (block.callout) {
      const icons = { tip: "💡", warn: "⚠️", info: "ℹ️" };
      return h("div", { role: "note", class: `tut-callout ${block.callout}` },
        h("span", { class: "tut-callout-icon", text: icons[block.callout], "aria-hidden": "true" }),
        h("div", {}, h("strong", { text: L(block.title) }), lessonHTML("p", {}, block.text)));
    }
    if (block.table) {
      return h("div", { class: "table-wrap" }, h("table", { class: "data-table tut-table m-table" },
        h("thead", {}, h("tr", {}, ...block.table.head.map((c) => lessonHTML("th", {}, c)))),
        h("tbody", {}, ...block.table.rows.map((row) => h("tr", {}, ...row.map((c) => lessonHTML("td", {}, c)))))));
    }
    if (block.list) {
      const list = h(block.ordered ? "ol" : "ul", { class: "tut-list" });
      for (const item of block.list) list.append(lessonHTML("li", {}, item));
      return list;
    }
    if (block.links) {
      return h("div", { class: "m-links" }, h("span", { text: L({ tr: "Ayrıca bakın:", en: "See also:" }) }),
        ...block.links.map((place) => (chapterOf(place.chapter || CHAPTER)?.url || (place.chapter || CHAPTER) === CHAPTER
          ? h("a", {
            href: "#", "data-go": `${place.tab}:${place.section || ""}`, "data-chapter": place.chapter || CHAPTER,
            text: `${placeLabel({ chapter: CHAPTER, ...place })} →`,
          })
          // a chapter that is still to come: named, not linked
          : h("span", { class: "m-link-soon", text: `${placeLabel({ chapter: CHAPTER, ...place })} · ${L({ tr: "yakında", en: "coming soon" })}` }))));
    }
    if (block.code) {
      const lines = lessonText(block.code).split("\n").map((text, k) => ({ n: k + 1, text }));
      return codeEditor(lines, { file: L(block.name), notes: block.notes, caption: block.caption });
    }
    if (block.file) {
      const name = block.file.source ?? sourceFiles[0]?.name;
      const source = L_.sources.get(name);
      if (!source) {
        return h("p", { class: "placeholder", text: L({
          tr: `${name} bu derlemede yok.`,
          en: `${name} is missing from this build.`,
        }) });
      }
      const lines = sourceLines(block.file, source);
      if (!lines) return h("p", { class: "placeholder", text: `${L({ tr: "Kod parçası bulunamadı", en: "Code excerpt not found" })}: ${block.file.from}` });
      return codeEditor(lines, { file: name, fromFile: true, notes: block.file.notes, caption: block.file.caption });
    }
    if (block.term) return terminal(L(block.term), block.lines.map((line) => L(line)));
    if (block.run) return runBox(block.run);
    return null;
  }

  const goTo = (id) => navigate(tab, id);

  function render() {
    const main = $(config.main);
    const toc = $(config.toc);
    main.replaceChildren();
    toc.replaceChildren();
    sections.forEach((section, k) => {
      const title = L(section.title);
      const kicker = L(config.kicker || { tr: "Kısım", en: "Section" });
      const article = h("article", { class: "card tut-step m-step", id: section.id },
        h("header", { class: "tut-step-head" },
          h("span", { class: "tut-num", text: String(k + 1) }),
          h("div", {}, h("span", { class: "tut-kicker", text: L({ tr: `${kicker} ${k + 1} / ${sections.length}`, en: `${kicker} ${k + 1} of ${sections.length}` }) }),
            h("h2", { text: title }))));
      section.blocks.forEach((block, b) => {
        const node = renderBlock(block, `${section.id}:${b}`);
        if (node) article.append(node);
      });
      const nav = h("footer", { class: "tut-nav" });
      if (k > 0) {
        const prev = h("button", { type: "button", class: "btn small", text: `← ${L(sections[k - 1].title)}` });
        prev.addEventListener("click", () => goTo(sections[k - 1].id));
        nav.append(prev);
      }
      if (k < sections.length - 1) {
        const next = h("button", { type: "button", class: "btn small primary", text: `${L(sections[k + 1].title)} →` });
        next.addEventListener("click", () => goTo(sections[k + 1].id));
        nav.append(next);
      }
      article.append(nav);
      main.append(article);
      const link = h("button", { type: "button", "data-step": section.id },
        h("span", { class: "toc-num", text: String(k + 1) }), h("span", { text: title }));
      link.addEventListener("click", () => goTo(section.id));
      toc.append(h("li", {}, link));
    });

    // Highlight the section being read in the table of contents
    L_.observer?.disconnect();
    L_.observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const b of toc.querySelectorAll("button")) b.classList.toggle("active", b.dataset.step === entry.target.id);
      }
    }, { rootMargin: "-25% 0px -65% 0px" });
    for (const article of main.querySelectorAll(".m-step")) L_.observer.observe(article);
    L_.rendered = true;
    config.onRender?.();
  }

  /** The section nearest the top of the viewport and its offset (null if the tab is hidden). */
  function readingPosition() {
    const steps = [...$(config.main).querySelectorAll(".m-step")];
    if (!steps.length || !steps[0].getClientRects().length) return null;
    const topbar = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--topbar-h")) || 59;
    let anchor = steps[0];
    for (const step of steps) {
      if (step.getBoundingClientRect().top <= topbar + 40) anchor = step;
      else break;
    }
    return { id: anchor.id, top: anchor.getBoundingClientRect().top };
  }

  async function start() {
    if (L_.started) return;
    L_.started = true;
    for (const file of sourceFiles) {
      if (file.text) L_.sources.set(file.name, file.text.replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n"));
    }
    render();
  }

  document.addEventListener("tab-shown", (event) => {
    if (event.detail === tab) start();
  });

  document.addEventListener("lang-change", () => {
    if (!L_.rendered) return;
    const position = readingPosition();
    render();
    if (!position) return;
    const anchor = document.getElementById(position.id);
    if (!anchor) return;
    if (position.top >= 0) window.scrollBy({ top: anchor.getBoundingClientRect().top - position.top, behavior: "instant" });
    else anchor.scrollIntoView({ behavior: "instant", block: "start" });
  });

  return { state: L_.state, rerender: () => L_.rendered && render() };
}
