export const termChip = (key) => h("button", { type: "button", class: "gl-chip gl-term", "data-term": key, text: L(GLOSSARY[key].name) });

/** Fills a container with one term: name, other-language name, definition, model note, related, links. */
import { hooks } from "../page.ts";
import { I18N, L } from "../i18n.js";
import { $, CHAPTER, chapterOf, goToChapterPlace, h, hideTip, nav, navigate } from "../core.js";
import { GLOSSARY, GLOSSARY_GROUPS, otherLang, placeLabel, relatedTerms, termPlaces, termify } from "./terms.js";

export function renderTerm(key, { glossaryLink }) {
  const entry = GLOSSARY[key];
  const alt = entry.name[otherLang()];
  const nodes = [];
  const def = h("div", { class: "g-def" });
  def.innerHTML = termify(L(entry.def), key);   // static, author-written markup
  nodes.push(def);
  if (entry.model) {
    const label = L(entry.modelLabel || { tr: "SimpleCNN'de", en: "In SimpleCNN" });
    const note = h("p", { class: "g-model" }, h("b", { text: `${label}: ` }));
    note.insertAdjacentHTML("beforeend", termify(L(entry.model), key));
    nodes.push(note);
  }
  const related = relatedTerms(key);
  if (related.length) {
    nodes.push(h("div", { class: "g-see" }, h("span", { text: L({ tr: "İlgili:", en: "Related:" }) }), ...related.map(termChip)));
  }
  const links = h("div", { class: "g-links" });
  for (const place of termPlaces(key)) {
    if ((place.chapter || "cnn") !== CHAPTER && !chapterOf(place.chapter || "cnn")?.url) {   // that chapter is still to come
      links.append(h("span", { class: "g-place soon", text: `${placeLabel(place)} · ${L({ tr: "yakında", en: "coming soon" })}` }));
      continue;
    }
    const a = h("a", { href: "#", class: "g-place", text: `${placeLabel(place)} →` });
    a.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();   // not the page-wide data-go handler
      goToPlace(place);
    });
    links.append(a);
  }
  if (glossaryLink) {
    const a = h("a", { href: "#", class: "g-open", text: L({ tr: "Sözlükte aç", en: "Open in glossary" }) });
    a.addEventListener("click", (event) => {
      event.preventDefault();
      closeTerm();
      hooks.closeInfo?.();
      openGlossary(key);
    });
    links.append(a);
  }
  if (links.childNodes.length) nodes.push(links);
  return { title: L(entry.name), alt: alt !== L(entry.name) ? alt : "", nodes };
}

/** Follows a term's "where to see it" link: closes the popups and goes there. */
export function goToPlace(link) {
  closeTerm();
  hooks.closeInfo?.();
  if ((link.chapter || "cnn") !== CHAPTER) {   // another chapter's page
    goToChapterPlace({ ...link, chapter: link.chapter || "cnn" });   // a place without a chapter is in the CNN chapter
    return;
  }
  const moves = link.tab !== nav.tab || Boolean(link.section);   // navigate() does nothing otherwise
  if (glossary.open) closeGlossary({ fromLink: moves });
  if (moves) navigate(link.tab, link.section || null);
}

/** Places a floating box under an anchor (or above it when there is no room), inside the window. */
export function placeNear(el, anchor) {
  // Measure at the top-left corner: left at its old place, the box could widen the page
  // (on phones that widens the viewport too) and the measurement would be off
  el.style.left = "0px";
  el.style.top = "0px";
  const viewW = document.documentElement.clientWidth;
  const viewH = document.documentElement.clientHeight;
  const b = anchor.getBoundingClientRect();
  const width = el.offsetWidth;
  const height = el.offsetHeight;
  let left = b.left + b.width / 2 - width / 2;
  left = Math.max(12, Math.min(left, viewW - width - 12));
  let top = b.bottom + 8;
  if (top + height > viewH - 12 && b.top - height - 8 > 12) top = b.top - height - 8;
  el.style.left = `${left + window.scrollX}px`;
  el.style.top = `${top + window.scrollY}px`;
}

// =====================================================================
// Term card: opens next to a term link; terms inside it can be followed (with a back button)
// =====================================================================
export const termCard = { el: null, anchor: null, key: null, trail: [] };

export function buildTermCard() {
  const back = h("button", { type: "button", class: "btn small icon gl-card-back", text: "←" });
  back.addEventListener("click", () => {
    const previous = termCard.trail.pop();
    if (previous) showTerm(previous);
  });
  const close = h("button", { type: "button", class: "btn small icon gl-card-close", text: "×" });
  close.addEventListener("click", closeTerm);
  termCard.el = h("div", { class: "gl-card", role: "dialog", hidden: true },
    h("div", { class: "gl-card-head" }, back,
      h("div", { class: "gl-card-titles" }, h("strong", { class: "gl-card-title" }), h("span", { class: "gl-card-alt" })),
      close),
    h("div", { class: "gl-card-body" }));
  document.body.append(termCard.el);
}

export function showTerm(key) {
  termCard.key = key;
  const view = renderTerm(key, { glossaryLink: true });
  const card = termCard.el;
  card.querySelector(".gl-card-title").textContent = view.title;
  card.querySelector(".gl-card-alt").textContent = view.alt;
  card.querySelector(".gl-card-alt").hidden = !view.alt;
  card.querySelector(".gl-card-body").replaceChildren(...view.nodes);
  const back = card.querySelector(".gl-card-back");
  back.hidden = !termCard.trail.length;
  if (termCard.trail.length) {
    const previous = L(GLOSSARY[termCard.trail[termCard.trail.length - 1]].name);
    back.setAttribute("aria-label", `${L({ tr: "Geri", en: "Back" })}: ${previous}`);
    back.title = back.getAttribute("aria-label");
  }
  card.querySelector(".gl-card-close").setAttribute("aria-label", L({ tr: "Kapat", en: "Close" }));
  card.setAttribute("aria-label", view.title);
  placeNear(card, termCard.anchor);
}

export function openTerm(key, anchor) {
  if (termCard.anchor === anchor && termCard.key === key && !termCard.trail.length) {   // a second click closes it
    closeTerm();
    return;
  }
  termCard.anchor?.setAttribute("aria-expanded", "false");
  termCard.anchor = anchor;
  termCard.trail = [];
  anchor.setAttribute("aria-expanded", "true");
  hideTip();
  termCard.el.hidden = false;
  showTerm(key);
  termCard.el.querySelector(".gl-card-close").focus({ preventScroll: true });
}

export function closeTerm() {
  if (!termCard.anchor) return;
  termCard.anchor.setAttribute("aria-expanded", "false");
  const anchor = termCard.anchor;
  termCard.anchor = null;
  termCard.key = null;
  termCard.trail = [];
  termCard.el.hidden = true;
  if (termCard.el.contains(document.activeElement) && anchor.isConnected) anchor.focus({ preventScroll: true });
}

// =====================================================================
// Glossary panel
// =====================================================================
export const glossary = { el: null, list: null, search: null, empty: null, count: null, open: false, opener: null };

/** Lower case without accents, with Turkish dotted and dotless i treated alike, for searching. */
export const fold = (s) => s.toLocaleLowerCase("tr").normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/ı/g, "i");

export function buildGlossary() {
  const close = h("button", { type: "button", class: "btn small icon glossary-close", text: "×" });
  close.addEventListener("click", () => closeGlossary());
  glossary.search = h("input", { type: "search", class: "glossary-search", autocomplete: "off", spellcheck: "false" });
  glossary.search.addEventListener("input", filterGlossary);
  glossary.count = h("span", { class: "glossary-count" });
  glossary.list = h("div", { class: "glossary-list" });
  glossary.empty = h("p", { class: "glossary-empty", hidden: true });
  glossary.el = h("aside", { class: "glossary", id: "glossary", "aria-labelledby": "glossary-title", inert: "" },
    h("div", { class: "glossary-head" },
      h("div", {}, h("h2", { id: "glossary-title" }), glossary.count),
      close),
    h("div", { class: "glossary-tools" }, glossary.search),
    glossary.list,
    glossary.empty);
  document.body.append(glossary.el);
  renderGlossary();
}

export function renderGlossary() {
  glossary.el.querySelector("#glossary-title").textContent = L({ tr: "Sözlük", en: "Glossary" });
  glossary.el.querySelector(".glossary-close").setAttribute("aria-label", L({ tr: "Sözlüğü kapat", en: "Close the glossary" }));
  glossary.search.placeholder = L({ tr: "Kavram ara (Türkçe ya da İngilizce)…", en: "Search a term (English or Turkish)…" });
  glossary.search.setAttribute("aria-label", L({ tr: "Sözlükte ara", en: "Search the glossary" }));
  glossary.empty.textContent = L({ tr: "Bu aramayla eşleşen kavram yok.", en: "No term matches this search." });
  const collator = new Intl.Collator(I18N.lang);
  const sections = GLOSSARY_GROUPS.map((group) => {
    const keys = Object.keys(GLOSSARY)
      .filter((key) => GLOSSARY[key].group === group.key)
      .sort((a, b) => collator.compare(L(GLOSSARY[a].name), L(GLOSSARY[b].name)));
    return h("section", { class: "glossary-group", "data-group": group.key },
      h("h3", { text: L(group.title) }),
      ...keys.map((key) => {
        const view = renderTerm(key, { glossaryLink: false });
        return h("article", { class: "g-entry", id: `g-${key}`, "data-term-entry": key, tabindex: "-1" },
          h("h4", {}, h("span", { text: view.title }), view.alt ? h("span", { class: "g-alt", text: view.alt }) : null),
          ...view.nodes);
      }));
  });
  glossary.list.replaceChildren(...sections);
  filterGlossary();
}

export function filterGlossary() {
  const query = fold(glossary.search.value.trim());
  let shown = 0;
  for (const article of glossary.list.querySelectorAll(".g-entry")) {
    const entry = GLOSSARY[article.dataset.termEntry];
    const names = fold([entry.name.tr, entry.name.en, ...(entry.aka || [])].join(" "));
    const text = fold(article.textContent);
    const match = !query || names.includes(query) || text.includes(query);
    article.hidden = !match;
    article.classList.toggle("g-name-match", Boolean(query) && names.includes(query));
    if (match) shown++;
  }
  for (const section of glossary.list.querySelectorAll(".glossary-group")) {
    section.hidden = !section.querySelector(".g-entry:not([hidden])");
  }
  glossary.empty.hidden = shown > 0;
  const total = Object.keys(GLOSSARY).length;
  glossary.count.textContent = query
    ? L({ tr: `${shown} / ${total} kavram`, en: `${shown} of ${total} terms` })
    : L({ tr: `${total} kavram`, en: `${total} terms` });
}

/** Scrolls the panel to a term and highlights it. */
export function showGlossaryEntry(key, { smooth = true } = {}) {
  let article = glossary.list.querySelector(`#g-${CSS.escape(key)}`);
  if (article?.hidden) {   // hidden by the search: clear the search first
    glossary.search.value = "";
    filterGlossary();
    article = glossary.list.querySelector(`#g-${CSS.escape(key)}`);
  }
  if (!article) return;
  article.scrollIntoView({ behavior: smooth ? "smooth" : "instant", block: "start" });
  article.classList.remove("nav-flash");
  void article.getBoundingClientRect();
  article.classList.add("nav-flash");
  setTimeout(() => article.classList.remove("nav-flash"), 1800);
  article.focus({ preventScroll: true });
}

export function setGlossaryVisible(visible) {
  glossary.open = visible;
  glossary.el.classList.toggle("open", visible);
  glossary.el.toggleAttribute("inert", !visible);
  document.body.classList.toggle("glossary-open", visible);
  $("#glossary-btn")?.setAttribute("aria-expanded", String(visible));
}

/** Opens the panel as a history step (marked glossaryStep), so Back closes it. */
export function openGlossary(key = null) {
  if (!glossary.open) {
    glossary.opener = document.activeElement;
    history.replaceState({ ...history.state, tab: nav.tab, scroll: window.scrollY }, "");
    history.pushState({ ...history.state, glossary: true, glossaryStep: true, gscroll: 0 }, "");
    setGlossaryVisible(true);
  }
  if (key) showGlossaryEntry(key, { smooth: false });
  else {
    glossary.list.scrollTop = 0;
    glossary.search.focus({ preventScroll: true });
  }
}

/**
 * Closes the panel. When the current history entry is the one that opened it, that step is
 * undone (Back would do the same); otherwise the entry is just marked closed.
 */
export function closeGlossary({ fromLink = false } = {}) {
  if (!glossary.open) return;
  if (fromLink) {   // a link inside it is followed: keep the entry open for Back, with the list position
    history.replaceState({ ...history.state, gscroll: glossary.list.scrollTop }, "");
    setGlossaryVisible(false);
    return;
  }
  if (history.state?.glossaryStep) {
    history.back();   // the popstate handler hides it
  } else {
    history.replaceState({ ...history.state, glossary: false }, "");
    setGlossaryVisible(false);
  }
  if (glossary.opener?.isConnected && glossary.opener !== document.body) glossary.opener.focus({ preventScroll: true });
}

/** Used by the history handler in app.js: the panel follows the history entry. */
export function glossaryIsOpen() {
  return glossary.open;
}

export function syncGlossary(state) {
  const want = Boolean(state?.glossary);
  if (want === glossary.open) return;
  setGlossaryVisible(want);
  if (want) glossary.list.scrollTop = state.gscroll || 0;
}

// Remember where the list is scrolled, so Back / Forward reopen it at the same place
export let glossaryScrollTimer = 0;

// =====================================================================
// Wiring
// =====================================================================
/**
 * Static page texts (data-i18n) may hold [[key|text]] too; i18n.js puts the raw text back on
 * every language change, so the links are made again after it.
 */
export function linkStaticTerms() {
  for (const el of document.querySelectorAll("[data-i18n]")) {
    if (el.innerHTML.includes("[[")) el.innerHTML = termify(el.innerHTML);
  }
}

buildTermCard();
buildGlossary();
linkStaticTerms();

glossary.list.addEventListener("scroll", () => {
  clearTimeout(glossaryScrollTimer);
  glossaryScrollTimer = setTimeout(() => {
    if (glossary.open && history.state?.glossary) history.replaceState({ ...history.state, gscroll: glossary.list.scrollTop }, "");
  }, 250);
}, { passive: true });

$("#glossary-btn").addEventListener("click", () => (glossary.open ? closeGlossary() : openGlossary()));

document.addEventListener("click", (event) => {
  const term = event.target.closest?.(".gl-term");
  if (term) {
    event.preventDefault();
    event.stopImmediatePropagation();   // the info popup's "click outside" check must not see this click
    const key = term.dataset.term;
    if (glossary.el.contains(term)) {   // inside the panel: go to that entry
      showGlossaryEntry(key);
    } else if (termCard.el.contains(term)) {   // inside the card: follow it, with a way back
      termCard.trail.push(termCard.key);
      showTerm(key);
    } else {
      openTerm(key, term);
    }
    return;
  }
  // composedPath still includes elements the click handler replaced (the card re-renders itself)
  if (termCard.anchor && !event.composedPath().includes(termCard.el)) closeTerm();
});

// Esc closes the top-most layer only: the term card, then (in help.js) the info popup, then the panel
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || document.querySelector("dialog[open]")) return;   // a dialog (zoom) closes itself
  if (termCard.anchor) {
    closeTerm();
    event.stopImmediatePropagation();
  } else if (glossary.open && !hooks.infoPopOpen?.()) {
    if (document.activeElement === glossary.search && glossary.search.value) {
      glossary.search.value = "";   // first Esc clears the search
      filterGlossary();
    } else {
      closeGlossary();
    }
    event.stopImmediatePropagation();
  }
}, true);

window.addEventListener("resize", () => { if (termCard.anchor) placeNear(termCard.el, termCard.anchor); });
document.addEventListener("tab-shown", closeTerm);
document.addEventListener("lang-change", () => {
  linkStaticTerms();
  renderGlossary();
  if (termCard.anchor && termCard.key) showTerm(termCard.key);
  $("#glossary-btn .glossary-btn-text").textContent = L({ tr: "Sözlük", en: "Glossary" });
  $("#glossary-btn").title = L({ tr: "Kavramlar sözlüğü", en: "Glossary of terms" });
});
$("#glossary-btn").title = L({ tr: "Kavramlar sözlüğü", en: "Glossary of terms" });

// core.js reaches the panel through these hooks (it is loaded before the glossary)
hooks.glossaryIsOpen = glossaryIsOpen;
hooks.syncGlossary = syncGlossary;
