/*
 * A Practice tab: short questions with immediate feedback, laid out by lesson.js (sections, table
 * of contents, language changes). The questions and the progress are in practice.ts.
 *
 *   createPractice({ main: "#px-main", toc: "#px-toc", summary: "#px-summary", sections })
 *
 * A wrong answer says why (a typical wrong number: which slip it comes from) and can be tried
 * again; "Show the answer" finishes the question without counting it as correct. A prediction
 * question is answered once: the real model then runs. A "model" question loads its model first, so
 * it runs in the background and its result is saved.
 */
import { L } from "./i18n.js";
import { $, CHAPTER, h, locale, scrollMode } from "./core.js";
import { placeLabel } from "./glossary/terms.js";
import { createLesson, lessonHTML } from "./lesson.js";
import { createProgress, matches, mistakeOf, plainMath, readNumber } from "./practice.ts";

const say = (tr, en) => L({ tr, en });
const blankDraft = () => ({ pick: null, value: "", tried: false, notice: null, wrongPick: null, mistake: null, busy: false });
// No thousands separator: the answer shown can be typed back as it is
const showNumber = (value) => new Intl.NumberFormat(locale(), { maximumFractionDigits: 6, useGrouping: false }).format(value);
const isPrediction = (q) => q.kind === "predict" || q.kind === "model";

export function createPractice({ main, toc, summary, sections }) {
  const questions = sections.flatMap((s) => s.questions);
  const progress = createProgress(CHAPTER, questions.map((q) => q.id));
  const numberOf = new Map(questions.map((q, k) => [q.id, k + 1]));
  const drafts = new Map();   // picked or typed but not finished: survives a language change
  const runs = new Map();     // a prediction's model runs once per page: its result never changes
  const runOf = (q) => {
    if (!runs.has(q.id)) runs.set(q.id, q.run());
    return runs.get(q.id);
  };
  /** The result to show for a finished question, if it has one. */
  const runFor = (q, saved) => {
    if (q.kind === "predict") return runOf(q);
    if (q.kind === "model" && saved.run) {
      try {
        return { answer: saved.run.answer, result: q.describe(saved.run.values) };
      } catch {
        return null;   // a damaged saved run (too few numbers): the result is not shown, the tab still works
      }
    }
    return null;
  };
  const cards = new Map();    // the card on the page for each question (a language change rebuilds them)
  let confirming = false;
  let lesson = null;

  const draftOf = (q) => {
    if (!drafts.has(q.id)) {
      const saved = progress.get(q.id);
      drafts.set(q.id, { ...blankDraft(), pick: saved?.pick ?? null, value: saved?.value ?? "" });
    }
    return drafts.get(q.id);
  };

  /** What the feedback box says for this question now. */
  function feedbackNodes(q, draft, saved, run) {
    const verdict = (text) => h("p", { class: "px-verdict", text });
    const note = (html) => lessonHTML("p", {}, html);
    if (saved) {
      const nodes = [];
      if (saved.status === "correct") nodes.push(verdict(isPrediction(q) ? say("✓ Tahmininiz doğru.", "✓ Your prediction was right.") : say("✓ Doğru.", "✓ Correct.")));
      else if (saved.status === "missed") nodes.push(verdict(say("✕ Tahmininiz tutmadı.", "✕ Your prediction was off.")));
      else {
        const shown = q.kind === "number" ? showNumber(q.answer) : "";
        nodes.push(q.kind === "number"
          ? verdict(`${say("Doğru cevap", "The answer")}: ${shown}`)
          : verdict(say("Doğru cevap işaretlendi.", "The answer is marked.")));
      }
      if (run) nodes.push(h("p", { class: "px-run" }, h("strong", { text: `${say("Model çalıştı", "The model ran")}: ` }), lessonHTML("span", {}, run.result)));
      nodes.push(note(q.explain));
      if (q.tryIt) {
        nodes.push(h("p", { class: "m-links" }, h("a", {
          href: "#", "data-go": `${q.tryIt.tab}:${q.tryIt.section || ""}`, "data-chapter": CHAPTER,
          text: `${say("Kendiniz deneyin", "Try it yourself")}: ${placeLabel({ chapter: CHAPTER, ...q.tryIt })} →`,
        })));
      }
      return nodes;
    }
    if (draft.busy) return [verdict(say("Model çalışıyor…", "The model is running…"))];
    if (draft.notice === "failed") return [verdict(say("Modelin dosyaları yüklenemedi. Yeniden deneyin.", "The model's files could not be loaded. Try again."))];
    if (draft.notice === "pick") return [verdict(say("Önce bir seçenek işaretleyin.", "Pick an option first."))];
    if (draft.notice === "number") return [verdict(say("Bir sayı yazın, örneğin 0,5.", "Type a number, for example 0.5."))];
    if (draft.notice === "wrong") {
      // A typical wrong number gets its own explanation, any other the question's hint
      const why = q.kind === "number" ? (draft.mistake?.why ?? q.hint) : q.options[draft.wrongPick]?.why;
      return [verdict(say("✕ Henüz değil.", "✕ Not yet.")), why ? note(why) : null].filter(Boolean);
    }
    return [];
  }

  /** Saves a finished question and shows its result on the card that is on the page now. */
  function finish(q, answer, takeFocus = true, { background = false } = {}) {
    const draft = draftOf(q);
    progress.set(q.id, answer);
    draft.notice = null;
    draft.busy = false;
    const card = cards.get(q.id);
    card?.sync();
    changed({ background });
    if (!card || !takeFocus) return;
    // The button that had the focus is gone: the keyboard continues from the result
    card.feedback.focus({ preventScroll: true });
    card.feedback.scrollIntoView({ behavior: scrollMode(true), block: "nearest" });
  }

  function questionCard(q) {
    const draft = draftOf(q);
    const name = `px-${q.id}`;
    const inputs = [];
    const optionRows = [];
    let body;
    if (q.kind === "number") {
      const input = h("input", { type: "text", class: "px-input", name, autocomplete: "off", spellcheck: "false", "aria-label": plainMath(L(q.label)) });
      input.value = draft.value;
      input.addEventListener("input", () => { draft.value = input.value; });
      inputs.push(input);
      body = h("label", { class: "px-number" }, lessonHTML("span", {}, q.label), input);
    } else {
      body = h("div", { class: "px-options" });
      const texts = q.kind === "choice" ? q.options.map((o) => o.text) : q.options;
      texts.forEach((text, k) => {
        const input = h("input", { type: "radio", name, value: String(k), "aria-label": plainMath(L(text)) });
        input.checked = draft.pick === k;
        input.addEventListener("change", () => { draft.pick = k; });
        inputs.push(input);
        const mark = h("span", { class: "px-mark" });
        const row = h("label", { class: "px-opt" }, input, lessonHTML("span", { class: "px-opt-text" }, text), mark);
        optionRows.push({ row, mark });
        body.append(row);
      });
    }

    const check = h("button", { type: "submit", class: "btn small primary", text: isPrediction(q) ? say("Tahmini sına", "Test my prediction") : say("Kontrol et", "Check") });
    const reveal = h("button", { type: "button", class: "btn small", text: say("Cevabı göster", "Show the answer") });
    const again = h("button", { type: "button", class: "btn small", text: say("Yeniden çöz", "Try again") });
    const feedback = h("div", { class: "px-feedback", role: "status", tabindex: "-1" });
    const form = h("form", { class: "px-q", "data-q": q.id, novalidate: true },
      h("fieldset", {},
        h("legend", {}, h("span", { class: "px-qnum", text: `${say("Soru", "Question")} ${numberOf.get(q.id)}` }), lessonHTML("span", { class: "px-qtext" }, q.q)),
        body),
      h("div", { class: "px-actions" }, check, reveal, again),
      feedback);

    const sync = () => {
      const saved = progress.get(q.id);
      const run = saved ? runFor(q, saved) : null;
      const rightIndex = q.kind === "choice" ? q.answer : isPrediction(q) ? (run?.answer ?? q.answer) : undefined;
      form.dataset.state = saved?.status ?? (draft.notice === "wrong" ? "wrong" : "");
      for (const input of inputs) input.disabled = Boolean(saved) || draft.busy;
      check.hidden = Boolean(saved);
      check.disabled = draft.busy;
      reveal.hidden = Boolean(saved) || !draft.tried;
      again.hidden = !saved;
      optionRows.forEach(({ row, mark }, k) => {
        const right = Boolean(saved) && k === rightIndex;
        const wrong = saved ? saved.pick === k && k !== rightIndex : draft.notice === "wrong" && draft.wrongPick === k;
        row.classList.toggle("is-right", right);
        row.classList.toggle("is-wrong", wrong);
        mark.textContent = right ? say("✓ doğru", "✓ correct") : wrong ? say("✕ yanlış", "✕ wrong") : "";
      });
      feedback.replaceChildren(...feedbackNodes(q, draft, saved, run));
    };

    cards.set(q.id, { form, feedback, sync });

    /** Runs the model in the background; the reader may scroll, change language or reset meanwhile. */
    const runModel = () => {
      const pick = draft.pick;
      draft.busy = true;
      draft.notice = null;
      sync();
      feedback.focus({ preventScroll: true });   // the button that had the focus has just turned off
      q.run().then(
        (run) => {
          if (drafts.get(q.id) !== draft) return;   // the progress was reset while it ran
          // Back to the result only if the reader is still there: not after scrolling away from it
          const shown = cards.get(q.id)?.form;
          const box = shown?.getBoundingClientRect();
          const inView = Boolean(box) && box.bottom > 0 && box.top < window.innerHeight;
          const takeFocus = inView && shown.contains(document.activeElement);
          finish(q, { status: pick === run.answer ? "correct" : "missed", pick, run }, takeFocus, { background: true });
        },
        () => {
          if (drafts.get(q.id) !== draft) return;
          draft.busy = false;
          draft.notice = "failed";
          cards.get(q.id)?.sync();
        },
      );
    };

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (progress.get(q.id) || draft.busy) return;
      if (q.kind === "number") {
        const values = readNumber(draft.value);
        if (!values.length) draft.notice = "number";
        else if (values.some((value) => matches(value, q.answer, q.tolerance))) return finish(q, { status: "correct", value: draft.value });
        else {
          draft.tried = true;
          draft.notice = "wrong";
          draft.mistake = mistakeOf(q, values);
        }
        return sync();
      }
      if (draft.pick === null) {
        draft.notice = "pick";
        return sync();
      }
      if (q.kind === "model") return runModel();
      if (q.kind === "predict") return finish(q, { status: draft.pick === runOf(q).answer ? "correct" : "missed", pick: draft.pick });
      if (draft.pick === q.answer) return finish(q, { status: "correct", pick: draft.pick });
      draft.tried = true;
      draft.notice = "wrong";
      draft.wrongPick = draft.pick;
      return sync();
    });

    reveal.addEventListener("click", () => finish(q, {
      status: "revealed",
      ...(draft.pick === null ? {} : { pick: draft.pick }),
      ...(draft.value ? { value: draft.value } : {}),
    }));

    again.addEventListener("click", () => {
      progress.clear(q.id);
      Object.assign(draft, blankDraft());
      for (const input of inputs) {
        if (input.type === "radio") input.checked = false;
        else input.value = "";
      }
      sync();
      changed();
      inputs[0].focus({ preventScroll: true });
    });

    sync();
    return form;
  }

  /** A section whose questions are all finished gets a check mark in the table of contents. */
  function markSections() {
    const list = $(toc);
    for (const section of sections) {
      const button = list.querySelector(`[data-step="${section.id}"]`);
      if (!button) continue;
      const done = section.questions.every((q) => progress.get(q.id));
      const num = button.querySelector(".toc-num");
      button.classList.toggle("px-done", done);
      num.textContent = done ? "✓" : String(sections.indexOf(section) + 1);
      if (done) button.setAttribute("aria-label", `${L(section.title)} (${say("tamamlandı", "finished")})`);
      else button.removeAttribute("aria-label");
    }
  }

  function renderSummary(focus = null) {
    const { total, done, correct } = progress.summary();
    const count = h("p", { class: "px-count", tabindex: "-1" },
      h("strong", { text: `${say("Tamamlanan", "Finished")}: ${done} / ${total}` }),
      h("span", { text: ` · ${say("Doğru", "Correct")}: ${correct}` }));
    const bar = h("div", {
      class: "px-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": String(total), "aria-valuenow": String(done),
      "aria-label": say("Tamamlanan sorular", "Questions finished"),
    }, h("span", { style: `width:${total ? (done / total) * 100 : 0}%` }));
    const tools = h("div", { class: "px-tools" });
    let target = count;
    if (confirming) {
      const yes = h("button", { type: "button", class: "btn small", text: say("Evet, sıfırla", "Yes, reset") });
      const no = h("button", { type: "button", class: "btn small primary", text: say("Vazgeç", "Cancel") });
      yes.addEventListener("click", () => {
        confirming = false;
        progress.clearAll();
        drafts.clear();
        lesson?.rerender();
        renderSummary("count");
      });
      no.addEventListener("click", () => {
        confirming = false;
        renderSummary("reset");
      });
      tools.append(h("span", { text: say("Bu bölümdeki tüm cevaplar silinsin mi?", "Delete all answers in this chapter?") }), yes, no);
      if (focus === "confirm") target = no;
    } else if (done) {
      const reset = h("button", { type: "button", class: "btn small", text: say("İlerlemeyi sıfırla", "Reset progress") });
      reset.addEventListener("click", () => {
        confirming = true;
        renderSummary("confirm");
      });
      tools.append(reset);
      if (focus === "reset") target = reset;
    }
    $(summary).replaceChildren(
      h("div", { class: "px-sum-text" },
        h("h2", { text: say("Kendinizi sınayın", "Test yourself") }),
        h("p", { text: say(
          "Sorular Teori sekmesinin sırasını izler. Cevaplarınız yalnızca bu tarayıcıda saklanır, hiçbir yere gönderilmez.",
          "The questions follow the order of the Theory tab. Your answers stay in this browser and are not sent anywhere.",
        ) })),
      h("div", { class: "px-sum-state" }, count, bar, tools));
    if (focus) target.focus({ preventScroll: true });
  }

  /**
   * After a question is finished. A model run that ends in the background leaves an open "Delete
   * all answers?" question and the focus in the summary where they are.
   */
  function changed({ background = false } = {}) {
    markSections();
    if (background && confirming) return;
    confirming = false;
    const inSummary = $(summary).contains(document.activeElement);
    renderSummary(background && inSummary ? (done() ? "reset" : "count") : null);
  }
  const done = () => progress.summary().done > 0;

  const widgets = Object.fromEntries(questions.map((q) => [q.id, () => questionCard(q)]));
  lesson = createLesson({
    tab: "practice", main, toc, widgets,
    kicker: { tr: "Kısım", en: "Section" },
    onRender: markSections,
    sections: sections.map((s) => ({
      id: s.id,
      title: s.title,
      blocks: [{ p: s.intro }, ...s.questions.map((q) => ({ widget: q.id })), { links: [s.read] }],
    })),
  });

  renderSummary();
  document.addEventListener("lang-change", () => renderSummary());
}
