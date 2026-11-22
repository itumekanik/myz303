/**
 * The Practice tab without its page: the kinds of question, reading a typed number, and the
 * reader's progress. Progress stays in this browser (localStorage) and is never sent anywhere;
 * when storage is blocked it lasts until the page is closed.
 */
import type { Text } from "./chapters.ts";

export interface Place {
  tab: string;
  section?: string;
}

interface QuestionBase {
  id: string;
  q: Text;
  /** Shown once the question is finished. */
  explain: Text;
}

export interface ChoiceQuestion extends QuestionBase {
  kind: "choice";
  /** `why` says what is wrong with a wrong option. */
  options: { text: Text; why?: Text }[];
  answer: number;
  /** Works the answer out with the chapter's code, for the unit tests. */
  compute?: () => number;
}

export interface NumberQuestion extends QuestionBase {
  kind: "number";
  /** What to type, e.g. \(z =\). */
  label: Text;
  answer: number;
  hint: Text;
  compute: () => number;
}

/** The reader commits to a prediction, then the real model runs and shows what happens. */
export interface PredictQuestion extends QuestionBase {
  kind: "predict";
  options: Text[];
  answer: number;
  run: () => { answer: number; result: Text };
  tryIt: Place;
}

export type Question = ChoiceQuestion | NumberQuestion | PredictQuestion;

export interface PracticeSection {
  id: string;
  title: Text;
  intro: Text;
  read: Place;
  questions: Question[];
}

/** A number as a student types it: 0,5 or 0.5, with a plain or typographic minus. Null if it is not one. */
export function parseNumber(text: string): number | null {
  const clean = text.trim().replace(/[−–]/g, "-");
  if (!/^[+-]?(\d+([.,]\d*)?|[.,]\d+)$/.test(clean)) return null;
  return Number(clean.replace(",", "."));
}

export const matches = (value: number, answer: number): boolean => Math.abs(value - answer) < 1e-6;

/** A label with formulas as plain text, the name of its input for screen readers. */
export function plainMath(text: string): string {
  return text
    .replace(/\\[()]/g, "")
    .replace(/\\hat\s*y/g, "ŷ")
    .replace(/\\eta/g, "η")
    .replace(/\{,\}/g, ",")
    .replace(/\\[ ,;]/g, " ")
    .replace(/_\{?(\w+)\}?/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

export type Status = "correct" | "revealed" | "missed";

export interface Answer {
  status: Status;
  pick?: number;
  value?: string;
}

type Store = Pick<Storage, "getItem" | "setItem">;
type Saved = Record<string, { total: number; answers: Record<string, Answer> }>;

const KEY = "practice-v1";

function browserStore(): Store | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

function readAll(store: Store | null): Saved {
  try {
    const data: unknown = JSON.parse(store?.getItem(KEY) ?? "{}");
    return data && typeof data === "object" && !Array.isArray(data) ? (data as Saved) : {};
  } catch {
    return {};
  }
}

const validAnswer = (a: unknown): a is Answer =>
  Boolean(a) && typeof a === "object" && ["correct", "revealed", "missed"].includes((a as Answer).status);

/** The progress of one chapter, for the questions it has now. */
export function createProgress(chapter: string, ids: readonly string[], store: Store | null = browserStore()) {
  const stored: unknown = readAll(store)[chapter]?.answers;
  const answers = new Map<string, Answer>();
  if (stored && typeof stored === "object") {
    for (const id of ids) {
      const a = (stored as Record<string, unknown>)[id];
      if (validAnswer(a)) answers.set(id, a);
    }
  }

  const save = (): void => {
    if (!store) return;
    try {
      const all = readAll(store); // the other chapters stay as they are
      if (answers.size) all[chapter] = { total: ids.length, answers: Object.fromEntries(answers) };
      else delete all[chapter];
      store.setItem(KEY, JSON.stringify(all));
    } catch {
      /* storage full or blocked: the progress lasts until the page is closed */
    }
  };

  return {
    get: (id: string): Answer | undefined => answers.get(id),
    set(id: string, answer: Answer): void {
      answers.set(id, answer);
      save();
    },
    clear(id: string): void {
      answers.delete(id);
      save();
    },
    clearAll(): void {
      answers.clear();
      save();
    },
    summary() {
      const done = answers.size;
      const correct = [...answers.values()].filter((a) => a.status === "correct").length;
      return { total: ids.length, done, correct };
    },
  };
}

/** For the start page, which does not load the questions. Null until something is finished. */
export function chapterSummary(chapter: string, store: Store | null = browserStore()): { total: number; done: number } | null {
  const saved = readAll(store)[chapter];
  if (!saved || !Number.isInteger(saved.total) || saved.total <= 0 || !saved.answers || typeof saved.answers !== "object") return null;
  const done = Math.min(saved.total, Object.values(saved.answers).filter(validAnswer).length);
  return done ? { total: saved.total, done } : null;
}
