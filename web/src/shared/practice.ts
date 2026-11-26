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
  /** How far a typed value may be from `answer`, for a rounded result (default: exact). */
  tolerance?: number;
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

/** What a model question keeps of its run: numbers only, so what is read back from storage stays harmless. */
export interface ModelRun {
  answer: number;
  values: number[];
}

/** A prediction tested on a model that has to be loaded first, so its result is saved with the answer. */
export interface ModelQuestion extends QuestionBase {
  kind: "model";
  options: Text[];
  answer: number;
  run: () => Promise<ModelRun>;
  /** The result of a run, in words. */
  describe: (values: number[]) => Text;
  tryIt: Place;
}

export type Question = ChoiceQuestion | NumberQuestion | PredictQuestion | ModelQuestion;

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

/**
 * Every value a typed number can mean. A point or comma before three digits may group thousands
 * ("1.024" is 1024 in Turkish, "1,024" in English) or be a decimal one, so both readings count:
 * "1.024" → [1.024, 1024], "1.024.000" → [1024000], "1.024,5" → [1024.5]. Empty if it is no number.
 */
export function readNumber(text: string): number[] {
  const clean = text.trim().replace(/[−–]/g, "-");
  const values: number[] = [];
  const plain = parseNumber(clean);
  if (plain !== null) values.push(plain);
  const grouped = /^([+-]?)([1-9]\d{0,2}(?:([.,])\d{3})+)(?:(?!\3)[.,](\d*))?$/.exec(clean);
  if (grouped) {
    const [, sign, digits, separator, fraction] = grouped;
    const value = Number(`${sign}${digits.split(separator).join("")}.${fraction || "0"}`);
    if (!values.includes(value)) values.push(value);
  }
  return values;
}

export const matches = (value: number, answer: number, tolerance = 1e-6): boolean => Math.abs(value - answer) <= tolerance;

/** A label with formulas as plain text, the name of its input for screen readers. */
export function plainMath(text: string): string {
  return text
    .replace(/\\[()]/g, "")
    .replace(/\\hat\s*y/g, "ŷ")
    .replace(/\\eta/g, "η")
    .replace(/\\sigma/g, "σ")
    .replace(/\\delta/g, "δ")
    .replace(/\\theta/g, "θ")
    .replace(/\\approx/g, "≈")
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
  run?: ModelRun;
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

const validRun = (r: unknown): r is ModelRun =>
  Boolean(r) &&
  typeof r === "object" &&
  Number.isInteger((r as ModelRun).answer) &&
  Array.isArray((r as ModelRun).values) &&
  (r as ModelRun).values.every((v) => Number.isFinite(v));

/** The answer as read from storage, without a damaged run. */
function cleaned(a: Answer): Answer {
  if (a.run === undefined || validRun(a.run)) return a;
  const { run: _damaged, ...rest } = a;
  return rest;
}

/** The progress of one chapter, for the questions it has now. */
export function createProgress(chapter: string, ids: readonly string[], store: Store | null = browserStore()) {
  const record = readAll(store)[chapter];
  const stored: unknown = record?.answers;
  const answers = new Map<string, Answer>();
  if (stored && typeof stored === "object") {
    for (const id of ids) {
      const a = (stored as Record<string, unknown>)[id];
      if (validAnswer(a)) answers.set(id, cleaned(a));
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
  // A newer site may have more, fewer or renamed questions: the start page reads the stored record,
  // so it is brought up to date as soon as the chapter opens
  const outdated =
    record !== undefined &&
    (record?.total !== ids.length || !stored || typeof stored !== "object" || Object.keys(stored).length !== answers.size);
  if (outdated) save();

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
