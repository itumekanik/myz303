/**
 * Finds the chapters of the learning path: every src/chapters/<id>/chapter.json.
 * See src/chapters/chapter.schema.json for the fields.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const TAB_IDS = ["train", "infer", "detect", "theory", "play", "math", "code", "practice"];
const isText = (v) => v && typeof v.tr === "string" && typeof v.en === "string";

/** Throws with a readable message when a chapter.json is wrong. */
export function validateChapter(c, folder) {
  const fail = (msg) => {
    throw new Error(`src/chapters/${folder}/chapter.json: ${msg}`);
  };
  if (c.id !== folder) fail(`"id" must be the folder name "${folder}"`);
  if (!/^[a-z][a-z0-9-]*$/.test(c.id)) fail(`"id" may contain only a-z, 0-9 and "-"`);
  if (!Number.isInteger(c.order) || c.order < 1) fail(`"order" must be a whole number ≥ 1`);
  if (!["ready", "planned"].includes(c.status)) fail(`"status" must be "ready" or "planned"`);
  for (const key of ["year", "short"]) if (typeof c[key] !== "string") fail(`"${key}" must be a string`);
  for (const key of ["title", "sub"]) if (!isText(c[key])) fail(`"${key}" must be { "tr": …, "en": … }`);
  if (c.status === "ready") {
    const p = c.page;
    if (!p) fail(`a ready chapter needs "page"`);
    if (!isText(p.title)) fail(`"page.title" must be { "tr": …, "en": … }`);
    if (p.brand !== undefined && !isText(p.brand)) fail(`"page.brand" must be { "tr": …, "en": … }`);
    if (!Array.isArray(p.tabs) || !p.tabs.length || p.tabs.some((t) => !TAB_IDS.includes(t)))
      fail(`"page.tabs" must list tabs from ${TAB_IDS.join(", ")}`);
  }
  return c;
}

/**
 * All chapters, in path order, each with its number (1, 2, …), its page url (null while planned)
 * and the folder it came from.
 */
export function loadChapters(chaptersDir) {
  const folders = readdirSync(chaptersDir, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name);
  const chapters = [];
  for (const folder of folders) {
    const file = join(chaptersDir, folder, "chapter.json");
    if (!existsSync(file)) continue;
    const c = validateChapter(JSON.parse(readFileSync(file, "utf8")), folder);
    if (c.status === "ready") {
      for (const part of ["main.js", "panels.html", "logo.svg"]) {
        if (!existsSync(join(chaptersDir, folder, part))) throw new Error(`src/chapters/${folder}: a ready chapter needs ${part}`);
      }
    }
    chapters.push({ ...c, dir: join(chaptersDir, folder) });
  }
  chapters.sort((a, b) => a.order - b.order);
  const orders = chapters.map((c) => c.order);
  const twice = orders.find((o, i) => orders.indexOf(o) !== i);
  if (twice !== undefined) throw new Error(`two chapters have "order": ${twice}`);
  return chapters.map((c, i) => ({ ...c, number: i + 1, url: c.status === "ready" ? `${c.id}.html` : null }));
}

/** What the pages need at run time (shared/chapters.ts, __CHAPTERS__). */
export const runtimeList = (chapters) =>
  chapters.map(({ id, number, url, year, short, title, sub }) => ({ id, number, url, year, short, title, sub }));
