/**
 * The learning path. scripts/build.mjs reads every src/chapters/<id>/chapter.json, sorts them by
 * "order" and injects the list as __CHAPTERS__, so adding a chapter folder is enough to put it
 * on the path (in the chapter menu, on the start page and in glossary links).
 */

export interface Text {
  tr: string;
  en: string;
}

export interface ChapterInfo {
  id: string;
  /** 1, 2, 3 … in path order. */
  number: number;
  /** The chapter's page ("perceptron.html"), or null while the chapter is still to come. */
  url: string | null;
  year: string;
  short: string;
  title: Text;
  sub: Text;
}

declare const __CHAPTERS__: ChapterInfo[];

export const CHAPTERS: ChapterInfo[] = __CHAPTERS__;
