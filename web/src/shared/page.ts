/**
 * What the shared code needs to know about the page it runs on.
 *
 * - TABS: the page's tabs. Each chapter fills it (Object.assign) before the page starts; core.js
 *   reads it when it shows a tab or parses the address.
 * - hooks: functions of modules that load after core.js (the glossary panel), which core.js calls
 *   only later, at run time. Going through hooks keeps the module graph free of cycles.
 */

export interface TabSpec {
  /** Selector of the tab button. */
  tab: string;
  /** Selector of the tab's panel. */
  panel: string;
  /** Address of the tab, e.g. "#teori". */
  hash: string;
}

export const TABS: Record<string, TabSpec> = {};

export interface PageHooks {
  /** Set by the glossary panel. */
  glossaryIsOpen?: () => boolean;
  syncGlossary?: (state: { glossary?: boolean; term?: string }) => void;
  /** Set by a page with info popups (the CNN chapter's Training tab). */
  closeInfo?: () => void;
  infoPopOpen?: () => boolean;
  /** Set by a page whose Coding tab names its steps (the CNN chapter). */
  codeSteps?: { id: string; title: unknown }[];
}

export const hooks: PageHooks = {};
