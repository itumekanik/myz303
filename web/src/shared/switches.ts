/**
 * The theme switch (light / dark / system) and the language switch in the top bar, shared by
 * the chapter pages (core.js) and the start page (app/landing.ts).
 *
 * The theme choice is saved in this browser; "system" follows the operating system. A change
 * fires "theme-change" on document, so canvases can repaint with the new colours.
 */
import { setLang } from "./i18n.js";

export type ThemeChoice = "light" | "dark" | "system";

const systemDark = matchMedia("(prefers-color-scheme: dark)");

export const themeChoice = (): ThemeChoice => (document.documentElement.dataset.theme as ThemeChoice | undefined) ?? "system";

function markThemeButtons(): void {
  for (const b of document.querySelectorAll<HTMLButtonElement>("#theme-switch button")) {
    b.setAttribute("aria-pressed", String(b.dataset.themeChoice === themeChoice()));
  }
}

export function applyTheme(choice: ThemeChoice): void {
  if (choice === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
  try {
    if (choice === "system") localStorage.removeItem("theme");
    else localStorage.setItem("theme", choice);
  } catch {
    /* storage blocked: the choice lasts until the page is reloaded */
  }
  markThemeButtons();
  document.dispatchEvent(new CustomEvent("theme-change"));
}

/** Wires the #theme-switch and #lang-switch buttons of the top bar. */
export function wireSwitches(): void {
  for (const b of document.querySelectorAll<HTMLButtonElement>("#theme-switch button")) {
    b.addEventListener("click", () => applyTheme(b.dataset.themeChoice as ThemeChoice));
  }
  systemDark.addEventListener("change", () => {
    if (themeChoice() === "system") document.dispatchEvent(new CustomEvent("theme-change"));
  });
  markThemeButtons();
  // Static text swaps in i18n.js; every view redraws itself on "lang-change"
  for (const b of document.querySelectorAll<HTMLButtonElement>("#lang-switch button")) {
    b.addEventListener("click", () => setLang(b.dataset.lang));
  }
}
