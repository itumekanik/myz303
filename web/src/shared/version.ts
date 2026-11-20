/**
 * The site's version: the newest release in CHANGELOG.md, injected by scripts/build.mjs as
 * __RELEASE__ (see scripts/release.mjs). Shown at the bottom of the start page and of the
 * chapter menu, with a link to the list of changes.
 */
import { I18N, L } from "./i18n.js";

export interface Release {
  /** "1.0.0" */
  version: string;
  /** The release date, "2026-11-20". */
  date: string;
  /** The commit the site was built from (short hash), or null when built outside git. */
  commit: string | null;
}

declare const __RELEASE__: Release;

export const RELEASE: Release = __RELEASE__;

export const REPO = "https://github.com/itumekanik/myz303";
export const CHANGELOG_URL = `${REPO}/blob/main/CHANGELOG.md`;

/** The release date in the page's language: "20 Kasım 2026", "20 November 2026". */
export function releaseDate(): string {
  return new Intl.DateTimeFormat(I18N.lang === "en" ? "en-GB" : "tr-TR", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${RELEASE.date}T00:00:00Z`));
}

/**
 * "Sürüm 1.0.0 · 20 Kasım 2026 · Neler değişti?" (`full`: also the build's commit). The short
 * form keeps the date and commit in its tooltip.
 */
export function versionLine(className: string, { full = false } = {}): HTMLElement {
  const line = document.createElement("p");
  line.className = className;
  const build = RELEASE.commit ? `${L({ tr: "derleme", en: "build" })} ${RELEASE.commit}` : null;
  const version = document.createElement("span");
  version.className = "version-num";
  version.textContent = `${L({ tr: "Sürüm", en: "Version" })} ${RELEASE.version}`;
  const changes = document.createElement("a");
  changes.href = CHANGELOG_URL;
  changes.target = "_blank"; // a chapter page keeps its state (a training run in the CNN chapter)
  changes.rel = "noopener";
  changes.textContent = L({ tr: "Neler değişti?", en: "What's new?" });
  const parts: (string | null)[] = full ? [releaseDate(), build] : [];
  if (!full) line.title = [releaseDate(), build].filter(Boolean).join(" · ");
  line.append(version, ...parts.filter((p): p is string => p !== null).map((p) => ` · ${p}`), " · ", changes);
  return line;
}
