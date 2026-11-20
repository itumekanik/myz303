/**
 * The site's version, shown on the start page and in every chapter's menu. CHANGELOG.md (in the
 * repository root) is the source: its newest "## [x.y.z] - YYYY-MM-DD" heading is the version and
 * date of this build, and package.json's "version" must say the same, so releasing is one edit in
 * each and a forgotten one fails the build.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const RELEASE = /^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})\s*$/m;

/** The newest release in a changelog's text: { version, date }. */
export function latestRelease(changelog) {
  const match = RELEASE.exec(changelog.replace(/\r\n/g, "\n"));
  if (!match) throw new Error('CHANGELOG.md: no release heading like "## [1.2.0] - 2026-11-20"');
  return { version: match[1], date: match[2] };
}

/** The commit the build was made from (short hash), or null outside a git checkout. */
function commit(root) {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], {
      cwd: root,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

/** { version, date, commit } of this build; web/ is `root`, CHANGELOG.md is next to it. */
export function readRelease(root) {
  const release = latestRelease(readFileSync(join(root, "..", "CHANGELOG.md"), "utf8"));
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8")).version;
  if (pkg !== release.version) {
    throw new Error(`CHANGELOG.md's newest release is ${release.version} but package.json says ${pkg}: update both`);
  }
  return { ...release, commit: commit(root) };
}
