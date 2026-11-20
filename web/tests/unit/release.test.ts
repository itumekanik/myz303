import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { latestRelease, readRelease } from "../../scripts/release.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "../..");

describe("release", () => {
  it("takes the newest release heading, not the ones below it", () => {
    const text = "# Changelog\r\n\r\nIntro.\r\n\r\n## [1.2.0] - 2026-12-01\r\n\r\n- x\r\n\r\n## [1.1.0] - 2026-11-25\r\n";
    assert.deepEqual(latestRelease(text), { version: "1.2.0", date: "2026-12-01" });
  });

  it("refuses a changelog without a release", () => {
    assert.throws(() => latestRelease("# Changelog\n\n## Unreleased\n"), /no release heading/);
  });

  it("CHANGELOG.md and package.json give the same version", () => {
    const release = readRelease(root);
    const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8")) as { version: string };
    assert.equal(release.version, pkg.version);
    assert.match(release.date, /^\d{4}-\d{2}-\d{2}$/);
  });
});
