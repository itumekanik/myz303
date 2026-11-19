import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { loadChapters, runtimeList, validateChapter } from "../../scripts/chapters.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const ok = {
  id: "demo",
  order: 1,
  status: "ready",
  year: "2000",
  short: "D",
  title: { tr: "Deneme", en: "Demo" },
  sub: { tr: "a", en: "b" },
  page: { title: { tr: "D", en: "D" }, tabs: ["theory"] },
};

describe("chapter discovery (scripts/chapters.mjs)", () => {
  it("finds the chapters of the course in path order, the CNN still to come", () => {
    const chapters = loadChapters(join(here, "../../src/chapters"));
    assert.deepEqual(
      chapters.map((c) => c.id),
      ["perceptron", "mlp", "cnn"],
    );
    assert.deepEqual(
      chapters.map((c) => c.number),
      [1, 2, 3],
    );
    assert.deepEqual(
      chapters.map((c) => c.url),
      ["perceptron.html", "mlp.html", null],
    );
    assert.ok(!("dir" in runtimeList(chapters)[0]));
  });

  it("accepts a valid chapter.json", () => {
    assert.equal(validateChapter(ok, "demo"), ok);
    assert.equal(validateChapter({ ...ok, status: "planned", page: undefined }, "demo").status, "planned");
  });

  const wrong: [object, RegExp][] = [
    [{ ...ok, id: "other" }, /folder name/],
    [{ ...ok, order: 0 }, /order/],
    [{ ...ok, status: "draft" }, /status/],
    [{ ...ok, title: { tr: "x" } }, /title/],
    [{ ...ok, page: undefined }, /needs "page"/],
    [{ ...ok, page: { ...ok.page, tabs: ["nope"] } }, /tabs/],
  ];
  wrong.forEach(([chapter, message], k) => {
    it(`rejects a wrong chapter.json (${k + 1}: ${message.source})`, () => {
      assert.throws(() => validateChapter(chapter, "demo"), message);
    });
  });

  it("requires the files of a ready chapter and unique orders", () => {
    const dir = mkdtempSync(join(tmpdir(), "chapters-"));
    mkdirSync(join(dir, "demo"));
    writeFileSync(join(dir, "demo", "chapter.json"), JSON.stringify(ok));
    assert.throws(() => loadChapters(dir), /needs main\.js/);
    for (const f of ["main.js", "panels.html", "logo.svg"]) writeFileSync(join(dir, "demo", f), "");
    assert.equal(loadChapters(dir).length, 1);
    mkdirSync(join(dir, "twin"));
    writeFileSync(join(dir, "twin", "chapter.json"), JSON.stringify({ ...ok, id: "twin", status: "planned" }));
    assert.throws(() => loadChapters(dir), /two chapters have "order": 1/);
  });
});
