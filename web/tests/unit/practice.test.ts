import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { PRACTICE } from "../../src/chapters/perceptron/questions.ts";
import { ruleStep, stepFn, trainFromZero, zOf } from "../../src/chapters/perceptron/model.ts";
import { chapterSummary, createProgress, matches, parseNumber, plainMath } from "../../src/shared/practice.ts";

function memoryStore(initial?: string) {
  let data = initial ?? null;
  return {
    getItem: () => data,
    setItem: (_key: string, value: string) => {
      data = value;
    },
    read: () => data,
  };
}

const questions = PRACTICE.flatMap((s) => s.questions);
const texts = (value: unknown): string[] => {
  if (typeof value === "function" || value === null || typeof value !== "object") return [];
  if ("tr" in value && "en" in value) return [String(value.tr), String(value.en)];
  return Object.values(value).flatMap(texts);
};

describe("a typed number (practice.ts)", () => {
  const good: [string, number][] = [
    ["0,3", 0.3],
    ["0.3", 0.3],
    [" -0,15 ", -0.15],
    ["−0.15", -0.15],
    [",5", 0.5],
    [".5", 0.5],
    ["+2", 2],
    ["64", 64],
    ["1,", 1],
  ];
  for (const [text, value] of good) it(`reads "${text}"`, () => assert.equal(parseNumber(text), value));
  for (const text of ["", " ", "abc", "1,2,3", "1.2.3", "1,2.3", "-", ",", "1e3", "Infinity", "NaN", "0x10", "2 3"]) {
    it(`rejects "${text}"`, () => assert.equal(parseNumber(text), null));
  }
  it("accepts a sum's rounding error, not a near miss", () => {
    assert.ok(matches(0.1 + 0.2, 0.3));
    assert.ok(!matches(0.31, 0.3));
  });
});

describe("labels for screen readers (practice.ts)", () => {
  it("turns a formula into plain text", () => {
    assert.equal(plainMath(String.raw`\(\hat y = 0\)`), "ŷ = 0");
    assert.equal(plainMath(String.raw`\((1,\ 1,\ -0{,}5)\)`), "(1, 1, -0,5)");
    assert.equal(plainMath(String.raw`yeni \(w_1 =\)`), "yeni w1 =");
    assert.equal(plainMath(String.raw`En küçük \(\eta\) öğrenir`), "En küçük η öğrenir");
  });

  it("leaves no TeX in the name of any input", () => {
    for (const q of questions) {
      const labels = q.kind === "number" ? [q.label] : q.kind === "choice" ? q.options.map((o) => o.text) : q.options;
      for (const label of labels) {
        for (const text of [label.tr, label.en]) {
          const name = plainMath(text);
          assert.ok(name, `${q.id}: an empty name`);
          assert.doesNotMatch(name, /[\\{}_]/, `${q.id}: ${name}`);
        }
      }
    }
  });
});

describe("progress (practice.ts)", () => {
  it("is saved, read back and counted", () => {
    const store = memoryStore();
    const first = createProgress("demo", ["a", "b", "c"], store);
    first.set("a", { status: "correct", pick: 1 });
    first.set("b", { status: "revealed" });
    assert.deepEqual(first.summary(), { total: 3, done: 2, correct: 1 });
    const again = createProgress("demo", ["a", "b", "c"], store);
    assert.deepEqual(again.get("a"), { status: "correct", pick: 1 });
    assert.deepEqual(chapterSummary("demo", store), { total: 3, done: 2 });
  });

  it("keeps the other chapters, and forgets a chapter when it is reset", () => {
    const store = memoryStore();
    createProgress("one", ["a"], store).set("a", { status: "correct" });
    const two = createProgress("two", ["b"], store);
    two.set("b", { status: "missed" });
    two.clearAll();
    assert.equal(chapterSummary("two", store), null);
    assert.deepEqual(chapterSummary("one", store), { total: 1, done: 1 });
  });

  it("ignores questions that no longer exist and damaged records", () => {
    const store = memoryStore(
      JSON.stringify({ demo: { total: 3, answers: { gone: { status: "correct" }, a: { status: "nope" }, b: { status: "missed" } } } }),
    );
    const p = createProgress("demo", ["a", "b"], store);
    assert.equal(p.get("a"), undefined);
    assert.deepEqual(p.summary(), { total: 2, done: 1, correct: 0 });
    for (const broken of ["not json", "[]", "null", '{"demo":5}', '{"demo":{"total":"x"}}']) {
      const s = memoryStore(broken);
      assert.equal(chapterSummary("demo", s), null);
      assert.equal(createProgress("demo", ["a"], s).summary().done, 0);
    }
  });

  it("works without storage, and when writing fails", () => {
    const none = createProgress("demo", ["a"], null);
    none.set("a", { status: "correct" });
    assert.equal(none.summary().done, 1);
    const full = createProgress("demo", ["a"], {
      getItem: () => null,
      setItem: () => {
        throw new Error("quota");
      },
    });
    full.set("a", { status: "correct" });
    assert.equal(full.summary().done, 1);
  });
});

describe("the perceptron model", () => {
  it("outputs 1 only for z > 0", () => {
    assert.equal(stepFn(1e-9), 1);
    assert.equal(stepFn(0), 0);
    assert.equal(stepFn(-1), 0);
  });

  it("applies the rule only after a mistake", () => {
    const model = { w: [0, 0], b: 0 };
    assert.equal(ruleStep(model, [1, 1], 0, 0.5).e, 0);
    assert.deepEqual(model, { w: [0, 0], b: 0 });
    const step = ruleStep(model, [1, 0], 1, 0.5);
    assert.equal(step.e, 1);
    assert.deepEqual(model, { w: [0.5, 0], b: 0.5 });
  });

  it("learns AND, never XOR, and XOR with the product feature", () => {
    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    const and = trainFromZero(
      X.map((x, i) => ({ x, y: [0, 0, 0, 1][i]! })),
      0.25,
      100,
    );
    assert.ok(and.converged);
    X.forEach((x, i) => assert.equal(stepFn(zOf(and.model.w, and.model.b, x)), [0, 0, 0, 1][i]));
    assert.ok(
      !trainFromZero(
        X.map((x, i) => ({ x, y: [0, 1, 1, 0][i]! })),
        0.25,
        1000,
      ).converged,
    );
    assert.ok(
      trainFromZero(
        X.map(([a, b], i) => ({ x: [a!, b!, a! * b!], y: [0, 1, 1, 0][i]! })),
        0.25,
        200,
      ).converged,
    );
  });
});

describe("the perceptron's questions", () => {
  it("have unique ids, in the sections of the Theory tab", () => {
    const ids = questions.map((q) => q.id);
    assert.equal(new Set(ids).size, ids.length);
    assert.equal(ids.length, 15);
    assert.deepEqual(
      PRACTICE.map((s) => s.read.section),
      ["p-neuron", "p-boundary", "p-rule", "p-converge", "p-xor"],
    );
  });

  it("are written in both languages", () => {
    for (const text of texts(PRACTICE)) assert.ok(text.trim(), "an empty text");
    for (const q of questions) {
      assert.notEqual(q.q.tr, q.q.en, q.id);
      assert.notEqual(q.explain.tr, q.explain.en, q.id);
    }
  });

  it("do not give away the answer by its position", () => {
    const positions = new Set(questions.filter((q) => q.kind === "choice").map((q) => q.answer));
    assert.ok(positions.size >= 3, "the right option is in the same few places");
  });

  for (const q of questions) {
    it(`${q.id}: the stated answer is what the model computes`, () => {
      if (q.kind === "number") {
        assert.ok(matches(q.compute(), q.answer), `computed ${q.compute()}, stated ${q.answer}`);
      } else if (q.kind === "predict") {
        const run = q.run();
        assert.equal(run.answer, q.answer);
        assert.ok(q.answer >= 0 && q.answer < q.options.length);
        assert.ok(run.result.tr && run.result.en);
      } else {
        assert.ok(q.answer >= 0 && q.answer < q.options.length);
        if (q.compute) assert.equal(q.compute(), q.answer);
        q.options.forEach((option, k) => {
          if (k !== q.answer) assert.ok(option.why, `option ${k} does not say why it is wrong`);
        });
      }
    });
  }
});
