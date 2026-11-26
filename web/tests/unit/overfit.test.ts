import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { OVERFIT, advanceOverfit, startOverfit } from "../../src/chapters/mlp/overfit.ts";

/** What Theory 6 of the MLP chapter says about its demo (the first data set). */
describe("the overfitting demo (mlp/overfit.ts)", () => {
  const finished = (hidden: number) => {
    const run = startOverfit(hidden);
    advanceOverfit(run, OVERFIT.epochs);
    return { run, end: run.history.at(-1)! };
  };

  it("uses 12 training and 60 validation points, and trains on the training points only", () => {
    const { train, val } = startOverfit(2);
    assert.equal(train.length, 12);
    assert.equal(val.length, 60);
  });

  it("with 2 neurons the two errors stay close: no overfitting to speak of", () => {
    const { run, end } = finished(2);
    assert.equal(end.epoch, OVERFIT.epochs);
    assert.ok(end.val < run.best.val * 1.2, `validation ${end.val} against its lowest ${run.best.val}`);
  });

  it("with 40 neurons the training error goes to zero while the validation error rises again", () => {
    const { run, end } = finished(40);
    assert.ok(end.train < 0.01, `training error ${end.train}`);
    assert.ok(run.best.epoch > 0 && run.best.epoch < 1000, `lowest validation error at epoch ${run.best.epoch}`);
    assert.ok(end.val > run.best.val * 1.5, `validation ${end.val} against its lowest ${run.best.val}`);
  });

  it("keeps the network of the best moment, as early stopping would", () => {
    const { run, end } = finished(40);
    assert.equal(run.best.net.epoch, run.best.epoch);
    assert.notEqual(run.best.net, run.net);
    assert.ok(run.best.val < end.val);
  });
});
