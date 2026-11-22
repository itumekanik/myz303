/**
 * The perceptron itself: the forward computation and the learning rule. The Theory demos, the
 * Playground and the Practice questions all run this code, and the unit tests check the answers
 * of the questions against it.
 */
export interface Model {
  w: number[];
  b: number;
}

export interface Example {
  x: readonly number[];
  y: number;
}

export const stepFn = (z: number): 0 | 1 => (z > 0 ? 1 : 0);

export function zOf(w: readonly number[], b: number, x: readonly number[]): number {
  let z = 0;
  for (let i = 0; i < w.length; i++) z += w[i]! * x[i]!;
  return z + b;
}

/** One step of the perceptron rule on example (x, y). Returns what happened. */
export function ruleStep(model: Model, x: readonly number[], y: number, eta: number) {
  const z = zOf(model.w, model.b, x);
  const yHat = stepFn(z);
  const e = y - yHat;
  const before = { w: [...model.w], b: model.b };
  if (e) {
    for (let i = 0; i < model.w.length; i++) model.w[i]! += eta * e * x[i]!;
    model.b += eta * e;
  }
  return { x, y, z, yHat, e, before, after: { w: [...model.w], b: model.b } };
}

/** Trains from zero weights, in the given order, until an epoch without a mistake or the limit. */
export function trainFromZero(points: readonly Example[], eta: number, maxEpochs: number) {
  const model: Model = { w: new Array<number>(points[0]!.x.length).fill(0), b: 0 };
  let epochs = 0;
  let lastMistakes = 0;
  while (epochs < maxEpochs) {
    epochs++;
    lastMistakes = 0;
    for (const p of points) if (ruleStep(model, p.x, p.y, eta).e) lastMistakes++;
    if (!lastMistakes) break;
  }
  return { model, epochs, lastMistakes, converged: lastMistakes === 0 };
}
