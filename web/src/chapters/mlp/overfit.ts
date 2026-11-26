/**
 * Chapter 2, Theory 6: memorizing or generalizing. A network with one hidden layer of tanh neurons
 * is fitted to 12 noisy points of a smooth curve, and 60 more points of the same curve that it
 * never trains on (the validation set) show how well it generalizes. The Theory demo and the
 * Practice question both run this code; the unit tests check the question's answer against it.
 */
import { NET, type Example, type Net } from "../../engines/net.js";

/** The settings of the demo: few, noisy points and Adam with a small step, so the curves unfold slowly. */
export const OVERFIT = { trainPoints: 12, valPoints: 60, noise: 0.3, lr: 0.01, epochs: 4000, every: 10, sizes: [2, 8, 40] } as const;

/** The curve the points are taken from; the network never sees it, only the noisy points. */
export const truth = (x: number): number => 0.8 * Math.sin(2 * x);

/** The same "random" numbers on every computer (mulberry32). */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (rand: () => number): number => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

/** n points spread over [−1, 1], each the curve's value plus noise. */
function noisyPoints(seed: number, n: number): Example[] {
  const rand = seeded(seed);
  return Array.from({ length: n }, (_, i) => {
    const x = -1 + (2 * (i + 0.2 + 0.6 * rand())) / n;
    return { x: [x], y: truth(x) + OVERFIT.noise * gauss(rand) };
  });
}

/** The training and the validation points of data set number `data` (1, 2, 3 …). */
export function overfitData(data = 1): { train: Example[]; val: Example[] } {
  return { train: noisyPoints(data * 2 + 1, OVERFIT.trainPoints), val: noisyPoints(data * 2 + 2, OVERFIT.valPoints) };
}

/** The mean squared error, (ŷ − y)² averaged over the points. */
export const meanSquaredError = (net: Net, data: Example[]): number =>
  data.reduce((sum, d) => sum + (NET.predict(net, d.x) - d.y) ** 2, 0) / data.length;

export interface OverfitPoint {
  epoch: number;
  train: number;
  val: number;
}

export interface OverfitRun {
  hidden: number;
  net: Net;
  train: Example[];
  val: Example[];
  /** Both errors every `OVERFIT.every` epochs, from epoch 0. */
  history: OverfitPoint[];
  /** The network at its lowest validation error so far: where early stopping would have stopped. */
  best: { epoch: number; val: number; net: Net };
  state: object;
}

/** A new run: 1 input → `hidden` tanh neurons → 1 linear output, not trained yet. */
export function startOverfit(hidden: number, data = 1): OverfitRun {
  const { train, val } = overfitData(data);
  const net = NET.create([1, hidden, 1], { act: "tanh", out: "linear", rand: seeded(7 + hidden) });
  const first = { epoch: 0, train: meanSquaredError(net, train), val: meanSquaredError(net, val) };
  return { hidden, net, train, val, history: [first], best: { epoch: 0, val: first.val, net: NET.clone(net) }, state: {} };
}

/** Trains `epochs` more epochs (full batch, squared error, Adam), up to OVERFIT.epochs in all. */
export function advanceOverfit(run: OverfitRun, epochs: number): void {
  for (let k = 0; k < epochs && run.net.epoch < OVERFIT.epochs; k++) {
    NET.trainEpoch(run.net, run.train, { lr: OVERFIT.lr, loss: "mse", optimizer: "adam", state: run.state });
    if (run.net.epoch % OVERFIT.every) continue;
    const point = { epoch: run.net.epoch, train: meanSquaredError(run.net, run.train), val: meanSquaredError(run.net, run.val) };
    run.history.push(point);
    if (point.val < run.best.val) run.best = { epoch: point.epoch, val: point.val, net: NET.clone(run.net) };
  }
}
