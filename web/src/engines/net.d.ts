/** Types of net.js, the small MLP engine of the chapter demos (forward, backprop, SGD / Adam). */

export type Activation = "sigmoid" | "tanh" | "relu" | "linear" | "step";
export type Loss = "bce" | "mse";

export interface Layer {
  act: Activation;
  /** W[j][i]: from input i to neuron j. */
  W: number[][];
  b: number[];
}

export interface Net {
  sizes: number[];
  layers: Layer[];
  epoch: number;
}

/** A gradient (or Adam moment) with the shape of the layers. */
export interface LayerGrad {
  W: number[][];
  b: number[];
}

export interface Example {
  x: number[];
  /** Target: 0 or 1 for cross-entropy, any number for the squared error. */
  y: number;
}

export interface AdamState {
  t?: number;
  m?: LayerGrad[];
  v?: LayerGrad[];
}

export interface ApplyOptions {
  lr: number;
  optimizer?: "sgd" | "adam";
  /** Adam's moments; pass the same object at every step. */
  state?: AdamState | null;
}

export interface TrainOptions extends Partial<ApplyOptions> {
  /** Examples per step; data.length (the default) is full-batch gradient descent. */
  batch?: number;
  loss?: Loss;
  rand?: () => number;
}

export interface Forward {
  zs: number[][];
  /** as[0] is the input. */
  as: number[][];
  out: number[];
}

export declare const NET: {
  ACT: Record<Activation, { f: (z: number) => number; d: (z: number, a: number) => number }>;
  sigmoid(z: number): number;
  /** Weights from N(0, 1/n_in) (N(0, 2/n_in) for ReLU), zero biases. */
  create(sizes: number[], options?: { act?: Activation; out?: Activation; rand?: () => number }): Net;
  clone(net: Net): Net;
  forward(net: Net, x: number[]): Forward;
  predict(net: Net, x: number[]): number;
  /** A fast, allocation-free predictor for two inputs; fills hidden[l][j] when given. */
  predictor(net: Net): (x1: number, x2: number, hidden?: number[][] | null) => number;
  lossOne(p: number, y: number, loss?: Loss): number;
  lossOf(net: Net, data: Example[], loss?: Loss): number;
  accuracy(net: Net, data: Example[]): number;
  backward(net: Net, cache: Forward, y: number, loss?: Loss): { grads: LayerGrad[]; deltas: number[][] };
  /** Mean gradient of the loss over a batch. */
  gradient(net: Net, batch: Example[], loss?: Loss): LayerGrad[];
  apply(net: Net, grads: LayerGrad[], options: ApplyOptions): void;
  trainEpoch(net: Net, data: Example[], options?: TrainOptions): void;
  /** Central differences, for checking gradient(). */
  numericGradient(net: Net, batch: Example[], loss?: Loss, eps?: number): LayerGrad[];
  /** Σ n_l (n_{l-1} + 1). */
  paramCount(sizes: number[]): number;
};
