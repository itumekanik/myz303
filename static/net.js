"use strict";
/*
 * A small fully connected network (MLP) in plain JavaScript, for the chapter pages' demos:
 * forward pass, backpropagation, gradient descent (plain, or Adam) and a numerical gradient
 * check. Self-contained: no DOM, no other script.
 *
 *   const net = NET.create([2, 4, 1], { act: "tanh", out: "sigmoid", rand });
 *   NET.trainEpoch(net, data, { lr: 0.3, batch: 10, rand });   // data: [{ x: [..], y }]
 *   NET.predict(net, [0.5, -1]);                                // → output (a probability)
 *
 * Layer l has W[j][i] (from input i to neuron j), b[j] and an activation. The loss is binary
 * cross-entropy with a sigmoid output (the output error is then ŷ − y), or the mean squared
 * error ½(ŷ − y)² (any output activation).
 */
const NET = (() => {
const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
/** f: the activation; d(z, a): its derivative, given z and a = f(z). */
const ACT = {
  sigmoid: { f: sigmoid, d: (z, a) => a * (1 - a) },
  tanh: { f: Math.tanh, d: (z, a) => 1 - a * a },
  relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
  linear: { f: (z) => z, d: () => 1 },
  step: { f: (z) => (z > 0 ? 1 : 0), d: () => 0 },
};
const gauss = (rand) => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

/** New network with random weights: N(0, 1/n_in) (N(0, 2/n_in) for ReLU) and zero biases. */
function create(sizes, { act = "tanh", out = "sigmoid", rand = Math.random } = {}) {
  const layers = [];
  for (let l = 1; l < sizes.length; l++) {
    const nIn = sizes[l - 1];
    const fn = l === sizes.length - 1 ? out : act;
    const scale = Math.sqrt((fn === "relu" ? 2 : 1) / nIn);
    layers.push({
      act: fn,
      W: Array.from({ length: sizes[l] }, () => Array.from({ length: nIn }, () => gauss(rand) * scale)),
      b: new Array(sizes[l]).fill(0),
    });
  }
  return { sizes: [...sizes], layers, epoch: 0 };
}

const clone = (net) => ({ sizes: [...net.sizes], epoch: net.epoch,
  layers: net.layers.map((L) => ({ act: L.act, W: L.W.map((r) => [...r]), b: [...L.b] })) });

/** Every layer's z and a (as[0] is the input). */
function forward(net, x) {
  const zs = [];
  const as = [x];
  let a = x;
  for (const L of net.layers) {
    const z = L.W.map((row, j) => { let s = L.b[j]; for (let i = 0; i < row.length; i++) s += row[i] * a[i]; return s; });
    const f = ACT[L.act].f;
    a = z.map(f);
    zs.push(z);
    as.push(a);
  }
  return { zs, as, out: a };
}
const predict = (net, x) => forward(net, x).out[0];

/**
 * A fast predict for drawing (no allocations per call): returns (x1, x2) → output, and fills
 * `hidden` with every neuron's activation when given (layer by layer, neuron by neuron).
 */
function predictor(net) {
  const bufs = net.layers.map((L) => new Float64Array(L.b.length));
  const fs = net.layers.map((L) => ACT[L.act].f);
  return (x1, x2, hidden = null) => {
    let prev = null;
    let k = 0;
    for (let l = 0; l < net.layers.length; l++) {
      const L = net.layers[l];
      const out = bufs[l];
      for (let j = 0; j < L.b.length; j++) {
        const row = L.W[j];
        let s = L.b[j];
        if (l === 0) s += row[0] * x1 + (row.length > 1 ? row[1] * x2 : 0);
        else for (let i = 0; i < row.length; i++) s += row[i] * prev[i];
        out[j] = fs[l](s);
        if (hidden) hidden[k++] = out[j];
      }
      prev = out;
    }
    return prev[0];
  };
}

/** The loss of one prediction. */
function lossOne(p, y, loss = "bce") {
  if (loss === "mse") return 0.5 * (p - y) ** 2;
  const e = 1e-12;
  return -(y * Math.log(Math.max(p, e)) + (1 - y) * Math.log(Math.max(1 - p, e)));
}
const lossOf = (net, data, loss = "bce") =>
  data.length ? data.reduce((s, d) => s + lossOne(predict(net, d.x), d.y, loss), 0) / data.length : 0;
const accuracy = (net, data) =>
  data.length ? data.filter((d) => (predict(net, d.x) > 0.5 ? 1 : 0) === d.y).length / data.length : 0;

/** Backpropagation for one example: the gradient of every weight and bias, and the deltas ∂L/∂z. */
function backward(net, cache, y, loss = "bce") {
  const n = net.layers.length;
  const deltas = new Array(n);
  const last = net.layers[n - 1];
  deltas[n - 1] = cache.as[n].map((a, j) => {
    const yj = Array.isArray(y) ? y[j] : y;
    if (loss === "bce" && last.act === "sigmoid") return a - yj;   // σ' cancels with the loss's 1/(ŷ(1−ŷ))
    return (a - yj) * ACT[last.act].d(cache.zs[n - 1][j], a);
  });
  for (let l = n - 2; l >= 0; l--) {
    const next = net.layers[l + 1];
    const d = ACT[net.layers[l].act].d;
    deltas[l] = cache.zs[l].map((z, j) => {
      let s = 0;
      for (let k = 0; k < next.W.length; k++) s += next.W[k][j] * deltas[l + 1][k];
      return s * d(z, cache.as[l + 1][j]);
    });
  }
  const grads = deltas.map((delta, l) => ({ W: delta.map((dj) => cache.as[l].map((a) => dj * a)), b: [...delta] }));
  return { grads, deltas };
}

/** The average gradient over some examples. */
function gradient(net, batch, loss = "bce") {
  const sum = net.layers.map((L) => ({ W: L.W.map((r) => r.map(() => 0)), b: L.b.map(() => 0) }));
  for (const d of batch) {
    const { grads } = backward(net, forward(net, d.x), d.y, loss);
    grads.forEach((g, l) => {
      g.b.forEach((v, j) => { sum[l].b[j] += v; });
      g.W.forEach((row, j) => row.forEach((v, i) => { sum[l].W[j][i] += v; }));
    });
  }
  const k = 1 / Math.max(1, batch.length);
  for (const g of sum) { g.b = g.b.map((v) => v * k); g.W = g.W.map((r) => r.map((v) => v * k)); }
  return sum;
}

/** One update: plain gradient descent, or Adam (opt.state keeps its moving averages). */
function apply(net, grads, { lr, optimizer = "sgd", state = null }) {
  if (optimizer === "adam") {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    if (!state.m) {
      state.t = 0;
      state.m = grads.map((g) => ({ W: g.W.map((r) => r.map(() => 0)), b: g.b.map(() => 0) }));
      state.v = grads.map((g) => ({ W: g.W.map((r) => r.map(() => 0)), b: g.b.map(() => 0) }));
    }
    state.t += 1;
    const c1 = 1 - b1 ** state.t;
    const c2 = 1 - b2 ** state.t;
    const upd = (m, v, g) => {
      const mi = b1 * m + (1 - b1) * g;
      const vi = b2 * v + (1 - b2) * g * g;
      return [mi, vi, (lr * (mi / c1)) / (Math.sqrt(vi / c2) + eps)];
    };
    net.layers.forEach((L, l) => {
      L.b.forEach((_, j) => {
        const [m, v, step] = upd(state.m[l].b[j], state.v[l].b[j], grads[l].b[j]);
        state.m[l].b[j] = m; state.v[l].b[j] = v; L.b[j] -= step;
        L.W[j].forEach((__, i) => {
          const [m2, v2, s2] = upd(state.m[l].W[j][i], state.v[l].W[j][i], grads[l].W[j][i]);
          state.m[l].W[j][i] = m2; state.v[l].W[j][i] = v2; L.W[j][i] -= s2;
        });
      });
    });
    return;
  }
  net.layers.forEach((L, l) => {
    L.b.forEach((_, j) => {
      L.b[j] -= lr * grads[l].b[j];
      L.W[j].forEach((__, i) => { L.W[j][i] -= lr * grads[l].W[j][i]; });
    });
  });
}

/** One pass over the data in batches (batch = data.length: full-batch gradient descent). */
function trainEpoch(net, data, { lr = 0.1, batch = data.length, loss = "bce", optimizer = "sgd", state = null, rand = Math.random } = {}) {
  if (!data.length) return;
  let order = data;
  if (batch < data.length) {
    order = [...data];
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
  }
  for (let s = 0; s < order.length; s += batch) {
    apply(net, gradient(net, order.slice(s, s + batch), loss), { lr, optimizer, state });
  }
  net.epoch += 1;
}

/** Central differences: ∂L/∂θ ≈ (L(θ + ε) − L(θ − ε)) / 2ε, for every parameter. */
function numericGradient(net, batch, loss = "bce", eps = 1e-5) {
  const L0 = () => batch.reduce((s, d) => s + lossOne(predict(net, d.x), d.y, loss), 0) / batch.length;
  return net.layers.map((L) => {
    const nudge = (get, set) => {
      const old = get();
      set(old + eps); const up = L0();
      set(old - eps); const down = L0();
      set(old);
      return (up - down) / (2 * eps);
    };
    return {
      W: L.W.map((row, j) => row.map((_, i) => nudge(() => L.W[j][i], (v) => { L.W[j][i] = v; }))),
      b: L.b.map((_, j) => nudge(() => L.b[j], (v) => { L.b[j] = v; })),
    };
  });
}

const paramCount = (sizes) => sizes.slice(1).reduce((s, n, l) => s + n * sizes[l] + n, 0);

return { ACT, sigmoid, create, clone, forward, predict, predictor, backward, gradient, apply, trainEpoch,
  lossOne, lossOf, accuracy, numericGradient, paramCount };
})();
if (typeof module !== "undefined") module.exports = NET;
