# ----------------------------------------------------------------------
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
============================
         Perceptron
============================

Rosenblatt's perceptron (1958), written from scratch with NumPy: a single
artificial neuron that learns the logic gates AND, OR and NAND from their four
examples, and fails on XOR, which no single straight line can separate.

    python Perceptron.py

The Perceptron chapter of the dashboard (http://127.0.0.1:8000/perceptron)
explains every line of this file and trains the same perceptron in the browser.
"""
import numpy as np


#--------------- Hyperparameters ---------------#
learning_rate = 0.1  # eta: how far one correction moves the weights
max_epochs = 20  # an epoch is one pass over all the examples
seed = 0  # fixed random seed: the same start gives the same run

#------------------- The data ------------------#
# Every possible input of a gate with two inputs, one example per row
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

# The answer (target) of each gate for those four inputs
GATES = {
    "AND":  np.array([0, 0, 0, 1]),
    "OR":   np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR":  np.array([0, 1, 1, 0]),
}


def step(z):
    """Step (Heaviside) activation: 1 where z > 0, otherwise 0."""
    return (z > 0).astype(int)


class Perceptron:
    """A single neuron: a weighted sum of the inputs plus a bias, then a step."""

    def __init__(self, n_inputs, learning_rate=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.5, n_inputs)  # small random weights
        self.b = 0.0  # bias
        self.lr = learning_rate

    def predict(self, X):
        z = X @ self.w + self.b  # the weighted sum of every row of X
        return step(z)

    def fit(self, X, y, max_epochs=20):
        """Perceptron learning rule. Returns the number of mistakes in each epoch."""
        history = []
        for epoch in range(max_epochs):
            mistakes = 0
            for x_i, y_i in zip(X, y):
                y_hat = step(x_i @ self.w + self.b)
                error = y_i - y_hat  # +1, -1 or 0
                if error != 0:  # wrong answer: move the boundary towards the right side
                    self.w += self.lr * error * x_i
                    self.b += self.lr * error
                    mistakes += 1
            history.append(mistakes)
            if mistakes == 0:  # every example is right: the perceptron has converged
                break
        return history


if __name__ == "__main__":
    for name, y in GATES.items():
        model = Perceptron(n_inputs=2, learning_rate=learning_rate, seed=seed)
        history = model.fit(X, y, max_epochs=max_epochs)
        predictions = model.predict(X)
        w1, w2 = model.w

        print(f"--- {name} ---")
        print(f"Mistakes per epoch: {history}")
        print(f"Boundary: {w1:.2f}*x1 {w2:+.2f}*x2 {model.b:+.2f} = 0")
        print(f"Predictions: {predictions}  targets: {y}  accuracy: {(predictions == y).mean():.0%}")
        if history[-1] == 0:
            print(f"Converged after {len(history)} epochs.\n")
        else:
            print(f"No convergence in {max_epochs} epochs: no straight line separates XOR.\n")
