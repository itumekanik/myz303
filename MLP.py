"""
A multilayer perceptron (MLP) from scratch, with NumPy.

Chapter 2 of the learning path: a network with one hidden layer, trained with
backpropagation and gradient descent. It first checks backpropagation against
numerical derivatives, then learns XOR (which a single perceptron cannot learn)
and finally the "two moons" dataset, whose boundary is a curve.

Run:  python MLP.py
"""
import numpy as np


#--------------- Hyperparameters ---------------#
hidden_units = 8
learning_rate = 0.5
epochs_xor = 2000
epochs_moons = 300
batch_size = 16
seed = 0
#-----------------------------------------------#

rng = np.random.default_rng(seed)

# XOR: the four inputs and their labels
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0], dtype=float)


def make_moons(n, noise=0.25):
    """Two interleaving half circles, one per class."""
    t = rng.uniform(0, np.pi, n)
    labels = rng.integers(0, 2, n)
    x = np.where(labels == 0, np.cos(t), 1 - np.cos(t))
    y = np.where(labels == 0, np.sin(t), 0.5 - np.sin(t))
    X = np.stack([x, y], axis=1) + rng.normal(0, noise, (n, 2))
    return X, labels.astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh_derivative(a):
    """The derivative of tanh, written with its output a = tanh(z)."""
    return 1 - a ** 2


class MLP:
    """Inputs -> one hidden layer (tanh) -> one output (sigmoid)."""

    def __init__(self, n_inputs, n_hidden):
        # Random weights scaled by 1/sqrt(number of inputs); zero biases
        self.W1 = rng.normal(0, 1 / np.sqrt(n_inputs), (n_inputs, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        """Forward pass for a batch X of shape (n, n_inputs); keeps what backward needs."""
        self.X = X
        self.Z1 = X @ self.W1 + self.b1        # (n, n_hidden)
        self.H = np.tanh(self.Z1)              # the hidden layer's outputs
        self.Z2 = self.H @ self.W2 + self.b2   # (n, 1)
        self.Y_hat = sigmoid(self.Z2)          # the probability of class 1
        return self.Y_hat[:, 0]

    def loss(self, y_hat, y):
        """Mean binary cross-entropy."""
        eps = 1e-12
        return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    def backward(self, y):
        """Gradients of the mean loss for every parameter (call forward first)."""
        n = len(y)
        delta2 = (self.Y_hat - y[:, None]) / n                    # dL/dZ2
        dW2 = self.H.T @ delta2                                   # (n_hidden, 1)
        db2 = delta2.sum(axis=0)
        delta1 = (delta2 @ self.W2.T) * tanh_derivative(self.H)   # dL/dZ1
        dW1 = self.X.T @ delta1                                   # (n_inputs, n_hidden)
        db1 = delta1.sum(axis=0)
        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        """Gradient descent: move every parameter a little against its gradient."""
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def fit(self, X, y, epochs, lr, batch_size, log_every):
        """Mini-batch gradient descent over shuffled data."""
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = order[start:start + batch_size]
                self.forward(X[batch])
                self.step(self.backward(y[batch]), lr)
            if epoch % log_every == 0:
                y_hat = self.forward(X)
                accuracy = np.mean((y_hat > 0.5) == y)
                print(f"  epoch {epoch:5d}   loss {self.loss(y_hat, y):.4f}   accuracy {accuracy:.0%}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


def gradient_check(model, X, y, eps=1e-5):
    """Compares backpropagation with numerical derivatives; returns the largest difference."""
    model.forward(X)
    grads = model.backward(y)
    params = [model.W1, model.b1, model.W2, model.b2]
    worst = 0.0
    for param, grad in zip(params, grads):
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps
            up = model.loss(model.forward(X), y)
            param[idx] = old - eps
            down = model.loss(model.forward(X), y)
            param[idx] = old
            numeric = (up - down) / (2 * eps)
            worst = max(worst, abs(numeric - grad[idx]))
    return worst


if __name__ == "__main__":
    # 1. Is backpropagation right? Compare it with numerical derivatives.
    model = MLP(n_inputs=2, n_hidden=hidden_units)
    print(f"Gradient check: largest difference {gradient_check(model, X_xor, y_xor):.1e}")

    # 2. XOR, the problem a single perceptron cannot solve
    print("\nXOR")
    model = MLP(n_inputs=2, n_hidden=hidden_units)
    model.fit(X_xor, y_xor, epochs=epochs_xor, lr=learning_rate, batch_size=4, log_every=500)
    print("  predictions:", model.predict(X_xor), "  labels:", y_xor.astype(int))

    # 3. Two moons: the boundary has to be a curve
    print("\nTwo moons")
    X_train, y_train = make_moons(400)
    X_test, y_test = make_moons(200)
    model = MLP(n_inputs=2, n_hidden=hidden_units)
    model.fit(X_train, y_train, epochs=epochs_moons, lr=learning_rate, batch_size=batch_size, log_every=100)
    test_accuracy = np.mean(model.predict(X_test) == y_test)
    print(f"  test accuracy: {test_accuracy:.1%}")
