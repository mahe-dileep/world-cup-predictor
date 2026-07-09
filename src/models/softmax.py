"""A mini softmax (multinomial logistic regression) engine, from scratch in numpy.

Transparent core model for the three-way result (Home / Draw / Away): standardise
features, add a bias, and fit softmax + cross-entropy by full-batch gradient
descent with L2. Deterministic. No sklearn — the whole model is ~40 lines so you
can read exactly what it does.
"""
from __future__ import annotations
import numpy as np


class SoftmaxRegression:
    def __init__(self, l2: float = 1e-3, lr: float = 0.1, epochs: int = 3000):
        self.l2, self.lr, self.epochs = l2, lr, epochs

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)      # stability
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def _design(self, X):
        Xs = (np.asarray(X, float) - self.mean_) / self.std_
        return np.hstack([np.ones((len(Xs), 1)), Xs])   # bias column

    def fit(self, X, y, classes=None):
        X = np.asarray(X, float)
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-8
        Xs = self._design(X)
        self.classes_ = np.array(sorted(set(y)) if classes is None else classes)
        idx = {c: i for i, c in enumerate(self.classes_)}
        n, d, K = len(Xs), Xs.shape[1], len(self.classes_)
        Y = np.zeros((n, K))
        for i, yi in enumerate(y):
            Y[i, idx[yi]] = 1.0
        self.W = np.zeros((d, K))
        for _ in range(self.epochs):
            grad = Xs.T @ (self._softmax(Xs @ self.W) - Y) / n
            grad[1:] += self.l2 * self.W[1:]      # regularise weights, not bias
            self.W -= self.lr * grad
        return self

    def predict_proba(self, X):
        return self._softmax(self._design(X) @ self.W)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
