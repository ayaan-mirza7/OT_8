"""
Modeling Engine and Evaluation Primitives.
Module of realistic production codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import math
from typing import Sequence

class LinearRegressionModel:
    def __init__(self, lr: float = 0.01, max_iters: int = 100):
        self.lr = lr
        self.max_iters = max_iters
        self.weights = []
        self.bias = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.max_iters):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                y_pred = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                err = y_pred - y[i]
                for j in range(n_features):
                    dw[j] += err * X[i][j]
                db += err
            for j in range(n_features):
                self.weights[j] -= (self.lr * dw[j]) / n_samples
            self.bias -= (self.lr * db) / n_samples
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        return [sum(w * x for w, x in zip(self.weights, row)) + self.bias for row in X]

class LogisticBinaryClassifier:
    def __init__(self, lr: float = 0.05, max_iters: int = 80):
        self.lr = lr
        self.max_iters = max_iters
        self.weights = []
        self.bias = 0.0

    def _sigmoid(self, z: float) -> float:
        z_clamped = max(-15.0, min(15.0, z))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.max_iters):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                z = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                p = self._sigmoid(z)
                err = p - y[i]
                for j in range(n_features):
                    dw[j] += err * X[i][j]
                db += err
            for j in range(n_features):
                self.weights[j] -= (self.lr * dw[j]) / n_samples
            self.bias -= (self.lr * db) / n_samples
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        return [self._sigmoid(sum(w * x for w, x in zip(self.weights, row)) + self.bias) for row in X]

    def predict(self, X: Sequence[Sequence[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

class ModelMetrics:
    @staticmethod
    def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)

    @staticmethod
    def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return matches / max(len(y_true), 1)

    @staticmethod
    def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall < 1e-12:
            return 0.0
        return 2.0 * (precision * recall) / (precision + recall)
