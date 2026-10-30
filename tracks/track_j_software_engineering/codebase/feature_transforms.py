"""
Feature Transformation & Engineering Pipeline.
Module of realistic production codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import math
from typing import Sequence

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: Sequence[Sequence[float]]):
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Cannot fit scaler on empty data")
        n_features = len(X[0])
        means = [0.0] * n_features
        for row in X:
            for j in range(n_features):
                means[j] += row[j]
        means = [m / n_samples for m in means]

        variances = [0.0] * n_features
        for row in X:
            for j in range(n_features):
                variances[j] += (row[j] - means[j]) ** 2
        stds = [math.sqrt(v / max(n_samples - 1, 1)) if v > 1e-12 else 1.0 for v in variances]

        self.mean_ = means
        self.std_ = stds
        return self

    def transform(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before transform")
        transformed = []
        for row in X:
            scaled_row = [(val - m) / s for val, m, s in zip(row, self.mean_, self.std_)]
            transformed.append(scaled_row)
        return transformed

    def fit_transform(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        return self.fit(X).transform(X)

class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.min_val = feature_range[0]
        self.max_val = feature_range[1]
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X: Sequence[Sequence[float]]):
        if not X:
            raise ValueError("Empty data")
        n_features = len(X[0])
        mins = list(X[0])
        maxs = list(X[0])
        for row in X[1:]:
            for j in range(n_features):
                if row[j] < mins[j]:
                    mins[j] = row[j]
                if row[j] > maxs[j]:
                    maxs[j] = row[j]
        self.data_min_ = mins
        self.data_max_ = maxs
        return self

    def transform(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        diff = self.max_val - self.min_val
        transformed = []
        for row in X:
            scaled = []
            for j, val in enumerate(row):
                span = self.data_max_[j] - self.data_min_[j]
                if span < 1e-12:
                    scaled.append(self.min_val)
                else:
                    norm = (val - self.data_min_[j]) / span
                    scaled.append(norm * diff + self.min_val)
            transformed.append(scaled)
        return transformed

class OneHotEncoder:
    def __init__(self):
        self.categories_ = {}

    def fit(self, categories_list: Sequence[str]):
        unique_cats = sorted(list(set(categories_list)))
        self.categories_ = {cat: idx for idx, cat in enumerate(unique_cats)}
        return self

    def transform(self, item: str) -> list[int]:
        n = len(self.categories_)
        vector = [0] * n
        if item in self.categories_:
            vector[self.categories_[item]] = 1
        return vector

class RollingWindowStats:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = []

    def update(self, val: float) -> dict[str, float]:
        self.buffer.append(val)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        mean_val = sum(self.buffer) / len(self.buffer)
        var_val = sum((x - mean_val) ** 2 for x in self.buffer) / max(len(self.buffer), 1)
        return {
            'rolling_mean': mean_val,
            'rolling_std': math.sqrt(var_val),
            'min': min(self.buffer),
            'max': max(self.buffer)
        }
