"""
Modular Neural Network implementation demonstrating Backpropagation as distinct from the Optimizer.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

class SmallMLP:
    """
    2-Layer Multi-Layer Perceptron: Input -> Linear -> ReLU -> Linear -> Sigmoid -> Binary Cross-Entropy.
    Clearly demonstrates:
    1. Forward propagation (activations, cache)
    2. Backpropagation (chain rule, gradient calculation) - STRICTLY DISTINCT from optimization
    3. Parameter update delegated to optimizer instance
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 8, out_features: int = 1, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Xavier / He initialization
        self.W1 = rng.standard_normal((in_features, hidden_dim)) * np.sqrt(2.0 / in_features)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, out_features)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(out_features)
        
        self.cache = {}
        self.grads = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass saving activations for backpropagation."""
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)  # ReLU
        Z2 = A1 @ self.W2 + self.b2
        A2 = 1.0 / (1.0 + np.exp(-np.clip(Z2, -15.0, 15.0)))  # Sigmoid
        
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray, l2_reg: float = 1e-4) -> float:
        """Binary cross-entropy loss + L2 regularization."""
        eps = 1e-12
        bce = -np.mean(y_true * np.log(y_pred + eps) + (1.0 - y_true) * np.log(1.0 - y_pred + eps))
        reg = 0.5 * l2_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(bce + reg)

    def backward(self, y_true: np.ndarray, l2_reg: float = 1e-4) -> dict[str, np.ndarray]:
        """
        Pure Backpropagation: computes analytical gradients with respect to parameters
        using the multivariable chain rule.
        Does NOT update parameters.
        """
        N = len(y_true)
        X = self.cache['X']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        Z1 = self.cache['Z1']

        # dLoss / dZ2
        dZ2 = (A2 - y_true) / N
        dW2 = A1.T @ dZ2 + l2_reg * self.W2
        db2 = np.sum(dZ2, axis=0)

        # dLoss / dA1 -> dLoss / dZ1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0.0)  # ReLU derivative
        dW1 = X.T @ dZ1 + l2_reg * self.W1
        db1 = np.sum(dZ1, axis=0)

        self.grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }
        return self.grads

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_flat_params(self, flat: np.ndarray):
        s_w1 = self.W1.size
        s_b1 = self.b1.size
        s_w2 = self.W2.size
        s_b2 = self.b2.size
        
        self.W1 = flat[:s_w1].reshape(self.W1.shape)
        self.b1 = flat[s_w1:s_w1+s_b1].reshape(self.b1.shape)
        self.W2 = flat[s_w1+s_b1:s_w1+s_b1+s_w2].reshape(self.W2.shape)
        self.b2 = flat[s_w1+s_b1+s_w2:].reshape(self.b2.shape)

    def get_flat_grads(self) -> np.ndarray:
        return np.concatenate([
            self.grads['W1'].ravel(),
            self.grads['b1'].ravel(),
            self.grads['W2'].ravel(),
            self.grads['b2'].ravel()
        ])
