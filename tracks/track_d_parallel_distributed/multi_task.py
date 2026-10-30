"""
Multi-Task Optimization: Equal Weighting vs MGDA vs GradNorm.
Solves multi-objective optimization with competing gradients.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np

class TwoTaskModel:
    """
    Shared representation layer + Two task heads:
    Input X (dim 6) -> Shared representation H (dim 4)
    Head 1: Binary Classification Task 1 (e.g. category A)
    Head 2: Regression Task 2 (e.g. continuous target)
    """
    def __init__(self, in_dim: int = 6, shared_dim: int = 4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W_shared = rng.standard_normal((in_dim, shared_dim)) * 0.5
        self.w_task1 = rng.standard_normal((shared_dim, 1)) * 0.5
        self.w_task2 = rng.standard_normal((shared_dim, 1)) * 0.5

    def forward(self, X: np.ndarray):
        H = np.tanh(X @ self.W_shared)
        y1_pred = 1.0 / (1.0 + np.exp(-np.clip(H @ self.w_task1, -10.0, 10.0)))
        y2_pred = H @ self.w_task2
        return H, y1_pred, y2_pred

    def compute_task_losses_and_grads(self, X: np.ndarray, y1: np.ndarray, y2: np.ndarray):
        """Computes losses and shared representation gradients for both tasks."""
        H, y1_pred, y2_pred = self.forward(X)
        N = len(X)
        eps = 1e-12

        # Task 1 Loss: Binary Cross Entropy
        l1 = -np.mean(y1 * np.log(y1_pred + eps) + (1.0 - y1) * np.log(1.0 - y1_pred + eps))
        # Task 2 Loss: Mean Squared Error
        l2 = float(np.mean((y2_pred - y2) ** 2))

        # Gradient of L1 w.r.t shared weights
        dy1 = (y1_pred - y1) / N
        dH_1 = dy1 @ self.w_task1.T * (1.0 - H ** 2)
        g1_shared = (X.T @ dH_1).ravel()

        # Gradient of L2 w.r.t shared weights
        dy2 = (2.0 * (y2_pred - y2)) / N
        dH_2 = dy2 @ self.w_task2.T * (1.0 - H ** 2)
        g2_shared = (X.T @ dH_2).ravel()

        return l1, l2, g1_shared, g2_shared

def solve_mgda_2task(g1: np.ndarray, g2: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Solves min-norm point in convex hull: min_{alpha in [0, 1]} ||alpha * g1 + (1 - alpha) * g2||^2.
    Analytical solution for 2 tasks:
    alpha = ((g2 - g1)^T g2) / ||g1 - g2||^2, clipped to [0, 1].
    """
    diff = g1 - g2
    diff_norm_sq = float(np.dot(diff, diff))
    if diff_norm_sq < 1e-12:
        alpha = 0.5
    else:
        alpha = float(np.dot(g2, g2 - g1) / diff_norm_sq)
        alpha = max(0.0, min(1.0, alpha))
    
    combined_grad = alpha * g1 + (1.0 - alpha) * g2
    return alpha, 1.0 - alpha, combined_grad

def run_multi_task_experiment(n_steps: int = 100, lr: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    N = 200
    X = rng.standard_normal((N, 6))
    y1 = (X[:, :1] + X[:, 1:2] > 0).astype(np.float64)
    # Competing target with conflicting negative correlations
    y2 = -2.0 * X[:, :1] + 1.5 * X[:, 2:3]

    methods = ['Equal_Weighting', 'MGDA', 'GradNorm']
    results = {}

    for method in methods:
        model = TwoTaskModel(in_dim=6, shared_dim=4, seed=seed)
        l1_history = []
        l2_history = []
        w_task = np.array([0.5, 0.5])
        l1_init, l2_init = None, None

        for step in range(n_steps):
            l1, l2, g1, g2 = model.compute_task_losses_and_grads(X, y1, y2)
            l1_history.append(l1)
            l2_history.append(l2)

            if l1_init is None:
                l1_init, l2_init = l1, l2

            if method == 'Equal_Weighting':
                combined_grad = 0.5 * g1 + 0.5 * g2
            elif method == 'MGDA':
                _, _, combined_grad = solve_mgda_2task(g1, g2)
            elif method == 'GradNorm':
                # Dynamically balance gradient norms
                norm1 = np.linalg.norm(g1) + 1e-8
                norm2 = np.linalg.norm(g2) + 1e-8
                # Relative loss ratios
                r1 = (l1 / l1_init)
                r2 = (l2 / l2_init)
                r_mean = (r1 + r2) / 2.0
                target_ratio1 = (r1 / r_mean) ** 1.5
                target_ratio2 = (r2 / r_mean) ** 1.5
                w1 = target_ratio1 / norm1
                w2 = target_ratio2 / norm2
                # Renormalize weights
                w_sum = w1 + w2
                w1, w2 = w1 / w_sum, w2 / w_sum
                combined_grad = w1 * g1 + w2 * g2

            # Parameter update
            model.W_shared -= lr * combined_grad.reshape(model.W_shared.shape)

        results[method] = {
            'l1_history': l1_history,
            'l2_history': l2_history,
            'final_l1': l1_history[-1],
            'final_l2': l2_history[-1],
            'combined_final_loss': l1_history[-1] + l2_history[-1]
        }
    return results
