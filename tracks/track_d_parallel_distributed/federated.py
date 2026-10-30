"""
Federated Optimization: Federated Averaging (FedAvg) vs Centralized Training.
Examines communication rounds, local steps E, communication overhead, and client drift under non-IID partitions.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np

class FederatedDataset:
    """Creates non-IID partitioned datasets for K clients."""
    def __init__(self, num_clients: int = 5, n_samples_per_client: int = 80, dim: int = 4, seed: int = 42):
        self.K = num_clients
        rng = np.random.default_rng(seed)
        
        # Underlying true weights with client-specific bias/drift
        self.w_true = rng.standard_normal(dim)
        self.client_data = []

        for k in range(num_clients):
            # Non-IID feature distribution shift
            mean_shift = rng.uniform(-1.0, 1.0, size=dim)
            X_k = rng.standard_normal((n_samples_per_client, dim)) + mean_shift
            noise = rng.normal(0.0, 0.2, size=(n_samples_per_client, 1))
            # Binary classification labels via logistic sigmoid
            logits = X_k @ self.w_true.reshape(-1, 1) + noise
            y_k = (logits > 0).astype(np.float64)
            self.client_data.append((X_k, y_k))

        # Centralized pooled dataset
        all_X = np.vstack([c[0] for c in self.client_data])
        all_y = np.vstack([c[1] for c in self.client_data])
        self.centralized_data = (all_X, all_y)

def compute_loss_and_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray, l2_reg: float = 1e-3):
    N = len(y)
    y_flat = y.ravel()
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
    eps = 1e-12
    loss = -np.mean(y_flat * np.log(p + eps) + (1.0 - y_flat) * np.log(1.0 - p + eps)) + 0.5 * l2_reg * np.sum(w ** 2)
    grad = (X.T @ (p - y_flat)) / N + l2_reg * w
    return loss, grad

def run_centralized_training(fed_data: FederatedDataset, total_steps: int = 100, lr: float = 0.1) -> dict:
    X_all, y_all = fed_data.centralized_data
    w = np.zeros(X_all.shape[1])
    history = []
    start_t = time.perf_counter()

    for _ in range(total_steps):
        loss, grad = compute_loss_and_grad(w, X_all, y_all)
        history.append(float(loss))
        w = w - lr * grad

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Centralized SGD',
        'final_loss': float(history[-1]),
        'runtime_ms': runtime * 1000.0,
        'communication_rounds': 0,
        'total_bytes_transferred': 0,
        'history': history
    }

def run_fedavg(fed_data: FederatedDataset, rounds: int = 20, local_epochs: int = 5, lr: float = 0.1) -> dict:
    dim = fed_data.centralized_data[0].shape[1]
    w_global = np.zeros(dim)
    history = []
    total_bytes = 0
    start_t = time.perf_counter()

    param_bytes = dim * 8  # 8 bytes per float64

    for r in range(rounds):
        # Evaluate global loss on pooled data
        loss_val, _ = compute_loss_and_grad(w_global, fed_data.centralized_data[0], fed_data.centralized_data[1])
        history.append(float(loss_val))

        client_weights = []
        # Client updates
        for k in range(fed_data.K):
            X_k, y_k = fed_data.client_data[k]
            w_local = w_global.copy()
            # Local SGD steps
            for _ in range(local_epochs):
                _, grad_k = compute_loss_and_grad(w_local, X_k, y_k)
                w_local = w_local - lr * grad_k
            client_weights.append(w_local)
            # Uplink communication: send w_local to server
            total_bytes += param_bytes

        # Downlink communication: broadcast w_global to clients
        total_bytes += fed_data.K * param_bytes

        # Server aggregation: Federated Averaging
        w_global = np.mean(client_weights, axis=0)

    final_loss, _ = compute_loss_and_grad(w_global, fed_data.centralized_data[0], fed_data.centralized_data[1])
    history.append(float(final_loss))
    runtime = time.perf_counter() - start_t

    return {
        'method': f'FedAvg (E={local_epochs})',
        'local_epochs': local_epochs,
        'rounds': rounds,
        'final_loss': float(final_loss),
        'runtime_ms': runtime * 1000.0,
        'communication_rounds': rounds,
        'total_bytes_transferred': total_bytes,
        'history': history
    }
