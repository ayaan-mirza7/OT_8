"""
Quantum Machine Learning (QML) vs Classical Classifier Benchmark.
Trains a Variational Quantum Classifier (VQC) with angle embedding vs an equivalent Classical MLP on the same dataset.
Reports accuracy, training time, parameter count, and number of circuit evaluations.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np
from tracks.track_c_quantum.simulator import QuantumState, ry_gate, rz_gate
from tracks.track_c_quantum.gradients import parameter_shift_gradient
from tracks.track_a_continuous.neural_net import SmallMLP
from tracks.track_a_continuous.optimizers import Adam

class VariationalQuantumClassifier:
    """
    VQC with:
    1. Angle Embedding: maps classical 2D features (x1, x2) to Ry(x1) |0>, Ry(x2) |0>
    2. Strongly Entangling Ansatz: parameterized Ry, Rz, and CNOT entangling gates
    3. Measurement of <Z_0> mapped to class probability via sigmoid: p = 1 / (1 + exp(-<Z_0>))
    """
    def __init__(self, num_qubits: int = 2, layers: int = 2):
        self.n = num_qubits
        self.layers = layers
        self.num_params = 2 * num_qubits * layers
        self.circuit_evaluations = 0

    def predict_sample(self, x: np.ndarray, params: np.ndarray) -> float:
        """Runs quantum circuit for a single sample x = [x0, x1]."""
        self.circuit_evaluations += 1
        qs = QuantumState(self.n)
        # Feature encoding (angle embedding)
        for q in range(self.n):
            qs.apply_single_gate(ry_gate(x[q]), q)

        # Variational ansatz
        p_idx = 0
        for _ in range(self.layers):
            for q in range(self.n):
                qs.apply_single_gate(ry_gate(params[p_idx]), q)
                p_idx += 1
                qs.apply_single_gate(rz_gate(params[p_idx]), q)
                p_idx += 1
            for q in range(self.n - 1):
                qs.apply_cnot(q, q + 1)

        # Expectation of Z_0
        exp_z = qs.expectation_z(0)
        # Sigmoid probability
        prob = 1.0 / (1.0 + np.exp(-3.0 * exp_z))
        return prob

    def loss(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        eps = 1e-12
        total_loss = 0.0
        for i in range(len(X)):
            p = self.predict_sample(X[i], params)
            total_loss -= (y[i] * np.log(p + eps) + (1.0 - y[i]) * np.log(1.0 - p + eps))
        return total_loss / len(X)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 25, lr: float = 0.1, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=self.num_params)
        start_t = time.perf_counter()
        self.circuit_evaluations = 0
        loss_history = []

        m = np.zeros_like(params)
        v = np.zeros_like(params)

        for epoch in range(1, epochs + 1):
            current_loss = self.loss(X, y, params)
            loss_history.append(current_loss)

            # Gradient w.r.t params via parameter-shift
            grad = parameter_shift_gradient(lambda p: self.loss(X, y, p), params)

            # Adam update
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad ** 2)
            m_hat = m / (1.0 - (0.9 ** epoch))
            v_hat = v / (1.0 - (0.999 ** epoch))
            params -= (lr / (np.sqrt(v_hat) + 1e-8)) * m_hat

        runtime = time.perf_counter() - start_t
        # Compute final training accuracy
        preds = np.array([self.predict_sample(X[i], params) >= 0.5 for i in range(len(X))])
        acc = float(np.mean(preds == y.ravel()))

        return {
            'model': 'Variational Quantum Classifier (VQC)',
            'accuracy': acc,
            'final_loss': float(loss_history[-1]),
            'runtime_ms': runtime * 1000.0,
            'parameter_count': self.num_params,
            'circuit_evaluations': self.circuit_evaluations
        }

def run_classical_classifier_baseline(X: np.ndarray, y: np.ndarray, epochs: int = 80, seed: int = 42) -> dict:
    """Equivalent classical MLP trained on identical data."""
    start_t = time.perf_counter()
    y_col = y.reshape(-1, 1)
    mlp = SmallMLP(in_features=2, hidden_dim=4, out_features=1, seed=seed)
    opt = Adam(lr=0.08)

    for epoch in range(epochs):
        _ = mlp.forward(X)
        mlp.backward(y_col)
        params = mlp.get_flat_params()
        grads = mlp.get_flat_grads()
        mlp.set_flat_params(opt.step(params, grads))

    runtime = time.perf_counter() - start_t
    preds = (mlp.forward(X) >= 0.5).astype(int)
    acc = float(np.mean(preds == y_col))
    param_count = mlp.get_flat_params().size

    return {
        'model': 'Classical 2-Layer MLP',
        'accuracy': acc,
        'final_loss': float(mlp.loss(mlp.forward(X), y_col)),
        'runtime_ms': runtime * 1000.0,
        'parameter_count': param_count,
        'circuit_evaluations': 0
    }
