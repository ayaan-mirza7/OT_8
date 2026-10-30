"""
Variational Quantum Eigensolver (VQE).
Finds the ground state energy of a molecular/spin Hamiltonian:
H = sum_i h_i * Z_i + sum_{(i,j)} J_{ij} * Z_i Z_j + sum_i g_i * X_i
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np
from tracks.track_c_quantum.simulator import QuantumState, ry_gate, rz_gate, X_GATE
from tracks.track_c_quantum.gradients import parameter_shift_gradient

class VQE:
    def __init__(self, num_qubits: int, layers: int = 2):
        self.n = num_qubits
        self.layers = layers
        # Each layer has: Ry(theta) and Rz(phi) on all qubits = 2 * n params per layer
        self.num_params = 2 * self.n * self.layers

    def construct_ansatz(self, params: np.ndarray) -> QuantumState:
        qs = QuantumState(self.n)
        p_idx = 0
        for l in range(self.layers):
            # Single-qubit rotations
            for q in range(self.n):
                qs.apply_single_gate(ry_gate(params[p_idx]), q)
                p_idx += 1
                qs.apply_single_gate(rz_gate(params[p_idx]), q)
                p_idx += 1
            # Entangling linear CNOT chain
            for q in range(self.n - 1):
                qs.apply_cnot(q, q + 1)
        return qs

    def compute_energy(self, params: np.ndarray, h_z: float = 1.0, J_zz: float = -1.0, g_x: float = 0.5) -> float:
        """Evaluates expectation value of transverse Ising Hamiltonian: H = sum h*Z_i + sum J*Z_i Z_{i+1} + sum g*X_i."""
        qs = self.construct_ansatz(params)
        energy = 0.0

        # Z terms
        for q in range(self.n):
            energy += h_z * qs.expectation_z(q)

        # ZZ interaction terms
        for q in range(self.n - 1):
            energy += J_zz * qs.expectation_zz(q, q + 1)

        # X terms (via <+ | Z | +> basis rotation or direct state dot)
        # For simplicity, X expectation:
        for q in range(self.n):
            # <psi | X_q | psi>
            qs_copy = self.construct_ansatz(params)
            qs_copy.apply_single_gate(X_GATE, q)
            # Re(<psi | X_q | psi>)
            energy += g_x * np.real(np.vdot(qs.state, qs_copy.state))

        return float(energy)

    def optimize_ground_state(self, max_iters: int = 50, lr: float = 0.05, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=self.num_params)
        history = []

        def cost_fn(p):
            return self.compute_energy(p)

        m = np.zeros_like(params)
        v = np.zeros_like(params)

        for t in range(1, max_iters + 1):
            current_e = cost_fn(params)
            history.append(current_e)

            grad = parameter_shift_gradient(cost_fn, params)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad ** 2)
            m_hat = m / (1.0 - (0.9 ** t))
            v_hat = v / (1.0 - (0.999 ** t))
            params = params - (lr / (np.sqrt(v_hat) + 1e-8)) * m_hat

        final_e = cost_fn(params)
        history.append(final_e)

        return {
            'final_energy': final_e,
            'optimal_params': params.tolist(),
            'history': history
        }
