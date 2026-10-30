"""
Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut.
Solves the exact Graph Max-Cut instance formulated in Track B.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np
from tracks.track_c_quantum.simulator import QuantumState, H_GATE, rx_gate
from tracks.track_c_quantum.gradients import parameter_shift_gradient, spsa_gradient_estimate

class QAOA:
    def __init__(self, adj_matrix: np.ndarray, p_layers: int = 2):
        self.adj = np.asarray(adj_matrix, dtype=np.float64)
        self.n = self.adj.shape[0]
        self.p = p_layers
        self.edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj[i, j] != 0:
                    self.edges.append((i, j, self.adj[i, j]))

    def construct_state(self, gamma: np.ndarray, beta: np.ndarray, noise_rate: float = 0.0) -> QuantumState:
        qs = QuantumState(self.n)
        # 1. Uniform superposition: Apply H to all qubits
        for q in range(self.n):
            qs.apply_single_gate(H_GATE, q)

        # 2. Alternating p layers of Problem & Mixer unitaries
        for l in range(self.p):
            g = gamma[l]
            b = beta[l]
            
            # Problem Unitary: exp(-i * gamma * H_C)
            # H_C = sum_{(u,v)} w_{uv} * (I - Z_u Z_v) / 2
            for u, v, w in self.edges:
                qs.apply_rzz(u, v, -g * w)
                if noise_rate > 0:
                    qs.apply_depolarizing_channel(noise_rate)

            # Mixer Unitary: exp(-i * beta * H_M) = prod Rx(2 * beta)
            for q in range(self.n):
                qs.apply_single_gate(rx_gate(2.0 * b), q)
                if noise_rate > 0:
                    qs.apply_depolarizing_channel(noise_rate)

        return qs

    def compute_expected_cut(self, params: np.ndarray, noise_rate: float = 0.0) -> float:
        """
        Computes expected Cut value <psi | H_C | psi>.
        params is array of length 2*p: [gamma_1..gamma_p, beta_1..beta_p]
        """
        gamma = params[:self.p]
        beta = params[self.p:]
        qs = self.construct_state(gamma, beta, noise_rate=noise_rate)

        # Expected cut = sum_{(u,v)} w_{uv} * 0.5 * (1 - <Z_u Z_v>)
        expected_cut = 0.0
        for u, v, w in self.edges:
            zz = qs.expectation_zz(u, v)
            expected_cut += 0.5 * w * (1.0 - zz)
        return float(expected_cut)

    def optimize_params(
        self,
        method: str = "parameter-shift",
        max_iters: int = 40,
        lr: float = 0.08,
        seed: int = 42
    ) -> dict:
        """Optimizes (gamma, beta) to maximize Expected Cut."""
        rng = np.random.default_rng(seed)
        # Initial parameters
        params = rng.uniform(0.1, np.pi, size=2 * self.p)
        history = []

        # Objective to MINIMIZE is -Expected_Cut
        def cost_fn(p):
            return -self.compute_expected_cut(p)

        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, max_iters + 1):
            current_cut = self.compute_expected_cut(params)
            history.append(current_cut)

            if method == "parameter-shift":
                grad = parameter_shift_gradient(cost_fn, params)
            elif method == "spsa":
                grad = spsa_gradient_estimate(cost_fn, params, c_k=0.1, rng=rng)
            else:
                raise ValueError(f"Unknown method {method}")

            # Adam update
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - (beta1 ** t))
            v_hat = v / (1.0 - (beta2 ** t))
            params = params - (lr / (np.sqrt(v_hat) + eps)) * m_hat

        final_cut = self.compute_expected_cut(params)
        history.append(final_cut)

        return {
            'optimal_params': params.tolist(),
            'expected_cut': final_cut,
            'history': history
        }

    def sample_best_solution(self, optimal_params: list[float], shots: int = 1000, seed: int = 42) -> dict:
        """Measures statevector and extracts top cut bitstrings."""
        params = np.asarray(optimal_params)
        gamma = params[:self.p]
        beta = params[self.p:]
        qs = self.construct_state(gamma, beta)
        counts = qs.sample_measurements(shots=shots, seed=seed)

        # Evaluate cut for sampled bitstrings
        best_bitstring = None
        best_cut = -1.0
        weighted_cut_sum = 0.0

        for bitstr, count in counts.items():
            bits = [int(b) for b in bitstr]
            cut = 0.0
            for u, v, w in self.edges:
                if bits[u] != bits[v]:
                    cut += w
            weighted_cut_sum += cut * count
            if cut > best_cut:
                best_cut = cut
                best_bitstring = bitstr

        return {
            'sampled_best_bitstring': best_bitstring,
            'sampled_max_cut': best_cut,
            'sampled_average_cut': weighted_cut_sum / shots,
            'top_counts': dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5])
        }
