"""
Barren Plateau Investigation.
Demonstrates the exponential decay of gradient variance with increasing number of qubits N:
Var_{theta} [ d <H> / d theta_k ] ~ O(2^{-N})
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np
from tracks.track_c_quantum.simulator import QuantumState, ry_gate, rz_gate
from tracks.track_c_quantum.gradients import parameter_shift_gradient

def sample_gradient_variance(num_qubits: int, depth: int = 3, num_samples: int = 40, seed: int = 42) -> dict:
    """
    Computes empirical variance of gradients for a random hardware-efficient ansatz.
    Cost Hamiltonian H = Z_0 Z_1.
    """
    rng = np.random.default_rng(seed)
    num_params = 2 * num_qubits * depth
    grads_first_param = []

    def run_circuit(params: np.ndarray) -> float:
        qs = QuantumState(num_qubits)
        p_idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qs.apply_single_gate(ry_gate(params[p_idx]), q)
                p_idx += 1
                qs.apply_single_gate(rz_gate(params[p_idx]), q)
                p_idx += 1
            for q in range(num_qubits - 1):
                qs.apply_cnot(q, q + 1)
        # Expectation of Z_0 Z_1
        return qs.expectation_zz(0, 1)

    for _ in range(num_samples):
        random_params = rng.uniform(-np.pi, np.pi, size=num_params)
        # Evaluate gradient of first parameter using parameter-shift
        p_plus = random_params.copy()
        p_minus = random_params.copy()
        p_plus[0] += np.pi / 2.0
        p_minus[0] -= np.pi / 2.0
        g = 0.5 * (run_circuit(p_plus) - run_circuit(p_minus))
        grads_first_param.append(g)

    grads_arr = np.array(grads_first_param)
    variance = float(np.var(grads_arr))
    mean_val = float(np.mean(grads_arr))

    return {
        'num_qubits': num_qubits,
        'depth': depth,
        'samples': num_samples,
        'mean_gradient': mean_val,
        'gradient_variance': variance
    }
