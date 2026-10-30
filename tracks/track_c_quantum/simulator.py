"""
Quantum Statevector & Density Matrix Simulator.
Features:
1. Exact unitary gate evolution on 2^N statevectors
2. Expectation values of Pauli strings (Z_i Z_j, X_i, etc.)
3. Finite measurement shots sampling with multinomial statistics
4. Density matrix & Kraus operator noise simulation (depolarizing & amplitude damping)
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

# Elementary single-qubit gates
I2 = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)

def rx_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
        [-1j * np.sin(theta / 2.0), np.cos(theta / 2.0)]
    ], dtype=complex)

def ry_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0), np.cos(theta / 2.0)]
    ], dtype=complex)

def rz_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2.0), 0.0],
        [0.0, np.exp(1j * theta / 2.0)]
    ], dtype=complex)

class QuantumState:
    """Pure statevector simulator for n qubits."""
    def __init__(self, num_qubits: int):
        self.n = num_qubits
        self.dim = 1 << num_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0  # |00...0>

    def reset(self):
        self.state.fill(0.0)
        self.state[0] = 1.0

    def apply_single_gate(self, gate: np.ndarray, target: int):
        """Applies a 2x2 unitary gate to target qubit."""
        # Reshape to (2, 2, ..., 2)
        state_tensor = self.state.reshape([2] * self.n)
        # Tensordot along target axis
        state_tensor = np.tensordot(gate, state_tensor, axes=([1], [target]))
        # Move target axis back to original position
        state_tensor = np.moveaxis(state_tensor, 0, target)
        self.state = state_tensor.flatten()

    def apply_cnot(self, control: int, target: int):
        """Applies CNOT (CX) gate."""
        state_tensor = self.state.reshape([2] * self.n)
        # Index slicing where control == 1
        slices_0 = [slice(None)] * self.n
        slices_1 = [slice(None)] * self.n
        slices_0[control] = 1
        slices_0[target] = 0
        slices_1[control] = 1
        slices_1[target] = 1

        val0 = state_tensor[tuple(slices_0)].copy()
        val1 = state_tensor[tuple(slices_1)].copy()
        state_tensor[tuple(slices_0)] = val1
        state_tensor[tuple(slices_1)] = val0
        self.state = state_tensor.flatten()

    def apply_rzz(self, q1: int, q2: int, gamma: float):
        """Applies exp(-i * gamma * Z1 * Z2 / 2) interaction gate."""
        phase = np.exp(-1j * gamma / 2.0)
        conj_phase = np.exp(1j * gamma / 2.0)
        
        # When q1 == q2, eigenvalue of Z1*Z2 is +1 (phase)
        # When q1 != q2, eigenvalue is -1 (conj_phase)
        indices = np.arange(self.dim)
        bit1 = (indices >> (self.n - 1 - q1)) & 1
        bit2 = (indices >> (self.n - 1 - q2)) & 1
        same_parity = (bit1 == bit2)

        multipliers = np.where(same_parity, phase, conj_phase)
        self.state *= multipliers

    def expectation_z(self, qubit: int) -> float:
        """Computes <Z_i>."""
        probs = np.abs(self.state) ** 2
        indices = np.arange(self.dim)
        bit = (indices >> (self.n - 1 - qubit)) & 1
        signs = 1.0 - 2.0 * bit  # bit 0 -> +1, bit 1 -> -1
        return float(np.sum(probs * signs))

    def expectation_zz(self, q1: int, q2: int) -> float:
        """Computes <Z_q1 Z_q2>."""
        probs = np.abs(self.state) ** 2
        indices = np.arange(self.dim)
        bit1 = (indices >> (self.n - 1 - q1)) & 1
        bit2 = (indices >> (self.n - 1 - q2)) & 1
        signs = np.where(bit1 == bit2, 1.0, -1.0)
        return float(np.sum(probs * signs))

    def sample_measurements(self, shots: int = 1024, seed: int = 42) -> dict[str, int]:
        """Samples discrete projective measurements simulating quantum hardware shot noise."""
        probs = np.abs(self.state) ** 2
        # Normalize to avoid floating point sum drift
        probs = probs / np.sum(probs)
        rng = np.random.default_rng(seed)
        counts_raw = rng.multinomial(shots, probs)
        
        results = {}
        fmt = f"0{self.n}b"
        for idx, count in enumerate(counts_raw):
            if count > 0:
                bitstring = format(idx, fmt)
                results[bitstring] = int(count)
        return results

    def apply_depolarizing_channel(self, error_rate: float, rng: np.random.Generator = None):
        """Simulates single-qubit depolarizing error channel."""
        if error_rate <= 0:
            return
        if rng is None:
            rng = np.random.default_rng()

        for q in range(self.n):
            if rng.random() < error_rate:
                pauli_choice = rng.integers(0, 3)
                if pauli_choice == 0:
                    self.apply_single_gate(X_GATE, q)
                elif pauli_choice == 1:
                    self.apply_single_gate(Y_GATE, q)
                else:
                    self.apply_single_gate(Z_GATE, q)
