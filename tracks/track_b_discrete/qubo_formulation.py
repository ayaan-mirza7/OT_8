"""
QUBO and Ising Model Formulations.
Covers:
1. Max-Cut Graph Partitioning (Ising Hamiltonian & QUBO matrix)
2. Binary-to-Spin Transformation (x_i in {0,1} <-> s_i in {-1, +1})
3. Penalty Terms for Constraints (quadratic penalty expansion)
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

class MaxCutQUBO:
    """
    Max-Cut graph formulation:
    Maximize Cut(x) = sum_{(i,j) in E} w_{ij} * (x_i + x_j - 2 * x_i * x_j)
    Equivalently minimize E(x) = x^T Q x
    """
    def __init__(self, adj_matrix: np.ndarray):
        self.adj = np.asarray(adj_matrix, dtype=np.float64)
        self.n = self.adj.shape[0]
        self.Q = self._build_qubo_matrix()

    def _build_qubo_matrix(self) -> np.ndarray:
        Q = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                w = self.adj[i, j]
                if w != 0:
                    # Cut term: w * (x_i + x_j - 2 * x_i * x_j)
                    # Minimizing -Cut: Q[i,i] -= w, Q[j,j] -= w, Q[i,j] += 2*w
                    Q[i, i] -= w
                    Q[j, j] -= w
                    Q[i, j] += 2.0 * w
        return Q

    def evaluate_qubo(self, x: np.ndarray) -> float:
        """Computes x^T Q x."""
        x = np.asarray(x, dtype=np.float64)
        return float(x.T @ self.Q @ x)

    def evaluate_cut(self, x: np.ndarray) -> float:
        """Computes cut weight directly."""
        cut = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if x[i] != x[j]:
                    cut += self.adj[i, j]
        return float(cut)

    def binary_to_spin(self, x: np.ndarray) -> np.ndarray:
        """Transforms binary x in {0, 1} to spin s in {-1, +1}: s_i = 1 - 2*x_i."""
        return 1 - 2 * np.asarray(x, dtype=int)

    def spin_to_binary(self, s: np.ndarray) -> np.ndarray:
        """Transforms spin s in {-1, +1} to binary x in {0, 1}: x_i = (1 - s_i) / 2."""
        return ((1 - np.asarray(s, dtype=int)) // 2).astype(int)

    def ising_energy(self, s: np.ndarray) -> float:
        """
        Ising Hamiltonian: H = 0.5 * sum_{(i,j)} w_{ij} * s_i * s_j
        Cut = 0.5 * sum w_{ij} * (1 - s_i * s_j)
        """
        s = np.asarray(s, dtype=np.float64)
        total_weight = np.sum(np.triu(self.adj, 1))
        spin_interaction = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                spin_interaction += self.adj[i, j] * s[i] * s[j]
        cut = 0.5 * (total_weight - spin_interaction)
        return float(cut)


class ConstrainedFeatureSelectionQUBO:
    """
    Feature Selection formulated as QUBO with Cardinality Penalty:
    Minimizes: - x^T C x + alpha * x^T S x + lambda * (sum(x_i) - K)^2
    where C is feature-target relevance, S is feature-feature redundancy,
    and lambda * (sum x_i - K)^2 is the quadratic penalty term.
    """
    def __init__(self, relevance: np.ndarray, redundancy: np.ndarray, k_target: int, penalty_lambda: float = 10.0):
        self.n = len(relevance)
        self.k_target = k_target
        self.penalty_lambda = penalty_lambda
        
        Q = np.zeros((self.n, self.n), dtype=np.float64)
        # 1. Relevance: -relevance on diagonal
        for i in range(self.n):
            Q[i, i] -= relevance[i]
            
        # 2. Redundancy: off-diagonal
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[i, j] += redundancy[i, j]

        # 3. Penalty term: lambda * (sum x_i - K)^2
        # (sum x_i - K)^2 = sum x_i^2 + 2 sum_{i<j} x_i x_j - 2 K sum x_i + K^2
        # Since x_i^2 = x_i:
        # Diagonal update: lambda * (1 - 2*K)
        # Off-diagonal update: 2 * lambda
        for i in range(self.n):
            Q[i, i] += penalty_lambda * (1.0 - 2.0 * k_target)
            for j in range(i + 1, self.n):
                Q[i, j] += 2.0 * penalty_lambda
                
        self.Q = Q
        self.constant_offset = penalty_lambda * (k_target ** 2)

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(x.T @ self.Q @ x + self.constant_offset)
