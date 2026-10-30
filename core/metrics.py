"""
Core utilities and mathematical metrics for Optimization Techniques Assignment 8.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

def amdahls_law(speedup_parallel_portion: float, parallel_fraction: float) -> float:
    """Computes theoretical speedup under Amdahl's Law: S(p) = 1 / ((1 - f) + f / s)."""
    assert 0.0 <= parallel_fraction <= 1.0, "Parallel fraction must be in [0, 1]"
    assert speedup_parallel_portion > 0, "Speedup portion must be positive"
    return 1.0 / ((1.0 - parallel_fraction) + (parallel_fraction / speedup_parallel_portion))

def gustafsons_law(cores: int, parallel_fraction: float) -> float:
    """Computes scaled speedup under Gustafson's Law: S(p) = (1 - f) + f * p."""
    assert 0.0 <= parallel_fraction <= 1.0, "Parallel fraction must be in [0, 1]"
    assert cores >= 1, "Cores must be >= 1"
    return (1.0 - parallel_fraction) + parallel_fraction * cores

def jains_fairness_index(allocations: list | np.ndarray) -> float:
    """Computes Jain's Fairness Index: J = (sum(x_i))^2 / (n * sum(x_i^2))."""
    x = np.asarray(allocations, dtype=np.float64)
    if len(x) == 0 or np.all(x == 0):
        return 1.0
    return float((np.sum(x) ** 2) / (len(x) * np.sum(x ** 2)))

def speedup_and_efficiency(t_serial: float, t_parallel: float, num_workers: int):
    """Computes speedup S = T_1 / T_p and parallel efficiency E = S / p."""
    speedup = t_serial / max(t_parallel, 1e-9)
    efficiency = speedup / max(num_workers, 1)
    return float(speedup), float(efficiency)

def apfd_score(test_order: list[int], fault_matrix: np.ndarray) -> float:
    """
    Computes Average Percentage of Faults Detected (APFD) for test prioritization.
    formula: APFD = 1 - (sum(TF_i) / (n * m)) + 1 / (2 * n)
    where:
      n = number of tests
      m = number of faults
      TF_i = position of first test detecting fault i
    """
    n_tests, n_faults = fault_matrix.shape
    if n_faults == 0:
        return 1.0
    
    tf = []
    for f in range(n_faults):
        first_pos = None
        for rank, test_idx in enumerate(test_order, start=1):
            if fault_matrix[test_idx, f] > 0:
                first_pos = rank
                break
        tf.append(first_pos if first_pos is not None else n_tests)
    
    return float(1.0 - (sum(tf) / (n_tests * n_faults)) + (1.0 / (2.0 * n_tests)))
