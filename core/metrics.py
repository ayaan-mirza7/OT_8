"""
Core utilities and mathematical metrics for Optimization Techniques Assignment 8.
Student: Ayaan Mirza (Roll No: 241070043)
"""

from __future__ import annotations
import numpy as np

__all__ = [
    "amdahls_law",
    "gustafsons_law",
    "jains_fairness_index",
    "speedup_and_efficiency",
    "apfd_score",
]

def amdahls_law(speedup_parallel_portion: float, parallel_fraction: float) -> float:
    """
    Computes theoretical speedup under Amdahl's Law:
    S(p) = 1 / ((1 - f) + f / s)

    Args:
        speedup_parallel_portion: Speedup factor of the parallelizable part (> 0).
        parallel_fraction: Proportion of execution time that is parallelizable [0, 1].

    Returns:
        float: Theoretical overall system speedup.
    """
    assert 0.0 <= parallel_fraction <= 1.0, "Parallel fraction must be in [0, 1]"
    assert speedup_parallel_portion > 0, "Speedup portion must be positive"
    return 1.0 / ((1.0 - parallel_fraction) + (parallel_fraction / speedup_parallel_portion))

def gustafsons_law(cores: int, parallel_fraction: float) -> float:
    """
    Computes scaled speedup under Gustafson's Law:
    S(p) = (1 - f) + f * p

    Args:
        cores: Number of parallel processing cores (>= 1).
        parallel_fraction: Proportion of parallelizable workload [0, 1].

    Returns:
        float: Scaled speedup factor under weak scaling.
    """
    assert 0.0 <= parallel_fraction <= 1.0, "Parallel fraction must be in [0, 1]"
    assert cores >= 1, "Cores must be >= 1"
    return (1.0 - parallel_fraction) + parallel_fraction * cores

def jains_fairness_index(allocations: list[float] | np.ndarray) -> float:
    """
    Computes Jain's Fairness Index:
    J(x) = (sum(x_i))^2 / (n * sum(x_i^2))

    Args:
        allocations: Resource allocation vector across entities.

    Returns:
        float: Fairness index between 1/n (worst) and 1.0 (perfectly fair).
    """
    x = np.asarray(allocations, dtype=np.float64)
    if len(x) == 0 or np.all(x == 0):
        return 1.0
    return float((np.sum(x) ** 2) / (len(x) * np.sum(x ** 2)))

def speedup_and_efficiency(t_serial: float, t_parallel: float, num_workers: int) -> tuple[float, float]:
    """
    Computes parallel speedup and efficiency:
    S = T_1 / T_p,  E = S / p

    Args:
        t_serial: Single-core execution runtime in seconds.
        t_parallel: Multi-core parallel runtime in seconds.
        num_workers: Total worker count.

    Returns:
        tuple[float, float]: (Speedup factor, parallel efficiency).
    """
    speedup = t_serial / max(t_parallel, 1e-9)
    efficiency = speedup / max(num_workers, 1)
    return float(speedup), float(efficiency)

def apfd_score(test_order: list[int], fault_matrix: np.ndarray) -> float:
    """
    Computes Average Percentage of Faults Detected (APFD) for test prioritization:
    APFD = 1 - (sum(TF_i) / (n * m)) + 1 / (2 * n)

    Args:
        test_order: Ordered permutation of test suite indices.
        fault_matrix: Binary fault matrix of shape (n_tests, n_faults).

    Returns:
        float: APFD metric score in [0, 1].
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
