"""
Test-Suite Minimization (TSM) and Prioritization (TSP) Engine.
Formulated as 0-1 Integer Linear Programming (Set Cover) and Greedy APFD Optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from core.metrics import apfd_score

class TestMetadata:
    def __init__(self, test_id: int, name: str, duration_ms: float, covered_branches: set[int]):
        self.id = test_id
        self.name = name
        self.duration_ms = duration_ms
        self.covered_branches = covered_branches

def build_test_suite_model(num_tests: int = 40, num_branches: int = 50, seed: int = 42):
    """
    Constructs realistic test suite metadata with heterogeneous execution times,
    overlapping branch coverage, and synthetic fault injection matrix.
    """
    rng = np.random.default_rng(seed)
    tests = []

    for i in range(num_tests):
        # Durations range from 5ms (fast unit test) to 120ms (integration/data parser test)
        dur = float(round(rng.uniform(5.0, 100.0), 1))
        # Each test covers 4 to 12 random branches
        k_cov = int(rng.integers(4, 13))
        cov = set(rng.choice(num_branches, size=k_cov, replace=False).tolist())
        tests.append(TestMetadata(i, f"test_{i:02d}", dur, cov))

    # Ensure all branches are covered by at least one test
    for b in range(num_branches):
        if not any(b in t.covered_branches for t in tests):
            # Assign to random test
            pick = int(rng.integers(0, num_tests))
            tests[pick].covered_branches.add(b)

    # Synthetic Fault Matrix: 10 hidden regression faults
    num_faults = 10
    fault_matrix = np.zeros((num_tests, num_faults), dtype=int)
    for f in range(num_faults):
        # Each fault is triggered by 2 to 6 specific tests
        n_detectors = int(rng.integers(2, 7))
        detecting_tests = rng.choice(num_tests, size=n_detectors, replace=False)
        for t_idx in detecting_tests:
            fault_matrix[t_idx, f] = 1

    return tests, num_branches, fault_matrix

def solve_greedy_test_minimization(tests: list[TestMetadata], total_branches: int) -> dict:
    """Greedy Set Cover heuristic: iteratively picks test that covers most uncovered branches per millisecond."""
    start_t = time.perf_counter()
    uncovered = set(range(total_branches))
    selected = []
    current_time = 0.0

    while uncovered:
        best_test = None
        best_efficiency = -1.0
        best_new_cov = set()

        for t in tests:
            if t.id not in selected:
                new_cov = t.covered_branches.intersection(uncovered)
                if len(new_cov) > 0:
                    efficiency = len(new_cov) / t.duration_ms
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_test = t
                        best_new_cov = new_cov

        if best_test is None:
            break

        selected.append(best_test.id)
        current_time += best_test.duration_ms
        uncovered.difference_update(best_new_cov)

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Greedy Additional Coverage',
        'selected_test_ids': selected,
        'num_selected_tests': len(selected),
        'total_duration_ms': current_time,
        'branches_covered': total_branches - len(uncovered),
        'coverage_percentage': ((total_branches - len(uncovered)) / total_branches) * 100.0,
        'optimization_runtime_ms': runtime * 1000.0
    }

def solve_ilp_test_minimization(tests: list[TestMetadata], total_branches: int) -> dict:
    """
    Exact 0-1 Integer Linear Programming (Set Cover):
    Minimize sum_i duration_i * x_i
    Subject to sum_{i: b in B_i} x_i >= 1 for all branches b
    """
    start_t = time.perf_counter()
    n = len(tests)
    c = np.array([t.duration_ms for t in tests])

    # Constraint matrix A: (num_branches, n)
    A = np.zeros((total_branches, n))
    for b in range(total_branches):
        for i, t in enumerate(tests):
            if b in t.covered_branches:
                A[b, i] = 1.0

    constraints = LinearConstraint(A, lb=np.ones(total_branches), ub=np.full(total_branches, np.inf))
    integrality = np.ones(n)
    bounds = Bounds(lb=0.0, ub=1.0)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    runtime = time.perf_counter() - start_t

    sol = np.round(res.x).astype(int)
    selected_ids = [i for i, val in enumerate(sol) if val == 1]
    tot_dur = float(res.fun)

    # Verify coverage
    covered = set()
    for idx in selected_ids:
        covered.update(tests[idx].covered_branches)

    return {
        'method': 'Exact ILP Set Cover',
        'selected_test_ids': selected_ids,
        'num_selected_tests': len(selected_ids),
        'total_duration_ms': tot_dur,
        'branches_covered': len(covered),
        'coverage_percentage': (len(covered) / total_branches) * 100.0,
        'optimization_runtime_ms': runtime * 1000.0
    }

def prioritize_tests_greedy(test_ids: list[int], tests: list[TestMetadata], fault_matrix: np.ndarray) -> list[int]:
    """Prioritizes tests to maximize rate of fault detection (APFD)."""
    # Sort by number of detectable faults / duration
    scores = []
    for tid in test_ids:
        n_faults = np.sum(fault_matrix[tid])
        dur = tests[tid].duration_ms
        scores.append((n_faults / dur, tid))
    # Highest score first
    scores.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in scores]
