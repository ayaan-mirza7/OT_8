"""
Complexity and Algorithmic Advantage Comparison: Grover vs Classical Linear Search.
Contrasts theoretical asymptotic query counts with physical gate counts and classical simulation wall-clock times.
Demonstrates why classical simulation of quantum algorithms suffers exponential overhead.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np
import pandas as pd
from tracks.track_i_quantum_advantage.algorithms import run_grovers_search

def run_classical_linear_search(database_size: int, target_idx: int) -> tuple[int, float]:
    """Classical sequential linear search: checks items until target is found."""
    start_t = time.perf_counter()
    queries = 0
    for i in range(database_size):
        queries += 1
        if i == target_idx:
            break
    runtime_ms = (time.perf_counter() - start_t) * 1000.0
    return queries, runtime_ms

def compare_grover_vs_linear_search():
    # Test across n = 3, 4, 5, 6, 7, 8 qubits (N = 8 to 256)
    results = []

    for n in [3, 4, 5, 6, 7, 8]:
        N = 1 << n
        target = N - 3  # towards end to test worst/average case

        # Classical Search
        class_queries, class_time_ms = run_classical_linear_search(N, target)

        # Quantum Grover Simulation
        start_t = time.perf_counter()
        q_res = run_grovers_search(num_qubits=n, target_index=target)
        grover_sim_time_ms = (time.perf_counter() - start_t) * 1000.0

        # Estimated physical QPU gate count ~ O(n^2 * sqrt(N))
        est_gates = int(round(q_res['optimal_iterations'] * (n * 2 + 2 * n)))

        results.append({
            'num_qubits': n,
            'database_size_N': N,
            'classical_queries': class_queries,
            'classical_runtime_ms': round(class_time_ms, 5),
            'quantum_queries': q_res['quantum_queries'],
            'theoretical_speedup_queries': round(class_queries / q_res['quantum_queries'], 2),
            'quantum_gate_count': est_gates,
            'grover_simulation_time_ms': round(grover_sim_time_ms, 3),
            'simulation_overhead_ratio': round(grover_sim_time_ms / max(class_time_ms, 1e-5), 1),
            'grover_success_prob': round(q_res['success_probability'], 3)
        })

    return pd.DataFrame(results)
