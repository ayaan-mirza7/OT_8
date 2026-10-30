"""
Classical Solvers for QUBO and Max-Cut Problems.
Provides the classical baseline against which Track C (Quantum QAOA) is benchmarked.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np
from tracks.track_b_discrete.qubo_formulation import MaxCutQUBO

def solve_exact_maxcut(qubo: MaxCutQUBO) -> dict:
    """Exact exhaustive search over all 2^N binary assignments."""
    start_t = time.perf_counter()
    n = qubo.n
    assert n <= 20, "Exact solver only practical for n <= 20"

    best_cut = -1.0
    best_x = None

    for i in range(1 << n):
        x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=int)
        cut = qubo.evaluate_cut(x)
        if cut > best_cut:
            best_cut = cut
            best_x = x

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Exact Exhaustive Search',
        'cut_value': float(best_cut),
        'qubo_energy': qubo.evaluate_qubo(best_x),
        'best_state': best_x.tolist(),
        'runtime_ms': float(runtime * 1000.0)
    }


def solve_simulated_annealing_maxcut(
    qubo: MaxCutQUBO,
    t_init: float = 10.0,
    t_min: float = 1e-3,
    alpha: float = 0.95,
    steps_per_t: int = 50,
    seed: int = 42
) -> dict:
    """Metropolis-Hastings Simulated Annealing for Max-Cut QUBO."""
    start_t = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n

    # Random initial bitstring
    current_x = rng.integers(0, 2, size=n)
    current_energy = qubo.evaluate_qubo(current_x)
    current_cut = qubo.evaluate_cut(current_x)

    best_x = current_x.copy()
    best_cut = current_cut
    best_energy = current_energy

    T = t_init
    energy_history = [current_energy]

    while T > t_min:
        for _ in range(steps_per_t):
            # Select random bit to flip
            flip_idx = rng.integers(0, n)
            candidate_x = current_x.copy()
            candidate_x[flip_idx] = 1 - candidate_x[flip_idx]

            candidate_energy = qubo.evaluate_qubo(candidate_x)
            delta_e = candidate_energy - current_energy

            # Metropolis acceptance (minimizing energy)
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_x = candidate_x
                current_energy = candidate_energy
                current_cut = qubo.evaluate_cut(current_x)

                if current_cut > best_cut:
                    best_cut = current_cut
                    best_x = current_x.copy()
                    best_energy = current_energy

        energy_history.append(current_energy)
        T *= alpha

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Simulated Annealing',
        'cut_value': float(best_cut),
        'qubo_energy': float(best_energy),
        'best_state': best_x.tolist(),
        'runtime_ms': float(runtime * 1000.0),
        'history': energy_history
    }


def solve_tabu_search_maxcut(
    qubo: MaxCutQUBO,
    max_iters: int = 150,
    tabu_tenure: int = 4,
    seed: int = 42
) -> dict:
    """Tabu Search metaheuristic with short-term recency memory."""
    start_t = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n

    current_x = rng.integers(0, 2, size=n)
    best_x = current_x.copy()
    best_cut = qubo.evaluate_cut(current_x)

    tabu_list = np.zeros(n, dtype=int)  # stores iteration until which bit is tabu

    for it in range(1, max_iters + 1):
        best_neighbor_x = None
        best_neighbor_cut = -1.0
        best_flip_idx = -1

        for flip_idx in range(n):
            cand_x = current_x.copy()
            cand_x[flip_idx] = 1 - cand_x[flip_idx]
            cand_cut = qubo.evaluate_cut(cand_x)

            # Check tabu status with aspiration criterion (allow if strictly beats global best)
            is_tabu = tabu_list[flip_idx] >= it
            if (not is_tabu) or (cand_cut > best_cut):
                if cand_cut > best_neighbor_cut:
                    best_neighbor_cut = cand_cut
                    best_neighbor_x = cand_x
                    best_flip_idx = flip_idx

        if best_neighbor_x is not None:
            current_x = best_neighbor_x
            tabu_list[best_flip_idx] = it + tabu_tenure
            if best_neighbor_cut > best_cut:
                best_cut = best_neighbor_cut
                best_x = current_x.copy()

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Tabu Search',
        'cut_value': float(best_cut),
        'qubo_energy': qubo.evaluate_qubo(best_x),
        'best_state': best_x.tolist(),
        'runtime_ms': float(runtime * 1000.0)
    }
