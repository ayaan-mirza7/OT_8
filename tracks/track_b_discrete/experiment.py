"""
Track B Experiment Runner.
Executes benchmarks for:
1. Combinatorial Solvers (Greedy vs DP vs Branch-and-Bound vs ILP)
2. Max-Cut QUBO / Ising formulation (Exact vs Simulated Annealing vs Tabu Search)
3. Constrained Feature Selection with quadratic penalties
4. Saves baseline instances for Track C (QAOA)
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_b_discrete.combinatorial_solvers import (
    KnapsackProblem, solve_greedy_knapsack, solve_dp_knapsack,
    solve_branch_and_bound_knapsack, solve_ilp_knapsack
)
from tracks.track_b_discrete.qubo_formulation import MaxCutQUBO, ConstrainedFeatureSelectionQUBO
from tracks.track_b_discrete.classical_qubo_solvers import (
    solve_exact_maxcut, solve_simulated_annealing_maxcut, solve_tabu_search_maxcut
)

def run_knapsack_suite():
    # Realistic test instance with 16 items
    rng = np.random.default_rng(42)
    n = 16
    values = rng.integers(10, 100, size=n).tolist()
    weights = rng.integers(5, 50, size=n).tolist()
    capacity = int(0.5 * sum(weights))

    problem = KnapsackProblem(values, weights, capacity)

    results = [
        solve_greedy_knapsack(problem),
        solve_dp_knapsack(problem),
        solve_branch_and_bound_knapsack(problem),
        solve_ilp_knapsack(problem)
    ]
    df = pd.DataFrame(results)
    optimal_val = df['value'].max()
    df['optimality_gap_%'] = ((optimal_val - df['value']) / optimal_val) * 100.0
    return df, problem

def run_maxcut_suite():
    # 8-node benchmark graph (e.g., 3-regular cubic graph / cycle with chords)
    # Perfect size for classical exact verification and Track C Quantum QAOA simulation!
    n = 8
    adj = np.zeros((n, n), dtype=np.float64)
    edges = [
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
        (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 0, 1.0),
        (0, 4, 1.5), (1, 5, 1.5), (2, 6, 1.5), (3, 7, 1.5)
    ]
    for u, v, w in edges:
        adj[u, v] = w
        adj[v, u] = w

    qubo = MaxCutQUBO(adj)

    exact_res = solve_exact_maxcut(qubo)
    sa_res = solve_simulated_annealing_maxcut(qubo, seed=42)
    tabu_res = solve_tabu_search_maxcut(qubo, seed=42)

    results = [exact_res, sa_res, tabu_res]
    df = pd.DataFrame(results)

    # Save exact instance and baseline for Track C
    baseline_payload = {
        'num_nodes': n,
        'adjacency_matrix': adj.tolist(),
        'qubo_matrix': qubo.Q.tolist(),
        'exact_max_cut': exact_res['cut_value'],
        'exact_optimal_state': exact_res['best_state'],
        'sa_cut': sa_res['cut_value'],
        'tabu_cut': tabu_res['cut_value']
    }
    with open("tracks/track_b_discrete/results/maxcut_baseline.json", "w") as f:
        json.dump(baseline_payload, f, indent=2)

    return df, qubo, sa_res

def run_penalty_ablation():
    """Demonstrates how quadratic penalty lambda forces cardinality constraint satisfaction."""
    relevance = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    redundancy = np.full((6, 6), 0.2)
    np.fill_diagonal(redundancy, 0.0)
    k_target = 3  # exactly 3 features

    penalty_lambdas = [0.1, 1.0, 5.0, 20.0]
    records = []

    for lam in penalty_lambdas:
        fs_qubo = ConstrainedFeatureSelectionQUBO(relevance, redundancy, k_target=k_target, penalty_lambda=lam)
        # Solve via small brute force
        best_e = float('inf')
        best_x = None
        for i in range(1 << 6):
            x = np.array([(i >> bit) & 1 for bit in range(6)], dtype=int)
            e = fs_qubo.evaluate(x)
            if e < best_e:
                best_e = e
                best_x = x
        selected_count = int(np.sum(best_x))
        satisfied = (selected_count == k_target)
        records.append({
            'penalty_lambda': lam,
            'k_target': k_target,
            'actual_selected': selected_count,
            'constraint_satisfied': satisfied,
            'best_solution': best_x.tolist(),
            'qubo_energy': best_e
        })
    return pd.DataFrame(records)

def main():
    print("[Track B] Running Discrete Optimization & QUBO Experiments...")
    os.makedirs("tracks/track_b_discrete/results", exist_ok=True)

    # 1. Knapsack suite
    knapsack_df, knapsack_prob = run_knapsack_suite()
    knapsack_df.to_csv("tracks/track_b_discrete/results/knapsack_benchmark.csv", index=False)

    # 2. Max-Cut QUBO suite
    maxcut_df, qubo, sa_res = run_maxcut_suite()
    maxcut_df.to_csv("tracks/track_b_discrete/results/maxcut_qubo_benchmark.csv", index=False)

    # 3. Penalty ablation
    penalty_df = run_penalty_ablation()
    penalty_df.to_csv("tracks/track_b_discrete/results/penalty_ablation.csv", index=False)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Track B: Discrete Optimization & QUBO Baselines\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: Combinatorial Runtime vs Solution Quality
    ax = axes[0]
    methods = knapsack_df['method'].tolist()
    values = knapsack_df['value'].tolist()
    runtimes = knapsack_df['runtime_ms'].tolist()
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos - 0.2, values, width=0.4, label='Knapsack Value', color='#2563eb')
    ax.set_ylabel("Total Objective Value", color='#2563eb')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x_pos + 0.2, runtimes, 'r-o', lw=2, label='Runtime (ms)')
    ax2.set_ylabel("Runtime (ms) [log scale]", color='r')
    ax2.set_yscale('log')
    ax.set_title("Combinatorial Knapsack Comparison")

    # Plot 2: Simulated Annealing Energy Trajectory for QUBO
    ax = axes[1]
    ax.plot(sa_res['history'], color='#059669', lw=2)
    ax.set_title("Simulated Annealing Energy Trajectory (Max-Cut QUBO)")
    ax.set_xlabel("Cooling Step")
    ax.set_ylabel("QUBO Energy E(x)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_b_discrete/results/track_b_discrete_plots.png", dpi=300)
    plt.close()

    print("[Track B] Completed successfully. Artifacts saved in tracks/track_b_discrete/results/")

if __name__ == '__main__':
    main()
