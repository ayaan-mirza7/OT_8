"""
Track I Quantum Algorithms and QML Advantage Experiment Runner.
Executes:
1. Suite of 7 fundamental quantum algorithms
2. QML (VQC) vs Classical MLP comparative trial
3. Grover vs Classical Linear Search complexity scaling
4. Detailed breakdown of quantum limits: simulation overhead, barren plateaus, and data loading
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_i_quantum_advantage.algorithms import (
    run_deutsch_jozsa, run_bernstein_vazirani, run_grovers_search,
    run_qft_demonstration, run_quantum_phase_estimation,
    run_shor_order_finding, run_quantum_annealing
)
from tracks.track_i_quantum_advantage.qml_classifier import (
    VariationalQuantumClassifier, run_classical_classifier_baseline
)
from tracks.track_i_quantum_advantage.complexity_comparison import compare_grover_vs_linear_search

def main():
    print("[Track I] Running Quantum Algorithms & QML Advantage Benchmark...")
    os.makedirs("tracks/track_i_quantum_advantage/results", exist_ok=True)

    # 1. Run all 7 Quantum Algorithms
    algo_results = [
        run_deutsch_jozsa(num_qubits=3, is_constant=False),
        run_bernstein_vazirani(hidden_secret="1011"),
        run_grovers_search(num_qubits=4, target_index=9),
        run_qft_demonstration(n_qubits=3),
        run_quantum_phase_estimation(true_phase=0.375, n_count_qubits=3),
        run_shor_order_finding(a=7, N=15),
        run_quantum_annealing(n_spins=6, total_steps=40)
    ]
    with open("tracks/track_i_quantum_advantage/results/quantum_algorithms_suite.json", "w") as f:
        # Convert non-serializable objects
        clean_res = []
        for r in algo_results:
            c = dict(r)
            if 'history' in c:
                del c['history']
            clean_res.append(c)
        json.dump(clean_res, f, indent=2)

    # 2. QML vs Classical Classifier Trial
    rng = np.random.default_rng(42)
    # Generate small 2D dataset suitable for VQC angle embedding
    N = 40
    X = rng.uniform(-1.5, 1.5, (N, 2))
    # Decision boundary: x0 * x1 > 0 (quadrants 1 & 3 vs 2 & 4)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.float64)

    vqc = VariationalQuantumClassifier(num_qubits=2, layers=2)
    vqc_res = vqc.train(X, y, epochs=15, lr=0.1, seed=42)
    mlp_res = run_classical_classifier_baseline(X, y, epochs=50, seed=42)

    qml_df = pd.DataFrame([vqc_res, mlp_res])
    qml_df.to_csv("tracks/track_i_quantum_advantage/results/qml_vs_classical.csv", index=False)

    # 3. Grover vs Classical Complexity Comparison
    grover_df = compare_grover_vs_linear_search()
    grover_df.to_csv("tracks/track_i_quantum_advantage/results/grover_vs_linear_search.csv", index=False)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Track I: Quantum Advantage & Complexity Trade-Offs\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: Query Complexity (Grover O(sqrt(N)) vs Classical O(N))
    ax = axes[0]
    ax.plot(grover_df['database_size_N'], grover_df['classical_queries'], 'r-o', lw=2, label='Classical Linear Queries O(N)')
    ax.plot(grover_df['database_size_N'], grover_df['quantum_queries'], 'b--s', lw=2, label='Quantum Grover Queries O(sqrt(N))')
    ax.set_xlabel("Database Size (N)")
    ax.set_ylabel("Query Evaluations")
    ax.set_title("Algorithmic Advantage: Query Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Classical Simulation Wall-Clock Time (The Honest Negative Result)
    ax = axes[1]
    ax.plot(grover_df['database_size_N'], grover_df['classical_runtime_ms'], 'r-o', lw=2, label='Classical Search Wall-Clock (ms)')
    ax.plot(grover_df['database_size_N'], grover_df['grover_simulation_time_ms'], 'purple', linestyle='--', marker='^', lw=2, label='Grover Classical Sim Wall-Clock (ms)')
    ax.set_yscale('log')
    ax.set_xlabel("Database Size (N)")
    ax.set_ylabel("Wall-Clock Time (ms) [log scale]")
    ax.set_title("Simulation Reality: Classical Sim Overhead")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_i_quantum_advantage/results/track_i_quantum_plots.png", dpi=300)
    plt.close()

    print("[Track I] Completed successfully. Artifacts saved in tracks/track_i_quantum_advantage/results/")

if __name__ == '__main__':
    main()
