"""
Track D Experiment Runner:
Submodular Optimization, Parallel Computing, Federated Learning, and Multi-Task Optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_d_parallel_distributed.submodular import FacilityLocationSubmodular
from tracks.track_d_parallel_distributed.parallel_runtime import (
    run_serial_workload, run_parallel_workload, WorkStealingSimulation
)
from tracks.track_d_parallel_distributed.federated import (
    FederatedDataset, run_centralized_training, run_fedavg
)
from tracks.track_d_parallel_distributed.multi_task import run_multi_task_experiment
from core.metrics import amdahls_law, gustafsons_law, speedup_and_efficiency

def run_submodular_suite():
    print(" -> Running Submodular Facility Location...")
    fac = FacilityLocationSubmodular(num_facilities=12, num_targets=35, seed=42)
    verify_res = fac.verify_diminishing_returns(num_trials=100, seed=42)

    k_values = [2, 3, 4, 5, 6]
    records = []
    for k in k_values:
        greedy_res = fac.greedy_selection(k)
        exact_res = fac.exact_optimal_selection(k)
        approx_ratio = greedy_res['objective_value'] / exact_res['optimal_value']
        records.append({
            'k': k,
            'greedy_value': greedy_res['objective_value'],
            'optimal_value': exact_res['optimal_value'],
            'empirical_ratio': approx_ratio,
            'theoretical_guarantee': 1.0 - (1.0 / np.e),  # ~0.6321
            'guarantee_satisfied': approx_ratio >= (1.0 - (1.0 / np.e)),
            'greedy_runtime_ms': greedy_res['runtime_ms'],
            'exact_runtime_ms': exact_res['runtime_ms']
        })
    return pd.DataFrame(records), verify_res

def run_parallel_scaling_suite():
    print(" -> Running Parallel Worker Scaling & Amdahl/Gustafson Analysis...")
    # Generate heterogeneous tasks
    rng = np.random.default_rng(42)
    tasks = [(i, int(rng.integers(1500, 3500))) for i in range(48)]

    # Serial baseline
    t1, _ = run_serial_workload(tasks)

    worker_counts = [1, 2, 4, 8]
    records = []
    # Assume 90% parallelizable fraction
    parallel_frac = 0.88

    for p in worker_counts:
        if p == 1:
            tp = t1
        else:
            tp, _ = run_parallel_workload(tasks, num_workers=p)
        sp, eff = speedup_and_efficiency(t1, tp, p)
        amdahl_sp = amdahls_law(p, parallel_frac)
        gustafson_sp = gustafsons_law(p, parallel_frac)

        records.append({
            'workers': p,
            'measured_time_sec': tp,
            'empirical_speedup': sp,
            'parallel_efficiency': eff,
            'amdahl_theoretical_speedup': amdahl_sp,
            'gustafson_theoretical_speedup': gustafson_sp
        })

    # Work Stealing Simulation
    task_costs = [t[1] for t in tasks]
    ws_sim = WorkStealingSimulation(task_costs, num_workers=4)
    static_res = ws_sim.run_static_partitioning()
    stealing_res = ws_sim.run_work_stealing()

    ws_comparison = {
        'static_makespan': static_res['makespan'],
        'static_imbalance': static_res['imbalance_ratio'],
        'work_stealing_makespan': stealing_res['makespan'],
        'work_stealing_imbalance': stealing_res['imbalance_ratio'],
        'steals_count': stealing_res['steals_count']
    }

    return pd.DataFrame(records), ws_comparison

def run_federated_suite():
    print(" -> Running Federated Learning (FedAvg vs Centralized)...")
    fed_data = FederatedDataset(num_clients=5, n_samples_per_client=60, dim=4, seed=42)
    cent_res = run_centralized_training(fed_data, total_steps=80, lr=0.1)

    fed_e1 = run_fedavg(fed_data, rounds=16, local_epochs=1, lr=0.1)
    fed_e5 = run_fedavg(fed_data, rounds=16, local_epochs=5, lr=0.1)
    fed_e10 = run_fedavg(fed_data, rounds=16, local_epochs=10, lr=0.1)

    fed_df = pd.DataFrame([
        {
            'Method': cent_res['method'],
            'Final_Loss': cent_res['final_loss'],
            'Rounds': 0,
            'Total_Bytes': cent_res['total_bytes_transferred'],
            'Runtime_ms': cent_res['runtime_ms']
        },
        {
            'Method': fed_e1['method'],
            'Final_Loss': fed_e1['final_loss'],
            'Rounds': fed_e1['rounds'],
            'Total_Bytes': fed_e1['total_bytes_transferred'],
            'Runtime_ms': fed_e1['runtime_ms']
        },
        {
            'Method': fed_e5['method'],
            'Final_Loss': fed_e5['final_loss'],
            'Rounds': fed_e5['rounds'],
            'Total_Bytes': fed_e5['total_bytes_transferred'],
            'Runtime_ms': fed_e5['runtime_ms']
        },
        {
            'Method': fed_e10['method'],
            'Final_Loss': fed_e10['final_loss'],
            'Rounds': fed_e10['rounds'],
            'Total_Bytes': fed_e10['total_bytes_transferred'],
            'Runtime_ms': fed_e10['runtime_ms']
        }
    ])
    return fed_df, [cent_res, fed_e1, fed_e5, fed_e10]

def main():
    print("[Track D] Running Submodular, Parallel, and Distributed Optimization...")
    os.makedirs("tracks/track_d_parallel_distributed/results", exist_ok=True)

    # 1. Submodular
    submod_df, verify_res = run_submodular_suite()
    submod_df.to_csv("tracks/track_d_parallel_distributed/results/submodular_benchmark.csv", index=False)
    with open("tracks/track_d_parallel_distributed/results/submodular_verification.json", "w") as f:
        json.dump(verify_res, f, indent=2)

    # 2. Parallel Scaling
    parallel_df, ws_comp = run_parallel_scaling_suite()
    parallel_df.to_csv("tracks/track_d_parallel_distributed/results/parallel_scaling.csv", index=False)
    with open("tracks/track_d_parallel_distributed/results/work_stealing_comparison.json", "w") as f:
        json.dump(ws_comp, f, indent=2)

    # 3. Federated Learning
    fed_df, fed_runs = run_federated_suite()
    fed_df.to_csv("tracks/track_d_parallel_distributed/results/federated_benchmark.csv", index=False)

    # 4. Multi-Task Optimization
    mt_res = run_multi_task_experiment(n_steps=80, lr=0.04, seed=42)
    mt_df = pd.DataFrame([
        {
            'Method': k,
            'Final_Task1_Loss': v['final_l1'],
            'Final_Task2_Loss': v['final_l2'],
            'Combined_Loss': v['combined_final_loss']
        }
        for k, v in mt_res.items()
    ])
    mt_df.to_csv("tracks/track_d_parallel_distributed/results/multi_task_benchmark.csv", index=False)

    # Visualization Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Track D: Submodular, Parallel & Distributed Optimization\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: Submodular Greedy Ratio vs (1 - 1/e) Guarantee
    ax = axes[0, 0]
    ax.plot(submod_df['k'], submod_df['empirical_ratio'], 'o-', label='Empirical Ratio (Greedy / Opt)', color='#2563eb', lw=2)
    ax.axhline(1.0 - (1.0 / np.e), color='r', linestyle='--', label='Theoretical Guarantee (1 - 1/e ~ 0.632)', lw=1.8)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Submodular Facility Location Approximation Ratio")
    ax.set_xlabel("Selected Facilities (k)")
    ax.set_ylabel("Approximation Ratio")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Parallel Speedup vs Amdahl & Gustafson Laws
    ax = axes[0, 1]
    ax.plot(parallel_df['workers'], parallel_df['empirical_speedup'], 'o-', label='Empirical Speedup', color='#059669', lw=2)
    ax.plot(parallel_df['workers'], parallel_df['amdahl_theoretical_speedup'], '--', label="Amdahl's Law (f=0.88)", color='#dc2626')
    ax.plot(parallel_df['workers'], parallel_df['gustafson_theoretical_speedup'], ':', label="Gustafson's Law (f=0.88)", color='#7c3aed')
    ax.plot(parallel_df['workers'], parallel_df['workers'], 'k-.', alpha=0.3, label='Linear Ideal')
    ax.set_title("Parallel Speedup: Empirical vs Amdahl & Gustafson")
    ax.set_xlabel("Worker Processes (p)")
    ax.set_ylabel("Speedup S(p)")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Federated Learning Convergence Trajectory
    ax = axes[1, 0]
    for r in fed_runs:
        ax.plot(r['history'], label=r['method'], lw=1.8)
    ax.set_title("Federated Learning Loss Convergence")
    ax.set_xlabel("Rounds / Steps")
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Multi-Task Optimization (Task 1 vs Task 2 Loss)
    ax = axes[1, 1]
    for method, data in mt_res.items():
        ax.plot(data['l1_history'], label=f"{method} (T1)", lw=1.8)
        ax.plot(data['l2_history'], linestyle='--', label=f"{method} (T2)", lw=1.8)
    ax.set_title("Multi-Task Optimization Trajectory (MGDA vs Equal)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Task Loss")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_d_parallel_distributed/results/track_d_plots.png", dpi=300)
    plt.close()

    print("[Track D] Completed successfully. Artifacts saved in tracks/track_d_parallel_distributed/results/")

if __name__ == '__main__':
    main()
