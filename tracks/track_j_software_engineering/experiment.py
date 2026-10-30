"""
Track J Software-Engineering Optimization Experiment Runner.
Executes:
1. Baseline test suite execution profiling
2. Test-Suite Minimization via Greedy vs Exact ILP Set Cover
3. Test Prioritization & Average Percentage of Faults Detected (APFD) evaluation
4. Verification of 100% branch coverage preservation
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_j_software_engineering.test_optimizer import (
    build_test_suite_model,
    solve_greedy_test_minimization,
    solve_ilp_test_minimization,
    prioritize_tests_greedy
)
from core.metrics import apfd_score

def main():
    print("[Track J] Running Software-Engineering Optimization Benchmark...")
    os.makedirs("tracks/track_j_software_engineering/results", exist_ok=True)

    # 1. Build test model for codebase
    tests, total_branches, fault_matrix = build_test_suite_model(num_tests=40, num_branches=50, seed=42)

    # Baseline: Execute all 40 tests in arbitrary order
    all_ids = list(range(len(tests)))
    base_duration = sum(t.duration_ms for t in tests)
    base_apfd = apfd_score(all_ids, fault_matrix)

    baseline_record = {
        'method': 'Baseline All Tests (Unoptimized)',
        'num_selected_tests': len(tests),
        'total_duration_ms': base_duration,
        'duration_savings_%': 0.0,
        'branches_covered': total_branches,
        'coverage_percentage': 100.0,
        'apfd_score': base_apfd,
        'optimization_runtime_ms': 0.0
    }

    # 2. Greedy Minimization
    greedy_res = solve_greedy_test_minimization(tests, total_branches)
    greedy_order = prioritize_tests_greedy(greedy_res['selected_test_ids'], tests, fault_matrix)
    greedy_apfd = apfd_score(greedy_order, fault_matrix)
    greedy_record = {
        'method': greedy_res['method'],
        'num_selected_tests': greedy_res['num_selected_tests'],
        'total_duration_ms': round(greedy_res['total_duration_ms'], 2),
        'duration_savings_%': round(((base_duration - greedy_res['total_duration_ms']) / base_duration) * 100.0, 2),
        'branches_covered': greedy_res['branches_covered'],
        'coverage_percentage': greedy_res['coverage_percentage'],
        'apfd_score': round(greedy_apfd, 4),
        'optimization_runtime_ms': round(greedy_res['optimization_runtime_ms'], 2)
    }

    # 3. Exact ILP Minimization
    ilp_res = solve_ilp_test_minimization(tests, total_branches)
    ilp_order = prioritize_tests_greedy(ilp_res['selected_test_ids'], tests, fault_matrix)
    ilp_apfd = apfd_score(ilp_order, fault_matrix)
    ilp_record = {
        'method': ilp_res['method'],
        'num_selected_tests': ilp_res['num_selected_tests'],
        'total_duration_ms': round(ilp_res['total_duration_ms'], 2),
        'duration_savings_%': round(((base_duration - ilp_res['total_duration_ms']) / base_duration) * 100.0, 2),
        'branches_covered': ilp_res['branches_covered'],
        'coverage_percentage': ilp_res['coverage_percentage'],
        'apfd_score': round(ilp_apfd, 4),
        'optimization_runtime_ms': round(ilp_res['optimization_runtime_ms'], 2)
    }

    # Strict Correctness Assertion: 100% Branch Coverage Preserved!
    assert ilp_res['branches_covered'] == total_branches, "Correctness Failure: Branches missed!"
    print(f" -> Correctness Check PASSED: 100% branch coverage ({total_branches}/{total_branches}) strictly preserved!")
    print(f" -> Baseline Duration: {base_duration:.1f} ms | ILP Minimized Duration: {ilp_res['total_duration_ms']:.1f} ms ({ilp_record['duration_savings_%']}% savings)")
    print(f" -> APFD Fault Detection: Baseline {base_apfd:.3f} -> Prioritized {ilp_apfd:.3f}")

    summary_df = pd.DataFrame([baseline_record, greedy_record, ilp_record])
    summary_df.to_csv("tracks/track_j_software_engineering/results/test_optimization_summary.csv", index=False)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Track J: CI Test-Suite Minimization & Prioritization\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: CI Build & Test Duration Comparison
    ax = axes[0]
    methods = [r['method'].replace(' ', '\n') for r in [baseline_record, greedy_record, ilp_record]]
    durs = [r['total_duration_ms'] for r in [baseline_record, greedy_record, ilp_record]]
    bars = ax.bar(methods, durs, color=['#ef4444', '#f59e0b', '#10b981'], width=0.45)
    ax.set_ylabel("CI Test Execution Time (ms)")
    ax.set_title("Test Suite Execution Time Reduction")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval:.1f} ms", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: APFD Fault Detection Curves
    ax = axes[1]
    # Fault accumulation curve
    n_faults = fault_matrix.shape[1]
    
    def cumulative_faults(order):
        detected = set()
        curve = [0.0]
        for tid in order:
            for f in range(n_faults):
                if fault_matrix[tid, f] > 0:
                    detected.add(f)
            curve.append((len(detected) / n_faults) * 100.0)
        return curve

    base_curve = cumulative_faults(all_ids)
    ilp_curve = cumulative_faults(ilp_order)

    ax.plot(np.linspace(0, 100, len(base_curve)), base_curve, 'r--', label=f'Baseline Arbitrary (APFD={base_apfd:.2f})', lw=2)
    ax.plot(np.linspace(0, 100, len(ilp_curve)), ilp_curve, 'g-o', label=f'Optimized & Prioritized (APFD={ilp_apfd:.2f})', lw=2)
    ax.set_xlabel("% Tests Executed")
    ax.set_ylabel("% Faults Detected")
    ax.set_title("Fault Detection Velocity (APFD Curve)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_j_software_engineering/results/track_j_software_engineering_plots.png", dpi=300)
    plt.close()

    print("[Track J] Completed successfully. Artifacts saved in tracks/track_j_software_engineering/results/")

if __name__ == '__main__':
    main()
