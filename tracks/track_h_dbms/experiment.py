"""
Track H DBMS Optimization Experiment Runner.
Executes TPC-H workload optimization on SQLite:
1. Baseline query plans (full scans) and execution latencies
2. Index selection via Greedy vs Exact ILP Knapsack formulations
3. Post-optimization query plans (index seeks) and throughput measurement
4. Strict result set equivalence verification
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import time
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_h_dbms.schema_and_data import create_tpch_database
from tracks.track_h_dbms.index_optimizer import (
    TPCH_QUERIES, CANDIDATE_INDEXES,
    benchmark_workload, evaluate_index_benefits,
    solve_greedy_index_selection, solve_ilp_index_selection
)

def verify_query_results_identical(conn: sqlite3.Connection, baseline_results: dict[str, list]):
    """Strict correctness verification: asserts query outputs are byte-for-byte identical."""
    cur = conn.cursor()
    for q_name, sql in TPCH_QUERIES.items():
        cur.execute(sql)
        current_res = cur.fetchall()
        expected_res = baseline_results[q_name]
        assert current_res == expected_res, f"Correctness Violation: Query {q_name} results differ!"
    print(" -> Strict Correctness Check PASSED: All query results are 100% identical!")

def main():
    print("[Track H] Running DBMS Optimization Experiment Suite...")
    os.makedirs("tracks/track_h_dbms/results", exist_ok=True)

    # 1. Initialize SQLite Database
    conn = create_tpch_database(db_path=":memory:", num_customers=1000, num_orders=3000, seed=42)

    # 2. Capture baseline result sets for strict verification
    cur = conn.cursor()
    baseline_query_results = {}
    for q_name, sql in TPCH_QUERIES.items():
        cur.execute(sql)
        baseline_query_results[q_name] = cur.fetchall()

    # 3. Benchmark Baseline (No Secondary Indexes)
    base_total, base_q_times, base_plans = benchmark_workload(conn, repetitions=6)
    print(f" -> Baseline Total Workload Latency: {base_total:.2f} ms")

    # 4. Profile individual index benefits
    benefits = evaluate_index_benefits(conn)

    # 5. Formulate & Solve Index Selection under Storage Budget B = 200 KB
    budget_kb = 200.0
    greedy_res = solve_greedy_index_selection(benefits, storage_budget_kb=budget_kb)
    ilp_res = solve_ilp_index_selection(benefits, storage_budget_kb=budget_kb)

    # Compare Solvers Table
    solver_df = pd.DataFrame([greedy_res, ilp_res])
    solver_df.to_csv("tracks/track_h_dbms/results/index_selection_solvers.csv", index=False)

    # 6. Apply Optimal ILP Selected Indexes to Database
    for idx in ilp_res['selected_indices']:
        cur.execute(CANDIDATE_INDEXES[idx]['sql'])
    cur.execute("ANALYZE;")
    conn.commit()

    # 7. Benchmark Optimized Database
    opt_total, opt_q_times, opt_plans = benchmark_workload(conn, repetitions=6)
    print(f" -> Optimized Total Workload Latency: {opt_total:.2f} ms")
    speedup = base_total / max(opt_total, 1e-6)
    print(f" -> Speedup: {speedup:.2f}x ({((base_total - opt_total) / base_total)*100.0:.1f}% latency reduction)")

    # 8. Verify Result Equality
    verify_query_results_identical(conn, baseline_query_results)

    # Save Query-by-Query Comparison Table
    query_records = []
    for q_name in TPCH_QUERIES.keys():
        query_records.append({
            'Query': q_name,
            'Baseline_Latency_ms': round(base_q_times[q_name], 3),
            'Optimized_Latency_ms': round(opt_q_times[q_name], 3),
            'Speedup': round(base_q_times[q_name] / max(opt_q_times[q_name], 1e-6), 2),
            'Baseline_Plan': " | ".join(base_plans[q_name]),
            'Optimized_Plan': " | ".join(opt_plans[q_name])
        })
    query_df = pd.DataFrame(query_records)
    query_df.to_csv("tracks/track_h_dbms/results/query_latency_and_plans.csv", index=False)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Track H: DBMS Workload Optimization & Query Plans\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: Query Latencies Before vs After
    ax = axes[0]
    q_names = list(TPCH_QUERIES.keys())
    x_pos = np.arange(len(q_names))
    b_lat = [base_q_times[q] for q in q_names]
    o_lat = [opt_q_times[q] for q in q_names]

    ax.bar(x_pos - 0.2, b_lat, width=0.4, label='Baseline (Table Scans)', color='#ef4444')
    ax.bar(x_pos + 0.2, o_lat, width=0.4, label='Optimized (Index Seeks)', color='#10b981')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Query Latency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Total Workload Latency & Speedup
    ax = axes[1]
    cats = ['Baseline', 'Optimized']
    totals = [base_total, opt_total]
    bars = ax.bar(cats, totals, color=['#dc2626', '#059669'], width=0.4)
    ax.set_ylabel("Total Workload Latency (ms)")
    ax.set_title(f"Total Workload Speedup: {speedup:.2f}x")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f} ms", ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_h_dbms/results/track_h_dbms_plots.png", dpi=300)
    plt.close()

    print("[Track H] Completed successfully. Artifacts saved in tracks/track_h_dbms/results/")

if __name__ == '__main__':
    main()
