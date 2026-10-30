"""
Track E Case Study Experiment Runner.
Executes the full Section 4 workflow for Sustainable Datacenter Optimization:
1. Baseline measurement
2. Method comparison (Naive vs Green Greedy vs Genetic Algorithm)
3. 24-hour load shifting & carbon abatement evaluation
4. Strategy-selection table generation
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_e_case_study.problem_formulation import DatacenterOptimizationProblem
from tracks.track_e_case_study.solvers import solve_baseline_naive, solve_green_greedy, solve_genetic_algorithm
from tracks.track_e_case_study.strategy_table import get_strategy_selection_table

def main():
    print("[Track E] Running Main Real-Process Case Study (Green Cloud Datacenter)...")
    os.makedirs("tracks/track_e_case_study/results", exist_ok=True)

    # 1. Strategy selection table
    strat_df = get_strategy_selection_table()
    strat_df.to_csv("tracks/track_e_case_study/results/strategy_selection_table.csv", index=False)

    # 2. Problem instance
    problem = DatacenterOptimizationProblem(num_jobs=45, num_servers=6, time_slots=24, seed=42)

    # 3. Execute methods
    naive_res = solve_baseline_naive(problem)
    greedy_res = solve_green_greedy(problem)
    ga_res = solve_genetic_algorithm(problem, pop_size=35, generations=40, seed=42)

    methods_res = [naive_res, greedy_res, ga_res]
    summary_rows = []
    for r in methods_res:
        summary_rows.append({
            'Method': r['method'],
            'Total_Cost_$': round(r['total_cost'], 2),
            'Energy_Cost_$': round(r['energy_cost'], 2),
            'Carbon_Emissions_kg': round(r['carbon_emissions_g'] / 1000.0, 2),
            'Carbon_Cost_$': round(r['carbon_cost'], 2),
            'SLA_Violations': r['sla_violations'],
            'Runtime_ms': round(r['runtime_ms'], 2),
            'Cost_Reduction_%': round(((naive_res['total_cost'] - r['total_cost']) / naive_res['total_cost']) * 100.0, 2),
            'Carbon_Abatement_%': round(((naive_res['carbon_emissions_g'] - r['carbon_emissions_g']) / naive_res['carbon_emissions_g']) * 100.0, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("tracks/track_e_case_study/results/case_study_summary.csv", index=False)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Track E: Green Datacenter Workload & Energy-Carbon Optimization\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: 24-Hour Grid Tariffs & Carbon Intensity
    ax = axes[0, 0]
    hours = np.arange(problem.T)
    ax.plot(hours, problem.tariffs, 'r-o', lw=2, label='Electricity Tariff ($/kWh)')
    ax.set_ylabel("Tariff ($/kWh)", color='r')
    ax.set_xlabel("Hour of Day")
    ax.set_title("24-Hour Dynamic Grid Conditions")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(hours, problem.carbon_intensity, 'g--s', lw=2, label='Carbon Intensity (gCO2/kWh)')
    ax2.set_ylabel("Carbon Intensity (gCO2/kWh)", color='g')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # Plot 2: Cost Comparison
    ax = axes[0, 1]
    names = [r['Method'] for r in summary_rows]
    costs = [r['Total_Cost_$'] for r in summary_rows]
    bars = ax.bar(names, costs, color=['#ef4444', '#10b981', '#3b82f6'], width=0.5)
    ax.set_ylabel("Total Operational Cost ($)")
    ax.set_title("Operational Cost Comparison (Baseline vs Optimized)")
    ax.set_xticklabels(names, rotation=12, ha='right', fontsize=9)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"${yval:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Carbon Emissions Comparison
    ax = axes[1, 0]
    carbons = [r['Carbon_Emissions_kg'] for r in summary_rows]
    bars = ax.bar(names, carbons, color=['#f97316', '#059669', '#6366f1'], width=0.5)
    ax.set_ylabel("Total Carbon Emissions (kg CO2)")
    ax.set_title("Carbon Emissions Abatement")
    ax.set_xticklabels(names, rotation=12, ha='right', fontsize=9)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} kg", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: GA Optimization Convergence
    ax = axes[1, 1]
    ax.plot(ga_res['history'], color='#2563eb', lw=2)
    ax.set_title("Genetic Algorithm Convergence Trajectory")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Total Cost ($)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_e_case_study/results/track_e_case_study_plots.png", dpi=300)
    plt.close()

    print("[Track E] Completed successfully. Artifacts saved in tracks/track_e_case_study/results/")

if __name__ == '__main__':
    main()
