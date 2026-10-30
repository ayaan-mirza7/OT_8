"""
Track G xv6 Operating-System Optimization Experiment Runner.
Benchmarks Default Round-Robin vs Priority vs MLFQ vs Stride schedulers
across 3 distinct workloads under cycle-accurate simulation.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_g_xv6.emulator import Process, run_simulation

def generate_workload_1_cpu_bound() -> list[Process]:
    """Workload 1: Compute-bound processes with staggered arrivals."""
    return [
        Process(pid=1, name="cpu_calc_1", arrival_time=0, total_cpu_needed=40, io_frequency=0, priority=5, tickets=10),
        Process(pid=2, name="cpu_calc_2", arrival_time=2, total_cpu_needed=35, io_frequency=0, priority=7, tickets=20),
        Process(pid=3, name="cpu_calc_3", arrival_time=4, total_cpu_needed=50, io_frequency=0, priority=3, tickets=10),
        Process(pid=4, name="cpu_calc_4", arrival_time=6, total_cpu_needed=30, io_frequency=0, priority=6, tickets=15),
        Process(pid=5, name="cpu_calc_5", arrival_time=8, total_cpu_needed=45, io_frequency=0, priority=8, tickets=25),
        Process(pid=6, name="cpu_calc_6", arrival_time=10, total_cpu_needed=25, io_frequency=0, priority=4, tickets=10),
    ]

def generate_workload_2_interactive() -> list[Process]:
    """Workload 2: Latency-sensitive I/O bursty processes."""
    return [
        Process(pid=1, name="shell_ui", arrival_time=0, total_cpu_needed=20, io_frequency=2, io_wait_time=3, priority=9, tickets=20),
        Process(pid=2, name="editor", arrival_time=1, total_cpu_needed=18, io_frequency=2, io_wait_time=2, priority=8, tickets=20),
        Process(pid=3, name="net_daemon", arrival_time=3, total_cpu_needed=25, io_frequency=3, io_wait_time=4, priority=8, tickets=20),
        Process(pid=4, name="terminal", arrival_time=5, total_cpu_needed=15, io_frequency=2, io_wait_time=2, priority=9, tickets=30),
        Process(pid=5, name="key_listener", arrival_time=7, total_cpu_needed=12, io_frequency=1, io_wait_time=2, priority=10, tickets=30),
        Process(pid=6, name="logger", arrival_time=8, total_cpu_needed=22, io_frequency=4, io_wait_time=3, priority=7, tickets=15),
    ]

def generate_workload_3_mixed() -> list[Process]:
    """Workload 3: Mixed compute & interactive (evaluating starvation & interactive response)."""
    return [
        # Heavy CPU hogs
        Process(pid=1, name="heavy_matrix", arrival_time=0, total_cpu_needed=70, io_frequency=0, priority=3, tickets=10),
        Process(pid=2, name="crypto_hash", arrival_time=0, total_cpu_needed=80, io_frequency=0, priority=3, tickets=10),
        # Burst interactive tasks
        Process(pid=3, name="gui_refresh", arrival_time=5, total_cpu_needed=14, io_frequency=2, io_wait_time=2, priority=9, tickets=30),
        Process(pid=4, name="user_keystroke", arrival_time=15, total_cpu_needed=8, io_frequency=1, io_wait_time=1, priority=10, tickets=40),
        Process(pid=5, name="network_ping", arrival_time=25, total_cpu_needed=10, io_frequency=2, io_wait_time=2, priority=8, tickets=25),
        Process(pid=6, name="disk_sync", arrival_time=35, total_cpu_needed=16, io_frequency=3, io_wait_time=3, priority=7, tickets=20),
    ]

def main():
    print("[Track G] Running xv6 OS Optimization Benchmark Suite...")
    os.makedirs("tracks/track_g_xv6/results", exist_ok=True)

    workloads = [
        ("Workload 1 (CPU-Bound)", generate_workload_1_cpu_bound()),
        ("Workload 2 (Interactive I/O)", generate_workload_2_interactive()),
        ("Workload 3 (Mixed Starvation Risk)", generate_workload_3_mixed())
    ]
    schedulers = ["Round_Robin", "Priority", "MLFQ", "Stride"]

    all_records = []
    for wl_name, wl_procs in workloads:
        for sched in schedulers:
            res = run_simulation(sched, wl_procs)
            res['workload'] = wl_name
            all_records.append(res)

    results_df = pd.DataFrame(all_records)
    results_df.to_csv("tracks/track_g_xv6/results/xv6_scheduler_benchmarks.csv", index=False)

    # Focus analysis: Mixed workload response time & turnaround time
    mixed_df = results_df[results_df['workload'] == "Workload 3 (Mixed Starvation Risk)"]
    print(" -> Mixed Workload Scheduler Comparison:")
    for _, row in mixed_df.iterrows():
        print(f"    [{row['scheduler']}] Avg Turnaround: {row['avg_turnaround_time']:.1f}t, Avg Response: {row['avg_response_time']:.1f}t, Fairness: {row['jains_fairness']:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Track G: xv6 OS Scheduler Optimization Benchmarks\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: Mixed Workload Response Time vs Turnaround Time
    ax = axes[0]
    x_pos = np.arange(len(schedulers))
    sched_names = [s.replace('_', ' ') for s in schedulers]
    resp_times = mixed_df['avg_response_time'].tolist()
    turn_times = mixed_df['avg_turnaround_time'].tolist()

    ax.bar(x_pos - 0.2, resp_times, width=0.4, label='Avg Response Time (ticks)', color='#3b82f6')
    ax.bar(x_pos + 0.2, turn_times, width=0.4, label='Avg Turnaround Time (ticks)', color='#ef4444')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sched_names)
    ax.set_ylabel("Ticks")
    ax.set_title("Mixed Workload: Response vs Turnaround Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Jain's Fairness Index across Schedulers
    ax = axes[1]
    wl_names = [w[0] for w in workloads]
    for sched, col in zip(schedulers, ['#64748b', '#f59e0b', '#10b981', '#8b5cf6']):
        sub = results_df[results_df['scheduler'] == sched]
        ax.plot(wl_names, sub['jains_fairness'], 'o-', lw=2, label=sched.replace('_', ' '), color=col)
    ax.set_ylabel("Jain's Fairness Index (0 to 1)")
    ax.set_title("Scheduler Fairness Comparison")
    ax.set_xticklabels(wl_names, rotation=10, ha='right', fontsize=8)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_g_xv6/results/track_g_xv6_plots.png", dpi=300)
    plt.close()

    print("[Track G] Completed successfully. Artifacts saved in tracks/track_g_xv6/results/")

if __name__ == '__main__':
    main()
