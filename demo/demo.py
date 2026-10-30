"""
Interactive Demonstration CLI for Optimization Techniques Assignment 8.
Student: Ayaan Mirza (Roll No: 241070043)
Allows quick interactive inspection and verification of any track from A to J.
"""

import sys
import time
import os

# Ensure workspace root is in sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

def print_header():
    print("=" * 76)
    print("   OPTIMIZATION TECHNIQUES — ASSIGNMENT 8: VIBE CODING CHALLENGE")
    print("   Student Name: Ayaan Mirza  |  Roll No: 241070043")
    print("=" * 76)

def run_track(choice: str):
    c = choice.upper().strip()
    if c == 'A':
        from tracks.track_a_continuous.experiment import main as run_a
        run_a()
    elif c == 'B':
        from tracks.track_b_discrete.experiment import main as run_b
        run_b()
    elif c == 'C':
        from tracks.track_c_quantum.experiment import main as run_c
        run_c()
    elif c == 'D':
        from tracks.track_d_parallel_distributed.experiment import main as run_d
        run_d()
    elif c == 'E':
        from tracks.track_e_case_study.experiment import main as run_e
        run_e()
    elif c == 'F':
        from tracks.track_f_compiler.experiment import main as run_f
        run_f()
    elif c == 'G':
        from tracks.track_g_xv6.experiment import main as run_g
        run_g()
    elif c == 'H':
        from tracks.track_h_dbms.experiment import main as run_h
        run_h()
    elif c == 'I':
        from tracks.track_i_quantum_advantage.experiment import main as run_i
        run_i()
    elif c == 'J':
        from tracks.track_j_software_engineering.experiment import main as run_j
        run_j()
    elif c == 'ALL':
        tracks = [('A', 'tracks.track_a_continuous.experiment'),
                  ('B', 'tracks.track_b_discrete.experiment'),
                  ('C', 'tracks.track_c_quantum.experiment'),
                  ('D', 'tracks.track_d_parallel_distributed.experiment'),
                  ('E', 'tracks.track_e_case_study.experiment'),
                  ('F', 'tracks.track_f_compiler.experiment'),
                  ('G', 'tracks.track_g_xv6.experiment'),
                  ('H', 'tracks.track_h_dbms.experiment'),
                  ('I', 'tracks.track_i_quantum_advantage.experiment'),
                  ('J', 'tracks.track_j_software_engineering.experiment')]
        for name, mod in tracks:
            print(f"\n>>> Executing Track {name} Benchmark...")
            __import__(mod, fromlist=['main']).main()
    else:
        print(f"Unknown track selection '{choice}'. Enter A-J or ALL.")

def main():
    print_header()
    if len(sys.argv) > 1:
        run_track(sys.argv[1])
        return

    print("\nSelect an Experiment Track to run:")
    print("  [A] Track A: Continuous Optimization (GD, Momentum, Adam, Newton, MLP Backprop)")
    print("  [B] Track B: Discrete Optimization (Knapsack, B&B, DP, ILP, Max-Cut QUBO/Ising)")
    print("  [C] Track C: Quantum Optimization (QAOA, VQE, Parameter-Shift, SPSA, Barren Plateaus)")
    print("  [D] Track D: Submodular, Parallel & Distributed (Facility Coverage, Amdahl, FedAvg, MGDA)")
    print("  [E] Track E: Main Case Study (Green Cloud Datacenter Energy-Carbon Optimization)")
    print("  [F] Track F: Compiler Optimization (Constant Folding, CSE, DCE, LICM, ILP RegAlloc)")
    print("  [G] Track G: xv6 OS Schedulers (MLFQ, Stride, Round-Robin, Priority across 3 Workloads)")
    print("  [H] Track H: DBMS Optimization (TPC-H Workload Index Knapsack & Query Plan Seeks)")
    print("  [I] Track I: Quantum Algorithms & QML (7 Algorithms, VQC vs MLP, Grover Advantage)")
    print("  [J] Track J: Software-Engineering (1,000+ LOC Codebase Test Minimization & Prioritization)")
    print("  [ALL] Run All 10 Tracks in sequence")
    print("  [Q] Quit")

    try:
        choice = input("\nEnter choice [A-J, ALL, Q]: ").strip()
        if choice.upper() != 'Q' and choice != '':
            run_track(choice)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting demonstration.")

if __name__ == '__main__':
    main()
