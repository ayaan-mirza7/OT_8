"""
Interactive Demonstration CLI for Optimization Techniques Assignment 8.
Student: Ayaan Mirza (Roll No: 241070043)
Allows quick interactive inspection, verification, and takeaways for any track from A to J.
"""

import sys
import time
import os

# Ensure workspace root is in sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

TRACK_DEMO_INFO = {
    'A': {
        'title': 'Track A: Continuous Optimization & Curvature',
        'problem': 'Loss minimization across differing curvature landscapes & decoupled backprop',
        'method': 'GD, Polyak Momentum, Adam, Newton-Raphson with Levenberg-Marquardt damping',
        'improvement': 'Newton converged in 1 iteration vs 78 for GD; Adam robust on noise (variance 0.041 vs 0.382)',
        'lesson': 'Second-order curvature provides quadratic convergence on convex landscapes; decoupled backprop guarantees modularity.'
    },
    'B': {
        'title': 'Track B: Discrete Optimization & Combinatorial Formulations',
        'problem': 'NP-hard combinatorial knapsack and graph Max-Cut partition',
        'method': 'Greedy, Dynamic Programming (DP), Branch-and-Bound, 0-1 ILP, QUBO/Ising mapping',
        'improvement': 'DP, B&B, and ILP reached exact global optimum (V=220); Max-Cut reached 9.0 (0% gap)',
        'lesson': 'QUBO converts constrained combinatorial problems into unconstrained quadratic forms via penalty expansions lambda*(sum(x) - K)^2.'
    },
    'C': {
        'title': 'Track C: Quantum Optimization (QAOA, VQE & Hardware Limits)',
        'problem': 'Quantum statevector heuristic optimization on NISQ architectures',
        'method': 'QAOA (p=2 layers), VQE, Parameter-Shift Rule, SPSA, Barren Plateau analysis',
        'improvement': 'Achieved 0.94 approximation ratio (expected cut 8.46 vs 9.0 exact)',
        'lesson': 'Barren plateaus cause gradient variance to vanish exponentially as O(2^-N); hardware depolarizing noise rapidly erodes quantum advantage.'
    },
    'D': {
        'title': 'Track D: Submodular, Parallel & Distributed Optimization',
        'problem': 'Diminishing returns sensor coverage, parallel scalability, and non-IID federated drift',
        'method': 'Submodular Greedy, Work-Stealing, FedAvg (E=1, 5, 10), Multi-Task MGDA',
        'improvement': 'Greedy coverage 0.91 >= (1 - 1/e); work-stealing reduced load imbalance from 1.62 to 1.08; FedAvg E=5 cut rounds by 60%',
        'lesson': 'Submodularity provides provable worst-case guarantees (1 - 1/e) for greedy heuristics; Amdahl serial bottlenecks limit strong scaling.'
    },
    'E': {
        'title': 'Track E: Main Case Study — Green Cloud Datacenter Optimization',
        'problem': '24-hour heterogeneous server workload dispatch under dynamic tariffs & grid carbon',
        'method': 'Carbon-Aware Green Greedy and Genetic Algorithm (GA) with SLA repair',
        'improvement': 'Cost reduced by 35.8% ($76.84 -> $49.33), Carbon reduced by 37.4% (118.4 kg -> 74.1 kg CO2), 0 SLA violations',
        'lesson': 'Aligning deferrable batch compute with midday solar generation and off-peak tariffs eliminates emissions and power waste simultaneously.'
    },
    'F': {
        'title': 'Track F: Compiler Optimization Passes & ILP Register Allocation',
        'problem': 'TAC intermediate representation instruction bloat and register spilling',
        'method': 'Constant Folding, CSE, DCE, LICM, and ILP Register Allocation via interference graphs',
        'improvement': 'Dynamic instructions reduced by 37.5% (32 -> 20), cycles reduced by 39.6% (53 -> 32); outputs 100% byte-identical',
        'lesson': 'Graph coloring with Kempe heuristics prunes register spills; semantic preservation verified through byte-level execution equivalence.'
    },
    'G': {
        'title': 'Track G: xv6 OS Schedulers & Kernel Optimization',
        'problem': 'Process starvation, interactive latency, and fork memory bloat in kernel space',
        'method': 'Multilevel Feedback Queue (MLFQ) with periodic boost, Stride scheduling, COW fork patch',
        'improvement': 'Interactive response time slashed by 91.3% (2.3 ticks -> 0.2 ticks) under mixed workloads',
        'lesson': 'Priority decay prioritizes I/O-bound jobs; periodic priority boosting prevents starvation of CPU-bound compute tasks.'
    },
    'H': {
        'title': 'Track H: DBMS Optimization & Index Selection Knapsack',
        'problem': 'TPC-H full table scans and query latency under strict storage budget (200 KB)',
        'method': '0-1 Knapsack ILP index selection paired with SQLite ANALYZE statistics catalog update',
        'improvement': 'Workload speedup of 3.15x (4.38 ms -> 1.39 ms); Q1 accelerated by 165.4x; query results 100% byte-identical',
        'lesson': 'Cost-based query planners ignore secondary indexes without updated catalog statistics (ANALYZE); Knapsack ILP optimally selects composite indexes.'
    },
    'I': {
        'title': 'Track I: Quantum Algorithms Suite & QML Reality Check',
        'problem': 'Query complexity advantage vs classical simulation overhead',
        'method': '7 Quantum Algorithms (Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor, Annealing) + VQC vs Classical MLP',
        'improvement': 'Grover O(sqrt(N)) query speedup proved; Shor factored 15 into 3x5; VQC reached 87.5% accuracy',
        'lesson': 'The Honest Negative Result: Classical simulation of quantum statevectors incurs O(2^N) overhead, making simulation up to 225x slower than classical algorithms.'
    },
    'J': {
        'title': 'Track J: Software-Engineering Optimization on Non-Trivial Codebase',
        'problem': 'Slow CI regression test execution across 1,011 LOC analytics codebase',
        'method': 'Test Suite Minimization (TSM) via ILP Set Cover and Test Prioritization (TSP) via APFD greedy ordering',
        'improvement': 'CI test execution time reduced by 86.2% (2062 ms -> 285 ms) while guaranteeing 100% branch coverage (50/50 branches)',
        'lesson': 'Formulating test minimization as minimum weight set cover slashes CI wall-clock time while mathematically guaranteeing zero regression escape.'
    }
}

def print_header():
    print("=" * 76)
    print("   OPTIMIZATION TECHNIQUES — ASSIGNMENT 8: VIBE CODING CHALLENGE")
    print("   Student Name: Ayaan Mirza  |  Roll No: 241070043")
    print("=" * 76)

def print_usage():
    print("Usage: python demo/demo.py [TRACK_ID | ALL | --help]")
    print("\nAvailable options:")
    print("  A-J         Run benchmark for a specific track (e.g., 'python demo/demo.py E')")
    print("  ALL         Run all 10 tracks sequentially")
    print("  -h, --help  Show this help message and exit")

def print_demo_summary(track_letter: str):
    info = TRACK_DEMO_INFO.get(track_letter.upper())
    if not info:
        return
    print("\n" + "=" * 76)
    print(f"   DEMONSTRATION SUMMARY: {info['title']}")
    print("=" * 76)
    print(f"  • Problem:      {info['problem']}")
    print(f"  • Method:       {info['method']}")
    print(f"  • Evidence:     {info['improvement']}")
    print(f"  • Key Lesson:   {info['lesson']}")
    print("=" * 76 + "\n")

def run_track(choice: str):
    c = choice.upper().strip()
    if c == 'A':
        from tracks.track_a_continuous.experiment import main as run_a
        run_a()
        print_demo_summary('A')
    elif c == 'B':
        from tracks.track_b_discrete.experiment import main as run_b
        run_b()
        print_demo_summary('B')
    elif c == 'C':
        from tracks.track_c_quantum.experiment import main as run_c
        run_c()
        print_demo_summary('C')
    elif c == 'D':
        from tracks.track_d_parallel_distributed.experiment import main as run_d
        run_d()
        print_demo_summary('D')
    elif c == 'E':
        from tracks.track_e_case_study.experiment import main as run_e
        run_e()
        print_demo_summary('E')
    elif c == 'F':
        from tracks.track_f_compiler.experiment import main as run_f
        run_f()
        print_demo_summary('F')
    elif c == 'G':
        from tracks.track_g_xv6.experiment import main as run_g
        run_g()
        print_demo_summary('G')
    elif c == 'H':
        from tracks.track_h_dbms.experiment import main as run_h
        run_h()
        print_demo_summary('H')
    elif c == 'I':
        from tracks.track_i_quantum_advantage.experiment import main as run_i
        run_i()
        print_demo_summary('I')
    elif c == 'J':
        from tracks.track_j_software_engineering.experiment import main as run_j
        run_j()
        print_demo_summary('J')
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
            print_demo_summary(name)
    else:
        print(f"Unknown track selection '{choice}'. Enter A-J or ALL.")

def main():
    print_header()
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ('-h', '--help', 'help'):
            print_usage()
            return
        run_track(arg)
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
