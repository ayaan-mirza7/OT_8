"""
Chapter 6 Strategy-Selection Matrix for Track E Case Study.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import pandas as pd

def get_strategy_selection_table() -> pd.DataFrame:
    """
    Constructs the required Chapter 6 strategy-selection table.
    Categorizes optimization methods by:
    - Variable Type
    - Problem Size / Complexity
    - Exact vs Heuristic
    - Target Hardware
    - Parallelism Potential
    - Convergence Guarantee
    - Communication Overhead
    """
    data = [
        {
            'Algorithm': 'Baseline Naive First-Fit',
            'Variable_Type': 'Discrete (Assignment, Time)',
            'Problem_Size': 'O(N) Real-Time',
            'Approach': 'Heuristic (Myopic)',
            'Hardware': 'Single-Core CPU / Microcontroller',
            'Parallelism': 'Sequential per job arrival',
            'Convergence': 'None (Immediate greedy assignment)',
            'Suitability': 'Poor: Ignores carbon & tariff spikes'
        },
        {
            'Algorithm': 'Mixed Integer Linear Programming (MILP)',
            'Variable_Type': 'Mixed (Binary x_{ijt}, Continuous f_{jt})',
            'Problem_Size': 'Small to Medium (N <= 100, T <= 24)',
            'Approach': 'Exact (Branch-and-Cut Simplex)',
            'Hardware': 'High-Memory Multicore Server',
            'Parallelism': 'Parallel branch-and-bound tree search',
            'Convergence': 'Global Optimum with verifiable duality gap',
            'Suitability': 'Strong for offline planning; exponential worst-case'
        },
        {
            'Algorithm': 'Green Carbon-Aware Greedy',
            'Variable_Type': 'Discrete (Server ID, Window Start)',
            'Problem_Size': 'Large (N >= 10,000)',
            'Approach': 'Heuristic (Marginal Cost Search)',
            'Hardware': 'Edge Gateway / Standard CPU',
            'Parallelism': 'Embarrassingly parallel across jobs',
            'Convergence': 'Local myopic minimum; fast polynomial runtime',
            'Suitability': 'High: Exceptional speed-quality trade-off for streaming'
        },
        {
            'Algorithm': 'Genetic Algorithm (GA / Metaheuristic)',
            'Variable_Type': 'Discrete / Permutation Chromosomes',
            'Problem_Size': 'Medium to Large (NP-hard combinatorial)',
            'Approach': 'Stochastic Population Heuristic',
            'Hardware': 'Multi-core CPU / GPU Vectorized',
            'Parallelism': 'Embarrassingly parallel fitness evaluations',
            'Convergence': 'Asymptotic empirical convergence to near-optima',
            'Suitability': 'High: Handles non-linear cubic DVFS power curves'
        },
        {
            'Algorithm': 'Quantum QAOA (Track C)',
            'Variable_Type': 'Binary / Spin (Ising QUBO)',
            'Problem_Size': 'NISQ Limit (N <= 50 qubits currently)',
            'Approach': 'Hybrid Quantum-Classical Heuristic',
            'Hardware': 'Superconducting / Trapped-ion QPU',
            'Parallelism': 'Quantum state superposition 2^N',
            'Convergence': 'Bound by circuit depth p, noise & barren plateaus',
            'Suitability': 'Emerging research; classical heuristics dominate in wall-clock time'
        }
    ]
    return pd.DataFrame(data)
