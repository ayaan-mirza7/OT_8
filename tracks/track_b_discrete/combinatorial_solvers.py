"""
Combinatorial Solvers for Track B: Discrete Optimization.
Implements Greedy, Dynamic Programming, Branch-and-Bound, and Integer Linear Programming (ILP).
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np
from scipy.optimize import milp, LinearConstraint

class KnapsackProblem:
    """Standard 0-1 Knapsack problem representation."""
    def __init__(self, values: list[float], weights: list[float], capacity: float):
        self.values = np.asarray(values, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.capacity = float(capacity)
        self.n = len(values)

    def evaluate(self, selection: np.ndarray) -> tuple[float, float, bool]:
        tot_w = float(np.sum(selection * self.weights))
        tot_v = float(np.sum(selection * self.values))
        valid = tot_w <= self.capacity
        return tot_v, tot_w, valid


def solve_greedy_knapsack(problem: KnapsackProblem) -> dict:
    """Greedy heuristic based on value-to-weight ratio."""
    start_t = time.perf_counter()
    ratios = problem.values / np.maximum(problem.weights, 1e-9)
    sorted_indices = np.argsort(-ratios)

    selection = np.zeros(problem.n, dtype=int)
    current_w = 0.0
    current_v = 0.0

    for idx in sorted_indices:
        if current_w + problem.weights[idx] <= problem.capacity:
            selection[idx] = 1
            current_w += problem.weights[idx]
            current_v += problem.values[idx]

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Greedy',
        'value': current_v,
        'weight': current_w,
        'selection': selection.tolist(),
        'runtime_ms': runtime * 1000.0
    }


def solve_dp_knapsack(problem: KnapsackProblem) -> dict:
    """Exact Dynamic Programming solution (integer capacities)."""
    start_t = time.perf_counter()
    W = int(round(problem.capacity))
    weights_int = [int(round(w)) for w in problem.weights]
    values = problem.values
    n = problem.n

    dp = np.zeros((n + 1, W + 1), dtype=np.float64)

    for i in range(1, n + 1):
        w_i = weights_int[i - 1]
        v_i = values[i - 1]
        for w in range(W + 1):
            if w_i <= w:
                dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - w_i] + v_i)
            else:
                dp[i, w] = dp[i - 1, w]

    # Backtrack selection
    selection = np.zeros(n, dtype=int)
    w_curr = W
    for i in range(n, 0, -1):
        if dp[i, w_curr] != dp[i - 1, w_curr]:
            selection[i - 1] = 1
            w_curr -= weights_int[i - 1]

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Dynamic Programming',
        'value': float(dp[n, W]),
        'weight': float(np.sum(selection * problem.weights)),
        'selection': selection.tolist(),
        'runtime_ms': runtime * 1000.0
    }


def solve_branch_and_bound_knapsack(problem: KnapsackProblem) -> dict:
    """Exact combinatorial tree search with linear fractional relaxation upper bounds."""
    start_t = time.perf_counter()
    n = problem.n
    ratios = problem.values / np.maximum(problem.weights, 1e-9)
    order = np.argsort(-ratios)
    sorted_v = problem.values[order]
    sorted_w = problem.weights[order]

    best_val = 0.0
    best_sol = np.zeros(n, dtype=int)

    def compute_bound(level: int, current_w: float, current_v: float) -> float:
        if current_w > problem.capacity:
            return 0.0
        bound = current_v
        w_rem = problem.capacity - current_w
        for i in range(level, n):
            if sorted_w[i] <= w_rem:
                w_rem -= sorted_w[i]
                bound += sorted_v[i]
            else:
                bound += sorted_v[i] * (w_rem / sorted_w[i])
                break
        return bound

    class Node:
        def __init__(self, level: int, current_w: float, current_v: float, bound: float, sol: list[int]):
            self.level = level
            self.current_w = current_w
            self.current_v = current_v
            self.bound = bound
            self.sol = sol

    root = Node(0, 0.0, 0.0, compute_bound(0, 0.0, 0.0), [])
    stack = [root]

    while stack:
        node = stack.pop()
        if node.bound <= best_val:
            continue

        if node.level == n:
            if node.current_v > best_val:
                best_val = node.current_v
                best_sol_reordered = np.zeros(n, dtype=int)
                for idx, included in enumerate(node.sol):
                    best_sol_reordered[order[idx]] = included
                best_sol = best_sol_reordered
            continue

        # Option 1: Include item at node.level
        next_w = node.current_w + sorted_w[node.level]
        next_v = node.current_v + sorted_v[node.level]
        if next_w <= problem.capacity:
            b1 = compute_bound(node.level + 1, next_w, next_v)
            if b1 > best_val:
                stack.append(Node(node.level + 1, next_w, next_v, b1, node.sol + [1]))

        # Option 2: Exclude item at node.level
        b0 = compute_bound(node.level + 1, node.current_w, node.current_v)
        if b0 > best_val:
            stack.append(Node(node.level + 1, node.current_w, node.current_v, b0, node.sol + [0]))

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Branch and Bound',
        'value': best_val,
        'weight': float(np.sum(best_sol * problem.weights)),
        'selection': best_sol.tolist(),
        'runtime_ms': runtime * 1000.0
    }


def solve_ilp_knapsack(problem: KnapsackProblem) -> dict:
    """Exact Integer Linear Programming (ILP) using branch-and-cut simplex."""
    start_t = time.perf_counter()
    c = -problem.values  # minimize -values
    A = problem.weights.reshape(1, -1)
    constraints = LinearConstraint(A, -np.inf, problem.capacity)
    from scipy.optimize import Bounds
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(problem.n)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

    runtime = time.perf_counter() - start_t
    sel = np.round(res.x).astype(int)
    return {
        'method': 'Integer Linear Programming',
        'value': float(-res.fun),
        'weight': float(np.sum(sel * problem.weights)),
        'selection': sel.tolist(),
        'runtime_ms': runtime * 1000.0
    }
