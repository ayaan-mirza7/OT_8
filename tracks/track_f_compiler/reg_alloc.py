"""
Register Allocation Formulated as an Optimization Problem (Graph Colouring).
Implements:
1. Liveness Analysis & Variable Interference Graph Construction
2. Chapter 6 Greedy Kempe Graph Colouring Heuristic (Chaitin-Briggs)
3. Exact 0-1 Integer Linear Programming (ILP) formulation minimizing spill cost
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from tracks.track_f_compiler.ir import Instruction, is_constant

class LivenessAnalyzer:
    """Computes live ranges and variable interference edges."""
    def __init__(self, instructions: list[Instruction]):
        self.instructions = instructions
        self.variables = self._extract_variables()
        self.interference_graph: dict[str, set[str]] = {v: set() for v in self.variables}
        self.spill_costs: dict[str, float] = {v: 1.0 for v in self.variables}
        self._build_interference()

    def _extract_variables(self) -> list[str]:
        vars_found = set()
        for inst in self.instructions:
            if inst.dest and not is_constant(inst.dest) and inst.op != 'LABEL':
                vars_found.add(inst.dest)
            if inst.arg1 and not is_constant(inst.arg1):
                vars_found.add(inst.arg1)
            if inst.arg2 and not is_constant(inst.arg2):
                vars_found.add(inst.arg2)
        return sorted(list(vars_found))

    def _build_interference(self):
        # Backward sweep to compute live variables at each instruction
        live_now = set()
        for inst in reversed(self.instructions):
            if inst.dest and inst.dest in self.variables:
                # Dest interferes with all currently live variables
                for live_v in live_now:
                    if live_v != inst.dest:
                        self.interference_graph[inst.dest].add(live_v)
                        self.interference_graph[live_v].add(inst.dest)
                # Def kills variable
                live_now.discard(inst.dest)

            # Uses generate liveness
            if inst.arg1 and inst.arg1 in self.variables:
                live_now.add(inst.arg1)
                self.spill_costs[inst.arg1] += 2.0  # frequently used variables cost more to spill
            if inst.arg2 and inst.arg2 in self.variables:
                live_now.add(inst.arg2)
                self.spill_costs[inst.arg2] += 2.0


def solve_greedy_register_allocation(analyzer: LivenessAnalyzer, k_registers: int = 3) -> dict:
    """Greedy Kempe / Chaitin-Briggs heuristic for graph coloring."""
    start_t = time.perf_counter()
    graph = {v: set(neighbors) for v, neighbors in analyzer.interference_graph.items()}
    variables = sorted(list(graph.keys()), key=lambda v: len(graph[v]), reverse=True)

    color_assignment = {}
    spilled = []

    for var in variables:
        # Find forbidden colors from already colored neighbors
        forbidden_colors = set()
        for neighbor in graph[var]:
            if neighbor in color_assignment:
                forbidden_colors.add(color_assignment[neighbor])

        # Assign lowest available color in [0, k_registers - 1]
        assigned = None
        for c in range(k_registers):
            if c not in forbidden_colors:
                assigned = c
                break

        if assigned is not None:
            color_assignment[var] = assigned
        else:
            spilled.append(var)

    runtime = time.perf_counter() - start_t
    tot_spill_cost = sum(analyzer.spill_costs[v] for v in spilled)
    return {
        'method': 'Greedy Chaitin-Briggs',
        'k_registers': k_registers,
        'allocated_registers': color_assignment,
        'spilled_variables': spilled,
        'num_spills': len(spilled),
        'total_spill_cost': tot_spill_cost,
        'runtime_ms': runtime * 1000.0
    }


def solve_ilp_register_allocation(analyzer: LivenessAnalyzer, k_registers: int = 3) -> dict:
    """
    Exact Integer Linear Programming (ILP) formulation:
    Variables:
      x[v, c] in {0, 1}: variable v colored with register c
      s[v] in {0, 1}: variable v spilled to stack
    Minimize sum_v SpillCost(v) * s[v]
    Subject to:
      1. sum_c x[v, c] + s[v] = 1 for all v
      2. x[u, c] + x[v, c] <= 1 for all (u, v) in E, for all c
    """
    start_t = time.perf_counter()
    vars_list = analyzer.variables
    n_vars = len(vars_list)
    var_to_idx = {v: i for i, v in enumerate(vars_list)}

    # Total decision variables: n_vars * k_registers + n_vars (spills)
    total_vars = n_vars * k_registers + n_vars

    def x_idx(v_idx, c):
        return v_idx * k_registers + c

    def s_idx(v_idx):
        return n_vars * k_registers + v_idx

    # Cost vector c_obj
    c_obj = np.zeros(total_vars)
    for i, v in enumerate(vars_list):
        c_obj[s_idx(i)] = analyzer.spill_costs[v]

    # Constraints
    A_rows = []
    b_l = []
    b_u = []

    # 1. Assignment constraint: sum_c x[v, c] + s[v] == 1
    for i in range(n_vars):
        row = np.zeros(total_vars)
        for c in range(k_registers):
            row[x_idx(i, c)] = 1.0
        row[s_idx(i)] = 1.0
        A_rows.append(row)
        b_l.append(1.0)
        b_u.append(1.0)

    # 2. Interference constraint: for each edge (u, v) in E, and each c: x[u, c] + x[v, c] <= 1
    seen_edges = set()
    for u, neighbors in analyzer.interference_graph.items():
        u_idx = var_to_idx[u]
        for v in neighbors:
            v_idx = var_to_idx[v]
            edge = tuple(sorted((u_idx, v_idx)))
            if edge not in seen_edges:
                seen_edges.add(edge)
                for c in range(k_registers):
                    row = np.zeros(total_vars)
                    row[x_idx(u_idx, c)] = 1.0
                    row[x_idx(v_idx, c)] = 1.0
                    A_rows.append(row)
                    b_l.append(-np.inf)
                    b_u.append(1.0)

    A_mat = np.array(A_rows)
    constraints = LinearConstraint(A_mat, b_l, b_u)
    integrality = np.ones(total_vars)
    bounds = Bounds(lb=0.0, ub=1.0)

    res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
    runtime = time.perf_counter() - start_t

    sol = np.round(res.x).astype(int)
    color_assignment = {}
    spilled = []

    for i, v in enumerate(vars_list):
        if sol[s_idx(i)] == 1:
            spilled.append(v)
        else:
            for c in range(k_registers):
                if sol[x_idx(i, c)] == 1:
                    color_assignment[v] = c
                    break

    return {
        'method': 'Exact ILP Graph Colouring',
        'k_registers': k_registers,
        'allocated_registers': color_assignment,
        'spilled_variables': spilled,
        'num_spills': len(spilled),
        'total_spill_cost': float(res.fun),
        'runtime_ms': runtime * 1000.0
    }
