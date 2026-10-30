"""
DBMS Index Selection Formulated as 0-1 Knapsack / ILP Optimization.
Implements:
1. Workload Profiling & Latency Benefit Calculation
2. Greedy Knapsack Selection (Benefit-to-Storage Ratio)
3. Exact 0-1 Integer Linear Programming (ILP) under Storage Budget B
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import sqlite3
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Workload Queries with selective filters benefiting heavily from secondary indexes
TPCH_QUERIES = {
    'Q1': """
        SELECT l_orderkey, l_extendedprice, l_discount
        FROM lineitem
        WHERE l_shipdate = '2024-06-15' AND l_discount <= 0.05
    """,
    'Q2': """
        SELECT c.c_name, o.o_orderdate, l.l_extendedprice
        FROM orders o
        JOIN lineitem l ON o.o_orderkey = l.l_orderkey
        JOIN customer c ON o.o_custkey = c.c_custkey
        WHERE o.o_orderdate = '2024-05-10'
    """,
    'Q3': """
        SELECT count(*), sum(l_extendedprice)
        FROM lineitem
        WHERE l_partkey = 142
    """,
    'Q4': """
        SELECT o_orderkey, o_orderdate, o_totalprice
        FROM orders
        WHERE o_custkey = 250
    """
}

# Candidate Indexes with SQL definition and estimated storage footprint (KB)
CANDIDATE_INDEXES = [
    {
        'id': 'idx_lineitem_ship_disc',
        'sql': "CREATE INDEX idx_lineitem_ship_disc ON lineitem(l_shipdate, l_discount);",
        'drop_sql': "DROP INDEX IF EXISTS idx_lineitem_ship_disc;",
        'storage_kb': 75.0,
        'targeted_queries': ['Q1']
    },
    {
        'id': 'idx_orders_date',
        'sql': "CREATE INDEX idx_orders_date ON orders(o_orderdate);",
        'drop_sql': "DROP INDEX IF EXISTS idx_orders_date;",
        'storage_kb': 50.0,
        'targeted_queries': ['Q2']
    },
    {
        'id': 'idx_lineitem_orderkey',
        'sql': "CREATE INDEX idx_lineitem_orderkey ON lineitem(l_orderkey);",
        'drop_sql': "DROP INDEX IF EXISTS idx_lineitem_orderkey;",
        'storage_kb': 80.0,
        'targeted_queries': ['Q2']
    },
    {
        'id': 'idx_lineitem_partkey',
        'sql': "CREATE INDEX idx_lineitem_partkey ON lineitem(l_partkey);",
        'drop_sql': "DROP INDEX IF EXISTS idx_lineitem_partkey;",
        'storage_kb': 60.0,
        'targeted_queries': ['Q3']
    },
    {
        'id': 'idx_orders_custkey',
        'sql': "CREATE INDEX idx_orders_custkey ON orders(o_custkey);",
        'drop_sql': "DROP INDEX IF EXISTS idx_orders_custkey;",
        'storage_kb': 45.0,
        'targeted_queries': ['Q4']
    }
]

def benchmark_workload(conn: sqlite3.Connection, repetitions: int = 15) -> tuple[float, dict[str, float], dict[str, list]]:
    """Measures total execution latency and captures EXPLAIN QUERY PLAN."""
    cur = conn.cursor()
    total_time = 0.0
    query_times = {}
    query_plans = {}

    for q_name, sql in TPCH_QUERIES.items():
        # Get query plan
        plan_cur = cur.execute(f"EXPLAIN QUERY PLAN {sql}")
        query_plans[q_name] = [row[3] for row in plan_cur.fetchall()]

        # Measure runtime
        start_t = time.perf_counter()
        for _ in range(repetitions):
            cur.execute(sql)
            _ = cur.fetchall()
        q_elapsed = (time.perf_counter() - start_t) / repetitions * 1000.0  # ms
        query_times[q_name] = q_elapsed
        total_time += q_elapsed

    return total_time, query_times, query_plans

def evaluate_index_benefits(conn: sqlite3.Connection) -> np.ndarray:
    """Measures baseline latency and individual index benefits Delta_Latency."""
    cur = conn.cursor()
    for cand in CANDIDATE_INDEXES:
        cur.execute(cand['drop_sql'])
    cur.execute("ANALYZE;")
    conn.commit()

    base_total, _, _ = benchmark_workload(conn)
    benefits = []

    for cand in CANDIDATE_INDEXES:
        cur.execute(cand['sql'])
        cur.execute("ANALYZE;")
        conn.commit()

        new_total, _, _ = benchmark_workload(conn)
        benefit = max(0.05, base_total - new_total)
        benefits.append(benefit)

        # Drop index
        cur.execute(cand['drop_sql'])
        cur.execute("ANALYZE;")
        conn.commit()

    return np.array(benefits)

def solve_greedy_index_selection(benefits: np.ndarray, storage_budget_kb: float) -> dict:
    """Greedy Knapsack: Selects indexes by Benefit / Storage ratio."""
    start_t = time.perf_counter()
    sizes = np.array([cand['storage_kb'] for cand in CANDIDATE_INDEXES])
    ratios = benefits / sizes
    sorted_order = np.argsort(-ratios)

    selected = []
    used_storage = 0.0
    total_benefit = 0.0

    for idx in sorted_order:
        if used_storage + sizes[idx] <= storage_budget_kb:
            selected.append(idx)
            used_storage += sizes[idx]
            total_benefit += benefits[idx]

    runtime = time.perf_counter() - start_t
    return {
        'method': 'Greedy Benefit-to-Storage',
        'selected_indexes': [CANDIDATE_INDEXES[i]['id'] for i in selected],
        'selected_indices': selected,
        'total_benefit_ms': float(total_benefit),
        'storage_used_kb': float(used_storage),
        'runtime_ms': runtime * 1000.0
    }

def solve_ilp_index_selection(benefits: np.ndarray, storage_budget_kb: float) -> dict:
    """Exact 0-1 Integer Linear Program (ILP) Knapsack Solver."""
    start_t = time.perf_counter()
    sizes = np.array([cand['storage_kb'] for cand in CANDIDATE_INDEXES])
    n = len(benefits)

    c = -benefits
    A = sizes.reshape(1, -1)
    constraints = LinearConstraint(A, -np.inf, storage_budget_kb)
    integrality = np.ones(n)
    bounds = Bounds(lb=0.0, ub=1.0)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    runtime = time.perf_counter() - start_t

    sol = np.round(res.x).astype(int)
    selected_indices = [i for i, val in enumerate(sol) if val == 1]
    used_storage = sum(sizes[i] for i in selected_indices)
    total_benefit = float(-res.fun)

    return {
        'method': 'Exact ILP Knapsack',
        'selected_indexes': [CANDIDATE_INDEXES[i]['id'] for i in selected_indices],
        'selected_indices': selected_indices,
        'total_benefit_ms': total_benefit,
        'storage_used_kb': float(used_storage),
        'runtime_ms': runtime * 1000.0
    }
