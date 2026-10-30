# Prompt Log: Track H (DBMS Workload Optimization & Index Selection)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track H — DBMS Optimization

## 1. Initial Prompt
> "Build an in-memory SQLite TPC-H synthetic dataset generator with Customer, Orders, LineItem, and Part tables. Formulate Secondary Index Selection as a 0-1 Knapsack problem under storage budget $B$, comparing Greedy benefit-to-storage ratio against exact 0-1 ILP. Benchmark before-and-after query execution plans using EXPLAIN QUERY PLAN, report latency, throughput, and storage cost, and rigorously verify that query results are identical."

## 2. AI Response & Observed Error
The assistant implemented `schema_and_data.py`, `index_optimizer.py`, and `experiment.py`. On the initial run, the queries were non-selective (e.g. `WHERE l_shipdate >= '2024-05-01'`) and `ANALYZE;` was not executed. Consequently, SQLite's cost-based query planner chose sequential table scans, yielding a negative speedup (0.79x). In addition, `np` was missing from `experiment.py` imports.

## 3. Revised Prompt (Correction)
> "Import `numpy as np` in `experiment.py`. Refactor the query workload to include highly selective predicate filters (e.g. equality on indexed foreign keys and dates). Execute `ANALYZE;` after creating candidate indexes so SQLite's internal catalog statistics trigger index seeks instead of sequential table scans."

## 4. Resulting Code Changes
- Added `import numpy as np` to `experiment.py`.
- Injected `ANALYZE;` after index creation in `index_optimizer.py` and `experiment.py`.
- Query execution plans transformed from `SCAN lineitem` to `SEARCH lineitem USING INDEX idx_lineitem_ship_disc`.
- Overall workload latency dropped from **4.38 ms to 1.39 ms (3.15x speedup)**.
- Strict verification confirmed 100% identical result sets across all queries.
