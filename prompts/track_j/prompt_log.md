# Prompt Log: Track J (Software-Engineering Optimization)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track J — Software-Engineering Optimization

## 1. Initial Prompt
> "Build a realistic, non-trivial multi-module production codebase exceeding 1,000 lines of code across data ingestion, feature transforms, model estimators, API rate limiting, graph analytics, cache management, query processing, and CI pipeline execution. Implement Test-Suite Minimization (TSM) formulated as 0-1 Integer Linear Programming (Set Cover) and greedy additional coverage, and Test Prioritization (TSP) using APFD (Average Percentage of Faults Detected). Benchmark execution speedup and verify that 100% branch coverage is strictly preserved."

## 2. AI Response Summary
The assistant structured `codebase/` into 8 production modules plus `test_suite.py` containing 40 unit and integration tests. It verified that total codebase size reached **1,011 lines of code**, satisfying Section 6's scale requirement.

## 3. Revised Prompt
> "Formulate the ILP set cover problem in `test_optimizer.py` with binary decision variables $x_i \in \{0, 1\}$ representing test inclusion, minimizing total execution time $\sum t_i x_i$ subject to covering all branches. Implement `apfd_score` in `core/metrics.py` to evaluate fault detection velocity."

## 4. Resulting Code Changes
- Created `test_optimizer.py` with greedy and ILP set cover solvers.
- Built test accumulation visualization comparing APFD fault detection curves.
- Benchmark confirmed: ILP minimization reduced test suite runtime from **2062.2 ms to 285.0 ms (86.2% time reduction)** while maintaining **100% branch coverage (50/50 branches)**.
