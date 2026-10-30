# Prompt Log: Track B (Discrete Optimization & QUBO)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track B — Discrete Optimization, ILP & QUBO

## 1. Initial Prompt
> "Implement 0-1 Knapsack combinatorial solvers using Greedy (ratio-based), Dynamic Programming ($O(N \cdot W)$), Branch-and-Bound with fractional LP relaxation, and Integer Linear Programming (ILP). Formulate Graph Max-Cut as a QUBO and Ising model with binary-to-spin variable transformation ($x_i = (1 - s_i)/2$), and implement classical Simulated Annealing and Tabu Search to solve it as the baseline for Track C. Formulate Constrained Feature Selection with a quadratic penalty term $\lambda (\sum x_i - K)^2$."

## 2. AI Response & Observed Error
The AI drafted `combinatorial_solvers.py` and `qubo_formulation.py`. However, for `scipy.optimize.milp`, it initially supplied `bounds = [(0, 1) for _ in range(n)]`, which crashed due to SciPy API incompatibility with `Bounds`.

## 3. Revised Prompt (Correction)
> "Fix the bounds parameter for `scipy.optimize.milp`. Import `scipy.optimize.Bounds` and pass `bounds = Bounds(lb=0.0, ub=1.0)`. Ensure `integrality = np.ones(n)` is properly defined before calling `milp`."

## 4. Resulting Code Changes
- Updated `solve_ilp_knapsack` to use `Bounds(lb=0, ub=1)`.
- Verified penalty expansion for feature selection: $\lambda (\sum x_i - K)^2 = \lambda [\sum x_i (1 - 2K) + 2 \sum_{i < j} x_i x_j + K^2]$.
- Saved baseline graph instance to `tracks/track_b_discrete/results/maxcut_baseline.json`.

## 5. Verification Evidence
- Verified via `test_track_b_knapsack_optimality` and `test_track_b_maxcut_qubo_equivalence`.
- Both DP and ILP produced the exact optimal knapsack value (220.0), while Simulated Annealing found the optimal cut weight (9.0).
