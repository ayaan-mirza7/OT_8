# Prompt Log: Track D (Submodular, Parallel & Distributed Optimization)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track D — Submodular, Parallel, and Distributed Optimization

## 1. Initial Prompt
> "Implement facility location coverage as a monotone submodular objective. Write an empirical verification routine for the diminishing returns condition $f(A \cup \{e\}) - f(A) \ge f(B \cup \{e\}) - f(B)$ for $A \subseteq B$, and compare Greedy selection against the exact combinatorial optimum to verify the $(1 - 1/e) \approx 0.632$ guarantee. Implement multi-worker data parallelism, measure empirical speedup/efficiency, and analyze them against Amdahl's and Gustafson's laws. Simulate work-stealing vs static scheduling. Implement FedAvg over non-IID partitions, and multi-task optimization using MGDA and GradNorm on a two-task model."

## 2. AI Response & Observed Error
The assistant implemented `submodular.py`, `parallel_runtime.py`, `federated.py`, and `multi_task.py`. During the first run of `federated.py`, a broadcasting error occurred: `ValueError: operands could not be broadcast together with shapes (4, 300) (4,)` because client label arrays were 2D columns $(N, 1)$ while weights were 1D vectors $(D,)$.

## 3. Revised Prompt (Correction)
> "In `compute_loss_and_grad`, flatten the target label array `y` using `y.ravel()` so that vector subtraction with 1D predicted probabilities `(p - y_flat)` does not trigger erroneous 2D outer broadcasting."

## 4. Resulting Code Changes
- Fixed array broadcasting in `federated.py`.
- Formulated analytical MGDA convex-hull minimum norm point solver $\min_{\alpha \in [0, 1]} \|\alpha g_1 + (1 - \alpha) g_2\|^2$.
- Verified $(1 - 1/e)$ approximation ratio: Greedy achieved 0.91 on facility coverage (surpassing the 0.632 lower bound).
- Simulated Amdahl's Law asymptote ($S \le 8.33$ for $f=0.88$) vs Gustafson scaled speedup.
