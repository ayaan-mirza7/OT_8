"""
Submodular Optimization Module.
Demonstrates:
1. Facility Location / Maximum Coverage submodular objective
2. Numerical verification of diminishing returns property:
   f(A U {e}) - f(A) >= f(B U {e}) - f(B) for all A subset B
3. Greedy selection algorithm with theoretical (1 - 1/e) ~ 0.632 approximation guarantee
4. Exact combinatorial baseline to evaluate true empirical approximation ratio
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np

class FacilityLocationSubmodular:
    """
    Facility location / sensor coverage objective:
    f(S) = sum_{j in Targets} max_{i in S} Coverage(i, j)
    Known to be monotone submodular: f(A) <= f(B) for A subset B,
    and satisfies diminishing returns.
    """
    def __init__(self, num_facilities: int = 14, num_targets: int = 40, seed: int = 42):
        self.n_fac = num_facilities
        self.n_tar = num_targets
        rng = np.random.default_rng(seed)
        
        # Coordinates in [0, 10] x [0, 10]
        self.fac_locs = rng.uniform(0.0, 10.0, (num_facilities, 2))
        self.tar_locs = rng.uniform(0.0, 10.0, (num_targets, 2))
        
        # Pairwise distance and coverage matrix
        dists = np.linalg.norm(self.fac_locs[:, None, :] - self.tar_locs[None, :, :], axis=2)
        # Coverage decay with distance: exp(-dist / 3.0)
        self.cov_matrix = np.exp(-dists / 3.0)

    def evaluate(self, selected_set: set[int] | list[int]) -> float:
        """Evaluates f(S). f(empty) = 0."""
        if not selected_set:
            return 0.0
        indices = list(selected_set)
        # For each target j, take max coverage over all selected facilities i in S
        sub_cov = self.cov_matrix[indices, :]
        max_cov_per_target = np.max(sub_cov, axis=0)
        return float(np.sum(max_cov_per_target))

    def verify_diminishing_returns(self, num_trials: int = 100, seed: int = 42) -> dict:
        """
        Empirically tests whether f satisfies diminishing returns:
        For random subsets A subset B and candidate element e not in B:
        Check Delta_A = f(A U {e}) - f(A) >= Delta_B = f(B U {e}) - f(B).
        """
        rng = np.random.default_rng(seed)
        violations = 0
        min_slack = float('inf')

        for _ in range(num_trials):
            # Pick random subset B
            k_b = rng.integers(2, self.n_fac)
            all_indices = list(range(self.n_fac))
            rng.shuffle(all_indices)
            
            b_list = all_indices[:k_b]
            # A is a strict subset of B
            k_a = rng.integers(1, k_b)
            a_list = b_list[:k_a]
            
            # Element e not in B
            remaining = [idx for idx in range(self.n_fac) if idx not in b_list]
            if not remaining:
                continue
            e = rng.choice(remaining)

            f_a = self.evaluate(a_list)
            f_ae = self.evaluate(a_list + [e])
            delta_a = f_ae - f_a

            f_b = self.evaluate(b_list)
            f_be = self.evaluate(b_list + [e])
            delta_b = f_be - f_b

            slack = delta_a - delta_b
            if slack < min_slack:
                min_slack = slack

            # Allow small numerical tolerance
            if delta_a < delta_b - 1e-9:
                violations += 1

        return {
            'trials': num_trials,
            'violations': violations,
            'is_submodular': violations == 0,
            'min_slack': float(min_slack)
        }

    def greedy_selection(self, k: int) -> dict:
        """
        Greedy algorithm for cardinality constrained submodular maximization:
        Starts with S = empty, iteratively adds element with maximum marginal gain.
        Guarantee: f(S_greedy) >= (1 - 1/e) * f(S_opt) ~ 0.632 * f(S_opt).
        """
        start_t = time.perf_counter()
        selected = []
        marginal_gains = []

        for _ in range(k):
            best_cand = None
            best_gain = -1.0

            current_val = self.evaluate(selected)
            for cand in range(self.n_fac):
                if cand not in selected:
                    val_with_cand = self.evaluate(selected + [cand])
                    gain = val_with_cand - current_val
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand

            if best_cand is not None:
                selected.append(best_cand)
                marginal_gains.append(best_gain)

        runtime = time.perf_counter() - start_t
        final_val = self.evaluate(selected)
        return {
            'selected_facilities': selected,
            'objective_value': final_val,
            'marginal_gains': marginal_gains,
            'runtime_ms': runtime * 1000.0
        }

    def exact_optimal_selection(self, k: int) -> dict:
        """Exhaustive search over all subsets of size k."""
        import itertools
        start_t = time.perf_counter()
        best_val = -1.0
        best_set = None

        for combo in itertools.combinations(range(self.n_fac), k):
            val = self.evaluate(combo)
            if val > best_val:
                best_val = val
                best_set = combo

        runtime = time.perf_counter() - start_t
        return {
            'optimal_facilities': list(best_set),
            'optimal_value': best_val,
            'runtime_ms': runtime * 1000.0
        }
