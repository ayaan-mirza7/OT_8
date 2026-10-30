"""
Parallel Runtime Optimization and Scheduling.
Implements:
1. Multi-process Data Parallelism
2. Work Stealing Queue vs Static Chunking (load balancing)
3. Amdahl's Law and Gustafson's Law analysis
4. Speedup and Efficiency measurement
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def compute_heavy_task(item: tuple[int, int]) -> float:
    """Simulates realistic compute-bound numerical workload with heterogeneous task durations."""
    task_id, difficulty = item
    # CPU heavy loop: Monte Carlo estimation of high-dimensional integral
    rng = np.random.default_rng(task_id + 100)
    samples = rng.standard_normal((difficulty, 8))
    # Non-linear function evaluation
    val = np.sum(np.exp(-0.5 * np.sum(samples ** 2, axis=1)))
    return float(val)

def run_serial_workload(tasks: list[tuple[int, int]]) -> tuple[float, list[float]]:
    start_t = time.perf_counter()
    results = [compute_heavy_task(t) for t in tasks]
    elapsed = time.perf_counter() - start_t
    return elapsed, results

def run_parallel_workload(tasks: list[tuple[int, int]], num_workers: int) -> tuple[float, list[float]]:
    start_t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_heavy_task, tasks))
    elapsed = time.perf_counter() - start_t
    return elapsed, results

class WorkStealingSimulation:
    """
    Simulates work-stealing scheduling (Blumofe & Leiserson)
    vs static equal partitioning under heterogeneous workloads.
    """
    def __init__(self, task_costs: list[float], num_workers: int = 4):
        self.costs = task_costs
        self.P = num_workers

    def run_static_partitioning(self) -> dict:
        """Assigns tasks in static round-robin chunks without rebalancing."""
        worker_loads = [0.0] * self.P
        for idx, cost in enumerate(self.costs):
            worker_id = idx % self.P
            worker_loads[worker_id] += cost
        makespan = max(worker_loads)
        return {
            'schedule': 'Static Partitioning',
            'makespan': makespan,
            'worker_loads': worker_loads,
            'imbalance_ratio': max(worker_loads) / max(min(worker_loads), 1e-6)
        }

    def run_work_stealing(self, steal_overhead: float = 0.05) -> dict:
        """
        Simulates work-stealing deques:
        When a worker finishes its local queue, it steals a task from the busiest worker.
        """
        # Initial distribution
        deques = [collections.deque() for _ in range(self.P)]
        for idx, cost in enumerate(self.costs):
            deques[idx % self.P].append(cost)

        worker_times = [0.0] * self.P
        steals_count = 0

        while any(len(d) > 0 for d in deques):
            # Find worker with minimum accumulated time
            idle_worker = int(np.argmin(worker_times))
            if len(deques[idle_worker]) > 0:
                cost = deques[idle_worker].popleft()
                worker_times[idle_worker] += cost
            else:
                # Steal from busiest deque
                busiest_worker = max(range(self.P), key=lambda w: sum(deques[w]))
                if len(deques[busiest_worker]) > 0:
                    stolen_cost = deques[busiest_worker].pop()  # steal from tail
                    worker_times[idle_worker] += stolen_cost + steal_overhead
                    steals_count += 1
                else:
                    break

        makespan = max(worker_times)
        return {
            'schedule': 'Work Stealing',
            'makespan': makespan,
            'worker_loads': worker_times,
            'steals_count': steals_count,
            'imbalance_ratio': max(worker_times) / max(min(worker_times), 1e-6)
        }
