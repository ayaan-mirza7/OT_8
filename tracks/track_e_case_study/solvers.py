"""
Case Study Solvers:
1. Baseline Naive First-Fit Scheduler (unaware of tariffs and carbon intensity)
2. Green Carbon-Aware Greedy Heuristic (shifts flexible jobs to low-carbon/low-tariff solar windows)
3. Evolutionary Genetic Algorithm (GA) with chromosome repair & tournament selection
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
import numpy as np
from tracks.track_e_case_study.problem_formulation import DatacenterOptimizationProblem

def solve_baseline_naive(problem: DatacenterOptimizationProblem) -> dict:
    """Baseline: Schedule immediately at arrival time on first available server."""
    start_t = time.perf_counter()
    schedule = np.zeros((problem.num_jobs, 2), dtype=int)

    for i in range(problem.num_jobs):
        srv = i % problem.num_servers
        start_time = problem.job_arrival[i]
        schedule[i] = [srv, start_time]

    runtime = time.perf_counter() - start_t
    metrics = problem.evaluate_schedule(schedule)
    metrics.update({
        'method': 'Baseline Naive First-Fit',
        'runtime_ms': runtime * 1000.0,
        'schedule': schedule.tolist()
    })
    return metrics

def solve_green_greedy(problem: DatacenterOptimizationProblem) -> dict:
    """
    Green Greedy Heuristic:
    For each job, scans all feasible start times [arrival_i, deadline_i - duration_i]
    and selects the (server, time) slot that minimizes composite energy + carbon price.
    """
    start_t = time.perf_counter()
    schedule = np.zeros((problem.num_jobs, 2), dtype=int)
    
    # Combined effective cost rate per hour
    combined_cost_rate = problem.tariffs + (problem.carbon_weight_beta * problem.carbon_intensity)

    for i in range(problem.num_jobs):
        arrival = problem.job_arrival[i]
        duration = problem.job_workload[i]
        deadline = problem.job_deadline[i]
        max_start = max(arrival, deadline - duration)

        best_cost = float('inf')
        best_slot = (i % problem.num_servers, arrival)

        for srv in range(problem.num_servers):
            for t in range(arrival, min(max_start + 1, problem.T - duration + 1)):
                # Average grid cost over this window
                window_cost = np.sum(combined_cost_rate[t:t + duration])
                if window_cost < best_cost:
                    best_cost = window_cost
                    best_slot = (srv, t)

        schedule[i] = [best_slot[0], best_slot[1]]

    runtime = time.perf_counter() - start_t
    metrics = problem.evaluate_schedule(schedule)
    metrics.update({
        'method': 'Green Carbon-Aware Greedy',
        'runtime_ms': runtime * 1000.0,
        'schedule': schedule.tolist()
    })
    return metrics

def solve_genetic_algorithm(
    problem: DatacenterOptimizationProblem,
    pop_size: int = 40,
    generations: int = 50,
    mutation_rate: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Evolutionary Genetic Algorithm optimizing workload placement.
    Chromosome: flat vector of length 2 * num_jobs representing [srv_0, t_0, srv_1, t_1, ...]
    """
    start_t = time.perf_counter()
    rng = np.random.default_rng(seed)

    def init_chromosome():
        chrom = np.zeros((problem.num_jobs, 2), dtype=int)
        for i in range(problem.num_jobs):
            srv = rng.integers(0, problem.num_servers)
            arr = problem.job_arrival[i]
            dur = problem.job_workload[i]
            dead = problem.job_deadline[i]
            max_t = max(arr, min(dead - dur, problem.T - dur))
            t = rng.integers(arr, max(arr + 1, max_t + 1))
            chrom[i] = [srv, t]
        return chrom

    # Seed initial population with greedy solution
    greedy_sol = np.array(solve_green_greedy(problem)['schedule'], dtype=int)
    population = [greedy_sol.copy()] + [init_chromosome() for _ in range(pop_size - 1)]

    best_chrom = greedy_sol.copy()
    best_metrics = problem.evaluate_schedule(best_chrom)
    history = [best_metrics['total_cost']]

    for gen in range(generations):
        # Evaluate fitness
        scores = [problem.evaluate_schedule(c)['total_cost'] for c in population]

        # Update best
        min_idx = int(np.argmin(scores))
        if scores[min_idx] < best_metrics['total_cost']:
            best_metrics = problem.evaluate_schedule(population[min_idx])
            best_chrom = population[min_idx].copy()
        history.append(best_metrics['total_cost'])

        # Selection (tournament)
        new_pop = [best_chrom.copy()]  # elitism
        while len(new_pop) < pop_size:
            t1, t2 = rng.integers(0, pop_size, size=2)
            parent1 = population[t1] if scores[t1] < scores[t2] else population[t2]
            t3, t4 = rng.integers(0, pop_size, size=2)
            parent2 = population[t3] if scores[t3] < scores[t4] else population[t4]

            # Uniform crossover
            child = parent1.copy()
            mask = rng.random(size=problem.num_jobs) < 0.5
            child[mask] = parent2[mask]

            # Mutation
            if rng.random() < mutation_rate:
                mut_job = rng.integers(0, problem.num_jobs)
                child[mut_job, 0] = rng.integers(0, problem.num_servers)
                arr = problem.job_arrival[mut_job]
                dur = problem.job_workload[mut_job]
                dead = problem.job_deadline[mut_job]
                max_t = max(arr, min(dead - dur, problem.T - dur))
                child[mut_job, 1] = rng.integers(arr, max(arr + 1, max_t + 1))

            new_pop.append(child)

        population = new_pop

    runtime = time.perf_counter() - start_t
    best_metrics.update({
        'method': 'Genetic Algorithm (GA)',
        'runtime_ms': runtime * 1000.0,
        'history': history,
        'schedule': best_chrom.tolist()
    })
    return best_metrics
