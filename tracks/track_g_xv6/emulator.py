"""
Cycle-Accurate xv6 Kernel Scheduling Simulator.
Simulates:
1. Default xv6 Round-Robin (RR)
2. Priority Preemptive Scheduler
3. Multilevel Feedback Queue (MLFQ) with Aging
4. Stride Scheduling (Proportional Share)
Benchmarked across 3 workloads:
- Workload 1: CPU-Bound Compute
- Workload 2: Interactive I/O Burst
- Workload 3: Mixed Heterogeneous Workload
Student: Ayaan Mirza (Roll No: 241070043)
"""

from dataclasses import dataclass, field
import numpy as np
from core.metrics import jains_fairness_index

@dataclass
class Process:
    pid: int
    name: str
    arrival_time: int
    total_cpu_needed: int
    remaining_cpu: int = 0
    io_frequency: int = 0         # yield every io_frequency ticks (0 if purely CPU-bound)
    io_wait_time: int = 2         # ticks spent waiting on I/O
    priority: int = 5             # 1 (lowest) to 10 (highest)
    tickets: int = 10             # for Stride scheduling

    # Runtime tracking
    first_run_time: int = -1
    completion_time: int = -1
    current_io_wait: int = 0
    mlfq_queue: int = 0           # 0 (Q0: 1 tick), 1 (Q1: 4 ticks), 2 (Q2: 8 ticks)
    mlfq_ticks_in_q: int = 0
    stride_pass: int = 0
    allocated_ticks: int = 0

def run_simulation(scheduler_type: str, workload: list[Process], boost_interval: int = 100, max_ticks: int = 10000) -> dict:
    # Deep copy workload
    processes: list[Process] = [
        Process(
            pid=p.pid, name=p.name, arrival_time=p.arrival_time,
            total_cpu_needed=p.total_cpu_needed, remaining_cpu=p.total_cpu_needed,
            io_frequency=p.io_frequency, io_wait_time=p.io_wait_time,
            priority=p.priority, tickets=p.tickets
        )
        for p in workload
    ]

    time_quanta = {0: 1, 1: 4, 2: 8} if scheduler_type == "MLFQ" else {0: 1}
    current_tick = 0
    completed = []
    ready_queue: list[Process] = []
    io_blocked: list[Process] = []

    # Calculate strides
    LARGE_CONSTANT = 10000
    for p in processes:
        p.stride_pass = 0

    while len(completed) < len(processes) and current_tick < max_ticks:
        # 1. Admit arriving processes
        for p in processes:
            if p.arrival_time == current_tick:
                p.mlfq_queue = 0
                p.mlfq_ticks_in_q = 0
                ready_queue.append(p)

        # 2. Advance I/O blocked processes
        new_io_blocked = []
        for p in io_blocked:
            p.current_io_wait -= 1
            if p.current_io_wait <= 0:
                # I/O completed: re-enter ready queue
                ready_queue.append(p)
            else:
                new_io_blocked.append(p)
        io_blocked = new_io_blocked

        # 3. MLFQ Periodic Priority Boost (starvation prevention)
        if scheduler_type == "MLFQ" and current_tick > 0 and (current_tick % boost_interval == 0):
            for p in ready_queue:
                p.mlfq_queue = 0
                p.mlfq_ticks_in_q = 0

        if not ready_queue:
            current_tick += 1
            continue

        # 4. Pick next process according to scheduler policy
        if scheduler_type == "Round_Robin":
            selected_proc = ready_queue.pop(0)

        elif scheduler_type == "Priority":
            # Highest priority first
            best_idx = int(np.argmax([p.priority for p in ready_queue]))
            selected_proc = ready_queue.pop(best_idx)

        elif scheduler_type == "MLFQ":
            # Select process from highest non-empty queue Q0 -> Q1 -> Q2
            min_q = min(p.mlfq_queue for p in ready_queue)
            # Pick first available at this queue level (FIFO within queue)
            for idx, p in enumerate(ready_queue):
                if p.mlfq_queue == min_q:
                    selected_proc = ready_queue.pop(idx)
                    break

        elif scheduler_type == "Stride":
            # Select process with minimum pass value
            min_pass_idx = int(np.argmin([p.stride_pass for p in ready_queue]))
            selected_proc = ready_queue.pop(min_pass_idx)
        else:
            raise ValueError(f"Unknown scheduler {scheduler_type}")

        # Mark first run
        if selected_proc.first_run_time == -1:
            selected_proc.first_run_time = current_tick

        # Execute 1 tick
        selected_proc.remaining_cpu -= 1
        selected_proc.mlfq_ticks_in_q += 1
        selected_proc.allocated_ticks += 1
        stride_val = LARGE_CONSTANT // max(selected_proc.tickets, 1)
        selected_proc.stride_pass += stride_val
        current_tick += 1

        # Check completion
        if selected_proc.remaining_cpu <= 0:
            selected_proc.completion_time = current_tick
            completed.append(selected_proc)
            continue

        # Check I/O trigger
        if selected_proc.io_frequency > 0 and (selected_proc.allocated_ticks % selected_proc.io_frequency == 0):
            selected_proc.current_io_wait = selected_proc.io_wait_time
            io_blocked.append(selected_proc)
            continue

        # Check MLFQ quantum expiration
        if scheduler_type == "MLFQ":
            q_limit = time_quanta[selected_proc.mlfq_queue]
            if selected_proc.mlfq_ticks_in_q >= q_limit:
                selected_proc.mlfq_ticks_in_q = 0
                if selected_proc.mlfq_queue < 2:
                    selected_proc.mlfq_queue += 1  # Demote
            ready_queue.append(selected_proc)
        else:
            ready_queue.append(selected_proc)

    # Calculate metrics
    turnaround_times = [p.completion_time - p.arrival_time for p in completed]
    waiting_times = [t - p.total_cpu_needed for t, p in zip(turnaround_times, completed)]
    response_times = [p.first_run_time - p.arrival_time for p in completed]
    allocations = [p.allocated_ticks for p in completed]
    fairness = jains_fairness_index(allocations)

    return {
        'scheduler': scheduler_type,
        'avg_turnaround_time': float(np.mean(turnaround_times)),
        'avg_waiting_time': float(np.mean(waiting_times)),
        'avg_response_time': float(np.mean(response_times)),
        'jains_fairness': float(fairness),
        'total_ticks': current_tick,
        'completed_processes': len(completed)
    }
