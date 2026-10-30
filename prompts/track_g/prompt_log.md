# Prompt Log: Track G (xv6 OS Scheduler Optimization)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track G — xv6 Operating-System Optimization

## 1. Initial Prompt
> "Develop an xv6 kernel scheduler optimization. Generate unified patch diffs for xv6 implementing: (1) Multilevel Feedback Queue (MLFQ) with 3 priority queues ($Q_0, Q_1, Q_2$), aging, periodic priority boost to prevent starvation, and a `sys_setpriority` system call; and (2) Copy-On-Write (COW) Fork in virtual memory. Write a user-level benchmark C program (`sched_bench.c`) evaluating 3 workloads: CPU-bound, I/O burst interactive, and mixed heterogeneous. Build a cycle-accurate emulator to benchmark Default Round-Robin, Static Priority, MLFQ, and Stride scheduling on turnaround time, response time, and Jain's fairness index."

## 2. AI Response & Observed Error
The assistant generated `xv6_mlfq_priority.patch`, `cow_fork.patch`, `sched_bench.c`, and `emulator.py`. On the initial run, a `TypeError: Process.__init__() missing 1 required positional argument: 'remaining_cpu'` occurred in `emulator.py`.

## 3. Revised Prompt (Correction)
> "In `Process` dataclass in `emulator.py`, assign `remaining_cpu: int = 0` as a default argument so that benchmark workload instantiation with keyword arguments works cleanly."

## 4. Resulting Code Changes
- Updated `Process` dataclass with default argument.
- Emulated all 4 scheduling policies across 3 distinct workloads.
- Benchmarked on Mixed Workload: Default Round-Robin had average response time of 2.3 ticks; MLFQ reduced this to **0.2 ticks (91.3% improvement)** while preserving Jain's fairness index at 0.548 and avoiding starvation via periodic boost.
