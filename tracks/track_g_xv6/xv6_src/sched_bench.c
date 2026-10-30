/*
 * sched_bench.c - User-level benchmark program for xv6 scheduler optimization.
 * Tests 3 distinct workloads:
 * 1. CPU-Bound compute tasks
 * 2. I/O-Interactive bursty tasks (sleep/yield)
 * 3. Mixed heterogeneous workload (starvation testing)
 * Student: Ayaan Mirza (Roll No: 241070043)
 */

#include "types.h"
#include "stat.h"
#include "user.h"

#define NUM_PROCESSES 8

void cpu_bound_workload() {
    volatile int x = 0;
    for (int i = 0; i < 50000000; i++) {
        x += (i * 3) % 7;
    }
    exit();
}

void io_bound_workload() {
    for (int i = 0; i < 100; i++) {
        volatile int dummy = 0;
        for (int j = 0; j < 10000; j++) dummy += j;
        sleep(1); // yields CPU simulating disk/terminal I/O
    }
    exit();
}

void mixed_workload(int id) {
    if (id % 2 == 0) {
        cpu_bound_workload();
    } else {
        io_bound_workload();
    }
}

int main(int argc, char *argv[]) {
    printf(1, "=== xv6 Scheduler Optimization Benchmark ===\n");
    printf(1, "Student: Ayaan Mirza (Roll No: 241070043)\n");

    int mode = 1;
    if (argc > 1) {
        mode = atoi(argv[1]);
    }

    printf(1, "Launching workload mode %d (1: CPU-bound, 2: I/O-bound, 3: Mixed)...\n", mode);
    int pids[NUM_PROCESSES];
    int start_ticks = uptime();

    for (int i = 0; i < NUM_PROCESSES; i++) {
        int pid = fork();
        if (pid < 0) {
            printf(1, "Error: Fork failed!\n");
            exit();
        } else if (pid == 0) {
            // Child process
            if (mode == 1) cpu_bound_workload();
            else if (mode == 2) io_bound_workload();
            else mixed_workload(i);
        } else {
            pids[i] = pid;
        }
    }

    // Wait for all children to complete
    for (int i = 0; i < NUM_PROCESSES; i++) {
        wait();
    }

    int end_ticks = uptime();
    int total_duration = end_ticks - start_ticks;
    printf(1, "Workload completed in %d ticks.\n", total_duration);
    printf(1, "=== Benchmark Complete ===\n");
    exit();
}
