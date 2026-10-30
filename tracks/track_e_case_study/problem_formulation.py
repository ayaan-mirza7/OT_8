"""
Track E: Main Real-Process Case Study.
Domain: Sustainable Cloud Datacenter Workload Allocation & Dynamic Energy-Carbon Cost Optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import numpy as np

class DatacenterOptimizationProblem:
    """
    Mathematical formulation of 24-hour green datacenter scheduling.
    
    Decision Variables:
    - x[i, j, t] in {0, 1}: Binary assignment of Job i to Server j in time window t
    - f[j, t] in [f_min, f_max]: Continuous clock frequency / DVFS level for server j
    - y[j, t] in {0, 1}: Server operational state (1: active, 0: sleep/low-power)

    Objective Function:
    Minimize Total Operational Cost = Energy Cost + Carbon Penalty + SLA Violation Cost
    J = sum_t [ (Tariff(t) + beta * CarbonIntensity(t)) * sum_j Power(j, t) ] + Penalty * sum_i max(0, finish_i - deadline_i)
    """
    def __init__(self, num_jobs: int = 40, num_servers: int = 6, time_slots: int = 24, seed: int = 42):
        self.num_jobs = num_jobs
        self.num_servers = num_servers
        self.T = time_slots
        rng = np.random.default_rng(seed)

        # 1. Job specifications
        self.job_workload = rng.integers(2, 6, size=num_jobs)      # CPU cycles required
        self.job_ram = rng.integers(4, 16, size=num_jobs)          # GB RAM required
        self.job_arrival = rng.integers(0, 14, size=num_jobs)      # Arrival hour
        # Deadline between arrival + duration and hour 23
        self.job_deadline = self.job_arrival + self.job_workload + rng.integers(2, 7, size=num_jobs)
        self.job_deadline = np.clip(self.job_deadline, 0, self.T - 1)

        # 2. Server capacities
        self.server_cpu_cap = np.array([12, 16, 16, 20, 24, 32][:num_servers], dtype=float)
        self.server_ram_cap = np.array([32, 64, 64, 64, 128, 128][:num_servers], dtype=float)
        self.p_idle = 80.0     # Watts idle power
        self.p_max = 280.0     # Watts peak power

        # 3. Dynamic 24-hour Grid Environment
        hours = np.arange(self.T)
        # Time-of-Use Electricity Tariff ($/kWh): peaks at evening hours (17-21)
        self.tariffs = 0.10 + 0.18 * np.exp(-0.5 * ((hours - 19) / 2.5) ** 2) + 0.05 * np.exp(-0.5 * ((hours - 9) / 2.0) ** 2)
        
        # Carbon Intensity (gCO2 / kWh): solar surplus drops carbon at midday (11-15)
        solar_generation = np.maximum(0.0, np.sin((hours - 6) * np.pi / 12))
        self.carbon_intensity = 450.0 - 280.0 * solar_generation + rng.normal(0, 15, size=self.T)
        self.carbon_intensity = np.maximum(80.0, self.carbon_intensity)

        # Weights
        self.carbon_weight_beta = 0.0005  # $/gCO2
        self.sla_penalty_per_hour = 25.0   # $/hour overdue

    def evaluate_schedule(self, schedule_matrix: np.ndarray) -> dict:
        """
        schedule_matrix shape: (num_jobs, 2)
        schedule_matrix[i] = [assigned_server, start_time]
        """
        server_cpu_usage = np.zeros((self.num_servers, self.T))
        server_ram_usage = np.zeros((self.num_servers, self.T))
        server_active = np.zeros((self.num_servers, self.T), dtype=bool)

        sla_violations = 0
        sla_penalty = 0.0
        capacity_violations = 0

        for i in range(self.num_jobs):
            srv = int(schedule_matrix[i, 0])
            start_t = int(schedule_matrix[i, 1])
            duration = int(self.job_workload[i])
            finish_t = start_t + duration

            # Arrival check
            if start_t < self.job_arrival[i]:
                sla_violations += 1
                sla_penalty += (self.job_arrival[i] - start_t) * self.sla_penalty_per_hour * 2

            # Deadline SLA check
            if finish_t > self.job_deadline[i]:
                overdue = finish_t - self.job_deadline[i]
                sla_violations += 1
                sla_penalty += overdue * self.sla_penalty_per_hour

            # Accumulate server load
            for t in range(start_t, min(finish_t, self.T)):
                server_cpu_usage[srv, t] += 1
                server_ram_usage[srv, t] += self.job_ram[i]
                server_active[srv, t] = True

        # Check capacity constraints
        for s in range(self.num_servers):
            for t in range(self.T):
                if server_cpu_usage[s, t] > self.server_cpu_cap[s]:
                    capacity_violations += (server_cpu_usage[s, t] - self.server_cpu_cap[s])
                if server_ram_usage[s, t] > self.server_ram_cap[s]:
                    capacity_violations += (server_ram_usage[s, t] - self.server_ram_cap[s]) / 10.0

        # Compute power and cost
        # Power model: P(u) = P_idle + (P_max - P_idle) * u^2 (superlinear energy scaling)
        cpu_util = server_cpu_usage / self.server_cpu_cap[:, None]
        power_watts = np.where(server_active, self.p_idle + (self.p_max - self.p_idle) * (cpu_util ** 2), 10.0) # 10W deep sleep

        # Total kWh
        kwh_per_slot = power_watts / 1000.0  # 1 hour slots
        energy_cost = np.sum(kwh_per_slot * self.tariffs[None, :])
        carbon_emissions = np.sum(kwh_per_slot * self.carbon_intensity[None, :])
        carbon_cost = carbon_emissions * self.carbon_weight_beta

        total_cost = energy_cost + carbon_cost + sla_penalty + (capacity_violations * 100.0)

        return {
            'total_cost': float(total_cost),
            'energy_cost': float(energy_cost),
            'carbon_emissions_g': float(carbon_emissions),
            'carbon_cost': float(carbon_cost),
            'sla_penalty': float(sla_penalty),
            'sla_violations': int(sla_violations),
            'capacity_violations': float(capacity_violations),
            'feasible': (capacity_violations == 0 and sla_violations == 0)
        }
