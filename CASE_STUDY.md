# Main Process Case Study: Green Cloud Datacenter Optimization
### Dynamic Energy-Carbon Workload Scheduling & Cost Minimization
**Course:** Optimization Techniques — Lab Assignment 8 (The 14-Day Vibe-Coding Challenge)  
**Student Name:** Ayaan Mirza  
**Roll No:** 241070043  
**Track:** Track E (Main Process Case Study)  
**Implementation Code:** [tracks/track_e_case_study/](tracks/track_e_case_study/)  

---

## 1. Executive Summary & Process Selection

### The Process Chosen
We selected **Sustainable Cloud Datacenter Workload Allocation & Dynamic Energy-Carbon Cost Management**. Datacenters currently account for approximately 1.5% to 2% of total worldwide electrical power consumption, with carbon footprints continuing to expand rapidly under AI and cloud computing demands.

### Empirical Evidence of Inefficiency in the Baseline
Profiling the naive baseline scheduling mechanism (First-Fit Greedy dispatch) on a cluster of heterogeneous servers under a 24-hour synthetic workload ($N=40$ compute jobs, $M=6$ servers) revealed two major systemic inefficiencies:
1. **Idle Server Power Waste:** Servers remained active in low-power idle states consuming $80\text{ W}$ each even when running only a single small task or zero workload, rather than consolidating tasks and placing unused servers into deep sleep.
2. **Tariff and Grid Carbon Desynchronization:** Baseline First-Fit schedules tasks immediately upon arrival regardless of ambient grid conditions. This causes large batch compute jobs to run during the evening peak (hours 17:00–21:00) when Time-of-Use tariffs spike to $\$0.28/\text{kWh}$ and grid carbon intensity peaks at $450\text{ gCO}_2/\text{kWh}$ (due to sunset and fossil peaking plants).

---

## 2. Mathematical Problem Formulation

### Decision Variables
For $N$ batch jobs, $M$ heterogeneous servers, and $T = 24$ discrete 1-hour time slots ($t \in \{0, 1, \dots, 23\}$):
- $x_{ijt} \in \{0, 1\}$: Binary variable indicating if Job $i$ runs on Server $j$ at time slot $t$.
- $s_i \in [a_i, d_i - w_i]$: Integer start time slot for Job $i$.
- $u_{jt} \in [0.0, 1.0]$: Continuous CPU utilization ratio of Server $j$ at hour $t$:
  $$u_{jt} = \frac{\sum_{i=1}^N x_{ijt} \cdot \text{CPU}_i}{\text{CapCPU}_j}$$
- $y_{jt} \in \{0, 1\}$: Operational power state of Server $j$ at hour $t$ ($1$ if $\sum_i x_{ijt} > 0$, else $0$).

### Objective Function
Minimize total 24-hour operational cost, balancing energy expenditure, carbon abatement penalties, and SLA violation penalties:
$$\min \quad J = \sum_{t=0}^{23} \left[ \left(\text{Tariff}(t) + \beta \cdot \text{CarbonIntensity}(t)\right) \cdot \sum_{j=1}^M P(u_{jt}, y_{jt}) \right] + \sum_{i=1}^N \lambda \cdot \max(0, \text{finish}_i - \text{deadline}_i)$$

Where:
- $\text{Tariff}(t)$ is the dynamic Time-of-Use grid electricity price ($\$ / \text{kWh}$).
- $\text{CarbonIntensity}(t)$ is grid emissions in $\text{gCO}_2 / \text{kWh}$, with solar surplus lowering emissions between 11:00 and 15:00.
- $\beta = 0.0005\ \$ / \text{gCO}_2$ is the carbon tax / abatement weighting parameter.
- $\lambda = \$25.00 / \text{hour}$ is the financial penalty for exceeding a job's SLA deadline.
- $P(u_{jt}, y_{jt})$ is the non-linear power consumption function (Watts):
  $$P(u_{jt}, y_{jt}) = y_{jt} \cdot P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) \cdot u_{jt}^2$$
  with $P_{\text{idle}} = 80\text{ W}$ and $P_{\text{max}} = 280\text{ W}$.

### Constraints & Assumptions
1. **Job Release Time (Arrival):** A job cannot start before it arrives:
   $$s_i \ge a_i \quad \forall i \in \{1, \dots, N\}$$
2. **Non-Preemptive Execution:** Once started, Job $i$ runs contiguously on Server $j$ for its full required duration $w_i$:
   $$\sum_{t=s_i}^{s_i + w_i - 1} x_{ijt} = w_i \quad \text{and} \quad \text{finish}_i = s_i + w_i$$
3. **Server Capacity Constraints (Hard):** Aggregate CPU and RAM demands cannot exceed server physical limits at any hour:
   $$\sum_{i=1}^N x_{ijt} \cdot \text{CPU}_i \le \text{CapCPU}_j \quad \forall j, t$$
   $$\sum_{i=1}^N x_{ijt} \cdot \text{RAM}_i \le \text{CapRAM}_j \quad \forall j, t$$
4. **SLA Deadlines (Soft / Hard Penalty):** Each job has an SLA deadline $d_i$. Any overrun incurs heavy penalty $\lambda$.

---

## 3. Comparison of Solvers & Experimental Results

Three distinct scheduling strategies were benchmarked on the exact same synthetic workload and grid trace (`seed=42`):

| Metric / Parameter | Baseline: Naive First-Fit | Method 1: Green Carbon-Aware Greedy | Method 2: Genetic Algorithm (GA) | Measured GA Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Electricity Cost (\$)** | \$76.84 | \$54.91 | **\$49.33** | **-35.8% cost reduction** |
| **Carbon Footprint ($\text{kg } CO_2$)** | 118.4 kg | 81.5 kg | **74.1 kg** | **-37.4% carbon cut** |
| **SLA Deadline Violations** | 0 | 0 | **0** | **100% SLA compliant** |
| **Capacity Constraint Violations** | 0 | 0 | **0** | **0 violations** |
| **Compute Execution Runtime** | 4.2 ms | 18.5 ms | **842.1 ms** | Metaheuristic trade-off |

### Detailed Strategy Breakdown
1. **Baseline (First-Fit Greedy):**
   - Dispatches jobs immediately at $t = a_i$ to the first server with sufficient CPU/RAM headroom.
   - Result: Peak workloads coincide directly with expensive, carbon-heavy peak grid hours.
2. **Green Carbon-Aware Greedy:**
   - Evaluates the marginal composite cost $(\text{Tariff}(t) + \beta \cdot \text{Carbon}(t)) \cdot \Delta P$ across all feasible time windows $t \in [a_i, d_i - w_i]$ on all available servers.
   - Shifts deferrable batch workloads into the midday solar surplus window (hours 11:00–15:00).
   - Achieves 28.5% cost reduction and 31.2% carbon reduction in under 20 ms.
3. **Genetic Algorithm (GA) with SLA Repair:**
   - Population size: 50, Generations: 100, Tournament Selection ($k=3$), Two-point crossover ($p_c = 0.85$), Gaussian mutation ($p_m = 0.15$).
   - Infeasible chromosomes (violating capacity or deadlines) are penalized and repaired using an active capacity projector.
   - Achieves the global Pareto frontier: **\$49.33 total cost (35.8% savings)** and **74.1 kg $CO_2$ (37.4% savings)** with zero SLA violations.

---

## 4. Correctness & Constraint Verification

All experimental outputs were validated programmatically:
- **Zero SLA Violations:** Verified $\text{finish}_i \le d_i$ for all $N=40$ jobs.
- **Zero Capacity Exceedance:** Verified $\max_{j, t}(u_{jt}) \le 1.0$ and $\max_{j, t}(\text{RAM}_{jt}) \le \text{CapRAM}_j$.
- **Convergence Plots:** High-resolution multi-panel plots generated in [track_e_case_study_plots.png](tracks/track_e_case_study/results/track_e_case_study_plots.png).
- **Summary Metrics:** Stored in [case_study_summary.csv](tracks/track_e_case_study/results/case_study_summary.csv).

---

## 5. Trade-Off Analysis & Scalability

1. **Metaheuristic Runtime vs Quality:** The GA delivers an additional 7.3% cost improvement over Green Greedy, but requires $842\text{ ms}$ vs $18.5\text{ ms}$. For real-time dispatch ($N > 10,000$ jobs), Green Greedy provides the superior throughput/latency balance.
2. **Battery Energy Storage Systems (BESS):** Future extensions could integrate on-site battery storage, enabling the datacenter to absorb excess solar power during midday and discharge during evening tariff peaks.
3. **Distributed Workload Migration:** Cross-datacenter geographical workload shifting could route batch jobs to regions with currently active renewable generation.
