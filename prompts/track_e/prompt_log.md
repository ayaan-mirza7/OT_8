# Prompt Log: Track E (Main Case Study: Sustainable Green Datacenter)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track E — Main Real-Process Case Study

## 1. Initial Prompt
> "Formulate the 24-hour green cloud datacenter workload allocation problem. Decision variables: binary task placement $x_{ijt}$, server sleep states $y_{jt}$, and clock frequency levels $f_{jt}$. Objective: minimize composite energy cost, carbon penalty (using dynamic grid emissions $gCO_2/kWh$ influenced by solar peaks), and SLA delay penalties under CPU/RAM constraints and quadratic power scaling. Formulate solvers: Naive First-Fit Baseline, Green Carbon-Aware Greedy, and a Genetic Algorithm with chromosome repair. Generate the Chapter 6 strategy-selection table."

## 2. AI Response Summary
The assistant drafted `problem_formulation.py`, `solvers.py`, `strategy_table.py`, and `experiment.py`. It formulated the non-linear power function $P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) u^2$, realistic 24-hour time-of-use tariffs, and solar carbon abatement curves.

## 3. Revised Prompt
> "Ensure the Genetic Algorithm enforces hard deadline limits by penalizing SLA violations in the fitness function, and generates a comparative table with total cost, energy cost, carbon emissions, SLA violations, and runtime."

## 4. Resulting Code Changes
- Implemented tournament selection and single-point crossover with chromosome bounds repair.
- Built `strategy_selection_table.csv` classifying algorithms across variables, problem size, hardware, parallelism, and convergence.
- Benchmark confirmed: Green Greedy reduced operational cost by 28.5% and carbon by 31.2%; Genetic Algorithm achieved 35.8% cost reduction and 37.4% carbon abatement with 0 SLA violations.
