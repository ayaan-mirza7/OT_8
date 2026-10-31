# Optimization Techniques — Lab Assignment 8: Optimize Anything
### The 14-Day Vibe-Coding Challenge
**Student Name:** Ayaan Mirza  
**Roll No:** 241070043  
**Course:** Optimization Techniques  
**Repository:** Public Submission Package  

---

## 🚀 Overview

This repository contains the complete submission for **Assignment 8: The 14-Day Vibe-Coding Challenge**. The project implements, benchmarks, and verifies all ten compulsory experiment tracks (**Tracks A to J**) covering the complete Chapter 6 Optimization Techniques syllabus, a real-process main case study on green cloud datacenter optimization, compiler passes, operating system schedulers, DBMS query plans, quantum algorithms, and software engineering optimization on a 1,000+ line codebase.

All tracks follow the compulsory 7-step optimization problem-solving process:
1. **Discover the process** & identify inefficiencies
2. **Establish the baseline** with reproducible metrics
3. **Formulate the problem** (decision variables, objective, constraints)
4. **Select the method** from Chapter 6
5. **Use vibe coding** with documented AI prompts and revisions
6. **Experiment and compare** baseline vs optimized
7. **Analyze trade-offs** and limitations

---

## 📋 Repository Structure

```text
OT_8/
├── README.md                                  <- Project documentation and execution instructions
├── CHAPTER_6_CHECKLIST.md                     <- Complete syllabus-to-code mapping
├── AI_TOOLS_REGISTER.md                       <- AI tools register, failure log & comparisons
├── TECHNICAL_REPORT.md                        <- Full technical report & answers to 10 final questions
├── requirements.txt                           <- Python dependencies
├── core/                                      <- Shared metrics (Amdahl, Gustafson, Jain fairness, APFD)
│   ├── __init__.py
│   └── metrics.py
├── tracks/
│   ├── track_a_continuous/                   <- Track A: Continuous Optimization & Decoupled Backprop
│   ├── track_b_discrete/                     <- Track B: Discrete Solvers, ILP & Max-Cut QUBO/Ising
│   ├── track_c_quantum/                      <- Track C: QAOA, VQE, Parameter-Shift, Barren Plateaus
│   ├── track_d_parallel_distributed/         <- Track D: Submodular, Parallel, FedAvg & Multi-Task MGDA
│   ├── track_e_case_study/                   <- Track E: Green Cloud Datacenter Optimization
│   ├── track_f_compiler/                     <- Track F: TAC Compiler Passes & ILP Register Allocation
│   ├── track_g_xv6/                          <- Track G: xv6 Kernel Schedulers & COW Fork Patches
│   ├── track_h_dbms/                         <- Track H: TPC-H Database Index Knapsack & Seeks
│   ├── track_i_quantum_advantage/            <- Track I: 7 Quantum Algorithms, VQC vs MLP, Grover Advantage
│   └── track_j_software_engineering/         <- Track J: 1,000+ LOC Codebase Test Minimization (TSM/TSP)
├── prompts/                                   <- Prompt logs per track (A to J)
├── tests/                                     <- Automated pytest validation test suite
└── demo/                                      <- Interactive demonstration CLI (demo.py)
```

---

## 🛠️ Setup & Installation

### Requirements
- Python 3.10+ (tested on Python 3.14.2 64-bit on Windows)
- Git

### Installation
Clone the repository and install the standard scientific computing packages:
```bash
git clone https://github.com/ayaanmirza/OT_8.git
cd OT_8

# Optional: Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```

---

## ⚡ Execution Instructions

### 1. Run Automated Test Suite
To execute the comprehensive correctness test suite across all tracks:
```bash
python -m pytest tests/
```
To run tests for an isolated track (e.g., Track A or Track E):
```bash
python -m pytest tests/test_all_tracks.py -k "track_a"
python -m pytest tests/test_all_tracks.py -k "track_e"
```
Expected output:
```text
tests\test_all_tracks.py ............ [100%]
12 passed in 1.43s
```

### 2. Run Interactive Demonstration CLI
To launch the interactive benchmark runner:
```bash
python demo/demo.py
```
Or run any individual track directly from the command line:
```bash
python demo/demo.py E      # Run Main Case Study (Green Datacenter)
python demo/demo.py F      # Run Track F Compiler Passes & Verification
python demo/demo.py ALL    # Run all 10 tracks sequentially
```

### 3. Run Individual Track Benchmarks
Each track can also be executed as an independent module:
```bash
python -m tracks.track_a_continuous.experiment
python -m tracks.track_b_discrete.experiment
python -m tracks.track_c_quantum.experiment
python -m tracks.track_d_parallel_distributed.experiment
python -m tracks.track_e_case_study.experiment
python -m tracks.track_f_compiler.experiment
python -m tracks.track_g_xv6.experiment
python -m tracks.track_h_dbms.experiment
python -m tracks.track_i_quantum_advantage.experiment
python -m tracks.track_j_software_engineering.experiment
```

---

## 📊 Summary of Experimental Results Across All 10 Tracks

| Track | Domain / Method | Baseline Metric | Optimized Metric | Measured Improvement | Correctness Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | Continuous & Curvature | GD: 78 iterations to tol | Newton: 1 iteration; Adam: stable on noise | Machine precision on convex; robust on saddle | Decoupled analytical backprop verified |
| **Track B** | Combinatorial & QUBO | Random assignment: Cut=4.5 | Exact / SA / Tabu: Cut=**9.0** | Global optimum cut achieved; 0% gap in ILP | Exact exhaustive search baseline |
| **Track C** | Quantum QAOA & VQE | Classical exact cut: 9.0 | QAOA expected cut: **8.46** | **0.94 approximation ratio**; VQE converged | Statevector unitarity & shot convergence |
| **Track D** | Submodular & Parallel | Static load balance: 1.62 | Work-stealing: **1.08**; FedAvg $E=5$ | Greedy coverage **0.91** $\ge (1 - 1/e) \approx 0.632$ | Zero submodularity violations in 100 trials |
| **Track E** | Green Datacenter Case Study | First-Fit: \$76.84, 118.4 kg $CO_2$ | GA: **\$49.33, 74.1 kg $CO_2$** | **35.8% cost cut, 37.4% carbon cut** | **0 SLA violations**; hard constraints satisfied |
| **Track F** | Compiler Passes & RegAlloc | Dynamic Insts: 32, Cycles: 53 | Dynamic Insts: **20**, Cycles: **32** | **37.5% instruction reduction, 39.6% cycle cut** | Byte-identical output `[40, 6550]` |
| **Track G** | xv6 OS Scheduling | Round-Robin resp: 2.3 ticks | MLFQ resp: **0.2 ticks** | **91.3% interactive response speedup** | xv6 boots cleanly; user benchmarks pass |
| **Track H** | DBMS Workload Indexing | Table Scans: 4.38 ms | Index Seeks: **1.39 ms** (Q1: 0.01 ms) | **3.15x overall speedup (165.4x on Q1)** | Query results 100% byte-identical |
| **Track I** | Quantum Algorithms & QML | Classical search: $O(N)$ queries | Grover: $O(\sqrt{N})$ queries | Asymptotic quantum query speedup proved | 7 algorithms verified; honest negative sim result |
| **Track J** | Software Eng (1,011 LOC) | All tests: 2062.2 ms | ILP Minimized: **285.0 ms** | **86.2% CI test time reduction** | **100% branch coverage (50/50)** preserved |

---

## 🤖 AI Tools Disclosure & Vibe Coding Compliance

In accordance with Sections 8 and 9 of the assignment instructions:
- **AI Tools Used:** Antigravity IDE (Gemini 3.8 Flash), Claude Code (Claude 3.5 Sonnet), and ChatGPT (GPT-4o).
- **Prompt Logs:** Maintained with original prompts, revisions, code links, and verification notes in `prompts/track_a/` through `prompts/track_j/`.
- **Documented Failures Fixed:**
  1. *SciPy MILP Bounds Format:* Corrected list-of-tuples to `scipy.optimize.Bounds(lb=0, ub=1)`.
  2. *Federated Gradient Broadcasting:* Fixed 2D label column mismatch with 1D probability vector using `y.ravel()`.
  3. *SQLite Index Planner Statistics:* Added `ANALYZE;` to ensure the SQLite cost-based planner selected index seeks over table scans, increasing speedup from 0.79x to **3.15x**.
- **Full Disclosure:** Complete details and side-by-side tool comparisons are provided in [AI_TOOLS_REGISTER.md](file:///AI_TOOLS_REGISTER.md).

---

## 📄 Key Documentation Links
- [Complete Technical Report](file:///TECHNICAL_REPORT.md)
- [Chapter 6 Coverage Checklist](file:///CHAPTER_6_CHECKLIST.md)
- [AI Tools Register & Failure Log](file:///AI_TOOLS_REGISTER.md)

---
*Created by Ayaan Mirza (Roll No: 241070043) for Optimization Techniques Lab Assignment 8.*
