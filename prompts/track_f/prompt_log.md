# Prompt Log: Track F (Compiler Optimization Passes & Register Allocation)
**Student:** Ayaan Mirza (Roll No: 241070043)  
**Track:** Track F — Compiler Optimization

## 1. Initial Prompt
> "Build a mini Three-Address Code (TAC) intermediate representation with basic blocks. Implement four optimization passes: Constant Folding & Forward Propagation, Common Subexpression Elimination (CSE), Dead Code Elimination (DCE), and Loop-Invariant Code Motion (LICM). Formulate Register Allocation as an optimization problem: build variable interference graphs from backward liveness analysis, implement the Greedy Chaitin-Briggs Kempe graph colouring heuristic, and formulate exact 0-1 ILP minimizing spill costs. Build a virtual machine execution engine to verify output equality between unoptimized and optimized programs."

## 2. AI Response Summary
The assistant drafted `ir.py`, `passes.py`, `reg_alloc.py`, `code_gen.py`, and `experiment.py`. It formulated the ILP register allocation model with assignment and conflict constraints.

## 3. Revised Prompt
> "Add safety guards in `constant_folding_and_propagation` so that constants are invalidated across label branch targets to prevent unsafe forward propagation into loops. Add cycle count estimation in the Virtual Machine to quantify performance speedups beyond raw instruction counts."

## 4. Resulting Code Changes
- Enhanced `passes.py` with label-based constant boundary invalidation.
- Built `reg_alloc.py` using `scipy.optimize.milp` with bounds `Bounds(lb=0.0, ub=1.0)`.
- Verified strict correctness: `res_unopt['output'] == res_opt['output']` evaluated to `True` with byte-identical outputs `[40, 6550]`, while dynamic instructions were reduced by 37.5% and cycle counts by 39.8%.
