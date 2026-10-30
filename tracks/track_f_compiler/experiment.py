"""
Track F Compiler Optimization Experiment Runner.
Executes optimization passes, register allocation formulation, and output verification.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt

from tracks.track_f_compiler.ir import Instruction
from tracks.track_f_compiler.passes import (
    constant_folding_and_propagation,
    common_subexpression_elimination,
    dead_code_elimination,
    loop_invariant_code_motion
)
from tracks.track_f_compiler.reg_alloc import (
    LivenessAnalyzer,
    solve_greedy_register_allocation,
    solve_ilp_register_allocation
)
from tracks.track_f_compiler.code_gen import VirtualMachine

def generate_benchmark_program() -> list[Instruction]:
    """
    Constructs a rich benchmark TAC program featuring:
    - Constant folding opportunities (e.g. 15 * 4, 100 - 25)
    - Common subexpressions (x + y computed repeatedly)
    - Dead code assignments never used
    - Loop with an invariant expression inside body
    """
    instructions = [
        # 1. Constant folding opportunities
        Instruction('ASSIGN', '15', None, 'c1'),
        Instruction('ASSIGN', '4', None, 'c2'),
        Instruction('MUL', 'c1', 'c2', 'folded_const'),      # 15 * 4 = 60
        Instruction('SUB', '100', '25', 'folded_sub'),       # 100 - 25 = 75

        # 2. Dead code (never used in outputs)
        Instruction('MUL', 'folded_const', '99', 'dead_var1'),
        Instruction('ADD', 'dead_var1', '1000', 'dead_var2'),

        # 3. Base variables
        Instruction('ASSIGN', '7', None, 'x'),
        Instruction('ASSIGN', '3', None, 'y'),

        # 4. Common subexpression elimination (x + y repeated)
        Instruction('ADD', 'x', 'y', 't1'),                  # t1 = 7 + 3 = 10
        Instruction('MUL', 't1', '2', 'val1'),               # val1 = 20
        Instruction('ADD', 'x', 'y', 't2'),                  # redundant! x + y
        Instruction('ADD', 'val1', 't2', 'val2'),            # val2 = 30
        Instruction('ADD', 'x', 'y', 't3'),                  # redundant! x + y
        Instruction('ADD', 'val2', 't3', 'out_accum'),       # out_accum = 40

        # 5. Loop initialization
        Instruction('ASSIGN', '0', None, 'i'),
        Instruction('ASSIGN', '0', None, 'out_sum'),
        Instruction('ASSIGN', '50', None, 'limit'),

        # Loop header
        Instruction('LABEL', None, None, 'loop_start'),
        Instruction('CMP_LT', 'i', 'limit', 'cond'),
        Instruction('JUMP_IF_FALSE', 'cond', None, 'loop_end'),

        # Loop body with loop-invariant computation
        Instruction('MUL', '7', '8', 'inv_factor'),          # Invariant! 7 * 8 = 56
        Instruction('ADD', 'inv_factor', 'folded_sub', 'inv_total'), # Invariant! 56 + 75 = 131
        Instruction('ADD', 'out_sum', 'inv_total', 'out_sum'),

        # Counter increment
        Instruction('ADD', 'i', '1', 'i'),
        Instruction('JUMP', None, None, 'loop_start'),

        Instruction('LABEL', None, None, 'loop_end'),
        Instruction('PRINT', 'out_accum', None, None),
        Instruction('PRINT', 'out_sum', None, None)
    ]
    return instructions

def main():
    print("[Track F] Running Compiler Optimization Passes & Verification...")
    os.makedirs("tracks/track_f_compiler/results", exist_ok=True)

    unopt_prog = generate_benchmark_program()

    # Apply Pass 1: Constant Folding & Propagation
    t0 = time.perf_counter()
    pass1 = constant_folding_and_propagation(unopt_prog)
    t_pass1 = (time.perf_counter() - t0) * 1000.0

    # Apply Pass 2: Common Subexpression Elimination
    t0 = time.perf_counter()
    pass2 = common_subexpression_elimination(pass1)
    t_pass2 = (time.perf_counter() - t0) * 1000.0

    # Apply Pass 3: Dead Code Elimination
    t0 = time.perf_counter()
    pass3 = dead_code_elimination(pass2)
    t_pass3 = (time.perf_counter() - t0) * 1000.0

    # Apply Pass 4: Loop-Invariant Code Motion
    t0 = time.perf_counter()
    pass4 = loop_invariant_code_motion(pass3)
    t_pass4 = (time.perf_counter() - t0) * 1000.0

    total_opt_time = t_pass1 + t_pass2 + t_pass3 + t_pass4

    # Execute Unoptimized Program on VM
    vm_unopt = VirtualMachine(unopt_prog)
    res_unopt = vm_unopt.execute()

    # Execute Fully Optimized Program on VM
    vm_opt = VirtualMachine(pass4)
    res_opt = vm_opt.execute()

    # STRICT CORRECTNESS VERIFICATION
    assert res_unopt['output'] == res_opt['output'], "Correctness Violation: Program outputs differ!"
    assert res_unopt['final_env']['out_sum'] == res_opt['final_env']['out_sum'], "State mismatch on out_sum!"
    print(f" -> Output Verification PASSED: Outputs are byte-identical! {res_opt['output']}")

    # Pass-by-Pass Pipeline Table
    stages = [
        ('Unoptimized', unopt_prog),
        ('Pass 1: Constant Folding', pass1),
        ('Pass 2: CSE', pass2),
        ('Pass 3: DCE', pass3),
        ('Pass 4: LICM (Fully Optimized)', pass4)
    ]
    pipeline_records = []
    for stage_name, insts in stages:
        vm = VirtualMachine(insts)
        res = vm.execute()
        pipeline_records.append({
            'Stage': stage_name,
            'Code_Size_Insts': res['static_code_size'],
            'Dynamic_Insts_Executed': res['dynamic_inst_count'],
            'Cycle_Count': res['cycle_count'],
            'Runtime_ms': res['runtime_ms']
        })
    pipeline_df = pd.DataFrame(pipeline_records)
    pipeline_df.to_csv("tracks/track_f_compiler/results/optimization_pipeline_summary.csv", index=False)

    # Register Allocation Formulation Experiment (Greedy vs ILP)
    analyzer = LivenessAnalyzer(pass4)
    k_regs = 3
    greedy_reg = solve_greedy_register_allocation(analyzer, k_registers=k_regs)
    ilp_reg = solve_ilp_register_allocation(analyzer, k_registers=k_regs)

    reg_df = pd.DataFrame([greedy_reg, ilp_reg])
    reg_df.to_csv("tracks/track_f_compiler/results/register_allocation_comparison.csv", index=False)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Track F: Compiler Optimization Metrics & Register Allocation\nAyaan Mirza (Roll No: 241070043)", fontsize=13, fontweight='bold')

    # Plot 1: Reduction in Dynamic Instructions & Cycles across Passes
    ax = axes[0]
    stages_short = ['Unopt', 'ConstFold', 'CSE', 'DCE', 'LICM (Final)']
    cycles = pipeline_df['Cycle_Count'].tolist()
    dyn_insts = pipeline_df['Dynamic_Insts_Executed'].tolist()
    x = range(len(stages_short))

    ax.plot(x, dyn_insts, 'b-o', lw=2, label='Dynamic Instructions')
    ax.plot(x, cycles, 'r--s', lw=2, label='Estimated Cycle Count')
    ax.set_xticks(x)
    ax.set_xticklabels(stages_short, rotation=15, ha='right')
    ax.set_ylabel("Execution Count")
    ax.set_title("Instruction and Cycle Count Reduction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Register Allocation Spills & Runtime
    ax = axes[1]
    methods = [greedy_reg['method'], ilp_reg['method']]
    spill_costs = [greedy_reg['total_spill_cost'], ilp_reg['total_spill_cost']]
    bars = ax.bar(methods, spill_costs, color=['#f59e0b', '#10b981'], width=0.4)
    ax.set_ylabel("Total Spill Penalty Cost")
    ax.set_title("Register Allocation: Greedy vs Exact ILP")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"Cost: {yval:.1f}", ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tracks/track_f_compiler/results/track_f_compiler_plots.png", dpi=300)
    plt.close()

    print("[Track F] Completed successfully. Artifacts saved in tracks/track_f_compiler/results/")

if __name__ == '__main__':
    main()
