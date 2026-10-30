"""
Compiler Optimization Passes:
1. Constant Folding and Propagation
2. Common Subexpression Elimination (CSE)
3. Dead Code Elimination (DCE)
4. Loop-Invariant Code Motion (LICM) and Strength Reduction
Student: Ayaan Mirza (Roll No: 241070043)
"""

from tracks.track_f_compiler.ir import Instruction, is_constant, evaluate_const_op

def constant_folding_and_propagation(instructions: list[Instruction]) -> list[Instruction]:
    """Pass 1: Constant Folding and Forward Propagation."""
    const_table = {}
    optimized = []

    for inst in instructions:
        if inst.op == 'LABEL':
            # Invalidate constants across jump targets to ensure block safety
            const_table.clear()
            optimized.append(inst)
            continue

        arg1 = const_table.get(inst.arg1, inst.arg1)
        arg2 = const_table.get(inst.arg2, inst.arg2)

        if inst.op in ('ADD', 'SUB', 'MUL', 'DIV', 'CMP_LT'):
            if is_constant(arg1) and is_constant(arg2):
                folded_val = evaluate_const_op(inst.op, arg1, arg2)
                const_table[inst.dest] = folded_val
                optimized.append(Instruction('ASSIGN', folded_val, None, inst.dest))
                continue
            else:
                # If variable was updated, invalidate
                if inst.dest in const_table:
                    del const_table[inst.dest]
                optimized.append(Instruction(inst.op, arg1, arg2, inst.dest))
                continue

        elif inst.op == 'ASSIGN':
            if is_constant(arg1):
                const_table[inst.dest] = arg1
            elif arg1 in const_table:
                const_table[inst.dest] = const_table[arg1]
            else:
                if inst.dest in const_table:
                    del const_table[inst.dest]
            optimized.append(Instruction('ASSIGN', arg1, None, inst.dest))
            continue

        elif inst.op in ('JUMP_IF_FALSE', 'PRINT'):
            optimized.append(Instruction(inst.op, arg1, arg2, inst.dest))
            continue

        optimized.append(inst)

    return optimized

def common_subexpression_elimination(instructions: list[Instruction]) -> list[Instruction]:
    """Pass 2: Available Expressions / Common Subexpression Elimination (CSE)."""
    expr_table = {}  # (op, arg1, arg2) -> var_name
    var_dependencies = {}  # var -> list of expr keys
    optimized = []

    for inst in instructions:
        if inst.op == 'LABEL' or inst.op == 'JUMP':
            expr_table.clear()
            var_dependencies.clear()
            optimized.append(inst)
            continue

        if inst.op in ('ADD', 'SUB', 'MUL', 'DIV', 'CMP_LT'):
            # Normalize commutative expressions
            if inst.op in ('ADD', 'MUL') and inst.arg1 > inst.arg2:
                key = (inst.op, inst.arg2, inst.arg1)
            else:
                key = (inst.op, inst.arg1, inst.arg2)

            if key in expr_table:
                # Re-use existing computation!
                reused_var = expr_table[key]
                optimized.append(Instruction('ASSIGN', reused_var, None, inst.dest))
            else:
                expr_table[key] = inst.dest
                var_dependencies.setdefault(inst.arg1, []).append(key)
                var_dependencies.setdefault(inst.arg2, []).append(key)
                optimized.append(inst)

            # Invalidate any expressions depending on overwritten dest
            if inst.dest in var_dependencies:
                for dep_key in var_dependencies[inst.dest]:
                    if dep_key in expr_table:
                        del expr_table[dep_key]
                del var_dependencies[inst.dest]
            continue

        elif inst.op == 'ASSIGN':
            if inst.dest in var_dependencies:
                for dep_key in var_dependencies[inst.dest]:
                    if dep_key in expr_table:
                        del expr_table[dep_key]
                del var_dependencies[inst.dest]
            optimized.append(inst)
            continue

        optimized.append(inst)

    return optimized

def dead_code_elimination(instructions: list[Instruction]) -> list[Instruction]:
    """Pass 3: Backward Liveness & Dead Code Elimination."""
    # Iterative DCE until fixed point
    changed = True
    current = instructions

    while changed:
        changed = False
        used_vars = set()
        for inst in current:
            if inst.op == 'PRINT' and inst.arg1:
                used_vars.add(inst.arg1)
            elif inst.op == 'JUMP_IF_FALSE' and inst.arg1:
                used_vars.add(inst.arg1)
            elif inst.op in ('ADD', 'SUB', 'MUL', 'DIV', 'CMP_LT'):
                if inst.arg1 and not is_constant(inst.arg1):
                    used_vars.add(inst.arg1)
                if inst.arg2 and not is_constant(inst.arg2):
                    used_vars.add(inst.arg2)
            elif inst.op == 'ASSIGN':
                if inst.arg1 and not is_constant(inst.arg1):
                    used_vars.add(inst.arg1)

        new_instructions = []
        for inst in current:
            # If assignment or arithmetic defines a variable never used anywhere and has no side effects, drop it
            if inst.op in ('ASSIGN', 'ADD', 'SUB', 'MUL', 'DIV', 'CMP_LT'):
                # Essential variables: e.g. final return or output variables
                if inst.dest and inst.dest not in used_vars and not inst.dest.startswith("out_"):
                    changed = True
                    continue  # eliminated!
            new_instructions.append(inst)

        current = new_instructions

    return current

def loop_invariant_code_motion(instructions: list[Instruction]) -> list[Instruction]:
    """Pass 4: Hoists instructions inside loop bodies that do not depend on loop counter variables."""
    # Detect simple loops delineated by LABEL loop_start -> ... -> JUMP loop_start
    in_loop = False
    loop_body = []
    loop_modified_vars = set()
    pre_instructions = []
    post_instructions = []

    for inst in instructions:
        if inst.op == 'LABEL' and 'loop_start' in inst.dest:
            in_loop = True
            loop_body = []
            loop_modified_vars = set()
            continue

        if in_loop and inst.op == 'JUMP' and 'loop_start' in inst.dest:
            in_loop = False
            # Find invariant instructions in loop_body
            hoisted = []
            remaining_body = []

            for body_inst in loop_body:
                if body_inst.op in ('ADD', 'SUB', 'MUL', 'DIV'):
                    # Invariant if operands are not modified within the loop
                    op1_invariant = is_constant(body_inst.arg1) or (body_inst.arg1 not in loop_modified_vars)
                    op2_invariant = is_constant(body_inst.arg2) or (body_inst.arg2 not in loop_modified_vars)
                    if op1_invariant and op2_invariant:
                        hoisted.append(body_inst)
                        continue
                remaining_body.append(body_inst)

            pre_instructions.extend(hoisted)
            pre_instructions.append(Instruction('LABEL', None, None, 'loop_start'))
            pre_instructions.extend(remaining_body)
            pre_instructions.append(inst)
            continue

        if in_loop:
            if inst.dest:
                loop_modified_vars.add(inst.dest)
            loop_body.append(inst)
        else:
            pre_instructions.append(inst)

    return pre_instructions
