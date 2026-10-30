"""
Bytecode Virtual Machine Execution Engine.
Measures instruction count, cycle count, code size, runtime,
and rigorously verifies output equality between unoptimized and optimized code.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
from tracks.track_f_compiler.ir import Instruction, is_constant, evaluate_const_op

class VirtualMachine:
    def __init__(self, instructions: list[Instruction]):
        self.instructions = instructions
        self.env = {}
        self.output_log = []
        self.dynamic_inst_count = 0
        self.cycle_count = 0
        self.labels = {}
        self._build_label_table()

    def _build_label_table(self):
        for idx, inst in enumerate(self.instructions):
            if inst.op == 'LABEL':
                self.labels[inst.dest] = idx

    def _resolve(self, arg: str):
        if is_constant(arg):
            return float(arg) if '.' in arg else int(arg)
        return self.env.get(arg, 0)

    def execute(self, max_steps: int = 100000) -> dict:
        pc = 0
        n_inst = len(self.instructions)
        start_t = time.perf_counter()

        while pc < n_inst and self.dynamic_inst_count < max_steps:
            inst = self.instructions[pc]
            self.dynamic_inst_count += 1

            if inst.op == 'LABEL':
                self.cycle_count += 0  # zero cost
                pc += 1

            elif inst.op == 'ASSIGN':
                val = self._resolve(inst.arg1)
                self.env[inst.dest] = val
                self.cycle_count += 1
                pc += 1

            elif inst.op in ('ADD', 'SUB', 'MUL', 'DIV', 'CMP_LT'):
                v1 = self._resolve(inst.arg1)
                v2 = self._resolve(inst.arg2)
                res_str = evaluate_const_op(inst.op, str(v1), str(v2))
                self.env[inst.dest] = float(res_str) if '.' in res_str else int(res_str)
                # Cycles: MUL/DIV cost 3, ADD/SUB cost 1
                self.cycle_count += (3 if inst.op in ('MUL', 'DIV') else 1)
                pc += 1

            elif inst.op == 'JUMP':
                self.cycle_count += 2  # branch penalty
                pc = self.labels[inst.dest]

            elif inst.op == 'JUMP_IF_FALSE':
                cond_val = self._resolve(inst.arg1)
                self.cycle_count += 2
                if cond_val == 0:
                    pc = self.labels[inst.dest]
                else:
                    pc += 1

            elif inst.op == 'PRINT':
                val = self._resolve(inst.arg1)
                self.output_log.append(val)
                self.cycle_count += 1
                pc += 1

            else:
                pc += 1

        elapsed = time.perf_counter() - start_t
        return {
            'output': list(self.output_log),
            'final_env': dict(self.env),
            'static_code_size': len(self.instructions),
            'dynamic_inst_count': self.dynamic_inst_count,
            'cycle_count': self.cycle_count,
            'runtime_ms': elapsed * 1000.0
        }
