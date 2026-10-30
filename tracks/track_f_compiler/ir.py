"""
Intermediate Representation (Three-Address Code) and Control Flow Graph.
Student: Ayaan Mirza (Roll No: 241070043)
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Instruction:
    op: str                 # 'ASSIGN', 'ADD', 'SUB', 'MUL', 'DIV', 'LABEL', 'JUMP', 'JUMP_IF_FALSE', 'PRINT', 'CMP_LT'
    arg1: Optional[str]     # variable or constant
    arg2: Optional[str]     # variable or constant
    dest: Optional[str]     # destination variable or label

    def __str__(self):
        if self.op == 'LABEL':
            return f"{self.dest}:"
        elif self.op == 'ASSIGN':
            return f"  {self.dest} = {self.arg1}"
        elif self.op in ('ADD', 'SUB', 'MUL', 'DIV', 'CMP_LT'):
            op_sym = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/', 'CMP_LT': '<'}[self.op]
            return f"  {self.dest} = {self.arg1} {op_sym} {self.arg2}"
        elif self.op == 'JUMP':
            return f"  jump {self.dest}"
        elif self.op == 'JUMP_IF_FALSE':
            return f"  jump_if_false {self.arg1} -> {self.dest}"
        elif self.op == 'PRINT':
            return f"  print {self.arg1}"
        return f"  {self.op} {self.arg1} {self.arg2} {self.dest}"

class BasicBlock:
    def __init__(self, block_id: int):
        self.id = block_id
        self.instructions: list[Instruction] = []
        self.predecessors: list['BasicBlock'] = []
        self.successors: list['BasicBlock'] = []
        # Liveness sets
        self.gen_set: set[str] = set()
        self.kill_set: set[str] = set()
        self.in_set: set[str] = set()
        self.out_set: set[str] = set()

def is_constant(val: Optional[str]) -> bool:
    if val is None:
        return False
    try:
        int(val)
        return True
    except ValueError:
        try:
            float(val)
            return True
        except ValueError:
            return False

def evaluate_const_op(op: str, c1: str, c2: str) -> str:
    v1 = float(c1) if '.' in c1 else int(c1)
    v2 = float(c2) if '.' in c2 else int(c2)
    if op == 'ADD':
        res = v1 + v2
    elif op == 'SUB':
        res = v1 - v2
    elif op == 'MUL':
        res = v1 * v2
    elif op == 'DIV':
        res = v1 // v2 if isinstance(v1, int) and isinstance(v2, int) and v2 != 0 else (v1 / v2 if v2 != 0 else 0)
    elif op == 'CMP_LT':
        res = 1 if v1 < v2 else 0
    else:
        raise ValueError(f"Unknown op {op}")
    return str(res)
