"""
Continuous Integration Pipeline & Build Dependency Graph Execution Engine.
Module of realistic production codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import time
from enum import Enum
from typing import Callable, Any, Optional

class BuildStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class BuildStage:
    def __init__(self, name: str, action: Callable[[], bool], estimated_cost_sec: float = 1.0):
        self.name = name
        self.action = action
        self.estimated_cost = estimated_cost_sec
        self.dependencies: set['BuildStage'] = set()
        self.status = BuildStatus.PENDING
        self.runtime_sec = 0.0
        self.error_message: Optional[str] = None

    def add_dependency(self, stage: 'BuildStage'):
        self.dependencies.add(stage)

class PipelineDAG:
    def __init__(self, name: str):
        self.name = name
        self.stages: dict[str, BuildStage] = {}

    def add_stage(self, stage: BuildStage):
        self.stages[stage.name] = stage

    def link(self, parent_name: str, child_name: str):
        parent = self.stages[parent_name]
        child = self.stages[child_name]
        child.add_dependency(parent)

    def validate_acyclic(self) -> bool:
        visited = {}  # 0: unvisited, 1: visiting, 2: visited
        for name in self.stages:
            visited[name] = 0

        def dfs(curr: str) -> bool:
            visited[curr] = 1
            stage = self.stages[curr]
            for dep in stage.dependencies:
                if visited[dep.name] == 1:
                    return False  # Cycle!
                if visited[dep.name] == 0:
                    if not dfs(dep.name):
                        return False
            visited[curr] = 2
            return True

        for name in self.stages:
            if visited[name] == 0:
                if not dfs(name):
                    return False
        return True

    def execute_sequential(self) -> dict[str, Any]:
        """Executes all stages sequentially according to dependency resolution."""
        start_t = time.perf_counter()
        execution_order = []
        completed = set()
        failed = False

        while len(completed) < len(self.stages):
            progress_made = False
            for name, stage in self.stages.items():
                if stage.status == BuildStatus.PENDING:
                    # Check if all dependencies are successfully satisfied
                    deps_satisfied = all(dep in completed for dep in stage.dependencies)
                    if deps_satisfied:
                        stage.status = BuildStatus.RUNNING
                        s_time = time.perf_counter()
                        try:
                            ok = stage.action()
                            stage.status = BuildStatus.SUCCESS if ok else BuildStatus.FAILED
                        except Exception as e:
                            stage.status = BuildStatus.FAILED
                            stage.error_message = str(e)
                        stage.runtime_sec = time.perf_counter() - s_time
                        completed.add(stage)
                        execution_order.append(stage.name)
                        progress_made = True

                        if stage.status == BuildStatus.FAILED:
                            failed = True
                            break
            if failed or not progress_made:
                break

        total_elapsed = time.perf_counter() - start_t
        return {
            'pipeline': self.name,
            'success': not failed,
            'total_elapsed_sec': total_elapsed,
            'execution_order': execution_order,
            'stages_completed': len(completed)
        }
