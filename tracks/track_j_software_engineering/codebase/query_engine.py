"""
In-Memory Relational Query Processing Engine.
Module of realistic production analytics codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

from typing import Any, Callable, Optional

class TableRecord:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    def get(self, col: str, default: Any = None) -> Any:
        return self.data.get(col, default)

    def __getitem__(self, col: str) -> Any:
        return self.data[col]

    def __setitem__(self, col: str, val: Any):
        self.data[col] = val

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

class QueryContext:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = [TableRecord(r) for r in records]

    def filter(self, predicate: Callable[[TableRecord], bool]) -> 'QueryContext':
        filtered = [r for r in self.records if predicate(r)]
        return QueryContext([r.to_dict() for r in filtered])

    def project(self, columns: list[str]) -> 'QueryContext':
        projected = []
        for r in self.records:
            sub = {col: r.get(col) for col in columns}
            projected.append(sub)
        return QueryContext(projected)

    def sort_by(self, key: str, reverse: bool = False) -> 'QueryContext':
        sorted_records = sorted(self.records, key=lambda r: r.get(key), reverse=reverse)
        return QueryContext([r.to_dict() for r in sorted_records])

    def limit(self, n: int) -> 'QueryContext':
        return QueryContext([r.to_dict() for r in self.records[:n]])

    def hash_join(
        self,
        other: 'QueryContext',
        left_on: str,
        right_on: str,
        prefix_other: str = "right_"
    ) -> 'QueryContext':
        """Performs classical hash join between two in-memory datasets."""
        # 1. Build hash table on 'other'
        hash_table: dict[Any, list[TableRecord]] = {}
        for r in other.records:
            key = r.get(right_on)
            hash_table.setdefault(key, []).append(r)

        # 2. Probe hash table with self
        joined_results = []
        for l_rec in self.records:
            l_key = l_rec.get(left_on)
            if l_key in hash_table:
                for r_rec in hash_table[l_key]:
                    merged = dict(l_rec.data)
                    for k, v in r_rec.data.items():
                        merged[f"{prefix_other}{k}"] = v
                    joined_results.append(merged)

        return QueryContext(joined_results)

    def group_by_aggregate(
        self,
        group_col: str,
        agg_specs: dict[str, tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """
        agg_specs format: {output_name: (column, 'sum' | 'count' | 'avg' | 'min' | 'max')}
        """
        groups: dict[Any, list[TableRecord]] = {}
        for r in self.records:
            key = r.get(group_col)
            groups.setdefault(key, []).append(r)

        results = []
        for g_val, group_records in groups.items():
            row = {group_col: g_val}
            for out_name, (col, agg_fn) in agg_specs.items():
                vals = [r.get(col) for r in group_records if r.get(col) is not None]
                if agg_fn == 'count':
                    row[out_name] = len(vals)
                elif agg_fn == 'sum':
                    row[out_name] = sum(vals)
                elif agg_fn == 'avg':
                    row[out_name] = sum(vals) / max(len(vals), 1)
                elif agg_fn == 'min':
                    row[out_name] = min(vals) if vals else None
                elif agg_fn == 'max':
                    row[out_name] = max(vals) if vals else None
                else:
                    raise ValueError(f"Unsupported aggregate function: {agg_fn}")
            results.append(row)

        return results

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def count(self) -> int:
        return len(self.records)
