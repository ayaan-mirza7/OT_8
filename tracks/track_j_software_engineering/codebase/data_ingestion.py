"""
Data Ingestion and Schema Validation Pipeline.
Module of realistic production codebase for Track J software optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import csv
import json
import io
from typing import Any, Optional

class DataValidationError(Exception):
    pass

class SchemaField:
    def __init__(self, name: str, field_type: type, required: bool = True, default: Any = None):
        self.name = name
        self.field_type = field_type
        self.required = required
        self.default = default

    def validate(self, value: Any) -> Any:
        if value is None or value == "":
            if self.required:
                if self.default is not None:
                    return self.default
                raise DataValidationError(f"Field '{self.name}' is required but was empty/null.")
            return self.default
        try:
            return self.field_type(value)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"Field '{self.name}' invalid type: expected {self.field_type.__name__}, got {type(value).__name__}") from exc

class RecordSchema:
    def __init__(self, fields: list[SchemaField]):
        self.fields = {f.name: f for f in fields}

    def parse_record(self, raw_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}
        for name, field in self.fields.items():
            raw_val = raw_dict.get(name, None)
            cleaned[name] = field.validate(raw_val)
        return cleaned

class CSVStreamParser:
    def __init__(self, schema: RecordSchema, delimiter: str = ","):
        self.schema = schema
        self.delimiter = delimiter
        self.processed_count = 0
        self.error_count = 0

    def parse_csv_content(self, csv_text: str) -> list[dict[str, Any]]:
        reader = csv.DictReader(io.StringIO(csv_text), delimiter=self.delimiter)
        valid_records = []
        for row in reader:
            try:
                record = self.schema.parse_record(row)
                valid_records.append(record)
                self.processed_count += 1
            except DataValidationError:
                self.error_count += 1
        return valid_records

class JSONStreamParser:
    def __init__(self, schema: RecordSchema):
        self.schema = schema
        self.processed_count = 0
        self.error_count = 0

    def parse_json_lines(self, jsonl_text: str) -> list[dict[str, Any]]:
        records = []
        for line in jsonl_text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                raw_dict = json.loads(line)
                clean_rec = self.schema.parse_record(raw_dict)
                records.append(clean_rec)
                self.processed_count += 1
            except (json.JSONDecodeError, DataValidationError):
                self.error_count += 1
        return records

class BatchAggregator:
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.current_batch = []

    def add(self, item: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
        self.current_batch.append(item)
        if len(self.current_batch) >= self.batch_size:
            flush = list(self.current_batch)
            self.current_batch.clear()
            return flush
        return None

    def flush_remaining(self) -> list[dict[str, Any]]:
        flush = list(self.current_batch)
        self.current_batch.clear()
        return flush
