"""
Realistic Multi-Module Analytics and ML Data Infrastructure Codebase.
Scale: Over 1,000 Lines of Code.
Modules:
- data_ingestion: Schema validation and streaming parsers
- feature_transforms: Scalers, encoders, and rolling statistics
- model_engine: Linear and logistic estimators with metrics
- api_service: Token bucket rate limiter and request dispatching
- graph_analytics: Directed graphs, PageRank, and Dijkstra shortest paths
- cache_manager: LRU, LFU, and TTL cache policies
- query_engine: Relational operators (filter, project, hash join, group-by)
- ci_pipeline: Pipeline DAG execution and dependency tracking
- test_suite: 40 unit and integration tests

Student: Ayaan Mirza (Roll No: 241070043)
"""

from tracks.track_j_software_engineering.codebase.data_ingestion import SchemaField, RecordSchema, CSVStreamParser, JSONStreamParser, BatchAggregator
from tracks.track_j_software_engineering.codebase.feature_transforms import StandardScaler, MinMaxScaler, OneHotEncoder, RollingWindowStats
from tracks.track_j_software_engineering.codebase.model_engine import LinearRegressionModel, LogisticBinaryClassifier, ModelMetrics
from tracks.track_j_software_engineering.codebase.api_service import TokenBucketRateLimiter, APIRequest, APIRouter
from tracks.track_j_software_engineering.codebase.graph_analytics import GraphNode, DirectedGraph
from tracks.track_j_software_engineering.codebase.cache_manager import LRUCache, LFUCache, TTLCache
from tracks.track_j_software_engineering.codebase.query_engine import TableRecord, QueryContext
from tracks.track_j_software_engineering.codebase.ci_pipeline import BuildStatus, BuildStage, PipelineDAG

__all__ = [
    "SchemaField", "RecordSchema", "CSVStreamParser", "JSONStreamParser", "BatchAggregator",
    "StandardScaler", "MinMaxScaler", "OneHotEncoder", "RollingWindowStats",
    "LinearRegressionModel", "LogisticBinaryClassifier", "ModelMetrics",
    "TokenBucketRateLimiter", "APIRequest", "APIRouter",
    "GraphNode", "DirectedGraph",
    "LRUCache", "LFUCache", "TTLCache",
    "TableRecord", "QueryContext",
    "BuildStatus", "BuildStage", "PipelineDAG"
]
