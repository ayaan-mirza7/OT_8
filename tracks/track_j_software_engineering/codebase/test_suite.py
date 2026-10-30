"""
Comprehensive Test Suite for Codebase (>30 unit & integration tests).
Used for Test-Suite Minimization (TSM) and Prioritization (TSP) optimization.
Student: Ayaan Mirza (Roll No: 241070043)
"""

from tracks.track_j_software_engineering.codebase.data_ingestion import (
    SchemaField, RecordSchema, CSVStreamParser, JSONStreamParser, BatchAggregator, DataValidationError
)
from tracks.track_j_software_engineering.codebase.feature_transforms import (
    StandardScaler, MinMaxScaler, OneHotEncoder, RollingWindowStats
)
from tracks.track_j_software_engineering.codebase.model_engine import (
    LinearRegressionModel, LogisticBinaryClassifier, ModelMetrics
)
from tracks.track_j_software_engineering.codebase.api_service import (
    TokenBucketRateLimiter, APIRequest, APIRouter
)

# Test functions with simulated execution & explicit branch coverage metadata
def test_01_schema_field_valid():
    f = SchemaField("age", int, required=True)
    assert f.validate("25") == 25

def test_02_schema_field_required_missing():
    f = SchemaField("age", int, required=True)
    try:
        f.validate(None)
        assert False
    except DataValidationError:
        assert True

def test_03_schema_field_default():
    f = SchemaField("active", bool, required=False, default=True)
    assert f.validate(None) is True

def test_04_csv_parser_valid():
    schema = RecordSchema([SchemaField("id", int), SchemaField("score", float)])
    parser = CSVStreamParser(schema)
    res = parser.parse_csv_content("id,score\n1,95.5\n2,88.0\n")
    assert len(res) == 2
    assert res[0]["id"] == 1

def test_05_csv_parser_error_handling():
    schema = RecordSchema([SchemaField("id", int)])
    parser = CSVStreamParser(schema)
    res = parser.parse_csv_content("id\n1\ninvalid\n3\n")
    assert len(res) == 2
    assert parser.error_count == 1

def test_06_json_parser_valid():
    schema = RecordSchema([SchemaField("id", int), SchemaField("val", str)])
    parser = JSONStreamParser(schema)
    lines = '{"id": 1, "val": "ok"}\n{"id": 2, "val": "fine"}'
    res = parser.parse_json_lines(lines)
    assert len(res) == 2

def test_07_json_parser_malformed():
    schema = RecordSchema([SchemaField("id", int)])
    parser = JSONStreamParser(schema)
    res = parser.parse_json_lines('{"id": 1}\nnot_a_json\n{"id": 2}')
    assert len(res) == 2
    assert parser.error_count == 1

def test_08_batch_aggregator_flush():
    agg = BatchAggregator(batch_size=3)
    assert agg.add({"x": 1}) is None
    assert agg.add({"x": 2}) is None
    b = agg.add({"x": 3})
    assert len(b) == 3
    agg.add({"x": 4})
    assert len(agg.flush_remaining()) == 1

def test_09_standard_scaler_basic():
    X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    scaler = StandardScaler().fit(X)
    res = scaler.transform(X)
    assert len(res) == 3
    assert abs(res[1][0]) < 1e-6  # mean should be ~ 0

def test_10_standard_scaler_unfitted():
    scaler = StandardScaler()
    try:
        scaler.transform([[1.0]])
        assert False
    except RuntimeError:
        assert True

def test_11_minmax_scaler():
    X = [[0.0], [5.0], [10.0]]
    scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(X)
    res = scaler.transform(X)
    assert res[0][0] == 0.0
    assert res[2][0] == 1.0

def test_12_one_hot_encoder():
    enc = OneHotEncoder().fit(["cat", "dog", "bird"])
    v = enc.transform("cat")
    assert sum(v) == 1
    assert enc.transform("unknown") == [0, 0, 0]

def test_13_rolling_window_stats():
    roller = RollingWindowStats(window_size=3)
    _ = roller.update(10.0)
    _ = roller.update(20.0)
    st = roller.update(30.0)
    assert st["rolling_mean"] == 20.0
    assert st["min"] == 10.0
    assert st["max"] == 30.0

def test_14_linear_regression_fit_predict():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [2.0, 4.0, 6.0, 8.0]
    model = LinearRegressionModel(lr=0.05, max_iters=200).fit(X, y)
    preds = model.predict([[5.0]])
    assert abs(preds[0] - 10.0) < 0.5

def test_15_logistic_classifier():
    X = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [0, 0, 1, 1]
    clf = LogisticBinaryClassifier(lr=0.1, max_iters=150).fit(X, y)
    preds = clf.predict([[3.0], [-3.0]])
    assert preds[0] == 1
    assert preds[1] == 0

def test_16_model_metrics():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 0, 1, 0]
    acc = ModelMetrics.accuracy(y_true, y_pred)
    assert acc == 0.8
    f1 = ModelMetrics.f1_score(y_true, y_pred)
    assert f1 > 0.7

def test_17_token_bucket_rate_limiter():
    limiter = TokenBucketRateLimiter(capacity=2, refill_rate_per_sec=0.1)
    assert limiter.consume(1) is True
    assert limiter.consume(1) is True
    assert limiter.consume(1) is False

def test_18_api_router_dispatch_200():
    router = APIRouter()
    router.register_endpoint("/health", lambda req: {"status": "healthy"})
    req = APIRequest("/health", "client_1", {})
    res = router.dispatch(req)
    assert res["status"] == 200

def test_19_api_router_dispatch_404():
    router = APIRouter()
    req = APIRequest("/non_existent", "client_1", {})
    res = router.dispatch(req)
    assert res["status"] == 404

def test_20_api_router_rate_limit():
    router = APIRouter()
    router.register_endpoint("/ping", lambda req: {"pong": True})
    for _ in range(5):
        router.dispatch(APIRequest("/ping", "user_x", {}))
    # 6th request should be rate limited
    res = router.dispatch(APIRequest("/ping", "user_x", {}))
    assert res["status"] == 429

# Additional Tests for Graph Analytics & Caching
from tracks.track_j_software_engineering.codebase.graph_analytics import DirectedGraph
from tracks.track_j_software_engineering.codebase.cache_manager import LRUCache, LFUCache, TTLCache

def test_21_graph_topological_sort():
    g = DirectedGraph()
    g.add_edge("build", "test")
    g.add_edge("test", "deploy")
    order = g.topological_sort()
    assert order == ["build", "test", "deploy"]

def test_22_graph_cycle_detection():
    g = DirectedGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "A")
    try:
        g.topological_sort()
        assert False
    except ValueError:
        assert True

def test_23_dijkstra_shortest_path():
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_edge("B", "C", 2.0)
    g.add_edge("A", "C", 5.0)
    dists = g.dijkstra_shortest_path("A")
    assert dists["C"] == 3.0

def test_24_pagerank():
    g = DirectedGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")
    pr = g.pagerank()
    assert len(pr) == 3
    assert abs(pr["A"] - pr["B"]) < 1e-4

def test_25_lru_cache_eviction():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # evicts "b"
    assert cache.get("b") is None
    assert cache.get("c") == 3

def test_26_lfu_cache_eviction():
    cache = LFUCache(capacity=2)
    cache.put("x", 10)
    cache.put("y", 20)
    _ = cache.get("x")
    _ = cache.get("x")
    cache.put("z", 30)  # should evict "y" which has lower frequency
    assert cache.get("y") is None
    assert cache.get("x") == 10

def test_27_ttl_cache_expiration():
    cache = TTLCache(default_ttl_sec=0.1)
    cache.set("temp", 999)
    assert cache.get("temp") == 999
    import time
    time.sleep(0.15)
    assert cache.get("temp") is None

def test_28_rolling_stats_underflow():
    roller = RollingWindowStats(window_size=2)
    st = roller.update(5.0)
    assert st["rolling_mean"] == 5.0

def test_29_linear_regression_metrics():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.2]
    mse = ModelMetrics.mean_squared_error(y_true, y_pred)
    assert mse < 0.05

def test_30_api_router_500_exception():
    router = APIRouter()
    def failing_handler(req):
        raise RuntimeError("Internal crash")
    router.register_endpoint("/fail", failing_handler)
    res = router.dispatch(APIRequest("/fail", "client_err", {}))
    assert res["status"] == 500

# Tests for In-Memory Query Engine & CI Pipeline
from tracks.track_j_software_engineering.codebase.query_engine import QueryContext
from tracks.track_j_software_engineering.codebase.ci_pipeline import PipelineDAG, BuildStage, BuildStatus

def test_31_query_filter_and_project():
    data = [
        {"id": 1, "dept": "IT", "salary": 90000},
        {"id": 2, "dept": "HR", "salary": 65000},
        {"id": 3, "dept": "IT", "salary": 110000}
    ]
    ctx = QueryContext(data)
    res = ctx.filter(lambda r: r.get("dept") == "IT").project(["id", "salary"]).to_list()
    assert len(res) == 2
    assert "dept" not in res[0]

def test_32_query_sort_and_limit():
    data = [{"val": 30}, {"val": 10}, {"val": 50}, {"val": 20}]
    ctx = QueryContext(data)
    sorted_top2 = ctx.sort_by("val", reverse=True).limit(2).to_list()
    assert sorted_top2[0]["val"] == 50
    assert sorted_top2[1]["val"] == 30

def test_33_query_hash_join():
    customers = QueryContext([{"cid": 1, "name": "Alice"}, {"cid": 2, "name": "Bob"}])
    orders = QueryContext([{"oid": 101, "cid": 1, "amt": 250}, {"oid": 102, "cid": 1, "amt": 450}])
    joined = customers.hash_join(orders, left_on="cid", right_on="cid")
    assert joined.count() == 2
    res = joined.to_list()
    assert res[0]["name"] == "Alice"
    assert res[0]["right_amt"] == 250

def test_34_query_group_by_aggregates():
    items = [
        {"category": "Electronics", "price": 100},
        {"category": "Electronics", "price": 300},
        {"category": "Books", "price": 25}
    ]
    ctx = QueryContext(items)
    aggs = ctx.group_by_aggregate("category", {
        "total_rev": ("price", "sum"),
        "avg_price": ("price", "avg"),
        "item_count": ("price", "count")
    })
    assert len(aggs) == 2
    elec = next(a for a in aggs if a["category"] == "Electronics")
    assert elec["total_rev"] == 400
    assert elec["avg_price"] == 200.0

def test_35_ci_pipeline_dag_valid():
    dag = PipelineDAG("build_test_deploy")
    dag.add_stage(BuildStage("lint", lambda: True, 0.1))
    dag.add_stage(BuildStage("unit_tests", lambda: True, 0.3))
    dag.add_stage(BuildStage("deploy", lambda: True, 0.2))
    dag.link("lint", "unit_tests")
    dag.link("unit_tests", "deploy")
    assert dag.validate_acyclic() is True
    res = dag.execute_sequential()
    assert res["success"] is True
    assert res["execution_order"] == ["lint", "unit_tests", "deploy"]

def test_36_ci_pipeline_cycle_detected():
    dag = PipelineDAG("cyclic_dag")
    dag.add_stage(BuildStage("s1", lambda: True))
    dag.add_stage(BuildStage("s2", lambda: True))
    dag.link("s1", "s2")
    dag.link("s2", "s1")
    assert dag.validate_acyclic() is False

def test_37_ci_pipeline_failure_halts():
    dag = PipelineDAG("failing_pipeline")
    dag.add_stage(BuildStage("stage_ok", lambda: True))
    dag.add_stage(BuildStage("stage_fail", lambda: False))
    dag.add_stage(BuildStage("stage_downstream", lambda: True))
    dag.link("stage_ok", "stage_fail")
    dag.link("stage_fail", "stage_downstream")
    res = dag.execute_sequential()
    assert res["success"] is False
    assert "stage_downstream" not in res["execution_order"]

def test_38_lru_cache_hit_rate():
    cache = LRUCache(capacity=5)
    cache.put("k1", 100)
    _ = cache.get("k1")  # hit
    _ = cache.get("k2")  # miss
    assert cache.hit_rate() == 0.5

def test_39_ttl_cache_purge():
    cache = TTLCache(default_ttl_sec=0.05)
    cache.set("a", 1)
    cache.set("b", 2)
    import time
    time.sleep(0.08)
    purged = cache.purge_expired()
    assert purged == 2

def test_40_model_metrics_edge_cases():
    # Test F1 zero division safety
    f1 = ModelMetrics.f1_score([0, 0], [0, 0])
    assert f1 == 0.0


