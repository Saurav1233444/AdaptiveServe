import concurrent.futures

from adaptiveserve.feedback import FeedbackStore


def test_empty_metrics_and_persisted_feedback(tmp_path):
    db = FeedbackStore(tmp_path / "feedback.sqlite")
    assert db.metrics()["requests"] == 0
    assert db.metrics()["average_latency_ms"] is None
    def record(i):
        db.record({"request_id": str(i), "selected_model": "model", "latency_ms": i + 1,
                   "cpu_percent": 40, "memory_mb": 100})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(20)))
    metrics = FeedbackStore(tmp_path / "feedback.sqlite").metrics()
    assert metrics["requests"] == 20
    assert metrics["average_latency_ms"] == 10.5
    assert metrics["model_counts"] == {"model": 20}
    assert len(metrics["recent"]) == 20
