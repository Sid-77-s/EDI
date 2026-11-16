import json
import time

from integration_live_demo import _LatencyTrace


def test_latency_trace_records_monotonic_correlated_spans(tmp_path):
    path = tmp_path / "latency.jsonl"
    trace = _LatencyTrace(path)
    correlation_id = "turn-test-1"
    start_ns = time.perf_counter_ns() - 2_000_000

    trace.record("stage_start", correlation_id=correlation_id)
    trace.record("stage_complete", correlation_id=correlation_id, started_ns=start_ns)
    trace.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["stage"] for record in records] == ["stage_start", "stage_complete"]
    assert all(record["correlation_id"] == correlation_id for record in records)
    assert records[0]["monotonic_ns"] <= records[1]["monotonic_ns"]
    assert records[1]["duration_ms"] >= 2.0
