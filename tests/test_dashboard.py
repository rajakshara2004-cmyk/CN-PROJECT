"""Automated test for Streamlit dashboard data processing, charts, and summaries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard.app import load_dataset  # noqa: E402
from streaming import metrics  # noqa: E402


def test_dashboard_dataset_load():
    df = load_dataset()
    assert not df.empty, "dataset should not be empty"
    assert "protocol" in df.columns
    assert "throughput_mbps" in df.columns
    assert "condition" in df.columns
    print(f"PASS  dashboard dataset loaded successfully ({len(df)} rows)")


def test_dashboard_aggregations():
    df = load_dataset()
    completed = df[df["completed"].astype(str) == "True"]
    assert not completed.empty, "expected completed rows"
    grouped = completed.groupby(["condition", "protocol"])["throughput_mbps"].mean()
    assert len(grouped) > 0
    print(f"PASS  dashboard grouped aggregations verified ({len(grouped)} groups)")


if __name__ == "__main__":
    test_dashboard_dataset_load()
    test_dashboard_aggregations()
    print("ALL DASHBOARD TESTS PASSED")
