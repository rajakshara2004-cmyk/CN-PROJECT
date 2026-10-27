"""Tests for metric calculations, counter parsing, CSV logging and the trial runner."""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streaming import metrics, run_experiment  # noqa: E402
from streaming.video_common import VideoStats  # noqa: E402


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_video_stats():
    """Hand-checked values: transit 10,14,10,12 ms -> jitter 0.25, 0.484375, 0.5791015625."""
    st = VideoStats(stall_gap=0.25)
    arrivals = [1.000, 1.100, 1.500, 1.600]
    transit_ms = [10, 14, 10, 12]
    for a, tm in zip(arrivals, transit_ms):
        st.on_chunk(a, a - tm / 1000.0, 1000)
    r = st.summary(session_t0=0.5)
    assert r["chunks"] == 4 and r["bytes"] == 4000
    assert close(r["lat_avg_ms"], 11.5) and close(r["lat_min_ms"], 10) and close(r["lat_max_ms"], 14)
    assert close(r["jitter_ms"], 0.5791015625), r["jitter_ms"]
    assert r["stalls"] == 1 and close(r["stall_total_ms"], 400.0, 1e-3), r
    assert close(r["startup_delay_ms"], 500.0, 1e-3), r["startup_delay_ms"]
    assert close(r["duration_s"], 0.6) and close(r["mbps"], 4000 * 8 / 0.6 / 1e6)
    assert VideoStats().summary()["startup_delay_ms"] is None
    print("PASS  latency, jitter, stall, startup and throughput calculations")


def test_parsers():
    snmp = ("Ip: Forwarding DefaultTTL InReceives\nIp: 1 64 1000\n"
            "Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens AttemptFails "
            "EstabResets CurrEstab InSegs OutSegs RetransSegs InErrs OutRsts InCsumErrors\n"
            "Tcp: 1 200 120000 -1 10 5 0 0 3 1000 900 42 0 1 0\n")
    assert metrics.parse_proc_snmp(snmp) == 42
    assert metrics.parse_proc_snmp("garbage") is None
    netstat = ("TCP Statistics for IPv4\n\n  Active Opens                        = 812\n"
               "  Segments Received                   = 51234\n"
               "  Segments Sent                       = 49876\n"
               "  Segments Retransmitted              = 77\n")
    assert metrics.parse_netstat_tcp(netstat) == 77
    assert metrics.parse_netstat_tcp("nothing here") is None
    print("PASS  TCP retransmission counter parsing")


def test_csv():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "experiments.csv"
        metrics.append_row({"protocol": "tcp", "trial": 1, "throughput_mbps": 1.1}, path)
        metrics.append_row({"protocol": "udp", "trial": 2, "loss_pct": 0.5}, path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == metrics.COLUMNS
        assert len(rows) == 2 and rows[0]["protocol"] == "tcp" and rows[1]["loss_pct"] == "0.5"
        assert rows[0]["loss_pct"] == ""   # unavailable values stay blank

        path.write_text("old_a,old_b\n1,2\n", encoding="utf-8")   # file with a different header
        metrics.append_row({"protocol": "tcp"}, path)
        assert len(list(Path(d).glob("experiments_old_*.csv"))) == 1, "old file was not kept"
        with open(path, newline="", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 1
    print("PASS  CSV logging (header, blanks, old-file protection)")


def test_trials():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "trials.csv"
        t = run_experiment.run_trial("tcp", 3, "test", 1, csv_path=path)
        u = run_experiment.run_trial("udp", 3, "test", 1, csv_path=path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for row, proto in zip(rows, ("tcp", "udp")):
        assert row["protocol"] == proto and row["completed"] == "True", row["error"]
        assert row["impairment_applied"] == "no"
        assert int(row["recv_bytes"]) > 0 and int(row["recv_bytes"]) == int(row["sent_bytes"]), row
        assert row["sha256_match"] == "True"
        assert 0.3 <= float(row["throughput_mbps"]) <= 3.0, row["throughput_mbps"]
        assert float(row["startup_delay_ms"]) > 0 and float(row["total_time_s"]) >= 2.5
    assert t["tcp_retrans_source"] != "" and (t["tcp_retrans_segments"] == "" or t["tcp_retrans_segments"] >= 0)
    assert u["lost_chunks"] == 0 and u["tcp_retrans_segments"] == ""
    print("PASS  TCP and UDP trials logged correctly")
    print(f"      TCP: {run_experiment.describe(t)}")
    print(f"      UDP: {run_experiment.describe(u)}")


if __name__ == "__main__":
    test_video_stats()
    test_parsers()
    test_csv()
    test_trials()
    print("ALL METRICS TESTS PASSED")