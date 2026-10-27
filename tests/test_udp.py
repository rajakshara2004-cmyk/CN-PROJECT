"""Automated UDP tests: loss/duplicate/reorder detection, normal run, timeouts."""
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming import udp_receiver, udp_sender  # noqa: E402
from streaming.udp_common import TYPE_DATA, TYPE_END, pack_packet  # noqa: E402

HOST, PORT = config.HOST, config.UDP_TEST_PORT


def start_receiver(**kwargs):
    ready, out = threading.Event(), {}

    def run():
        try:
            out["r"] = udp_receiver.serve_once(HOST, PORT, ready_event=ready, **kwargs)
        except Exception as e:  # noqa: BLE001
            out["err"] = e
            ready.set()

    t = threading.Thread(target=run)
    t.start()
    assert ready.wait(5), "receiver did not start"
    return t, out


def test_detection():
    """Send seqs 0,1,2,2,5,4,7 and declare 8 packets: expect 2 lost, 1 dup, 1 out of order."""
    t, out = start_receiver(idle_timeout=3.0, grace=0.5)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for seq in (0, 1, 2, 2, 5, 4, 7):
            s.sendto(pack_packet(TYPE_DATA, seq, time.perf_counter(), b"x" * 100), (HOST, PORT))
        for _ in range(3):
            s.sendto(pack_packet(TYPE_END, 8, time.perf_counter()), (HOST, PORT))
            time.sleep(0.01)
    t.join(10)
    assert "err" not in out, out.get("err")
    r = out["r"]
    assert r["received"] == 7, r
    assert r["unique"] == 6, r
    assert r["duplicates"] == 1, r
    assert r["out_of_order"] == 1, r
    assert r["expected"] == 8 and r["lost"] == 2, r
    assert abs(r["loss_pct"] - 25.0) < 1e-9, r
    print("PASS  loss / duplicate / out-of-order detection")


def test_normal_run():
    t, out = start_receiver(idle_timeout=3.0, grace=0.5)
    s = udp_sender.send_stream(HOST, PORT, packets=500, rate_pps=2000)
    t.join(10)
    assert "err" not in out, out.get("err")
    r = out["r"]
    assert s["packets"] == 500
    assert r["unique"] == 500 and r["lost"] == 0 and r["duplicates"] == 0, r
    print(f"PASS  500-packet run (jitter {r['jitter_ms']:.3f} ms, avg latency {r['lat_avg_ms']:.3f} ms)")


def test_wait_timeout():
    try:
        udp_receiver.serve_once(HOST, PORT, wait_timeout=1.0)
    except TimeoutError:
        print("PASS  receiver timeout when no sender")
        return
    raise AssertionError("expected TimeoutError")


def test_stop_event():
    stop = threading.Event()
    out = {}
    ready = threading.Event()

    def run():
        out["r"] = udp_receiver.serve_once(HOST, PORT, ready_event=ready, stop_event=stop)

    t = threading.Thread(target=run)
    t.start()
    assert ready.wait(5)
    stop.set()
    t.join(5)
    assert not t.is_alive() and out["r"] is None
    print("PASS  receiver stops cleanly on stop signal")


def test_bad_arguments():
    for kwargs in ({"packets": 0, "rate_pps": 10},
                   {"packets": 10, "rate_pps": 0},
                   {"packets": 10, "rate_pps": 10, "packet_size": 5}):
        try:
            udp_sender.send_stream(HOST, PORT, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    print("PASS  invalid sender arguments rejected")


if __name__ == "__main__":
    test_detection()
    test_normal_run()
    test_wait_timeout()
    test_stop_event()
    test_bad_arguments()
    print("ALL UDP TESTS PASSED")