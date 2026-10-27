"""Automated TCP tests: integrity for several sizes + no-receiver handling."""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming import tcp_receiver, tcp_sender  # noqa: E402


def roundtrip(size: int) -> None:
    ready, out = threading.Event(), {}

    def run():
        try:
            out["r"] = tcp_receiver.serve_once(
                config.HOST, config.TCP_TEST_PORT, 15, ready_event=ready
            )
        except Exception as e:  # noqa: BLE001
            out["err"] = e
            ready.set()

    t = threading.Thread(target=run)
    t.start()
    assert ready.wait(5), "receiver did not start"
    payload = os.urandom(size)
    s = tcp_sender.send_payload(config.HOST, config.TCP_TEST_PORT, payload, 5)
    t.join(15)
    assert "err" not in out, f"receiver error: {out.get('err')}"
    r = out["r"]
    assert r["bytes"] == size, f"size mismatch {r['bytes']} != {size}"
    assert r["integrity_ok"] and s["acked"], "integrity check failed"
    print(f"PASS  {size:>9} bytes  ({r['mbps']:.1f} Mbps)")


def refused() -> None:
    # Windows may report a closed port as a timeout instead of "refused".
    try:
        tcp_sender.send_payload(config.HOST, 5999, b"x", 5)
    except (ConnectionRefusedError, TimeoutError):
        print("PASS  no-receiver case handled")
        return
    raise AssertionError("expected a connection error")


if __name__ == "__main__":
    for size in (0, 1, 1000, 5_000_000):
        roundtrip(size)
    refused()
    print("ALL TCP TESTS PASSED")