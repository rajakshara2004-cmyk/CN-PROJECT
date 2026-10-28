"""Tests for application-level impairment proxies (UDP & TCP)."""
import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming import udp_receiver, udp_sender  # noqa: E402
from streaming.impairment import (CONDITIONS, TcpImpairmentProxy,  # noqa: E402
                                  UdpImpairmentProxy, get_impairment_config)
from streaming.udp_common import TYPE_DATA, TYPE_END, pack_packet  # noqa: E402


def test_udp_proxy_loss():
    """Verify UDP proxy drops approximately expected percentage of data packets."""
    listen_port = 5020
    dest_port = 5021
    ready, out = threading.Event(), {}

    def run_rx():
        try:
            out["r"] = udp_receiver.serve_once(config.HOST, dest_port, ready_event=ready,
                                                idle_timeout=2.0, grace=0.3)
        except Exception as e:  # noqa: BLE001
            out["err"] = e
            ready.set()

    t_rx = threading.Thread(target=run_rx, daemon=True)
    t_rx.start()
    assert ready.wait(5), "receiver did not start"

    with UdpImpairmentProxy(config.HOST, listen_port, config.HOST, dest_port,
                            loss_pct=25.0) as proxy:
        assert proxy.ready_event.wait(5), "proxy did not start"
        # Send 100 packets to proxy
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for i in range(100):
                s.sendto(pack_packet(TYPE_DATA, i, time.perf_counter(), b"p" * 100),
                         (config.HOST, listen_port))
                time.sleep(0.005)
            # Send 3 END packets
            for _ in range(3):
                s.sendto(pack_packet(TYPE_END, 100, time.perf_counter()),
                         (config.HOST, listen_port))
                time.sleep(0.01)

    t_rx.join(10)
    assert "err" not in out, out.get("err")
    r = out["r"]
    assert r is not None, "receiver got no data"
    # Expected 100 packets, around 25% dropped (allow reasonable binomial range: 10 to 40)
    assert 10 <= r["lost"] <= 40, f"unexpected loss count: {r['lost']}"
    assert proxy.stats["dropped"] == r["lost"]
    print(f"PASS  UDP proxy loss: {r['lost']} / {r['expected']} lost ({r['loss_pct']:.1f}%)")


def test_udp_proxy_delay():
    """Verify UDP proxy adds scheduled latency."""
    listen_port = 5022
    dest_port = 5023
    ready, out = threading.Event(), {}

    def run_rx():
        try:
            out["r"] = udp_receiver.serve_once(config.HOST, dest_port, ready_event=ready,
                                                idle_timeout=2.0, grace=0.3)
        except Exception as e:  # noqa: BLE001
            out["err"] = e
            ready.set()

    t_rx = threading.Thread(target=run_rx, daemon=True)
    t_rx.start()
    assert ready.wait(5)

    with UdpImpairmentProxy(config.HOST, listen_port, config.HOST, dest_port,
                            delay_ms=40.0) as proxy:
        assert proxy.ready_event.wait(5)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for i in range(10):
                s.sendto(pack_packet(TYPE_DATA, i, time.perf_counter(), b"x" * 100),
                         (config.HOST, listen_port))
                time.sleep(0.01)
            for _ in range(3):
                s.sendto(pack_packet(TYPE_END, 10, time.perf_counter()),
                         (config.HOST, listen_port))
                time.sleep(0.01)

    t_rx.join(10)
    r = out["r"]
    assert r is not None
    assert r["lat_avg_ms"] >= 35.0, f"latency too low: {r['lat_avg_ms']:.2f} ms"
    print(f"PASS  UDP proxy delay: avg latency {r['lat_avg_ms']:.2f} ms (target 40ms)")


def test_condition_configs():
    for name in CONDITIONS:
        cfg = get_impairment_config(name)
        assert cfg["name"] == name
        if name == "normal":
            assert cfg["applied"] is False
        else:
            assert cfg["applied"] is True
            assert cfg["qdisc"] == "application_proxy"
    print("PASS  condition configuration profiles verified")


if __name__ == "__main__":
    test_condition_configs()
    test_udp_proxy_loss()
    test_udp_proxy_delay()
    print("ALL IMPAIRMENT PROXY TESTS PASSED")
