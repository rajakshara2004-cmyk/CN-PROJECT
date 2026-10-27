"""UDP receiver: detects loss, duplicates, reordering; measures latency and jitter."""
import argparse
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming.udp_common import TYPE_DATA, TYPE_END, unpack_header  # noqa: E402


class UdpStats:
    """Collects per-packet statistics. Jitter uses the RFC 3550 estimator."""

    def __init__(self):
        self.seen = set()
        self.received = 0        # every data packet that arrived (incl. duplicates)
        self.duplicates = 0
        self.out_of_order = 0
        self.malformed = 0
        self.highest = -1
        self.bytes = 0
        self.latencies_ms = []
        self.jitter_ms = 0.0
        self._prev_transit = None
        self.first_t = None
        self.last_t = None

    def on_packet(self, seq: int, send_ts: float, recv_ts: float, nbytes: int):
        self.received += 1
        self.bytes += nbytes
        if self.first_t is None:
            self.first_t = recv_ts
        self.last_t = recv_ts

        if seq in self.seen:
            self.duplicates += 1
            return
        self.seen.add(seq)

        if seq < self.highest:
            self.out_of_order += 1
        else:
            self.highest = seq

        transit = (recv_ts - send_ts) * 1000.0
        self.latencies_ms.append(transit)
        if self._prev_transit is not None:
            d = abs(transit - self._prev_transit)
            self.jitter_ms += (d - self.jitter_ms) / 16.0
        self._prev_transit = transit

    def summary(self, expected=None) -> dict:
        from_end = expected is not None
        if expected is None:
            expected = self.highest + 1   # tail loss cannot be seen without the END packet
        unique = len(self.seen)
        lost = max(0, expected - unique)
        duration = (self.last_t - self.first_t) if self.first_t is not None else 0.0
        lat = self.latencies_ms
        return {
            "expected": expected,
            "expected_from_end_packet": from_end,
            "received": self.received,
            "unique": unique,
            "lost": lost,
            "loss_pct": (lost / expected * 100.0) if expected > 0 else 0.0,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "malformed": self.malformed,
            "bytes": self.bytes,
            "duration_s": duration,
            "mbps": (self.bytes * 8 / duration / 1e6) if duration > 0 else 0.0,
            "lat_avg_ms": (sum(lat) / len(lat)) if lat else 0.0,
            "lat_min_ms": min(lat) if lat else 0.0,
            "lat_max_ms": max(lat) if lat else 0.0,
            "jitter_ms": self.jitter_ms,
        }


def serve_once(host, port, wait_timeout=60.0, idle_timeout=3.0, grace=0.5,
               ready_event=None, stop_event=None):
    """Receive one UDP stream and return a summary dict.

    Ends when: the END packet arrived and `grace` seconds passed (to collect
    late packets), or no packet arrived for `idle_timeout` seconds, or stop_event is set.
    Raises TimeoutError if nothing arrives within wait_timeout.
    Returns None if stopped before any packet arrived.
    """
    stats = UdpStats()
    expected = None
    end_time = None
    last_rx = None

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        s.bind((host, port))
        s.settimeout(0.2)
        if ready_event is not None:
            ready_event.set()
        started = time.monotonic()

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.monotonic()
            if last_rx is None:
                if now - started > wait_timeout:
                    raise TimeoutError(f"No packets received within {wait_timeout:.0f} s")
            else:
                if end_time is not None and now - end_time >= grace:
                    break
                if now - last_rx > idle_timeout:
                    break
            try:
                data, _addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            except ConnectionResetError:   # Windows quirk, safe to ignore
                continue
            recv_ts = time.perf_counter()
            try:
                ptype, seq, send_ts = unpack_header(data)
            except (ValueError, struct.error):
                stats.malformed += 1
                continue
            last_rx = time.monotonic()
            if ptype == TYPE_END:
                expected = seq
                if end_time is None:
                    end_time = last_rx
            elif ptype == TYPE_DATA:
                stats.on_packet(seq, send_ts, recv_ts, len(data))
            else:
                stats.malformed += 1

    if stats.received == 0:
        return None
    return stats.summary(expected)


def print_summary(r: dict) -> None:
    src = "END packet" if r["expected_from_end_packet"] else "highest sequence seen"
    print(f"Expected packets : {r['expected']} (from {src})")
    print(f"Received (all)   : {r['received']}   Unique: {r['unique']}")
    print(f"Lost             : {r['lost']} ({r['loss_pct']:.2f} %)")
    print(f"Duplicates       : {r['duplicates']}")
    print(f"Out of order     : {r['out_of_order']}")
    print(f"Malformed        : {r['malformed']}")
    print(f"Data received    : {r['bytes']} bytes in {r['duration_s']:.3f} s ({r['mbps']:.2f} Mbps)")
    print(f"Latency (ms)     : avg {r['lat_avg_ms']:.3f}  min {r['lat_min_ms']:.3f}  max {r['lat_max_ms']:.3f}")
    print(f"Jitter (ms)      : {r['jitter_ms']:.3f}")


def main():
    p = argparse.ArgumentParser(description="UDP test receiver")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.UDP_TEST_PORT)
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the first packet")
    p.add_argument("--idle", type=float, default=3.0, help="stop after this many silent seconds")
    a = p.parse_args()

    print(f"Listening on {a.host}:{a.port} (Ctrl+C to stop)")
    try:
        r = serve_once(a.host, a.port, a.wait, a.idle)
    except KeyboardInterrupt:
        print("Stopped by user")
        return 0
    except (TimeoutError, OSError) as e:
        print(f"ERROR: {e}")
        return 1

    if r is None:
        print("No packets received")
        return 1
    print_summary(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())