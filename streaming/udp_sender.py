"""UDP sender: sends numbered, timestamped packets at a fixed rate."""
import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming.udp_common import HEADER, TYPE_DATA, TYPE_END, pack_packet  # noqa: E402


def send_stream(host, port, packets, rate_pps, packet_size=config.UDP_PAYLOAD_SIZE,
                stop_event=None):
    """Send `packets` data packets at `rate_pps`, then 5 END packets. Returns a dict."""
    if packets <= 0:
        raise ValueError("packets must be > 0")
    if rate_pps <= 0:
        raise ValueError("rate must be > 0")
    if not (HEADER.size <= packet_size <= 65507):
        raise ValueError(f"packet size must be between {HEADER.size} and 65507")

    payload = bytes(packet_size - HEADER.size)
    interval = 1.0 / rate_pps
    sent = 0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        t0 = time.perf_counter()
        for i in range(packets):
            if stop_event is not None and stop_event.is_set():
                break
            target = t0 + i * interval
            while True:                      # paced sending
                remaining = target - time.perf_counter()
                if remaining <= 0:
                    break
                if remaining > 0.002:
                    time.sleep(remaining - 0.001)
            try:
                s.sendto(pack_packet(TYPE_DATA, i, time.perf_counter(), payload), (host, port))
            except ConnectionResetError:     # Windows: ICMP "port unreachable"
                pass
            sent += 1
        seconds = time.perf_counter() - t0

        for _ in range(5):                   # END packet repeated in case one is lost
            try:
                s.sendto(pack_packet(TYPE_END, sent, time.perf_counter()), (host, port))
            except ConnectionResetError:
                pass
            time.sleep(0.01)

    return {
        "packets": sent,
        "bytes": sent * packet_size,
        "seconds": seconds,
        "mbps": (sent * packet_size * 8 / seconds / 1e6) if seconds > 0 else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="UDP test sender")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.UDP_TEST_PORT)
    p.add_argument("--packets", type=int, default=1000)
    p.add_argument("--rate", type=float, default=1000.0, help="packets per second")
    p.add_argument("--size", type=int, default=config.UDP_PAYLOAD_SIZE, help="bytes per packet")
    a = p.parse_args()

    print(f"Sending {a.packets} packets of {a.size} bytes at {a.rate:g} pkt/s to {a.host}:{a.port}")
    try:
        r = send_stream(a.host, a.port, a.packets, a.rate, a.size)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    except OSError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("Stopped by user")
        return 0

    print(f"Sent: {r['packets']} packets, {r['bytes']} bytes in {r['seconds']:.3f} s ({r['mbps']:.2f} Mbps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())