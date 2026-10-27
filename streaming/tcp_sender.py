"""TCP sender: sends random or file data to the receiver and checks the ACK."""
import argparse
import hashlib
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming.tcp_receiver import HEADER, recv_exact  # noqa: E402


def send_payload(host, port, payload: bytes, timeout=config.SOCKET_TIMEOUT):
    """Send payload; return a result dict. Raises on connection problems."""
    digest = hashlib.sha256(payload).digest()
    t0 = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(HEADER.pack(len(payload), digest))
        view = memoryview(payload)
        for i in range(0, len(payload), config.TCP_BUFFER_SIZE):
            s.sendall(view[i:i + config.TCP_BUFFER_SIZE])
        ack = recv_exact(s, 2)
    seconds = time.perf_counter() - t0
    return {
        "bytes": len(payload),
        "seconds": seconds,
        "mbps": (len(payload) * 8 / seconds / 1e6) if seconds > 0 else 0.0,
        "acked": ack == b"OK",
    }


def main():
    p = argparse.ArgumentParser(description="TCP test sender")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.TCP_TEST_PORT)
    p.add_argument("--size-mb", type=float, default=5.0, help="random payload size in MB")
    p.add_argument("--file", help="send this file instead of random data")
    a = p.parse_args()

    try:
        payload = Path(a.file).read_bytes() if a.file else os.urandom(int(a.size_mb * 1_000_000))
    except OSError as e:
        print(f"ERROR: cannot read file: {e}")
        return 1

    print(f"Sending {len(payload)} bytes to {a.host}:{a.port}")
    try:
        r = send_payload(a.host, a.port, payload)
    except (ConnectionRefusedError, socket.timeout):
        print("ERROR: could not connect. No receiver is listening, or it did not respond. Start tcp_receiver first.")
        return 1
    except (ConnectionError, OSError) as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("Stopped by user")
        return 0

    print(f"Sent        : {r['bytes']} bytes in {r['seconds']:.3f} s ({r['mbps']:.1f} Mbps)")
    print(f"Receiver ACK: {'OK (data verified)' if r['acked'] else 'FAILED (data corrupted)'}")
    return 0 if r["acked"] else 2


if __name__ == "__main__":
    sys.exit(main())