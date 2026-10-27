"""TCP receiver: accepts one connection, receives data, verifies integrity."""
import argparse
import hashlib
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Header: total payload length (8 bytes) + SHA-256 digest (32 bytes)
HEADER = struct.Struct("!Q32s")


def recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ConnectionError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed before all data was received")
        buf.extend(chunk)
    return bytes(buf)


def serve_once(host, port, wait_timeout=60.0, ready_event=None, stop_event=None):
    """Wait for one sender, receive its payload, return a result dict.

    Returns None if stopped via stop_event before a sender connected.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        srv.settimeout(1.0)  # short timeout so Ctrl+C works on Windows
        if ready_event is not None:
            ready_event.set()

        deadline = time.monotonic() + wait_timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            if time.monotonic() > deadline:
                raise TimeoutError(f"No sender connected within {wait_timeout:.0f} s")
            try:
                conn, addr = srv.accept()
                break
            except socket.timeout:
                continue

        with conn:
            conn.settimeout(config.SOCKET_TIMEOUT)
            total, expected_digest = HEADER.unpack(recv_exact(conn, HEADER.size))

            sha = hashlib.sha256()
            received = 0
            t0 = time.perf_counter()
            while received < total:
                chunk = conn.recv(min(config.TCP_BUFFER_SIZE, total - received))
                if not chunk:
                    raise ConnectionError(
                        f"Connection closed after {received} of {total} bytes"
                    )
                sha.update(chunk)
                received += len(chunk)
            seconds = time.perf_counter() - t0

            integrity_ok = sha.digest() == expected_digest
            conn.sendall(b"OK" if integrity_ok else b"NO")

            return {
                "peer": f"{addr[0]}:{addr[1]}",
                "bytes": received,
                "seconds": seconds,
                "mbps": (received * 8 / seconds / 1e6) if seconds > 0 else 0.0,
                "integrity_ok": integrity_ok,
            }


def main():
    p = argparse.ArgumentParser(description="TCP test receiver")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.TCP_TEST_PORT)
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for a sender")
    a = p.parse_args()

    print(f"Listening on {a.host}:{a.port} (Ctrl+C to stop)")
    try:
        r = serve_once(a.host, a.port, a.wait)
    except KeyboardInterrupt:
        print("Stopped by user")
        return 0
    except (TimeoutError, ConnectionError, OSError) as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Sender      : {r['peer']}")
    print(f"Received    : {r['bytes']} bytes in {r['seconds']:.3f} s ({r['mbps']:.1f} Mbps)")
    print(f"Integrity   : {'OK' if r['integrity_ok'] else 'FAILED'}")
    return 0 if r["integrity_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())