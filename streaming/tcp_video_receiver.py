"""TCP video receiver: measures the stream, optionally plays it and saves it."""
import argparse
import hashlib
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming.tcp_receiver import recv_exact  # noqa: E402
from streaming.video_common import (  # noqa: E402
    STREAM_TIMEOUT, TCP_FRAME, VideoStats, print_summary, start_player, stop_process)


def _recv_frame_header(conn):
    """Return the unpacked frame header, or None if the sender closed cleanly."""
    data = conn.recv(TCP_FRAME.size)
    if not data:
        return None
    if len(data) < TCP_FRAME.size:
        data += recv_exact(conn, TCP_FRAME.size - len(data))
    return TCP_FRAME.unpack(data)


def serve_once(host, port, play=False, save_path=None, wait_timeout=60.0,
               ready_event=None, stop_event=None):
    """Receive one video stream. Returns a result dict, or None if stopped before a sender came.

    Raises TimeoutError if no sender connects within wait_timeout.
    """
    stats = VideoStats()
    sha = hashlib.sha256()
    player = start_player("TCP video stream") if play else None
    save = open(save_path, "wb") if save_path else None
    error = None
    t0 = None
    peer = "-"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            srv.settimeout(1.0)   # short timeout so Ctrl+C works on Windows
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

            peer = f"{addr[0]}:{addr[1]}"
            with conn:
                conn.settimeout(STREAM_TIMEOUT)
                try:
                    (t0,) = struct.unpack("!d", recv_exact(conn, 8))
                    while not (stop_event is not None and stop_event.is_set()):
                        head = _recv_frame_header(conn)
                        if head is None:
                            break   # sender finished and closed the connection
                        _seq, send_ts, length = head
                        payload = recv_exact(conn, length)
                        recv_ts = time.perf_counter()
                        stats.on_chunk(recv_ts, send_ts, length)
                        sha.update(payload)
                        if save is not None:
                            save.write(payload)
                        if player is not None:
                            try:
                                player.stdin.write(payload)
                                player.stdin.flush()
                            except (OSError, ValueError):   # player window was closed
                                stop_process(player)
                                player = None
                except (ConnectionError, socket.timeout, struct.error) as e:
                    error = str(e) or type(e).__name__
    finally:
        if save is not None:
            save.close()
        stop_process(player, grace=3.0)

    r = stats.summary(t0)
    r.update(sha256=sha.hexdigest(), completed=error is None, error=error, peer=peer)
    return r


def main():
    p = argparse.ArgumentParser(description="TCP video receiver")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.VIDEO_TCP_PORT)
    p.add_argument("--play", action="store_true", help="show the video in an ffplay window")
    p.add_argument("--save", help="save the received stream to this .ts file")
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for a sender")
    a = p.parse_args()

    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    print(f"Listening on {a.host}:{a.port} (Ctrl+C to stop)")
    try:
        r = serve_once(a.host, a.port, a.play, a.save, a.wait)
    except KeyboardInterrupt:
        print("Stopped by user")
        return 0
    except (TimeoutError, OSError) as e:
        print(f"ERROR: {e}")
        return 1

    if r is None:
        print("Stopped before a sender connected")
        return 1
    print(f"Sender           : {r['peer']}")
    print_summary(r)
    print(f"SHA-256          : {r['sha256']}")
    print(f"Stream complete  : {'yes' if r['completed'] else 'NO - ' + str(r['error'])}")
    return 0 if r["completed"] else 2


if __name__ == "__main__":
    sys.exit(main())