"""UDP video receiver: measures loss, delay, jitter and stalls; optionally plays and saves."""
import argparse
import hashlib
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming.udp_common import HEADER, TYPE_DATA, TYPE_END, unpack_header  # noqa: E402
from streaming.video_common import (  # noqa: E402
    VideoStats, print_summary, start_player, stop_process)

TYPE_START = 2   # carries the sender's session start time in the timestamp field


def serve_once(host, port, play=False, save_path=None, wait_timeout=60.0,
               idle_timeout=3.0, grace=0.5, ready_event=None, stop_event=None):
    """Receive one UDP video stream and return a result dict.

    Ends when the END packet arrived and `grace` seconds passed, or when no packet
    arrived for `idle_timeout` seconds, or when stop_event is set.
    Raises TimeoutError if nothing arrives within wait_timeout.
    Returns None if stopped before any packet arrived.
    """
    stats = VideoStats()
    sha = hashlib.sha256()
    seen = set()
    duplicates = out_of_order = malformed = 0
    highest = -1
    expected = None
    end_time = None
    last_rx = None
    t0 = None
    peer = "-"
    player = start_player("UDP video stream") if play else None
    save = open(save_path, "wb") if save_path else None

    try:
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
                    data, addr = s.recvfrom(65535)
                except socket.timeout:
                    continue
                except ConnectionResetError:   # Windows quirk, safe to ignore
                    continue
                recv_ts = time.perf_counter()
                try:
                    ptype, seq, ts = unpack_header(data)
                except (ValueError, struct.error):
                    malformed += 1
                    continue
                last_rx = time.monotonic()

                if ptype == TYPE_START:
                    if t0 is None:
                        t0 = ts
                elif ptype == TYPE_END:
                    expected = seq
                    if end_time is None:
                        end_time = last_rx
                elif ptype == TYPE_DATA:
                    if seq in seen:
                        duplicates += 1
                        continue
                    seen.add(seq)
                    if peer == "-":
                        peer = f"{addr[0]}:{addr[1]}"
                    if seq < highest:
                        out_of_order += 1
                    else:
                        highest = seq
                    payload = data[HEADER.size:]
                    stats.on_chunk(recv_ts, ts, len(payload))
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
                else:
                    malformed += 1
    finally:
        if save is not None:
            save.close()
        stop_process(player, grace=3.0)

    if last_rx is None:
        return None

    from_end = expected is not None
    if expected is None:
        expected = highest + 1   # tail loss cannot be seen without the END packet
    unique = len(seen)
    lost = max(0, expected - unique)
    r = stats.summary(t0)
    r.update(
        expected=expected, expected_from_end_packet=from_end, unique=unique, lost=lost,
        loss_pct=(lost / expected * 100.0) if expected > 0 else 0.0,
        duplicates=duplicates, out_of_order=out_of_order, malformed=malformed,
        sha256=sha.hexdigest(), peer=peer,
    )
    return r


def main():
    p = argparse.ArgumentParser(description="UDP video receiver")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.VIDEO_UDP_PORT)
    p.add_argument("--play", action="store_true", help="show the video in an ffplay window")
    p.add_argument("--save", help="save the received stream to this .ts file")
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the first packet")
    p.add_argument("--idle", type=float, default=3.0, help="stop after this many silent seconds")
    a = p.parse_args()

    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    print(f"Listening on {a.host}:{a.port} (Ctrl+C to stop)")
    try:
        r = serve_once(a.host, a.port, a.play, a.save, a.wait, a.idle)
    except KeyboardInterrupt:
        print("Stopped by user")
        return 0
    except (TimeoutError, OSError) as e:
        print(f"ERROR: {e}")
        return 1

    if r is None:
        print("No packets received")
        return 1
    src = "END packet" if r["expected_from_end_packet"] else "highest sequence seen"
    print(f"Sender           : {r['peer']}")
    print(f"Expected chunks  : {r['expected']} (from {src})")
    print(f"Unique received  : {r['unique']}")
    print(f"Lost             : {r['lost']} ({r['loss_pct']:.2f} %)")
    print(f"Duplicates       : {r['duplicates']}")
    print(f"Out of order     : {r['out_of_order']}")
    print(f"Malformed        : {r['malformed']}")
    print_summary(r)
    print(f"SHA-256          : {r['sha256']}  (matches the sender only if nothing was lost or reordered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())