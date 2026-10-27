"""TCP video sender: encodes a video with FFmpeg in real time and streams it over TCP."""
import argparse
import hashlib
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming.video_common import (  # noqa: E402
    CHUNK_BYTES, STREAM_TIMEOUT, TCP_FRAME, start_encoder, stop_process)


def stream(host, port, video, duration=0.0, stop_event=None, connect_timeout=5.0):
    """Stream `video` to a TCP receiver. duration 0 = whole video. Returns a result dict."""
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    sha = hashlib.sha256()
    seq = 0
    total = 0
    first_send = None
    proc = None
    t0 = time.perf_counter()   # session start, used for the startup-delay measurement
    try:
        with socket.create_connection((host, port), timeout=connect_timeout) as s:
            s.settimeout(STREAM_TIMEOUT)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.sendall(struct.pack("!d", t0))
            proc = start_encoder(video, duration)
            while not (stop_event is not None and stop_event.is_set()):
                chunk = proc.stdout.read(CHUNK_BYTES)
                if not chunk:
                    break
                now = time.perf_counter()
                if first_send is None:
                    first_send = now
                s.sendall(TCP_FRAME.pack(seq, now, len(chunk)) + chunk)
                sha.update(chunk)
                total += len(chunk)
                seq += 1
    finally:
        stop_process(proc)

    stopped = stop_event is not None and stop_event.is_set()
    if seq == 0 and not stopped:
        raise RuntimeError("FFmpeg produced no video data. See logs\\ffmpeg_sender.log")
    seconds = (time.perf_counter() - first_send) if first_send is not None else 0.0
    return {
        "chunks": seq,
        "bytes": total,
        "seconds": seconds,
        "mbps": (total * 8 / seconds / 1e6) if seconds > 0 else 0.0,
        "sha256": sha.hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description="TCP video sender")
    p.add_argument("--host", default=config.HOST)
    p.add_argument("--port", type=int, default=config.VIDEO_TCP_PORT)
    p.add_argument("--video", default=str(config.DEFAULT_VIDEO))
    p.add_argument("--duration", type=float, default=0.0, help="seconds to stream (0 = whole video)")
    a = p.parse_args()

    print(f"Streaming {a.video} to {a.host}:{a.port} over TCP")
    try:
        r = stream(a.host, a.port, a.video, a.duration)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except (ConnectionRefusedError, TimeoutError):
        print("ERROR: could not connect. No receiver is listening, or it did not respond. "
              "Start tcp_video_receiver first.")
        return 1
    except ConnectionError as e:
        print(f"ERROR: connection lost: {e}")
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    except OSError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("Stopped by user")
        return 0

    print(f"Sent             : {r['chunks']} chunks, {r['bytes']} bytes in {r['seconds']:.3f} s ({r['mbps']:.3f} Mbps)")
    print(f"SHA-256          : {r['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())