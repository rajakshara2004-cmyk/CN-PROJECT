"""Run repeatable TCP/UDP streaming trials and log every measurement to results/experiments.csv."""
import argparse
import platform
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming import (metrics, tcp_video_receiver, tcp_video_sender,  # noqa: E402
                       udp_video_receiver, udp_video_sender)
from streaming.video_common import CHUNK_BYTES  # noqa: E402

PROTOCOLS = {
    "tcp": (tcp_video_receiver, tcp_video_sender, config.VIDEO_TCP_PORT),
    "udp": (udp_video_receiver, udp_video_sender, config.VIDEO_UDP_PORT),
}


def _v(d, key):
    """Value from a result dict for the CSV: blank if missing, rounded if a float."""
    if d is None:
        return ""
    v = d.get(key, "")
    if v is None:
        return ""
    return round(v, 4) if isinstance(v, float) else v


def run_trial(protocol, duration=10.0, condition="normal", trial=1, video=None,
              host=config.HOST, run_id=None, csv_path=metrics.CSV_PATH, impairment=None):
    """Run one sender/receiver trial on this machine, append a CSV row, and return it.

    `impairment` is an optional dict (delay_ms, jitter_ms, loss_pct, rate_mbit, applied) that is
    only RECORDED. It does not change the network. Phase 6 applies and verifies real conditions.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}")
    receiver, sender, port = PROTOCOLS[protocol]
    video = Path(video) if video else config.DEFAULT_VIDEO
    imp = impairment or {}
    ready, stop, out = threading.Event(), threading.Event(), {}

    def run_receiver():
        try:
            out["r"] = receiver.serve_once(host, port, play=False, save_path=None,
                                           wait_timeout=30, ready_event=ready, stop_event=stop)
        except Exception as e:  # noqa: BLE001
            out["err"] = e
            ready.set()

    t = threading.Thread(target=run_receiver, daemon=True)
    t.start()
    if not ready.wait(10):
        stop.set()
        raise RuntimeError("receiver did not start")

    before, source = metrics.read_tcp_retransmits() if protocol == "tcp" else (None, "")
    s = send_err = None
    t_start = time.perf_counter()
    try:
        s = sender.stream(host, port, video, duration)
    except Exception as e:  # noqa: BLE001
        send_err = e
        stop.set()
    total_time = time.perf_counter() - t_start

    t.join((duration if duration > 0 else 60) + 60)
    if t.is_alive():
        stop.set()
        t.join(5)
        out.setdefault("err", TimeoutError("receiver did not finish"))
    after = metrics.read_tcp_retransmits()[0] if protocol == "tcp" else None

    r = out.get("r")
    error = ""
    if send_err is not None:
        error = f"sender: {send_err}"
    elif "err" in out:
        error = f"receiver: {out['err']}"
    elif r is None:
        error = "receiver got no data"
    elif protocol == "tcp" and not r["completed"]:
        error = f"receiver: {r['error']}"
    elif protocol == "udp" and not r["expected_from_end_packet"]:
        error = "END packet not received, tail loss not counted"

    retrans = ""
    if protocol == "tcp":
        if before is not None and after is not None:
            retrans = after - before
        else:
            source = "unavailable"
    u = r if protocol == "udp" else None   # UDP-only fields

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id or uuid.uuid4().hex[:8],
        "protocol": protocol, "condition": condition, "trial": trial,
        "video": video.name, "duration_cfg_s": duration,
        "bitrate": config.VIDEO_BITRATE, "width": config.VIDEO_WIDTH,
        "height": config.VIDEO_HEIGHT, "fps": config.VIDEO_FPS, "chunk_bytes": CHUNK_BYTES,
        "netem_delay_ms": imp.get("delay_ms", 0), "netem_jitter_ms": imp.get("jitter_ms", 0),
        "netem_loss_pct": imp.get("loss_pct", 0), "netem_rate_mbit": imp.get("rate_mbit", 0),
        "impairment_applied": "yes" if imp.get("applied") else "no", "impairment_verified": "yes" if imp.get("verified") else "no", "netem_qdisc": imp.get("qdisc", ""),
        "platform": f"{platform.system()} {platform.machine()}",
        "sent_chunks": _v(s, "chunks"), "sent_bytes": _v(s, "bytes"),
        "recv_chunks": _v(r, "chunks"), "recv_bytes": _v(r, "bytes"),
        "total_time_s": round(total_time, 3),
        "stream_duration_s": _v(r, "duration_s"), "throughput_mbps": _v(r, "mbps"),
        "lat_avg_ms": _v(r, "lat_avg_ms"), "lat_min_ms": _v(r, "lat_min_ms"),
        "lat_max_ms": _v(r, "lat_max_ms"), "jitter_ms": _v(r, "jitter_ms"),
        "startup_delay_ms": _v(r, "startup_delay_ms"),
        "stalls": _v(r, "stalls"), "stall_total_ms": _v(r, "stall_total_ms"),
        "expected_chunks": _v(u, "expected"), "lost_chunks": _v(u, "lost"),
        "loss_pct": _v(u, "loss_pct"), "duplicates": _v(u, "duplicates"),
        "out_of_order": _v(u, "out_of_order"),
        "tcp_retrans_segments": retrans, "tcp_retrans_source": source,
        "sha256_match": (s["sha256"] == r["sha256"]) if (s and r) else "",
        "completed": not error, "error": error,
    }
    metrics.append_row(row, csv_path)
    return row


def describe(row: dict) -> str:
    def f(key, fmt):
        v = row[key]
        return "n/a" if v == "" else format(v, fmt)

    text = (f"{f('throughput_mbps', '.2f')} Mbps, latency {f('lat_avg_ms', '.2f')} ms, "
            f"jitter {f('jitter_ms', '.2f')} ms, startup {f('startup_delay_ms', '.0f')} ms, "
            f"stalls {f('stalls', 'd')}")
    if row["protocol"] == "udp":
        text += f", lost {f('lost_chunks', 'd')} ({f('loss_pct', '.2f')} %)"
    else:
        text += f", retransmits {f('tcp_retrans_segments', 'd')}"
    return text + ("  OK" if row["completed"] else f"  FAILED: {row['error']}")


def main():
    p = argparse.ArgumentParser(description="Run TCP/UDP streaming trials and log results")
    p.add_argument("--protocol", choices=["tcp", "udp", "both"], default="both")
    p.add_argument("--duration", type=float, default=10.0,
                   help="seconds of video per trial (0 = whole video)")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--condition", default="normal", help="label stored with each row")
    p.add_argument("--video", default=str(config.DEFAULT_VIDEO))
    p.add_argument("--csv", default=str(metrics.CSV_PATH))
    a = p.parse_args()

    if a.trials < 1 or a.duration < 0:
        print("ERROR: --trials must be at least 1 and --duration must not be negative")
        return 1
    if not Path(a.video).is_file():
        print(f"ERROR: video file not found: {a.video}")
        return 1

    protocols = ["tcp", "udp"] if a.protocol == "both" else [a.protocol]
    total = a.trials * len(protocols)
    run_id = uuid.uuid4().hex[:8]
    print(f"Run {run_id}: {total} trials of {a.duration:g} s, condition '{a.condition}'")
    done = failures = 0
    try:
        for trial in range(1, a.trials + 1):
            for proto in protocols:   # alternate protocols so slow drift affects both equally
                done += 1
                row = run_trial(proto, a.duration, a.condition, trial, a.video,
                                run_id=run_id, csv_path=a.csv)
                failures += 0 if row["completed"] else 1
                print(f"[{done}/{total}] {proto} trial {trial}: {describe(row)}")
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopped by user")
    except PermissionError:
        print(f"ERROR: cannot write {a.csv}. Close it if it is open in Excel, then run again.")
        return 1
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Saved {done} rows to {a.csv} ({failures} failed)")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())