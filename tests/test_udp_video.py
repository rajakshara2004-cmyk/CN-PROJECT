"""Automated UDP video tests: clean round trip, loss detection, error handling."""
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming import udp_video_receiver, udp_video_sender  # noqa: E402
from streaming.udp_common import TYPE_DATA, TYPE_END, pack_packet  # noqa: E402
from streaming.udp_video_receiver import TYPE_START  # noqa: E402

HOST, PORT = config.HOST, config.VIDEO_UDP_PORT


def start_receiver(**kwargs):
    ready, out = threading.Event(), {}

    def run():
        try:
            out["r"] = udp_video_receiver.serve_once(HOST, PORT, ready_event=ready, **kwargs)
        except Exception as e:  # noqa: BLE001
            out["err"] = e
            ready.set()

    t = threading.Thread(target=run)
    t.start()
    assert ready.wait(5), "receiver did not start"
    return t, out


def test_roundtrip():
    config.ensure_dirs()
    save = config.RESULTS_DIR / "test_udp_video.ts"
    t, out = start_receiver(play=False, save_path=save, wait_timeout=20)
    s = udp_video_sender.stream(HOST, PORT, config.DEFAULT_VIDEO, duration=3)
    t.join(30)
    assert "err" not in out, out.get("err")
    r = out["r"]
    assert r["expected_from_end_packet"], "END packet was not received"
    assert r["unique"] == s["chunks"], (r["unique"], s["chunks"])
    assert r["lost"] == 0 and r["duplicates"] == 0 and r["out_of_order"] == 0, r
    assert r["sha256"] == s["sha256"], "SHA-256 mismatch"
    assert save.stat().st_size == r["bytes"] == s["bytes"], "saved file size mismatch"
    assert 2.0 <= r["duration_s"] <= 6.0, f"unexpected duration {r['duration_s']:.2f} s"
    assert r["startup_delay_ms"] is not None and r["startup_delay_ms"] > 0

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        res = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", str(save)],
            capture_output=True, text=True, timeout=30)
        lines = {ln.strip() for ln in res.stdout.splitlines() if ln.strip()}
        assert lines == {"h264,640,360"}, res.stdout + res.stderr
        probe = "saved file is H.264 640x360"
    else:
        probe = "ffprobe not found, file format not checked"
    print(f"PASS  video round trip ({r['bytes']} bytes, 0 lost, SHA-256 match, {probe})")
    print(f"      startup {r['startup_delay_ms']:.0f} ms, avg latency {r['lat_avg_ms']:.2f} ms, "
          f"jitter {r['jitter_ms']:.2f} ms, stalls {r['stalls']}")


def test_loss_detection():
    """Send chunks 0,1,2,2,5,4,7 and declare 8: expect 2 lost, 1 duplicate, 1 out of order."""
    t, out = start_receiver(idle_timeout=3.0, grace=0.5)
    addr = (HOST, PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(pack_packet(TYPE_START, 0, time.perf_counter()), addr)
        for seq in (0, 1, 2, 2, 5, 4, 7):
            s.sendto(pack_packet(TYPE_DATA, seq, time.perf_counter(), b"x" * 188), addr)
        for _ in range(3):
            s.sendto(pack_packet(TYPE_END, 8, time.perf_counter()), addr)
            time.sleep(0.01)
    t.join(10)
    assert "err" not in out, out.get("err")
    r = out["r"]
    assert r["unique"] == 6 and r["duplicates"] == 1 and r["out_of_order"] == 1, r
    assert r["expected"] == 8 and r["lost"] == 2, r
    assert abs(r["loss_pct"] - 25.0) < 1e-9, r
    assert r["bytes"] == 6 * 188, r
    assert r["startup_delay_ms"] is not None
    print("PASS  loss / duplicate / out-of-order detection")


def test_wait_timeout():
    try:
        udp_video_receiver.serve_once(HOST, PORT, wait_timeout=1.0)
    except TimeoutError:
        print("PASS  receiver timeout when no sender")
        return
    raise AssertionError("expected TimeoutError")


def test_stop_event():
    stop, out, ready = threading.Event(), {}, threading.Event()

    def run():
        out["r"] = udp_video_receiver.serve_once(HOST, PORT, ready_event=ready, stop_event=stop)

    t = threading.Thread(target=run)
    t.start()
    assert ready.wait(5)
    stop.set()
    t.join(5)
    assert not t.is_alive() and out["r"] is None
    print("PASS  receiver stops cleanly on stop signal")


def test_missing_video():
    try:
        udp_video_sender.stream(HOST, PORT, config.VIDEO_DIR / "does_not_exist.mp4")
    except FileNotFoundError:
        print("PASS  missing video file handled")
        return
    raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    test_roundtrip()
    test_loss_detection()
    test_wait_timeout()
    test_stop_event()
    test_missing_video()
    print("ALL UDP VIDEO TESTS PASSED")