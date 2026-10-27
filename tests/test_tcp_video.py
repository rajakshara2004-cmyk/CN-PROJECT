"""Automated TCP video tests: integrity, playable output, error handling."""
import shutil
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from streaming import tcp_video_receiver, tcp_video_sender  # noqa: E402

HOST, PORT = config.HOST, config.VIDEO_TCP_PORT


def test_roundtrip():
    config.ensure_dirs()
    save = config.RESULTS_DIR / "test_tcp_video.ts"
    ready, out = threading.Event(), {}

    def run():
        try:
            out["r"] = tcp_video_receiver.serve_once(
                HOST, PORT, play=False, save_path=save, wait_timeout=20, ready_event=ready)
        except Exception as e:  # noqa: BLE001
            out["err"] = e
            ready.set()

    t = threading.Thread(target=run)
    t.start()
    assert ready.wait(5), "receiver did not start"
    s = tcp_video_sender.stream(HOST, PORT, config.DEFAULT_VIDEO, duration=3)
    t.join(30)
    assert "err" not in out, out.get("err")
    r = out["r"]
    assert r["completed"], r["error"]
    assert r["bytes"] > 0 and r["bytes"] == s["bytes"], (r["bytes"], s["bytes"])
    assert r["sha256"] == s["sha256"], "SHA-256 mismatch"
    assert save.stat().st_size == r["bytes"], "saved file size mismatch"
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
    print(f"PASS  video round trip ({r['bytes']} bytes, SHA-256 match, {probe})")
    print(f"      startup {r['startup_delay_ms']:.0f} ms, avg latency {r['lat_avg_ms']:.2f} ms, "
          f"jitter {r['jitter_ms']:.2f} ms, stalls {r['stalls']}")


def test_no_receiver():
    try:
        tcp_video_sender.stream(HOST, 5998, config.DEFAULT_VIDEO, duration=1, connect_timeout=3)
    except (ConnectionRefusedError, TimeoutError):
        print("PASS  no-receiver case handled")
        return
    raise AssertionError("expected a connection error")


def test_missing_video():
    try:
        tcp_video_sender.stream(HOST, PORT, config.VIDEO_DIR / "does_not_exist.mp4")
    except FileNotFoundError:
        print("PASS  missing video file handled")
        return
    raise AssertionError("expected FileNotFoundError")


def test_wait_timeout():
    try:
        tcp_video_receiver.serve_once(HOST, PORT, wait_timeout=1.0)
    except TimeoutError:
        print("PASS  receiver wait timeout")
        return
    raise AssertionError("expected TimeoutError")


if __name__ == "__main__":
    test_roundtrip()
    test_no_receiver()
    test_missing_video()
    test_wait_timeout()
    print("ALL TCP VIDEO TESTS PASSED")