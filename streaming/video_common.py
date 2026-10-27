"""Shared video-streaming helpers: FFmpeg encoder/player, TCP framing, statistics."""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CHUNK_BYTES = config.UDP_PAYLOAD_SIZE   # 1316 bytes = 7 MPEG-TS packets
STALL_GAP_S = 0.25                      # arrival gap that counts as a stall
STREAM_TIMEOUT = 15.0                   # seconds of silence before a stream is abandoned
TCP_FRAME = struct.Struct("!IdH")       # sequence, send time, payload length


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"{name} not found on PATH. Close and reopen VS Code, then run '{name} -version'."
        )
    return path


def encoder_command(video, duration: float = 0.0) -> list:
    """FFmpeg command: real-time pacing (-re), identical settings for TCP and UDP."""
    cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
           "-re", "-i", str(video)]
    if duration and duration > 0:
        cmd += ["-t", str(duration)]
    cmd += [
        "-an",
        "-vf", f"scale={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT},fps={config.VIDEO_FPS}",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-b:v", config.VIDEO_BITRATE, "-maxrate", config.VIDEO_BITRATE, "-bufsize", "500k",
        "-g", str(config.VIDEO_FPS), "-pix_fmt", "yuv420p",
        "-flush_packets", "1",
        "-f", "mpegts", "-mpegts_flags", "+resend_headers", "pipe:1",
    ]
    return cmd


def _open_log(name: str):
    config.ensure_dirs()
    return open(config.LOGS_DIR / name, "wb")


def start_encoder(video, duration: float = 0.0):
    """Start FFmpeg writing the MPEG-TS stream to its stdout pipe."""
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")
    log = _open_log("ffmpeg_sender.log")
    proc = subprocess.Popen(encoder_command(video, duration), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=log)
    proc.log_file = log
    return proc


def start_player(title: str = "Received stream"):
    """Start ffplay reading the stream from its stdin. Returns None if ffplay is missing."""
    try:
        ffplay = find_tool("ffplay")
    except FileNotFoundError as e:
        print(f"WARNING: {e} Playback disabled.")
        return None
    log = _open_log("ffplay.log")
    cmd = [ffplay, "-hide_banner", "-loglevel", "error",
           "-fflags", "nobuffer", "-flags", "low_delay",
           "-probesize", "100000", "-analyzeduration", "500000",
           "-window_title", title, "-autoexit", "-f", "mpegts", "-i", "pipe:0"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    proc.log_file = log
    return proc


def stop_process(proc, grace: float = 0.0) -> None:
    """Close a child process. `grace` lets a player finish playing buffered video."""
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except OSError:
        pass
    if grace > 0:
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    try:
        if proc.stdout:
            proc.stdout.close()
    except OSError:
        pass
    log = getattr(proc, "log_file", None)
    if log:
        log.close()


class VideoStats:
    """Arrival statistics for video chunks. Jitter uses the RFC 3550 estimator."""

    def __init__(self, stall_gap: float = STALL_GAP_S):
        self.stall_gap = stall_gap
        self.chunks = 0
        self.bytes = 0
        self.first_t = None
        self.last_t = None
        self.latencies_ms = []
        self.jitter_ms = 0.0
        self._prev_transit = None
        self.stalls = 0
        self.stall_total_s = 0.0

    def on_chunk(self, recv_ts: float, send_ts: float, nbytes: int) -> None:
        if self.first_t is None:
            self.first_t = recv_ts
        else:
            gap = recv_ts - self.last_t
            if gap > self.stall_gap:
                self.stalls += 1
                self.stall_total_s += gap
        self.last_t = recv_ts
        self.chunks += 1
        self.bytes += nbytes

        transit = (recv_ts - send_ts) * 1000.0
        self.latencies_ms.append(transit)
        if self._prev_transit is not None:
            d = abs(transit - self._prev_transit)
            self.jitter_ms += (d - self.jitter_ms) / 16.0
        self._prev_transit = transit

    def summary(self, session_t0=None) -> dict:
        dur = (self.last_t - self.first_t) if self.first_t is not None else 0.0
        lat = self.latencies_ms
        startup = None
        if session_t0 is not None and self.first_t is not None:
            startup = (self.first_t - session_t0) * 1000.0
        return {
            "chunks": self.chunks,
            "bytes": self.bytes,
            "duration_s": dur,
            "mbps": (self.bytes * 8 / dur / 1e6) if dur > 0 else 0.0,
            "lat_avg_ms": (sum(lat) / len(lat)) if lat else 0.0,
            "lat_min_ms": min(lat) if lat else 0.0,
            "lat_max_ms": max(lat) if lat else 0.0,
            "jitter_ms": self.jitter_ms,
            "startup_delay_ms": startup,
            "stalls": self.stalls,
            "stall_total_ms": self.stall_total_s * 1000.0,
        }


def print_summary(r: dict) -> None:
    sd = r.get("startup_delay_ms")
    print(f"Chunks received  : {r['chunks']}")
    print(f"Data received    : {r['bytes']} bytes in {r['duration_s']:.3f} s ({r['mbps']:.3f} Mbps)")
    print(f"Latency (ms)     : avg {r['lat_avg_ms']:.3f}  min {r['lat_min_ms']:.3f}  max {r['lat_max_ms']:.3f}")
    print(f"Jitter (ms)      : {r['jitter_ms']:.3f}")
    print(f"Startup delay    : {'n/a' if sd is None else f'{sd:.0f} ms'}")
    print(f"Stalls (> {STALL_GAP_S * 1000:.0f} ms gap): {r['stalls']} (total {r['stall_total_ms']:.0f} ms)")