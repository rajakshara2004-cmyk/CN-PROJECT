"""Central configuration for the TCP vs UDP streaming project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

DEFAULT_VIDEO = VIDEO_DIR / "sample.mp4"

# Network settings (single laptop = loopback)
HOST = "127.0.0.1"
TCP_TEST_PORT = 5001
UDP_TEST_PORT = 5002
VIDEO_TCP_PORT = 5003
VIDEO_UDP_PORT = 5004

# Socket settings
TCP_BUFFER_SIZE = 65536
UDP_PAYLOAD_SIZE = 1316      # 7 x 188-byte MPEG-TS packets
SOCKET_TIMEOUT = 5.0         # seconds

# Video settings (same for TCP and UDP)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
VIDEO_FPS = 25
VIDEO_BITRATE = "1M"


def ensure_dirs() -> None:
    for d in (VIDEO_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Base dir     :", BASE_DIR)
    print("Video dir    :", VIDEO_DIR)
    print("Results dir  :", RESULTS_DIR)
    print("Logs dir     :", LOGS_DIR)
    print("TCP/UDP test :", TCP_TEST_PORT, UDP_TEST_PORT)
    print("Video ports  :", VIDEO_TCP_PORT, VIDEO_UDP_PORT)
    print("CONFIG OK")