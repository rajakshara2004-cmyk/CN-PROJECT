"""CSV logging of experiment results and TCP retransmission counters."""
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CSV_PATH = config.RESULTS_DIR / "experiments.csv"

COLUMNS = [
    "timestamp", "run_id", "protocol", "condition", "trial",
    "video", "duration_cfg_s", "bitrate", "width", "height", "fps", "chunk_bytes",
    "netem_delay_ms", "netem_jitter_ms", "netem_loss_pct", "netem_rate_mbit",
    "impairment_applied", "impairment_verified", "netem_qdisc", "platform",
    "sent_chunks", "sent_bytes", "recv_chunks", "recv_bytes",
    "total_time_s", "stream_duration_s", "throughput_mbps",
    "lat_avg_ms", "lat_min_ms", "lat_max_ms", "jitter_ms", "startup_delay_ms",
    "stalls", "stall_total_ms",
    "expected_chunks", "lost_chunks", "loss_pct", "duplicates", "out_of_order",
    "tcp_retrans_segments", "tcp_retrans_source",
    "sha256_match", "completed", "error",
]


def append_row(row: dict, path=CSV_PATH) -> None:
    """Append one result row. If an old file has a different header, it is renamed, not overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path.rename(path.with_name(f"{path.stem}_old_{stamp}{path.suffix}"))
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})


def parse_proc_snmp(text: str):
    """Linux: read RetransSegs from the contents of /proc/net/snmp. Returns int or None."""
    tcp_lines = [ln for ln in text.splitlines() if ln.startswith("Tcp:")]
    if len(tcp_lines) < 2:
        return None
    names = tcp_lines[0].split()[1:]
    values = tcp_lines[1].split()[1:]
    if "RetransSegs" not in names:
        return None
    try:
        return int(values[names.index("RetransSegs")])
    except (IndexError, ValueError):
        return None


def parse_netstat_tcp(text: str):
    """Windows: read 'Segments Retransmitted' from `netstat -s -p tcp`. Returns int or None."""
    m = re.search(r"Segments Retransmitted\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def read_tcp_retransmits():
    """Return (system-wide retransmitted segment count, source). Count is None if unavailable."""
    try:
        if sys.platform.startswith("linux"):
            value = parse_proc_snmp(Path("/proc/net/snmp").read_text())
            if value is not None:
                return value, "linux:/proc/net/snmp"
        elif sys.platform == "win32":
            out = subprocess.run(["netstat", "-s", "-p", "tcp"], capture_output=True,
                                 text=True, timeout=15).stdout
            value = parse_netstat_tcp(out)
            if value is not None:
                return value, "windows:netstat -s -p tcp"
    except (OSError, subprocess.SubprocessError):
        pass
    return None, "unavailable"