"""Print mean and standard deviation per condition and protocol from the results CSV."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streaming import metrics  # noqa: E402

COLS = ["throughput_mbps", "lat_avg_ms", "jitter_ms", "startup_delay_ms",
        "stalls", "loss_pct", "tcp_retrans_segments", "total_time_s"]


def main():
    p = argparse.ArgumentParser(description="Summarise repeated trials")
    p.add_argument("--csv", default=str(metrics.CSV_PATH))
    a = p.parse_args()

    path = Path(a.csv)
    if not path.exists():
        print(f"ERROR: {path} not found. Run streaming\\run_experiment.py first.")
        return 1
    df = pd.read_csv(path)
    ok = df[df["completed"].astype(str) == "True"]
    print(f"{len(df)} rows, {len(ok)} completed, {len(df) - len(ok)} failed (failed rows are excluded)")
    if ok.empty:
        return 1

    rows = []
    for (cond, proto), g in ok.groupby(["condition", "protocol"]):
        row = {"condition": cond, "protocol": proto, "trials": len(g)}
        for c in COLS:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if vals.empty:
                row[c] = "n/a"
            elif len(vals) > 1:
                row[c] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
            else:
                row[c] = f"{vals.mean():.3f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())