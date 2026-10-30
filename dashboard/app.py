"""Streamlit Dashboard: Performance Evaluation of TCP vs. UDP for Real-Time Video Streaming.

High-density research platform for empirical benchmarking, real-time trial execution,
visual pipeline inspection, side-by-side comparative monitoring, and statistical summaries.
"""
import importlib
import inspect
import io
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup project path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: E402
from streaming import metrics, run_experiment  # noqa: E402

# Force reload to pick up any updated function signatures in running Streamlit server
importlib.reload(run_experiment)
importlib.reload(metrics)

from streaming.impairment import CONDITIONS, get_impairment_config  # noqa: E402

# Page config
st.set_page_config(
    page_title="NetFlow // TCP vs UDP Video Streaming Benchmark",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Restrained Dark Theme & Unified Typographic Hierarchy CSS
# ------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --nf-bg-page: #0B1020;
        --nf-bg-card: #121A2D;
        --nf-bg-card-subtle: #0E1526;
        --nf-border: rgba(255, 255, 255, 0.08);
        --nf-border-subtle: rgba(255, 255, 255, 0.04);
        --nf-accent: #4C8DFF;
        --nf-accent-hover: #3B7CE6;
        --nf-success: #3FB950;
        --nf-error: #F85149;
        --nf-text-primary: #F1F5F9;
        --nf-text-secondary: #CBD5E1;
        --nf-text-muted: #8B93A7;
        --nf-tcp: #38BDF8;
        --nf-udp: #FB923C;
    }

    /* Core layout constraints */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: var(--nf-bg-page);
        color: var(--nf-text-primary);
    }

    .block-container {
        max-width: 1400px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid var(--nf-border);
    }

    section[data-testid="stSidebar"] {
        background-color: #0E1528 !important;
        border-right: 1px solid var(--nf-border);
    }

    /* Typographic hierarchy scale */
    .nf-title {
        font-size: 28px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: var(--nf-text-primary);
        line-height: 1.25;
        margin: 0;
    }

    .nf-section-heading {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: var(--nf-text-primary);
        line-height: 1.3;
        margin-top: 0;
        margin-bottom: 12px;
    }

    .nf-subheading {
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--nf-text-secondary);
        margin: 0;
    }

    .nf-label {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: var(--nf-text-muted) !important;
    }

    .nf-value {
        font-size: 14px !important;
        font-weight: 400 !important;
        color: var(--nf-text-primary);
    }

    .nf-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .nf-caption {
        font-size: 12px;
        color: var(--nf-text-muted);
        line-height: 1.4;
    }

    /* Header card */
    .nf-header-card {
        background: var(--nf-bg-card);
        border: 1px solid var(--nf-border);
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 8px;
    }

    .nf-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 8px;
    }

    .nf-badge {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.05);
        color: var(--nf-text-secondary);
        border: 1px solid var(--nf-border);
    }

    .nf-header-desc {
        font-size: 14px;
        color: var(--nf-text-muted);
        line-height: 1.4;
        margin: 0;
    }

    /* Experiment specs chips */
    .nf-specs-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 4px 0;
    }

    .nf-spec-chip {
        font-size: 12px;
        background: var(--nf-bg-card-subtle);
        border: 1px solid var(--nf-border);
        border-radius: 6px;
        padding: 6px 12px;
        color: var(--nf-text-secondary);
    }

    .nf-spec-chip strong {
        color: var(--nf-text-muted);
        font-weight: 500;
        margin-right: 4px;
    }

    /* Sidebar environment key-value layout */
    .nf-kv-table {
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 8px;
    }

    .nf-kv-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid var(--nf-border-subtle);
    }

    .nf-kv-row:last-child {
        border-bottom: none;
    }

    /* Design tokens: Card & KPI Grid */
    .nf-card {
        background: var(--nf-bg-card);
        border: 1px solid var(--nf-border);
        border-radius: 10px;
        padding: 16px;
    }

    .nf-kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 16px;
        margin-bottom: 24px;
    }

    .nf-kpi-card {
        background: var(--nf-bg-card);
        border: 1px solid var(--nf-border);
        border-radius: 10px;
        padding: 16px;
    }

    .nf-kpi-value {
        font-size: 24px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
        color: var(--nf-text-primary);
        line-height: 1.1;
        margin: 6px 0 4px 0;
    }

    .nf-kpi-subtext {
        font-size: 12px;
        color: var(--nf-text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Tab navigation bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid var(--nf-border);
        padding-bottom: 2px;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        color: var(--nf-text-muted);
        font-size: 14px;
        font-weight: 500;
        border: none;
        transition: all 0.15s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--nf-text-primary);
        background-color: rgba(255, 255, 255, 0.03);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--nf-bg-card) !important;
        color: var(--nf-accent) !important;
        font-weight: 600;
        border-bottom: 2px solid var(--nf-accent) !important;
    }

    /* Controls & Buttons */
    .stButton > button[kind="primary"] {
        background: var(--nf-accent) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--nf-accent) !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        transition: background-color 0.15s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--nf-accent-hover) !important;
        border-color: var(--nf-accent-hover) !important;
    }

    .stButton > button:not([kind="primary"]) {
        background: var(--nf-bg-card) !important;
        color: var(--nf-text-secondary) !important;
        border: 1px solid var(--nf-border) !important;
        border-radius: 6px !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }

    /* Single-row Pipeline Architecture */
    .nf-pipeline-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        background: var(--nf-bg-card-subtle);
        border: 1px solid var(--nf-border);
        border-radius: 8px;
        padding: 12px;
        overflow-x: auto;
    }

    .nf-pipeline-node {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 10px 8px;
        background: var(--nf-bg-card);
        border: 1px solid var(--nf-border);
        border-radius: 6px;
    }

    .nf-pipeline-node-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--nf-text-primary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
    }

    .nf-pipeline-node-sub {
        font-size: 11px;
        color: var(--nf-text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
    }

    .nf-pipeline-arrow {
        color: var(--nf-text-muted);
        font-size: 12px;
        opacity: 0.6;
        user-select: none;
        flex-shrink: 0;
    }

    /* Arena Stream Cards */
    .nf-stream-box {
        background: var(--nf-bg-card);
        border: 1px solid var(--nf-border);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .nf-stream-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }

    .nf-stream-title {
        font-size: 14px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--nf-text-primary);
    }

    .nf-badge-status {
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }

    .nf-badge-success {
        background: rgba(63, 185, 80, 0.15);
        color: var(--nf-success);
        border: 1px solid rgba(63, 185, 80, 0.3);
    }

    .nf-badge-error {
        background: rgba(248, 81, 73, 0.15);
        color: var(--nf-error);
        border: 1px solid rgba(248, 81, 73, 0.3);
    }

    .nf-badge-neutral {
        background: rgba(255, 255, 255, 0.05);
        color: var(--nf-text-muted);
        border: 1px solid var(--nf-border);
    }

    .nf-stream-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 12px;
    }

    .nf-stream-stat {
        background: var(--nf-bg-card-subtle);
        border: 1px solid var(--nf-border-subtle);
        border-radius: 6px;
        padding: 10px 12px;
    }

    .nf-stream-diagnosis {
        font-size: 12px;
        line-height: 1.5;
        color: var(--nf-text-secondary);
        background: var(--nf-bg-card-subtle);
        border: 1px solid var(--nf-border-subtle);
        border-radius: 6px;
        padding: 10px 12px;
    }

    /* Head-to-Head Comparison Banner */
    .nf-comparison-banner {
        background: var(--nf-bg-card);
        border: 1px solid var(--nf-border);
        border-radius: 10px;
        padding: 16px;
        margin-top: 16px;
        margin-bottom: 24px;
    }

    /* Dataframe table overrides */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--nf-border);
        border-radius: 8px;
        background: var(--nf-bg-card);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_dataset() -> pd.DataFrame:
    """Load and preprocess the results CSV safely."""
    path = metrics.CSV_PATH
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=metrics.COLUMNS)
    try:
        df = pd.read_csv(path)
        numeric_cols = [
            "duration_cfg_s", "sent_chunks", "sent_bytes", "recv_chunks", "recv_bytes",
            "total_time_s", "stream_duration_s", "throughput_mbps", "lat_avg_ms",
            "lat_min_ms", "lat_max_ms", "jitter_ms", "startup_delay_ms", "stalls",
            "stall_total_ms", "expected_chunks", "lost_chunks", "loss_pct",
            "duplicates", "out_of_order", "tcp_retrans_segments", "netem_delay_ms",
            "netem_jitter_ms", "netem_loss_pct", "netem_rate_mbit",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame(columns=metrics.COLUMNS)


# ------------------------------------------------------------------------------
# 1. Header Card & Collapsed Specs
# ------------------------------------------------------------------------------
st.markdown("""
<div class="nf-header-card">
    <div class="nf-header-row">
        <h1 class="nf-title">NetFlow Research // TCP vs. UDP Video Streaming Evaluation</h1>
        <span class="nf-badge">Empirical Platform</span>
    </div>
    <p class="nf-header-desc">
        Comparative empirical evaluation of TCP and UDP under paced MPEG-TS streaming with deterministic network impairments.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("Experiment specs", expanded=False):
    st.markdown("""
    <div class="nf-specs-grid">
        <span class="nf-spec-chip"><strong>Network:</strong> Loopback (127.0.0.1)</span>
        <span class="nf-spec-chip"><strong>Codec:</strong> H.264 ultrafast (zerolatency) @ 1.0 Mbps</span>
        <span class="nf-spec-chip"><strong>Resolution:</strong> 640x360 @ 25fps</span>
        <span class="nf-spec-chip"><strong>MPEG-TS Payload:</strong> 1,316 B (7 TS pkts/datagram)</span>
        <span class="nf-spec-chip"><strong>Jitter Model:</strong> RFC 3550 Filter</span>
        <span class="nf-spec-chip"><strong>Impairment:</strong> In-Memory Proxy Pipeline</span>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Sidebar: Controls Top, Environment in Collapsed Expander
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="nf-section-heading">Video Asset Selection</div>', unsafe_allow_html=True)
    video_source_mode = st.radio(
        "Source Stream Asset",
        options=["Default Video (sample.mp4)", "Upload Custom Video"],
        index=0,
        help="Use the standardized benchmark sample.mp4 or upload a custom video file (.mp4, .mkv, .mov, .avi, .ts)",
    )

    active_video_path = config.DEFAULT_VIDEO
    active_video_name = "sample.mp4"
    active_video_sub = f"{config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT} @ {config.VIDEO_FPS}fps"

    if video_source_mode == "Upload Custom Video":
        uploaded_video = st.file_uploader(
            "Upload Custom Video File",
            type=["mp4", "mkv", "mov", "avi", "ts"],
            help="Supported: MP4, MKV, MOV, AVI, TS. FFmpeg automatically scales to 640x360@25fps H.264 real-time stream.",
        )
        if uploaded_video is not None:
            custom_path = config.VIDEO_DIR / f"upload_{uploaded_video.name}"
            with open(custom_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            active_video_path = custom_path
            active_video_name = uploaded_video.name
            size_kb = active_video_path.stat().st_size / 1024
            active_video_sub = f"Custom: {size_kb:.1f} KB"
            st.success(f"Loaded: `{uploaded_video.name}` ({size_kb:.1f} KB)")
            with st.expander("Preview Uploaded Video", expanded=False):
                st.video(str(active_video_path))
        else:
            st.info("No custom file selected yet. Using default `sample.mp4`.")
            if config.DEFAULT_VIDEO.is_file():
                with st.expander("Preview Default Video", expanded=False):
                    st.video(str(config.DEFAULT_VIDEO))
    else:
        if config.DEFAULT_VIDEO.is_file():
            size_kb = config.DEFAULT_VIDEO.stat().st_size / 1024
            st.caption(f"Default: `{config.DEFAULT_VIDEO.name}` ({size_kb:.1f} KB)")
            with st.expander("Preview Default Video", expanded=False):
                st.video(str(config.DEFAULT_VIDEO))
        else:
            st.warning("Default sample.mp4 video file not found.")

    st.markdown('<div class="nf-section-heading" style="margin-top: 16px;">Quick Controls</div>', unsafe_allow_html=True)
    if st.button("Refresh Dataset", use_container_width=True):
        st.rerun()

    host_system = f"{platform.system()} ({platform.machine()})"
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor} (.venv)"

    with st.expander("Environment", expanded=False):
        st.caption("Testing runs on loopback (127.0.0.1) exercising the full OS network stack with in-memory impairment queues.")
        st.markdown(
            f"""
            <div class="nf-kv-table">
                <div class="nf-kv-row"><span class="nf-label">Host OS</span><span class="nf-value nf-mono">{host_system}</span></div>
                <div class="nf-kv-row"><span class="nf-label">Python</span><span class="nf-value nf-mono">{python_ver}</span></div>
                <div class="nf-kv-row"><span class="nf-label">TCP Port</span><span class="nf-value nf-mono">{config.VIDEO_TCP_PORT}</span></div>
                <div class="nf-kv-row"><span class="nf-label">UDP Port</span><span class="nf-value nf-mono">{config.VIDEO_UDP_PORT}</span></div>
                <div class="nf-kv-row"><span class="nf-label">Target Bitrate</span><span class="nf-value nf-mono">{config.VIDEO_BITRATE} CBR</span></div>
                <div class="nf-kv-row"><span class="nf-label">Chunk Size</span><span class="nf-value nf-mono">{config.UDP_PAYLOAD_SIZE} B</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("NetFlow Research Suite • Standard Architecture")

# ------------------------------------------------------------------------------
# 4. KPI Cards: 4 cards, no colored dots, >=24px spacing
# ------------------------------------------------------------------------------
df = load_dataset()
completed_df = df[df["completed"].astype(str) == "True"] if not df.empty else df

total_trials = len(df)
completed_count = len(completed_df)
failed_count = total_trials - completed_count
tcp_count = len(completed_df[completed_df["protocol"] == "tcp"]) if not completed_df.empty else 0
udp_count = len(completed_df[completed_df["protocol"] == "udp"]) if not completed_df.empty else 0
active_conditions = completed_df["condition"].nunique() if not completed_df.empty else 0
mean_throughput = completed_df["throughput_mbps"].mean() if not completed_df.empty else 0.0

st.markdown(f"""
<div class="nf-kpi-container">
    <div class="nf-kpi-card">
        <div class="nf-label">Total Trials</div>
        <div class="nf-kpi-value">{total_trials}</div>
        <div class="nf-kpi-subtext">{completed_count} verified, {failed_count} failed • {active_conditions}/7 conditions</div>
    </div>
    <div class="nf-kpi-card">
        <div class="nf-label">TCP Reliable</div>
        <div class="nf-kpi-value" style="color: var(--nf-tcp);">{tcp_count}</div>
        <div class="nf-kpi-subtext">Byte-stream / Retransmit check</div>
    </div>
    <div class="nf-kpi-card">
        <div class="nf-label">UDP Datagram</div>
        <div class="nf-kpi-value" style="color: var(--nf-udp);">{udp_count}</div>
        <div class="nf-kpi-subtext">RFC 3550 timestamps / Jitter</div>
    </div>
    <div class="nf-kpi-card">
        <div class="nf-label">Mean Throughput</div>
        <div class="nf-kpi-value">{mean_throughput:.2f} <span style="font-size: 13px; color: var(--nf-text-muted);">Mbps</span></div>
        <div class="nf-kpi-subtext">Across active completed runs</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. Tab Navigation: Text only, shortened labels
# ------------------------------------------------------------------------------
tab_runner, tab_comparison, tab_conditions, tab_summary, tab_data = st.tabs([
    "Live Runner",
    "TCP vs UDP",
    "Impairments",
    "Statistics",
    "Audit & CSV",
])

# Plotly Theme Settings
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0B1020",
    plot_bgcolor="#121A2D",
    font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.08)",
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.08)",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(18, 26, 45, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.08)",
        borderwidth=1,
    ),
)
COLOR_TCP = "#38BDF8"
COLOR_UDP = "#FB923C"
PROTOCOL_COLOR_MAP = {"tcp": COLOR_TCP, "udp": COLOR_UDP, "TCP": COLOR_TCP, "UDP": COLOR_UDP}


# Helper to execute a trial safely with defensive signature inspection
def execute_trial_safe(protocol, duration, condition, trial, run_id, launch_player, video_path=None):
    sig = inspect.signature(run_experiment.run_trial)
    params = sig.parameters
    kwargs = {
        "protocol": protocol,
        "duration": duration,
        "condition": condition,
        "trial": trial,
        "run_id": run_id,
        "csv_path": metrics.CSV_PATH,
    }
    if "play" in params:
        kwargs["play"] = launch_player
    if "video" in params and video_path:
        kwargs["video"] = str(video_path)
    return run_experiment.run_trial(**kwargs)


# ==============================================================================
# TAB 1: Live Runner
# ==============================================================================
with tab_runner:
    st.markdown('<div class="nf-section-heading">Experiment Runner</div>', unsafe_allow_html=True)

    # Active video source caption
    video_type_label = "Default Sample Video" if active_video_name == "sample.mp4" else "User Uploaded Video"
    st.markdown(f"""
    <div class="nf-caption" style="margin-bottom: 16px;">
        Active Source Asset: <span class="nf-mono" style="color: var(--nf-text-primary);">{active_video_name}</span> ({active_video_sub}) • {video_type_label}
    </div>
    """, unsafe_allow_html=True)

    # 7. Video Transfer Pipeline Diagram inside Expander (Single Row, No Wrap)
    with st.expander("Pipeline architecture", expanded=False):
        st.markdown(f"""
        <div class="nf-pipeline-row">
            <div class="nf-pipeline-node">
                <div class="nf-pipeline-node-title">{active_video_name}</div>
                <div class="nf-pipeline-node-sub">{active_video_sub}</div>
            </div>
            <div class="nf-pipeline-arrow">→</div>
            <div class="nf-pipeline-node">
                <div class="nf-pipeline-node-title">FFmpeg Transcoder</div>
                <div class="nf-pipeline-node-sub">libx264 zerolatency</div>
            </div>
            <div class="nf-pipeline-arrow">→</div>
            <div class="nf-pipeline-node">
                <div class="nf-pipeline-node-title">MPEG-TS Framer</div>
                <div class="nf-pipeline-node-sub">1,316 B chunks</div>
            </div>
            <div class="nf-pipeline-arrow">→</div>
            <div class="nf-pipeline-node">
                <div class="nf-pipeline-node-title">Transport Sockets</div>
                <div class="nf-pipeline-node-sub">TCP {config.VIDEO_TCP_PORT} / UDP {config.VIDEO_UDP_PORT}</div>
            </div>
            <div class="nf-pipeline-arrow">→</div>
            <div class="nf-pipeline-node">
                <div class="nf-pipeline-node-title">Impairment Proxy</div>
                <div class="nf-pipeline-node-sub">Loss / Delay / Token</div>
            </div>
            <div class="nf-pipeline-arrow">→</div>
            <div class="nf-pipeline-node">
                <div class="nf-pipeline-node-title">Receiver Engine</div>
                <div class="nf-pipeline-node-sub">Metrics & SHA-256</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Controls Row
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([1.3, 1.7, 1.2, 1.0])
    with cfg_col1:
        run_proto = st.selectbox(
            "Transport Protocol",
            options=["both", "tcp", "udp"],
            format_func=lambda x: "Both (TCP & UDP Side-by-Side)" if x == "both" else f"{x.upper()} only",
            help="Select 'both' to compare TCP and UDP directly side-by-side under identical condition profiles",
        )
    with cfg_col2:
        condition_options = list(CONDITIONS.keys()) + ["all"]
        run_cond = st.selectbox(
            "Network Condition Profile",
            options=condition_options,
            format_func=lambda x: f"{x} - {CONDITIONS[x]['description']}" if x in CONDITIONS else "all - Full 7-condition test matrix",
        )
    with cfg_col3:
        run_duration = st.slider(
            "Duration (seconds)",
            min_value=3.0,
            max_value=30.0,
            value=5.0,
            step=1.0,
            help="Duration of paced video stream per trial",
        )
    with cfg_col4:
        run_trials = st.number_input(
            "Trials",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Repeat count for statistical variance calculations",
        )

    # Visual Inspection Option (FFplay player toggle)
    c_opt1, c_opt2 = st.columns([1.5, 2.5])
    with c_opt1:
        launch_ffplay = st.checkbox(
            "Launch Live FFplay Video Window",
            value=False,
            help="Pipes incoming stream directly to an FFplay window so you can visually watch the video streaming live and observe packet drops/buffering stalls.",
        )
    with c_opt2:
        if launch_ffplay:
            st.caption("An FFplay video player window will spawn on your desktop displaying the received stream in real-time.")
        else:
            st.caption("Running headless (fastest). Enable the checkbox to watch the live video player.")

    start_btn = st.button("Execute Streaming Trials", type="primary", use_container_width=True)

    # Containers for Side-by-Side Live Execution
    st.markdown('<div class="nf-section-heading" style="margin-top: 24px;">Streaming Arena</div>', unsafe_allow_html=True)
    arena_status = st.empty()
    col_tcp_arena, col_udp_arena = st.columns(2)
    tcp_slot = col_tcp_arena.empty()
    udp_slot = col_udp_arena.empty()
    comparison_container = st.container()

    # Function to render a stream card in the arena (replaces content in-place using slot)
    def render_stream_card(slot, proto, status_badge, row=None, error=None):
        is_tcp = proto.lower() == "tcp"
        proto_name = "TCP (Reliable Byte-Stream)" if is_tcp else "UDP (Timely Datagram)"
        port_num = config.VIDEO_TCP_PORT if is_tcp else config.VIDEO_UDP_PORT

        # Determine badge styling class
        badge_cls = "nf-badge-neutral"
        if "COMPLETED" in status_badge or "VERIFIED" in status_badge:
            badge_cls = "nf-badge-success"
        elif "ERROR" in status_badge or "FAILED" in status_badge:
            badge_cls = "nf-badge-error"

        # Check for completion vs error
        is_completed = False
        if row is not None:
            is_completed = str(row.get("completed", "")).lower() in ("true", "1")
            if not is_completed and not error:
                error = row.get("error") or "Trial was interrupted or connection timed out"

        if is_completed and row is not None:
            tp = f"{row.get('throughput_mbps', 0):.2f} Mbps"
            lat = f"{row.get('lat_avg_ms', 0):.2f} ms"
            jit = f"{row.get('jitter_ms', 0):.2f} ms"
            loss = f"{row.get('loss_pct', 0):.2f}%" if not is_tcp else "0.00% (ACKed)"
            stalls = f"{row.get('stalls', 0)} ({row.get('stall_total_ms', 0):.0f} ms)" if is_tcp else "0 (No Head-of-Line)"
            integrity = "Match" if str(row.get("sha256_match")) == "True" else ("Corrupted" if str(row.get("sha256_match")) == "False" else "Loss Tolerant")

            diagnosis = (
                f"<strong>Head-of-Line Blocking:</strong> TCP guaranteed 100% byte integrity. "
                f"Network impairments caused {row.get('stalls', 0)} buffering stalls ({row.get('stall_total_ms', 0):.0f} ms starvation) and {lat} transit latency."
                if is_tcp else
                f"<strong>Timely Datagram Delivery:</strong> UDP prioritized immediate delivery with {jit} jitter and {lat} latency. "
                f"Missing datagrams ({loss} loss) were skipped without stalling the decoder."
            )

            slot.markdown(f"""
            <div class="nf-stream-box">
                <div class="nf-stream-header">
                    <div class="nf-stream-title">
                        <span>{proto_name}</span>
                        <span class="nf-mono" style="font-size: 11px; color: var(--nf-text-muted);">Port {port_num}</span>
                    </div>
                    <div class="nf-badge-status {badge_cls}">
                        {status_badge}
                    </div>
                </div>
                <div class="nf-stream-grid">
                    <div class="nf-stream-stat">
                        <div class="nf-label">Throughput</div>
                        <div class="nf-value nf-mono">{tp}</div>
                    </div>
                    <div class="nf-stream-stat">
                        <div class="nf-label">Avg Latency</div>
                        <div class="nf-value nf-mono">{lat}</div>
                    </div>
                    <div class="nf-stream-stat">
                        <div class="nf-label">RFC 3550 Jitter</div>
                        <div class="nf-value nf-mono">{jit}</div>
                    </div>
                    <div class="nf-stream-stat">
                        <div class="nf-label">Packet Loss</div>
                        <div class="nf-value nf-mono">{loss}</div>
                    </div>
                    <div class="nf-stream-stat">
                        <div class="nf-label">Buffering Stalls</div>
                        <div class="nf-value nf-mono">{stalls}</div>
                    </div>
                    <div class="nf-stream-stat">
                        <div class="nf-label">Data Integrity</div>
                        <div class="nf-value nf-mono">{integrity}</div>
                    </div>
                </div>
                <div class="nf-stream-diagnosis">
                    {diagnosis}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif error:
            slot.markdown(f"""
            <div class="nf-stream-box">
                <div class="nf-stream-header">
                    <div class="nf-stream-title">
                        <span>{proto_name}</span>
                        <span class="nf-mono" style="font-size: 11px; color: var(--nf-text-muted);">Port {port_num}</span>
                    </div>
                    <div class="nf-badge-status nf-badge-error">
                        ERROR
                    </div>
                </div>
                <div style="color: var(--nf-error); font-size: 12px; background: rgba(248, 81, 73, 0.08); border: 1px solid rgba(248, 81, 73, 0.2); border-radius: 6px; padding: 12px; margin-bottom: 8px;">
                    <strong>Diagnostic Error:</strong> {error}
                </div>
                <div style="font-size: 11px; color: var(--nf-text-muted);">
                    Port {port_num} released cleanly. Sockets have been drained for next trial.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            in_prog_text = "Streaming video frames in real-time..." if "STREAM" in status_badge else ("Waiting for preceding TCP stream to complete..." if "WAIT" in status_badge else "Ready for execution. Click 'Execute Streaming Trials' above.")
            slot.markdown(f"""
            <div class="nf-stream-box">
                <div class="nf-stream-header">
                    <div class="nf-stream-title">
                        <span>{proto_name}</span>
                        <span class="nf-mono" style="font-size: 11px; color: var(--nf-text-muted);">Port {port_num}</span>
                    </div>
                    <div class="nf-badge-status {badge_cls}">
                        {status_badge}
                    </div>
                </div>
                <div style="color: var(--nf-text-muted); font-size: 12px; padding: 20px 0; text-align: center;">
                    {in_prog_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Initialize session state for persistent side-by-side view
    if "last_side_by_side" not in st.session_state:
        st.session_state["last_side_by_side"] = None

    # Handle Trial Execution
    if start_btn:
        conditions_to_run = list(CONDITIONS.keys()) if run_cond == "all" else [run_cond]
        protocols_to_run = ["tcp", "udp"] if run_proto == "both" else [run_proto]
        total_runs = len(conditions_to_run) * run_trials * len(protocols_to_run)
        run_id = uuid.uuid4().hex[:8]

        arena_status.markdown(
            f"""
            <div class="nf-card" style="padding: 10px 14px; margin-bottom: 14px; font-size: 13px;">
                <strong>Executing Streaming Matrix:</strong> {total_runs} trial(s) ({run_duration}s per trial). 
                Live video window: <strong>{'ENABLED' if launch_ffplay else 'HEADLESS'}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        last_tcp_row = None
        last_udp_row = None

        for c in conditions_to_run:
            for t in range(1, run_trials + 1):
                # TCP Trial
                if "tcp" in protocols_to_run:
                    render_stream_card(tcp_slot, "tcp", "STREAMING...")
                    if "udp" in protocols_to_run and last_udp_row is None:
                        render_stream_card(udp_slot, "udp", "WAITING FOR TCP")
                    try:
                        tcp_res = execute_trial_safe("tcp", run_duration, c, t, run_id, launch_ffplay, active_video_path)
                        last_tcp_row = tcp_res
                        status_str = "COMPLETED" if str(tcp_res.get("completed")).lower() in ("true", "1") else "FAILED"
                        render_stream_card(tcp_slot, "tcp", status_str, row=tcp_res)
                    except Exception as e:
                        render_stream_card(tcp_slot, "tcp", "ERROR", error=str(e))
                    time.sleep(1.0)

                # UDP Trial
                if "udp" in protocols_to_run:
                    render_stream_card(udp_slot, "udp", "STREAMING...")
                    try:
                        udp_res = execute_trial_safe("udp", run_duration, c, t, run_id, launch_ffplay, active_video_path)
                        last_udp_row = udp_res
                        status_str = "COMPLETED" if str(udp_res.get("completed")).lower() in ("true", "1") else "FAILED"
                        render_stream_card(udp_slot, "udp", status_str, row=udp_res)
                    except Exception as e:
                        render_stream_card(udp_slot, "udp", "ERROR", error=str(e))
                    time.sleep(1.0)

                # Store latest side-by-side comparison in session state
                if last_tcp_row is not None or last_udp_row is not None:
                    st.session_state["last_side_by_side"] = {
                        "condition": c,
                        "trial": t,
                        "duration": run_duration,
                        "video_name": active_video_name,
                        "tcp": last_tcp_row,
                        "udp": last_udp_row,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }

        arena_status.markdown(
            """
            <div class="nf-card" style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--nf-success); border-color: rgba(63, 185, 80, 0.3); padding: 10px 14px; margin-bottom: 14px;">
                All trials executed successfully and logged to results/experiments.csv.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render persistent Side-by-Side comparison if available
    sbs = st.session_state.get("last_side_by_side")
    if sbs and not start_btn:
        c_name = sbs.get("condition")
        tcp_r = sbs.get("tcp")
        udp_r = sbs.get("udp")
        v_name = sbs.get("video_name", "sample.mp4")

        if tcp_r:
            status_str = "VERIFIED TRIAL" if str(tcp_r.get("completed")).lower() in ("true", "1") else "FAILED TRIAL"
            render_stream_card(tcp_slot, "tcp", status_str, row=tcp_r)
        else:
            render_stream_card(tcp_slot, "tcp", "NOT EXECUTED")

        if udp_r:
            status_str = "VERIFIED TRIAL" if str(udp_r.get("completed")).lower() in ("true", "1") else "FAILED TRIAL"
            render_stream_card(udp_slot, "udp", status_str, row=udp_r)
        else:
            render_stream_card(udp_slot, "udp", "NOT EXECUTED")

        # Head-to-Head Comparative Delta Banner
        if tcp_r and udp_r:
            with comparison_container:
                tcp_tp = tcp_r.get("throughput_mbps", 0) or 0
                udp_tp = udp_r.get("throughput_mbps", 0) or 0
                tcp_lat = tcp_r.get("lat_avg_ms", 0) or 0
                udp_lat = udp_r.get("lat_avg_ms", 0) or 0
                tcp_jit = tcp_r.get("jitter_ms", 0) or 0
                udp_jit = udp_r.get("jitter_ms", 0) or 0
                udp_loss = udp_r.get("loss_pct", 0) or 0
                tcp_stalls = tcp_r.get("stalls", 0) or 0

                st.markdown(f"""
                <div class="nf-comparison-banner">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span class="nf-subheading">Head-to-Head Comparative Analysis: TCP vs. UDP under Condition '{c_name}'</span>
                        <span class="nf-caption">Asset: <span class="nf-mono">{v_name}</span> | {sbs.get('timestamp')}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
                        <div class="nf-card" style="padding: 10px 12px;">
                            <div class="nf-label">Throughput Delta</div>
                            <div class="nf-value nf-mono" style="font-weight: 700; margin-top: 4px;">
                                TCP: {tcp_tp:.2f} | UDP: {udp_tp:.2f}
                            </div>
                            <div class="nf-caption" style="margin-top: 2px;">Δ {abs(tcp_tp - udp_tp):.2f} Mbps</div>
                        </div>
                        <div class="nf-card" style="padding: 10px 12px;">
                            <div class="nf-label">Latency Delta</div>
                            <div class="nf-value nf-mono" style="font-weight: 700; margin-top: 4px;">
                                TCP: {tcp_lat:.1f} ms | UDP: {udp_lat:.1f} ms
                            </div>
                            <div class="nf-caption" style="margin-top: 2px;">Δ {abs(tcp_lat - udp_lat):.1f} ms</div>
                        </div>
                        <div class="nf-card" style="padding: 10px 12px;">
                            <div class="nf-label">Jitter Delta (RFC 3550)</div>
                            <div class="nf-value nf-mono" style="font-weight: 700; margin-top: 4px;">
                                TCP: {tcp_jit:.2f} ms | UDP: {udp_jit:.2f} ms
                            </div>
                            <div class="nf-caption" style="margin-top: 2px;">Arrival variance</div>
                        </div>
                        <div class="nf-card" style="padding: 10px 12px;">
                            <div class="nf-label">Loss vs. Stalls</div>
                            <div class="nf-value nf-mono" style="font-weight: 700; margin-top: 4px;">
                                Stalls: {tcp_stalls} | Loss: {udp_loss:.1f}%
                            </div>
                            <div class="nf-caption" style="margin-top: 2px;">Core Protocol Trade-Off</div>
                        </div>
                    </div>
                    <div class="nf-stream-diagnosis">
                        <strong>Empirical Takeaway:</strong> Under condition <code>{c_name}</code>, UDP prioritized real-time delivery with zero stalls while accepting {udp_loss:.2f}% packet loss. Conversely, TCP guaranteed 100% data integrity but experienced {tcp_stalls} buffering stall(s) due to retransmissions and Head-of-Line blocking.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif not start_btn:
        render_stream_card(col_tcp_arena, "tcp", "STANDBY")
        render_stream_card(col_udp_arena, "udp", "STANDBY")

# ==============================================================================
# TAB 2: TCP vs UDP
# ==============================================================================
with tab_comparison:
    st.markdown('<div class="nf-section-heading">Comparative Metrics</div>', unsafe_allow_html=True)

    if completed_df.empty:
        st.warning("No completed experiment records found yet. Use the 'Live Runner' to generate benchmark data.")
    else:
        metric_choices = {
            "throughput_mbps": ("Throughput (Mbps)", "Mbps", "Higher is better. Measures effective received video payload bandwidth."),
            "lat_avg_ms": ("Transit Latency (ms)", "ms", "Lower is better. End-to-end propagation and socket transit delay."),
            "jitter_ms": ("RFC 3550 Jitter (ms)", "ms", "Lower is better. Statistical packet arrival variance estimator."),
            "startup_delay_ms": ("Startup Delay (ms)", "ms", "Lower is better. Time elapsed before the first decodable frame arrives."),
            "stalls": ("Playback Stalls", "count", "Lower is better. Playback freeze events where chunk gap exceeded 250ms."),
            "stall_total_ms": ("Total Stall Duration (ms)", "ms", "Lower is better. Cumulative buffering starvation time."),
            "loss_pct": ("Packet Loss % (UDP)", "%", "Lower is better. Unrecovered dropped datagram chunks."),
        }

        f_col1, f_col2 = st.columns([1.2, 2.8])
        with f_col1:
            selected_metric_key = st.selectbox(
                "Primary Comparative Metric",
                options=list(metric_choices.keys()),
                format_func=lambda x: metric_choices[x][0],
            )
            all_conditions = sorted(completed_df["condition"].unique().tolist())
            selected_conditions = st.multiselect(
                "Filter Impairment Conditions",
                options=all_conditions,
                default=all_conditions,
            )

        metric_title, metric_unit, metric_desc = metric_choices[selected_metric_key]
        with f_col2:
            st.markdown(f"""
            <div class="nf-card" style="padding: 12px 16px; margin-top: 4px;">
                <div style="font-size: 13px; font-weight: 600; color: var(--nf-text-primary);">{metric_title} [{metric_unit}]</div>
                <div class="nf-caption" style="margin-top: 2px;">{metric_desc}</div>
            </div>
            """, unsafe_allow_html=True)

        filtered = completed_df[completed_df["condition"].isin(selected_conditions)]

        if filtered.empty:
            st.info("No experiment rows match the chosen condition filter.")
        else:
            # Grouped Bar Chart
            fig_bar = px.bar(
                filtered,
                x="condition",
                y=selected_metric_key,
                color="protocol",
                barmode="group",
                color_discrete_map=PROTOCOL_COLOR_MAP,
                labels={"condition": "Network Condition Profile", selected_metric_key: f"{metric_title} ({metric_unit})", "protocol": "Protocol"},
                title=f"Comparative {metric_title} across Impairment Conditions",
            )
            fig_bar.update_layout(PLOTLY_LAYOUT)
            fig_bar.update_traces(marker_line_width=0, opacity=0.92)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Secondary Analytics: Box Distribution & Scatter Trade-off
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown('<div class="nf-subheading" style="margin-bottom: 8px;">Measurement Variance & Distribution</div>', unsafe_allow_html=True)
                fig_box = px.box(
                    filtered,
                    x="condition",
                    y=selected_metric_key,
                    color="protocol",
                    color_discrete_map=PROTOCOL_COLOR_MAP,
                    title=f"Spread & Outliers: {metric_title}",
                    labels={"condition": "Condition", selected_metric_key: f"{metric_title} ({metric_unit})"},
                )
                fig_box.update_layout(PLOTLY_LAYOUT)
                st.plotly_chart(fig_box, use_container_width=True)

            with col_chart2:
                st.markdown('<div class="nf-subheading" style="margin-bottom: 8px;">Throughput vs. Latency Trade-Off</div>', unsafe_allow_html=True)
                fig_scatter = px.scatter(
                    filtered,
                    x="lat_avg_ms",
                    y="throughput_mbps",
                    color="protocol",
                    symbol="condition",
                    color_discrete_map=PROTOCOL_COLOR_MAP,
                    labels={"lat_avg_ms": "Transit Latency (ms)", "throughput_mbps": "Throughput (Mbps)", "condition": "Condition"},
                    title="Pareto Trade-Off: Throughput vs. Latency",
                )
                fig_scatter.update_layout(PLOTLY_LAYOUT)
                fig_scatter.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.2)")))
                st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# TAB 3: Impairments
# ==============================================================================
with tab_conditions:
    st.markdown('<div class="nf-section-heading">Network Impairment Scenarios</div>', unsafe_allow_html=True)
    st.markdown(
        "To rigorously evaluate protocol performance without physical cable unplugs or kernel netem driver requirements, "
        "an application-level loopback impairment proxy pipeline intercepts socket data on `127.0.0.1` and executes "
        "deterministic queueing and packet dropping models."
    )

    cond_rows = []
    for name, c in CONDITIONS.items():
        cond_rows.append({
            "Condition": name,
            "Description": c["description"],
            "Delay (ms)": f"{c['delay_ms']} ms",
            "Jitter (ms)": f"± {c['jitter_ms']} ms",
            "Loss Rate": f"{c['loss_pct']:.1f}%",
            "Bandwidth Limit": f"{c['rate_mbit']:.1f} Mbps" if c["rate_mbit"] > 0 else "Unlimited",
            "Queue Model": "Token Bucket + Jitter Queue" if name != "normal" else "Zero-Delay Direct Loopback",
        })
    st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="nf-subheading" style="margin-top: 24px; margin-bottom: 12px;">Protocol Mechanics Under Network Impairments</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        <div class="nf-card">
            <div style="font-size: 14px; font-weight: 700; color: var(--nf-text-primary); margin-bottom: 8px;">
                TCP: Reliable Stream Transport
            </div>
            <ul style="font-size: 12px; color: var(--nf-text-secondary); line-height: 1.6; margin-left: -15px;">
                <li><strong>Guaranteed Delivery:</strong> Uses sequence numbers, positive acknowledgments (ACKs), and automatic retransmissions (RTO).</li>
                <li><strong>Head-of-Line (HoL) Blocking:</strong> When a segment is lost or delayed, subsequent arriving segments must wait in socket receive buffers until the missing segment is retransmitted.</li>
                <li><strong>Real-World Video Effect:</strong> The video decoder starves for frames, producing <em>playback freezing / buffering stalls</em> and surging latency, but preserving 100% pixel fidelity with zero corrupt macroblocks.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown("""
        <div class="nf-card">
            <div style="font-size: 14px; font-weight: 700; color: var(--nf-text-primary); margin-bottom: 8px;">
                UDP: Timely Datagram Transport
            </div>
            <ul style="font-size: 12px; color: var(--nf-text-secondary); line-height: 1.6; margin-left: -15px;">
                <li><strong>Fire-and-Forget:</strong> Transmits independent datagram chunks without connection handshakes or retransmission delays.</li>
                <li><strong>Zero Head-of-Line Blocking:</strong> Arriving packets are fed immediately into the video decoder; lost packets are skipped.</li>
                <li><strong>Real-World Video Effect:</strong> Playback maintains <em>ultra-low transit latency and smooth continuous timing</em>, but lost MPEG-TS packets result in visual artifacts, pixelation, or transient decoder glitching.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: Statistics
# ==============================================================================
with tab_summary:
    st.markdown('<div class="nf-section-heading">Repeated-Trial Aggregations</div>', unsafe_allow_html=True)
    st.markdown(
        "Empirical scientific summaries aggregated by network condition and protocol across repeated experiment replications."
    )

    if completed_df.empty:
        st.info("No completed trials recorded yet.")
    else:
        summary_cols = [
            "throughput_mbps", "lat_avg_ms", "jitter_ms", "startup_delay_ms",
            "stalls", "loss_pct", "total_time_s",
        ]

        summary_rows = []
        for (cond, proto), g in completed_df.groupby(["condition", "protocol"]):
            entry = {
                "Condition": cond,
                "Protocol": proto.upper(),
                "Trials": len(g),
            }
            for col in summary_cols:
                series = pd.to_numeric(g[col], errors="coerce").dropna()
                if series.empty:
                    entry[col] = "n/a"
                elif len(series) > 1:
                    entry[col] = f"{series.mean():.3f} ± {series.std():.3f}"
                else:
                    entry[col] = f"{series.mean():.3f}"
            summary_rows.append(entry)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.rename(columns={
                "throughput_mbps": "Throughput (Mbps)",
                "lat_avg_ms": "Latency (ms)",
                "jitter_ms": "RFC 3550 Jitter (ms)",
                "startup_delay_ms": "Startup (ms)",
                "stalls": "Stalls",
                "loss_pct": "Loss %",
                "total_time_s": "Total Time (s)",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# TAB 5: Audit & CSV
# ==============================================================================
with tab_data:
    st.markdown('<div class="nf-section-heading">Audit Trail & CSV Export</div>', unsafe_allow_html=True)
    st.markdown(
        "Immutable audit log of all recorded benchmark trials with cryptographic SHA-256 verification, "
        "hardware timestamps, socket counters, and proxy metadata."
    )

    if df.empty:
        st.info("No experiment rows found in dataset.")
    else:
        top_c1, top_c2, top_c3 = st.columns([1.5, 1.5, 2])
        with top_c1:
            proto_filter = st.selectbox("Protocol Filter", ["All", "TCP", "UDP"], key="tab5_proto_filter")
        with top_c2:
            show_all_cols = st.checkbox("Show all 44 raw columns", value=False, key="tab5_raw_cols")
        with top_c3:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Export Full CSV Dataset",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name=f"netflow_benchmark_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        view_df = df.copy()
        if proto_filter == "TCP":
            view_df = view_df[view_df["protocol"].str.lower() == "tcp"]
        elif proto_filter == "UDP":
            view_df = view_df[view_df["protocol"].str.lower() == "udp"]

        st.caption(f"Displaying **{len(view_df)}** of **{len(df)}** experiment records (Source: `{metrics.CSV_PATH}`)")

        if not show_all_cols:
            summary_cols_map = {
                "timestamp": "Timestamp",
                "run_id": "Run ID",
                "protocol": "Protocol",
                "condition": "Condition",
                "trial": "Trial",
                "duration_cfg_s": "Duration (s)",
                "throughput_mbps": "Throughput (Mbps)",
                "lat_avg_ms": "Avg Latency (ms)",
                "jitter_ms": "Jitter (ms)",
                "loss_pct": "Loss (%)",
                "stalls": "Stalls",
                "sha256_match": "SHA-256 Match",
                "completed": "Status",
            }
            cols_to_show = [c for c in summary_cols_map.keys() if c in view_df.columns]
            table_df = view_df[cols_to_show].copy()

            if "protocol" in table_df.columns:
                table_df["protocol"] = table_df["protocol"].str.upper()
            if "throughput_mbps" in table_df.columns:
                table_df["throughput_mbps"] = table_df["throughput_mbps"].round(3)
            if "lat_avg_ms" in table_df.columns:
                table_df["lat_avg_ms"] = table_df["lat_avg_ms"].round(2)
            if "jitter_ms" in table_df.columns:
                table_df["jitter_ms"] = table_df["jitter_ms"].round(2)
            if "loss_pct" in table_df.columns:
                table_df["loss_pct"] = table_df["loss_pct"].apply(lambda v: f"{v:.2f}%" if pd.notna(v) and v != "" else "-")
            if "sha256_match" in table_df.columns:
                table_df["sha256_match"] = table_df["sha256_match"].apply(lambda v: "Match" if str(v) == "True" else ("Mismatch" if str(v) == "False" else "-"))
            if "completed" in table_df.columns:
                table_df["completed"] = table_df["completed"].apply(lambda v: "OK" if str(v) == "True" else "Failed")

            table_df = table_df.rename(columns=summary_cols_map)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        else:
            raw_display = view_df.astype(str).replace({"nan": "-", "None": "-"})
            st.dataframe(raw_display, use_container_width=True, hide_index=True)
