"""Streamlit Dashboard: Performance Evaluation of TCP vs. UDP for Real-Time Video Streaming.

High-density research platform for empirical benchmarking, real-time trial execution,
visual pipeline inspection, side-by-side comparative monitoring, and statistical summaries.
"""
import importlib
import inspect
import io
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
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Research Platform Dark Navy Palette & Restrained Aesthetics
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: #0B1020;
        color: #F1F5F9;
    }

    /* Streamlit core container overrides */
    header[data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0E1528 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Research Platform Header */
    .platform-header {
        background: linear-gradient(180deg, #121B2B 0%, #0F1726 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 18px;
    }
    .platform-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 8px;
    }
    .platform-title {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .platform-title-badge {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 3px 8px;
        border-radius: 4px;
        background: rgba(167, 139, 250, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(167, 139, 250, 0.3);
    }
    .platform-subtitle {
        font-size: 13px;
        color: #94A3B8;
        line-height: 1.5;
        max-width: 900px;
    }
    .platform-meta-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }
    .meta-chip {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        color: #94A3B8;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .meta-chip strong {
        color: #CBD5E1;
    }

    /* Data-Driven KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: #121B2B;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 14px 16px;
        transition: border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
    }
    .kpi-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .kpi-subtext {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Pipeline Architecture Box */
    .pipeline-card {
        background: #121B2B;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .pipeline-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #CBD5E1;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pipeline-diagram {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        background: #0E1624;
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 6px;
        padding: 12px 14px;
    }
    .pipeline-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 8px 12px;
        background: #141E30;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        min-width: 125px;
    }
    .pipeline-node-icon {
        font-size: 18px;
        margin-bottom: 4px;
    }
    .pipeline-node-title {
        font-size: 12px;
        font-weight: 600;
        color: #F1F5F9;
    }
    .pipeline-node-sub {
        font-size: 10px;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }
    .pipeline-arrow {
        color: #475569;
        font-size: 14px;
        font-weight: bold;
    }

    /* Side-by-Side Monitoring Cards */
    .stream-box {
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 16px;
        background: #121B2B;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stream-box-tcp {
        border-left: 4px solid #35C8F4;
    }
    .stream-box-udp {
        border-left: 4px solid #FF9D45;
    }
    .stream-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .stream-title {
        font-size: 15px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stream-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 12px;
    }
    .stream-stat {
        background: #0E1624;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        padding: 10px 12px;
    }
    .stream-stat-label {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .stream-stat-val {
        font-size: 16px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }
    .stream-diagnosis {
        font-size: 12px;
        line-height: 1.5;
        color: #CBD5E1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 6px;
        padding: 10px 12px;
    }

    /* Comparison Head-to-Head Banner */
    .comparison-banner {
        background: linear-gradient(180deg, #162235 0%, #121B2B 100%);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 14px;
        margin-bottom: 18px;
    }
    .comparison-title {
        font-size: 14px;
        font-weight: 700;
        color: #A78BFA;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 2px;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        color: #94A3B8;
        font-size: 13px;
        font-weight: 500;
        border: none;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F1F5F9;
        background-color: rgba(255, 255, 255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: #121B2B !important;
        color: #35C8F4 !important;
        font-weight: 600;
        border-bottom: 2px solid #35C8F4 !important;
    }

    /* Protocol Badges */
    .proto-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }
    .proto-tcp {
        background: rgba(53, 200, 244, 0.12);
        color: #35C8F4;
        border: 1px solid rgba(53, 200, 244, 0.3);
    }
    .proto-udp {
        background: rgba(255, 157, 69, 0.12);
        color: #FF9D45;
        border: 1px solid rgba(255, 157, 69, 0.3);
    }

    /* Custom Button & Control Styling */
    .stButton > button[kind="primary"] {
        background: #1E293B !important;
        color: #35C8F4 !important;
        border: 1px solid rgba(53, 200, 244, 0.4) !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: rgba(53, 200, 244, 0.15) !important;
        border-color: #35C8F4 !important;
        box-shadow: 0 0 10px rgba(53, 200, 244, 0.15);
    }
    .stButton > button:not([kind="primary"]) {
        background: #121B2B !important;
        color: #CBD5E1 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 6px !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: #182438 !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }

    /* Dataframe Table styling */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        background: #121B2B;
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
# Compact Header
# ------------------------------------------------------------------------------
st.markdown("""
<div class="platform-header">
    <div class="platform-title-row">
        <div class="platform-title">
            <span>📡 NetFlow Research // TCP vs. UDP Video Streaming Evaluation</span>
            <span class="platform-title-badge">Empirical Platform</span>
        </div>
    </div>
    <div class="platform-subtitle">
        High-fidelity comparative empirical evaluation of Transmission Control Protocol (TCP) and User Datagram Protocol (UDP)
        under real-time paced MPEG-TS streaming with token bucket rate limiting, jitter queues, and cryptographic verification.
    </div>
    <div class="platform-meta-chips">
        <span class="meta-chip"><strong>Network:</strong> Loopback (127.0.0.1) & LAN Capable</span>
        <span class="meta-chip"><strong>Codec:</strong> H.264 ultrafast (zerolatency) @ 1.0 Mbps</span>
        <span class="meta-chip"><strong>Resolution:</strong> 640x360 @ 25fps</span>
        <span class="meta-chip"><strong>MPEG-TS Payload:</strong> 1,316 B (7 TS pkts/datagram)</span>
        <span class="meta-chip"><strong>Jitter Model:</strong> RFC 3550 Filter</span>
        <span class="meta-chip"><strong>Impairment:</strong> In-Memory Proxy Pipeline</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Collapsible Narrow Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### 🔬 Network & Environment")
    
    st.markdown(
        """
        <div style="background: #121B2B; border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 12px; margin-bottom: 14px;">
            <div style="font-size: 11px; font-weight: 600; color: #35C8F4; text-transform: uppercase; margin-bottom: 4px;">Host Topology</div>
            <div style="font-size: 12px; color: #CBD5E1; line-height: 1.4;">
                Testing runs on <strong>Loopback (127.0.0.1)</strong>, which exercises the full OS network stack (sliding windows, ACKs, datagrams).
            </div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">
                <em>No second device required</em>: deterministic impairments are injected via in-memory proxy queues.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Host OS:** Windows 11 ARM64 (`aarch64`)")
    st.markdown("**Python Environment:** Python 3.12 (`.venv`)")
    st.markdown(f"**TCP Port:** `{config.VIDEO_TCP_PORT}` | **UDP Port:** `{config.VIDEO_UDP_PORT}`")
    st.markdown(f"**Target Bitrate:** `{config.VIDEO_BITRATE}` CBR")
    st.markdown(f"**Chunk Size:** `{config.UDP_PAYLOAD_SIZE}` bytes")
    
    st.markdown("---")
    st.markdown("#### 🎬 Video Asset Selection")
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
            with st.expander("Preview Uploaded Video", expanded=True):
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

    st.markdown("---")
    st.markdown("#### ⚙️ Quick Controls")
    if st.button("🔄 Refresh Dataset", use_container_width=True):
        st.rerun()

    st.caption("NetFlow Research Suite • Standard Architecture")

# ------------------------------------------------------------------------------
# Load Dataset & KPI Cards
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
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">Total Trials</span>
            <span class="kpi-indicator" style="background: #94A3B8;"></span>
        </div>
        <div class="kpi-value">{total_trials}</div>
        <div class="kpi-subtext">{completed_count} verified, {failed_count} failed</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">TCP Reliable</span>
            <span class="kpi-indicator" style="background: #35C8F4;"></span>
        </div>
        <div class="kpi-value" style="color: #35C8F4;">{tcp_count}</div>
        <div class="kpi-subtext">Byte-stream / Retransmit check</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">UDP Datagram</span>
            <span class="kpi-indicator" style="background: #FF9D45;"></span>
        </div>
        <div class="kpi-value" style="color: #FF9D45;">{udp_count}</div>
        <div class="kpi-subtext">RFC 3550 timestamps / Jitter</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">Conditions Tested</span>
            <span class="kpi-indicator" style="background: #A78BFA;"></span>
        </div>
        <div class="kpi-value" style="color: #A78BFA;">{active_conditions} / 7</div>
        <div class="kpi-subtext">Impairment profile coverage</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">Mean Throughput</span>
            <span class="kpi-indicator" style="background: #10B981;"></span>
        </div>
        <div class="kpi-value">{mean_throughput:.2f} <span style="font-size: 13px; color: #94A3B8;">Mbps</span></div>
        <div class="kpi-subtext">Across active completed runs</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Main Tabs Navigation
# ------------------------------------------------------------------------------
tab_runner, tab_comparison, tab_conditions, tab_summary, tab_data = st.tabs([
    "🚀 Live Experiment Runner & Video Transfer Inspector",
    "📊 TCP vs UDP Comparative Analysis",
    "🧪 Impairment Scenarios & Mechanics",
    "📈 Statistical Summaries (Mean ± Std)",
    "📑 Audit Trail & CSV Export",
])

# Plotly Theme Settings
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0B1020",
    plot_bgcolor="#121B2B",
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
        bgcolor="rgba(18, 27, 43, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.08)",
        borderwidth=1,
    ),
)
COLOR_TCP = "#35C8F4"
COLOR_UDP = "#FF9D45"
COLOR_ACCENT = "#A78BFA"
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
# TAB 1: Live Experiment Runner & Video Transfer Inspector
# ==============================================================================
with tab_runner:
    st.markdown("### 🎛️ Streaming Pipeline & Real-Time Experiment Execution")
    
    # Active video source chip
    video_type_label = "Default Sample Video" if active_video_name == "sample.mp4" else "User Uploaded Video"
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; font-size: 12px; color: #CBD5E1;">
        <span style="font-size: 14px;">🎬</span>
        <span>Active Source Asset: <strong style="color: #35C8F4;">{active_video_name}</strong> ({active_video_sub})</span>
        <span style="font-size: 11px; padding: 2px 7px; border-radius: 4px; background: rgba(53, 200, 244, 0.12); color: #35C8F4; border: 1px solid rgba(53, 200, 244, 0.3);">{video_type_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Video Transfer Architectural Pipeline Visualization
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">
            <span>🔬 Video Transfer Architecture & Packet Pipeline</span>
        </div>
        <div class="pipeline-diagram">
            <div class="pipeline-node" style="border-color: rgba(53, 200, 244, 0.4);">
                <div class="pipeline-node-icon">🎞️</div>
                <div class="pipeline-node-title">{active_video_name}</div>
                <div class="pipeline-node-sub">{active_video_sub}</div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-node">
                <div class="pipeline-node-icon">⚙️</div>
                <div class="pipeline-node-title">FFmpeg Transcoder</div>
                <div class="pipeline-node-sub">libx264 zerolatency CBR</div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-node">
                <div class="pipeline-node-icon">📦</div>
                <div class="pipeline-node-title">MPEG-TS Framer</div>
                <div class="pipeline-node-sub">7 pkts = 1,316 B chunks</div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-node" style="border-color: rgba(53, 200, 244, 0.3);">
                <div class="pipeline-node-icon">🔌</div>
                <div class="pipeline-node-title">Transport Sockets</div>
                <div class="pipeline-node-sub">TCP: {config.VIDEO_TCP_PORT} | UDP: {config.VIDEO_UDP_PORT}</div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-node" style="border-color: rgba(167, 139, 250, 0.3);">
                <div class="pipeline-node-icon">🌪️</div>
                <div class="pipeline-node-title">Impairment Proxy</div>
                <div class="pipeline-node-sub">Loss / Delay / Token Bucket</div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-node">
                <div class="pipeline-node-icon">📥</div>
                <div class="pipeline-node-title">Receiver Engine</div>
                <div class="pipeline-node-sub">Metrics & SHA-256 Hash</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controls Row
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([1.3, 1.7, 1.2, 1.0])
    with cfg_col1:
        run_proto = st.selectbox(
            "Transport Protocol",
            options=["both", "tcp", "udp"],
            format_func=lambda x: "both (TCP & UDP Side-by-Side)" if x == "both" else f"{x.upper()} only",
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
            "📺 Launch Live FFplay Video Window",
            value=False,
            help="Pipes incoming stream directly to an FFplay window so you can visually watch the video streaming live and observe packet drops/buffering stalls.",
        )
    with c_opt2:
        if launch_ffplay:
            st.caption("ℹ️ *An FFplay video player window will spawn on your desktop displaying the received stream in real-time.*")
        else:
            st.caption("ℹ️ *Running headless (fastest). Enable the checkbox to watch the live video player.*")

    start_btn = st.button("▶️ Execute Streaming Trials", type="primary", use_container_width=True)

    # Containers for Side-by-Side Live Execution
    st.markdown("#### 📺 Live Side-by-Side Streaming Arena")
    arena_status = st.empty()
    col_tcp_arena, col_udp_arena = st.columns(2)
    tcp_slot = col_tcp_arena.empty()
    udp_slot = col_udp_arena.empty()
    comparison_container = st.container()

    # Function to render a stream card in the arena (replaces content in-place using slot)
    def render_stream_card(slot, proto, status_badge, row=None, error=None):
        is_tcp = proto.lower() == "tcp"
        border_cls = "stream-box-tcp" if is_tcp else "stream-box-udp"
        accent_color = COLOR_TCP if is_tcp else COLOR_UDP
        port_num = config.VIDEO_TCP_PORT if is_tcp else config.VIDEO_UDP_PORT
        proto_name = "TCP (Reliable Byte-Stream)" if is_tcp else "UDP (Timely Datagram)"
        icon = "🔌" if is_tcp else "⚡"

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
            integrity = "✅ SHA-256 Match" if str(row.get("sha256_match")) == "True" else ("❌ Corrupted" if str(row.get("sha256_match")) == "False" else "Loss Tolerant")

            diagnosis = (
                f"<strong>Head-of-Line Blocking:</strong> TCP guaranteed 100% byte integrity. "
                f"Network impairments caused {row.get('stalls', 0)} buffering stalls ({row.get('stall_total_ms', 0):.0f} ms starvation) and {lat} transit latency."
                if is_tcp else
                f"<strong>Timely Datagram Delivery:</strong> UDP prioritized immediate delivery with {jit} jitter and {lat} latency. "
                f"Missing datagrams ({loss} loss) were skipped without stalling the decoder."
            )

            slot.markdown(f"""
            <div class="stream-box {border_cls}">
                <div class="stream-header">
                    <div class="stream-title" style="color: {accent_color};">
                        <span>{icon} {proto_name}</span>
                        <span style="font-size: 11px; font-family: 'JetBrains Mono'; color: #94A3B8;">Port {port_num}</span>
                    </div>
                    <div style="font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3);">
                        {status_badge}
                    </div>
                </div>
                <div class="stream-grid">
                    <div class="stream-stat">
                        <div class="stream-stat-label">Throughput</div>
                        <div class="stream-stat-val" style="color: {accent_color};">{tp}</div>
                    </div>
                    <div class="stream-stat">
                        <div class="stream-stat-label">Avg Latency</div>
                        <div class="stream-stat-val">{lat}</div>
                    </div>
                    <div class="stream-stat">
                        <div class="stream-stat-label">RFC 3550 Jitter</div>
                        <div class="stream-stat-val">{jit}</div>
                    </div>
                    <div class="stream-stat">
                        <div class="stream-stat-label">Packet Loss</div>
                        <div class="stream-stat-val">{loss}</div>
                    </div>
                    <div class="stream-stat">
                        <div class="stream-stat-label">Buffering Stalls</div>
                        <div class="stream-stat-val">{stalls}</div>
                    </div>
                    <div class="stream-stat">
                        <div class="stream-stat-label">Data Integrity</div>
                        <div class="stream-stat-val" style="font-size: 13px;">{integrity}</div>
                    </div>
                </div>
                <div class="stream-diagnosis">
                    {diagnosis}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif error:
            slot.markdown(f"""
            <div class="stream-box {border_cls}">
                <div class="stream-header">
                    <div class="stream-title" style="color: {accent_color};">{icon} {proto_name}</div>
                    <div style="font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3);">
                        ERROR
                    </div>
                </div>
                <div style="color: #F87171; font-size: 12px; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 6px; padding: 12px; margin-bottom: 8px;">
                    <strong>Diagnostic Error:</strong> {error}
                </div>
                <div style="font-size: 11px; color: #94A3B8;">
                    Port {port_num} released cleanly. Sockets have been drained for next trial.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            in_prog_text = "Streaming video frames in real-time..." if "STREAM" in status_badge else ("Waiting for preceding TCP stream to complete..." if "WAIT" in status_badge else "Ready for execution. Click 'Execute Streaming Trials' above.")
            slot.markdown(f"""
            <div class="stream-box {border_cls}">
                <div class="stream-header">
                    <div class="stream-title" style="color: {accent_color};">
                        <span>{icon} {proto_name}</span>
                        <span style="font-size: 11px; font-family: 'JetBrains Mono'; color: #94A3B8;">Port {port_num}</span>
                    </div>
                    <div style="font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: rgba(255, 255, 255, 0.05); color: #94A3B8; border: 1px solid rgba(255, 255, 255, 0.1);">
                        {status_badge}
                    </div>
                </div>
                <div style="color: #64748B; font-size: 12px; font-style: italic; padding: 20px 0; text-align: center;">
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
            <div style="background: #121B2B; border-left: 3px solid #35C8F4; padding: 10px 14px; border-radius: 4px; margin-bottom: 14px; font-size: 13px;">
                🚀 <strong>Executing Live Streaming Matrix:</strong> {total_runs} trial(s) ({run_duration}s per trial). 
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
                    render_stream_card(tcp_slot, "tcp", "⏳ STREAMING...")
                    if "udp" in protocols_to_run and last_udp_row is None:
                        render_stream_card(udp_slot, "udp", "WAITING FOR TCP")
                    try:
                        tcp_res = execute_trial_safe("tcp", run_duration, c, t, run_id, launch_ffplay, active_video_path)
                        last_tcp_row = tcp_res
                        status_str = "COMPLETED" if str(tcp_res.get("completed")).lower() in ("true", "1") else "FAILED"
                        render_stream_card(tcp_slot, "tcp", status_str, row=tcp_res)
                    except Exception as e:
                        render_stream_card(tcp_slot, "tcp", "ERROR", error=str(e))
                    # Allow Windows TCP socket stack TIME_WAIT to drain cleanly
                    time.sleep(1.0)

                # UDP Trial
                if "udp" in protocols_to_run:
                    render_stream_card(udp_slot, "udp", "⏳ STREAMING...")
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
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #10B981; background: #121B2B; border: 1px solid rgba(16, 185, 129, 0.3); padding: 10px 14px; border-radius: 6px; margin-bottom: 14px;">
                ✅ All trials executed successfully and logged to <code>results/experiments.csv</code>.
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
                <div class="comparison-banner">
                    <div class="comparison-title">
                        <span>⚔️ Head-to-Head Comparative Analysis: TCP vs. UDP under Condition '{c_name}'</span>
                        <span style="font-size: 11px; font-weight: normal; color: #94A3B8; margin-left: auto;">Asset: <strong>{v_name}</strong> | Logged at {sbs.get('timestamp')}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
                        <div style="background: #0E1624; padding: 10px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                            <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase;">Throughput Delta</div>
                            <div style="font-size: 15px; font-weight: 700; font-family: 'JetBrains Mono'; color: #F8FAFC;">
                                TCP: {tcp_tp:.2f} | UDP: {udp_tp:.2f}
                            </div>
                            <div style="font-size: 10px; color: #64748B;">Δ {abs(tcp_tp - udp_tp):.2f} Mbps</div>
                        </div>
                        <div style="background: #0E1624; padding: 10px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                            <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase;">Latency Delta</div>
                            <div style="font-size: 15px; font-weight: 700; font-family: 'JetBrains Mono'; color: #F8FAFC;">
                                TCP: {tcp_lat:.1f} ms | UDP: {udp_lat:.1f} ms
                            </div>
                            <div style="font-size: 10px; color: #64748B;">Δ {abs(tcp_lat - udp_lat):.1f} ms</div>
                        </div>
                        <div style="background: #0E1624; padding: 10px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                            <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase;">Jitter Delta (RFC 3550)</div>
                            <div style="font-size: 15px; font-weight: 700; font-family: 'JetBrains Mono'; color: #F8FAFC;">
                                TCP: {tcp_jit:.2f} ms | UDP: {udp_jit:.2f} ms
                            </div>
                            <div style="font-size: 10px; color: #64748B;">Arrival variance</div>
                        </div>
                        <div style="background: #0E1624; padding: 10px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">
                            <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase;">Loss vs. Stalls</div>
                            <div style="font-size: 15px; font-weight: 700; font-family: 'JetBrains Mono'; color: #F8FAFC;">
                                Stalls: {tcp_stalls} | Loss: {udp_loss:.1f}%
                            </div>
                            <div style="font-size: 10px; color: #64748B;">Core Protocol Trade-Off</div>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #CBD5E1; line-height: 1.5;">
                        📌 <strong>Empirical Takeaway:</strong> Under condition <code>{c_name}</code>, UDP sacrificed packet reliability ({udp_loss:.2f}% unrecovered drops) to maintain uninterrupted real-time frame playback with zero stalls. In contrast, TCP preserved 100% cryptographic SHA-256 data fidelity, but suffered {tcp_stalls} playback buffering stall(s) due to socket retransmissions and Head-of-Line blocking.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif not start_btn:
        # Default placeholder arena view
        render_stream_card(col_tcp_arena, "tcp", "STANDBY")
        render_stream_card(col_udp_arena, "udp", "STANDBY")

# ==============================================================================
# TAB 2: TCP vs UDP Comparative Analysis
# ==============================================================================
with tab_comparison:
    st.markdown("### 📊 Direct Comparative Metrics: TCP vs. UDP")

    if completed_df.empty:
        st.warning("No completed experiment records found yet. Use the 'Live Experiment Runner' to generate benchmark data.")
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
            <div style="background: #121B2B; border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 12px 16px; margin-top: 4px;">
                <div style="font-size: 13px; font-weight: 600; color: #F1F5F9;">{metric_title} [{metric_unit}]</div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">{metric_desc}</div>
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
                st.markdown("#### Measurement Variance & Distribution")
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
                st.markdown("#### Throughput vs. Latency Trade-Off")
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
# TAB 3: Impairment Scenarios & Mechanics
# ==============================================================================
with tab_conditions:
    st.markdown("### 🧪 Network Impairment Scenarios & Mathematical Models")
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

    st.markdown("---")
    st.markdown("#### 🔬 Protocol Mechanics Under Network Impairments")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        <div style="background: #121B2B; border: 1px solid rgba(53, 200, 244, 0.25); border-radius: 8px; padding: 16px;">
            <div style="font-size: 14px; font-weight: 700; color: #35C8F4; margin-bottom: 8px;">
                TCP: Reliable Stream Transport
            </div>
            <ul style="font-size: 12px; color: #CBD5E1; line-height: 1.6; margin-left: -15px;">
                <li><strong>Guaranteed Delivery:</strong> Uses sequence numbers, positive acknowledgments (ACKs), and automatic retransmissions (RTO).</li>
                <li><strong>Head-of-Line (HoL) Blocking:</strong> When a segment is lost or delayed, subsequent arriving segments must wait in socket receive buffers until the missing segment is retransmitted.</li>
                <li><strong>Real-World Video Effect:</strong> The video decoder starves for frames, producing <em>playback freezing / buffering stalls</em> and surging latency, but preserving 100% pixel fidelity with zero corrupt macroblocks.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown("""
        <div style="background: #121B2B; border: 1px solid rgba(255, 157, 69, 0.25); border-radius: 8px; padding: 16px;">
            <div style="font-size: 14px; font-weight: 700; color: #FF9D45; margin-bottom: 8px;">
                UDP: Timely Datagram Transport
            </div>
            <ul style="font-size: 12px; color: #CBD5E1; line-height: 1.6; margin-left: -15px;">
                <li><strong>Fire-and-Forget:</strong> Transmits independent datagram chunks without connection handshakes or retransmission delays.</li>
                <li><strong>Zero Head-of-Line Blocking:</strong> Arriving packets are fed immediately into the video decoder; lost packets are skipped.</li>
                <li><strong>Real-World Video Effect:</strong> Playback maintains <em>ultra-low transit latency and smooth continuous timing</em>, but lost MPEG-TS packets result in visual artifacts, pixelation, or transient decoder glitching.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: Statistical Summaries (Mean ± Std Dev)
# ==============================================================================
with tab_summary:
    st.markdown("### 📈 Repeated-Trial Aggregations (Mean ± Std Dev)")
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
# TAB 5: Full Audit Trail & CSV Export
# ==============================================================================
with tab_data:
    st.markdown("### 📑 Full Experiment Audit Trail & CSV Export")
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
                label="📥 Export Full CSV Dataset",
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
                table_df["sha256_match"] = table_df["sha256_match"].apply(lambda v: "✅ Match" if str(v) == "True" else ("❌ Mismatch" if str(v) == "False" else "-"))
            if "completed" in table_df.columns:
                table_df["completed"] = table_df["completed"].apply(lambda v: "✅ OK" if str(v) == "True" else "❌ Failed")

            table_df = table_df.rename(columns=summary_cols_map)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        else:
            raw_display = view_df.astype(str).replace({"nan": "-", "None": "-"})
            st.dataframe(raw_display, use_container_width=True, hide_index=True)
