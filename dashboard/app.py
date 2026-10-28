"""Streamlit Dashboard: Performance Evaluation of TCP vs. UDP for Real-Time Video Streaming.

Provides interactive experiment orchestration, real-time trial execution,
side-by-side comparative analytics, statistical summaries, and audit trail export.
"""
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
from streaming.impairment import CONDITIONS, get_impairment_config  # noqa: E402

# Page config
st.set_page_config(
    page_title="TCP vs UDP Video Streaming Benchmark",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Top banner card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
    }

    /* Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-desc {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
    }

    /* Condition pill badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-tcp {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-udp {
        background-color: rgba(249, 115, 22, 0.15);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }

    /* Custom tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #94a3b8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.12) !important;
        color: #38bdf8 !important;
        font-weight: 600;
        border-bottom: 2px solid #38bdf8 !important;
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
        # Ensure standard types
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


# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📡 TCP vs. UDP Real-Time Video Streaming Evaluation</div>
    <div class="hero-subtitle">
        Performance evaluation under normal conditions and network impairments (Bandwidth, Delay, Jitter, Packet Loss).
        Paced MPEG-TS real-time streaming with RFC 3550 jitter and cryptographic integrity verification.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar System Specs
with st.sidebar:
    st.markdown("### 🖥️ Environment & Specs")
    st.markdown(f"**OS / Host:** `{metrics.sys.platform.upper()}` (ARM64 Windows)")
    st.markdown(f"**Encoder:** `libx264 ultrafast` (zerolatency)")
    st.markdown(f"**Target Bitrate:** `{config.VIDEO_BITRATE}` ({config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT} @ {config.VIDEO_FPS}fps)")
    st.markdown(f"**Payload Chunk:** `{config.UDP_PAYLOAD_SIZE}` bytes (7 TS pkts)")
    st.markdown(f"**Ports:** TCP {config.VIDEO_TCP_PORT} | UDP {config.VIDEO_UDP_PORT}")
    st.markdown("---")

    st.markdown("### ⚙️ Quick Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("TCP vs UDP Research Project • Phases 1–8 Active")

# Load data
df = load_dataset()
completed_df = df[df["completed"].astype(str) == "True"] if not df.empty else df

# Top Status Indicators
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
with col_kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Trials</div>
        <div class="metric-value">{len(df)}</div>
        <div class="metric-desc">{len(completed_df)} completed, {len(df) - len(completed_df)} failed</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    tcp_count = len(completed_df[completed_df["protocol"] == "tcp"]) if not completed_df.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TCP Runs</div>
        <div class="metric-value" style="color: #38bdf8;">{tcp_count}</div>
        <div class="metric-desc">Reliable byte-stream</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    udp_count = len(completed_df[completed_df["protocol"] == "udp"]) if not completed_df.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">UDP Runs</div>
        <div class="metric-value" style="color: #fb923c;">{udp_count}</div>
        <div class="metric-desc">Datagram with RFC 3550</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    n_conds = completed_df["condition"].nunique() if not completed_df.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Conditions</div>
        <div class="metric-value">{n_conds} / 7</div>
        <div class="metric-desc">Impairment profiles</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi5:
    last_time = df["timestamp"].iloc[-1] if not df.empty and "timestamp" in df.columns else "None"
    if last_time != "None" and "T" in str(last_time):
        last_time = str(last_time).split("T")[1][:8]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Run</div>
        <div class="metric-value" style="font-size: 20px;">{last_time}</div>
        <div class="metric-desc">Audit timestamp</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Tab Navigation
tab_runner, tab_comparison, tab_conditions, tab_summary, tab_data = st.tabs([
    "🚀 Live Experiment Runner",
    "📊 TCP vs UDP Comparative Analysis",
    "🧪 Impairment Scenarios",
    "📈 Statistical Summaries (Mean ± Std)",
    "📑 Audit Trail & CSV Export",
])

# ==============================================================================
# TAB 1: Live Experiment Runner
# ==============================================================================
with tab_runner:
    st.markdown("### 🎛️ Configure & Execute Streaming Trials")
    st.markdown(
        "Run automated, repeatable trials on loopback. Conditions other than `normal` "
        "automatically utilize the isolated **Application-Level Impairment Proxy** to inject "
        "reproducible latency, jitter, loss, or bandwidth limits."
    )

    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    with cfg_col1:
        run_proto = st.selectbox(
            "Protocol",
            options=["both", "tcp", "udp"],
            help="Select 'both' to alternate TCP and UDP trials sequentially",
        )
    with cfg_col2:
        condition_options = list(CONDITIONS.keys()) + ["all"]
        run_cond = st.selectbox(
            "Condition Profile",
            options=condition_options,
            format_func=lambda x: f"{x} - {CONDITIONS[x]['description']}" if x in CONDITIONS else "all - Full 7-condition test matrix",
        )
    with cfg_col3:
        run_duration = st.slider(
            "Stream Duration (seconds)",
            min_value=3.0,
            max_value=30.0,
            value=5.0,
            step=1.0,
            help="Duration of real-time video encoded and streamed per trial",
        )
    with cfg_col4:
        run_trials = st.number_input(
            "Trials per Condition",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Number of repeated trials for statistical replication (3+ recommended)",
        )

    start_btn = st.button("▶️ Start Experiment Run", type="primary", use_container_width=True)

    if start_btn:
        conditions_to_run = list(CONDITIONS.keys()) if run_cond == "all" else [run_cond]
        protocols_to_run = ["tcp", "udp"] if run_proto == "both" else [run_proto]
        total_runs = len(conditions_to_run) * run_trials * len(protocols_to_run)

        st.info(f"Starting Run: {total_runs} total trial(s) ({len(conditions_to_run)} condition(s), {run_trials} repeat(s) @ {run_duration}s each).")
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        log_box = st.container()

        run_id = uuid.uuid4().hex[:8]
        current_step = 0

        for c in conditions_to_run:
            for t in range(1, run_trials + 1):
                for p in protocols_to_run:
                    current_step += 1
                    status_box.markdown(
                        f"⏳ **Executing [{current_step}/{total_runs}]:** "
                        f"Protocol `{p.upper()}` | Trial `{t}` | Condition `{c}`..."
                    )

                    try:
                        row = run_experiment.run_trial(
                            protocol=p,
                            duration=run_duration,
                            condition=c,
                            trial=t,
                            run_id=run_id,
                            csv_path=metrics.CSV_PATH,
                        )
                        res_desc = run_experiment.describe(row)
                        with log_box:
                            if row["completed"]:
                                st.success(f"**[{current_step}/{total_runs}] {p.upper()} Trial {t} ({c})**: {res_desc}")
                            else:
                                st.error(f"**[{current_step}/{total_runs}] {p.upper()} Trial {t} ({c})**: {res_desc}")
                    except Exception as err:
                        with log_box:
                            st.error(f"Error in {p.upper()} trial {t} ({c}): {err}")

                    progress_bar.progress(current_step / total_runs)
                    time.sleep(0.5)

        status_box.markdown("✅ **Experiment run completed! Results saved to `results/experiments.csv`.**")
        st.balloons()
        time.sleep(1.0)
        st.rerun()

# ==============================================================================
# TAB 2: TCP vs UDP Comparative Analysis
# ==============================================================================
with tab_comparison:
    st.markdown("### 📊 Direct Comparative Metrics (TCP vs. UDP)")

    if completed_df.empty:
        st.warning("No completed experiment runs found yet. Use the 'Live Experiment Runner' to generate baseline data.")
    else:
        metric_choices = {
            "throughput_mbps": "Throughput (Mbps)",
            "lat_avg_ms": "Average End-to-End Latency (ms)",
            "jitter_ms": "RFC 3550 Jitter (ms)",
            "startup_delay_ms": "Initial Startup Delay (ms)",
            "stalls": "Stall Count (>250ms gap)",
            "stall_total_ms": "Total Stall Duration (ms)",
            "loss_pct": "Packet Loss % (UDP only)",
            "total_time_s": "Total Elapsed Transfer Time (s)",
        }

        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            selected_metric = st.selectbox(
                "Select Metric to Compare",
                options=list(metric_choices.keys()),
                format_func=lambda x: metric_choices[x],
            )

            all_conditions = sorted(completed_df["condition"].unique().tolist())
            selected_conditions = st.multiselect(
                "Filter Conditions",
                options=all_conditions,
                default=all_conditions,
            )

        with f_col2:
            st.markdown(f"**Metric Focus: {metric_choices[selected_metric]}**")
            if selected_metric == "loss_pct":
                st.caption("ℹ️ *Note: UDP directly records dropped packets. TCP provides reliable in-order transport; kernel/proxy packet loss induces RTO retransmission delay and stalls rather than data loss.*")
            elif selected_metric == "jitter_ms":
                st.caption("ℹ️ *Note: Jitter is calculated using the RFC 3550 statistical variance estimator $J = J + (|D| - J) / 16$.*")
            else:
                st.caption("ℹ️ *Comparison reflects empirical measurements captured with identical video sources, bitrates, and resolution.*")

        filtered = completed_df[completed_df["condition"].isin(selected_conditions)]

        if filtered.empty:
            st.info("No data matches the selected condition filter.")
        else:
            # Bar chart comparing TCP vs UDP across conditions
            fig_bar = px.bar(
                filtered,
                x="condition",
                y=selected_metric,
                color="protocol",
                barmode="group",
                color_discrete_map={"tcp": "#38bdf8", "udp": "#fb923c"},
                title=f"{metric_choices[selected_metric]} by Condition and Protocol",
                labels={"condition": "Network Condition", selected_metric: metric_choices[selected_metric], "protocol": "Protocol"},
                template="plotly_dark",
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                paper_bgcolor="rgba(11, 15, 25, 0.0)",
                font=dict(family="Inter, sans-serif"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Box plot showing distribution
            if len(filtered) > 4:
                st.markdown("#### Measurement Variance & Distribution")
                fig_box = px.box(
                    filtered,
                    x="condition",
                    y=selected_metric,
                    color="protocol",
                    color_discrete_map={"tcp": "#38bdf8", "udp": "#fb923c"},
                    title=f"Spread & Outliers: {metric_choices[selected_metric]}",
                    template="plotly_dark",
                )
                fig_box.update_layout(
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    paper_bgcolor="rgba(11, 15, 25, 0.0)",
                    font=dict(family="Inter, sans-serif"),
                )
                st.plotly_chart(fig_box, use_container_width=True)

# ==============================================================================
# TAB 3: Impairment Scenarios
# ==============================================================================
with tab_conditions:
    st.markdown("### 🧪 Network Impairment Profiles & Mechanics")
    st.markdown(
        "Each impairment condition tests specific boundary characteristics of real-time streaming protocols. "
        "Review the exact impairment parameters below:"
    )

    cond_cards = []
    for name, c in CONDITIONS.items():
        cond_cards.append({
            "Condition": name,
            "Description": c["description"],
            "Delay (ms)": c["delay_ms"],
            "Jitter (ms)": c["jitter_ms"],
            "Loss (%)": c["loss_pct"],
            "Rate Limit (Mbps)": f"{c['rate_mbit']:.1f}" if c["rate_mbit"] > 0 else "Unlimited",
            "Impairment Layer": "Application Proxy (Loopback)" if name != "normal" else "None (Native)",
        })
    st.dataframe(pd.DataFrame(cond_cards), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔬 Protocol Dynamics Under Impairments")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        **TCP (Transmission Control Protocol)**
        - **Reliability:** 100% data integrity guaranteed via sequence numbers and retransmissions.
        - **Head-of-Line (HoL) Blocking:** When a segment is delayed or lost, the receiver cannot deliver subsequent arriving packets to the video decoder until the missing segment is retransmitted.
        - **Impact on Video:** Results in playback buffering stalls, spikes in transit latency, and delayed startup time, but zero video artifacting.
        """)
    with col_d2:
        st.markdown("""
        **UDP (User Datagram Protocol)**
        - **Timeliness:** Fire-and-forget datagram delivery with zero retransmission delay.
        - **No Head-of-Line Blocking:** Packets are processed in real-time as they arrive; missing packets are skipped.
        - **Impact on Video:** Maintains smooth playback timing and low latency even under heavy network loss, but can cause visual macroblocking, dropped frames, or decode corruption.
        """)

# ==============================================================================
# TAB 4: Statistical Summaries (Mean ± Std Dev)
# ==============================================================================
with tab_summary:
    st.markdown("### 📈 Repeated-Trial Aggregation (Mean ± Std Dev)")

    if completed_df.empty:
        st.info("No completed trials to summarize.")
    else:
        summary_cols = [
            "throughput_mbps", "lat_avg_ms", "jitter_ms", "startup_delay_ms",
            "stalls", "loss_pct", "total_time_s",
        ]

        rows = []
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
            rows.append(entry)

        summary_df = pd.DataFrame(rows)
        st.dataframe(
            summary_df.rename(columns={
                "throughput_mbps": "Throughput (Mbps)",
                "lat_avg_ms": "Latency (ms)",
                "jitter_ms": "Jitter (ms)",
                "startup_delay_ms": "Startup (ms)",
                "stalls": "Stalls",
                "loss_pct": "Loss %",
                "total_time_s": "Total Time (s)",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# TAB 5: Data Audit Trail & CSV Export
# ==============================================================================
with tab_data:
    st.markdown("### 📑 Full Audit Trail & CSV Export")
    st.markdown(
        "Complete record of all trials executed. Every row includes network conditions, "
        "encoder settings, cryptographic hash verification, and raw metrics."
    )

    if df.empty:
        st.info("No experiments logged in CSV.")
    else:
        # Action controls
        top_c1, top_c2, top_c3 = st.columns([1.5, 1.5, 2])
        with top_c1:
            proto_filter = st.selectbox("Filter Protocol", ["All", "TCP", "UDP"], key="tab5_proto")
        with top_c2:
            show_all_cols = st.checkbox("Show all 44 raw columns", value=False, key="tab5_cols")
        with top_c3:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Full CSV Dataset",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name=f"streaming_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Apply filter
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
