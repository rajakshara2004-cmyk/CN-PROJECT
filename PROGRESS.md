# Project Progress: Performance Evaluation of TCP vs. UDP for Real-Time Video Streaming

## 1. System & Environment Details
- **Operating System:** Windows 11 on ARM64 (`aarch64`)
- **Python Environment:** `.venv\Scripts\python.exe` (Python 3.12.10 ARM64)
- **Key Installed Dependencies:**
  - `numpy`: 2.5.4
  - `pandas`: 3.0.6
  - `plotly`: 7.1.0
  - `matplotlib`: 3.11.2
  - `streamlit`: 1.66.0
  - `ffmpeg`: 9.0.2 (`C:\Users\AKSHARA RAJ\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.EXE`)
  - `ffplay`: 9.0.2 (`...bin\ffplay.EXE`)
  - `ffprobe`: 9.0.2 (`...bin\ffprobe.EXE`)
- **WSL Status:** WSL2 Ubuntu is operational (`Linux 6.18.54.1-microsoft-standard-WSL2 aarch64`) with `/usr/sbin/tc` installed.

---

## 2. Milestone Verification Status (Phases 1–8)

| Phase | Description | Command Executed | Result | Evidence / Observed Output |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Environment, imports, configurations, paths | `python check_env.py`<br>`python config.py` | **PASS** | Verified libraries, ffmpeg/ffplay paths, dirs, and default ports. |
| **Phase 2** | Raw TCP socket connection, binary transfer, integrity check, failure handling | `python tests/test_tcp.py` | **PASS** | Roundtrips for 0B, 1B, 1KB, 5MB verified SHA-256 match & ACK; connection refusal verified. |
| **Phase 3** | UDP sequence tracking, timestamps, loss/duplicate/out-of-order detection, timeouts | `python tests/test_udp.py` | **PASS** | Verified artificial 8-packet loss/dup/reorder sequence; 500-packet stream at 2000 pps; clean timeout & stop event. |
| **Phase 4** | Real-time continuous video streaming over TCP and UDP, MPEG-TS piping, playback | `python tests/test_tcp_video.py`<br>`python tests/test_udp_video.py` | **PASS** | FFmpeg streamed `data/videos/sample.mp4` (H.264 640x360@25fps); TS chunks received, SHA-256 match verified; `ffprobe` verified received video format; startup delay & stalls tracked. |
| **Phase 5** | Metrics calculations, RFC 3550 jitter, stall tracking, netstat parsing, CSV logging | `python tests/test_metrics.py` | **PASS** | Calculations for jitter, stalls, startup delay hand-verified; netstat parser passed; safe CSV header upgrade passed; TCP & UDP trial execution verified. |
| **Phase 6** | Network impairment framework (7 conditions: normal, bandwidth, delay, jitter, loss_low, loss_high, combined), queue pipelining, trial automation | `python tests/test_impairment.py`<br>`python streaming/run_experiment.py --list-conditions` | **PASS** | Pipelined `TcpImpairmentProxy` and `UdpImpairmentProxy` verified with target latency, jitter, loss %, and rate limiting; SHA-256 integrity maintained on TCP; CSV metadata populated (`impairment_applied: yes`, `impairment_verified: yes`). |
| **Phase 7** | Interactive Streamlit Dashboard | `streamlit run dashboard/app.py` | **PASS** | Initial 5-tab dashboard built with KPIs, trial execution, and charts. |
| **Phase 8** | Comprehensive End-to-End QA | `python tests/test_dashboard.py`<br>`python streaming/summarize_results.py` | **PASS** | All unit/integration test suites passing 100%. 21 completed trials logged in `results/experiments.csv`. |
| **Phase 9** | Research Platform Redesign & Video Transfer Visualizer | `streamlit run dashboard/app.py`<br>`python tests/test_dashboard.py` | **PASS** | Professional dark navy (#0B1020 / #121B2B) research platform theme, cyan (#35C8F4) TCP & orange (#FF9D45) UDP, side-by-side live execution arena, visual packet pipeline, live FFplay video playback window toggle, custom video upload with default sample fallback, and enhanced Plotly analytics. |

---

## 3. Supported Impairment Conditions in Phase 6
All conditions use repeatable mathematical queueing models in `streaming/impairment.py`:
1. `normal`: Baseline (delay 0ms, jitter 0ms, loss 0%, unlimited rate)
2. `bandwidth`: Limited bandwidth (0.8 Mbps rate limit token bucket)
3. `delay`: Increased latency (50 ms propagation delay)
4. `jitter`: Jitter (50 ms delay ± 20 ms random uniform jitter)
5. `loss_low`: Moderate packet loss (3.0% loss probability)
6. `loss_high`: Higher packet loss (10.0% loss probability)
7. `combined`: Combined impairments (50 ms delay, 15 ms jitter, 5.0% loss, 0.8 Mbps rate limit)

---

## 4. Empirical Benchmark Summary (21 Completed Trials)

| Condition | Protocol | Trials | Throughput (Mbps) | Latency (ms) | Jitter (ms) | Startup Delay (ms) | Packet Loss % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **normal** | TCP | 4 | 1.124 ± 0.051 | 0.322 ± 0.066 | 0.159 ± 0.039 | 1713.7 ± 996.6 | n/a (retransmit) |
| **normal** | UDP | 4 | 1.121 ± 0.043 | 0.294 ± 0.027 | 0.092 ± 0.016 | 1708.7 ± 1027.8 | 0.000 ± 0.000 |
| **bandwidth** | TCP | 2 | 0.794 ± 0.002 | 827.3 ± 152.6 | 19.27 ± 2.15 | 148.6 ± 18.9 | n/a (retransmit) |
| **bandwidth** | UDP | 1 | 0.793 | 924.6 | 21.34 | 172.0 | 0.000% |
| **delay** | TCP | 1 | 1.232 | 58.14 | 1.29 | 210.9 | n/a (retransmit) |
| **delay** | UDP | 1 | 1.181 | 58.32 | 1.64 | 254.3 | 0.000% |
| **jitter** | TCP | 1 | 1.216 | 57.98 | 11.51 | 215.5 | n/a (retransmit) |
| **jitter** | UDP | 1 | 1.190 | 58.21 | 11.00 | 182.2 | 0.000% |
| **loss_low** | TCP | 1 | 1.181 | 5.73 | 26.99 | 148.8 | n/a (retransmit) |
| **loss_low** | UDP | 1 | 1.156 | 0.68 | 0.09 | 194.3 | 3.049% |
| **loss_high** | TCP | 1 | 1.217 | 17.69 | 27.60 | 152.7 | n/a (retransmit) |
| **loss_high** | UDP | 1 | 1.104 | 0.91 | 0.11 | 197.3 | 6.707% |
| **combined** | TCP | 1 | 0.790 | 993.56 | 19.07 | 189.4 | n/a (retransmit) |
| **combined** | UDP | 1 | 0.794 | 901.03 | 22.84 | 268.8 | 4.268% |

---

## 5. How to Run the Project

### Running the Dashboard:
```powershell
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

### Running Automated Experiment Trials:
```powershell
# Run 3 trials of all 7 conditions (both TCP and UDP):
.\.venv\Scripts\python.exe streaming/run_experiment.py --duration 10.0 --trials 3 --condition all

# List all condition profiles:
.\.venv\Scripts\python.exe streaming/run_experiment.py --list-conditions

# Print statistical summaries:
.\.venv\Scripts\python.exe streaming/summarize_results.py
```

### Running the Test Suite:
```powershell
.\.venv\Scripts\python.exe tests/test_tcp.py
.\.venv\Scripts\python.exe tests/test_udp.py
.\.venv\Scripts\python.exe tests/test_tcp_video.py
.\.venv\Scripts\python.exe tests/test_udp_video.py
.\.venv\Scripts\python.exe tests/test_metrics.py
.\.venv\Scripts\python.exe tests/test_impairment.py
.\.venv\Scripts\python.exe tests/test_dashboard.py
```
