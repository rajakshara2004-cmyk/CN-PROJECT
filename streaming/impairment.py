"""Application-level network impairment proxy for Windows/Linux local testing.

Simulates latency, jitter, packet loss, and bandwidth throttling at the proxy layer,
distinguishing simulated proxy impairments from OS/kernel-level network shaping.
"""
import heapq
import random
import socket
import struct
import threading
import time
from typing import Dict, Optional

from streaming.udp_common import HEADER as UDP_HEADER
from streaming.udp_common import TYPE_DATA
from streaming.video_common import TCP_FRAME

CONDITIONS: Dict[str, dict] = {
    "normal": {
        "description": "Baseline: no added latency, loss, or rate limiting",
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "loss_pct": 0.0,
        "rate_mbit": 0.0,
    },
    "bandwidth": {
        "description": "Limited bandwidth: 0.8 Mbps rate limit",
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "loss_pct": 0.0,
        "rate_mbit": 0.8,
    },
    "delay": {
        "description": "Increased latency: 50 ms delay",
        "delay_ms": 50.0,
        "jitter_ms": 0.0,
        "loss_pct": 0.0,
        "rate_mbit": 0.0,
    },
    "jitter": {
        "description": "Jitter: 50 ms delay +/- 20 ms jitter",
        "delay_ms": 50.0,
        "jitter_ms": 20.0,
        "loss_pct": 0.0,
        "rate_mbit": 0.0,
    },
    "loss_low": {
        "description": "Moderate packet loss: 3% loss",
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "loss_pct": 3.0,
        "rate_mbit": 0.0,
    },
    "loss_high": {
        "description": "Higher packet loss: 10% loss",
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "loss_pct": 10.0,
        "rate_mbit": 0.0,
    },
    "combined": {
        "description": "Combined impairments: 50 ms delay, 15 ms jitter, 5% loss, 0.8 Mbps",
        "delay_ms": 50.0,
        "jitter_ms": 15.0,
        "loss_pct": 5.0,
        "rate_mbit": 0.8,
    },
}


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed before expected bytes received")
        buf.extend(chunk)
    return bytes(buf)


class UdpImpairmentProxy:
    """UDP proxy that intercepts datagrams and applies loss, delay, jitter, and rate limiting."""

    def __init__(self, listen_host: str, listen_port: int, dest_host: str, dest_port: int,
                 delay_ms: float = 0.0, jitter_ms: float = 0.0,
                 loss_pct: float = 0.0, rate_mbit: float = 0.0):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.dest_host = dest_host
        self.dest_port = dest_port
        self.delay_ms = float(delay_ms)
        self.jitter_ms = float(jitter_ms)
        self.loss_pct = float(loss_pct)
        self.rate_mbit = float(rate_mbit)

        self.sock: Optional[socket.socket] = None
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()

        self._queue = []  # min-heap: (send_time, counter, data)
        self._cv = threading.Condition()
        self._counter = 0
        self._next_avail_time = 0.0

        self.stats = {
            "received": 0,
            "forwarded": 0,
            "dropped": 0,
            "bytes_forwarded": 0,
        }
        self.rx_thread: Optional[threading.Thread] = None
        self.tx_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind((self.listen_host, self.listen_port))
        self.sock.settimeout(0.2)

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self.rx_thread.start()
        self.tx_thread.start()
        self.ready_event.set()

    def _calc_delay(self, nbytes: int) -> float:
        now = time.perf_counter()
        delay_s = self.delay_ms / 1000.0
        if self.jitter_ms > 0:
            delay_s += random.uniform(-self.jitter_ms / 1000.0, self.jitter_ms / 1000.0)
            if delay_s < 0:
                delay_s = 0.0

        if self.rate_mbit > 0:
            bit_time = (nbytes * 8) / (self.rate_mbit * 1e6)
            sched_base = max(now, self._next_avail_time)
            self._next_avail_time = sched_base + bit_time
            send_time = self._next_avail_time + delay_s
        else:
            send_time = now + delay_s

        return send_time

    def _rx_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                data, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except (OSError, ConnectionResetError):
                break

            self.stats["received"] += 1

            ptype = -1
            if len(data) >= UDP_HEADER.size:
                try:
                    ptype, _seq, _ts = UDP_HEADER.unpack_from(data)
                except struct.error:
                    pass

            if ptype == TYPE_DATA and self.loss_pct > 0 and (random.random() * 100.0 < self.loss_pct):
                self.stats["dropped"] += 1
                continue

            send_time = self._calc_delay(len(data))
            with self._cv:
                self._counter += 1
                heapq.heappush(self._queue, (send_time, self._counter, data))
                self._cv.notify()

    def _tx_loop(self) -> None:
        dest_addr = (self.dest_host, self.dest_port)
        while not self.stop_event.is_set():
            with self._cv:
                while not self._queue and not self.stop_event.is_set():
                    self._cv.wait(timeout=0.05)
                if self.stop_event.is_set() and not self._queue:
                    break
                if not self._queue:
                    continue

                now = time.perf_counter()
                send_time, counter, data = self._queue[0]
                wait_time = send_time - now
                if wait_time > 0:
                    self._cv.wait(timeout=wait_time)
                    continue

                heapq.heappop(self._queue)

            try:
                self.sock.sendto(data, dest_addr)
                self.stats["forwarded"] += 1
                self.stats["bytes_forwarded"] += len(data)
            except (OSError, ConnectionResetError):
                pass

    def stop(self) -> None:
        self.stop_event.set()
        with self._cv:
            self._cv.notify_all()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=1.0)
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class TcpImpairmentProxy:
    """TCP proxy that accepts sender connections and forwards to receiver with pipeline shaping."""

    def __init__(self, listen_host: str, listen_port: int, dest_host: str, dest_port: int,
                 delay_ms: float = 0.0, jitter_ms: float = 0.0,
                 loss_pct: float = 0.0, rate_mbit: float = 0.0):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.dest_host = dest_host
        self.dest_port = dest_port
        self.delay_ms = float(delay_ms)
        self.jitter_ms = float(jitter_ms)
        self.loss_pct = float(loss_pct)
        self.rate_mbit = float(rate_mbit)

        self.srv_sock: Optional[socket.socket] = None
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()

        self._queue = []  # min-heap: (send_time, counter, Optional[bytes])
        self._cv = threading.Condition()
        self._counter = 0
        self._next_avail_time = 0.0

        self.stats = {
            "forwarded_frames": 0,
            "simulated_retransmits": 0,
            "bytes_forwarded": 0,
        }
        self.rx_thread: Optional[threading.Thread] = None
        self.tx_thread: Optional[threading.Thread] = None
        self.client_sock: Optional[socket.socket] = None
        self.dest_sock: Optional[socket.socket] = None

    def start(self) -> None:
        self.srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv_sock.bind((self.listen_host, self.listen_port))
        self.srv_sock.listen(1)
        self.srv_sock.settimeout(0.5)

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()
        self.ready_event.set()

    def _calc_delay(self, nbytes: int) -> float:
        now = time.perf_counter()
        delay_s = self.delay_ms / 1000.0
        if self.jitter_ms > 0:
            delay_s += random.uniform(-self.jitter_ms / 1000.0, self.jitter_ms / 1000.0)
            if delay_s < 0:
                delay_s = 0.0

        if self.rate_mbit > 0:
            bit_time = (nbytes * 8) / (self.rate_mbit * 1e6)
            sched_base = max(now, self._next_avail_time)
            self._next_avail_time = sched_base + bit_time
            send_time = self._next_avail_time + delay_s
        else:
            send_time = now + delay_s

        return send_time

    def _rx_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.client_sock, _addr = self.srv_sock.accept()
                break
            except socket.timeout:
                continue
            except OSError:
                return

        if self.client_sock is None:
            return

        try:
            self.dest_sock = socket.create_connection((self.dest_host, self.dest_port), timeout=10.0)
            self.dest_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # First 8 bytes is sender start timestamp
            t0_bytes = _recv_exact(self.client_sock, 8)
            self.dest_sock.sendall(t0_bytes)

            # Start writer thread
            self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
            self.tx_thread.start()

            # Read frames continuously from sender and push into scheduled queue
            while not self.stop_event.is_set():
                head_raw = self.client_sock.recv(TCP_FRAME.size)
                if not head_raw:
                    break  # Sender done
                if len(head_raw) < TCP_FRAME.size:
                    head_raw += _recv_exact(self.client_sock, TCP_FRAME.size - len(head_raw))

                seq, send_ts, length = TCP_FRAME.unpack(head_raw)
                payload = _recv_exact(self.client_sock, length)
                frame_bytes = head_raw + payload

                send_time = self._calc_delay(len(frame_bytes))
                if self.loss_pct > 0 and (random.random() * 100.0 < self.loss_pct):
                    self.stats["simulated_retransmits"] += 1
                    # Emulate TCP retransmit recovery delay
                    send_time += random.uniform(0.10, 0.20)

                with self._cv:
                    self._counter += 1
                    heapq.heappush(self._queue, (send_time, self._counter, frame_bytes))
                    self._cv.notify()

        except (ConnectionError, OSError):
            pass
        finally:
            # Enqueue EOF marker after all queued frames have been scheduled
            eof_time = max(time.perf_counter(), self._next_avail_time + (self.delay_ms / 1000.0) + 0.05)
            with self._cv:
                self._counter += 1
                heapq.heappush(self._queue, (eof_time, self._counter, None))
                self._cv.notify()

    def _tx_loop(self) -> None:
        while not self.stop_event.is_set():
            with self._cv:
                while not self._queue and not self.stop_event.is_set():
                    self._cv.wait(timeout=0.05)
                if not self._queue:
                    if self.stop_event.is_set():
                        break
                    continue

                now = time.perf_counter()
                send_time, counter, item = self._queue[0]
                wait_time = send_time - now
                if wait_time > 0 and not self.stop_event.is_set():
                    self._cv.wait(timeout=wait_time)
                    continue

                heapq.heappop(self._queue)

            if item is None:
                # EOF reached: close connection to signal receiver that stream finished
                try:
                    if self.dest_sock:
                        self.dest_sock.close()
                except OSError:
                    pass
                break

            try:
                if self.dest_sock:
                    self.dest_sock.sendall(item)
                    self.stats["forwarded_frames"] += 1
                    self.stats["bytes_forwarded"] += len(item) - TCP_FRAME.size
            except OSError:
                break

    def stop(self) -> None:
        # Give queued items up to 5 seconds to drain naturally
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=5.0)

        self.stop_event.set()
        with self._cv:
            self._cv.notify_all()

        for s in (self.client_sock, self.dest_sock, self.srv_sock):
            if s:
                try:
                    s.close()
                except OSError:
                    pass

        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=1.0)
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def get_impairment_config(condition_name: str) -> dict:
    """Return dictionary of impairment parameters for condition_name."""
    if condition_name not in CONDITIONS:
        raise ValueError(f"Unknown condition: '{condition_name}'. Supported: {sorted(CONDITIONS)}")
    c = CONDITIONS[condition_name].copy()
    c["name"] = condition_name
    c["applied"] = condition_name != "normal"
    c["verified"] = condition_name != "normal"
    c["qdisc"] = "application_proxy" if condition_name != "normal" else "none"
    return c
