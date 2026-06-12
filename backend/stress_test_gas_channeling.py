import socket
import struct
import time
import numpy as np
import argparse
import threading
import urllib.request
import json
from datetime import datetime


class GasChannelingSimulator:
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.frame_id = 0
        self.width = 128
        self.height = 128
        self.max_value = 2 ** 14 - 1

        self.channeling_active = False
        self.channeling_start_time = 0
        self.channeling_position = (64, 64)
        self.channeling_intensity = 0

        self.stats_lock = threading.Lock()
        self.frames_sent = 0
        self.api_success_count = 0
        self.api_error_count = 0
        self.api_latencies = []

        self._stop = False

    def generate_normal_frame(self, frame_id: int) -> np.ndarray:
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)

        dist = np.sqrt(xx ** 2 + yy ** 2)

        t = frame_id * 0.05
        center_intensity = 0.85 + 0.1 * np.sin(t)

        base_profile = center_intensity * np.exp(-dist ** 2 * 2.5)
        ring1 = 0.12 * np.sin(dist * 18 + t * 2) * np.exp(-dist ** 2 * 1.5)
        ring2 = 0.08 * np.sin(dist * 30 - t * 1.5) * np.exp(-dist ** 2 * 0.8)

        angle = np.arctan2(yy, xx)
        sector_var = 0.06 * np.sin(angle * 4 + t * 0.8) * np.exp(-dist ** 2 * 1.0)

        noise = 0.02 * np.random.randn(self.height, self.width)

        total = base_profile + ring1 + ring2 + sector_var + noise
        total = np.clip(total, 0.05, 0.95)

        raw_values = (total * self.max_value).astype(np.uint16)
        raw_values = raw_values & 0x3FFF
        return raw_values

    def add_gas_channeling(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        if not self.channeling_active:
            return frame

        elapsed = time.time() - self.channeling_start_time

        cx, cy = self.channeling_position

        x = np.linspace(0, self.width - 1, self.width)
        y = np.linspace(0, self.height - 1, self.height)
        xx, yy = np.meshgrid(x, y)

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        intensity_factor = min(1.0, elapsed / 2.0)
        max_intensity = 0.95 + 0.05 * np.sin(frame_id * 0.2)

        channeling_profile = max_intensity * np.exp(-dist ** 2 * 0.02)

        oscillation = 0.1 * np.sin(elapsed * 10) * np.exp(-dist ** 2 * 0.01)
        channeling_profile += oscillation

        channeling_profile = channeling_profile * intensity_factor

        combined = np.maximum(frame.astype(np.float64) / self.max_value, channeling_profile)
        combined = np.clip(combined, 0, 1)

        result = (combined * self.max_value).astype(np.uint16)
        result = result & 0x3FFF
        return result

    def trigger_channeling(self):
        self.channeling_active = True
        self.channeling_start_time = time.time()
        self.channeling_position = (
            np.random.randint(20, 108),
            np.random.randint(20, 108),
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 GAS CHANNELING TRIGGERED at "
            f"position {self.channeling_position}"
        )

    def stop_channeling(self):
        self.channeling_active = False
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Gas channeling ended")

    def pack_frame(self, raw_frame: np.ndarray, frame_id: int) -> bytes:
        header = struct.pack("<I", frame_id)
        frame_bytes = raw_frame.tobytes()
        return header + frame_bytes

    def api_health_check(self):
        try:
            start = time.perf_counter()
            resp = urllib.request.urlopen("http://localhost:8000/api/health", timeout=1.0)
            latency = (time.perf_counter() - start) * 1000
            data = json.loads(resp.read().decode())

            with self.stats_lock:
                self.api_success_count += 1
                self.api_latencies.append(latency)
                if len(self.api_latencies) > 100:
                    self.api_latencies = self.api_latencies[-100:]

            return data
        except Exception as e:
            with self.stats_lock:
                self.api_error_count += 1
            return None

    def run_api_monitor(self):
        while not self._stop:
            self.api_health_check()
            time.sleep(0.2)

    def run(self, fps: int = 10, duration: int = 60):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.host, self.port))
            print(f"[SIM] Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            print(f"[SIM] Connection refused. Is the server running?")
            return

        monitor_thread = threading.Thread(target=self.run_api_monitor, daemon=True)
        monitor_thread.start()

        interval = 1.0 / fps
        print(f"[SIM] Starting stress test at {fps} FPS for {duration} seconds")
        print(f"[SIM] Gas channeling events will trigger every 15 seconds")
        print("=" * 70)

        start_time = time.time()
        next_channeling = start_time + 10

        try:
            while time.time() - start_time < duration and not self._stop:
                frame_start = time.perf_counter()

                raw_frame = self.generate_normal_frame(self.frame_id)
                raw_frame = self.add_gas_channeling(raw_frame, self.frame_id)

                if self.channeling_active and time.time() - self.channeling_start_time > 8:
                    self.stop_channeling()

                if time.time() > next_channeling and not self.channeling_active:
                    self.trigger_channeling()
                    next_channeling = time.time() + 15

                packet = self.pack_frame(raw_frame, self.frame_id)
                sock.sendall(packet)

                self.frames_sent += 1
                self.frame_id += 1

                if self.frame_id % 100 == 0:
                    self.print_stats()

                elapsed = time.perf_counter() - frame_start
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n[SIM] Interrupted by user")
        finally:
            self._stop = True
            sock.close()
            monitor_thread.join(timeout=2.0)

            print("\n" + "=" * 70)
            print("STRESS TEST SUMMARY")
            print("=" * 70)
            self.print_stats(final=True)

    def print_stats(self, final: bool = False):
        with self.stats_lock:
            avg_latency = (
                np.mean(self.api_latencies) if self.api_latencies else 0
            )
            max_latency = (
                np.max(self.api_latencies) if self.api_latencies else 0
            )
            p95_latency = (
                np.percentile(self.api_latencies, 95) if self.api_latencies else 0
            )

        status = "🔥 CHANNELING" if self.channeling_active else "✅ NORMAL"
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] {status} | "
            f"Frames: {self.frames_sent} | "
            f"API: {self.api_success_count} OK, {self.api_error_count} FAIL | "
            f"Latency: {avg_latency:.1f}ms avg, {p95_latency:.1f}ms p95, {max_latency:.1f}ms max"
        )

        if self.api_error_count > 0 and final:
            error_rate = self.api_error_count / (self.api_success_count + self.api_error_count) * 100
            print(f"\n⚠️  API Error Rate: {error_rate:.2f}%")
            if error_rate > 5:
                print("❌ FAIL: High API error rate during stress test")
            else:
                print("✅ PASS: API error rate within acceptable range")

        if max_latency > 1000 and final:
            print(f"❌ FAIL: Max API latency {max_latency:.1f}ms exceeds 1000ms")
        elif final:
            print(f"✅ PASS: API latency within acceptable range")


def main():
    parser = argparse.ArgumentParser(description="Gas Channeling Stress Test")
    parser.add_argument("--host", default="127.0.0.1", help="TCP server host")
    parser.add_argument("--port", type=int, default=8888, help="TCP server port")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    args = parser.parse_args()

    sim = GasChannelingSimulator(host=args.host, port=args.port)
    sim.run(fps=args.fps, duration=args.duration)


if __name__ == "__main__":
    main()
