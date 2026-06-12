import socket
import struct
import time
import numpy as np
import argparse
from app.config import config


class CameraSimulator:
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.frame_id = 0
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT
        self.max_value = 2 ** config.CAMERA_BIT_DEPTH - 1

    def generate_frame(self, frame_id: int) -> np.ndarray:
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

    def pack_frame(self, raw_frame: np.ndarray, frame_id: int) -> bytes:
        header = struct.pack("<I", frame_id)
        frame_bytes = raw_frame.tobytes()
        return header + frame_bytes

    def run(self, fps: int = 10):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.host, self.port))
            print(f"[SIM] Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            print(f"[SIM] Connection refused. Is the server running?")
            return

        interval = 1.0 / fps
        print(f"[SIM] Starting simulation at {fps} FPS...")

        try:
            while True:
                raw_frame = self.generate_frame(self.frame_id)
                packet = self.pack_frame(raw_frame, self.frame_id)
                sock.sendall(packet)
                self.frame_id += 1

                if self.frame_id % 100 == 0:
                    print(f"[SIM] Sent {self.frame_id} frames")

                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n[SIM] Stopped. Total frames sent: {self.frame_id}")
        finally:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description="IR Camera Data Simulator")
    parser.add_argument("--host", default="127.0.0.1", help="TCP server host")
    parser.add_argument("--port", type=int, default=8888, help="TCP server port")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    args = parser.parse_args()

    sim = CameraSimulator(host=args.host, port=args.port)
    sim.run(fps=args.fps)


if __name__ == "__main__":
    main()
