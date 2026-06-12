import asyncio
import struct
import numpy as np
import time
from collections import deque
from .config import config
from .data_manager import data_manager
from .temperature_inversion import temp_inversion
from .rbf_interpolation import rbf_interpolator


class FrameProcessingQueue:
    def __init__(self, max_queue_size: int = 5):
        self._queue: deque = deque(maxlen=max_queue_size)
        self._processing: bool = False
        self._lock = asyncio.Lock()
        self._total_received: int = 0
        self._total_processed: int = 0
        self._total_dropped: int = 0
        self._last_process_time: float = 0.0

    async def add_frame(self, raw_frame: np.ndarray, frame_id: int):
        self._total_received += 1

        async with self._lock:
            if len(self._queue) >= self._queue.maxlen:
                self._total_dropped += 1
                if self._total_dropped % 10 == 0:
                    print(
                        f"[Queue] Dropped {self._total_dropped} frames, "
                        f"processing too slow"
                    )

            self._queue.append((raw_frame, frame_id, time.perf_counter()))

        if not self._processing:
            asyncio.create_task(self._process_loop())

    async def _process_loop(self):
        async with self._lock:
            if self._processing:
                return
            self._processing = True

        try:
            while True:
                async with self._lock:
                    if not self._queue:
                        break
                    raw_frame, frame_id, receive_time = self._queue.popleft()

                queue_latency = (time.perf_counter() - receive_time) * 1000
                if queue_latency > 500:
                    print(
                        f"[Queue] Frame {frame_id} queued for {queue_latency:.1f}ms, "
                        f"may be stale"
                    )

                process_start = time.perf_counter()
                try:
                    await self._process_single_frame(raw_frame, frame_id)
                    self._total_processed += 1
                except Exception as e:
                    print(f"[Queue] Error processing frame {frame_id}: {e}")

                self._last_process_time = (time.perf_counter() - process_start) * 1000

        finally:
            async with self._lock:
                self._processing = False

    async def _process_single_frame(self, raw_frame: np.ndarray, frame_id: int):
        data_manager.update_raw_frame(raw_frame)

        timeout = config.FRAME_PROCESS_TIMEOUT
        temp_frame = await temp_inversion.invert_async(
            raw_frame, method="planck", timeout=timeout
        )
        data_manager.update_temperature_frame(temp_frame)

        interp_frame = await rbf_interpolator.interpolate_async(
            temp_frame, timeout=timeout
        )
        data_manager.update_interpolated_temp(interp_frame)

    def get_stats(self) -> dict:
        return {
            "total_received": self._total_received,
            "total_processed": self._total_processed,
            "total_dropped": self._total_dropped,
            "queue_size": len(self._queue),
            "last_process_time_ms": self._last_process_time,
            "is_processing": self._processing,
        }


class TCPServer:
    def __init__(self):
        self.server = None
        self._running = False
        self._frame_size = config.CAMERA_WIDTH * config.CAMERA_HEIGHT * 2
        self._header_size = 4
        self._total_frame_size = self._header_size + self._frame_size
        self._frame_queue = FrameProcessingQueue(max_queue_size=config.FRAME_QUEUE_MAX_SIZE)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"[TCP] Client connected: {addr}")

        buffer = bytearray()
        frame_count = 0

        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(
                        reader.read(65536),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if not data:
                    break

                buffer.extend(data)

                while len(buffer) >= self._total_frame_size:
                    header = bytes(buffer[: self._header_size])
                    frame_id = struct.unpack("<I", header)[0]

                    frame_bytes = bytes(
                        buffer[self._header_size : self._total_frame_size]
                    )
                    raw_frame = self._parse_frame(frame_bytes)

                    await self._frame_queue.add_frame(raw_frame, frame_id)

                    buffer = buffer[self._total_frame_size :]
                    frame_count += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TCP] Error handling client {addr}: {e}")
        finally:
            print(
                f"[TCP] Client disconnected: {addr}, frames received: {frame_count}"
            )
            print(f"[TCP] Queue stats: {self._frame_queue.get_stats()}")
            writer.close()
            await writer.wait_closed()

    def _parse_frame(self, frame_bytes: bytes) -> np.ndarray:
        raw_data = np.frombuffer(frame_bytes, dtype=np.uint16)
        raw_data = raw_data.reshape((config.CAMERA_HEIGHT, config.CAMERA_WIDTH))
        raw_data = raw_data & 0x3FFF
        return raw_data.astype(np.uint16)

    def get_queue_stats(self) -> dict:
        return self._frame_queue.get_stats()

    async def start(self):
        self._running = True
        self.server = await asyncio.start_server(
            self.handle_client,
            config.TCP_HOST,
            config.TCP_PORT,
            limit=4 * self._frame_size,
        )
        addr = self.server.sockets[0].getsockname()
        print(f"[TCP] Server listening on {addr}")

        asyncio.create_task(self._monitor_queue())

        async with self.server:
            await self.server.serve_forever()

    async def _monitor_queue(self):
        while self._running:
            await asyncio.sleep(5.0)
            stats = self._frame_queue.get_stats()
            if stats["total_received"] > 0:
                print(
                    f"[TCP] Queue: {stats['queue_size']} queued, "
                    f"{stats['total_processed']}/{stats['total_received']} processed, "
                    f"{stats['total_dropped']} dropped, "
                    f"{stats['last_process_time_ms']:.1f}ms/frame"
                )

    async def stop(self):
        self._running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        print("[TCP] Server stopped")


tcp_server = TCPServer()
