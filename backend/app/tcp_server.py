import asyncio
import struct
import numpy as np
import threading
from .config import config
from .data_manager import data_manager
from .temperature_inversion import temp_inversion
from .rbf_interpolation import rbf_interpolator


class TCPServer:
    def __init__(self):
        self.server = None
        self._running = False
        self._frame_size = config.CAMERA_WIDTH * config.CAMERA_HEIGHT * 2
        self._header_size = 4
        self._total_frame_size = self._header_size + self._frame_size

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"[TCP] Client connected: {addr}")

        buffer = bytearray()
        frame_count = 0

        try:
            while self._running:
                data = await reader.read(4096)
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

                    self._process_frame(raw_frame, frame_id)

                    buffer = buffer[self._total_frame_size :]
                    frame_count += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TCP] Error handling client {addr}: {e}")
        finally:
            print(f"[TCP] Client disconnected: {addr}, frames received: {frame_count}")
            writer.close()
            await writer.wait_closed()

    def _parse_frame(self, frame_bytes: bytes) -> np.ndarray:
        raw_data = np.frombuffer(frame_bytes, dtype=np.uint16)
        raw_data = raw_data.reshape((config.CAMERA_HEIGHT, config.CAMERA_WIDTH))
        raw_data = raw_data & 0x3FFF
        return raw_data.astype(np.uint16)

    def _process_frame(self, raw_frame: np.ndarray, frame_id: int):
        data_manager.update_raw_frame(raw_frame)

        temp_frame = temp_inversion.invert(raw_frame, method="planck")
        data_manager.update_temperature_frame(temp_frame)

        interp_frame = rbf_interpolator.interpolate(temp_frame)
        data_manager.update_interpolated_temp(interp_frame)

    async def start(self):
        self._running = True
        self.server = await asyncio.start_server(
            self.handle_client,
            config.TCP_HOST,
            config.TCP_PORT,
            limit=2 * self._frame_size,
        )
        addr = self.server.sockets[0].getsockname()
        print(f"[TCP] Server listening on {addr}")

        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        self._running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        print("[TCP] Server stopped")


tcp_server = TCPServer()
