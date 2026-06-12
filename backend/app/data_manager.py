import threading
import time
import numpy as np
from typing import Optional, Callable, List
from .config import config


class DataManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._raw_frame: Optional[np.ndarray] = None
        self._temperature_frame: Optional[np.ndarray] = None
        self._interpolated_temp: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._last_update: float = 0.0
        self._fps: float = 0.0
        self._frame_times: List[float] = []
        self._listeners: List[Callable] = []
        self._data_lock = threading.RLock()

    def update_raw_frame(self, raw_data: np.ndarray):
        with self._data_lock:
            self._raw_frame = raw_data.copy()
            self._frame_count += 1
            now = time.time()
            self._frame_times.append(now)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            if len(self._frame_times) >= 2:
                self._fps = len(self._frame_times) / (
                    self._frame_times[-1] - self._frame_times[0]
                )
            self._last_update = now

    def update_temperature_frame(self, temp_data: np.ndarray):
        with self._data_lock:
            self._temperature_frame = temp_data.copy()

    def update_interpolated_temp(self, interp_data: np.ndarray):
        with self._data_lock:
            self._interpolated_temp = interp_data.copy()
            self._notify_listeners()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._data_lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def get_temperature_frame(self) -> Optional[np.ndarray]:
        with self._data_lock:
            return (
                self._temperature_frame.copy()
                if self._temperature_frame is not None
                else None
            )

    def get_interpolated_temp(self) -> Optional[np.ndarray]:
        with self._data_lock:
            return (
                self._interpolated_temp.copy()
                if self._interpolated_temp is not None
                else None
            )

    def get_stats(self) -> dict:
        with self._data_lock:
            stats = {
                "frame_count": self._frame_count,
                "fps": round(self._fps, 2),
                "last_update": self._last_update,
            }
            if self._temperature_frame is not None:
                stats["min_temp"] = float(np.min(self._temperature_frame))
                stats["max_temp"] = float(np.max(self._temperature_frame))
                stats["avg_temp"] = float(np.mean(self._temperature_frame))
            return stats

    def add_listener(self, callback: Callable):
        with self._data_lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        with self._data_lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify_listeners(self):
        for listener in self._listeners:
            try:
                listener(self)
            except Exception:
                pass


data_manager = DataManager()
