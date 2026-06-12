import multiprocessing as mp
import numpy as np
import asyncio
import time
import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class ComputeResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    compute_time_ms: float = 0.0


def _init_worker_process():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import warnings
    warnings.filterwarnings("ignore")


def _rbf_interpolate_worker(
    temperature: np.ndarray,
    grid_size: int,
    camera_height: int,
    camera_width: int,
    function: str,
    epsilon: float,
) -> np.ndarray:
    from scipy.interpolate import RBFInterpolator, RegularGridInterpolator

    h, w = temperature.shape

    valid_mask = ~np.isnan(temperature) & ~np.isinf(temperature)
    y_coords, x_coords = np.where(valid_mask)
    values = temperature[valid_mask]

    edge_y = np.concatenate([
        np.zeros(w), np.full(w, h - 1),
        np.arange(h), np.arange(h)
    ])
    edge_x = np.concatenate([
        np.arange(w), np.arange(w),
        np.zeros(h), np.full(h, w - 1)
    ])
    edge_values = np.concatenate([
        temperature[0, :], temperature[-1, :],
        temperature[:, 0], temperature[:, -1]
    ])

    y_coords = np.concatenate([y_coords, edge_y])
    x_coords = np.concatenate([x_coords, edge_x])
    values = np.concatenate([values, edge_values])

    points = np.column_stack([y_coords, x_coords])

    x = np.linspace(0, camera_width - 1, grid_size)
    y = np.linspace(0, camera_height - 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_coords = np.column_stack([yy.ravel(), xx.ravel()])

    if len(points) < 4:
        return np.full((grid_size, grid_size), np.nan, dtype=np.float32)

    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    try:
        rbf = RBFInterpolator(
            points,
            values,
            kernel=function,
            epsilon=epsilon,
            smoothing=0.1,
        )
        interpolated = rbf(grid_coords)
    except Exception:
        y_src = np.arange(camera_height)
        x_src = np.arange(camera_width)
        interp = RegularGridInterpolator(
            (y_src, x_src), temperature,
            method="linear", bounds_error=False, fill_value=None
        )
        interpolated = interp(grid_coords)

    result = interpolated.reshape((grid_size, grid_size))
    clipped = np.clip(result, min_val, max_val)
    return clipped.astype(np.float32)


def _temp_inversion_worker(
    raw_frame: np.ndarray,
    method: str,
    max_value: int,
    absorption_coeffs: np.ndarray,
    stefan_boltzmann_constant: float,
    planck_c1: float,
    planck_c2: float,
    wavelength_um: float,
) -> np.ndarray:
    normalized = raw_frame.astype(np.float64) / max_value
    radiance = normalized * absorption_coeffs

    if method == "stefan_boltzmann":
        sigma = stefan_boltzmann_constant
        emissivity = 0.85
        M = radiance / (emissivity * sigma)
        T = np.power(M, 0.25)
    else:
        c1 = planck_c1
        c2 = planck_c2
        lambda_um = wavelength_um
        emissivity = 0.85

        L = radiance * 1e3
        L = np.maximum(L, 1e-10)

        numerator = c2 / lambda_um
        denominator = np.log(1.0 + (c1 * emissivity) / (np.pi * lambda_um ** 5 * L))
        T = numerator / denominator

    T_celsius = T - 273.15
    return T_celsius.astype(np.float32)


def _generate_contours_worker(
    temperature: np.ndarray,
    levels: int,
    camera_height: int,
    camera_width: int,
) -> list:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    x = np.linspace(0, camera_width - 1, temperature.shape[1])
    y = np.linspace(0, camera_height - 1, temperature.shape[0])
    cs = ax.contour(x, y, temperature, levels=levels)

    contours = []
    for i, level_value in enumerate(cs.levels):
        level_paths = []
        for segs in cs.allsegs[i]:
            if len(segs) > 0:
                level_paths.append(segs.tolist())
        contours.append({"level": float(level_value), "paths": level_paths})

    plt.close(fig)
    return contours


class ComputationPool:
    _instance = None
    _lock = mp.Lock()

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

        cpu_count = mp.cpu_count()
        self.n_workers = max(1, min(cpu_count - 1, config.COMPUTE_POOL_WORKERS))

        self._executor: Optional[ProcessPoolExecutor] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._active_tasks = 0
        self._max_pending = config.COMPUTE_POOL_MAX_PENDING
        self._task_lock = asyncio.Lock()

        self._default_timeout = config.COMPUTE_DEFAULT_TIMEOUT

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker_process,
            mp_context=ctx,
        )
        logger.info(f"[ComputePool] Started with {self.n_workers} workers")

    def shutdown(self, wait: bool = True):
        if self._executor:
            self._executor.shutdown(wait=wait)
            logger.info("[ComputePool] Shutdown complete")

    async def _run_in_pool(
        self,
        func,
        *args,
        timeout: Optional[float] = None,
    ) -> ComputeResult:
        if self._executor is None:
            return ComputeResult(success=False, error="Pool not initialized")

        async with self._task_lock:
            if self._active_tasks >= self._max_pending:
                return ComputeResult(success=False, error="Too many pending tasks")
            self._active_tasks += 1

        start_time = time.perf_counter()
        timeout = timeout if timeout is not None else self._default_timeout

        try:
            result = await asyncio.wait_for(
                self._loop.run_in_executor(self._executor, func, *args),
                timeout=timeout,
            )
            compute_time = (time.perf_counter() - start_time) * 1000
            return ComputeResult(
                success=True,
                data=result,
                compute_time_ms=compute_time,
            )
        except TimeoutError:
            compute_time = (time.perf_counter() - start_time) * 1000
            return ComputeResult(
                success=False,
                error=f"Timeout after {timeout}s",
                compute_time_ms=compute_time,
            )
        except Exception as e:
            compute_time = (time.perf_counter() - start_time) * 1000
            return ComputeResult(
                success=False,
                error=str(e),
                compute_time_ms=compute_time,
            )
        finally:
            async with self._task_lock:
                self._active_tasks -= 1

    async def rbf_interpolate(
        self,
        temperature: np.ndarray,
        grid_size: int,
        camera_height: int,
        camera_width: int,
        function: str,
        epsilon: float,
        timeout: Optional[float] = None,
    ) -> ComputeResult:
        return await self._run_in_pool(
            _rbf_interpolate_worker,
            temperature,
            grid_size,
            camera_height,
            camera_width,
            function,
            epsilon,
            timeout=timeout,
        )

    async def temperature_inversion(
        self,
        raw_frame: np.ndarray,
        method: str,
        max_value: int,
        absorption_coeffs: np.ndarray,
        stefan_boltzmann_constant: float,
        planck_c1: float,
        planck_c2: float,
        wavelength_um: float,
        timeout: Optional[float] = None,
    ) -> ComputeResult:
        return await self._run_in_pool(
            _temp_inversion_worker,
            raw_frame,
            method,
            max_value,
            absorption_coeffs,
            stefan_boltzmann_constant,
            planck_c1,
            planck_c2,
            wavelength_um,
            timeout=timeout,
        )

    async def generate_contours(
        self,
        temperature: np.ndarray,
        levels: int,
        camera_height: int,
        camera_width: int,
        timeout: Optional[float] = None,
    ) -> ComputeResult:
        return await self._run_in_pool(
            _generate_contours_worker,
            temperature,
            levels,
            camera_height,
            camera_width,
            timeout=timeout,
        )

    def get_stats(self) -> dict:
        return {
            "n_workers": self.n_workers,
            "active_tasks": self._active_tasks,
            "max_pending": self._max_pending,
            "default_timeout": self._default_timeout,
        }


compute_pool = ComputationPool()
