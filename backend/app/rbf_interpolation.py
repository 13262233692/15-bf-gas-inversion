import numpy as np
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import KDTree
from typing import Optional, Tuple
from .config import config
from .computation_pool import compute_pool


class RBFInterpolation:
    def __init__(self):
        self.grid_size = config.INTERPOLATION_GRID_SIZE
        self.function = config.RBF_FUNCTION
        self.epsilon = config.RBF_EPSILON
        self._grid_coords = self._create_grid()
        self._cache = None
        self._cache_hash = None
        self._kdtree = None
        self._last_points_hash = None

    def _create_grid(self):
        x = np.linspace(0, config.CAMERA_WIDTH - 1, self.grid_size)
        y = np.linspace(0, config.CAMERA_HEIGHT - 1, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([yy.ravel(), xx.ravel()])

    def _extract_valid_points(self, temperature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        return points, values

    def _compute_fast_rbf(
        self,
        points: np.ndarray,
        values: np.ndarray,
        k_neighbors: int = 32,
    ) -> np.ndarray:
        n_query = len(self._grid_coords)
        n_points = len(points)

        min_val = np.nanmin(values)
        max_val = np.nanmax(values)

        points_hash = hash(points.tobytes())
        if self._last_points_hash != points_hash or self._kdtree is None:
            self._kdtree = KDTree(points)
            self._last_points_hash = points_hash

        block_size = 8192
        result = np.zeros(n_query, dtype=np.float32)

        epsilon = self.epsilon

        for i in range(0, n_query, block_size):
            block_end = min(i + block_size, n_query)
            query_block = self._grid_coords[i:block_end]

            distances, indices = self._kdtree.query(
                query_block,
                k=min(k_neighbors, n_points),
                workers=-1,
            )

            if indices.ndim == 1:
                indices = indices[:, np.newaxis]
                distances = distances[:, np.newaxis]

            neighbor_values = values[indices]

            r = distances * epsilon
            phi = np.sqrt(r * r + 1.0)

            weights = 1.0 / (phi + 1e-10)
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            result[i:block_end] = np.sum(weights * neighbor_values, axis=1).astype(np.float32)

        result = np.clip(result, min_val, max_val)
        return result

    def interpolate(self, temperature: np.ndarray, use_approx: bool = True) -> np.ndarray:
        if temperature.shape != (config.CAMERA_HEIGHT, config.CAMERA_WIDTH):
            raise ValueError(
                f"Expected temperature shape ({config.CAMERA_HEIGHT}, {config.CAMERA_WIDTH}), "
                f"got {temperature.shape}"
            )

        points, values = self._extract_valid_points(temperature)

        if len(points) < 4:
            return np.full(
                (self.grid_size, self.grid_size), np.nan, dtype=np.float32
            )

        min_val = np.nanmin(values)
        max_val = np.nanmax(values)

        if use_approx:
            interpolated = self._compute_fast_rbf(points, values)
        else:
            try:
                rbf = RBFInterpolator(
                    points,
                    values,
                    kernel=self.function,
                    epsilon=self.epsilon,
                    smoothing=0.1,
                )
                interpolated = rbf(self._grid_coords)
            except Exception as e:
                print(f"[RBF] High-precision interpolation error: {e}, using approx")
                interpolated = self._compute_fast_rbf(points, values)

        result = interpolated.reshape((self.grid_size, self.grid_size))
        if not use_approx:
            result = np.clip(result, min_val, max_val)
        return result.astype(np.float32)

    async def interpolate_async(
        self,
        temperature: np.ndarray,
        timeout: Optional[float] = None,
        use_high_precision: bool = False,
    ) -> np.ndarray:
        if temperature.shape != (config.CAMERA_HEIGHT, config.CAMERA_WIDTH):
            raise ValueError(
                f"Expected temperature shape ({config.CAMERA_HEIGHT}, {config.CAMERA_WIDTH}), "
                f"got {temperature.shape}"
            )

        temp_hash = hash(temperature.tobytes())
        if self._cache_hash == temp_hash and self._cache is not None:
            return self._cache.copy()

        points, values = self._extract_valid_points(temperature)

        if len(points) < 4:
            result = np.full(
                (self.grid_size, self.grid_size), np.nan, dtype=np.float32
            )
            self._cache = result
            self._cache_hash = temp_hash
            return result

        if not use_high_precision:
            interpolated = self._compute_fast_rbf(points, values)
            result = interpolated.reshape((self.grid_size, self.grid_size))
            self._cache = result
            self._cache_hash = temp_hash
            return result

        result = await compute_pool.rbf_interpolate(
            temperature,
            self.grid_size,
            config.CAMERA_HEIGHT,
            config.CAMERA_WIDTH,
            self.function,
            self.epsilon,
            timeout=timeout,
        )

        if not result.success:
            print(f"[RBF] Async interpolation failed: {result.error}, using fast approx")
            interpolated = self._compute_fast_rbf(points, values)
            result_data = interpolated.reshape((self.grid_size, self.grid_size))
            self._cache = result_data
            self._cache_hash = temp_hash
            return result_data

        self._cache = result.data
        self._cache_hash = temp_hash
        return result.data

    def _bilinear_fallback(self, temperature: np.ndarray) -> np.ndarray:
        y = np.arange(config.CAMERA_HEIGHT)
        x = np.arange(config.CAMERA_WIDTH)
        interp = RegularGridInterpolator(
            (y, x), temperature, method="linear", bounds_error=False, fill_value=None
        )

        grid_y = np.linspace(0, config.CAMERA_HEIGHT - 1, self.grid_size)
        grid_x = np.linspace(0, config.CAMERA_WIDTH - 1, self.grid_size)
        gy, gx = np.meshgrid(grid_y, grid_x, indexing="ij")
        points = np.column_stack([gy.ravel(), gx.ravel()])

        result = interp(points).reshape((self.grid_size, self.grid_size))
        return result.astype(np.float32)

    def generate_contours(
        self, temperature: np.ndarray, levels: int = 20
    ) -> list:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            fig, ax = plt.subplots()
            x = np.linspace(0, config.CAMERA_WIDTH - 1, temperature.shape[1])
            y = np.linspace(0, config.CAMERA_HEIGHT - 1, temperature.shape[0])
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
        except Exception as e:
            print(f"[RBF] Contour generation error: {e}")
            return []

    async def generate_contours_async(
        self,
        temperature: np.ndarray,
        levels: int = 20,
        timeout: Optional[float] = None,
    ) -> list:
        result = await compute_pool.generate_contours(
            temperature,
            levels,
            config.CAMERA_HEIGHT,
            config.CAMERA_WIDTH,
            timeout=timeout,
        )

        if not result.success:
            print(f"[RBF] Async contour generation failed: {result.error}")
            return []

        return result.data

    def get_grid_size(self) -> int:
        return self.grid_size


rbf_interpolator = RBFInterpolation()
