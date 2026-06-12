import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay
from .config import config


class RBFInterpolation:
    def __init__(self):
        self.grid_size = config.INTERPOLATION_GRID_SIZE
        self.function = config.RBF_FUNCTION
        self.epsilon = config.RBF_EPSILON
        self._grid_coords = self._create_grid()
        self._mask = None

    def _create_grid(self):
        x = np.linspace(0, config.CAMERA_WIDTH - 1, self.grid_size)
        y = np.linspace(0, config.CAMERA_HEIGHT - 1, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([yy.ravel(), xx.ravel()])

    def _extract_valid_points(self, temperature: np.ndarray):
        h, w = temperature.shape
        valid_mask = ~np.isnan(temperature) & ~np.isinf(temperature)

        y_coords, x_coords = np.where(valid_mask)
        values = temperature[valid_mask]

        edge_y = np.concatenate(
            [
                np.zeros(w),
                np.full(w, h - 1),
                np.arange(h),
                np.arange(h),
            ]
        )
        edge_x = np.concatenate(
            [
                np.arange(w),
                np.arange(w),
                np.zeros(h),
                np.full(h, w - 1),
            ]
        )
        edge_values = np.concatenate(
            [
                temperature[0, :],
                temperature[-1, :],
                temperature[:, 0],
                temperature[:, -1],
            ]
        )

        y_coords = np.concatenate([y_coords, edge_y])
        x_coords = np.concatenate([x_coords, edge_x])
        values = np.concatenate([values, edge_values])

        points = np.column_stack([y_coords, x_coords])
        return points, values

    def interpolate(self, temperature: np.ndarray) -> np.ndarray:
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
            print(f"[RBF] Interpolation error: {e}, falling back to bilinear")
            interpolated = self._bilinear_fallback(temperature)

        result = interpolated.reshape((self.grid_size, self.grid_size))
        clipped = np.clip(result, np.nanmin(values), np.nanmax(values))
        return clipped.astype(np.float32)

    def _bilinear_fallback(self, temperature: np.ndarray) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator

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

    def get_grid_size(self) -> int:
        return self.grid_size


rbf_interpolator = RBFInterpolation()
