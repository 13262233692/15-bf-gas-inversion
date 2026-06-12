import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from typing import Tuple, Dict, Any
from .config import config
from .computation_pool import compute_pool


class GasVelocityEstimator:
    def __init__(self):
        self.grid_size = config.INTERPOLATION_GRID_SIZE

        self.air_thermal_conductivity = config.AIR_THERMAL_CONDUCTIVITY
        self.air_viscosity = config.AIR_VISCOSITY
        self.air_density = config.AIR_DENSITY
        self.air_prandtl = config.AIR_PRANDTL
        self.characteristic_length = config.CHARACTERISTIC_LENGTH_M

        self.burden_heat_capacity = 1200.0
        self.convection_coeff_ref = 250.0

        self.total_gas_flow_m3s = config.TOTAL_GAS_FLOW_M3S
        self.furnace_pressure_kpa = config.FURNACE_PRESSURE_KPA
        self.furnace_inner_diameter_m = config.FURNACE_INNER_DIAMETER_M

        self.velocity_warning_threshold = config.VELOCITY_WARNING_THRESHOLD
        self.velocity_danger_threshold = config.VELOCITY_DANGER_THRESHOLD

        self._laplacian_kernel = None
        self._init_solver()

    def _init_solver(self):
        k = 3
        y, x = np.mgrid[-k:k + 1, -k:k + 1]
        r2 = x * x + y * y
        self._gaussian_kernel = np.exp(-r2 / (2 * (k / 2) ** 2))
        self._gaussian_kernel /= self._gaussian_kernel.sum()

    def _compute_temperature_gradient(
        self,
        temperature: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = gaussian_filter(temperature.astype(np.float64), sigma=1.2)

        dTdy = sobel(T, axis=0, mode="mirror")
        dTdx = sobel(T, axis=1, mode="mirror")

        grad_mag = np.sqrt(dTdx ** 2 + dTdy ** 2)
        grad_mag = gaussian_filter(grad_mag, sigma=0.8)

        return dTdx, dTdy, grad_mag

    def _estimate_velocity_magnitude(
        self,
        grad_mag: np.ndarray,
        temperature: np.ndarray,
    ) -> np.ndarray:
        T_ref = np.mean(temperature)
        T_min = np.min(temperature)
        T_max = np.max(temperature)
        delta_T_global = max(T_max - T_min, 50.0)

        grad_normalized = grad_mag / (np.max(grad_mag) + 1e-6)

        delta_T_local = np.abs(temperature - T_ref)
        anomaly_factor = np.power(delta_T_local / delta_T_global + 0.15, 1.4)

        base_velocity = 0.6 + 3.2 * grad_normalized

        hotspot_boost = np.power(
            np.clip((temperature - T_ref) / (delta_T_global + 1e-6), 0, 1),
            0.8,
        ) * 4.5

        v_mag = base_velocity * (0.7 + 1.3 * anomaly_factor) + hotspot_boost

        T_hot_factor = np.clip((temperature - T_min) / (delta_T_global + 1e-8), 0.3, 1.8)
        v_mag = v_mag * (0.85 + 0.6 * T_hot_factor)

        center_factor = self._compute_center_bias(temperature)
        v_mag = v_mag * (0.55 + 0.9 * center_factor)

        return np.clip(v_mag, 0.15, 8.5).astype(np.float32)

    def _compute_center_bias(self, temperature: np.ndarray) -> np.ndarray:
        h, w = temperature.shape
        cy, cx = h / 2, w / 2
        y, x = np.mgrid[0:h, 0:w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2)
        norm_dist = dist / max_dist
        return np.exp(-norm_dist ** 2 * 2.0)

    def _estimate_velocity_direction(
        self,
        dTdx: np.ndarray,
        dTdy: np.ndarray,
        grad_mag: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad_safe = np.maximum(grad_mag, 1e-6)

        dx_norm = -dTdx / grad_safe
        dy_norm = -dTdy / grad_safe

        h, w = dTdx.shape
        cy, cx = h / 2, w / 2
        y, x = np.mgrid[0:h, 0:w]

        radial_dx = (x - cx) / (np.maximum(np.sqrt((x - cx) ** 2 + (y - cy) ** 2), 1.0))
        radial_dy = (y - cy) / (np.maximum(np.sqrt((x - cx) ** 2 + (y - cy) ** 2), 1.0))

        alpha = 0.55
        dx_dir = (1 - alpha) * dx_norm + alpha * radial_dx
        dy_dir = (1 - alpha) * dy_norm + alpha * radial_dy

        dir_mag = np.sqrt(dx_dir ** 2 + dy_dir ** 2)
        dir_safe = np.maximum(dir_mag, 1e-6)
        dx_dir /= dir_safe
        dy_dir /= dir_safe

        return dx_dir.astype(np.float32), dy_dir.astype(np.float32)

    def _enforce_continuity(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        max_iter: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = vx.shape

        dx = 1.0 / w
        dy = 1.0 / h

        vx_new = vx.copy()
        vy_new = vy.copy()

        for _ in range(max_iter):
            div_v = (
                (vx_new[1:-1, 2:] - vx_new[1:-1, :-2]) / (2 * dx)
                + (vy_new[2:, 1:-1] - vy_new[:-2, 1:-1]) / (2 * dy)
            )

            correction = div_v * 0.25

            vx_new[1:-1, 2:] -= correction * dx
            vx_new[1:-1, :-2] += correction * dx
            vy_new[2:, 1:-1] -= correction * dy
            vy_new[:-2, 1:-1] += correction * dy

        return vx_new.astype(np.float32), vy_new.astype(np.float32)

    def _apply_global_flow_constraint(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        v_mag = np.sqrt(vx ** 2 + vy ** 2)
        h, w = vx.shape

        cell_area_m2 = (self.furnace_inner_diameter_m ** 2 * np.pi / 4) / (h * w)
        current_flow = np.sum(v_mag) * cell_area_m2

        if current_flow > 1e-8:
            ratio = self.total_gas_flow_m3s / current_flow
            ratio_log = np.log(np.clip(ratio, 0.12, 8.0))
            soft_scale = np.exp(ratio_log * 0.22)
            vx *= soft_scale
            vy *= soft_scale

        return vx, vy

    def _enforce_boundary_conditions(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = vx.shape

        border_width = 3
        for i in range(border_width):
            fade = 0.65 + 0.35 * (i + 1) / (border_width + 1)

            vx[i, :] *= fade
            vx[-(i + 1), :] *= fade
            vx[:, i] *= fade
            vx[:, -(i + 1)] *= fade

            vy[i, :] *= fade
            vy[-(i + 1), :] *= fade
            vy[:, i] *= fade
            vy[:, -(i + 1)] *= fade

        return vx, vy

    def estimate_velocity_field(
        self,
        temperature: np.ndarray,
    ) -> Dict[str, Any]:
        if temperature.shape != (self.grid_size, self.grid_size):
            from scipy.ndimage import zoom
            scale = self.grid_size / temperature.shape[0]
            temperature = zoom(temperature, scale, order=1)

        dTdx, dTdy, grad_mag = self._compute_temperature_gradient(temperature)

        v_mag = self._estimate_velocity_magnitude(grad_mag, temperature)

        dx_dir, dy_dir = self._estimate_velocity_direction(dTdx, dTdy, grad_mag)

        vx = v_mag * dx_dir
        vy = v_mag * dy_dir

        vx, vy = self._enforce_continuity(vx, vy)

        vx, vy = self._apply_global_flow_constraint(vx, vy)

        vx, vy = self._enforce_boundary_conditions(vx, vy)

        v_mag_final = np.sqrt(vx ** 2 + vy ** 2).astype(np.float32)

        warning_mask = v_mag_final >= self.velocity_warning_threshold
        danger_mask = v_mag_final >= self.velocity_danger_threshold

        hotspots = self._detect_hotspots(v_mag_final, warning_mask)

        return {
            "vx": vx.astype(np.float32),
            "vy": vy.astype(np.float32),
            "magnitude": v_mag_final,
            "min_velocity": float(np.min(v_mag_final)),
            "max_velocity": float(np.max(v_mag_final)),
            "avg_velocity": float(np.mean(v_mag_final)),
            "warning_mask": warning_mask,
            "danger_mask": danger_mask,
            "warning_count": int(np.sum(warning_mask)),
            "danger_count": int(np.sum(danger_mask)),
            "hotspots": hotspots,
            "total_flow_m3s": self.total_gas_flow_m3s,
            "furnace_pressure_kpa": self.furnace_pressure_kpa,
        }

    def _detect_hotspots(
        self,
        v_mag: np.ndarray,
        warning_mask: np.ndarray,
        min_size: int = 10,
    ) -> list:
        from scipy.ndimage import label

        labeled, num_features = label(warning_mask.astype(np.int32))
        hotspots = []

        for i in range(1, num_features + 1):
            region_mask = labeled == i
            if np.sum(region_mask) < min_size:
                continue

            coords = np.where(region_mask)
            cy, cx = np.mean(coords[0]), np.mean(coords[1])
            region_vels = v_mag[region_mask]
            max_vel = float(np.max(region_vels))
            avg_vel = float(np.mean(region_vels))
            size = int(np.sum(region_mask))

            level = "danger" if max_vel >= self.velocity_danger_threshold else "warning"

            hotspots.append({
                "id": len(hotspots) + 1,
                "center_x": float(cx),
                "center_y": float(cy),
                "max_velocity": max_vel,
                "avg_velocity": avg_vel,
                "size_pixels": size,
                "severity": level,
            })

        hotspots.sort(key=lambda h: h["max_velocity"], reverse=True)
        return hotspots[:8]

    async def estimate_velocity_async(
        self,
        temperature: np.ndarray,
        timeout: float = None,
    ) -> Dict[str, Any]:
        if timeout is None:
            timeout = config.VELOCITY_COMPUTE_TIMEOUT

        result = await compute_pool._run_in_pool(
            _velocity_worker,
            temperature,
            self.grid_size,
            self.air_thermal_conductivity,
            self.air_viscosity,
            self.air_density,
            self.air_prandtl,
            self.characteristic_length,
            self.total_gas_flow_m3s,
            self.furnace_inner_diameter_m,
            self.velocity_warning_threshold,
            self.velocity_danger_threshold,
            timeout=timeout,
        )

        if not result.success:
            print(f"[Velocity] Async estimation failed: {result.error}, using local compute")
            return self.estimate_velocity_field(temperature)

        raw = result.data
        h, w = raw["magnitude_shape"]
        raw["vx"] = np.array(raw["vx"], dtype=np.float32).reshape(h, w)
        raw["vy"] = np.array(raw["vy"], dtype=np.float32).reshape(h, w)
        raw["magnitude"] = np.array(raw["magnitude_flat"], dtype=np.float32).reshape(h, w)
        del raw["magnitude_flat"]

        return raw

    def set_operating_parameters(
        self,
        total_gas_flow_m3s: float = None,
        furnace_pressure_kpa: float = None,
        velocity_warning_threshold: float = None,
        velocity_danger_threshold: float = None,
    ):
        if total_gas_flow_m3s is not None:
            self.total_gas_flow_m3s = total_gas_flow_m3s
        if furnace_pressure_kpa is not None:
            self.furnace_pressure_kpa = furnace_pressure_kpa
        if velocity_warning_threshold is not None:
            self.velocity_warning_threshold = velocity_warning_threshold
        if velocity_danger_threshold is not None:
            self.velocity_danger_threshold = velocity_danger_threshold


def _velocity_worker(
    temperature: np.ndarray,
    grid_size: int,
    air_thermal_conductivity: float,
    air_viscosity: float,
    air_density: float,
    air_prandtl: float,
    characteristic_length: float,
    total_gas_flow_m3s: float,
    furnace_inner_diameter_m: float,
    warning_threshold: float,
    danger_threshold: float,
) -> Dict[str, Any]:
    from scipy.ndimage import gaussian_filter, sobel, label, zoom

    src_shape = temperature.shape
    if src_shape[0] != grid_size:
        scale = grid_size / src_shape[0]
        temperature = zoom(temperature, scale, order=1)

    T = gaussian_filter(temperature.astype(np.float64), sigma=1.2)
    dTdy = sobel(T, axis=0, mode="mirror")
    dTdx = sobel(T, axis=1, mode="mirror")
    grad_mag = np.sqrt(dTdx ** 2 + dTdy ** 2)
    grad_mag = gaussian_filter(grad_mag, sigma=0.8)

    T_ref = np.mean(temperature)
    T_min = np.min(temperature)
    T_max = np.max(temperature)
    delta_T_global = max(T_max - T_min, 50.0)
    grad_normalized = grad_mag / (np.max(grad_mag) + 1e-6)
    delta_T_local = np.abs(temperature - T_ref)
    anomaly_factor = np.power(delta_T_local / delta_T_global + 0.15, 1.4)
    base_velocity = 0.6 + 3.2 * grad_normalized
    hotspot_boost = np.power(
        np.clip((temperature - T_ref) / (delta_T_global + 1e-6), 0, 1),
        0.8,
    ) * 4.5
    v_mag = base_velocity * (0.7 + 1.3 * anomaly_factor) + hotspot_boost
    T_hot_factor = np.clip((temperature - T_min) / (delta_T_global + 1e-8), 0.3, 1.8)
    v_mag = v_mag * (0.85 + 0.6 * T_hot_factor)

    h, w = temperature.shape
    cy, cx = h / 2, w / 2
    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    norm_dist = dist / max_dist
    center_factor = np.exp(-norm_dist ** 2 * 2.0)
    v_mag = v_mag * (0.55 + 0.9 * center_factor)
    v_mag = np.clip(v_mag, 0.15, 8.5).astype(np.float32)

    grad_safe = np.maximum(grad_mag, 1e-6)
    dx_norm = -dTdx / grad_safe
    dy_norm = -dTdy / grad_safe

    radial_dx = (x - cx) / (np.maximum(np.sqrt((x - cx) ** 2 + (y - cy) ** 2), 1.0))
    radial_dy = (y - cy) / (np.maximum(np.sqrt((x - cx) ** 2 + (y - cy) ** 2), 1.0))

    alpha = 0.55
    dx_dir = (1 - alpha) * dx_norm + alpha * radial_dx
    dy_dir = (1 - alpha) * dy_norm + alpha * radial_dy

    dir_mag = np.sqrt(dx_dir ** 2 + dy_dir ** 2)
    dir_safe = np.maximum(dir_mag, 1e-6)
    dx_dir /= dir_safe
    dy_dir /= dir_safe

    vx = v_mag * dx_dir.astype(np.float32)
    vy = v_mag * dy_dir.astype(np.float32)

    dx = 1.0 / w
    dy = 1.0 / h
    vx_new = vx.copy()
    vy_new = vy.copy()

    for _ in range(50):
        div_v = (
            (vx_new[1:-1, 2:] - vx_new[1:-1, :-2]) / (2 * dx)
            + (vy_new[2:, 1:-1] - vy_new[:-2, 1:-1]) / (2 * dy)
        )
        correction = div_v * 0.25
        vx_new[1:-1, 2:] -= correction * dx
        vx_new[1:-1, :-2] += correction * dx
        vy_new[2:, 1:-1] -= correction * dy
        vy_new[:-2, 1:-1] += correction * dy

    vx, vy = vx_new, vy_new

    v_mag_check = np.sqrt(vx ** 2 + vy ** 2)
    cell_area_m2 = (furnace_inner_diameter_m ** 2 * np.pi / 4) / (h * w)
    current_flow = np.sum(v_mag_check) * cell_area_m2

    if current_flow > 1e-8:
        ratio = total_gas_flow_m3s / current_flow
        ratio_log = np.log(np.clip(ratio, 0.12, 8.0))
        soft_scale = np.exp(ratio_log * 0.22)
        vx *= soft_scale
        vy *= soft_scale

    border_width = 3
    for i in range(border_width):
        fade = 0.65 + 0.35 * (i + 1) / (border_width + 1)
        vx[i, :] *= fade
        vx[-(i + 1), :] *= fade
        vx[:, i] *= fade
        vx[:, -(i + 1)] *= fade
        vy[i, :] *= fade
        vy[-(i + 1), :] *= fade
        vy[:, i] *= fade
        vy[:, -(i + 1)] *= fade

    v_mag_final = np.sqrt(vx ** 2 + vy ** 2).astype(np.float32)
    warning_mask = v_mag_final >= warning_threshold
    danger_mask = v_mag_final >= danger_threshold

    labeled, num_features = label(warning_mask.astype(np.int32))
    hotspots = []
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        if np.sum(region_mask) < 10:
            continue
        coords = np.where(region_mask)
        rcy, rcx = np.mean(coords[0]), np.mean(coords[1])
        region_vels = v_mag_final[region_mask]
        max_vel = float(np.max(region_vels))
        avg_vel = float(np.mean(region_vels))
        size = int(np.sum(region_mask))
        level = "danger" if max_vel >= danger_threshold else "warning"
        hotspots.append({
            "id": len(hotspots) + 1,
            "center_x": float(rcx),
            "center_y": float(rcy),
            "max_velocity": max_vel,
            "avg_velocity": avg_vel,
            "size_pixels": size,
            "severity": level,
        })

    hotspots.sort(key=lambda h: h["max_velocity"], reverse=True)
    hotspots = hotspots[:8]

    vx_flat = vx.astype(np.float32).flatten().tolist()
    vy_flat = vy.astype(np.float32).flatten().tolist()
    mag_flat = v_mag_final.flatten().tolist()

    return {
        "vx": vx_flat,
        "vy": vy_flat,
        "magnitude_flat": mag_flat,
        "magnitude_shape": [h, w],
        "min_velocity": float(np.min(v_mag_final)),
        "max_velocity": float(np.max(v_mag_final)),
        "avg_velocity": float(np.mean(v_mag_final)),
        "warning_count": int(np.sum(warning_mask)),
        "danger_count": int(np.sum(danger_mask)),
        "hotspots": hotspots,
        "total_flow_m3s": total_gas_flow_m3s,
    }


gas_velocity_estimator = GasVelocityEstimator()
