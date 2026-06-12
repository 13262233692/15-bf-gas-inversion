import numpy as np
import os
from typing import Optional
from .config import config
from .computation_pool import compute_pool


class TemperatureInversion:
    def __init__(self):
        self.max_value = 2 ** config.CAMERA_BIT_DEPTH - 1
        self.absorption_coeffs = self._load_calibration()

    def _load_calibration(self) -> np.ndarray:
        if os.path.exists(config.CALIBRATION_FILE):
            return np.load(config.CALIBRATION_FILE)
        coeffs = self._generate_default_calibration()
        os.makedirs(os.path.dirname(config.CALIBRATION_FILE), exist_ok=True)
        np.save(config.CALIBRATION_FILE, coeffs)
        return coeffs

    def _generate_default_calibration(self) -> np.ndarray:
        x = np.linspace(0, 1, config.CAMERA_WIDTH)
        y = np.linspace(0, 1, config.CAMERA_HEIGHT)
        xx, yy = np.meshgrid(x, y)
        center_dist = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        edge_factor = 0.85 + 0.15 * np.cos(center_dist * np.pi)
        base_absorption = 0.92
        coeffs = base_absorption * edge_factor
        coeffs = np.clip(coeffs, 0.7, 1.0)
        return coeffs.astype(np.float64)

    def raw_to_radiance(self, raw_frame: np.ndarray) -> np.ndarray:
        normalized = raw_frame.astype(np.float64) / self.max_value
        radiance = normalized * self.absorption_coeffs
        return radiance

    def stefan_boltzmann_inversion(self, radiance: np.ndarray) -> np.ndarray:
        sigma = config.STEFAN_BOLTZMANN_CONSTANT
        emissivity = 0.85
        M = radiance / (emissivity * sigma)
        T = np.power(M, 0.25)
        return T

    def planck_inversion(self, radiance: np.ndarray) -> np.ndarray:
        c1 = config.PLANCK_C1
        c2 = config.PLANCK_C2
        lambda_um = config.WAVELENGTH_UM
        emissivity = 0.85

        L = radiance * 1e3
        L = np.maximum(L, 1e-10)

        numerator = c2 / lambda_um
        denominator = np.log(1.0 + (c1 * emissivity) / (np.pi * lambda_um ** 5 * L))
        T = numerator / denominator
        return T

    def invert(self, raw_frame: np.ndarray, method: str = "planck") -> np.ndarray:
        if raw_frame.shape != (config.CAMERA_HEIGHT, config.CAMERA_WIDTH):
            raise ValueError(
                f"Expected frame shape ({config.CAMERA_HEIGHT}, {config.CAMERA_WIDTH}), "
                f"got {raw_frame.shape}"
            )

        radiance = self.raw_to_radiance(raw_frame)

        if method == "stefan_boltzmann":
            temperature = self.stefan_boltzmann_inversion(radiance)
        elif method == "planck":
            temperature = self.planck_inversion(radiance)
        else:
            raise ValueError(f"Unknown inversion method: {method}")

        temperature_celsius = temperature - 273.15
        return temperature_celsius.astype(np.float32)

    async def invert_async(
        self,
        raw_frame: np.ndarray,
        method: str = "planck",
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        if raw_frame.shape != (config.CAMERA_HEIGHT, config.CAMERA_WIDTH):
            raise ValueError(
                f"Expected frame shape ({config.CAMERA_HEIGHT}, {config.CAMERA_WIDTH}), "
                f"got {raw_frame.shape}"
            )

        result = await compute_pool.temperature_inversion(
            raw_frame,
            method,
            self.max_value,
            self.absorption_coeffs,
            config.STEFAN_BOLTZMANN_CONSTANT,
            config.PLANCK_C1,
            config.PLANCK_C2,
            config.WAVELENGTH_UM,
            timeout=timeout,
        )

        if not result.success:
            print(f"[TempInversion] Async inversion failed: {result.error}, using local compute")
            return self.invert(raw_frame, method=method)

        return result.data

    def get_absorption_coeffs(self) -> np.ndarray:
        return self.absorption_coeffs.copy()


temp_inversion = TemperatureInversion()
