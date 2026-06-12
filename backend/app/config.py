import os
from dataclasses import dataclass


@dataclass
class Config:
    TCP_HOST: str = "0.0.0.0"
    TCP_PORT: int = 8888

    CAMERA_WIDTH: int = 128
    CAMERA_HEIGHT: int = 128
    CAMERA_BIT_DEPTH: int = 14
    CAMERA_FPS: int = 10

    STEFAN_BOLTZMANN_CONSTANT: float = 5.670374419e-8

    PLANCK_C1: float = 1.191042972e8
    PLANCK_C2: float = 14387.77736
    WAVELENGTH_UM: float = 10.0

    INTERPOLATION_GRID_SIZE: int = 256
    RBF_FUNCTION: str = "multiquadric"
    RBF_EPSILON: float = 10.0

    CALIBRATION_FILE: str = os.path.join(
        os.path.dirname(__file__), "calib", "absorption_coeffs.npy"
    )

    WS_UPDATE_INTERVAL: float = 0.1

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    COMPUTE_POOL_WORKERS: int = 4
    COMPUTE_POOL_MAX_PENDING: int = 8
    COMPUTE_DEFAULT_TIMEOUT: float = 2.0

    FRAME_QUEUE_MAX_SIZE: int = 3
    FRAME_PROCESS_TIMEOUT: float = 1.5


config = Config()
