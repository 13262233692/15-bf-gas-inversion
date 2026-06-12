import asyncio
import json
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import config
from .data_manager import data_manager
from .tcp_server import tcp_server
from .temperature_inversion import temp_inversion
from .rbf_interpolation import rbf_interpolator
from .computation_pool import compute_pool


app = FastAPI(title="高炉红外温度反演系统", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InversionRequest(BaseModel):
    raw_data: list
    method: str = "planck"


class InterpolationRequest(BaseModel):
    temperature_data: list
    grid_size: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    compute_pool.start(loop)
    asyncio.create_task(tcp_server.start())
    print("[API] Server starting up with isolated computation pool...")


@app.on_event("shutdown")
async def shutdown_event():
    await tcp_server.stop()
    compute_pool.shutdown(wait=False)
    print("[API] Server shutting down...")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "version": "2.0.0",
        "tcp_server": "running" if tcp_server._running else "stopped",
        "compute_pool": compute_pool.get_stats(),
        "queue": tcp_server.get_queue_stats(),
        "stats": data_manager.get_stats(),
    }


@app.get("/api/stats")
async def get_stats():
    return {
        "data": data_manager.get_stats(),
        "queue": tcp_server.get_queue_stats(),
        "compute_pool": compute_pool.get_stats(),
    }


@app.get("/api/raw-frame")
async def get_raw_frame():
    frame = data_manager.get_raw_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No raw frame available")
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "min": float(np.min(frame)),
        "max": float(np.max(frame)),
        "data": frame.flatten().tolist(),
    }


@app.get("/api/temperature-frame")
async def get_temperature_frame():
    frame = data_manager.get_temperature_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No temperature frame available")
    return {
        "shape": list(frame.shape),
        "min_temp": float(np.min(frame)),
        "max_temp": float(np.max(frame)),
        "avg_temp": float(np.mean(frame)),
        "data": frame.flatten().tolist(),
    }


@app.get("/api/interpolated-temp")
async def get_interpolated_temp():
    frame = data_manager.get_interpolated_temp()
    if frame is None:
        raise HTTPException(status_code=404, detail="No interpolated data available")
    return {
        "shape": list(frame.shape),
        "grid_size": frame.shape[0],
        "min_temp": float(np.min(frame)),
        "max_temp": float(np.max(frame)),
        "data": frame.flatten().tolist(),
    }


@app.get("/api/contours")
async def get_contours(levels: int = 20, use_interpolated: bool = True):
    if use_interpolated:
        temp_data = data_manager.get_interpolated_temp()
    else:
        temp_data = data_manager.get_temperature_frame()

    if temp_data is None:
        raise HTTPException(status_code=404, detail="No temperature data available")

    contours = await rbf_interpolator.generate_contours_async(
        temp_data, levels=levels, timeout=2.0
    )
    return {"levels": levels, "count": len(contours), "contours": contours}


@app.post("/api/invert")
async def invert_temperature(request: InversionRequest):
    try:
        raw_array = np.array(request.raw_data, dtype=np.uint16).reshape(
            (config.CAMERA_HEIGHT, config.CAMERA_WIDTH)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid raw data: {e}")

    try:
        temperature = await temp_inversion.invert_async(
            raw_array, method=request.method, timeout=2.0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "method": request.method,
        "shape": list(temperature.shape),
        "min_temp": float(np.min(temperature)),
        "max_temp": float(np.max(temperature)),
        "avg_temp": float(np.mean(temperature)),
        "data": temperature.flatten().tolist(),
    }


@app.post("/api/interpolate")
async def interpolate_temperature(request: InterpolationRequest):
    try:
        temp_array = np.array(request.temperature_data, dtype=np.float32).reshape(
            (config.CAMERA_HEIGHT, config.CAMERA_WIDTH)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid temperature data: {e}")

    original_grid = rbf_interpolator.grid_size
    if request.grid_size and request.grid_size != original_grid:
        rbf_interpolator.grid_size = request.grid_size
        rbf_interpolator._grid_coords = rbf_interpolator._create_grid()

    try:
        interpolated = await rbf_interpolator.interpolate_async(
            temp_array, timeout=2.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpolation error: {e}")
    finally:
        if request.grid_size and request.grid_size != original_grid:
            rbf_interpolator.grid_size = original_grid
            rbf_interpolator._grid_coords = rbf_interpolator._create_grid()

    return {
        "grid_size": interpolated.shape[0],
        "min_temp": float(np.min(interpolated)),
        "max_temp": float(np.max(interpolated)),
        "data": interpolated.flatten().tolist(),
    }


@app.get("/api/calibration")
async def get_calibration():
    coeffs = temp_inversion.get_absorption_coeffs()
    return {
        "shape": list(coeffs.shape),
        "min": float(np.min(coeffs)),
        "max": float(np.max(coeffs)),
        "data": coeffs.flatten().tolist(),
    }


@app.get("/api/compute-pool/stats")
async def get_compute_pool_stats():
    return compute_pool.get_stats()


@app.get("/api/velocity")
async def get_velocity_field():
    vel = data_manager.get_velocity_data()
    if vel is None:
        raise HTTPException(status_code=404, detail="No velocity data available")
    return {
        "shape": list(vel["magnitude"].shape),
        "grid_size": vel["magnitude"].shape[0],
        "min_velocity": float(vel["min_velocity"]),
        "max_velocity": float(vel["max_velocity"]),
        "avg_velocity": float(vel["avg_velocity"]),
        "warning_count": int(vel["warning_count"]),
        "danger_count": int(vel["danger_count"]),
        "total_flow_m3s": float(vel.get("total_flow_m3s", 0)),
        "vx": vel["vx"].flatten().tolist(),
        "vy": vel["vy"].flatten().tolist(),
        "magnitude": vel["magnitude"].flatten().tolist(),
        "hotspots": vel.get("hotspots", []),
    }


@app.get("/api/velocity/warnings")
async def get_velocity_warnings():
    vel = data_manager.get_velocity_data()
    if vel is None:
        return {"has_warning": False, "hotspots": [], "summary": {}}

    danger_count = int(vel["danger_count"])
    warn_count = int(vel["warning_count"])
    hotspots = vel.get("hotspots", [])

    status = "normal"
    if danger_count > 0:
        status = "danger"
    elif warn_count > 0:
        status = "warning"

    summary = {
        "status": status,
        "warning_count": warn_count,
        "danger_count": danger_count,
        "max_velocity": float(vel["max_velocity"]),
        "avg_velocity": float(vel["avg_velocity"]),
        "threshold_warning": config.VELOCITY_WARNING_THRESHOLD,
        "threshold_danger": config.VELOCITY_DANGER_THRESHOLD,
        "hotspot_count": len(hotspots),
    }

    return {
        "has_warning": warn_count > 0 or danger_count > 0,
        "summary": summary,
        "hotspots": hotspots,
    }


class OperatingParamsRequest(BaseModel):
    total_gas_flow_m3s: Optional[float] = None
    furnace_pressure_kpa: Optional[float] = None
    velocity_warning_threshold: Optional[float] = None
    velocity_danger_threshold: Optional[float] = None


@app.post("/api/velocity/params")
async def set_operating_params(request: OperatingParamsRequest):
    from .gas_velocity import gas_velocity_estimator
    gas_velocity_estimator.set_operating_parameters(
        total_gas_flow_m3s=request.total_gas_flow_m3s,
        furnace_pressure_kpa=request.furnace_pressure_kpa,
        velocity_warning_threshold=request.velocity_warning_threshold,
        velocity_danger_threshold=request.velocity_danger_threshold,
    )
    return {
        "status": "updated",
        "total_gas_flow_m3s": gas_velocity_estimator.total_gas_flow_m3s,
        "furnace_pressure_kpa": gas_velocity_estimator.furnace_pressure_kpa,
        "velocity_warning_threshold": gas_velocity_estimator.velocity_warning_threshold,
        "velocity_danger_threshold": gas_velocity_estimator.velocity_danger_threshold,
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    last_update = 0.0
    update_interval = config.WS_UPDATE_INTERVAL

    try:
        while True:
            await asyncio.sleep(update_interval)

            interp = data_manager.get_interpolated_temp()
            vel = data_manager.get_velocity_data()
            stats = data_manager.get_stats()
            queue_stats = tcp_server.get_queue_stats()
            pool_stats = compute_pool.get_stats()

            if interp is not None:
                data = {
                    "type": "temperature",
                    "stats": stats,
                    "queue": queue_stats,
                    "compute_pool": pool_stats,
                    "shape": list(interp.shape),
                    "min_temp": float(np.min(interp)),
                    "max_temp": float(np.max(interp)),
                    "data": interp.flatten().tolist(),
                }

                if vel is not None:
                    gs = vel["magnitude"].shape[0]
                    step = 4
                    sparse = np.zeros((gs // step, gs // step, 2), dtype=np.float32)
                    vy_s = vel["vy"][::step, ::step]
                    vx_s = vel["vx"][::step, ::step]
                    sparse[:, :, 0] = vy_s
                    sparse[:, :, 1] = vx_s

                    data["velocity"] = {
                        "grid_size": gs,
                        "step": step,
                        "sparse_shape": list(sparse.shape[:2]),
                        "min_velocity": float(vel["min_velocity"]),
                        "max_velocity": float(vel["max_velocity"]),
                        "avg_velocity": float(vel["avg_velocity"]),
                        "warning_count": int(vel["warning_count"]),
                        "danger_count": int(vel["danger_count"]),
                        "sparse_flat": sparse.flatten().tolist(),
                        "magnitude_64": vel["magnitude"][::4, ::4].flatten().tolist(),
                        "hotspots": vel.get("hotspots", []),
                        "total_flow_m3s": float(vel.get("total_flow_m3s", 0)),
                    }

                    danger = int(vel["danger_count"])
                    warn = int(vel["warning_count"])
                    status = "normal"
                    if danger > 0:
                        status = "danger"
                    elif warn > 0:
                        status = "warning"
                    data["velocity_warning"] = {
                        "status": status,
                        "has_warning": warn > 0 or danger > 0,
                        "threshold_warning": config.VELOCITY_WARNING_THRESHOLD,
                        "threshold_danger": config.VELOCITY_DANGER_THRESHOLD,
                    }

                await websocket.send_text(json.dumps(data))
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def root():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "BF Gas Inversion API v2.0"}
