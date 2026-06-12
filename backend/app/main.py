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
