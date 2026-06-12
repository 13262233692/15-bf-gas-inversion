@echo off
cd /d "%~dp0backend"

echo ========================================
echo   高炉红外温度反演系统 - 启动脚本
echo ========================================
echo.

echo [1/3] Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo [3/3] Starting FastAPI server...
echo.
echo Server will be available at:
echo   - Web UI:  http://localhost:8000/
echo   - API Docs: http://localhost:8000/docs
echo   - TCP Port: 8888
echo.
echo Press Ctrl+C to stop the server.
echo ========================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
