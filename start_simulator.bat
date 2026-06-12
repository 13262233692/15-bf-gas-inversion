@echo off
cd /d "%~dp0backend"

echo ========================================
echo   红外相机数据模拟器
echo ========================================
echo.

python simulator.py --fps 10

pause
