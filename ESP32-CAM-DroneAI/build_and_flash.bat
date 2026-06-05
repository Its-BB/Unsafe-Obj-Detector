@echo off
echo ESP32-CAM-DroneAI build
idf.py build
echo.
echo Flash: idf.py -p COM3 flash monitor
pause
