# ESP32-CAM DroneAI Project
cmake_minimum_required(VERSION 3.5)

# Set the target to ESP32 (ESP32-CAM module)
set(IDF_TARGET esp32)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32-cam-droneai)