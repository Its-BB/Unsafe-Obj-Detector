# ESP32-CAM firmware

Firmware for ESP32-CAM: joins your WiFi, serves a live MJPEG stream at `http://<device-ip>/stream`.

## Setup

1. Copy `wifi_config.h.example` to `main/wifi_config.h` and set your 2.4 GHz WiFi name and password.
2. Open this folder in an ESP-IDF shell.
3. Build and flash:

```bat
idf.py build
idf.py -p COM3 flash monitor
```

Or run `build_and_flash.bat` (build only; set your COM port for flash).

4. Note the IP from the serial monitor, then open `http://<ip>/` or `http://<ip>/stream`.

## Use with the Python app

In the parent project `config.yaml`:

```yaml
detection:
  use_remote_stream: true
  remote_stream_url: "http://<device-ip>/stream"
```

Test the stream from the repo root: `python fetch_video.py`

## Layout

```
main/
  main.c            Application entry
  wifi_station.c    Connect to your router
  web_server.c      HTTP + MJPEG /stream
  simple_camera.c   OV2640 via esp32-camera
  wifi_config.h     Your credentials (not committed — copy from example)
components/esp32-camera/
```

## Notes

- ESP32 needs **2.4 GHz** WiFi.
- If flash fails, hold BOOT, tap RST, release BOOT, then flash again.
