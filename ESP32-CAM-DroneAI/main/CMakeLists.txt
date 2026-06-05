idf_component_register(
    SRCS "main.c" "wifi_station.c" "web_server.c" "simple_camera.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_netif esp_event esp_http_server nvs_flash
    PRIV_REQUIRES esp32-camera
)
