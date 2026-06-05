#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "wifi_station.h"
#include "wifi_config.h"
#include "web_server.h"
#include "camera.h"

static const char* TAG = "MAIN";

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-CAM starting, free heap: %lu", esp_get_free_heap_size());

    if (camera_init() != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed");
        return;
    }

    if (wifi_station_init() != ESP_OK) {
        ESP_LOGE(TAG, "WiFi init failed");
        return;
    }

    vTaskDelay(pdMS_TO_TICKS(2000));

    if (web_server_start() != ESP_OK) {
        ESP_LOGE(TAG, "Web server failed");
        return;
    }

    char ip_str[16] = { 0 };
    if (wifi_station_get_ip_string(ip_str) == ESP_OK) {
        ESP_LOGI(TAG, "Ready on WiFi: %s", WIFI_ID);
        ESP_LOGI(TAG, "Web: http://%s/", ip_str);
        ESP_LOGI(TAG, "Stream: http://%s/stream", ip_str);
    } else {
        ESP_LOGI(TAG, "Ready (check log for IP)");
    }

    while (1) {
        ESP_LOGI(TAG, "heap %lu | camera %s",
                 esp_get_free_heap_size(),
                 camera_is_initialized() ? "online" : "offline");
        vTaskDelay(pdMS_TO_TICKS(60000));
    }
}
