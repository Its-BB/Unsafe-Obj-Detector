#ifndef WIFI_STATION_H
#define WIFI_STATION_H

#include <stdbool.h>
#include "esp_err.h"

esp_err_t wifi_station_init(void);
bool wifi_station_is_connected(void);
esp_err_t wifi_station_get_ip_string(char* ip_str);

#endif
