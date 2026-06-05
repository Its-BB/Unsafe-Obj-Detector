#ifndef CAMERA_H
#define CAMERA_H

#include "esp_err.h"
#include "esp_camera.h"
#include <stdint.h>
#include <stdbool.h>

esp_err_t camera_init(void);
camera_config_t* camera_get_config(void);
esp_err_t camera_set_quality(uint8_t quality);
esp_err_t camera_set_framesize(framesize_t framesize);
esp_err_t camera_set_brightness(int brightness);
esp_err_t camera_set_contrast(int contrast);
camera_fb_t* camera_capture_frame(void);
void camera_return_frame(camera_fb_t* fb);
bool camera_is_initialized(void);
bool frame2jpg(camera_fb_t* fb, uint8_t quality, uint8_t** out_buf, size_t* out_len);

#endif
