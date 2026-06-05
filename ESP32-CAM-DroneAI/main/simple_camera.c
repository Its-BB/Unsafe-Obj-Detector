#include "camera.h"
#include "esp_log.h"
#include "camera_pins.h"
#include <string.h>

static const char* TAG = "CAMERA";
static bool camera_initialized = false;

esp_err_t camera_init(void)
{
    ESP_LOGI(TAG, "Initializing camera");

    camera_config_t config = {0};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.frame_size = FRAMESIZE_QVGA;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return err;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL && s->id.PID == OV2640_PID) {
        s->set_framesize(s, FRAMESIZE_QVGA);
        s->set_quality(s, 12);
    }

    camera_initialized = true;
    ESP_LOGI(TAG, "Camera ready");
    return ESP_OK;
}

camera_config_t* camera_get_config(void)
{
    static camera_config_t config = {0};
    return &config;
}

esp_err_t camera_set_quality(uint8_t quality)
{
    if (!camera_initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_quality(s, quality);
    }
    return ESP_OK;
}

esp_err_t camera_set_framesize(framesize_t framesize)
{
    if (!camera_initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_framesize(s, framesize);
    }
    return ESP_OK;
}

esp_err_t camera_set_brightness(int brightness)
{
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_brightness(s, brightness);
    }
    return ESP_OK;
}

esp_err_t camera_set_contrast(int contrast)
{
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_contrast(s, contrast);
    }
    return ESP_OK;
}

camera_fb_t* camera_capture_frame(void)
{
    if (!camera_initialized) {
        return NULL;
    }
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Capture failed");
    }
    return fb;
}

void camera_return_frame(camera_fb_t *fb)
{
    if (fb) {
        esp_camera_fb_return(fb);
    }
}

bool camera_is_initialized(void)
{
    return camera_initialized;
}

bool frame2jpg(camera_fb_t* fb, uint8_t quality, uint8_t** out_buf, size_t* out_len)
{
    (void)quality;
    if (!fb || !out_buf || !out_len) {
        return false;
    }
    *out_len = fb->len;
    *out_buf = malloc(fb->len);
    if (!*out_buf) {
        return false;
    }
    memcpy(*out_buf, fb->buf, fb->len);
    return true;
}
