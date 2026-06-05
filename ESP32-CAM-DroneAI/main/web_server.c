#include "web_server.h"
#include "camera.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_system.h"

static const char* TAG = "WEB_SERVER";
static httpd_handle_t server = NULL;

static const char* index_html = 
"<!DOCTYPE html>"
"<html>"
"<head>"
"    <title>ESP32-CAM Live Stream</title>"
"    <meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no'>"
"    <style>"
"        * { margin: 0; padding: 0; box-sizing: border-box; }"
"        html, body { width: 100%; height: 100%; overflow: hidden; }"
"        body { background: #000; font-family: Arial, sans-serif; }"
"        .container { width: 100%; height: 100%; display: flex; flex-direction: column; }"
"        .header { background: #111; color: #fff; padding: 10px 15px; text-align: center; border-bottom: 2px solid #00ff00; }"
"        .header h1 { font-size: 20px; margin: 0; }"
"        .stream-container { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }"
"        .stream-container img { width: 100%; height: 100%; object-fit: contain; }"
"        .footer { background: #111; color: #aaa; padding: 8px 15px; text-align: center; font-size: 12px; border-top: 1px solid #333; }"
"    </style>"
"</head>"
"<body>"
"    <div class='container'>"
"        <div class='header'>"
"            <h1>ESP32-CAM Live Stream</h1>"
"        </div>"
"        <div class='stream-container'>"
"            <img src='/stream' alt='Camera Stream' onerror='this.src=/stream'>"
"        </div>"
"        <div class='footer'>"
"            <p>Connected to your WiFi | Stream: /stream | FPS: ~5</p>"
"        </div>"
"    </div>"
"</body>"
"</html>";

#define STREAM_CONTENT_TYPE "multipart/x-mixed-replace;boundary=123456789000000000000987654321"
#define STREAM_BOUNDARY "\r\n--123456789000000000000987654321\r\n"
#define STREAM_PART "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"

static esp_err_t index_handler(httpd_req_t *req)
{
    ESP_LOGI(TAG, "Serving main page");
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_html, strlen(index_html));
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    char part_buf[128];

    ESP_LOGI(TAG, "Starting MJPEG stream");

    httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "5");

    while (true) {
        fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
            break;
        }

        _jpg_buf = fb->buf;
        _jpg_buf_len = fb->len;

        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        }
        
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 128, STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }

        if (fb) {
            camera_return_frame(fb);
            fb = NULL;
        }

        if (res != ESP_OK) {
            ESP_LOGI(TAG, "Stream ended");
            break;
        }

        vTaskDelay(pdMS_TO_TICKS(200));
    }

    return res;
}

esp_err_t web_server_start(void)
{
    ESP_LOGI(TAG, "Starting web server");

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_open_sockets = 2;
    config.max_uri_handlers = 8;
    config.core_id = 1;
    config.stack_size = 16384;
    config.send_wait_timeout = 5;

    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Error starting server!");
        return ESP_FAIL;
    }

    httpd_uri_t index_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = index_handler,
        .user_ctx = NULL
    };
    httpd_register_uri_handler(server, &index_uri);

    httpd_uri_t stream_uri = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = stream_handler,
        .user_ctx = NULL
    };
    httpd_register_uri_handler(server, &stream_uri);

    ESP_LOGI(TAG, "Web server started on port 80");
    ESP_LOGI(TAG, "Use the device IP from serial monitor");

    return ESP_OK;
}

esp_err_t web_server_stop(void)
{
    if (server) {
        return httpd_stop(server);
    }
    return ESP_OK;
}