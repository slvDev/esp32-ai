/* DFR1154 sensor bring-up and the LM/perception coexistence harness.
 * See dfr1154_sensors.h for why this exists and where the pin map came from. */

#include "dfr1154_sensors.h"

#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#if SENSOR_HAS_CAMERA
#include "esp_camera.h"
#endif

static volatile uint32_t s_frames = 0, s_fails = 0;
static volatile uint64_t s_bytes  = 0;
static bool s_cam_up = false;
static int64_t s_t0 = 0;

#if SENSOR_HAS_CAMERA
/* Free-running capture.
 *
 * PRIORITY 1, PINNED TO CPU1.
 *
 * Both choices are measurements in disguise and are stated here so the result
 * is interpretable. The LM's matvec worker is priority 2 on CPU0 and the decode
 * loop is the main task on CPU1 (CONFIG_ESP_MAIN_TASK_AFFINITY_CPU1). Putting
 * capture at priority 1 on CPU1 means perception is strictly subordinate to
 * inference on the core that runs inference, and never preempts the worker at
 * all. So the FPS this reports is what perception can scavenge, and any tok/s
 * loss is attributable to bus contention and cache pressure rather than to the
 * scheduler taking cycles away from the LM. That is the conservative pairing:
 * it isolates the memory-system cost, which is the part we cannot design
 * around, from the scheduling cost, which we can. */
static void capture_task(void *arg) {
  (void)arg;
  for (;;) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      s_fails++;
      vTaskDelay(1);
      continue;
    }
    s_frames++;
    s_bytes += fb->len;
    esp_camera_fb_return(fb);
    /* Yield explicitly. Without this a priority-1 task that never blocks can
     * still starve equal-priority work on the same core between tick
     * boundaries, and the idle task never runs to service the WDT. */
    taskYIELD();
  }
}
#endif

bool dfr_sensors_begin(void) {
  printf("sensors: arm SENSOR_MODE=%d (%s)\n", SENSOR_MODE,
         SENSOR_MODE == 0   ? "LM only"
         : SENSOR_MODE == 1 ? "camera only"
                            : "LM + camera concurrent");
  printf("sensors: internal SRAM before bring-up %u B, PSRAM %u B\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

#if SENSOR_HAS_CAMERA
  camera_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0 = DFR_Y2_GPIO;  cfg.pin_d1 = DFR_Y3_GPIO;
  cfg.pin_d2 = DFR_Y4_GPIO;  cfg.pin_d3 = DFR_Y5_GPIO;
  cfg.pin_d4 = DFR_Y6_GPIO;  cfg.pin_d5 = DFR_Y7_GPIO;
  cfg.pin_d6 = DFR_Y8_GPIO;  cfg.pin_d7 = DFR_Y9_GPIO;
  cfg.pin_xclk = DFR_XCLK_GPIO;
  cfg.pin_pclk = DFR_PCLK_GPIO;
  cfg.pin_vsync = DFR_VSYNC_GPIO;
  cfg.pin_href = DFR_HREF_GPIO;
  cfg.pin_sccb_sda = DFR_SIOD_GPIO;
  cfg.pin_sccb_scl = DFR_SIOC_GPIO;
  cfg.pin_pwdn = DFR_PWDN_GPIO;
  cfg.pin_reset = DFR_RESET_GPIO;
  cfg.xclk_freq_hz = 20000000;
  cfg.frame_size = FRAMESIZE_240X240;
  cfg.pixel_format = PIXFORMAT_RGB565;
  /* GRAB_LATEST, not WHEN_EMPTY: an agent wants the freshest observation, and
   * WHEN_EMPTY would let the driver block on our consumer and report a frame
   * rate that is really our consumer's rate. */
  cfg.grab_mode = CAMERA_GRAB_LATEST;
  cfg.fb_location = CAMERA_FB_IN_PSRAM;
  cfg.fb_count = DFR_FB_COUNT;

  int64_t t = esp_timer_get_time();
  esp_err_t err = esp_camera_init(&cfg);
  int64_t init_us = esp_timer_get_time() - t;
  if (err != ESP_OK) {
    printf("sensors: CAMERA INIT FAILED 0x%x -- arm %d is INVALID, do not "
           "compare its tok/s against the LM-only control\n", err, SENSOR_MODE);
    return false;
  }
  s_cam_up = true;
  printf("sensors: OV3660 up in %.0f ms, 240x240 RGB565, %d fb in PSRAM "
         "(%u B/frame)\n", init_us / 1000.0, DFR_FB_COUNT,
         (unsigned)DFR_FRAME_BYTES);

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    printf("sensors: sensor PID 0x%x %s\n", s->id.PID,
           s->id.PID == OV3660_PID ? "(OV3660, as expected)"
                                   : "(NOT OV3660 -- pin map suspect)");
    if (s->id.PID == OV3660_PID) s->set_vflip(s, 1);
  }
#endif

  printf("sensors: internal SRAM after bring-up  %u B, PSRAM %u B\n",
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  return true;
}

void dfr_capture_start(void) {
#if SENSOR_HAS_CAMERA
  if (!s_cam_up) return;
  s_frames = 0; s_fails = 0; s_bytes = 0;
  s_t0 = esp_timer_get_time();
  if (xTaskCreatePinnedToCore(capture_task, "cap", 4096, NULL, 1, NULL, 1)
      != pdPASS) {
    printf("sensors: capture task creation FAILED\n");
  }
#endif
}

void dfr_capture_stats(uint32_t *frames, uint64_t *bytes, uint32_t *fails) {
  if (frames) *frames = s_frames;
  if (bytes)  *bytes  = s_bytes;
  if (fails)  *fails  = s_fails;
}

void dfr_sensors_report(void) {
  if (!s_cam_up) {
    printf("sensors: no camera in this arm\n");
    return;
  }
  int64_t el = esp_timer_get_time() - s_t0;
  double sec = el / 1e6;
  printf("sensors: %u frames in %.2f s = %.2f FPS | %.2f MiB/s from sensor "
         "| %u grab failures\n",
         (unsigned)s_frames, sec, s_frames / sec,
         (double)s_bytes / 1048576.0 / sec, (unsigned)s_fails);
}
