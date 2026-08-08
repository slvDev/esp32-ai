/* Minimal Arduino compatibility layer, so the ESP-IDF port can include the
 * Arduino sketch verbatim instead of forking it.
 *
 * The sketch turned out to be ~95% plain ESP-IDF already: esp_timer, heap_caps,
 * esp_partition and FreeRTOS are native APIs. Only Serial, delay, pinMode /
 * digitalWrite and the setup/loop entry points are Arduino-isms, and all of
 * them are thin. Shimming those is far cheaper than maintaining two copies of a
 * kernel whose whole value is that its output is bit-identical across builds --
 * a silent divergence between branches would invalidate every comparison this
 * project makes.
 */
#pragma once

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"

/* Serial: a struct of function pointers rather than macros, because the sketch
 * calls it as `Serial.printf(...)` and a dotted name cannot be macro-replaced.
 * A varargs function pointer binds to printf directly. */
static inline void shim_println_s(const char *s) { printf("%s\n", s); }
static inline void shim_println_v(void) { printf("\n"); }
static inline void shim_begin(unsigned long baud) { (void)baud; }
/* The sketch calls exactly two forms: `Serial.write(bytes, len)` to emit a
 * token's UTF-8 bytes, and `Serial.print(">>> ")` for the prompt marker. Shimmed
 * to those signatures rather than to Arduino's full overload set. */
static inline void shim_write_buf(const uint8_t *b, size_t n) {
  fwrite(b, 1, n, stdout);
}
static inline void shim_print_s(const char *s) { fputs(s, stdout); }

/* availableForWrite gates a non-blocking emit: the sketch skips the write when
 * the TX buffer cannot take a whole token, so generation is never stalled by an
 * undrained host. stdio over USB-Serial-JTAG buffers for us, so reporting a
 * large window preserves the "never stall, never truncate mid-token" intent. */
static inline int shim_avail_write(void) { return 256; }
static inline void shim_flush(void) { fflush(stdout); }

struct ArduinoSerial {
  int (*printf)(const char *, ...);
  void (*println)(const char *);
  void (*begin)(unsigned long);
  void (*write)(const uint8_t *, size_t);
  void (*print)(const char *);
  int (*availableForWrite)(void);
  void (*flush)(void);
};

static const struct ArduinoSerial Serial = {
    printf, shim_println_s, shim_begin, shim_write_buf,
    shim_print_s, shim_avail_write, shim_flush,
};

/* FreeRTOS tick is 1 ms by default in this project's sdkconfig, so delay(0)
 * still yields -- which is what the sketch uses it for (feeding the task
 * watchdog every 8 tokens), not for timing. */
static inline void delay(uint32_t ms) {
  if (ms == 0) { taskYIELD(); return; }
  vTaskDelay(pdMS_TO_TICKS(ms));
}

static inline uint32_t millis(void) {
  return (uint32_t)(esp_timer_get_time() / 1000);
}

/* GPIO: the sketch blinks GPIO3 (the DFR1154's onboard LED, per the DFRobot
 * wiki pin map) as a liveness indicator. */
#define OUTPUT GPIO_MODE_OUTPUT
#define INPUT GPIO_MODE_INPUT
#define HIGH 1
#define LOW 0

static inline void pinMode(int pin, gpio_mode_t mode) {
  gpio_config_t c = {};
  c.pin_bit_mask = 1ULL << pin;
  c.mode = mode;
  c.pull_up_en = GPIO_PULLUP_DISABLE;
  c.pull_down_en = GPIO_PULLDOWN_DISABLE;
  c.intr_type = GPIO_INTR_DISABLE;
  gpio_config(&c);
}

static inline void digitalWrite(int pin, int level) {
  gpio_set_level((gpio_num_t)pin, level);
}
