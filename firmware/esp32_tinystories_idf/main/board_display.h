#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

esp_err_t board_display_init(void);
void board_display_puts(const uint8_t *bytes, size_t len);
void board_display_stats(float tokens_per_second, float milliseconds_per_token);
