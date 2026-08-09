#include "board_display.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "driver/ledc.h"
#include "esp_board_manager_includes.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_ops.h"
#include "esp_log.h"

namespace {

constexpr uint16_t kBackground = 0x0000; // RGB565 black
constexpr uint16_t kForeground = 0x07E0; // RGB565 green
constexpr int kGlyphWidth = 5;
constexpr int kGlyphHeight = 7;

const char *const TAG = "llm_display";

esp_lcd_panel_handle_t panel;
uint16_t *framebuffer;
uint16_t width;
uint16_t height;
uint16_t cursor_x;
uint16_t cursor_y;
uint8_t scale;
bool display_ready;

// Compact 5x7 font. Lowercase is deliberately rendered as uppercase: the
// TinyStories output stays readable while the complete font costs only 180 B.
constexpr uint8_t kDigits[10][kGlyphWidth] = {
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46}, {0x21, 0x41, 0x45, 0x4B, 0x31},
    {0x18, 0x14, 0x12, 0x7F, 0x10}, {0x27, 0x45, 0x45, 0x45, 0x39},
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x29, 0x1E},
};

constexpr uint8_t kLetters[26][kGlyphWidth] = {
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, {0x7F, 0x49, 0x49, 0x49, 0x36},
    {0x3E, 0x41, 0x41, 0x41, 0x22}, {0x7F, 0x41, 0x41, 0x22, 0x1C},
    {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x09, 0x01},
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, {0x7F, 0x08, 0x08, 0x08, 0x7F},
    {0x00, 0x41, 0x7F, 0x41, 0x00}, {0x20, 0x40, 0x41, 0x3F, 0x01},
    {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F},
    {0x3E, 0x41, 0x41, 0x41, 0x3E}, {0x7F, 0x09, 0x09, 0x09, 0x06},
    {0x3E, 0x41, 0x51, 0x21, 0x5E}, {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31}, {0x01, 0x01, 0x7F, 0x01, 0x01},
    {0x3F, 0x40, 0x40, 0x40, 0x3F}, {0x1F, 0x20, 0x40, 0x20, 0x1F},
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x07, 0x08, 0x70, 0x08, 0x07}, {0x61, 0x51, 0x49, 0x45, 0x43},
};

void punctuation_glyph(char c, uint8_t out[kGlyphWidth]) {
    const uint8_t *glyph = nullptr;
    static constexpr uint8_t exclamation[] = {0x00, 0x00, 0x5F, 0x00, 0x00};
    static constexpr uint8_t quote[] = {0x00, 0x07, 0x00, 0x07, 0x00};
    static constexpr uint8_t apostrophe[] = {0x00, 0x05, 0x03, 0x00, 0x00};
    static constexpr uint8_t comma[] = {0x00, 0x50, 0x30, 0x00, 0x00};
    static constexpr uint8_t dash[] = {0x08, 0x08, 0x08, 0x08, 0x08};
    static constexpr uint8_t period[] = {0x00, 0x60, 0x60, 0x00, 0x00};
    static constexpr uint8_t colon[] = {0x00, 0x36, 0x36, 0x00, 0x00};
    static constexpr uint8_t semicolon[] = {0x00, 0x56, 0x36, 0x00, 0x00};
    static constexpr uint8_t question[] = {0x02, 0x01, 0x51, 0x09, 0x06};
    static constexpr uint8_t slash[] = {0x20, 0x10, 0x08, 0x04, 0x02};
    static constexpr uint8_t paren_l[] = {0x00, 0x1C, 0x22, 0x41, 0x00};
    static constexpr uint8_t paren_r[] = {0x00, 0x41, 0x22, 0x1C, 0x00};
    switch (c) {
    case '!':
        glyph = exclamation;
        break;
    case '"':
        glyph = quote;
        break;
    case '\'':
        glyph = apostrophe;
        break;
    case ',':
        glyph = comma;
        break;
    case '-':
        glyph = dash;
        break;
    case '.':
        glyph = period;
        break;
    case ':':
        glyph = colon;
        break;
    case ';':
        glyph = semicolon;
        break;
    case '?':
        glyph = question;
        break;
    case '/':
        glyph = slash;
        break;
    case '(':
        glyph = paren_l;
        break;
    case ')':
        glyph = paren_r;
        break;
    default:
        glyph = question;
        break;
    }
    memcpy(out, glyph, kGlyphWidth);
}

void get_glyph(char c, uint8_t out[kGlyphWidth]) {
    if (c == ' ') {
        memset(out, 0, kGlyphWidth);
    } else if (c >= '0' && c <= '9') {
        memcpy(out, kDigits[c - '0'], kGlyphWidth);
    } else {
        c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
        if (c >= 'A' && c <= 'Z') {
            memcpy(out, kLetters[c - 'A'], kGlyphWidth);
        } else {
            punctuation_glyph(c, out);
        }
    }
}

void clear_screen() {
    memset(framebuffer, 0,
           static_cast<size_t>(width) * height * sizeof(uint16_t));
    cursor_x = scale * 2;
    cursor_y = scale * 2;
}

void newline() {
    cursor_x = scale * 2;
    cursor_y += (kGlyphHeight + 2) * scale;
    if (cursor_y + kGlyphHeight * scale > height) {
        clear_screen();
    }
}

void draw_character(char c) {
    const uint16_t cell_width = (kGlyphWidth + 1) * scale;
    if (cursor_x + cell_width > width) {
        newline();
    }

    uint8_t glyph[kGlyphWidth];
    get_glyph(c, glyph);
    for (int gx = 0; gx < kGlyphWidth; ++gx) {
        for (int gy = 0; gy < kGlyphHeight; ++gy) {
            const uint16_t color =
                (glyph[gx] & (1U << gy)) ? kForeground : kBackground;
            for (int sx = 0; sx < scale; ++sx) {
                for (int sy = 0; sy < scale; ++sy) {
                    const size_t x = cursor_x + gx * scale + sx;
                    const size_t y = cursor_y + gy * scale + sy;
                    framebuffer[y * width + x] = color;
                }
            }
        }
    }
    cursor_x += cell_width;
}

void write_text(const uint8_t *bytes, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        const uint8_t byte = bytes[i];
        if (byte == '\n' || byte == '\r') {
            if (byte == '\n') {
                newline();
            }
        } else if (byte >= 32 && byte < 127) {
            draw_character(static_cast<char>(byte));
        } else if ((byte & 0xC0) != 0x80) {
            draw_character('?');
        }
    }
}

void refresh() {
    const esp_err_t err =
        esp_lcd_panel_draw_bitmap(panel, 0, 0, width, height, framebuffer);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "LCD refresh failed: %s", esp_err_to_name(err));
    }
}

esp_err_t set_backlight(uint32_t percent) {
#if CONFIG_ESP_BOARD_DEV_LEDC_CTRL_SUPPORT
    if (!esp_board_manager_check_name(ESP_BOARD_DEVICE_NAME_LCD_BRIGHTNESS)) {
        return ESP_OK;
    }

    percent = percent > 100 ? 100 : percent;
    ESP_RETURN_ON_ERROR(esp_board_manager_init_device_by_name(
                            ESP_BOARD_DEVICE_NAME_LCD_BRIGHTNESS),
                        TAG, "initialize lcd_brightness failed");

    periph_ledc_handle_t *ledc_handle = nullptr;
    ESP_RETURN_ON_ERROR(esp_board_manager_get_device_handle(
                            ESP_BOARD_DEVICE_NAME_LCD_BRIGHTNESS,
                            reinterpret_cast<void **>(&ledc_handle)),
                        TAG, "get lcd_brightness handle failed");

    dev_ledc_ctrl_config_t *brightness_config = nullptr;
    ESP_RETURN_ON_ERROR(esp_board_manager_get_device_config(
                            ESP_BOARD_DEVICE_NAME_LCD_BRIGHTNESS,
                            reinterpret_cast<void **>(&brightness_config)),
                        TAG, "get lcd_brightness config failed");

    periph_ledc_config_t *ledc_config = nullptr;
    ESP_RETURN_ON_ERROR(esp_board_manager_get_periph_config(
                            brightness_config->ledc_name,
                            reinterpret_cast<void **>(&ledc_config)),
                        TAG, "get backlight LEDC config failed");

    const uint32_t max_duty = (1U << ledc_config->duty_resolution) - 1;
    const uint32_t duty = percent * max_duty / 100;
    ESP_RETURN_ON_ERROR(
        ledc_set_duty(ledc_handle->speed_mode, ledc_handle->channel, duty), TAG,
        "set backlight duty failed");
    ESP_RETURN_ON_ERROR(
        ledc_update_duty(ledc_handle->speed_mode, ledc_handle->channel), TAG,
        "update backlight duty failed");
#else
    (void)percent;
#endif
    return ESP_OK;
}

} // namespace

esp_err_t board_display_init(void) {
#if !CONFIG_ESP_BOARD_DEV_DISPLAY_LCD_SUPPORT
    ESP_LOGE(TAG, "selected Board Manager board has no display_lcd device");
    return ESP_ERR_NOT_SUPPORTED;
#else
    ESP_RETURN_ON_FALSE(
        esp_board_manager_check_name(ESP_BOARD_DEVICE_NAME_DISPLAY_LCD),
        ESP_ERR_NOT_FOUND, TAG, "selected board has no display_lcd device");
    ESP_RETURN_ON_ERROR(
        esp_board_manager_init_device_by_name(ESP_BOARD_DEVICE_NAME_DISPLAY_LCD),
        TAG, "initialize display_lcd failed");

    dev_display_lcd_handles_t *display = nullptr;
    ESP_RETURN_ON_ERROR(
        esp_board_manager_get_device_handle(ESP_BOARD_DEVICE_NAME_DISPLAY_LCD,
                                            reinterpret_cast<void **>(&display)),
        TAG, "get display_lcd handle failed");
    ESP_RETURN_ON_FALSE(display && display->panel_handle, ESP_ERR_INVALID_STATE,
                        TAG, "display_lcd returned an invalid panel handle");
    panel = display->panel_handle;

    dev_display_lcd_config_t *config = nullptr;
    ESP_RETURN_ON_ERROR(
        esp_board_manager_get_device_config(ESP_BOARD_DEVICE_NAME_DISPLAY_LCD,
                                            reinterpret_cast<void **>(&config)),
        TAG, "get display_lcd config failed");
    width = config->swap_xy ? config->lcd_height : config->lcd_width;
    height = config->swap_xy ? config->lcd_width : config->lcd_height;
    ESP_RETURN_ON_FALSE(width && height, ESP_ERR_INVALID_SIZE, TAG,
                        "Board Manager returned an invalid display resolution");

    scale = width >= 800 ? 3 : (width >= 480 ? 2 : 1);
    const size_t framebuffer_size =
        static_cast<size_t>(width) * height * sizeof(uint16_t);
    framebuffer = static_cast<uint16_t *>(heap_caps_aligned_alloc(
            64, framebuffer_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    ESP_RETURN_ON_FALSE(framebuffer, ESP_ERR_NO_MEM, TAG,
                        "allocate %u-byte text framebuffer failed",
                        static_cast<unsigned>(framebuffer_size));

    clear_screen();
    refresh();
    ESP_RETURN_ON_ERROR(set_backlight(100), TAG, "turn LCD backlight on failed");
    display_ready = true;
    ESP_LOGI(TAG, "Board display: %s (%s), %ux%u, text scale=%u", config->chip,
             config->sub_type, width, height, scale);
    return ESP_OK;
#endif
}

void board_display_puts(const uint8_t *bytes, size_t len) {
    if (!display_ready || !bytes || len == 0) {
        return;
    }
    write_text(bytes, len);
    refresh();
}

void board_display_stats(float tokens_per_second,
                         float milliseconds_per_token) {
    if (!display_ready) {
        return;
    }

    char stats[192];
    const int len = snprintf(stats, sizeof(stats),
                             "ESP32 PLE TINYLM\n\n"
                             "28.9M PARAMETERS\n"
                             "%.2F TOKENS/SECOND\n"
                             "%.1F MS/TOKEN",
                             tokens_per_second, milliseconds_per_token);
    clear_screen();
    if (len > 0) {
        write_text(reinterpret_cast<const uint8_t *>(stats),
                   static_cast<size_t>(len < static_cast<int>(sizeof(stats))
                                       ? len
                                       : sizeof(stats) - 1));
    }
    refresh();
}
