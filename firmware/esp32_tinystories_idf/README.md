# TinyStories ESP-IDF on-chip inference

This project is an ESP-IDF and ESP Board Manager port of [`firmware/esp32_tinystories`](../esp32_tinystories/README.md).

It runs the released 28.9M-parameter TinyStories PLE model and shares the portable inference runtime in [`runtime/llm.h`](../../runtime/llm.h) with the upstream Arduino firmware and host verification tools.

The application is not tied to one board because Board Manager supplies the target, memory configuration, display handle, display resolution, and optional backlight device.

## Project structure

- `main/esp32_tinystories_idf.cpp` contains model loading, memory placement, inference, profiling, and text generation.
- `main/board_display.cpp` renders generated text through the Board Manager `display_lcd` device.
- `generated/vocab.h` is generated from the tokenizer being deployed and is intentionally excluded from Git.
- `../../runtime/llm.h` is the shared PLE model loader and inference implementation.
- `../../artifacts/tinystories/model.bin` is the verified released model downloaded by the repository fetch script.

## Runtime layout

- Flash holds the memory-mapped token embedding and PLE table, which are accessed by row.
- PSRAM holds the staged int8 core and output-head weights, logits, and KV cache.
- Internal SRAM holds the hot scratch buffers and relocated RMSNorm vectors.
- FreeRTOS task notifications split sufficiently large staged matvec operations across both cores when the selected target has more than one core.
- Generated text is written to both the serial console and the LCD resolution reported by Board Manager.

The implementation uses the current published TinyStories runtime with int8 activations and stages both the dense core and output head instead of retaining the earlier output-head-only optimization.

## Requirements

- ESP-IDF 5.5 or later.
- Python package `esp-bmgr-assist` for the `idf.py bmgr` action.
- Hugging Face CLI command `hf` for the verified model fetch script.
- A supported Board Manager definition with at least 16 MB Flash, PSRAM, and an RGB565 `display_lcd` device.

## Prepare the released model

Training is not required for normal deployment because the repository publishes a verified TinyStories model and tokenizer.

Run the following command from the repository root to download, verify, and install the release under `artifacts/tinystories`:

```bash
scripts/fetch_model.sh tinystories
```

The fetch script validates the model and tokenizer against pinned byte sizes and SHA-256 values before replacing any local artifact.

Generate this ESP-IDF project's decode header from the tokenizer that will be deployed:

```bash
uv run python firmware/esp32_tinystories/tools/generate_vocab.py \
  --tokenizer artifacts/tinystories/tokenizer.json \
  --out firmware/esp32_tinystories_idf/generated/vocab.h
```

The generated header remains inside `firmware/esp32_tinystories_idf` and is ignored by Git.

Training and `research.tinystories.export` are only needed when reproducing or replacing the published model.

## Select the board

Load the ESP-IDF environment and install the Board Manager action helper once:

```bash
. "$IDF_PATH/export.sh"
python -m pip install esp-bmgr-assist
```

Enter this project, list the available boards, and generate the selected board configuration:

```bash
cd firmware/esp32_tinystories_idf
idf.py bmgr -l
idf.py bmgr -b <board_name>
```

Select ESP32-P4-Function-EV-Board with:

```bash
idf.py bmgr -b esp32_p4_function_ev_board
```

Select ESP32-S31-Korvo-1 with:

```bash
idf.py bmgr -b esp32_s31_korvo_1
```

Select ESP32-S3-LCD-EV-Board with its default 480x480 GC9503 LCD sub-board with:

```bash
idf.py bmgr -b esp32_s3_lcd_ev_board
```

Select its 800x480 ST7262 RGB LCD and GT1151 touch sub-board, whose silkscreen is ESP32-S3-LCD-EV-Board-SUB3, with:

```bash
idf.py bmgr -b esp32_s3_lcd_ev_board \
  -a sub_board_800_480_lcd
```

The `bmgr` action selects the board's SoC target and generates `components/gen_bmgr_codes`, so a separate `idf.py set-target` command is not required.

## Build and flash

Build the firmware with:

```bash
idf.py build
```

Flash the bootloader, partition table, application, and released model, then open the serial monitor:

```bash
idf.py -p PORT flash monitor
```

The full `flash` target always includes `artifacts/tinystories/model.bin`, so flashing fails before writing the board when the model has not been fetched.

After the model has been installed once, firmware-only changes can be updated without rewriting the model partition:

```bash
idf.py -p PORT app-flash monitor
```

Press `Ctrl-]` to exit the serial monitor.

## Board Manager initialization

At startup, the application performs the following operations:

1. `esp_board_manager_print_board_info()` prints the selected board metadata.
2. `esp_board_manager_init_device_by_name("display_lcd")` initializes the selected board's panel and returns a generic `esp_lcd_panel_handle_t`.
3. The optional `lcd_brightness` device is initialized and set to full brightness when it is present.
4. The `model` partition is memory-mapped and checked against the generated tokenizer vocabulary.
5. Hot scratch and normalization data are placed in internal SRAM while staged weights, logits, and cache data are placed in PSRAM.
6. The application enables dual-core matvec execution when the selected target provides more than one core.

## Regenerate the board configuration

Run the following sequence when changing boards or when stale `sdkconfig` values prevent new board defaults from taking effect:

```bash
idf.py bmgr -x
idf.py fullclean
idf.py bmgr -b <board_name>
```
