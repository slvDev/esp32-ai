# ESP32-S3 on-chip inference

This sketch runs the 28.9M-parameter PLE TinyLM on an ESP32-S3 N16R8. The model
lives in the custom `model` flash partition at `0x110000`; the tied
embedding/output head is staged in PSRAM at boot.

## Build and verify

If you want the full sequence (prepare/train/export/compile/flash) in one
command, run `uv run python flash.py --full-pipeline --force-train` from the
repo root.

Export the group-128 ragged-int4 model and verify the portable C runtime first.
This repo does not ship `runs/*.pt` checkpoints or the generated `firmware/model/model.bin`, so
you need a trained checkpoint in `runs/` before `src/export.py` can produce the artifact:

```bash
cd src
uv run python export.py
cd ..
cc -O3 -o /tmp/esp32-llm-verify firmware/host_verify/verify.c -lm
/tmp/esp32-llm-verify firmware/model/model.bin firmware/model/golden.txt
```

Build the device firmware with Arduino ESP32 core 3.3.10:

```bash
arduino-cli compile \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build \
  firmware/esp32_llm
```

## Flash and run

On Linux the board usually shows up as `/dev/ttyUSB0`; replace the port if your
system uses a different device name:

If you already have a checkpoint/model and just want compile+flash with USB
auto-scan, run `uv run python flash.py` from the repo root.

```bash
arduino-cli upload \
  -p /dev/ttyUSB0 \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-build \
  firmware/esp32_llm

esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 \
  write_flash 0x110000 firmware/model/model.bin

arduino-cli monitor -p /dev/ttyUSB0 --config baudrate=115200
```

The model payload only needs reflashing after a new export. Firmware-only
changes can be uploaded without rewriting the model partition.

The model used for the measurements below has SHA-256:

```text
21067f5d78113f6c64a8720b05ff7e5c774dab0276797a522f81a6797253d97c
```

Expected boot diagnostics for the current artifact:

```text
model: V=32768 D=96 L=6 H=4 F=66 P=128
head staged int8: 2.53 MB
PSRAM free after alloc: ~5100 KB
```

The current runtime measures 102.9ms per model step (9.72 tok/s compute-only);
attached serial runs measure ~9.5 tok/s including output. On-device profile:
57.6ms output head, 25.6ms attention, 8.5ms PLE path, 6.9ms FFN, 4.4ms input.
The head is staged as int8 with int8 activations (host-validated, val perplexity
delta ~0) and is now PSRAM-bandwidth-bound. The fp32 host golden still matches
PyTorch to 1e-5.
