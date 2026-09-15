#!/usr/bin/env bash
# Flash a model and its firmware to an ESP32-S3.
#
# The board has one model partition. Deploying either model replaces whichever
# one is currently installed; there is no repartitioning trick.
# pipefail: some steps below are piped, and `set -e` alone does not see a
# failure on the left of a pipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Run one step of the deployment. On success only the interesting lines are
# shown, because a working run should read as a short checklist. On failure the
# whole output is printed and the script stops: a compiler error truncated to
# its last two lines is worse than no summary at all.
#
# SHOW is a grep pattern for the lines worth keeping; without it the last $keep
# lines are shown. Carriage returns become newlines so esptool's progress bar
# does not arrive as one enormous line.
step() {
  local keep=$1 label=$2
  shift 2
  local out status=0
  out=$("$@" 2>&1) || status=$?
  out=$(printf '%s\n' "$out" | tr '\r' '\n')
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$out" >&2
    echo "failed: $label" >&2
    exit "$status"
  fi
  if [ -n "${SHOW:-}" ]; then
    printf '%s\n' "$out" | grep -E "$SHOW" || true
  else
    printf '%s\n' "$out" | tail -n "$keep"
  fi
}

usage() {
  cat >&2 <<'EOF'
usage: scripts/deploy.sh <tinystories|barista>

The model is mandatory: the board holds one at a time, and deploying replaces
whatever is already there, so which one is being written is never implied.

Generates that model's device headers, runs the applicable host gates including
the golden gate when present, compiles, and only then writes the model partition
and the firmware.

Artifact paths may be overridden: ARTIFACTS, MODEL, TOKENIZER, GOLDEN, and for
barista also VOCAB and LAYOUT.
EOF
  exit 2
}

# Argument first, before any tool lookup, build, or device access.
[ $# -eq 1 ] || usage
MODEL_KIND=$1

# --- model-specific configuration -------------------------------------------
# MODEL_KIND selects; MODEL stays the path to the binary being flashed.
case "$MODEL_KIND" in
  tinystories)
    SKETCH=firmware/esp32_tinystories
    ARTIFACTS=${ARTIFACTS:-artifacts/tinystories}
    MODEL=${MODEL:-$ARTIFACTS/model.bin}
    TOKENIZER=${TOKENIZER:-$ARTIFACTS/tokenizer.json}
    GOLDEN=${GOLDEN:-$ARTIFACTS/golden.txt}
    REQUIRED=("$MODEL" "$TOKENIZER")
    FETCH_HINT="  scripts/fetch_model.sh tinystories
  Downloads always install under artifacts/tinystories/. If a different path
  is selected here, copy the files across afterwards."
    ;;
  barista)
    SKETCH=firmware/esp32_barista
    ARTIFACTS=${ARTIFACTS:-artifacts/barista}
    MODEL=${MODEL:-$ARTIFACTS/model.bin}
    TOKENIZER=${TOKENIZER:-$ARTIFACTS/tokenizer.json}
    GOLDEN=${GOLDEN:-$ARTIFACTS/golden.txt}
    VOCAB=${VOCAB:-$ARTIFACTS/vocab.json}
    LAYOUT=${LAYOUT:-$ARTIFACTS/layout.json}
    REQUIRED=("$MODEL" "$TOKENIZER" "$VOCAB" "$LAYOUT")
    FETCH_HINT="  scripts/fetch_model.sh barista
  Downloads always install under artifacts/barista/. If a different path is
  selected here, copy the files across afterwards."
    ;;
  *)
    usage
    ;;
esac

# These generators run isolated from the project environment: none of them needs
# torch, and deploying should not synchronise it. The tokenizer version is
# pinned because `--with` does not consult uv.lock.
# Every generator and checker is given the selected paths explicitly. Deriving
# them again from defaults would let an ARTIFACTS override build headers from one
# directory while conformance silently checked another.
prepare_headers() {
  case "$MODEL_KIND" in
    tinystories)
      # Rebuild the decode header from the tokenizer being deployed, every time.
      # The firmware only checks that VOCAB_N equals the model's output_vocab, so
      # a stale header from a different tokenizer with the same entry count would
      # pass that check and decode every token wrongly.
      echo "=== generate $SKETCH/generated/vocab.h from $TOKENIZER ==="
      SHOW=wrote step 0 "generate vocab.h" \
        uv run --no-project --with 'tokenizers==0.23.1' python "$SKETCH/tools/generate_vocab.py" \
        --tokenizer "$TOKENIZER" --out "$SKETCH/generated/vocab.h"
      # The sketch takes a typed prompt, so it needs the encoder as well as the
      # decode table.
      echo "=== generate encoder asset from $TOKENIZER ==="
      SHOW=wrote step 0 "generate encoder asset" \
        uv run --no-project python "$SKETCH/tools/generate_tokenizer_header.py" \
        --tokenizer "$TOKENIZER" --out "$SKETCH/generated/tokenizer_encoder.h"
      ;;
    barista)
      echo "=== generate word tables from $VOCAB and $LAYOUT ==="
      SHOW=wrote step 0 "generate word tables" \
        uv run --no-project python "$SKETCH/tools/generate_vocab_headers.py" \
        --vocab "$VOCAB" --layout "$LAYOUT" \
        --out-dir "$SKETCH/generated"
      echo "=== generate encoder asset from $TOKENIZER ==="
      SHOW=wrote step 0 "generate encoder asset" \
        uv run --no-project python "$SKETCH/tools/generate_tokenizer_header.py" \
        --tokenizer "$TOKENIZER" \
        --out "$SKETCH/generated/tokenizer_encoder.h"
      ;;
  esac
}

extra_gates() {
  case "$MODEL_KIND" in
    tinystories)
      : # The device only decodes: there is no on-device encoder to check.
      ;;
    barista)
      # The device encodes questions, so its ids must equal the tokenizer's, and
      # the reused word mappings must resolve to the ids they claim.
      echo "=== host verify: device encoder vs Hugging Face, and the word map ==="
      step 1 "tokenizer conformance" \
        uv run --no-project --with 'tokenizers==0.23.1' python "$SKETCH/tools/verify_tokenizer.py" \
        --tokenizer "$TOKENIZER" --vocab "$VOCAB" --layout "$LAYOUT"
      ;;
  esac
}

# --- everything below is shared ---------------------------------------------
PART_OFFSET=0x110000

# CDC_ON_BOOT selects where Serial goes: cdc for the chip's native USB port
# (the default), default for UART0 when the cable is in the board's UART port.
CDC_ON_BOOT=${CDC_ON_BOOT:-cdc}
# CHIP selects the target: esp32s3 (the N16R8 the project was built for) or
# esp32p4 (ESP32-P4NRW32, 32 MB flash and PSRAM; Serial defaults to UART0 there
# because its dev boards expose a UART bridge, so pass CDC_ON_BOOT=default).
CHIP=${CHIP:-esp32s3}
case "$CHIP" in
  esp32s3)
    FQBN="esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=$CDC_ON_BOOT,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info"
    ;;
  esp32p4)
    FQBN="esp32:esp32:esp32p4:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=$CDC_ON_BOOT,UploadMode=default,FlashMode=qio,FlashFreq=80,FlashSize=32M,PartitionScheme=custom,PSRAM=enabled,DebugLevel=info"
    ;;
  *)
    echo "CHIP must be esp32s3 or esp32p4, got '$CHIP'" >&2
    exit 2
    ;;
esac

# -O3, overriding the Arduino core default of -Os. The runtime carries no
# per-function optimization attributes; this flag is the whole configuration.
OPT_FLAGS='-O3'

# --- locate tools without hardcoding a home directory -----------------------
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need arduino-cli

find_esptool() {
  for c in esptool esptool.py; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }
  done
  # Arduino bundles it; the version directory changes between core releases, so
  # glob rather than pin. ARDUINO_DATA_DIR overrides for non-default installs.
  local data="${ARDUINO_DATA_DIR:-$HOME/Library/Arduino15}"
  [ -d "$data" ] || data="${ARDUINO_DATA_DIR:-$HOME/.arduino15}"
  local found
  found=$(ls -d "$data"/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1 || true)
  [ -n "$found" ] && { echo "$found"; return; }
  echo "esptool not found. Install it (pip install esptool) or set ARDUINO_DATA_DIR." >&2
  exit 1
}
ESPTOOL=$(find_esptool)

# `|| true`: a failing glob would otherwise abort the assignment under `set -e`,
# before the message below can print.
PORT=${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)}
[ -n "$PORT" ] || { echo "no /dev/cu.usbmodem* found; plug the board in, or set PORT=..." >&2; exit 1; }

# Every required artifact is checked before anything is generated or built, so a
# missing file is reported as a list rather than discovered halfway through.
missing=0
for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] || { echo "missing artifact: $f" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
  echo "required artifacts are missing; the expected paths are listed above" >&2
  echo "$FETCH_HINT" >&2
  exit 1
fi

prepare_headers

CFLAGS='-O3 -Wall -Wextra'
# One workspace per run, holding the host gate binaries and the firmware build.
# A shared path lets two concurrent deployments compile over each other.
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/esp32ai-deploy-XXXXXX")
trap 'rm -rf -- "$RUN_DIR"' EXIT
GATE_DIR="$RUN_DIR/gates"
BUILD_DIR="$RUN_DIR/build"
mkdir -p "$GATE_DIR" "$BUILD_DIR"

# The golden is produced by the exporter, which this script does not invoke, so
# it may simply be absent. Skip visibly rather than fail: staging_verify below
# needs no golden and still runs.
if [ -f "$GOLDEN" ]; then
  echo "=== host verify: exact int4 path vs PyTorch golden ==="
  cc $CFLAGS -o "$GATE_DIR/llm_verify" runtime/host_verify/verify.c -lm
  step 2 "golden gate" "$GATE_DIR/llm_verify" "$MODEL" "$GOLDEN"
else
  echo "=== host verify: SKIPPED, no golden at $GOLDEN ==="
fi

# The device runs the staged int8 kernel through the platform hooks. verify.c
# does not reach that code - it exercises the exact int4 path - so without
# this the thing actually executing on the board has no host gate.
echo "=== host verify: int8 staging + platform hooks ==="
cc $CFLAGS -DLLM_INT8_ACT=1 -o "$GATE_DIR/llm_staging" runtime/host_verify/staging_verify.c -lm
step 3 "staging gate" "$GATE_DIR/llm_staging" "$MODEL"

extra_gates

# Compile and verify before writing either image: a build failure after the
# model flash would leave new weights under old firmware.
echo "=== compile $SKETCH ($OPT_FLAGS) ==="
step 2 "compile $SKETCH" \
  arduino-cli compile --fqbn "$FQBN" \
  --build-property "compiler.optimization_flags=$OPT_FLAGS" \
  --build-path "$BUILD_DIR" \
  "$SKETCH"

# Compute the same FNV-1a fingerprint the firmware prints at boot, and show it
# before writing anything, so what is about to be installed is stated up front.
FP=$(python3 -c '
import sys
d = open(sys.argv[1], "rb").read()
h = 2166136261
for b in d:
    h ^= b; h = (h * 16777619) & 0xFFFFFFFF
print("%08x" % h)' "$MODEL")
BYTES=$(wc -c < "$MODEL" | tr -d ' ')

echo
echo "=== about to flash, replacing the model now on the board ==="
printf "  model   : %s\n" "$MODEL_KIND"
printf "  chip    : %s\n" "$CHIP"
printf "  sketch  : %s\n" "$SKETCH"
printf "  binary  : %s\n" "$MODEL"
printf "  port    : %s\n" "$PORT"
printf "  bytes   : %s\n" "$BYTES"
printf "  expect  : fp=%s\n" "$FP"
echo

echo "=== flash model -> $PORT @ $PART_OFFSET ==="
step 2 "flash model" \
  "$ESPTOOL" --chip "$CHIP" --port "$PORT" --baud 921600 \
  write_flash "$PART_OFFSET" "$MODEL"

# --input-dir: `upload` does not rebuild, so point it at the build just made.
echo "=== upload firmware ==="
step 1 "upload firmware" \
  arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir "$BUILD_DIR" "$SKETCH"

echo
echo "flashed : $MODEL_KIND from $MODEL"
echo "expect  : fp=$FP  bytes=$BYTES"
echo "the board prints 'build: bytes=... fp=...' at boot - both must match."
