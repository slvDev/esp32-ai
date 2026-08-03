#!/usr/bin/env python3
"""Fetch the published model artifacts for the ESP32 firmware.

The trained 28.9M-parameter checkpoint is not committed (see .gitignore) and
reproducing it requires training. The released weights are hosted at
MODEL_URL once published (tracked in issues #5 and #7); this script downloads
model.bin, verifies its SHA-256 against the hash documented in
firmware/esp32_llm/README.md, and writes it into firmware/model/ next to the
golden files the host verifier needs.

Usage:
    python tools/fetch_model.py [--url URL] [--sha HEX]
    python tools/fetch_model.py --check-only

Stdlib only, so it runs anywhere.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "firmware", "model")
MODEL_BIN = os.path.join(OUT, "model.bin")

# Fill this in once the weights are released (issue #5). Keep in sync with the
# SHA-256 printed in firmware/esp32_llm/README.md.
MODEL_URL = ""

# SHA-256 of the artifact used for the on-device measurements, from
# firmware/esp32_llm/README.md.
EXPECTED_SHA = "21067f5d78113f6c64a8720b05ff7e5c774dab0276797a522f81a6797253d97c"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def check_only():
    if not os.path.exists(MODEL_BIN):
        print(f"not present: {MODEL_BIN}")
        return 1
    got = sha256_of(MODEL_BIN)
    ok = got == EXPECTED_SHA
    print(f"{MODEL_BIN}")
    print(f"  expected {EXPECTED_SHA}")
    print(f"  got      {got}  {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=MODEL_URL, help="direct URL to model.bin")
    ap.add_argument("--sha", default=EXPECTED_SHA, help="expected SHA-256")
    ap.add_argument("--check-only", action="store_true",
                    help="verify an already-downloaded model.bin and exit")
    args = ap.parse_args()

    if args.check_only:
        return check_only()

    if not args.url:
        print("MODEL_URL is not set yet -- the trained weights are not published.")
        print("Track issue #7 (https://github.com/slvDev/esp32-ai/issues/7) and")
        print("issue #5 for the release. Once available, set MODEL_URL or pass --url.")
        return 1

    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), "model.bin.download")
    print(f"downloading {args.url}")
    urllib.request.urlretrieve(args.url, tmp)
    got = sha256_of(tmp)
    if got != args.sha:
        print(f"SHA-256 mismatch: expected {args.sha}, got {got}")
        return 1
    shutil.move(tmp, MODEL_BIN)
    print(f"verified + saved {MODEL_BIN} ({os.path.getsize(MODEL_BIN) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
