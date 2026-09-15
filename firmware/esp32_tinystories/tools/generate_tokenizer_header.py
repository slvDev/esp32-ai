"""Build this sketch's encoder header from its tokenizer.

Writes generated/tokenizer_encoder.h: the BTK1 asset runtime/bpe_tokenizer.h
loads, so a prompt typed over serial is encoded on the device with the same
ByteLevel BPE the model was trained with.

The packer itself lives with Barista, which introduced the on-device encoder;
this is that tool pointed at the TinyStories tokenizer. The one difference is
<|endoftext|>, a special added token: the model emits it to end a story, and
a user never has to type it, so the packer is told to accept it.

  uv run python firmware/esp32_tinystories/tools/generate_tokenizer_header.py
"""

import argparse
import sys
from pathlib import Path

SKETCH = Path(__file__).resolve().parent.parent
ROOT = SKETCH.parent.parent
DEFAULT_TOKENIZER = ROOT / "artifacts" / "tinystories" / "tokenizer.json"
DEFAULT_OUT = SKETCH / "generated" / "tokenizer_encoder.h"

sys.path.insert(0, str(ROOT / "firmware" / "esp32_barista" / "tools"))
from generate_tokenizer_header import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Pack the TinyStories BPE encoder tables into a C header.")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="canonical tokenizer.json the asset is built from")
    ap.add_argument("--out", default=DEFAULT_OUT, help="header to write")
    ap.add_argument("--asset", default=None,
                    help="also write the raw BTK1 bytes here, for host checks")
    args = ap.parse_args()
    generate(args.tokenizer, args.out, args.asset, allow_special_added_tokens=True)


if __name__ == "__main__":
    main()
