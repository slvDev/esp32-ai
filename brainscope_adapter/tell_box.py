"""Send one prompt to the matchbox AND to brainscope, in lockstep.

The board encodes the prompt itself and decodes greedily on-chip; brainscope's
twin is bit-faithful to those weights, so fed the same prompt at temperature 0
it writes the SAME story - the OLED shows the words, the open brainscope tab
shows the layers producing them.

    sg dialout -c 'python brainscope_adapter/tell_box.py "One day, a little cat"'

The board listens between stories; if it is mid-story, the prompt waits in its
serial buffer until the current one finishes (up to ~45 s).
"""

import argparse
import subprocess
import sys
import urllib.request
import json

PORT = "/dev/ttyACM0"
BRAINSCOPE = "http://localhost:8010/v1/chat/completions"
UNPLUGGED = "http://localhost:8011/v1/chat/completions"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", help="ASCII story opening, e.g. 'Once upon a time'")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="brainscope-side cap; the board itself writes 200")
    ap.add_argument("--also-unplugged", action="store_true",
                    help="feed the same prompt to the flash-unplugged twin on "
                         ":8011 too - three minds, one prompt")
    args = ap.parse_args()
    if not args.prompt.isascii():
        sys.exit("the device tokenizer is ASCII-only (no diacritics)")

    subprocess.run(["stty", "-F", args.port, "115200", "raw", "-echo"], check=True)
    with open(args.port, "wb", buffering=0) as ser:
        ser.write(args.prompt.encode("ascii") + b"\n")
    print(f"box     <- {args.prompt!r}")

    def ask(url):
        req = urllib.request.Request(
            url,
            data=json.dumps({
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            }).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    print("brainscope: generating (watch the open tab)...")
    text = ask(BRAINSCOPE)
    print(f"twin story: {text[:160]}{'...' if len(text) > 160 else ''}")
    if args.also_unplugged:
        broken = ask(UNPLUGGED)
        print(f"unplugged : {broken[:80]!r}...")
    print("the OLED should be writing the same words as the twin, ~10 tok/s.")


if __name__ == "__main__":
    main()
