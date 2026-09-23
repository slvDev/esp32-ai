"""Gate: the brainscope twin must match the C runtime that ships to the board.

Two checks, both on the deployed artifact:
  1. C-vs-Python logits on a fixed prompt (same prompt family as the exporter's
     golden). The C side is runtime/llm.h - the code verify.c gates before
     flashing - so agreement here means brainscope shows the device's model,
     not an approximation of it.
  2. KV-cache parity: decoding token-by-token (what brainscope does) must equal
     one full forward.

Run from the repo root:
    python brainscope_adapter/verify_vs_c.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hf import build  # noqa: E402

PROMPT = [1, 500, 1000, 200, 42, 777, 13, 99]
TOLERANCE = 0.02  # same bar verify.c holds the C port to against PyTorch


def c_logits(artifacts: Path) -> np.ndarray:
    exe = ROOT / "brainscope_adapter" / "dump_logits"
    subprocess.run(
        ["cc", "-O3", "-Wall", "-Wextra", "-I", str(ROOT / "runtime"),
         "-o", str(exe), str(ROOT / "brainscope_adapter" / "dump_logits.c"), "-lm"],
        check=True)
    out = subprocess.run(
        [str(exe), str(artifacts / "model.bin"), *map(str, PROMPT)],
        check=True, capture_output=True, text=True).stdout
    return np.array([float(v) for v in out.split()], dtype=np.float32)


def main():
    artifacts = ROOT / "artifacts" / "tinystories"
    model = build(artifacts)
    ids = torch.tensor([PROMPT])

    with torch.no_grad():
        full = model(input_ids=ids).logits[0, -1].numpy()

    ref = c_logits(artifacts)
    assert full.shape == ref.shape, f"vocab mismatch {full.shape} vs {ref.shape}"
    diff = np.abs(full - ref)
    print(f"C vs Python: max abs diff {diff.max():.6f}  rms {np.sqrt((diff**2).mean()):.6f}")
    print(f"top token:   C={int(ref.argmax())}  Python={int(full.argmax())}")
    ok_c = diff.max() < TOLERANCE and ref.argmax() == full.argmax()

    with torch.no_grad():
        past = None
        for t in PROMPT:
            out = model(input_ids=torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
        step = out.logits[0, -1].numpy()
    cache_diff = np.abs(step - full).max()
    print(f"KV-cache vs full forward: max abs diff {cache_diff:.8f}")
    ok_cache = cache_diff < 1e-4

    if ok_c and ok_cache:
        print("PASS: brainscope twin matches the device runtime")
        return 0
    print("FAIL: twin diverges from the device runtime")
    return 2


if __name__ == "__main__":
    sys.exit(main())
