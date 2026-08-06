"""Serve the ESP32's deployed model in brainscope, untouched brainscope.

Registers the PLE architecture with transformers in-process (import ple_hf) and
then hands control to brainscope's own CLI, pointed at the twin built by
build_hf.py. Any extra arguments pass straight through to brainscope
(--port, --lens, --no-browser, ...).

Run with brainscope's environment:
    ~/projekty/brainscope/.venv/bin/python brainscope_adapter/serve.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BRAINSCOPE_REPO = Path.home() / "projekty" / "brainscope"
# PLE_HF_DIR selects which twin to serve, e.g. the --zero-ple-table build.
HF_DIR = Path(os.environ.get("PLE_HF_DIR", ROOT / "artifacts" / "tinystories" / "hf"))

sys.path.insert(0, str(HERE))
if BRAINSCOPE_REPO.is_dir():
    sys.path.insert(0, str(BRAINSCOPE_REPO))

import ple_hf  # noqa: F401  (registers ple-tinylm with transformers)

if not HF_DIR.is_dir():
    raise SystemExit(f"{HF_DIR} missing - run: python brainscope_adapter/build_hf.py")

from brainscope import server  # noqa: E402

extra = sys.argv[1:]
argv = ["brainscope", "--model", str(HF_DIR)]
if "--lens" not in extra:
    argv += ["--lens", "on"]  # 6 layers x 96 dims: the lens is free, keep it on
sys.argv = argv + extra
server.main()
