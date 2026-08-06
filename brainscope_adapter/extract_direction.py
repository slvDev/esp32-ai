"""Extract a steering direction for the ESP32 twin - brainscope untouched.

Same trick as serve.py: register the PLE architecture, then hand over to
brainscope's own extract CLI. Defaults bake in the demo: the "dark" story-mood
direction from mood_pairs.jsonl at layer 3 (of 6), written next to the twin.

    ~/projekty/brainscope/.venv/bin/python brainscope_adapter/extract_direction.py
    ~/projekty/brainscope/.venv/bin/python brainscope_adapter/serve.py \
        --directions artifacts/tinystories/dirs.json
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BRAINSCOPE_REPO = Path.home() / "projekty" / "brainscope"

sys.path.insert(0, str(HERE))
if BRAINSCOPE_REPO.is_dir():
    sys.path.insert(0, str(BRAINSCOPE_REPO))

import ple_hf  # noqa: F401  (registers ple-tinylm with transformers)

from brainscope import extract  # noqa: E402

defaults = {
    "--model": str(ROOT / "artifacts" / "tinystories" / "hf"),
    "--pairs": str(HERE / "mood_pairs.jsonl"),
    "--layer": "3",
    "--name": "dark",
    "--out": str(ROOT / "artifacts" / "tinystories" / "dirs.json"),
}
extra = sys.argv[1:]
argv = ["extract"]
for flag, value in defaults.items():
    if flag not in extra:
        argv += [flag, value]
sys.argv = argv + extra
extract.main()
