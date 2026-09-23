# The ESP32's model, under brainscope

Loads the exact model this repo flashes to an ESP32-S3 into
[brainscope](https://github.com/moudrkat/brainscope) - logit lens, attention,
per-layer activity, live. Not the training checkpoint: the int4 weights are
dequantized straight out of `artifacts/tinystories/model.bin`, the same bytes
the board mmaps from flash. `verify_vs_c.py` proves the twin against the C
runtime (`runtime/llm.h`) that ships to the device - last-position logits agree
to ~1e-5, the same fp32-from-int4 path `verify.c` gates before flashing.

brainscope itself is untouched: the PLE architecture is registered with
transformers in-process and served through brainscope's public CLI.

```bash
scripts/fetch_model.sh tinystories                      # if artifacts/ is empty
$BRAINSCOPE_PY brainscope_adapter/build_hf.py           # model.bin -> hf twin
$BRAINSCOPE_PY brainscope_adapter/verify_vs_c.py        # gate vs the C runtime
$BRAINSCOPE_PY brainscope_adapter/serve.py              # brainscope on :8010
```

`$BRAINSCOPE_PY` is any python with `torch`, `transformers` and brainscope
importable. `serve.py` looks for a brainscope checkout at
`~/projekty/brainscope`; with brainscope pip-installed, the path insert is
simply unused.

The model is not a chat model - it continues text. The tokenizer's chat
template therefore concatenates all message contents verbatim, which turns
brainscope's chat box into a continue-the-story box: type an opening, watch
6 layers x 96 dims write the rest at full visibility.

| file | role |
|---|---|
| `ple_bin.py` | parse + dequantize `model.bin` (int4 groups, fp16 scales) |
| `ple_hf.py` | the PLE architecture in brainscope's expected skeleton |
| `build_hf.py` | write `artifacts/tinystories/hf/` + smoke sample |
| `dump_logits.c` | C-runtime logits for a prompt, via `runtime/llm.h` |
| `verify_vs_c.py` | twin-vs-C gate + KV-cache parity |
| `serve.py` | register the architecture, hand over to brainscope |
