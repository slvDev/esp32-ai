"""Capture the REAL input to Gemma-3-270M's output head.

Why this exists
---------------
The CertiHead box bound is `u = sum_j max(h_j*lo_j, h_j*hi_j)`. Every dimension
contributes an independent slack term, so bound tightness is a function of D --
and that is exactly the property that already killed the Cauchy-Schwarz bound on
this project at D=96. Our deployed head is D=96. Gemma-3-270M's is D=640.

Whether CertiHead is a decoding primitive or a small-hidden-dimension method is
therefore an empirical question about slack-vs-D, and it has to be answered on
REAL hidden states. Random h is not admissible: the bound's tightness depends on
the correlation between h and the per-dimension row spread of the tile, and
random vectors destroy exactly that structure.

Two sets are captured because they are not the same distribution:
  TF  -- teacher-forced over real prose. Broad coverage of contexts.
  GEN -- captured DURING autoregressive generation. This is the deployment
         condition: it is the h that CertiHead would actually be handed.

Both are the tensor fed to lm_head, i.e. post-final-RMSNorm, captured with a
forward hook rather than reconstructed, so there is no chance of applying the
norm twice or missing a scaling factor.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Prompts for the GEN set. Deliberately spread across registers -- narrative,
# technical, dialogue, list-forming, code -- because the certificate's job is
# hardest when the next token is genuinely uncertain, and a single register
# would sample only one shape of logit distribution.
PROMPTS = [
    "Once upon a time in a small village by the sea,",
    "The main difference between a microcontroller and a microprocessor is",
    "Q: How do I reverse a linked list in C?\nA:",
    "Ingredients for a simple tomato soup:\n1.",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "The treaty was signed in the spring of 1815, and its consequences",
    "Dear Sir or Madam,\n\nI am writing to enquire about",
    "In this paper we present a new method for",
    "The three most common causes of engine failure are",
    "She opened the door and immediately knew that something",
    "Translate to French: The weather is beautiful today.\n",
    "A prime number is a natural number greater than 1 that",
    "SELECT name, COUNT(*) FROM orders WHERE",
    "The patient presented with a persistent cough and",
    "Chapter One\n\nIt was the kind of morning that",
    "To configure the network interface, first edit",
    "The derivative of sin(x) with respect to x is",
    "Breaking: officials confirmed early this morning that",
    "My grandmother always said that the secret to",
    "Summary of the meeting held on Tuesday:\n\n-",
]

# Real prose for the TF set. The model card is genuine technical English that is
# already on disk; using it avoids inventing a corpus and keeps the input
# auditable -- anyone re-running this reads the same bytes.
TF_SOURCE = MODEL / "README.md"


def main():
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.eval()

    # Hook the actual head input. Reconstructing it from output_hidden_states
    # invites an off-by-one-norm error; the hook cannot be wrong about what the
    # head was handed.
    grabbed = []

    def hook(_mod, inputs, _out):
        grabbed.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu())

    h = model.lm_head.register_forward_hook(hook)

    # ---- GEN: h as seen during real autoregressive decoding ----
    gen_rows = []
    with torch.no_grad():
        for p in PROMPTS:
            grabbed.clear()
            ids = tok(p, return_tensors="pt").to(dev)
            model.generate(**ids, max_new_tokens=n_gen, do_sample=False,
                           pad_token_id=tok.pad_token_id)
            # First call covers the prompt prefill; later calls are one step each.
            # Keep only the position that was actually decoded from, which is the
            # last row of every call -- that is the h the head consumed to emit a
            # token. Prefill's earlier rows are teacher-forcing, not decoding.
            for g in grabbed:
                gen_rows.append(g[-1:])
    gen = torch.cat(gen_rows).numpy().astype(np.float32)

    # ---- TF: teacher-forced over real prose ----
    text = TF_SOURCE.read_text(encoding="utf-8", errors="ignore")
    ids = tok(text, return_tensors="pt").input_ids[:, : 64 * 128].to(dev)
    grabbed.clear()
    with torch.no_grad():
        for i in range(0, ids.shape[1], 512):
            model(ids[:, i:i + 512])
    tf = torch.cat(grabbed).numpy().astype(np.float32)

    h.remove()

    np.save(OUT / "h_gen.npy", gen)
    np.save(OUT / "h_tf.npy", tf)
    meta = {
        "model": str(MODEL),
        "hidden_size": int(gen.shape[1]),
        "gen_rows": int(gen.shape[0]),
        "tf_rows": int(tf.shape[0]),
        "gen_prompts": len(PROMPTS),
        "gen_new_tokens_each": n_gen,
        "tf_source": str(TF_SOURCE),
        "capture": "forward hook on lm_head input (post final RMSNorm)",
        "dtype_forward": "float32",
    }
    (OUT / "capture_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"GEN |h| mean {np.linalg.norm(gen, axis=1).mean():.3f}   "
          f"TF |h| mean {np.linalg.norm(tf, axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
