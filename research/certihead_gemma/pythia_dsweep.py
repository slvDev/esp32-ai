"""n=2 is not a finding. Pythia makes it a controlled sweep.

The problem with EXP-123
------------------------
It compared exactly two heads and concluded CertiHead is a low-hidden-dimension
method:

    TinyStories 25353 x  96   inflation 0.74  headroom 1.15  ->  prunes 62%
    Gemma-270M 262144 x 640   inflation 8.23  headroom 0.70  ->  prunes  0%

Those two differ in D, in V, in tokenizer, in training corpus, in training
budget, in architecture and in who trained them. Attributing the difference to D
was the most consistent reading of the internal sweeps, not something the
comparison could establish. The D sweep inside Gemma was confounded too, because
truncating dimensions shrinks the true score along with the bound.

Pythia removes every confound at once
-------------------------------------
The suite trains one architecture on one corpus with one recipe at several
widths, and -- unlike both heads above -- does NOT tie the output head to the
input embedding, so these are real dedicated output matrices:

    pythia-14m    D=128    V=50304
    pythia-31m    D=256    V=50304
    pythia-70m    D=512    V=50304
    pythia-160m   D=768    V=50304

Same vocabulary, same data, same tokenizer, same recipe, four widths spanning
our D=96 and Gemma's D=640. If inflation rises with D across these four and
pruning dies somewhere in the middle, "low-D method" is a finding. If inflation
is flat, D was never the variable and EXP-123's conclusion has to be withdrawn.

Hidden states are captured the same way as for Gemma: a forward hook on the
output head during real autoregressive generation, so h is what the head is
actually handed at decode time rather than a teacher-forced approximation.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bound_vs_dim import certified_scan, oracle_incumbent

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
PROMPTS = [
    "Once upon a time in a small village by the sea,",
    "The main difference between a microcontroller and a microprocessor is",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "The treaty was signed in the spring of 1815, and its consequences",
    "In this paper we present a new method for",
    "She opened the door and immediately knew that something",
    "A prime number is a natural number greater than 1 that",
    "Chapter One\n\nIt was the kind of morning that",
]


def capture(name, n_new=48):
    """Hidden states entering the output head during real decoding."""
    p = MODELS / name
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32).to(dev).eval()

    grabbed = []
    head = model.get_output_embeddings()
    hk = head.register_forward_hook(
        lambda m, i, o: grabbed.append(i[0].detach().reshape(-1, i[0].shape[-1]).cpu()))
    rows = []
    with torch.no_grad():
        for pr in PROMPTS:
            grabbed.clear()
            ids = tok(pr, return_tensors="pt").to(dev)
            model.generate(**ids, max_new_tokens=n_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
            # last row of each call = the position actually decoded from
            rows += [g[-1:] for g in grabbed]
    hk.remove()
    H = torch.cat(rows).to(dev)
    W = head.weight.detach().float().to(dev)          # [V, D], untied
    del model
    torch.cuda.empty_cache()
    return W, H


def measure(W, H, T=128, chunk=32):
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    surv, infl, head = [], [], []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        g = sc.amax(dim=1)
        first, best = certified_scan(u, m)
        assert torch.equal(best, g), "UNSOUND: certified scan lost the true max"
        s = g.abs().clamp(min=1e-9)[:, None]
        surv.append(oracle_incumbent(u, g).float() / nt)
        infl.append(((u - m) / s).flatten())
        head.append(((g[:, None] - m) / s).flatten())
        del sc, u, m
    return (100 * torch.cat(surv).mean().item(),
            torch.cat(infl).median().item(),
            torch.cat(head).median().item())


def main():
    print("Pythia suite: one recipe, one corpus, one vocabulary (50304), "
          "untied heads, four widths\n")
    print(f"{'model':<14} {'D':>5} {'V':>7} {'survive %':>10} {'inflation':>10} "
          f"{'headroom':>9} {'ratio':>7}")
    res = []
    for name in ("pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m"):
        W, H = capture(name)
        s, inf, hd = measure(W, H)
        res.append(dict(model=name, D=W.shape[1], V=W.shape[0], survive=s,
                        inflation=inf, headroom=hd, ratio=hd / inf,
                        positions=H.shape[0]))
        print(f"{name:<14} {W.shape[1]:5d} {W.shape[0]:7d} {s:10.2f} {inf:10.3f} "
              f"{hd:9.3f} {hd/inf:7.3f}")
        del W, H
        torch.cuda.empty_cache()

    print("\nfor reference, from EXP-123 (different corpora/tokenizers/recipes):")
    print(f"{'TinyStories':<14} {96:5d} {25353:7d} {37.79:10.2f} {0.740:10.3f} "
          f"{1.150:9.3f} {1.554:7.3f}")
    print(f"{'Gemma-3-270M':<14} {640:5d} {262144:7d} {100.00:10.2f} {8.228:10.3f} "
          f"{0.697:9.3f} {0.085:7.3f}")
    (HERE / "data" / "pythia_dsweep.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
