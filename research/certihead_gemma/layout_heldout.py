"""CAMPAIGN 3, part 3 -- does the composite layout survive a held-out split?

`ever+norm` beat plain norm sort on TinyStories: 63.37% vs 62.21% pruned, rho
1.690 vs 1.610. It is built by partitioning rows on whether they were ever a
top-64 contender, then norm-sorting within each part -- taking headroom from the
partition and inflation from the norm order.

But `ever` is measured from a trajectory, so the layout is fitted to data. The
atlas put the contender set's first-half/second-half Jaccard at 0.651 for this
head, which is moderate, not high. A layout fitted to a calibration set and
evaluated on the same set will flatter itself.

This derives `ever` from the FIRST half of the positions and evaluates pruning on
the SECOND half only, against plain norm sort evaluated on the same second half.
Layout is an offline staging decision, so using calibration data is legitimate --
using the evaluation data is not.

Also reported: the layout derived from the second half and evaluated on the
first, so the result is not an artefact of which half happened to be used, and a
Gemma/Pythia column so a positive result is not assumed to transfer.
"""
import json
from pathlib import Path

import torch

from layout import load_ts, load_gemma, stats, evaluate

DATA = Path(__file__).resolve().parent / "data"


def composite(W, ever):
    V = W.shape[0]
    n = W.norm(dim=1)
    return torch.argsort(ever.long() * (V + 1) + torch.argsort(torch.argsort(n)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from pythia_dsweep import capture as pcap
    heads = [("TinyStories", load_ts), ("Gemma-270M", load_gemma)]
    for n in ("pythia-14m", "pythia-70m"):
        heads.append((n, lambda d, _n=n: pcap(_n)))
    out = []

    print("layout fitted on one half, pruning measured on the other\n")
    print(f"{'head':<14} {'fit/eval':<10} {'norm':>8} {'ever+norm':>10} "
          f"{'delta pp':>9} {'contender Jaccard':>18}")
    for name, loader in heads:
        W, H = loader(dev)
        half = H.shape[0] // 2
        A, B = H[:half], H[half:]
        _, everA, _ = stats(W, A)
        _, everB, _ = stats(W, B)
        jac = ((everA & everB).sum() / (everA | everB).sum().clamp(min=1)).item()
        nperm = torch.argsort(W.norm(dim=1))
        for tag, fit, ev in (("A -> B", everA, B), ("B -> A", everB, A)):
            _, _, pn = evaluate(W, ev, nperm)
            _, _, pc = evaluate(W, ev, composite(W, fit))
            print(f"{name:<14} {tag:<10} {pn:8.2f} {pc:10.2f} {pc-pn:+9.2f} "
                  f"{jac:18.3f}")
            out.append(dict(head=name, split=tag, norm=pn, composite=pc,
                            delta=pc - pn, jaccard=jac))
        del W, H
        torch.cuda.empty_cache()

    (DATA / "layout_heldout.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
