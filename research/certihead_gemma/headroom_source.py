"""Where does OUR head's headroom come from? The uncomfortable question.

Established so far: a tile is skippable iff inflation < headroom, and

    TinyStories 25353 x  96   inflation 0.74   headroom 1.15   -> prunes 62%
    Gemma-270M 262144 x 640   inflation >=6.47 headroom 0.70   -> prunes 0%

The negative on Gemma is robust: three bound families, tile sizes 32-512, V from
25k to 262k, and a perfect incumbent all leave 100% of tiles alive.

Which raises the question that decides whether the positive result means
anything. Headroom is the existence of tiles whose best row is far worse than the
global best. A head trained on a small corpus with a vocabulary it never fully
uses will have exactly that: regions of rows that are close to their
initialisation and align with nothing. Gemma's vocabulary is trained on
trillions of tokens and may simply have no dead regions.

If CertiHead's 62% pruning is really "62% of this model's vocabulary is
undertrained", then it is a property of our checkpoint and not a method. That is
worth knowing now.

The test: delete rows from the bottom of the norm distribution -- the most
likely home of undertrained rows -- and re-measure on the survivors. If pruning
survives the amputation, headroom is coming from genuine structure in a trained
head. If it collapses, it was coming from junk.

Reported alongside: how often each row is ever the argmax across the trajectory,
which says directly how much of the vocabulary this model actually uses.
"""
from pathlib import Path

import numpy as np
import torch

from bound_vs_dim import certified_scan, oracle_incumbent

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def survive(W, H, T=128, chunk=64):
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    if nt < 8:
        return float("nan"), float("nan")
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    s, inf = [], []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        g = sc.amax(dim=1)
        s.append((u >= g[:, None]).float().mean(dim=1))
        inf.append(((u - m) / g.abs().clamp(min=1e-9)[:, None]).flatten())
        del sc, u, m
    return 100 * torch.cat(s).mean().item(), torch.cat(inf).median().item()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = dict(l.split() for l in (DATA / "ts_meta.txt").read_text().split("\n") if l)
    V, D = int(meta["rows"]), int(meta["cols"])
    W = torch.from_numpy(
        np.fromfile(DATA / "ts_W.f32", dtype=np.float32).reshape(V, D)).to(dev)
    H = torch.from_numpy(
        np.fromfile(DATA / "ts_x.f32", dtype=np.float32).reshape(-1, D)).to(dev)

    # How much of this vocabulary does the model ever actually emit?
    wins = torch.zeros(V, dtype=torch.long, device=dev)
    for i in range(0, H.shape[0], 64):
        wins += torch.bincount((H[i:i + 64] @ W.T).argmax(dim=1), minlength=V)
    used = int((wins > 0).sum())
    print(f"TinyStories head {V} x {D}, {H.shape[0]} positions")
    print(f"distinct argmax tokens over the trajectory: {used} "
          f"({100*used/V:.2f}% of the head)\n")

    n = W.norm(dim=1)
    order = torch.argsort(n, descending=True)     # keep the largest-norm rows
    print(f"{'kept rows':>10} {'drop %':>7} {'survive %':>10} {'inflation':>10}   "
          f"note")
    for frac in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
        k = int(V * frac)
        keep = order[:k]
        # A dropped row can no longer be the argmax, so the ground truth moves
        # with the head. That is the point: this asks what the certificate does
        # on a head WITHOUT those rows, not what it does while ignoring them.
        s, inf = survive(W[keep], H)
        print(f"{k:10d} {100*(1-frac):7.0f} {s:10.2f} {inf:10.3f}   "
              f"{'baseline' if frac == 1.0 else ''}")

    print("\nsame amputation from the OTHER end (drop the largest norms):")
    rev = torch.argsort(n)                        # keep the smallest-norm rows
    for frac in (0.9, 0.75, 0.5):
        k = int(V * frac)
        s, inf = survive(W[rev[:k]], H)
        print(f"{k:10d} {100*(1-frac):7.0f} {s:10.2f} {inf:10.3f}")


if __name__ == "__main__":
    main()
