"""CAMPAIGN 3, part 2 -- what the norm permutation is actually for.

Two hypotheses have now died on this question:

    "inert rows drive pruning"        Spearman(inert fraction, pruning) = -0.943
    "inert TILES drive pruning"       Spearman(dead tiles, pruning)     = +0.214

The second died hard. The `everk` layout, which sorts contenders away from
non-contenders, achieves 88.9%-98.3% dead tiles on every head and prunes 21.9%
(TinyStories), 1.5% (Gemma), 0% (pythia-14m/70m). Meanwhile plain norm sort has
only 52% dead tiles on TinyStories and prunes 62.2%. Concentrating inert rows is
neither necessary nor sufficient.

The remaining explanation is that the norm sort is not working on headroom at
all. It works on INFLATION. `u_j - m_j` comes from the per-dimension spread of a
tile's rows, and rows of similar norm have similar spread, so norm-homogeneous
tiles have tight lo/hi. `everk` tiles are homogeneous in contender-status but
contain rows of every magnitude, so their bounds are loose.

That predicts:

  P1. Within a head, inflation should track within-tile norm dispersion, and
      norm sort should minimise inflation among all layouts tested.
  P2. Norm sort should ALSO buy headroom, but only on heads where ||w||
      correlates with competitiveness -- which would explain why it is
      transformative on TinyStories and inert on Pythia and Gemma.
  P3. A composite layout -- partition by contender status for headroom, sort by
      norm within each part for inflation -- should beat both, if the two
      mechanisms are separable.

P3 is the one that could produce a better layout for the board, so it is
constructed and measured rather than argued.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from layout import load_ts, load_gemma, stats, evaluate

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def inflation_of(W, H, perm, T=128, chunk=32):
    """Median inflation (u_j - m_j) normalised by the head's own score spread.

    Normalised by the interquartile range of the logits rather than |gmax|, so
    it is comparable across layouts of the SAME head without inheriting the
    scale bug EXP-125 had to withdraw.
    """
    V, D = W.shape
    Wp = W[perm]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    inf, spread = [], []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        inf.append((u - m).flatten())
        spread.append(sc.std(dim=1))
        del sc, u, m
    return (torch.cat(inf).median() / torch.cat(spread).median()).item()


def norm_dispersion(W, perm, T=128):
    """Mean within-tile coefficient of variation of ||w||."""
    n = W.norm(dim=1)[perm]
    nt = n.numel() // T
    t = n[: nt * T].view(nt, T)
    return (t.std(dim=1) / t.mean(dim=1).clamp(min=1e-9)).mean().item()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from pythia_dsweep import capture as pcap
    heads = [("TinyStories", load_ts), ("Gemma-270M", load_gemma)]
    for n in ("pythia-14m", "pythia-70m"):
        heads.append((n, lambda d, _n=n: pcap(_n)))
    g = torch.Generator().manual_seed(0)
    out = []

    print("P2 first: does ||w|| carry any information about competitiveness?\n")
    print(f"{'head':<14} {'corr(||w||, best score)':>24} {'corr(||w||, wins)':>19}")
    comp = {}
    for name, loader in heads:
        W, H = loader(dev)
        wins, ever, best = stats(W, H)
        n = W.norm(dim=1)
        rk = lambda x: x.float().argsort().argsort().float()
        c1 = torch.corrcoef(torch.stack([rk(n), rk(best)]))[0, 1].item()
        c2 = torch.corrcoef(torch.stack([rk(n), rk(wins)]))[0, 1].item()
        print(f"{name:<14} {c1:24.3f} {c2:19.3f}")
        comp[name] = (c1, c2)
        del W, H
        torch.cuda.empty_cache()

    print("\nP1 and P3: inflation, norm dispersion, and a composite layout\n")
    for name, loader in heads:
        W, H = loader(dev)
        V = W.shape[0]
        wins, ever, best = stats(W, H)
        n = W.norm(dim=1)
        # composite: contenders last, norm-sorted within each part
        key = ever.long() * (V + 1) + torch.argsort(torch.argsort(n))
        layouts = {
            "norm": torch.argsort(n),
            "everk": torch.argsort(ever.int()),
            "ever+norm": torch.argsort(key),
            "random": torch.randperm(V, generator=g).to(dev),
        }
        print(f"=== {name} ===")
        print(f"{'layout':<12} {'norm disp':>10} {'inflation':>10} "
              f"{'dead tiles':>11} {'rho p50':>9} {'prune %':>9}")
        for ln, perm in layouts.items():
            disp = norm_dispersion(W, perm)
            infl = inflation_of(W, H, perm)
            dt, r, p = evaluate(W, H, perm, ever=ever)
            print(f"{ln:<12} {disp:10.4f} {infl:10.3f} {dt:11.4f} {r:9.3f} {p:9.2f}")
            out.append(dict(head=name, layout=ln, norm_disp=disp, inflation=infl,
                            dead_tiles=dt, rho=r, prune=p,
                            corr_norm_best=comp[name][0]))
        print()
        del W, H
        torch.cuda.empty_cache()

    (DATA / "layout2.json").write_text(json.dumps(out, indent=2))
    a = np.array([[r["norm_disp"], r["inflation"]] for r in out])
    rk = lambda x: x.argsort().argsort()
    print(f"Spearman(within-tile norm dispersion, inflation) = "
          f"{np.corrcoef(rk(a[:,0]), rk(a[:,1]))[0,1]:.3f}   "
          f"over {len(a)} head x layout points")


if __name__ == "__main__":
    main()
