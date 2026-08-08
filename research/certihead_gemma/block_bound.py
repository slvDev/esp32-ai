"""A bound family that interpolates between the two we already know.

What the decomposition established
----------------------------------
A tile is skippable iff  inflation < headroom, both in units of |gmax|:

    TinyStories 25353 x  96   inflation 0.74   headroom 1.15   ratio 1.55
    Gemma-270M 262144 x 640   inflation 8.23   headroom 0.70   ratio 0.09

Headroom barely moves between the two models (0.70 vs 1.15). Inflation moves by
11x. And within Gemma, inflation tracks D almost alone -- 2.7 at D=96 rising to
11.0 at D=640 -- while a 10x change in V moves nothing (0.057 -> 0.085, all
still 100% surviving).

So the box bound's problem is structural: it takes an independent worst case in
every one of the D dimensions and adds them up. There is no cancellation, so
slack accumulates roughly linearly in D while the true score, which does cancel,
grows like sqrt(D). At D=96 that is affordable. At D=640 it is not.

The family
----------
Split the D dimensions into B contiguous blocks and apply Cauchy-Schwarz to the
residual WITHIN each block:

    u_j = <h, c_j> + sum_b ||h_b|| * max_i ||w_i,b - c_j,b||

with c_j the tile centroid. Sound for every B: each block term dominates that
block's residual contribution by Cauchy-Schwarz, and the sum of dominating terms
dominates the sum.

  B = 1    the centroid bound this project already tested and rejected at D=96,
           where it pruned nothing.
  B = D    per-dimension, i.e. essentially the box bound we deployed.
  1<B<D    new. Trades slack against index bytes: B blocks costs B floats per
           tile instead of 2*D.

The prediction worth testing
----------------------------
If inflation is dominated by counting D independent worst cases, then fewer,
larger blocks should shrink it at large D -- and the bound that LOST at D=96
should WIN at D=640. A crossover would mean the right certificate is a function
of the hidden dimension rather than a fixed design choice, which is a claim about
the algorithm, not about our board.

It would also be a second instance of this project's recurring failure mode: a
lever rejected under one configuration and never re-derived when the
configuration changed. The softmax head-split was the first.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def block_bounds(Wt, B):
    """Per-tile centroid and per-block max residual norm.

    Wt: [nt, T, D]  ->  cen [nt, D], res [nt, B], and the block edges.
    """
    nt, T, D = Wt.shape
    edges = [round(b * D / B) for b in range(B + 1)]
    cen = Wt.mean(dim=1)                                  # [nt, D]
    res = torch.empty(nt, B, device=Wt.device)
    for b in range(B):
        s, e = edges[b], edges[b + 1]
        d = Wt[:, :, s:e] - cen[:, None, s:e]             # [nt, T, blk]
        res[:, b] = d.norm(dim=2).amax(dim=1)             # max over rows in tile
    return cen, res, edges


def evaluate(W, H, T=128, blocks=(1, 2, 4, 8, 16, 32, 64), chunk=32, box=True):
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    Wf = Wp[: nt * T].T.contiguous()

    # Ground truth per chunk, computed once and reused across every bound.
    truth = []
    for i in range(0, H.shape[0], chunk):
        sc = H[i:i + chunk] @ Wf
        truth.append((sc.view(-1, nt, T).amax(dim=2), sc.amax(dim=1)))
        del sc

    rows = []
    if box:
        lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
        surv, infl = [], []
        for k, i in enumerate(range(0, H.shape[0], chunk)):
            h = H[i:i + chunk]
            u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
            m, g = truth[k]
            assert (u >= m - 1e-3 * u.abs().clamp(min=1)).all(), "box bound UNSOUND"
            surv.append((u >= g[:, None]).float().mean(dim=1))
            infl.append(((u - m) / g.abs().clamp(min=1e-9)[:, None]).flatten())
        rows.append(("box (=B:D)", 2 * D * 4, 100 * torch.cat(surv).mean().item(),
                     torch.cat(infl).median().item()))

    for B in blocks:
        cen, res, edges = block_bounds(Wt, B)
        surv, infl = [], []
        for k, i in enumerate(range(0, H.shape[0], chunk)):
            h = H[i:i + chunk]
            hb = torch.stack([h[:, edges[b]:edges[b + 1]].norm(dim=1)
                              for b in range(B)], dim=1)        # [b, B]
            u = h @ cen.T + hb @ res.T                          # [b, nt]
            m, g = truth[k]
            assert (u >= m - 1e-3 * u.abs().clamp(min=1)).all(), f"B={B} UNSOUND"
            surv.append((u >= g[:, None]).float().mean(dim=1))
            infl.append(((u - m) / g.abs().clamp(min=1e-9)[:, None]).flatten())
        # index cost per tile: centroid D floats + B block radii
        rows.append((f"blockCS B={B}", (D + B) * 4,
                     100 * torch.cat(surv).mean().item(),
                     torch.cat(infl).median().item()))
        del cen, res
    return rows, nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = dict(l.split() for l in (DATA / "ts_meta.txt").read_text().split("\n") if l)
    V, D = int(meta["rows"]), int(meta["cols"])
    Wts = torch.from_numpy(
        np.fromfile(DATA / "ts_W.f32", dtype=np.float32).reshape(V, D)).to(dev)
    Hts = torch.from_numpy(
        np.fromfile(DATA / "ts_x.f32", dtype=np.float32).reshape(-1, D)).to(dev)
    with safe_open(GEMMA / "model.safetensors", "pt") as f:
        Wg = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    Hg = torch.from_numpy(np.load(DATA / "h_gen.npy")).to(dev)[:256]

    for tag, W, H, blocks in (
            ("TinyStories 25353 x 96", Wts, Hts, (1, 2, 4, 8, 16, 32, 48)),
            ("Gemma-270M 262144 x 640", Wg, Hg, (1, 2, 4, 8, 16, 32, 64, 160, 320))):
        print(f"\n=== {tag}   tile {a.tile} ===")
        print(f"{'bound':<14} {'idx B/tile':>11} {'survive %':>10} {'inflation':>10}")
        rows, nt = evaluate(W, H, a.tile, blocks)
        best = min(r[2] for r in rows)
        for name, ib, s, inf in rows:
            mark = "  <-- best" if s == best else ""
            print(f"{name:<14} {ib:11d} {s:10.2f} {inf:10.3f}{mark}")
        print(f"({nt} tiles)")


if __name__ == "__main__":
    main()
