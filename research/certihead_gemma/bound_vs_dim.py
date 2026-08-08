"""Does the CertiHead box bound survive D=640?

The question
------------
CertiHead prunes a tile of output rows when a per-dimension box bound proves no
row in it can beat the incumbent:

    lo_j = min over rows in tile of w_j        hi_j = max over rows in tile of w_j
    u    = sum_j max(h_j * lo_j, h_j * hi_j)

Every dimension contributes an independent slack term. The bound is therefore
loosest exactly where D is largest, and this project has already watched a bound
die of exactly this: Cauchy-Schwarz pruned NOTHING at D=96 while the box bound
pruned to a third.

Our deployed head is 25,353 x 96. Gemma-3-270M's is 262,144 x 640 -- 10.3x the
vocabulary and 6.7x the hidden dimension. If slack grows fast enough with D, the
method is a small-hidden-dimension method and the paper's framing changes. That
outcome is not a failure; it is the answer, and it is cheaper to have it now than
in six weeks.

This measures the NECESSARY CONDITION only: exact fp32 arithmetic, no
quantization anywhere. If the bound cannot prune in fp32 it certainly cannot
prune once the index is quantized to int8 with outward rounding, so a negative
result here is decisive and a positive result still has to clear the quantized
gate afterwards.

Reported per (D, tile size):
  rows scanned   -- the metric that actually costs money; mean/p50/p90/p99
  slack ratio    -- u / (true max in tile), how much the bound overestimates
  exactness      -- certified argmax vs dense argmax; must be 1.000 or the
                    implementation is wrong, not the method
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
MODEL = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def load_head(dev):
    with safe_open(MODEL / "model.safetensors", "pt") as f:
        w = f.get_tensor("model.embed_tokens.weight")
    # Gemma ties the output head to the input embedding table; there is no
    # separate lm_head tensor and no logit softcapping in this config, so the
    # head is exactly h @ W.T and argmax over it is the deployed decision.
    return w.float().to(dev)


def certified_scan(u, tmax):
    """Rows-scanned model matching the firmware's scan loop.

    Tiles are visited in descending bound order. The incumbent is the best true
    score among tiles already scanned. The loop breaks at the first tile whose
    bound cannot beat the incumbent -- correct because the bounds are sorted
    descending and the incumbent only grows, so every later tile is also dead.

    u, tmax: [tokens, n_tiles]
    """
    order = torch.argsort(u, dim=1, descending=True)
    us = torch.gather(u, 1, order)
    ms = torch.gather(tmax, 1, order)
    # incumbent BEFORE scanning tile i = running max of tiles 0..i-1
    prefix = torch.cummax(ms, dim=1).values
    prefix = torch.cat([torch.full_like(prefix[:, :1], -float("inf")),
                        prefix[:, :-1]], dim=1)
    dead = us < prefix                     # tile i cannot contain the winner
    # first dead index == number of tiles scanned
    any_dead = dead.any(dim=1)
    first = torch.where(any_dead, dead.float().argmax(dim=1),
                        torch.full_like(dead[:, 0], dead.shape[1], dtype=torch.long))
    return first, ms.max(dim=1).values


def oracle_incumbent(u, true_max):
    """Tiles that survive against a PERFECT incumbent.

    The device seeds the incumbent from a hot set -- last position's winner and
    runners-up -- which is what makes the bounds bite in practice. Modelling that
    heuristic approximately would leave the result arguable.

    Instead, seed the incumbent with the true global maximum. No incumbent
    strategy can ever do better than knowing the answer in advance, so the tiles
    surviving here are a LOWER BOUND on the tiles any certified scan must visit,
    for any hot set, any warm start, any ordering.

    If this is ~100%, the bound is dead and no amount of incumbent engineering
    can revive it. That is a decisive negative, not a tuning problem.
    """
    return (u >= true_max.unsqueeze(1)).sum(dim=1)


def run(W, H, dims, tiles, dev, chunk=64):
    V, Dfull = W.shape
    out = []
    for D in dims:
        Wd = W[:, :D].contiguous()
        Hd = H[:, :D].contiguous()
        # Row permutation by ||w||: the ordering CertiHead stages on device.
        # It is what makes a tile's rows similar in magnitude, which is what
        # makes lo/hi tight. Measured on this project to matter more than the
        # bound formula itself.
        nrm = Wd.norm(dim=1)
        perm = torch.argsort(nrm)
        Wp = Wd[perm]
        for T in tiles:
            nt = V // T
            Wt = Wp[: nt * T].view(nt, T, D)
            lo = Wt.amin(dim=1)            # [nt, D]
            hi = Wt.amax(dim=1)
            scanned, oracle, slack, exact_ok, n = [], [], [], 0, 0
            for i in range(0, Hd.shape[0], chunk):
                h = Hd[i:i + chunk]
                hp = h.clamp(min=0)
                hn = h.clamp(max=0)
                u = hp @ hi.T + hn @ lo.T           # [b, nt] upper bound
                sc = h @ Wp[: nt * T].T             # [b, V] dense truth
                tm = sc.view(h.shape[0], nt, T).amax(dim=2)
                gmax = sc.amax(dim=1)
                first, best = certified_scan(u, tm)
                scanned.append(first.float() * T)
                oracle.append(oracle_incumbent(u, gmax).float() * T)
                # slack only where the tile has a positive true max, else the
                # ratio is meaningless / sign-flipped
                pos = tm > 0
                slack.append((u[pos] / tm[pos]).float())
                exact_ok += (best == gmax).sum().item()
                n += h.shape[0]
                del sc, u, tm
            s = torch.cat(scanned)
            o = torch.cat(oracle)
            sl = torch.cat(slack)
            q = torch.quantile(s, torch.tensor([.5, .9, .99], device=dev))
            out.append(dict(
                D=D, tile=T, n_tiles=nt,
                rows_mean=s.mean().item(), frac_mean=s.mean().item() / (nt * T),
                p50=q[0].item(), p90=q[1].item(), p99=q[2].item(),
                oracle_rows_mean=o.mean().item(),
                oracle_frac=o.mean().item() / (nt * T),
                slack_med=sl.median().item(),
                slack_p90=torch.quantile(sl, 0.9).item(),
                exact=exact_ok / n,
                index_bytes_int8=nt * D * 2,
            ))
            print(f"D={D:4d} tile={T:4d}  scan {100*out[-1]['frac_mean']:5.1f}%  "
                  f"ORACLE-INC {100*out[-1]['oracle_frac']:5.1f}%  "
                  f"slack med {out[-1]['slack_med']:6.2f}x "
                  f"p90 {out[-1]['slack_p90']:6.2f}x  "
                  f"exact {out[-1]['exact']:.4f}  "
                  f"idx {out[-1]['index_bytes_int8']/1024:7.0f} KiB")
            del lo, hi, Wt
            torch.cuda.empty_cache()
        del Wd, Wp
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="gen", choices=["gen", "tf"])
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--dims", type=int, nargs="+", default=[96, 160, 256, 384, 512, 640])
    ap.add_argument("--tiles", type=int, nargs="+", default=[64, 128, 256, 512])
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = load_head(dev)
    H = torch.from_numpy(np.load(HERE / "data" / f"h_{a.set}.npy")).to(dev)
    H = H[: a.tokens]
    print(f"head {tuple(W.shape)}   h set '{a.set}' {tuple(H.shape)}   dev {dev}\n")

    res = run(W, H, a.dims, a.tiles, dev)
    (HERE / "data" / f"bound_vs_dim_{a.set}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
