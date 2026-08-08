"""What actually decides whether a certified head can prune?

The norm-heterogeneity hypothesis is dead. It was tested in both directions and
failed both:

    TinyStories, rows unit-normalised (CV 0.401 -> 0.000)   38% -> 28%  BETTER
    Gemma, grafted heterogeneous norms (CV 0.029 -> 0.401)  100% -> 99% NO HELP
    Gemma, even wider log-uniform norms (CV 0.661)          100% -> 99% NO HELP

So stop guessing at properties of W and measure the quantity the algorithm
actually consumes.

The decomposition
-----------------
Per tile j: u_j is the bound, m_j the true max inside it, and gmax = max_j m_j.
A tile is skippable exactly when u_j < gmax. Split that condition:

    u_j < gmax   <=>   (u_j - m_j)  <  (gmax - m_j)
                        ^inflation      ^headroom

INFLATION is how much the bound overestimates its own tile -- a property of the
bound and the tile's internal spread.
HEADROOM is how far that tile's best row falls short of the global best -- a
property of how UNEQUAL the tiles are.

Pruning is not "is the bound tight". It is "is the bound tighter than the tiles
are unequal". A perfectly tight bound prunes nothing if all tiles are equally
good, and a loose bound prunes plenty if one tile is far better than the rest.

That reframing predicts the failure: in high dimension, 128 near-orthogonal rows
per tile and 2048 tiles give every tile almost the same best-case alignment with
h, so headroom collapses toward zero and no inflation, however small, fits under
it. Both quantities are measured here in units of |gmax| so the two heads are
comparable despite completely different logit scales.
"""
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def decompose(tag, W, H, T=128, chunk=32):
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()

    infl, head, surv = [], [], []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        g = sc.amax(dim=1, keepdim=True)
        scale = g.abs().clamp(min=1e-9)
        infl.append(((u - m) / scale).flatten())
        head.append(((g - m) / scale).flatten())
        surv.append((u >= g).float().mean(dim=1))
        del sc, u, m
    infl = torch.cat(infl)
    head = torch.cat(head)
    s = 100 * torch.cat(surv).mean().item()
    print(f"{tag:<40} inflation med {infl.median():7.3f}  "
          f"headroom med {head.median():7.3f}  "
          f"ratio {(head.median()/infl.median()):6.3f}   survive {s:6.2f}%")
    return infl.median().item(), head.median().item(), s


def main():
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

    print("all quantities in units of |gmax|; a tile survives iff inflation > headroom\n")
    decompose("TinyStories 25353 x 96", Wts, Hts)
    decompose("Gemma-270M 262144 x 640", Wg, Hg)

    # Isolate V from D. Truncating D is confounded -- it shrinks the true score
    # while leaving the bound's structure intact -- so vary V instead, which is
    # a clean subset operation on rows.
    print("\n--- Gemma, vocabulary subsampled (D held at 640) ---")
    g = torch.Generator(device="cpu").manual_seed(0)
    for v in (25353, 65536, 131072, 262144):
        idx = torch.randperm(Wg.shape[0], generator=g)[:v].to(dev)
        decompose(f"Gemma V={v:6d} D=640", Wg[idx], Hg)

    # And vary D on Gemma with V held down, acknowledging the truncation caveat:
    # this is a projection of the SAME head, so headroom and inflation are still
    # comparable to each other within each row of the table.
    print("\n--- Gemma, D truncated (V held at 25353) ---")
    idx = torch.randperm(Wg.shape[0], generator=g)[:25353].to(dev)
    Wsub = Wg[idx]
    for d in (96, 160, 320, 640):
        decompose(f"Gemma V=25353 D={d:4d}", Wsub[:, :d].contiguous(),
                  Hg[:, :d].contiguous())


if __name__ == "__main__":
    main()
