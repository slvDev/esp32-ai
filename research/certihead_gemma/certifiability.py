"""The scale-free version of the decomposition. EXP-123's metric was not.

The flaw
--------
EXP-123 reported inflation as (u_j - m_j)/|gmax| and headroom as
(gmax - m_j)/|gmax|. Both numerators are differences of inner products and are
meaningful. The DENOMINATOR is not: |gmax| is the magnitude of the largest logit,
which depends on where a given model happens to centre its logits. Comparing
those normalised numbers across six models with different logit scales compares
the normaliser as much as the quantity.

That is what produced the nonsense in the Pythia sweep: pythia-70m came out with
headroom 0.009 and margin 0.001, which reads as "the top two tokens are
indistinguishable" when it actually means its |gmax| is large relative to its
logit gaps.

The fix
-------
The algorithm's decision is  u_j < gmax. Rearranged:

    rho_j = (gmax - m_j) / (u_j - m_j)      tile j is prunable iff rho_j > 1

Both numerator and denominator are differences in the same units, so rho is
invariant to any rescaling of the logits and to the choice of normaliser. It is
also the whole story: the prunable fraction is exactly the fraction of tiles with
rho_j > 1, so median rho is a single honest summary of how certifiable a head is.

This does not change any measured pruning percentage -- those were computed from
the actual comparison, never from the normalised statistics. It changes only the
explanatory numbers, and it is those that were used to argue "inflation scales
with D".
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from pythia_dsweep import capture as pythia_capture

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def rho(W, H, T=128, chunk=32):
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    rs, prun, gaps = [], [], []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        top2 = sc.topk(2, dim=1).values
        g = top2[:, 0:1]
        # denominator is >= 0 by soundness; clamp only guards exact ties
        r = (g - m) / (u - m).clamp(min=1e-12)
        rs.append(r.flatten())
        prun.append((u < g).float().mean(dim=1))
        # top-1 gap in NATS, which is scale-free and interpretable
        gaps.append(torch.log_softmax(sc, dim=1).topk(2, dim=1).values.diff(dim=1).abs().squeeze(1))
        del sc, u, m
    return torch.cat(rs), torch.cat(prun), torch.cat(gaps)


def load_gemma(dev):
    with safe_open(GEMMA / "model.safetensors", "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    return W, torch.from_numpy(np.load(DATA / "h_gen.npy")).to(dev)[:256]


def load_ts(dev):
    meta = dict(l.split() for l in (DATA / "ts_meta.txt").read_text().split("\n") if l)
    V, D = int(meta["rows"]), int(meta["cols"])
    return (torch.from_numpy(np.fromfile(DATA / "ts_W.f32", dtype=np.float32)
                             .reshape(V, D)).to(dev),
            torch.from_numpy(np.fromfile(DATA / "ts_x.f32", dtype=np.float32)
                             .reshape(-1, D)).to(dev))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("rho = (gmax - m_j) / (u_j - m_j);  tile prunable iff rho > 1")
    print("top-1 gap in nats (log p1 - log p2), scale-free\n")
    print(f"{'model':<14} {'D':>5} {'V':>7} {'rho p50':>8} {'rho p90':>8} "
          f"{'prune %':>8} {'gap nats':>9}")
    rows = []
    todo = [("TinyStories", load_ts, 96, 25353), ("Gemma-270M", load_gemma, 640, 262144)]
    for name, loader, D, V in todo:
        W, H = loader(dev)
        r, p, gp = rho(W, H)
        q = torch.quantile(r.float(), torch.tensor([.5, .9], device=dev))
        print(f"{name:<14} {D:5d} {V:7d} {q[0]:8.3f} {q[1]:8.3f} "
              f"{100*p.mean():8.2f} {gp.median():9.3f}")
        rows.append(dict(model=name, D=D, V=V, rho_p50=q[0].item(),
                         rho_p90=q[1].item(), prune=100 * p.mean().item(),
                         gap_nats=gp.median().item()))
        del W, H
        torch.cuda.empty_cache()
    for name in ("pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m"):
        W, H = pythia_capture(name)
        r, p, gp = rho(W, H)
        q = torch.quantile(r.float(), torch.tensor([.5, .9], device=dev))
        print(f"{name:<14} {W.shape[1]:5d} {W.shape[0]:7d} {q[0]:8.3f} {q[1]:8.3f} "
              f"{100*p.mean():8.2f} {gp.median():9.3f}")
        rows.append(dict(model=name, D=W.shape[1], V=W.shape[0],
                         rho_p50=q[0].item(), rho_p90=q[1].item(),
                         prune=100 * p.mean().item(), gap_nats=gp.median().item()))
        del W, H
        torch.cuda.empty_cache()
    (DATA / "certifiability.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
