"""Is CertiHead powered by dimensionality, or by row-norm heterogeneity?

The observation
---------------
    TinyStories head  25,353 x  96   ||w|| CV = 0.401   prunes to 38%
    Gemma-3-270M head 262,144 x 640  ||w|| CV = 0.029   prunes to 100%

The obvious reading is "D=640 is too big for a box bound". That reading is
wrong, or at least unproven, because Gemma truncated to D=96 also prunes 0% --
at a MEDIAN SLACK OF 5.20x, which is TIGHTER than TinyStories' 6.53x. A bound
that is tighter and prunes less is not a bound that failed from dimensionality.

The other difference is stark: Gemma's rows are all but norm-equalised (mean
0.9955, range 0.80-1.12), while TinyStories' span 0.49-5.32.

Why that would matter
---------------------
CertiHead sorts rows by ||w|| and cuts them into tiles. The pruning comes from
tiles being DIFFERENT: a tile of small-norm rows gets a low bound and dies
against an incumbent set by a tile of large-norm rows. If every row has the same
norm, every tile looks alike, every bound lands in the same place, and nothing
can be eliminated -- not because the bound is loose, but because there is no
heterogeneity for it to exploit. Sorting by norm sorts noise.

The test
--------
Correlation is not enough, so intervene on each head directly:

  A. TinyStories, rows rescaled to UNIT norm. Same directions, same D, same V,
     same everything except the property under test. If pruning collapses toward
     100%, norm heterogeneity was carrying it.

  B. Gemma, rows rescaled to carry TinyStories' norm distribution. If pruning
     appears where there was none, the mechanism is confirmed from the opposite
     direction.

Neither rescaled head is a proposal -- both change the model's output. They are
mechanism probes: the question is what the certificate does, not what the model
says. Direction B in particular is a lower bound on what a real trained head
with heterogeneous norms could achieve, since these norms are grafted on rather
than learned.
"""
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from bound_vs_dim import certified_scan, oracle_incumbent

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def prune_pct(W, H, T, chunk=32):
    """Fraction of rows a certified scan must read. Norm-sorted, box bound."""
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    scan, orc, slack = [], [], []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        tm = sc.view(h.shape[0], nt, T).amax(dim=2)
        gmax = sc.amax(dim=1)
        first, best = certified_scan(u, tm)
        assert torch.equal(best, gmax), "UNSOUND: certified scan lost the true max"
        scan.append(first.float() * T)
        orc.append(oracle_incumbent(u, gmax).float() * T)
        pos = tm > 0
        slack.append((u[pos] / tm[pos]).float())
        del sc, u, tm
    tot = nt * T
    return (100 * torch.cat(scan).mean().item() / tot,
            100 * torch.cat(orc).mean().item() / tot,
            torch.cat(slack).median().item())


def report(tag, W, H, T=128):
    n = W.norm(dim=1)
    cv = (n.std() / n.mean()).item()
    s, o, sl = prune_pct(W, H, T)
    print(f"{tag:<46} CV {cv:5.3f}   scan {s:6.2f}%   oracle {o:6.2f}%   "
          f"slack {sl:6.2f}x")
    return cv, s, o


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    meta = dict(l.split() for l in (DATA / "ts_meta.txt").read_text().split("\n") if l)
    V, D = int(meta["rows"]), int(meta["cols"])
    Wts = torch.from_numpy(
        np.fromfile(DATA / "ts_W.f32", dtype=np.float32).reshape(V, D)).to(dev)
    Hts = torch.from_numpy(
        np.fromfile(DATA / "ts_x.f32", dtype=np.float32).reshape(-1, D)).to(dev)

    with safe_open(GEMMA / "model.safetensors", "pt") as f:
        Wg = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    Hg = torch.from_numpy(np.load(DATA / "h_gen.npy")).to(dev)[:256]

    nts = Wts.norm(dim=1)

    print("=== as measured ===")
    report("TinyStories  25353 x  96   (real)", Wts, Hts)
    report("Gemma-270M  262144 x 640   (real)", Wg, Hg)

    print("\n=== A: remove norm heterogeneity from OUR head ===")
    # Directions untouched; only the norms are flattened.
    report("TinyStories  unit-normalised rows", Wts / nts[:, None], Hts)

    print("\n=== B: graft our norm distribution onto Gemma ===")
    # Sample TinyStories' norms with replacement, assign them to Gemma's rows in
    # a random pairing so no directional structure is smuggled in.
    idx = torch.randint(0, nts.shape[0], (Wg.shape[0],), device=dev)
    target = nts[idx]
    report("Gemma-270M   grafted TinyStories norms",
           Wg / Wg.norm(dim=1, keepdim=True) * target[:, None], Hg)

    print("\n=== C: is it the SPREAD, or the SHAPE? ===")
    # Log-uniform norms over the same 11x range, i.e. heterogeneity with none of
    # TinyStories' particular distributional shape.
    lo, hi = nts.min().item(), nts.max().item()
    synth = torch.exp(torch.empty(Wg.shape[0], device=dev).uniform_(
        float(np.log(lo)), float(np.log(hi))))
    report("Gemma-270M   log-uniform norms (same range)",
           Wg / Wg.norm(dim=1, keepdim=True) * synth[:, None], Hg)


if __name__ == "__main__":
    main()
