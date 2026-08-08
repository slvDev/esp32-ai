"""Headroom is the scarce resource. What produces it?

EXP-123 concluded that inflation scales with D and that CertiHead is a low-D
method. The Pythia sweep refutes that:

    model         D     inflation  headroom   survive
    pythia-14m    128     3.498      0.605     99.75%
    pythia-31m    256     3.178      0.364     99.49%
    pythia-70m    512     0.561      0.009     99.49%
    pythia-160m   768     0.677      0.011     99.24%
    TinyStories    96     0.740      1.150     37.79%
    Gemma-270M    640     8.228      0.697    100.00%

Inflation does not rise with D; across Pythia it FALLS. And pythia-70m has a
TIGHTER bound than our own head (0.561 vs 0.740) while pruning nothing at all,
because its headroom is 0.009 against our 1.150 -- two orders of magnitude.

So the binding constraint is headroom, and D is not what sets it.

The hypothesis this file tests
------------------------------
Headroom is (gmax - m_j)/|gmax|: how far the typical tile's best row falls short
of the winner. That is large when one row wins by a wide margin and small when
many rows are nearly as good -- which is a statement about the shape of the
output distribution at that position, not about the weight matrix.

If so, headroom is a property of MODEL CONFIDENCE. TinyStories is a tiny model on
children's stories with a 25k vocabulary and is extremely peaked; Pythia and
Gemma are general models over 50k-262k vocabularies and are not.

That would mean certified pruning works exactly when the next token is nearly
determined -- and fails when it is genuinely uncertain. Measured per position
rather than per model, so the claim is testable within a single trajectory
instead of across six.

If it holds, the consequence is not small. It says the certificate is a function
of decoding ENTROPY, which is something a structured/grammar-constrained decoder
controls directly.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

from pythia_dsweep import capture as pythia_capture

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def per_position(W, H, T=128, chunk=32):
    """Per position: surviving-tile fraction, entropy, top-1 margin, headroom."""
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    out = {k: [] for k in ("surv", "ent", "margin", "headroom", "inflation")}
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        top2 = sc.topk(2, dim=1).values
        g = top2[:, 0]
        lp = torch.log_softmax(sc, dim=1)
        s = g.abs().clamp(min=1e-9)
        out["surv"].append((u >= g[:, None]).float().mean(dim=1))
        out["ent"].append(-(lp.exp() * lp).sum(dim=1))
        out["margin"].append((g - top2[:, 1]) / s)
        out["headroom"].append(((g[:, None] - m) / s[:, None]).median(dim=1).values)
        out["inflation"].append(((u - m) / s[:, None]).median(dim=1).values)
        del sc, u, m, lp
    return {k: torch.cat(v) for k, v in out.items()}


def load_gemma(dev):
    with safe_open(GEMMA / "model.safetensors", "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    H = torch.from_numpy(np.load(DATA / "h_gen.npy")).to(dev)[:256]
    return W, H


def load_ts(dev):
    meta = dict(l.split() for l in (DATA / "ts_meta.txt").read_text().split("\n") if l)
    V, D = int(meta["rows"]), int(meta["cols"])
    W = torch.from_numpy(
        np.fromfile(DATA / "ts_W.f32", dtype=np.float32).reshape(V, D)).to(dev)
    H = torch.from_numpy(
        np.fromfile(DATA / "ts_x.f32", dtype=np.float32).reshape(-1, D)).to(dev)
    return W, H


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    print(f"{'model':<14} {'D':>5} {'entropy':>8} {'margin':>8} {'headroom':>9} "
          f"{'survive %':>10}   {'corr(ent, survive)':>19}")
    for name, loader in (("TinyStories", load_ts),
                         ("Gemma-270M", load_gemma)):
        W, H = loader(dev)
        r = per_position(W, H)
        c = np.corrcoef(r["ent"].cpu().numpy(), r["surv"].cpu().numpy())[0, 1]
        print(f"{name:<14} {W.shape[1]:5d} {r['ent'].median():8.3f} "
              f"{r['margin'].median():8.3f} {r['headroom'].median():9.3f} "
              f"{100*r['surv'].mean():10.2f}   {c:19.3f}")
        rows.append(dict(model=name, D=W.shape[1],
                         entropy=r["ent"].median().item(),
                         margin=r["margin"].median().item(),
                         headroom=r["headroom"].median().item(),
                         survive=100 * r["surv"].mean().item(), corr=float(c)))
        del W, H
        torch.cuda.empty_cache()

    for name in ("pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m"):
        W, H = pythia_capture(name)
        r = per_position(W, H)
        c = np.corrcoef(r["ent"].cpu().numpy(), r["surv"].cpu().numpy())[0, 1]
        print(f"{name:<14} {W.shape[1]:5d} {r['ent'].median():8.3f} "
              f"{r['margin'].median():8.3f} {r['headroom'].median():9.3f} "
              f"{100*r['surv'].mean():10.2f}   {c:19.3f}")
        rows.append(dict(model=name, D=W.shape[1],
                         entropy=r["ent"].median().item(),
                         margin=r["margin"].median().item(),
                         headroom=r["headroom"].median().item(),
                         survive=100 * r["surv"].mean().item(), corr=float(c)))
        del W, H
        torch.cuda.empty_cache()

    # The within-model test is the one that matters: across models, entropy is
    # confounded with everything else. Within one trajectory, only the position
    # changes, so a relationship there is about decoding state alone.
    print("\n--- within TinyStories: pruning by entropy quartile ---")
    W, H = load_ts(dev)
    r = per_position(W, H)
    q = torch.quantile(r["ent"], torch.tensor([.25, .5, .75], device=dev))
    lab = ("Q1 most confident", "Q2", "Q3", "Q4 least confident")
    edges = [-1e9] + q.tolist() + [1e9]
    for k in range(4):
        sel = (r["ent"] > edges[k]) & (r["ent"] <= edges[k + 1])
        print(f"{lab[k]:<20} n={int(sel.sum()):4d}  entropy {r['ent'][sel].median():6.3f}  "
              f"headroom {r['headroom'][sel].median():6.3f}  "
              f"survive {100*r['surv'][sel].mean():6.2f}%")

    (DATA / "headroom_vs_entropy.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
