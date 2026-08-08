"""Dose-response test of the dead-weight mechanism, in both directions.

The claim from EXP-126
----------------------
CertiHead's pruning comes from rows that cannot win. Headroom is the existence
of tiles whose best row falls far short of the global best, and a tile is only
far short when it is full of non-competitive rows.

Why this cannot be tested by correlation
----------------------------------------
"Headroom comes from rows that score badly" is close to a restatement of
headroom's definition. Correlating the two would prove nothing. Two earlier
observations are also only consistent with the claim, not evidence for it: a
realistic grammar (which deletes non-competitive rows) lowered rho 1.610 ->
0.319, and amputating low-norm rows raised survival 37.79% -> 61.00%. Both are
deletions, both are confounded with changing V and the tiling.

So: intervene, hold V and D and the tiling fixed, and vary only the FRACTION of
rows that can win. If the mechanism is real this must produce a monotone
dose-response, and it must run in both directions.

  DILUTE   take a head that cannot be certified (pythia, Gemma) and make a
           fraction f of its rows non-competitive by scaling them down. Nothing
           is added or removed; V, D and tile boundaries are untouched. The
           mechanism predicts rho RISES with f.

  ENRICH   take our head, which can be certified, and replace its deadest
           fraction f with copies of competitive rows plus small noise. The
           mechanism predicts rho FALLS with f.

A prediction that only worked in the direction the hypothesis was invented from
would be weak. Requiring both is what makes this a test.

Neither construction is a proposal -- both change what the model outputs. They
are mechanism probes: the question is what the certificate does.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def rho_prune(W, H, T=128, chunk=32):
    """Median rho and prunable %, on the norm-sorted layout the device uses."""
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    rs, pr = [], []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        g = sc.amax(dim=1, keepdim=True)
        rs.append(((g - m) / (u - m).clamp(min=1e-12)).flatten())
        pr.append((u < g).float().mean(dim=1))
        del sc, u, m
    return torch.cat(rs).median().item(), 100 * torch.cat(pr).mean().item()


def competitiveness(W, H, chunk=64):
    """Per row, its best score over the trajectory. Low = cannot win anywhere."""
    V = W.shape[0]
    best = torch.full((V,), -1e30, device=W.device)
    for i in range(0, H.shape[0], chunk):
        best = torch.maximum(best, (H[i:i + chunk] @ W.T).amax(dim=0))
    return best


def dilute(W, f, alpha, gen):
    """Make a random fraction f of rows non-competitive by shrinking them.

    Scaling a row down shrinks every inner product it can produce, so it stops
    winning -- and it also shrinks its norm, which is how genuinely dead rows
    look in a real head. V, D and the tiling are unchanged; only which rows are
    contenders changes.
    """
    V = W.shape[0]
    k = int(V * f)
    if k == 0:
        return W
    idx = torch.randperm(V, generator=gen)[:k].to(W.device)
    out = W.clone()
    out[idx] *= alpha
    return out


def enrich(W, H, f, gen):
    """Replace the deadest fraction f with copies of live rows plus noise."""
    V = W.shape[0]
    k = int(V * f)
    if k == 0:
        return W
    comp = competitiveness(W, H)
    dead = torch.argsort(comp)[:k]                  # least competitive
    live = torch.argsort(comp, descending=True)[:max(V // 20, 1)]
    pick = live[torch.randint(0, live.numel(), (k,), generator=gen).to(W.device)]
    out = W.clone()
    noise = torch.randn(k, W.shape[1], generator=gen).to(W.device)
    noise *= 0.02 * W[pick].norm(dim=1, keepdim=True)
    out[dead] = W[pick] + noise
    return out


def load_ts(dev):
    meta = dict(l.split() for l in (DATA / "ts_meta.txt").read_text().split("\n") if l)
    V, D = int(meta["rows"]), int(meta["cols"])
    return (torch.from_numpy(np.fromfile(DATA / "ts_W.f32", dtype=np.float32)
                             .reshape(V, D)).to(dev),
            torch.from_numpy(np.fromfile(DATA / "ts_x.f32", dtype=np.float32)
                             .reshape(-1, D)).to(dev))


def load_gemma(dev):
    with safe_open(GEMMA / "model.safetensors", "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    return W, torch.from_numpy(np.load(DATA / "h_gen.npy")).to(dev)[:256]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    out = []
    fracs = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95)

    from pythia_dsweep import capture as pcap

    print("DILUTE -- make f of the rows non-competitive (alpha=0.25). "
          "V, D, tiling all fixed.")
    print("mechanism predicts rho RISES with f\n")
    print(f"{'head':<14} {'f':>6} {'rho p50':>9} {'prune %':>9}")
    for name, loader in (("pythia-14m", lambda d: pcap("pythia-14m")),
                         ("pythia-70m", lambda d: pcap("pythia-70m")),
                         ("Gemma-270M", load_gemma)):
        W, H = loader(dev)
        for f in fracs:
            r, p = rho_prune(dilute(W, f, 0.25, gen), H)
            print(f"{name:<14} {f:6.2f} {r:9.3f} {p:9.2f}")
            out.append(dict(head=name, mode="dilute", f=f, rho=r, prune=p))
        print()
        del W, H
        torch.cuda.empty_cache()

    print("ENRICH -- replace the deadest f with copies of live rows.")
    print("mechanism predicts rho FALLS with f\n")
    print(f"{'head':<14} {'f':>6} {'rho p50':>9} {'prune %':>9}")
    W, H = load_ts(dev)
    for f in fracs:
        r, p = rho_prune(enrich(W, H, f, gen), H)
        print(f"{'TinyStories':<14} {f:6.2f} {r:9.3f} {p:9.2f}")
        out.append(dict(head="TinyStories", mode="enrich", f=f, rho=r, prune=p))

    (DATA / "dose_response.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
