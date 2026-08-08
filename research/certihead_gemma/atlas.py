"""CAMPAIGN 2 -- Competitive Vocabulary Atlas.

Every competitiveness statistic here is computed from LOGITS ALONE. None of them
reference tiles, bounds, permutations or rho. That is deliberate: EXP-127
established causally that inert vocabulary drives certifiability, but every
measure of "inert" used so far was derived from the certificate itself, so
nothing measured could have failed to agree with it.

An atlas built independently can be asked a harder question: does a statistic
computed with no knowledge of CertiHead PREDICT CertiHead's behaviour on a head
it was never fitted to?

Statistics per row r, over a trajectory of hidden states:
    win        times r was the argmax
    topk       times r was in the top k, for k in 1/4/16/64/256
    neargap    median normalised gap to the winner when r is in the top 256
    everk      was r EVER in the top k
    bridesmaid often near the winner, never the winner

Per head:
    active_k   fraction of V that is ever top-k
    eff_supp   exp(entropy) averaged -- effective candidate count per position
    tail_mass  probability mass outside the top-k
    stability  Jaccard of the active set between the first and second halves of
               the trajectory (a discovery/held-out split, so "inactive" is not
               just "unseen")

Co-activity is also emitted here because CAMPAIGN 3 needs it: rows that are
top-k together should be fetched together.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")
KS = (1, 4, 16, 64, 256)


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


def atlas(W, H, chunk=32, coact_k=64):
    """Certificate-free competitiveness statistics + a top-k co-activity graph."""
    V, D = W.shape
    dev = W.device
    Wt = W.T.contiguous()
    topk_count = {k: torch.zeros(V, dtype=torch.int32, device=dev) for k in KS}
    ent, tail = [], []
    coact_rows = []                       # top-coact_k sets, for CAMPAIGN 3
    half = H.shape[0] // 2
    first_seen = torch.zeros(V, dtype=torch.bool, device=dev)
    second_seen = torch.zeros(V, dtype=torch.bool, device=dev)

    for i in range(0, H.shape[0], chunk):
        sc = H[i:i + chunk] @ Wt
        lp = torch.log_softmax(sc, dim=1)
        p = lp.exp()
        ent.append(-(p * lp).sum(dim=1))
        big = sc.topk(max(KS), dim=1).indices
        for k in KS:
            topk_count[k] += torch.bincount(big[:, :k].flatten(), minlength=V).int()
        tail.append(1.0 - p.gather(1, big[:, :max(KS)]).sum(dim=1))
        ck = big[:, :coact_k]
        coact_rows.append(ck.cpu())
        seen = torch.zeros(V, dtype=torch.bool, device=dev)
        seen[ck.flatten()] = True
        if i < half:
            first_seen |= seen
        else:
            second_seen |= seen
        del sc, lp, p
    ent = torch.cat(ent)
    tail = torch.cat(tail)
    inter = (first_seen & second_seen).sum().item()
    union = (first_seen | second_seen).sum().item()
    return dict(
        V=V, D=D, positions=H.shape[0],
        active=({f"top{k}": (topk_count[k] > 0).float().mean().item() for k in KS}),
        eff_support=ent.exp().mean().item(),
        entropy=ent.median().item(),
        tail_mass=tail.median().item(),
        stability=inter / max(union, 1),
        # the headline: how much of the vocabulary is never even a contender
        inert_top64=1.0 - (topk_count[64] > 0).float().mean().item(),
        inert_top256=1.0 - (topk_count[256] > 0).float().mean().item(),
        # bridesmaids: often near the top, never at it
        bridesmaid=((topk_count[64] > 0) & (topk_count[1] == 0)).float().mean().item(),
    ), torch.cat(coact_rows)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from pythia_dsweep import capture as pcap
    out = {}
    heads = [("TinyStories", load_ts), ("Gemma-270M", load_gemma)]
    for n in ("pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m"):
        heads.append((n, lambda d, _n=n: pcap(_n)))

    print(f"{'head':<14} {'V':>7} {'D':>5} {'inert@64':>9} {'inert@256':>10} "
          f"{'effsupp':>9} {'tailmass':>9} {'stability':>10} {'bridesmd':>9}")
    for name, loader in heads:
        W, H = loader(dev)
        a, coact = atlas(W, H)
        a["head"] = name
        out[name] = a
        np.save(DATA / f"coact_{name}.npy", coact.numpy().astype(np.int32))
        print(f"{name:<14} {a['V']:7d} {a['D']:5d} {a['inert_top64']:9.4f} "
              f"{a['inert_top256']:10.4f} {a['eff_support']:9.1f} "
              f"{a['tail_mass']:9.4f} {a['stability']:10.3f} {a['bridesmaid']:9.4f}")
        del W, H
        torch.cuda.empty_cache()

    (DATA / "atlas.json").write_text(json.dumps(out, indent=2))

    # Held-out prediction: fit nothing, just check whether the certificate-free
    # statistic ORDERS the heads the same way rho does.
    from certifiability import rho as rho_fn
    print("\n--- does a certificate-free statistic predict rho? ---")
    print(f"{'head':<14} {'inert@64':>9} {'rho p50':>9} {'prune %':>9}")
    pts = []
    for name, loader in heads:
        W, H = loader(dev)
        r, p, _ = rho_fn(W[torch.argsort(W.norm(dim=1))], H)
        pts.append((out[name]["inert_top64"], r.median().item(), p.mean().item() * 100))
        print(f"{name:<14} {out[name]['inert_top64']:9.4f} "
              f"{r.median().item():9.3f} {100*p.mean().item():9.2f}")
        del W, H
        torch.cuda.empty_cache()
    a = np.array(pts)
    if a[:, 0].std() > 0:
        print(f"\nSpearman(inert@64, rho)   = "
              f"{np.corrcoef(a[:, 0].argsort().argsort(), a[:, 1].argsort().argsort())[0,1]:.3f}")
        print(f"Spearman(inert@64, prune) = "
              f"{np.corrcoef(a[:, 0].argsort().argsort(), a[:, 2].argsort().argsort())[0,1]:.3f}")


if __name__ == "__main__":
    main()
