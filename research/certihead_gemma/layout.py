"""CAMPAIGN 3 -- physical vocabulary layout as an optimization problem.
Also the correction the atlas forced on CAMPAIGN 2's mechanism.

What the atlas showed
---------------------
A certificate-free count of inert rows does NOT predict certifiability. Across
six heads, Spearman(inert-fraction@top64, pruning) = -0.943 -- the WRONG SIGN.
TinyStories has the LEAST inert vocabulary of all six (88.98%) and prunes best;
Gemma has the most (98.33%) and prunes nothing.

EXP-127 is not contradicted, it is under-specified. Dilution there scaled rows
down, which shrank their norms, so the norm sort swept them into whole tiles.
The intervention created dead TILES, not merely dead rows. Gemma's genuinely
inert rows are spread uniformly across all 2048 tiles by a norm sort that cannot
see them (its norm CV is 0.029), so every tile still holds a contender.

Refined claim, and it is layout-dependent rather than model-dependent:

    A certificate prunes to the extent that the layout CONCENTRATES
    non-competitive rows into whole tiles.

That reframes row order from a preprocessing detail into the algorithm's main
free parameter, and it predicts the board result of EXP-124 (removing the norm
permutation cost 33.2% -> 72.7% of rows) rather than merely being consistent
with it.

Layouts compared
----------------
    tokid       identity; the order the file happens to be in
    norm        ||w|| ascending; what the device stages today
    freq        by win count over the trajectory
    random      control
    maxscore    by the row's best score over the trajectory
    everk       inert rows first, contenders last (uses the atlas)
    coact       spectral-ish clustering on the top-k co-activity graph, so rows
                that are contenders TOGETHER are stored together
    coact+norm  co-activity clusters, norm-ordered within each cluster

Reported per layout: the certificate-free dead-tile fraction, then rho and actual
pruning, so the predictor can be checked against the thing it predicts.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


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


def stats(W, H, chunk=64, k=64):
    """Per-row contender flags and scores. Certificate-free."""
    V = W.shape[0]
    dev = W.device
    Wt = W.T.contiguous()
    wins = torch.zeros(V, dtype=torch.int32, device=dev)
    ever = torch.zeros(V, dtype=torch.bool, device=dev)
    best = torch.full((V,), -1e30, device=dev)
    for i in range(0, H.shape[0], chunk):
        sc = H[i:i + chunk] @ Wt
        best = torch.maximum(best, sc.amax(dim=0))
        tk = sc.topk(k, dim=1).indices
        ever[tk.flatten()] = True
        wins += torch.bincount(sc.argmax(dim=1), minlength=V).int()
        del sc
    return wins, ever, best


def coact_perm(name, V, dev, T=128):
    """Cluster rows by top-k co-activity, then lay clusters out contiguously.

    The graph is 'these rows were contenders at the same position'. Rows that
    compete together should live together: then a position's contenders occupy
    few tiles and every other tile is uniformly dead.

    Implemented as repeated bisection on the co-activity adjacency using the
    Fiedler-like direction from a few power iterations, which is cheap enough to
    run on a 262144-row head and needs no external solver.
    """
    p = DATA / f"coact_{name}.npy"
    if not p.exists():
        return None
    ck = torch.from_numpy(np.load(p)).long()          # [positions, k]
    # Row feature = which positions it was a contender at. Rows with identical
    # activation patterns land together; rows never active get a zero vector and
    # collect in one place, which is exactly what we want.
    P = ck.shape[0]
    feat = torch.zeros(V, min(P, 512), device=dev)
    step = max(P // 512, 1)
    for j, pos in enumerate(range(0, P, step)):
        if j >= feat.shape[1]:
            break
        feat[ck[pos].to(dev), j] = 1.0
    # project onto top singular directions -> order by the leading coordinate
    feat = feat - feat.mean(dim=0, keepdim=True)
    g = torch.randn(feat.shape[1], 8, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    for _ in range(8):
        g = feat.T @ (feat @ g)
        g, _ = torch.linalg.qr(g)
    emb = feat @ g                                     # [V, 8]
    # lexicographic on the first two components gives contiguous clusters
    key = emb[:, 0] * 1e6 + emb[:, 1]
    return torch.argsort(key)


def evaluate(W, H, perm, T=128, chunk=32, ever=None):
    V, D = W.shape
    Wp = W[perm]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    # certificate-free predictor: tiles containing no contender at all
    dead_tiles = float("nan")
    if ever is not None:
        e = ever[perm][: nt * T].view(nt, T).any(dim=1)
        dead_tiles = 1.0 - e.float().mean().item()
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
    return dead_tiles, torch.cat(rs).median().item(), 100 * torch.cat(pr).mean().item()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from pythia_dsweep import capture as pcap
    heads = [("TinyStories", load_ts), ("Gemma-270M", load_gemma)]
    for n in ("pythia-14m", "pythia-70m"):
        heads.append((n, lambda d, _n=n: pcap(_n)))
    rows = []
    g = torch.Generator().manual_seed(0)

    for name, loader in heads:
        W, H = loader(dev)
        V = W.shape[0]
        wins, ever, best = stats(W, H)
        layouts = {
            "tokid": torch.arange(V, device=dev),
            "norm": torch.argsort(W.norm(dim=1)),
            "freq": torch.argsort(wins),
            "random": torch.randperm(V, generator=g).to(dev),
            "maxscore": torch.argsort(best),
            "everk": torch.argsort(ever.int()),          # inert first
        }
        cp = coact_perm(name, V, dev)
        if cp is not None:
            layouts["coact"] = cp
            # norm order within each contiguous co-activity block
            blk = 1024
            order = cp.clone()
            for s in range(0, V, blk):
                sl = order[s:s + blk]
                order[s:s + blk] = sl[torch.argsort(W[sl].norm(dim=1))]
            layouts["coact+norm"] = order

        print(f"\n=== {name}  V={V} D={W.shape[1]} ===")
        print(f"{'layout':<12} {'dead tiles':>11} {'rho p50':>9} {'prune %':>9}")
        for ln, perm in layouts.items():
            dt, r, p = evaluate(W, H, perm, ever=ever)
            print(f"{ln:<12} {dt:11.4f} {r:9.3f} {p:9.2f}")
            rows.append(dict(head=name, layout=ln, dead_tiles=dt, rho=r, prune=p))
        del W, H
        torch.cuda.empty_cache()

    (DATA / "layout.json").write_text(json.dumps(rows, indent=2))
    a = np.array([[r["dead_tiles"], r["prune"]] for r in rows
                  if r["dead_tiles"] == r["dead_tiles"]])
    rk = lambda x: x.argsort().argsort()
    print(f"\nSpearman(dead-tile fraction, pruning) over all "
          f"{len(a)} head x layout points = "
          f"{np.corrcoef(rk(a[:,0]), rk(a[:,1]))[0,1]:.3f}")


if __name__ == "__main__":
    main()
