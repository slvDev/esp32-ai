"""CAMPAIGN 1 -- an adaptive EXACT output executor, as a policy in code.

CertiHead is one executor, not the architecture. Four exact mechanisms are
implemented here behind one interface, each with a cost model calibrated to this
board's measured PSRAM behaviour, plus a selector that picks between them from
observable state.

Executors (all EXACT -- every one returns the dense argmax):
    DENSE       read every row.
    CERT        certified scan: bound every tile from an SRAM index, visit in
                bound order, stop when the incumbent dominates.
    LEGAL       read only the legal rows, wherever they lie.
    LEGAL_GRP   read only the legal rows, which the layout has made contiguous.

Cost model
----------
Time is modelled as bytes/effective-bandwidth, with efficiency taken from this
board's own chunk-size curve (EXP-114: 64 B loses 48% of peak, 256 B 19%,
1024 B 6%, 4096 B 1.4%, 6656 B 0.8%, 16384 B 0.4%). That curve is why row count
is the wrong objective: 64 scattered 48 B rows cost far more than 3 KiB
contiguous.

Index traffic is charged. EXP-113 established that not charging it inverts the
ranking of tile sizes, and this project has now made the same class of mistake
three times, so the index is a first-class term rather than a footnote.

Selector
--------
Thresholds are DERIVED, not chosen: the selector computes each executor's
predicted time and takes the minimum. Calibration data are the measured
efficiency curve and the head's own rho, both obtainable offline. The interesting
output is not the policy code -- it is the resulting regime map, which says which
executor owns which (K, rho, D) region and is checked against measured pruning.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")

# Board-measured PSRAM efficiency (EXP-114). Fraction of peak LOST vs chunk size.
BW_CHUNK = np.array([64., 256., 1024., 4096., 6656., 16384.])
BW_LOSS = np.array([0.48, 0.19, 0.06, 0.014, 0.008, 0.004])
PEAK_MIBS = 85.3          # board-measured peak, user capture
SRAM_BUDGET = 40 * 1024   # what the certificate index may occupy


def eff(nbytes):
    x = np.clip(np.asarray(nbytes, float), BW_CHUNK[0], BW_CHUNK[-1])
    return 1.0 - np.interp(np.log(x), np.log(BW_CHUNK), BW_LOSS)


def us(nbytes, chunk):
    """Microseconds to move nbytes in runs of `chunk` bytes."""
    return float(nbytes) / (PEAK_MIBS * 1.048576 * eff(chunk))


class Executor:
    """Exact output executors and their predicted cost, in microseconds."""

    @staticmethod
    def dense(V, D, **_):
        rb = D // 2
        return us(V * rb, V * rb), V, 0

    @staticmethod
    def cert(V, D, prune=0.0, tile=128, index_resident=True, **_):
        rb = D // 2
        rows = V * (1.0 - prune)
        # rows come in runs of `tile`; index is 2 int8 vectors per tile
        idx = (V // tile) * D * 2
        t = us(rows * rb, tile * rb)
        if not index_resident:
            t += us(idx, D * 2)          # index itself streamed from PSRAM
        # bound arithmetic: D MACs per tile, ~1 MAC/cycle/core at 240 MHz x2
        t += (V // tile) * D / (240.0 * 2)
        return t, rows, idx

    @staticmethod
    def legal(V, D, K=0, **_):
        rb = D // 2
        return us(K * rb, rb), K, 0      # scattered: each row is its own run

    @staticmethod
    def legal_grp(V, D, K=0, **_):
        rb = D // 2
        return us(K * rb, K * rb), K, 0  # one contiguous run


def select(V, D, K, prune, tile=128):
    """Pick the cheapest EXACT executor. Returns (name, us, rows, index bytes)."""
    cands = {"DENSE": Executor.dense(V, D)}
    idx = (V // tile) * D * 2
    if V // tile >= 2:
        cands["CERT"] = Executor.cert(V, D, prune=prune, tile=tile,
                                      index_resident=idx <= SRAM_BUDGET)
    if K and K < V:
        cands["LEGAL"] = Executor.legal(V, D, K=K)
        cands["LEGAL_GRP"] = Executor.legal_grp(V, D, K=K)
    best = min(cands, key=lambda k: cands[k][0])
    return best, cands


def measured_prune(W, H, tile=128, chunk=32):
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // tile
    if nt < 2:
        return 0.0
    Wt = Wp[: nt * tile].view(nt, tile, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * tile].T.contiguous()
    pr = []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        pr.append((u < sc.amax(dim=1, keepdim=True)).float().mean(dim=1))
        del sc, u
    return torch.cat(pr).mean().item()


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
    from pythia_dsweep import capture as pcap
    heads = [("TinyStories", load_ts), ("Gemma-270M", load_gemma)]
    for n in ("pythia-14m", "pythia-70m"):
        heads.append((n, lambda d, _n=n: pcap(_n)))
    out = []

    # Sanity: the model must reproduce the board before it is used to decide.
    p_ts = None
    print("--- cost-model calibration against the board ---")
    W, H = load_ts(dev)
    p_ts = measured_prune(W, H)
    t_cert, rows, idx = Executor.cert(W.shape[0], W.shape[1], prune=p_ts)
    t_dense, _, _ = Executor.dense(W.shape[0], W.shape[1])
    print(f"TinyStories measured prune {100*p_ts:.2f}%  "
          f"predicted head {t_cert/1000:.2f} ms vs dense {t_dense/1000:.2f} ms")
    print(f"board measured: head 6.4-6.5 ms certified, 8427/25353 rows "
          f"(33.2% scanned)\n")
    del W, H
    torch.cuda.empty_cache()

    print("--- regime map: which exact executor wins, and by how much ---")
    print(f"{'head':<13} {'D':>4} {'prune%':>7} {'K':>7} {'winner':>10} "
          f"{'us':>9} {'vs dense':>9} {'runner-up':>10}")
    for name, loader in heads:
        W, H = loader(dev)
        V, D = W.shape
        pr = measured_prune(W, H)
        for K in (0, 8, 32, 128, 512, 2048, 8192):
            best, cands = select(V, D, K, pr)
            t = cands[best][0]
            rest = sorted((v[0], k) for k, v in cands.items() if k != best)
            print(f"{name:<13} {D:4d} {100*pr:7.2f} {K if K else V:7d} "
                  f"{best:>10} {t:9.1f} {cands['DENSE'][0]/t:8.1f}x "
                  f"{rest[0][1]:>10}")
            out.append(dict(head=name, D=D, V=V, prune=pr, K=K or V,
                            winner=best, us=t, speedup=cands["DENSE"][0] / t))
        print()
        del W, H
        torch.cuda.empty_cache()

    (DATA / "executor.json").write_text(json.dumps(out, indent=2))

    print("--- the decision boundary the policy actually implements ---")
    for name in ("TinyStories", "Gemma-270M"):
        r = [x for x in out if x["head"] == name]
        V, D, pr = r[0]["V"], r[0]["D"], r[0]["prune"]
        lo, hi = 1, V
        while lo < hi:                      # largest K where a legal read wins
            mid = (lo + hi + 1) // 2
            if select(V, D, mid, pr)[0].startswith("LEGAL"):
                lo = mid
            else:
                hi = mid - 1
        print(f"{name:<13} legal-row read wins for K <= {lo} "
              f"({100*lo/V:.2f}% of vocabulary); above that "
              f"{'CERT' if pr > 0.05 else 'DENSE'}")


if __name__ == "__main__":
    main()
