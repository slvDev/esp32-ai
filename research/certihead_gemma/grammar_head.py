"""Grammar-conditioned certifiability, and what it costs to FETCH.

Two things this does NOT assume
-------------------------------
1. That grammar restriction raises headroom. It need not. Headroom is
   (g - m_j); restricting the legal set can lower m_j (competitors removed) but
   ALSO lowers g whenever the unconstrained winner is itself illegal. The sign is
   undetermined in general, so winner-legality is measured rather than assumed
   and is reported as its own column.
2. That fewer legal rows means less memory traffic. It does not, by itself. This
   project measured the PSRAM efficiency curve directly (EXP-114): a 64 B read
   loses 48% of peak bandwidth, 256 B loses 19%, 4096 B loses 1.4%. Sixty-four
   legal rows scattered across the vocabulary are sixty-four ~48 B reads, which
   at 48% loss costs nearly as much time as reading twice the bytes contiguously.

The second point is the actual systems question. Logit masking after a dense
projection saves nothing. Restricting which rows are SCORED saves arithmetic but
not necessarily bandwidth. Only restricting which rows are FETCHED, in
contiguous runs, saves both -- and that requires the legal set to be physically
grouped, which is a layout decision, not a decoding decision.

Layouts compared
----------------
  norm      rows ordered by ||w||, as the device stages them today
  grammar   legal rows moved into one contiguous run

Legal-set constructions
-----------------------
  freq   the K most frequently winning tokens over the trajectory. A realistic
         grammar state: it covers what the model actually emits.
  rand   K uniformly random tokens. Adversarial control -- a constraint that
         fights the model.
  win    per position, the true winner plus K-1 random others. Upper bound; the
         winner is legal by construction, so it isolates the effect of removing
         competitors from the effect of removing the answer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")

# Measured on this board, EXP-114: fraction of peak PSRAM bandwidth LOST at a
# given contiguous chunk size. Interpolated in log-size; clamped outside.
BW_CHUNK = np.array([64, 256, 1024, 4096, 6656, 16384], dtype=np.float64)
BW_LOSS = np.array([0.48, 0.19, 0.06, 0.014, 0.008, 0.004])


def efficiency(nbytes):
    """Fraction of peak bandwidth achieved for a contiguous read of nbytes."""
    x = np.clip(np.asarray(nbytes, dtype=np.float64), BW_CHUNK[0], BW_CHUNK[-1])
    return 1.0 - np.interp(np.log(x), np.log(BW_CHUNK), BW_LOSS)


def segments(idx_sorted):
    """Maximal runs of consecutive physical row indices -> (count, lengths)."""
    if idx_sorted.numel() == 0:
        return 0, torch.empty(0)
    d = torch.diff(idx_sorted)
    brk = (d != 1).nonzero().flatten()
    starts = torch.cat([torch.tensor([0], device=idx_sorted.device), brk + 1])
    ends = torch.cat([brk, torch.tensor([idx_sorted.numel() - 1],
                                        device=idx_sorted.device)])
    return starts.numel(), (ends - starts + 1).float()


def effective_bytes(seg_lens, row_bytes):
    """Time-equivalent bytes at peak: sum over runs of bytes / efficiency."""
    b = (seg_lens.cpu().numpy() * row_bytes)
    return float((b / efficiency(b)).sum())


def rho_of(W, H, T=128, chunk=32):
    """Median rho and prunable fraction for a head over given hidden states."""
    V, D = W.shape
    nt = max(V // T, 1)
    if nt < 2:
        return float("nan"), 0.0
    Wt = W[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = W[: nt * T].T.contiguous()
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


def run_model(name, W, H, Ks, dev, out):
    V, D = W.shape
    row_bytes = D // 2                       # int4, our deployment format
    perm = torch.argsort(W.norm(dim=1))      # the layout the device stages
    Wp = W[perm]
    phys_of_token = torch.empty(V, dtype=torch.long, device=dev)
    phys_of_token[perm] = torch.arange(V, device=dev)

    # Unrestricted control, on the layout actually used.
    r0, p0 = rho_of(Wp, H)
    dense_bytes = effective_bytes(torch.tensor([float(V)]), row_bytes)
    print(f"\n=== {name}  V={V} D={D}  int4 row={row_bytes} B ===")
    print(f"unrestricted: rho {r0:.3f}  prune {p0:.2f}%  "
          f"dense fetch {dense_bytes/1024:.0f} KiB/token")
    print(f"{'K':>6} {'set':>5} {'win legal':>10} {'rho':>7} {'prune %':>8} "
          f"{'segs(norm)':>11} {'KiB norm':>9} {'KiB gram':>9} {'vs dense':>9}")

    wins = torch.zeros(V, dtype=torch.long, device=dev)
    for i in range(0, H.shape[0], 64):
        wins += torch.bincount((H[i:i + 64] @ W.T).argmax(dim=1), minlength=V)
    freq_order = torch.argsort(wins, descending=True)
    g = torch.Generator(device="cpu").manual_seed(0)

    for K in Ks:
        if K >= V:
            continue
        for kind in ("freq", "rand", "win"):
            if kind == "freq":
                legal = freq_order[:K]
            elif kind == "rand":
                legal = torch.randperm(V, generator=g)[:K].to(dev)
            else:
                # winner-legal: measured per position would change the legal set
                # each step; use the trajectory's winners plus random padding so
                # one fixed set still contains every winner it can.
                w = (wins > 0).nonzero().flatten()
                pad = torch.randperm(V, generator=g)[:max(K - w.numel(), 0)].to(dev)
                legal = torch.cat([w[:K], pad])[:K]

            legal_mask = torch.zeros(V, dtype=torch.bool, device=dev)
            legal_mask[legal] = True
            # how often the unconstrained winner survives the constraint
            hits = 0
            for i in range(0, H.shape[0], 64):
                hits += legal_mask[(H[i:i + 64] @ W.T).argmax(dim=1)].sum().item()
            win_legal = 100 * hits / H.shape[0]

            # certifiability of the restricted head
            Wl = W[legal]
            rl, pl = rho_of(Wl[torch.argsort(Wl.norm(dim=1))], H)

            # physical fetch cost under the two layouts
            ph = torch.sort(phys_of_token[legal]).values
            ns, lens = segments(ph)
            kib_norm = effective_bytes(lens, row_bytes) / 1024
            kib_gram = effective_bytes(torch.tensor([float(K)]), row_bytes) / 1024

            print(f"{K:6d} {kind:>5} {win_legal:9.1f}% "
                  f"{rl if rl == rl else float('nan'):7.3f} {pl:8.2f} "
                  f"{ns:11d} {kib_norm:9.2f} {kib_gram:9.2f} "
                  f"{dense_bytes/1024/kib_gram:8.1f}x")
            out.append(dict(model=name, V=V, D=D, K=K, kind=kind,
                            win_legal=win_legal, rho=rl, prune=pl,
                            segs=int(ns), kib_norm=kib_norm, kib_gram=kib_gram,
                            kib_dense=dense_bytes / 1024,
                            rho_unrestricted=r0, prune_unrestricted=p0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4, 16, 64, 256, 1024])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = []

    W, H = load_ts(dev)
    run_model("TinyStories", W, H, a.K, dev, out)
    del W, H; torch.cuda.empty_cache()

    from pythia_dsweep import capture as pcap
    for n in ("pythia-14m", "pythia-70m"):
        W, H = pcap(n)
        run_model(n, W, H, a.K, dev, out)
        del W, H; torch.cuda.empty_cache()

    W, H = load_gemma(dev)
    run_model("Gemma-270M", W, H, a.K, dev, out)
    del W, H; torch.cuda.empty_cache()

    (DATA / "grammar_head.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
