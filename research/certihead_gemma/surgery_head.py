"""CAMPAIGN 6 (Part 2A) -- can changing the head beat executing it cleverly?

The regime map from CAMPAIGN 1 says the certified executor only wins on our own
head. On Gemma and Pythia, unrestricted, DENSE wins because the certificate
prunes nothing and its index is pure overhead. Part 1 has therefore run out of
exact levers for those heads.

Part 2A is allowed to change the weights. Three surgeries, each measured on the
quantity that matters (bytes the head must move per token) AND on quality
(cross-entropy against the unmodified head's own distribution, so the comparison
is against the model being replaced rather than against a corpus):

    LOWRANK   W ~ A B with rank r. Head bytes V*D/2 -> (V*r + r*D)/2.
    VQ        rows quantized to a codebook of C centroids per sub-block; a row
              becomes indices instead of values.
    INT4      the deployment baseline, for reference.

Reported per surgery:
    bytes/token, exact-argmax agreement with the original head, KL to the
    original distribution, and whether the surgery makes the head CERTIFIABLE
    (rho > 1) as a side effect -- which would be a Part 2A route to a Part 1
    mechanism.

Agreement is the honest headline for a frozen-source programme: a surgery that
moves the argmax has changed the model, and must be judged as a model change,
not as an inference optimization.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GEMMA = Path(r"C:\Users\marsm\Downloads\tmpz\rnd\models\gemma-3-270m-it")


def load_gemma(dev):
    with safe_open(GEMMA / "model.safetensors", "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().to(dev)
    return W, torch.from_numpy(np.load(DATA / "h_gen.npy")).to(dev)[:256]


def quality(W0, W1, H, chunk=32):
    """Argmax agreement and mean KL(original || surgered), over real states."""
    agree, kl, n = 0, 0.0, 0
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        a = h @ W0.T
        b = h @ W1.T
        agree += (a.argmax(1) == b.argmax(1)).sum().item()
        pa = torch.log_softmax(a, 1)
        pb = torch.log_softmax(b, 1)
        kl += (pa.exp() * (pa - pb)).sum(1).sum().item()
        n += h.shape[0]
        del a, b, pa, pb
    return 100.0 * agree / n, kl / n


def lowrank(W, r):
    # randomized SVD; full SVD on 262144x640 is unnecessary for this question
    g = torch.randn(W.shape[1], r + 32, device=W.device,
                    generator=torch.Generator(device=W.device).manual_seed(0))
    Y = W @ g
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ W
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    return (Q @ U * S) @ Vh, (W.shape[0] * r + r * W.shape[1])


def vq(W, C, sub):
    """Per-sub-block k-means (few Lloyd steps) -> row becomes C-way indices."""
    V, D = W.shape
    nb = D // sub
    out = torch.empty_like(W)
    for b in range(nb):
        X = W[:, b * sub:(b + 1) * sub]
        idx = torch.randperm(V, device=W.device)[:C]
        cen = X[idx].clone()
        for _ in range(6):
            d = torch.cdist(X, cen)
            a = d.argmin(1)
            for c in range(C):
                m = a == c
                if m.any():
                    cen[c] = X[m].mean(0)
        out[:, b * sub:(b + 1) * sub] = cen[d.argmin(1)]
    bits = V * nb * int(np.ceil(np.log2(C)))
    return out, bits // 8 + C * nb * sub * 2


def rho_of(W, H, T=128, chunk=32):
    V, D = W.shape
    Wp = W[torch.argsort(W.norm(dim=1))]
    nt = V // T
    Wt = Wp[: nt * T].view(nt, T, D)
    lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
    Wf = Wp[: nt * T].T.contiguous()
    rs = []
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk]
        u = h.clamp(min=0) @ hi.T + h.clamp(max=0) @ lo.T
        sc = h @ Wf
        m = sc.view(h.shape[0], nt, T).amax(dim=2)
        rs.append(((sc.amax(1, keepdim=True) - m) / (u - m).clamp(min=1e-12)).flatten())
        del sc, u, m
    return torch.cat(rs).median().item()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, H = load_gemma(dev)
    V, D = W.shape
    base_bytes = V * D // 2                      # int4 deployment baseline
    out = []
    print(f"Gemma-3-270M head {V} x {D};  int4 baseline {base_bytes/2**20:.1f} MiB/token")
    print(f"{'surgery':<16} {'MiB/token':>10} {'vs int4':>8} {'agree %':>8} "
          f"{'KL':>8} {'rho':>7}")

    sc = W.abs().amax(1, keepdim=True) / 7.0
    Wi4 = (torch.round(W / sc).clamp(-8, 7)) * sc
    a, k = quality(W, Wi4, H)
    print(f"{'int4 (baseline)':<16} {base_bytes/2**20:10.2f} {1.0:7.2f}x "
          f"{a:8.2f} {k:8.4f} {rho_of(Wi4, H):7.3f}")
    out.append(dict(surgery="int4", mib=base_bytes / 2**20, agree=a, kl=k))

    for r in (64, 128, 256):
        Wr, params = lowrank(W, r)
        by = params // 2
        a, k = quality(W, Wr, H)
        print(f"{'lowrank r=' + str(r):<16} {by/2**20:10.2f} "
              f"{base_bytes/by:7.2f}x {a:8.2f} {k:8.4f} {rho_of(Wr, H):7.3f}")
        out.append(dict(surgery=f"lowrank{r}", mib=by / 2**20, agree=a, kl=k))
        del Wr
        torch.cuda.empty_cache()

    for C, sub in ((256, 8), (256, 16)):
        Wv, by = vq(W, C, sub)
        a, k = quality(W, Wv, H)
        print(f"{'vq C=%d s=%d' % (C, sub):<16} {by/2**20:10.2f} "
              f"{base_bytes/by:7.2f}x {a:8.2f} {k:8.4f} {rho_of(Wv, H):7.3f}")
        out.append(dict(surgery=f"vq{C}_{sub}", mib=by / 2**20, agree=a, kl=k))
        del Wv
        torch.cuda.empty_cache()

    (DATA / "surgery_head.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
