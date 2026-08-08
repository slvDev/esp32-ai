"""Is our head certifiable because it is INT4?

Where the evidence stands
-------------------------
Scale-free certifiability, rho = (gmax - m_j)/(u_j - m_j), prunable iff rho > 1:

    TinyStories    D= 96  V= 25353   rho 1.610   prunes 62%
    pythia-14m     D=128  V= 50304   rho 0.169   prunes  0.25%
    pythia-31m     D=256  V= 50304   rho 0.112   prunes  0.51%
    pythia-70m     D=512  V= 50304   rho 0.016   prunes  0.51%
    pythia-160m    D=768  V= 50304   rho 0.017   prunes  0.76%
    Gemma-270M     D=640  V=262144   rho 0.081   prunes  0.00%

Six heads. One prunes, and it is ours. Within the Pythia family rho does fall
with D (0.169 -> 0.016 across 128 -> 512), so D is a real factor -- but
pythia-14m at D=128 is the closest model to our D=96 and is still 10x below the
threshold. D does not carry us from 0.169 to 1.610.

The untested structural difference
----------------------------------
Every other head here is native fp16/fp32. Ours is INT4 with one scale per row:
each row's 96 values are drawn from a 16-symbol alphabet scaled by that row's own
scale. That is not a small perturbation of a continuous matrix; it is a severe
constraint on how much rows inside a tile can differ per dimension, and lo/hi is
exactly a per-dimension spread measurement.

If int4 is what makes the certificate bite, the conclusion inverts: CertiHead is
not a quirk of one small model, it is a QUANTIZATION-AWARE method, and it should
transfer to any int4 head -- which is the deployment regime for every edge target
this project cares about.

The test quantizes each Pythia head with our exact scheme (per-row scale,
code = round(w/scale)+8 clamped to [0,15], value = (code-8)*scale), dequantizes,
and re-measures rho on the SAME hidden states. Only the head changes. This does
perturb the model's outputs, so it is a mechanism probe and not a proposal --
the question is what the certificate does, not what the model says.
"""
from pathlib import Path

import torch

from certifiability import rho, load_ts
from pythia_dsweep import capture as pythia_capture


def quantize_int4_perrow(W):
    """Our staging format: one fp16 scale per row, 4-bit codes offset by 8."""
    scale = W.abs().amax(dim=1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-12).half().float()          # fp16 scales, as stored
    code = torch.round(W / scale) + 8
    code = code.clamp(0, 15)
    return (code - 8) * scale


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("rho = (gmax - m_j)/(u_j - m_j), prunable iff rho > 1\n")
    print(f"{'model':<14} {'D':>5} {'rho fp32':>9} {'prune fp32':>11} "
          f"{'rho int4':>9} {'prune int4':>11}   effect")

    # Our own head is already int4-derived; dequantizing to fp32 and requantizing
    # is a no-op, so it appears once as the reference point.
    W, H = load_ts(dev)
    r, p, _ = rho(W, H)
    print(f"{'TinyStories':<14} {96:5d} {r.median():9.3f} {100*p.mean():10.2f}% "
          f"{'(already int4)':>21}")
    del W, H
    torch.cuda.empty_cache()

    for name in ("pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m"):
        W, H = pythia_capture(name)
        r0, p0, _ = rho(W, H)
        Wq = quantize_int4_perrow(W)
        r1, p1, _ = rho(Wq, H)
        m0, m1 = r0.median().item(), r1.median().item()
        eff = f"{m1/max(m0,1e-9):.2f}x"
        print(f"{name:<14} {W.shape[1]:5d} {m0:9.3f} {100*p0.mean():10.2f}% "
              f"{m1:9.3f} {100*p1.mean():10.2f}%   {eff}")
        del W, Wq, H
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
