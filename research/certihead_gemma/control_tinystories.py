"""Control: does the torch harness reproduce the C oracle on OUR head?

The Gemma probe reports that the box bound prunes nothing at D=640. That claim
is only worth as much as the code that produced it, and the code is a fresh
reimplementation in a different language and numeric library.

So it gets validated against the one case where the answer is already known.
certihead_dump.exe, on artifacts/tinystories/model.bin with norm-sorted rows,
box bound, tile 128, hot 3, over 1020 positions, reports:

    rows scored mean : 9581.3  (37.79% of head)
    argmax mismatches: 0

This script loads the weights and hidden states that same binary dumped, rebuilds
the permutation, tiling and bounds independently in torch, and must land in the
same place. Two numbers are produced:

    scan   -- no hot set, incumbent starts at -inf. Must be >= 37.79%, because
              the C harness gets a free high incumbent from the previous
              position's winner and this does not.
    oracle -- incumbent seeded with the true argmax. Must be <= 37.79%, because
              no warm start can beat knowing the answer.

If 37.79% falls inside that bracket the harness agrees with the C oracle and the
Gemma numbers stand. If it does not, the Gemma result is a bug in this file.
"""
from pathlib import Path

import numpy as np
import torch

from bound_vs_dim import certified_scan, oracle_incumbent

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
C_ORACLE_PCT = 37.79      # certihead_dump.exe, tile 128 / box / norm-sort / hot 3


def main():
    meta = dict(l.split() for l in (DATA / "ts_meta.txt").read_text().split("\n") if l)
    V, D = int(meta["rows"]), int(meta["cols"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    W = torch.from_numpy(
        np.fromfile(DATA / "ts_W.f32", dtype=np.float32).reshape(V, D)).to(dev)
    H = torch.from_numpy(
        np.fromfile(DATA / "ts_x.f32", dtype=np.float32).reshape(-1, D)).to(dev)
    print(f"TinyStories head {V} x {D}   hidden states {H.shape[0]}   dev {dev}")

    # Rebuild the permutation here rather than importing it: an independent
    # re-derivation is the point of a control.
    perm = torch.argsort(W.norm(dim=1))
    Wp = W[perm]

    print(f"\n{'tile':>5} {'scan %':>8} {'oracle %':>9} {'slack med':>10} "
          f"{'bracket vs C oracle 37.79%':>28}")
    for T in (64, 128, 256, 512):
        nt = V // T
        Wt = Wp[: nt * T].view(nt, T, D)
        lo, hi = Wt.amin(dim=1), Wt.amax(dim=1)
        sc = H @ Wp[: nt * T].T
        tm = sc.view(H.shape[0], nt, T).amax(dim=2)
        gmax = sc.amax(dim=1)
        u = H.clamp(min=0) @ hi.T + H.clamp(max=0) @ lo.T

        first, best = certified_scan(u, tm)
        assert torch.equal(best, gmax), "certified scan missed the true max -- UNSOUND"
        scan_pct = 100 * (first.float() * T).mean().item() / (nt * T)
        orc_pct = 100 * (oracle_incumbent(u, gmax).float() * T).mean().item() / (nt * T)
        pos = tm > 0
        slack = (u[pos] / tm[pos]).median().item()

        # Tolerance, not equality: the C harness prints 2 decimal places, so an
        # exact agreement still fails a strict <= bracket on the third.
        eps = 0.01
        ok = orc_pct <= C_ORACLE_PCT + eps and scan_pct >= C_ORACLE_PCT - eps
        note = "CONSISTENT" if ok else "*** DISAGREES WITH C ORACLE ***"
        if T != 128:
            note = "(tile != 128, bracket not required)"
        print(f"{T:5d} {scan_pct:8.2f} {orc_pct:9.2f} {slack:10.2f}x  {note:>28}")


if __name__ == "__main__":
    main()
