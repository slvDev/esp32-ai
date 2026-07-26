"""Single-command deploy for the ESP32 PLE demo.

This wraps the existing flow:
  1. prepare data/tokenizer if needed
  2. train the deploy checkpoint if needed
  3. export the model binary and golden reference
  4. generate firmware vocab assets
  5. compile and run the host verifier

The default configuration matches the deploy run documented in the repo:
vocab 32768, d_model 96, n_layers 6, ple_dim 128, target core 560000.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
FIRMWARE_MODEL = ROOT / "firmware" / "model"
FIRMWARE = ROOT / "firmware"
SRC = ROOT / "src"


def run_step(title: str, argv: list[str]) -> None:
    print(f"\n==> {title}", flush=True)
    print("    " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def data_artifacts_exist(vocab: int) -> bool:
    suffix = "" if vocab == 4096 else f"_v{vocab}"
    return all(
        (DATA / name).exists()
        for name in (
            f"train{suffix}.bin",
            f"val{suffix}.bin",
            f"bpe{vocab}.json",
        )
    )


def checkpoint_path(arm: str, tag: str, seed: int) -> Path:
    return RUNS / f"{arm}{('-' + tag) if tag else ''}-s{seed}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ple", choices=["baseline", "ple", "ple_notable", "fatembed", "bigcore"])
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--ple-dim", type=int, default=128)
    ap.add_argument("--target-core", type=int, default=560000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="cleandeploy")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-assets", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    FIRMWARE_MODEL.mkdir(parents=True, exist_ok=True)

    if not args.skip_data and not data_artifacts_exist(args.vocab):
        run_step(
            "prepare data",
            [
                sys.executable,
                str(ROOT / "data" / "prepare.py"),
                "--vocab",
                str(args.vocab),
            ],
        )
    elif not args.skip_data:
        print("\n==> prepare data: skipped (artifacts already exist)", flush=True)

    ckpt = checkpoint_path(args.arm, args.tag, args.seed)
    if not args.skip_train:
        if ckpt.exists() and not args.force_train:
            print(f"\n==> train: skipped (found {ckpt.relative_to(ROOT)})", flush=True)
        else:
            run_step(
                "train deploy checkpoint",
                [
                    sys.executable,
                    str(SRC / "train.py"),
                    "--arm",
                    args.arm,
                    "--vocab",
                    str(args.vocab),
                    "--d-model",
                    str(args.d_model),
                    "--n-layers",
                    str(args.n_layers),
                    "--ple-dim",
                    str(args.ple_dim),
                    "--target-core",
                    str(args.target_core),
                    "--batch-size",
                    str(args.batch_size),
                    "--seq-len",
                    str(args.seq_len),
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(args.seed),
                    "--tag",
                    args.tag,
                ],
            )

    if not ckpt.exists():
        raise FileNotFoundError(f"expected checkpoint not found: {ckpt}")

    if not args.skip_export:
        run_step("export model artifact", [sys.executable, str(SRC / "export.py"), str(ckpt)])

    if not args.skip_assets:
        run_step("generate firmware vocab assets", [sys.executable, str(SRC / "gen_assets.py")])

    if not args.skip_verify:
        verify_bin = Path("/tmp/esp32-llm-verify")
        run_step(
            "build host verifier",
            ["cc", "-O3", "-o", str(verify_bin), str(FIRMWARE / "host_verify" / "verify.c"), "-lm"],
        )
        run_step(
            "verify exported model",
            [str(verify_bin), str(FIRMWARE_MODEL / "model.bin"), str(FIRMWARE_MODEL / "golden.txt")],
        )

    print("\nDeploy flow complete.")


if __name__ == "__main__":
    raise SystemExit(main())