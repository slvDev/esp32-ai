"""Build and flash the ESP32-S3 sketch and model partition.

This script looks for the board on USB serial, preferring stable
`/dev/serial/by-id/*` entries and falling back to `/dev/ttyUSB*` and
`/dev/ttyACM*`.

It can run the full pipeline in one command:
    1. optional deploy flow (prepare/train/export/assets/verify)
    2. compile sketch
    3. upload sketch
    4. flash model partition

By default it uses a repo-local standalone Arduino CLI at
`.tools/arduino-cli/arduino-cli` when present, and falls back to `arduino-cli`
from PATH.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIRMWARE = ROOT / "firmware" / "esp32_llm"
FIRMWARE_COMMON = ROOT / "firmware" / "common"
MODEL_BIN = ROOT / "firmware" / "model" / "model.bin"
BUILD_PATH = Path("/tmp/esp32-llm-build")
LOCAL_ARDUINO_CLI = ROOT / ".tools" / "arduino-cli" / "arduino-cli"
FQBN = (
    "esp32:esp32:esp32s3:"
    "UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,"
    "CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,"
    "PSRAM=opi,DebugLevel=info"
)


def run_step(title: str, argv: list[str]) -> None:
    print(f"\n==> {title}", flush=True)
    print("    " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def resolve_arduino_cli(explicit: str | None) -> str:
    if explicit:
        return explicit
    if LOCAL_ARDUINO_CLI.exists():
        return str(LOCAL_ARDUINO_CLI)
    path = shutil.which("arduino-cli")
    if path:
        return path
    raise FileNotFoundError(
        "arduino-cli not found. Install it or pass --arduino-cli /path/to/arduino-cli"
    )


def resolve_esptool() -> str:
    for name in ("esptool.py", "esptool"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "esptool not found in PATH. Install it (for example: `uv add esptool`) "
        "or provide it on PATH as `esptool.py` or `esptool`."
    )


def find_usb_ports() -> list[str]:
    ports: list[str] = []

    for path in sorted(glob.glob("/dev/serial/by-id/*")):
        if os.path.exists(path):
            ports.append(os.path.realpath(path))

    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        for path in sorted(glob.glob(pattern)):
            if path not in ports:
                ports.append(path)

    return ports


def choose_port(explicit: str | None) -> str:
    if explicit:
        return explicit

    ports = find_usb_ports()
    if not ports:
        raise FileNotFoundError(
            "no USB serial port found; pass --port /dev/ttyUSB0 or check the board connection"
        )
    if len(ports) == 1:
        print(f"selected USB port: {ports[0]}")
        return ports[0]

    print("found multiple USB serial ports:")
    for port in ports:
        print(f"  {port}")
    print(f"using {ports[0]}")
    return ports[0]


def run_deploy(python_exe: str, deploy_args: list[str], force_train: bool) -> None:
    argv = [python_exe, str(ROOT / "deploy.py")]
    argv.extend(deploy_args)
    if force_train and "--force-train" not in deploy_args:
        argv.append("--force-train")
    run_step("full deploy flow", argv)


def compile_sketch(arduino_cli: str, build_path: Path) -> None:
    run_step(
        "compile sketch",
        [
            arduino_cli,
            "compile",
            "--fqbn",
            FQBN,
            "--build-property",
            "compiler.optimization_flags=-O3",
            "--build-property",
            f"compiler.cpp.extra_flags=-I{FIRMWARE_COMMON}",
            "--build-path",
            str(build_path),
            str(FIRMWARE),
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full-pipeline",
        action="store_true",
        help="run deploy.py first (prepare/train/export/assets/verify), then compile+flash",
    )
    ap.add_argument(
        "--deploy-arg",
        action="append",
        default=[],
        help="extra argument forwarded to deploy.py (repeatable)",
    )
    ap.add_argument(
        "--force-train",
        action="store_true",
        help="when using --full-pipeline, force a new training run",
    )
    ap.add_argument("--port", help="USB serial port; auto-detected if omitted")
    ap.add_argument("--build-path", default=str(BUILD_PATH))
    ap.add_argument("--arduino-cli", help="path to arduino-cli binary")
    ap.add_argument("--python-exe", default=sys.executable, help="python executable for deploy.py")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--skip-compile", action="store_true", help="skip arduino-cli compile")
    ap.add_argument("--skip-sketch", action="store_true", help="skip arduino-cli upload")
    ap.add_argument("--skip-model", action="store_true", help="skip model partition flash")
    ap.add_argument("--monitor", action="store_true", help="start arduino-cli monitor after flashing")
    args = ap.parse_args()

    build_path = Path(args.build_path)
    arduino_cli = resolve_arduino_cli(args.arduino_cli)
    esptool = resolve_esptool()

    if args.full_pipeline:
        run_deploy(args.python_exe, args.deploy_arg, args.force_train)

    if not args.skip_sketch and not args.skip_compile:
        compile_sketch(arduino_cli, build_path)

    if not args.skip_sketch:
        if not build_path.exists():
            raise FileNotFoundError(
                f"build path not found: {build_path}. Run compile first or remove --skip-compile."
            )

    port: str | None = None
    if not args.skip_sketch or not args.skip_model or args.monitor:
        port = choose_port(args.port)

    if not args.skip_sketch:
        run_step(
            "upload sketch",
            [
                arduino_cli,
                "upload",
                "-p",
                port,
                "--fqbn",
                FQBN,
                "--input-dir",
                str(build_path),
                str(FIRMWARE),
            ],
        )

    if not args.skip_model:
        if not MODEL_BIN.exists():
            raise FileNotFoundError(
                f"missing model artifact: {MODEL_BIN}. Run `uv run python deploy.py` or `uv run python src/export.py` first."
            )
        run_step(
            "flash model partition",
            [
                esptool,
                "--chip",
                "esp32s3",
                "--port",
                port,
                "--baud",
                str(args.baud),
                "write_flash",
                "0x110000",
                str(MODEL_BIN),
            ],
        )

    if args.monitor:
        run_step(
            "serial monitor",
            [arduino_cli, "monitor", "-p", port, "--config", "baudrate=115200"],
        )

    print("\nFlash flow complete.")


if __name__ == "__main__":
    raise SystemExit(main())