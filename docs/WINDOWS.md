# Running esp32-ai on Windows (WSL2 + usbipd)

`scripts/deploy.sh` assumes a Unix host with the board attached directly — `PORT`
defaults to `/dev/cu.usbmodem*`. On Windows the board is invisible to the Linux
toolchain until it is forwarded into WSL2, and the generated text turns out to
arrive on a *different* USB-C connector than the one used for flashing.

This guide covers both. Tested on Windows 11 + WSL2 (Ubuntu), an
ESP32-S3 DevKitC-1 N16R8, and the `tinystories` model.

---

## First: the board has two USB-C ports and they are not interchangeable

This is the single biggest time sink, so it goes first.

| Silkscreen | Chip | USB ID | Use it for |
| ---------- | ---- | ------ | ---------- |
| `UART` | CH343 bridge | `1a86:55d3` | **Flashing** |
| `USB`  | native USB-Serial-JTAG | `303a:1001` | **Watching the output** |

The firmware is built with `CDCOnBoot=cdc` and `USBMode=hwcdc` (see `FQBN` in
`scripts/deploy.sh`), so `Serial` binds to the SoC's USB-Serial-JTAG peripheral.
The CH343 bridge sits on UART0, which `Serial` no longer uses.

The trap is that the wrong port is not silent. The ROM bootloader always logs to
UART0, so the `UART` connector prints:

```
ESP-ROM:esp32s3-20210327
rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)
SPIWP:0xee
```

Text appears, the port looks correct, and the generated story never arrives.

**Flash on `UART`. Watch on `USB`.**

---

## Prerequisites

On Windows:

- WSL2 with a Linux distribution
- [usbipd-win](https://github.com/dorssel/usbipd-win) — forwards USB devices into WSL2

Inside WSL2, the usual project requirements: `git`, `build-essential`, `uv`,
`hf`, and `arduino-cli` with the `esp32:esp32` core.

---

## No OLED wired up? Set `USE_DISPLAY 0` first

The firmware defaults to driving an SSD1306/SH110X panel. With no panel on the
bus, `display_begin()` blocks and the board never reaches generation — it looks
like a broken flash rather than a missing peripheral.

In `firmware/esp32_tinystories/esp32_tinystories.ino`:

```c
#define USE_DISPLAY 0
```

Serial-only output works fine without any panel.

---

## Step 1 — Flash (through WSL2)

Connect the cable to the **`UART`** port.

In an **administrator** PowerShell:

```powershell
usbipd list                          # note the BUSID of 1a86:55d3
usbipd bind   --busid <BUSID>        # once per device; persists
usbipd attach --wsl --busid <BUSID>
```

Then in WSL2:

```bash
cd ~/esp32-ai
ls /dev/ttyACM*                      # confirm the board appeared
PORT=/dev/ttyACM0 scripts/deploy.sh tinystories
```

`deploy.sh` prints the fingerprint of what it wrote:

```
expect  : fp=a9bdd778  bytes=14912348
```

Keep it. The board prints the same pair at boot and the two must agree.

> `usbipd attach` does not survive unplugging the board or rebooting Windows, and
> the BUSID can change between connections. Re-run `usbipd list` before each
> attach.

---

## Step 2 — Watch the output (from Windows, not from WSL2)

Move the cable to the **`USB`** port.

It is tempting to reuse the WSL2 path and run `arduino-cli monitor`. Don't — you
will miss the beginning of every run, and the beginning is not recoverable.

**Why.** Pressing `RST` drops the USB device. Rebuilding
`board -> Windows -> usbipd -> vhci_hcd -> /dev/ttyACM0` took a **measured ~6 s**
in our setup:

```
=== /dev/ttyACM0 disconnected (10:45:38), waiting ===
cat: /dev/ttyACM0: No such device          <- repeats for ~6 seconds
=== reconnected to /dev/ttyACM0 (10:45:44) ===
 to touch it, but she was too small.       <- already mid-story
```

The board starts generating in under 5 s, so the monitor always reconnects late.
And the output is gone for good: `emit()` in the sketch checks
`Serial.availableForWrite()` and **drops the token** when no host is draining the
CDC buffer, rather than stalling generation. Nothing is buffered for a late
reader.

Reading the COM port directly from Windows re-enumerates in 1–2 s, which is fast
enough to catch the boot banner.

```powershell
# release the board back to Windows first
usbipd detach --busid <BUSID>
```

Windows then exposes it as a COM port (`Device Manager` -> `Ports`, or
`[System.IO.Ports.SerialPort]::GetPortNames()`), and any terminal that reopens
the port on disconnect will do. `tools/windows/` in this repository has a
PowerShell script that detaches, finds the port by VID/PID, and reconnects
automatically across resets.

A correct capture starts at the ROM banner and runs to the profile line:

```
=== ESP32-S3 PLE TinyLM ===
model: Vin=32768 Vout=25353 D=96 L=6 H=4 F=66 P=128  (mapped 15.6 MB)
build: bytes=14912348 fp=a9bdd778 sram=29320B psram=4.19MB
>>> Once upon a time, there was a little girl named Lily...
--- 200 tokens in 19.35 s ---
throughput: 10.34 tok/s   (94.7 ms/token)
```

Generation runs **once per boot** (`loop()` only delays), so each new story needs
an `RST`.

---

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| ROM banner appears, story never does | Cable on `UART` | Move it to `USB` |
| Monitor connected, nothing at all | Board already finished | Press `RST` |
| Story always starts mid-sentence | Watching through WSL2 | Read the COM port from Windows |
| Boots, prints the header, then hangs | `USE_DISPLAY 1`, no panel | Set it to `0` |
| No COM port after `detach` | usbipd service stopped | `Start-Service usbipd`, then detach again |
| `deploy.sh`: no port found | Not attached to WSL2 | Re-run `usbipd list` and `attach` |

One trap worth naming: stopping every process named `usbipd` also stops the
Windows **service**, and a subsequent `usbipd detach` then fails silently — the
device stays attached to WSL2 with no COM port on the Windows side. Start the
service before detaching.

---

## A measured note on the OLED

`RESULTS.md` publishes 9.88 tok/s end to end with the panel attached. Running the
same firmware with `USE_DISPLAY 0` and no panel, we measured:

| | Published (with OLED) | Measured (no OLED) |
| --- | ---: | ---: |
| Compute | 94.9 ms/token | 94.7 ms/token |
| End to end | 9.88 tok/s | 10.34 tok/s |

Compute time is unchanged — the 0.2% gap is measurement noise, and the model runs
at exactly the same speed. The end-to-end difference of ~0.46 tok/s (~4.5%) is
therefore the cost of refreshing the display, not an optimization.

*Caveat: this is a single clean end-to-end capture. The per-token compute figure
was stable across runs (19.34 s and 19.35 s for 200 tokens); the end-to-end
number would benefit from more repetitions.*

---

## Credits

The model, the architecture, and this repository are the work of
[slvDev](https://github.com/slvDev) (Viacheslav Sierbov), MIT licensed. Per-Layer
Embeddings are Google's design from Gemma 3n; TinyStories is Eldan & Li,
Microsoft Research.

This guide contributes only the Windows path: the two-connector finding, the
usbipd latency measurement, and the direct-COM workaround. It was written by
[MahomerLeon](https://github.com/MahomerLeon) from an actual first-run
deployment, debugging included.

### Written with Claude

Debugged and written with [Claude](https://claude.ai) (Anthropic). Every command
was run on real hardware; every number here was measured, not estimated.
