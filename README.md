# Running a 28.9M parameter LLM on a microcontroller

<p align="center">
  <a href="https://x.com/slvDev">𝕏 slvDev</a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/slvdev/">LinkedIn</a>
</p>

> **This fork adds an ESP32-P4 port.** The same model file and the same C runtime
> run 6x faster on an ESP32-P4 by using its vector unit, and the TinyStories
> firmware now takes a prompt typed over serial. The original ESP32-S3 project
> by [slvDev](https://github.com/slvDev/esp32-ai) is documented unchanged below;
> the port is offered upstream in
> [issue #23](https://github.com/slvDev/esp32-ai/issues/23).

## ESP32-P4 results

Board: Waveshare ESP32-P4-WIFI6 (ESP32-P4NRW32: two RISC-V cores at 360 MHz,
32 MB in-package PSRAM, 32 MB flash). Same `model.bin` as the S3, same
partition layout, quality unchanged. Numbers are for a 200-token story with a
typed prompt; the profile is milliseconds per token.

| step                                             | tok/s | ms/token | head | attention |
| ------------------------------------------------ | ----: | -------: | ---: | --------: |
| ESP32-S3 N16R8, the original                     |  10.5 |     93.3 | 58.6 |      17.7 |
| ESP32-P4, code unchanged                         |  17.0 |     57.2 | 38.3 |      10.2 |
| + int8 matvec on the PIE vector unit             |  39.6 |     23.8 | 12.6 |       8.0 |
| + attention on a quantized KV cache, vector dots |  43.0 |     21.8 | 12.6 |       6.0 |
| + output head kept int4, nibbles unpacked in registers |  54 |   17.2 |  8.3 |       5.7 |
| + attention heads split across both cores        |  57.5 |     16.0 |  8.3 |       4.5 |
| + L1 cache preload of the head rows              | **63** | **14.6** |  7.1 |       4.4 |

What the P4 build does differently, all in `runtime/llm_pie_dot.h`,
`runtime/llm_pie.h` and the TinyStories sketch:

- **Vector matvec.** `esp.vmulas.s8.xacc` multiplies 16 int8 pairs per
  instruction into a 40-bit accumulator. Staged rows are padded to 16 bytes
  (`LLM_STAGE_ALIGN`) so every load is a whole vector.
- **Quantized KV cache.** Keys are int8, values int16 stored transposed, so both
  attention passes are the same integer dot products. Host perplexity over
  32,768 validation predictions: 2.1012 nats with the fp32 cache, 2.1014 with
  the quantized one.
- **int4 head.** The output head stays packed in PSRAM and is unpacked with an
  AND and a 4-bit lane shift; the codes are used as 0..15 and `8 * sum(x)` is
  subtracted, which makes the result identical to the int8 kernel. The head is
  bound by PSRAM bandwidth, so half the bytes is the win.
- **Both cores.** The head and the large per-layer matvecs were already split;
  attention heads now are too.
- **Cache preload.** The L1 data cache's preload engine fetches the next 8 KB
  of head rows while the current block computes.

Every vector kernel is checked against the scalar reference at boot and must
match bit for bit; the board refuses to run otherwise.

### Running it on a P4

```bash
scripts/fetch_model.sh tinystories
CHIP=esp32p4 scripts/deploy.sh tinystories
```

If your board's USB connector is a UART bridge rather than the chip's native
USB (the Waveshare board is; it enumerates as "USB Single Serial"), add
`CDC_ON_BOOT=default` so `Serial` goes to UART0. Then open a serial monitor at
115200, wait for `READY>`, type a prompt and press return. Each story is
sampled, so every run differs; ASCII prompts only.

The 360 MHz is this chip revision's ceiling: the clock driver refuses 400 MHz
on revision v1.3 silicon.

Port by Barkın Sarıkartal. The design and the model are slvDev's; see the
credits at the end of this file.


![28.9M-parameter LLM running on an ESP32-S3](media/esp32-ple-demo.gif)

This is a 28.9 million parameter language model that generates text on an ESP32-S3
microcontroller. It runs on the chip itself, with nothing sent to a server, and it
displays generated text at 9.88 tokens per second on a small screen wired to the
chip. It fits because most of the model lives in flash instead of RAM, using
Per-Layer Embeddings, an idea from Google's Gemma 3n.

## The numbers

|              |                                                    |
| ------------ | -------------------------------------------------- |
| Parameters   | 28.9M stored (25M of them in a flash lookup table) |
| Chip         | ESP32-S3, 512KB SRAM, 8MB PSRAM and 16MB flash     |
| Speed        | 9.88 tok/s end to end, 94.9 ms/token of compute    |
| Connectivity | none, everything runs on the device                |
| Model size   | 14.9MB at 4-bit                                    |

## Why it is hard, and how it fits anyway

A microcontroller has very little fast memory. The ESP32-S3 gives you 512KB of
SRAM, and only the values touched many times per token can live there:
activations and norm weights. The dense core and output head, scanned once per
position, sit in PSRAM. What is left is the embedding tables, and their size is
what normally decides how big a model can be.

In this model, most parameters sit in an embedding table, which the model reads
from rather than computes on. So that 25-million-parameter table stays in slow
flash, and only the few rows each token needs are pulled from it, about 450
bytes. Most of the model is therefore never loaded to run it: it sits in flash
and is sampled a little at a time.

That idea is Google's Per-Layer Embeddings, from
[Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n). Here it runs
on the memory layout of a microcontroller instead of a phone or a GPU.

Each tier holds whatever is read at its own frequency:

```
  SRAM  (fast, tiny)   activations and norm weights, touched many times a token
  PSRAM (medium)       the core and output head, read once per position
  FLASH (huge, slow)   the 25M-param table, about 6 rows read per token (~450 B)
```

## What it does, and what it does not

The model was trained on TinyStories, so it writes short, simple stories and mostly
keeps them coherent. It will not answer questions, follow instructions, write code,
or know facts. That limit comes from the small part of the model that does the
reasoning, and the memory trick does not change it. What is interesting here is the
architecture, fitting a large model onto a tiny chip, rather than what a 28.9 million
parameter model can say.

## Models

- [Barista](https://huggingface.co/slvDev/esp32-ai-barista) - espresso question answering
- [TinyStories](https://huggingface.co/slvDev/esp32-ai-tinystories) - story generation

## Running it yourself

Download and deployment are separate operations: one reaches the network, the
other touches the board.

```bash
scripts/fetch_model.sh barista   # download, verify, install into artifacts/
scripts/deploy.sh barista        # generate headers, run gates, compile, flash
```

`tinystories` is the other model, and takes the same two commands. Both require
the model to be named, because the board holds one at a time and deploying
replaces it.

`fetch_model.sh` checks the inference assets against a SHA-256 and byte size
pinned in the script, and cross-checks the release's own `metadata.json` against
those same pins. It installs nothing unless every check passes, so a failed
download leaves what you already have untouched. `deploy.sh` downloads no model:
it works from whatever is already in `artifacts/<model>/`. It does run two of its
header tools through `uv`, which fetches one pinned wheel the first time on a
machine that has never cached it.

The firmware details and the boot output to expect live in
[`firmware/esp32_barista/README.md`](firmware/esp32_barista/README.md) and
[`firmware/esp32_tinystories/README.md`](firmware/esp32_tinystories/README.md). The reusable
architecture is in `src/`; the training, ablation and quantization code that
reproduces the published numbers is in `research/tinystories/`. The full method,
the ablations, and the on-chip measurements are written up in
[`RESULTS.md`](RESULTS.md).

## Credit

TinyStories is the dataset this trains on: short synthetic stories simple enough
that a small model can still learn to write coherently (Ronen Eldan and Yuanzhi Li,
Microsoft Research, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)). The other
half is Per-Layer Embeddings, Google's design from Gemma 3n, which is what
lets a big model fit on a small chip.

Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) is the
reference for training a small language model and running it in plain C.

## Measurements

Detailed measurements and ablations are documented in `RESULTS.md`.
