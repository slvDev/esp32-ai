# Running a 28.9M parameter LLM on an $8 microcontroller

<p align="center">
  Open to Work &nbsp;·&nbsp;
  <a href="https://x.com/slvDev">𝕏 slvDev</a> &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/slvdev/">LinkedIn</a>
</p>

![28.9M-parameter LLM running on an ESP32-S3](media/esp32-ple-demo.gif)

This is a 28.9 million parameter language model that generates text on an ESP32-S3,
a microcontroller that costs about $8. It runs on the chip itself, with nothing
sent to a server, and it writes each word to a small screen wired to the chip at
roughly 9 tokens per second. The last language model people ran on a chip like this had 260
thousand parameters, so this one holds about a hundred times more. It fits because
most of the model lives in flash instead of RAM, using an idea from Google's Gemma
models called Per-Layer Embeddings.

## Reproducing It

This repo is source-first: the generated training data, checkpoints, export
artifacts, and firmware assets are not committed. To reproduce the ESP32 story
demo from a fresh checkout, you need:

1. Python 3.12+
2. `uv`
3. A TinyStories download and BPE/data bins from `data/prepare.py`
4. A trained checkpoint in `runs/`
5. Arduino CLI plus an ESP32 Arduino core for the firmware build

The easiest path is the bundled wrapper:

```bash
uv run python deploy.py
```

That command does the full sequence in order: prepare the data, train the deploy
checkpoint, export `firmware/model/model.bin`, generate `firmware/esp32_llm/vocab.h`,
and verify the exported binary on the host.

If you already have the data and checkpoint, you can skip ahead with:

```bash
uv run python deploy.py --skip-data --skip-train
```

For a start-to-finish run that prepares data, trains, exports, compiles, and
flashes in one command, use:

```bash
uv run python flash.py --full-pipeline --force-train
```

If you want to reuse an existing checkpoint (skip retraining), run:

```bash
uv run python flash.py --full-pipeline
```

If the model/checkpoint already exists and you only want build+flash, use:

```bash
uv run python flash.py
```

The device sketch plays the same “Once upon a time” prompt that the firmware
README documents, so the on-chip demo is a small storyteller rather than a chat
assistant.

## The numbers

|              |                                                               |
| ------------ | ------------------------------------------------------------- |
| Parameters   | 28.9M stored (25M of them in a flash lookup table)            |
| Chip         | ESP32-S3, about $8, with 512KB SRAM, 8MB PSRAM and 16MB flash |
| Speed        | about 9.5 tok/s end to end (9.7 tok/s of pure compute)        |
| Connectivity | none, everything runs on the device                           |
| Model size   | 14.9MB at 4-bit                                               |

## Why it is hard, and how it fits anyway

A microcontroller has very little fast memory. The ESP32-S3 gives you 512KB of SRAM.
Normally the whole model has to be reachable from there, which keeps you stuck with
tiny models, and that is why the previous model on a chip like this had only 260
thousand parameters.

The way around it is to stop putting the model in fast memory at all. Most of a
language model's parameters sit in an embedding table, which the model reads from
rather than computes on. So you can leave that 25 million row table in slow flash
and pull only the few rows each token needs, about 450 bytes, while the small part
that does the actual work stays in fast memory. The large model then costs almost
nothing to run, because you never load most of it. It just sits in flash and gets
sampled a little at a time.

That idea is Google's Per-Layer Embeddings, from Gemma 3n and Gemma 4. Here it runs
on the memory layout of a microcontroller instead of a phone or a GPU. As far as I
can tell, nobody had tried it on a chip this small.

```
  SRAM  (fast, tiny)   the "thinking" core, used on every token
  PSRAM (medium)       the output head and working memory
  FLASH (huge, slow)   the 25M-param table, about 6 rows read per token (~450 B)
```

## What it does, and what it does not

The model was trained on TinyStories, so it writes short, simple stories and mostly
keeps them coherent. It will not answer questions, follow instructions, write code,
or know facts. That limit comes from the small part of the model that does the
reasoning, and the memory trick does not change it. What is interesting here is the
architecture, fitting a large model onto a tiny chip, rather than what a 28.9 million
parameter model can say.

## Running it yourself

The firmware, the wiring, and the flashing steps live in
[`firmware/esp32_llm/README.md`](firmware/esp32_llm/README.md). The training,
ablation, and quantization code is in `src/` and `experiments/`. The full method,
the ablations, and the on-chip measurements are written up in
[`RESULTS.md`](RESULTS.md).

## Credit

TinyStories is the dataset this trains on: short synthetic stories simple enough
that a small model can still learn to write coherently (Ronen Eldan and Yuanzhi Li,
Microsoft Research, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)). The other
half is Per-Layer Embeddings, Google's design from the Gemma models, which is what
lets a big model fit on a small chip.

Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) is why a lot of
people, me included, believe you can train a tiny language model and run it in plain
C at all. This grew out of that.

## How this actually went

I left the messy history in the repo on purpose. That includes a bug I found in my
own parameter accounting, which had inflated an early number, and the corrected
result that followed once I fixed it. The commit history and `RESULTS.md` show where
the numbers moved and why.
