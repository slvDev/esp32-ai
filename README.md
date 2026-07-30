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

### Quick sanity check
Before a full training run, you can confirm your environment is set up correctly with a fast smoke test:
\`\`\`
uv run python data/prepare.py
uv run python src/train.py --arm ple --steps 20 --batch-size 4 --seq-len 64 --eval-every 10 --tag smoke-test
\`\`\`
This runs in under a minute and confirms data prep, model build, and the training loop all work end to end.