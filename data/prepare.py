"""Download a slice of TinyStories, train a small BPE, emit uint16 token bins.

The vocab is deliberately fixed across every ablation arm: cross-entropy is only
comparable between models that share a tokenizer.
"""

import argparse
import os
import sys

import numpy as np
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
RAW = os.path.join(HERE, "tinystories_slice.txt")
VOCAB_SIZE = 4096  # overridden by --vocab; 4096 keeps the original bin names
# ~300MB of stories is ~75M tokens: enough to overtrain a 1M-param core well past
# its compute-optimal point, which is the regime we actually care about.
SLICE_BYTES = 300 * 1024 * 1024
VAL_FRACTION = 0.005


def download():
    if os.path.exists(RAW) and os.path.getsize(RAW) >= SLICE_BYTES * 0.99:
        print(f"already have {RAW}")
        return
    print(f"downloading first {SLICE_BYTES / 1e6:.0f}MB of TinyStories...")
    got = 0
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(RAW, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got >= SLICE_BYTES:
                    break
                if got % (25 << 20) < (1 << 20):
                    print(f"  {got / 1e6:.0f}MB", flush=True)
    print(f"done, {got / 1e6:.0f}MB")


def train_tokenizer(text):
    path = os.path.join(HERE, f"bpe{VOCAB_SIZE}.json")
    if os.path.exists(path):
        print(f"already have {path}")
        return Tokenizer.from_file(path)
    print(f"training BPE vocab={VOCAB_SIZE}...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # 40MB is plenty to fit a 4k merge table; using the full slice just burns time.
    tok.train_from_iterator([text[: 40 * 1024 * 1024]], trainer=trainer)
    tok.save(path)
    return tok


def main():
    global VOCAB_SIZE, SLICE_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--max-bytes", type=int, default=SLICE_BYTES,
                    help="bytes of TinyStories to download/train on (default "
                         f"{SLICE_BYTES / 1e6:.0f}MB; pass a smaller value for a "
                         "quick smoke test or CI)")
    args = ap.parse_args()
    VOCAB_SIZE = args.vocab
    SLICE_BYTES = args.max_bytes
    # vocab 4096 keeps the original train.bin/val.bin; others get suffixed names
    # so both datasets coexist and train.py can pick by --vocab.
    suffix = "" if VOCAB_SIZE == 4096 else f"_v{VOCAB_SIZE}"

    download()
    with open(RAW, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Drop the trailing partial story left by the byte-slice.
    text = text[: text.rfind("<|endoftext|>") + len("<|endoftext|>")]

    tok = train_tokenizer(text)
    eot = tok.token_to_id("<|endoftext|>")
    print(f"eot id = {eot}")

    print("encoding...")
    docs = text.split("<|endoftext|>")
    ids = []
    for i in range(0, len(docs), 20000):
        batch = [d for d in docs[i : i + 20000] if d.strip()]
        for enc in tok.encode_batch(batch):
            ids.extend(enc.ids)
            ids.append(eot)
        print(f"  {i + len(batch)}/{len(docs)} docs, {len(ids) / 1e6:.1f}M tokens", flush=True)

    dtype = np.uint16 if VOCAB_SIZE <= 65536 else np.uint32
    arr = np.array(ids, dtype=dtype)
    assert arr.max() < VOCAB_SIZE
    n_val = int(len(arr) * VAL_FRACTION)
    arr[:-n_val].tofile(os.path.join(HERE, f"train{suffix}.bin"))
    arr[-n_val:].tofile(os.path.join(HERE, f"val{suffix}.bin"))
    print(f"train {len(arr) - n_val:,} tokens / val {n_val:,} tokens")
    print(f"compression: {len(text) / len(arr):.2f} bytes/token")


if __name__ == "__main__":
    sys.exit(main())
