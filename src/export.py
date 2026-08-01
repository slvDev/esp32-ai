"""Export a trained PLE model to a flat binary the C inference can mmap, plus a
golden logits reference so the C port can be proven correct before it touches
hardware.

Format is deliberately dead-simple (the C reader is ~30 lines):
  [header: magic + int32 config fields]
  then, in a fixed order the C code hard-codes, each tensor as either
    QUANT: int4 codes packed 2-per-byte (group-wise, group=G along last dim)
           followed by fp16 scales, one per group
    FP32 : raw fp32 (norms only -- tiny)

Quantization matches src/quantize.py exactly (symmetric group-wise int4), so the
golden logits are the *4-bit* model's logits: C-vs-PyTorch then isolates port
correctness from quantization error, which was measured separately.
"""

import os
import struct
import sys

import numpy as np
import torch
import argparse
from model import Config, TinyLM
from quantize import quantize_groupwise

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
OUT = os.path.join(HERE, "..", "firmware", "model")
MAGIC = 0x504C4531  # "PLE1"
GROUP = 128  # uniform; fp16 scales + ragged packing keep the 28.9M model < 16MB


def quant_pack(w, group=GROUP):
    """Group-wise symmetric int4, ragged (no padding) with fp16 scales.

    Returns (packed_uint8[rows, ceil(cols/2)], scales_fp16[rows, n_groups], dq).
    The last group of each row may be shorter than `group`; the C reader derives
    n_groups=ceil(cols/group) and row_bytes=ceil(cols/2). Scales are rounded to
    fp16 *before* computing dq so the golden matches the C exactly. dq is the
    dequantized fp32 tensor the C reconstructs.
    """
    w = w.float()
    out_shape = w.shape
    x = w.reshape(-1, out_shape[-1])
    rows, cols = x.shape
    n_groups = (cols + group - 1) // group
    q = torch.zeros(rows, cols)
    dq = torch.zeros(rows, cols)
    scales = torch.zeros(rows, n_groups)
    for gi in range(n_groups):
        a, b = gi * group, min((gi + 1) * group, cols)
        seg = x[:, a:b]
        sc = (seg.abs().amax(dim=1, keepdim=True) / 7).clamp_min(1e-8)
        sc = sc.half().float()  # fp16-round the scale
        scales[:, gi] = sc.squeeze(1)
        qi = torch.clamp(torch.round(seg / sc), -7, 7)
        q[:, a:b] = qi
        dq[:, a:b] = qi * sc
    dq = dq.reshape(out_shape)

    codes = (q.to(torch.int16) + 8).to(torch.uint8).numpy()  # rows x cols, 0..15
    row_bytes = (cols + 1) // 2
    packed = np.zeros((rows, row_bytes), dtype=np.uint8)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed[:, : lo.shape[1]] = lo
    packed[:, : hi.shape[1]] |= (hi << 4)
    scales16 = scales.numpy().astype(np.float16)
    return packed.reshape(-1), scales16.reshape(-1), dq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ple-cleandeploy-s0",
                    help="run tag to export, matches runs/{tag}.pt")
    args = ap.parse_args()
    tag = args.tag
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(os.path.join(RUNS, f"{tag}.pt"), map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg)
    model.load_state_dict(ck["state"])
    model.eval()

    # Ordered list: (name, tensor, is_quant). C reads in exactly this order.
    plan = []
    sd = model.state_dict()

    def add(name, quant):
        plan.append((name, sd[name], quant))

    add("tok_emb.weight", True)           # tied: input embed + output head
    add("ple_model_proj.weight", True)
    add("ple_proj_norm.weight", False)
    add("ple_table.weight", True)         # the 25M-param table
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        add(p + "attn_norm.weight", False)
        add(p + "attn.qkv.weight", True)
        add(p + "attn.proj.weight", True)
        add(p + "ffn_norm.weight", False)
        add(p + "ffn.gate.weight", True)
        add(p + "ffn.up.weight", True)
        add(p + "ffn.down.weight", True)
        add(p + "ple_gate.weight", True)
        add(p + "ple_proj.weight", True)
        add(p + "ple_norm.weight", False)
    add("out_norm.weight", False)

    # Reconstruct a dequantized state dict to drive the golden forward.
    dq_sd = {k: v.clone() for k, v in sd.items()}
    blobs = []
    for name, t, quant in plan:
        if quant:
            packed, scales, dq = quant_pack(t)
            dq_sd[name] = dq
            blobs.append(("Q", name, t.shape, packed, scales))
        else:
            blobs.append(("F", name, t.shape, t.contiguous().numpy().astype(np.float32), None))

    # Write binary.
    path = os.path.join(OUT, "model.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        for v in [cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads,
                  cfg.ffn_hidden, cfg.ple_dim, cfg.seq_len, GROUP]:
            f.write(struct.pack("<i", v))
        f.write(struct.pack("<f", cfg.rope_theta))
        for entry in blobs:
            kind = entry[0]
            if kind == "Q":
                _, _, _, packed, scales = entry
                f.write(struct.pack("<i", GROUP))   # per-tensor group
                f.write(packed.tobytes())           # ragged int4 codes
                f.write(scales.tobytes())           # fp16 scales
            else:
                _, _, _, arr, _ = entry
                f.write(arr.tobytes())
    size = os.path.getsize(path)
    print(f"wrote {path}  ({size/1e6:.2f} MB)  {len(plan)} tensors")

    # Keep the tied output head == dequantized input embedding. state_dict lists
    # both keys for tied weights; without this the head silently stays fp32 and
    # the golden no longer matches the (fully-quantized) C port.
    if "head.weight" in dq_sd:
        dq_sd["head.weight"] = dq_sd["tok_emb.weight"]

    # Golden: load dequantized weights, forward a fixed prompt, dump last-pos logits.
    gold = TinyLM(cfg)
    gold.load_state_dict(dq_sd)
    gold.eval()
    prompt = [1, 500, 1000, 200, 42, 777, 13, 99]  # arbitrary fixed token ids
    ids = torch.tensor([prompt])
    with torch.no_grad():
        logits, _ = gold(ids)
    last = logits[0, -1].numpy().astype(np.float32)
    np.savez(os.path.join(OUT, "golden.npz"),
             prompt=np.array(prompt, dtype=np.int32), logits=last)
    # Text form for the C host verifier: plen, prompt ids, then V logits.
    with open(os.path.join(OUT, "golden.txt"), "w") as gf:
        gf.write(f"{len(prompt)}\n")
        gf.write(" ".join(str(t) for t in prompt) + "\n")
        gf.write("\n".join(f"{v:.6f}" for v in last) + "\n")
    top5 = last.argsort()[-5:][::-1]
    print(f"golden: prompt={prompt}")
    print(f"golden: last-pos top5 token ids = {top5.tolist()}")
    print(f"golden: logit range [{last.min():.3f}, {last.max():.3f}]")


if __name__ == "__main__":
    main()
