"""Parse artifacts/<model>/model.bin back into fp32 tensors.

The binary is the exporter's int4 group-quantized format (research/tinystories/
export.py). Dequantizing here reproduces exactly the weights the C runtime
reconstructs on the device: codes are (q+8) nibbles, scales are fp16, groups of
`group` along the last dim, ragged tail. So the model this yields is not "the
checkpoint" - it is the model the ESP32 actually runs, bit-faithful.
"""

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x00454C50  # "PLE\0"


@dataclass
class BinConfig:
    vocab_size: int
    out_vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_hidden: int
    ple_dim: int
    seq_len: int
    group: int
    rope_theta: float
    tied_head: bool


def _dequant(buf, off, shape, group):
    """Read one packed int4 tensor; returns (fp32 ndarray, new offset)."""
    (tensor_group,) = struct.unpack_from("<i", buf, off)
    off += 4
    assert tensor_group == group, f"group {tensor_group} != header {group}"
    cols = shape[-1]
    rows = int(np.prod(shape)) // cols
    row_bytes = (cols + 1) // 2
    n_groups = (cols + group - 1) // group

    packed = np.frombuffer(buf, np.uint8, rows * row_bytes, off).reshape(rows, row_bytes)
    off += rows * row_bytes
    scales = np.frombuffer(buf, np.float16, rows * n_groups, off).reshape(rows, n_groups)
    off += rows * n_groups * 2

    codes = np.empty((rows, cols), np.int8)
    lo = (packed & 0x0F).astype(np.int8)
    hi = (packed >> 4).astype(np.int8)
    codes[:, 0::2] = lo[:, : (cols + 1) // 2]
    codes[:, 1::2] = hi[:, : cols // 2]
    q = codes.astype(np.float32) - 8.0

    sc = np.repeat(scales.astype(np.float32), group, axis=1)[:, :cols]
    return (q * sc).reshape(shape), off


def _fp32(buf, off, shape):
    n = int(np.prod(shape))
    arr = np.frombuffer(buf, np.float32, n, off).reshape(shape).copy()
    return arr, off + n * 4


def load_model_bin(path):
    """Returns (BinConfig, {name: fp32 ndarray}) with the exporter's names."""
    buf = open(path, "rb").read()
    magic, version, header_bytes, flags = struct.unpack_from("<IIII", buf, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic:#x}")
    if version != 1:
        raise ValueError(f"{path}: unsupported format version {version}")
    vocab, out_vocab = struct.unpack_from("<II", buf, 16)
    d, L, H, F, P, S, G = struct.unpack_from("<7i", buf, 24)
    (theta,) = struct.unpack_from("<f", buf, 52)
    cfg = BinConfig(vocab, out_vocab, d, L, H, F, P, S, G, theta, bool(flags & 1))

    # Same fixed order the exporter writes and the C reader hard-codes.
    plan = [
        ("tok_emb.weight", (vocab, d), True),
        ("ple_model_proj.weight", (L * P, d), True),
        ("ple_proj_norm.weight", (P,), False),
        ("ple_table.weight", (vocab, L * P), True),
    ]
    for i in range(L):
        p = f"blocks.{i}."
        plan += [
            (p + "attn_norm.weight", (d,), False),
            (p + "attn.qkv.weight", (3 * d, d), True),
            (p + "attn.proj.weight", (d, d), True),
            (p + "ffn_norm.weight", (d,), False),
            (p + "ffn.gate.weight", (F, d), True),
            (p + "ffn.up.weight", (F, d), True),
            (p + "ffn.down.weight", (d, F), True),
            (p + "ple_gate.weight", (P, d), True),
            (p + "ple_proj.weight", (d, P), True),
            (p + "ple_norm.weight", (d,), False),
        ]
    plan.append(("out_norm.weight", (d,), False))

    off = header_bytes
    sd = {}
    for name, shape, quant in plan:
        if quant:
            sd[name], off = _dequant(buf, off, shape, G)
        else:
            sd[name], off = _fp32(buf, off, shape)
    if off != len(buf):
        raise ValueError(f"{path}: {len(buf) - off} trailing bytes after last tensor")
    return cfg, sd
