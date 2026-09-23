"""Build a transformers-loadable twin of the deployed ESP32 model.

Reads artifacts/tinystories/{model.bin,tokenizer.json} - the exact files
deploy.sh flashes - and writes artifacts/tinystories/hf/ with the dequantized
fp32 weights in the brainscope-compatible skeleton. Ends with a short greedy
sample as a smoke test.

Run from the repo root with a python that has torch + transformers:
    python brainscope_adapter/build_hf.py [--artifacts artifacts/tinystories]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ple_bin import load_model_bin
from ple_hf import PLEConfig, PLETinyLMForCausalLM, bin_to_hf_key


def build(artifacts: Path):
    cfg_bin, sd_bin = load_model_bin(artifacts / "model.bin")
    print(f"model.bin: Vin={cfg_bin.vocab_size} Vout={cfg_bin.out_vocab} "
          f"D={cfg_bin.d_model} L={cfg_bin.n_layers} H={cfg_bin.n_heads} "
          f"F={cfg_bin.ffn_hidden} P={cfg_bin.ple_dim} seq={cfg_bin.seq_len}")
    if not cfg_bin.tied_head:
        raise SystemExit("model.bin is untied; this builder handles the tied-head layout")

    cfg = PLEConfig(
        vocab_size=cfg_bin.vocab_size, out_vocab=cfg_bin.out_vocab,
        hidden_size=cfg_bin.d_model, num_hidden_layers=cfg_bin.n_layers,
        num_attention_heads=cfg_bin.n_heads, ffn_hidden=cfg_bin.ffn_hidden,
        ple_dim=cfg_bin.ple_dim, seq_len=cfg_bin.seq_len,
        rope_theta=cfg_bin.rope_theta,
        eos_token_id=0, pad_token_id=0,
    )
    model = PLETinyLMForCausalLM(cfg)
    sd = {bin_to_hf_key(k): torch.from_numpy(v) for k, v in sd_bin.items()}
    # Tied head: the first out_vocab rows of the (dequantized) embedding.
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"][: cfg.out_vocab].clone()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith((".cos", ".sin"))]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def save(model, artifacts: Path, out: Path):
    from transformers import PreTrainedTokenizerFast

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(artifacts / "tokenizer.json"),
        eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    # Not a chat model: the "chat" is the story so far, concatenated verbatim,
    # so brainscope's chat box behaves as a continue-the-story box.
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tok.save_pretrained(out)
    print(f"wrote {out}")
    return tok


def sample(model, tok, prompt="Once upon a time", n=40):
    ids = tok(prompt, return_tensors="pt").input_ids
    past = None
    out_ids = []
    with torch.no_grad():
        feed = ids
        for _ in range(n):
            o = model(input_ids=feed, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            if nxt == 0:
                break
            out_ids.append(nxt)
            feed = torch.tensor([[nxt]])
    print(f"smoke sample: {prompt!r} -> {tok.decode(out_ids)!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path,
                    default=Path(__file__).resolve().parents[1] / "artifacts" / "tinystories")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <artifacts>/hf)")
    ap.add_argument("--zero-ple-table", action="store_true",
                    help="write the twin with the flash-resident 25M-param PLE "
                         "table zeroed - the 'flash unplugged' ablation")
    args = ap.parse_args()
    model = build(args.artifacts)
    if args.zero_ple_table:
        with torch.no_grad():
            model.model.ple_table.weight.zero_()
        print("PLE table zeroed: the flash-resident parameters are unplugged")
    out = args.out or args.artifacts / "hf"
    tok = save(model, args.artifacts, out)
    sample(model, tok)
