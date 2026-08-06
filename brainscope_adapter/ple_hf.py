"""HF-transformers wrapper for the ESP32 PLE TinyLM.

The point: brainscope loads models with AutoModelForCausalLM.from_pretrained and
then introspects a llama-shaped skeleton - model.layers (each with .self_attn
and .mlp), model.norm, get_output_embeddings(). This module dresses the PLE
architecture (src/model.py) in exactly that skeleton, so brainscope's residual
stream hooks, logit lens and attention capture all land without touching
brainscope itself.

Importing this module registers the architecture with transformers' Auto
classes; after that, from_pretrained on a directory whose config.json says
`"model_type": "ple-tinylm"` just works in this process.

Forward math is a line-for-line port of src/model.py (arm="ple"), extended with
a KV cache because brainscope decodes one token at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class PLEConfig(PretrainedConfig):
    model_type = "ple-tinylm"

    def __init__(self, vocab_size=32768, out_vocab=25353, hidden_size=96,
                 num_hidden_layers=6, num_attention_heads=4, ffn_hidden=66,
                 ple_dim=128, seq_len=256, rope_theta=10000.0,
                 max_position_embeddings=2048, **kwargs):
        self.vocab_size = vocab_size
        self.out_vocab = out_vocab
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden = ffn_hidden
        self.ple_dim = ple_dim
        self.seq_len = seq_len
        self.rope_theta = rope_theta
        # RoPE is precomputed to this length so the UI can decode past the
        # trained context (quality degrades beyond seq_len, nothing crashes).
        self.max_position_embeddings = max_position_embeddings
        # The device ties the head to the FIRST out_vocab rows of the embedding;
        # transformers' whole-tensor tying would clobber that, so it stays off
        # and build_hf.py copies the rows instead.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _apply_rope(x, cos, sin):
    # x: (B, H, T, Dh); cos/sin already sliced to these T positions
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class PLEAttention(nn.Module):
    def __init__(self, cfg: PLEConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, past=None, output_attentions=False):
        B, T, C = x.shape
        H, Dh = self.cfg.num_attention_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        past_len = 0
        if past is not None:
            pk, pv = past
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        # Eager attention so the weights exist for brainscope to look at.
        scores = q @ k.transpose(-1, -2) * Dh**-0.5           # (B,H,T,past+T)
        kv_pos = torch.arange(k.shape[2], device=x.device)
        q_pos = torch.arange(T, device=x.device) + past_len
        mask = kv_pos[None, :] > q_pos[:, None]               # future positions
        scores = scores.masked_fill(mask[None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out), (k, v), (weights if output_attentions else None)


class PLEMLP(nn.Module):
    def __init__(self, cfg: PLEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PLEDecoderLayer(nn.Module):
    def __init__(self, cfg: PLEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size)
        self.self_attn = PLEAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size)
        self.mlp = PLEMLP(cfg)
        self.ple_gate = nn.Linear(cfg.hidden_size, cfg.ple_dim, bias=False)
        self.ple_proj = nn.Linear(cfg.ple_dim, cfg.hidden_size, bias=False)
        self.ple_norm = RMSNorm(cfg.hidden_size)

    def forward(self, x, cos, sin, ple, past=None, output_attentions=False):
        a, kv, w = self.self_attn(self.input_layernorm(x), cos, sin, past, output_attentions)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        g = F.gelu(self.ple_gate(x))
        x = x + self.ple_norm(self.ple_proj(g * ple))
        return x, kv, w


class PLEModel(nn.Module):
    def __init__(self, cfg: PLEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.ple_model_proj = nn.Linear(cfg.hidden_size, cfg.num_hidden_layers * cfg.ple_dim, bias=False)
        self.ple_proj_norm = RMSNorm(cfg.ple_dim)
        self.ple_table = nn.Embedding(cfg.vocab_size, cfg.num_hidden_layers * cfg.ple_dim)
        self.layers = nn.ModuleList(PLEDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size)

        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
        t = torch.arange(cfg.max_position_embeddings).float()
        freqs = torch.outer(t, inv)
        # persistent=True on purpose: from_pretrained fast-inits on the meta
        # device, so a non-persistent buffer would come back as uninitialized
        # memory. Shipping the tables in the checkpoint sidesteps that.
        self.register_buffer("cos", freqs.cos(), persistent=True)
        self.register_buffer("sin", freqs.sin(), persistent=True)


class PLETinyLMForCausalLM(PreTrainedModel):
    config_class = PLEConfig
    base_model_prefix = "model"
    _no_split_modules = ["PLEDecoderLayer"]

    def __init__(self, config: PLEConfig):
        super().__init__(config)
        self.model = PLEModel(config)
        # On the device the head IS the first out_vocab rows of the embedding
        # (tied, scanned once per token from PSRAM). Held as its own tensor here
        # so get_output_embeddings/logit lens see a plain Linear.
        self.lm_head = nn.Linear(config.hidden_size, config.out_vocab, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, past_key_values=None, use_cache=False,
                output_hidden_states=False, output_attentions=False,
                attention_mask=None, **kwargs):
        cfg = self.config
        core = self.model
        B, T = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        cos = core.cos[past_len:past_len + T]
        sin = core.sin[past_len:past_len + T]

        x = core.embed_tokens(input_ids)
        L, P = cfg.num_hidden_layers, cfg.ple_dim
        ple = core.ple_model_proj(x) * (cfg.hidden_size**-0.5)
        ple = core.ple_proj_norm(ple.view(B, T, L, P))
        table = core.ple_table(input_ids).view(B, T, L, P)
        # embed_scale sqrt(P) on the table, then average the two halves - the
        # undocumented-but-load-bearing scaling from Gemma, kept from training.
        ple = (ple + table * (P**0.5)) * (2**-0.5)

        hidden = [x] if output_hidden_states else None
        new_past = [] if use_cache else None
        attns = [] if output_attentions else None
        for i, layer in enumerate(core.layers):
            past = past_key_values[i] if past_key_values else None
            x, kv, w = layer(x, cos, sin, ple[:, :, i], past, output_attentions)
            if output_hidden_states:
                hidden.append(x)
            if use_cache:
                new_past.append(kv)
            if output_attentions:
                attns.append(w)

        logits = self.lm_head(core.norm(x))
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=tuple(new_past) if use_cache else None,
            hidden_states=tuple(hidden) if output_hidden_states else None,
            attentions=tuple(attns) if output_attentions else None,
        )


AutoConfig.register("ple-tinylm", PLEConfig)
AutoModelForCausalLM.register(PLEConfig, PLETinyLMForCausalLM)


# exporter tensor name -> wrapper tensor name
def bin_to_hf_key(name: str) -> str:
    out = (name
           .replace("tok_emb.weight", "embed_tokens.weight")
           .replace("out_norm.weight", "norm.weight"))
    if out.startswith("blocks."):
        out = (out.replace("blocks.", "layers.")
               .replace(".attn_norm.", ".input_layernorm.")
               .replace(".attn.qkv.", ".self_attn.qkv.")
               .replace(".attn.proj.", ".self_attn.o_proj.")
               .replace(".ffn_norm.", ".post_attention_layernorm.")
               .replace(".ffn.gate.", ".mlp.gate_proj.")
               .replace(".ffn.up.", ".mlp.up_proj.")
               .replace(".ffn.down.", ".mlp.down_proj."))
    return "model." + out
