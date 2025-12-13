"""Transformer-XL style relative positional encoding (relational) GPT.

Implements attention with relative position encodings as in Transformer-XL:
https://arxiv.org/abs/1901.02860

This provides an alternative to learned absolute positions, ALiBi, and RoPE.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import BaseAttention, LayerNorm, MLP, GPTConfig, GPT


class RelativePositionEmbedding(nn.Module):
    """Sinusoidal relative position encoding table for keys (content) and bias terms.

    We build a table of size (2*max_len-1, head_dim) for relative distances.
    """

    def __init__(self, head_dim: int, max_len: int = 1024):
        super().__init__()
        self.max_len = max_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, T: int, device, dtype):
        # distances range: -(T-1)..(T-1)
        max_rel = min(self.max_len, T)
        dist = torch.arange(-(max_rel - 1), max_rel, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", dist, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb.to(dtype)  # shape: (2*max_rel-1, head_dim)


class RelAttention(BaseAttention):
    """Multi-head attention with Transformer-XL relative position encodings.

    Uses content-based term (q @ k^T), content-position term via RPE, and bias.
    """

    def __init__(self, config):
        super().__init__(config)
        self.head_dim = config.n_embd // config.n_head
        self.rpe = RelativePositionEmbedding(self.head_dim, max_len=config.block_size)
        # a separate linear for positional encoding projection (as in Transformer-XL)
        self.c_pos = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # positional encodings
        p = self.c_pos(x)  # (B, T, C)
        p = p.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)

        # Build relative position index map [i,j] -> i-j shifted to be >=0
        device = x.device
        rel_max = min(T, self.rpe.max_len)
        # indices limited to rel_max range
        idx_i = torch.arange(T, device=device)
        idx_j = torch.arange(T, device=device)
        rel = (idx_i.view(T, 1) - idx_j.view(1, T)).clamp(-(rel_max - 1), rel_max - 1)
        rel = rel + (rel_max - 1)  # shift to [0, 2*rel_max-2]

        # get RPE table for current T
        rpe_table = self.rpe(T, device, q.dtype)  # (2*rel_max-1, hd)
        # gather per (i,j) relative embedding
        rpe = rpe_table[rel]  # (T, T, hd)
        rpe = rpe.unsqueeze(0).unsqueeze(0)  # (1,1,T,T,hd)

        # content-based term
        scale = 1.0 / math.sqrt(self.head_dim)
        att_content = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # content-position term (q interacts with rpe via dot over head_dim)
        # q: (B, nh, T, hd), rpe: (1,1,T,T,hd) -> (B, nh, T, T)
        att_pos = (q.unsqueeze(3) * rpe).sum(-1) * scale

        att = att_content + att_pos

        if self.flash:
            # flash kernels don't support custom bias easily; fall back to manual path
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=device), 1) == 1, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class RelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = RelAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class RelGPTConfig(GPTConfig):
    pass


class RelGPT(GPT):
    """GPT model using Transformer-XL-style relative position encodings."""

    def __init__(self, config):
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([RelBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
