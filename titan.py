"""Titan-style GPT with interleaved dense and sparse attention layers.

Alternates between full causal attention and sparse attention (strided or sliding window)
to reduce memory and enable longer context windows.

Inspired by Titan, BigBird, and Longformer architectures.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import BaseAttention, LayerNorm, MLP, GPTConfig, GPT


class DenseAttention(BaseAttention):
    """Standard causal self-attention (used in dense layers)."""

    def __init__(self, config):
        super().__init__(config)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SparseAttention(BaseAttention):
    """Sparse causal attention with configurable pattern (strided or sliding window)."""

    def __init__(self, config):
        super().__init__(config)
        self.sparse_mode = getattr(config, "sparse_mode", "strided")  # 'strided' or 'window'
        self.stride = getattr(config, "sparse_stride", 2)
        self.window_size = getattr(config, "sparse_window", 128)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(k.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # Build sparse mask
        device = x.device
        if self.sparse_mode == "strided":
            # Strided pattern: attend to every stride-th position + immediate predecessors
            mask = torch.ones(T, T, device=device, dtype=torch.bool)
            for i in range(T):
                # Always attend to immediate local context
                mask[i, max(0, i - 4) : i + 1] = False
                # Attend to strided positions
                strided_positions = torch.arange(0, i + 1, self.stride, device=device)
                mask[i, strided_positions] = False
        else:  # window
            # Sliding window: attend to last window_size tokens
            mask = torch.ones(T, T, device=device, dtype=torch.bool)
            for i in range(T):
                start = max(0, i - self.window_size + 1)
                mask[i, start : i + 1] = False

        att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class TitanBlock(nn.Module):
    """Transformer block with configurable attention type (dense or sparse)."""

    def __init__(self, config, use_sparse=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SparseAttention(config) if use_sparse else DenseAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TitanGPTConfig(GPTConfig):
    sparse_mode: str = "strided"  # 'strided' or 'window'
    sparse_stride: int = 2  # for strided mode
    sparse_window: int = 128  # for window mode
    sparse_layer_interval: int = 2  # alternate dense/sparse every N layers


class TitanGPT(GPT):
    """GPT with interleaved dense and sparse attention layers (Titan-style)."""

    def __init__(self, config):
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Build alternating dense/sparse blocks
        blocks = []
        for layer_idx in range(config.n_layer):
            use_sparse = (layer_idx % config.sparse_layer_interval) != 0
            blocks.append(TitanBlock(config, use_sparse=use_sparse))

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),  # Use learned positions
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(blocks),
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
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

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
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
