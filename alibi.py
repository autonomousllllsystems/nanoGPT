"""
ALiBi-enabled GPT model that derives from the base model.

Usage:
    from alibi import AlibiGPTConfig, AlibiGPT

Paper: https://arxiv.org/pdf/2108.12409
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import base classes from model
from model import BaseAttention, LayerNorm, MLP, Block, GPTConfig, GPT


class AlibiAttention(BaseAttention):
    """Attention with Attention with Linear Biases (ALiBi) instead of positional embeddings."""

    def __init__(self, config):
        super().__init__(config)
        # ALiBi uses relative position biases instead of learnable position embeddings
        # No need for causal mask buffer - we handle causality through ALiBi
        self.register_buffer("alibi_slopes", self._get_alibi_slopes(self.n_head), persistent=False)

    @staticmethod
    def _get_alibi_slopes(n_head: int):
        """Compute ALiBi slopes. From https://github.com/ofirpress/attention_with_linear_biases"""

        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(n_head).is_integer():
            slopes = get_slopes_power_of_2(n_head)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
            slopes.extend(extra[: n_head - closest_power_of_2])
        return torch.tensor(slopes)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Compute scaled dot-product attention: (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply ALiBi bias (causal + relative position)
        # Create relative position matrix: distance[i, j] = i - j
        device = x.device
        pos = torch.arange(T, device=device, dtype=torch.float32)
        
        # Distance matrix: (T, T) where [i,j] = i - j
        distances = pos.view(T, 1) - pos.view(1, T)  # Shape: (T, T)
        
        # ALiBi slopes are per head, need to multiply by distances
        # Shape: (nh,) * (1, T, T) -> (nh, T, T)
        alibi_slopes = self.alibi_slopes.to(device).view(self.n_head, 1, 1)
        alibi_bias = alibi_slopes * distances.unsqueeze(0)  # (nh, T, T)
        
        # Apply causal mask: set future positions to very negative value
        # Create causal mask: lower triangular = 0, upper triangular = 1
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        alibi_bias = alibi_bias.masked_fill(causal_mask.unsqueeze(0) == 1, float('-inf'))
        
        # Add ALiBi bias to attention scores
        att = att + alibi_bias.unsqueeze(0)  # Broadcast to batch: (B, nh, T, T)
        
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        # Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class AlibiBlock(nn.Module):
    """Transformer block using ALiBi attention."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = AlibiAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class AlibiGPTConfig(GPTConfig):
    """Config for ALiBi-based GPT. Inherits from GPTConfig."""

    pass


class AlibiGPT(GPT):
    """GPT model with ALiBi attention instead of positional embeddings."""

    def __init__(self, config):
        # Don't call super().__init__() as we need custom initialization
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Note: ALiBi doesn't use position embeddings (wpe)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([AlibiBlock(config) for _ in range(config.n_layer)]),
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
        """Return the number of parameters in the model. ALiBi has no position embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Only token embeddings, no position embeddings for ALiBi
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        """ALiBi doesn't have positional embeddings to crop."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
