"""
Unified configuration for GPT models (standard and ALiBi variants).
Supports easy model selection and parameter configuration.

Usage:
    # For standard GPT
    from config.gpt_config import GPTConfig, GPT
    config = GPTConfig()
    model = GPT(config)

    # For ALiBi GPT
    from config.gpt_config import AlibiGPTConfig, AlibiGPT
    config = AlibiGPTConfig()
    model = AlibiGPT(config)
"""

from dataclasses import dataclass
from model import GPT, GPTConfig
from alibi import AlibiGPT, AlibiGPTConfig

__all__ = ["GPTConfig", "GPT", "AlibiGPTConfig", "AlibiGPT"]

# Preset configurations for different model sizes
@dataclass
class SmallGPTConfig(GPTConfig):
    """Small GPT model configuration."""
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True


@dataclass
class MediumGPTConfig(GPTConfig):
    """Medium GPT model configuration."""
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


@dataclass
class SmallAlibiGPTConfig(AlibiGPTConfig):
    """Small ALiBi GPT model configuration."""
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True


@dataclass
class MediumAlibiGPTConfig(AlibiGPTConfig):
    """Medium ALiBi GPT model configuration."""
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
