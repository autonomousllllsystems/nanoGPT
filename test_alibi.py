"""Quick test to verify ALiBi implementation is correct."""

import torch
from alibi import AlibiGPTConfig, AlibiGPT
from model import GPTConfig, GPT

def test_alibi_implementation():
    """Test that ALiBi is properly implemented."""
    
    print("=" * 60)
    print("Testing ALiBi Implementation")
    print("=" * 60)
    
    # Create small test configs
    config = AlibiGPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=True
    )
    
    # Create models
    alibi_model = AlibiGPT(config)
    
    # Check 1: No position embeddings in ALiBi
    print("\n✓ Check 1: Position Embeddings")
    has_wpe = hasattr(alibi_model.transformer, 'wpe')
    print(f"  ALiBi has position embeddings (wpe): {has_wpe}")
    assert not has_wpe, "❌ ALiBi should NOT have position embeddings!"
    print("  ✓ PASS: ALiBi correctly has no position embeddings")
    
    # Check 2: ALiBi slopes are registered
    print("\n✓ Check 2: ALiBi Slopes")
    first_block = alibi_model.transformer.h[0]
    has_slopes = hasattr(first_block.attn, 'alibi_slopes')
    print(f"  ALiBi attention has slopes: {has_slopes}")
    if has_slopes:
        slopes = first_block.attn.alibi_slopes
        print(f"  Slopes shape: {slopes.shape}")
        print(f"  Slopes values: {slopes.tolist()}")
        assert slopes.shape[0] == config.n_head, f"❌ Should have {config.n_head} slopes!"
        print("  ✓ PASS: ALiBi slopes correctly initialized")
    else:
        print("  ❌ FAIL: ALiBi slopes not found!")
        return False
    
    # Check 3: Forward pass works and uses ALiBi
    print("\n✓ Check 3: Forward Pass")
    test_input = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq=64
    
    with torch.no_grad():
        logits, loss = alibi_model(test_input, test_input)
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    assert logits.shape == (2, 64, config.vocab_size), "❌ Wrong output shape!"
    print("  ✓ PASS: Forward pass works correctly")
    
    # Check 4: Compare parameter count (ALiBi should have fewer params - no wpe)
    print("\n✓ Check 4: Parameter Count")
    alibi_params = alibi_model.get_num_params()
    
    # Create standard GPT for comparison
    std_config = GPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=True
    )
    std_model = GPT(std_config)
    std_params = std_model.get_num_params()
    
    print(f"  Standard GPT params: {std_params:,}")
    print(f"  ALiBi GPT params: {alibi_params:,}")
    print(f"  Difference: {std_params - alibi_params:,} (should be ~{config.block_size * config.n_embd:,} for wpe)")
    
    expected_diff = config.block_size * config.n_embd
    actual_diff = std_params - alibi_params
    # Allow some tolerance for other differences
    assert abs(actual_diff - expected_diff) < 1000, f"❌ Parameter difference unexpected!"
    print("  ✓ PASS: ALiBi has fewer parameters (no position embeddings)")
    
    # Check 5: Verify ALiBi bias is applied during attention
    print("\n✓ Check 5: ALiBi Bias Application")
    # Access the attention module
    attn = alibi_model.transformer.h[0].attn
    
    # Do a forward pass to populate cache
    test_seq = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        _ = alibi_model(test_seq)
    
    # Check if ALiBi bias was cached
    has_cached_bias = attn._cached_alibi_bias is not None
    print(f"  ALiBi bias cached: {has_cached_bias}")
    if has_cached_bias:
        print(f"  Cached bias shape: {attn._cached_alibi_bias.shape}")
        print(f"  Expected shape: ({config.n_head}, 32, 32)")
        assert attn._cached_alibi_bias.shape == (config.n_head, 32, 32), "❌ Wrong bias shape!"
        print("  ✓ PASS: ALiBi bias correctly applied")
    else:
        print("  ❌ FAIL: ALiBi bias not cached!")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! ALiBi is correctly implemented!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_alibi_implementation()
    exit(0 if success else 1)
