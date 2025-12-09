"""
Test script to evaluate a trained model checkpoint and calculate BPC (Bits Per Character).

Usage:
$ python test.py --checkpoint out-enwik8-baseline/ckpt.pt --dataset enwik8
$ python test.py --checkpoint out-shakespeare-char/ckpt.pt --dataset shakespeare_char
"""

import os
import pickle
import argparse
import math
import numpy as np
import torch
from contextlib import nullcontext

from model import GPT, GPTConfig

# default config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

def test(checkpoint_path, dataset, eval_iters=200, batch_size=32, block_size=None):
    """
    Load a checkpoint and evaluate on test/validation set.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'out-enwik8-baseline/ckpt.pt')
        dataset: Dataset name ('enwik8', 'shakespeare_char', etc.)
        eval_iters: Number of batches to evaluate
        batch_size: Batch size for evaluation
        block_size: Context length for evaluation (uses checkpoint value if None)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config and state
    checkpoint_model_args = checkpoint['model_args']
    model_args = checkpoint_model_args.copy()
    
    # Use block_size from checkpoint if not specified
    if block_size is None:
        block_size = model_args.get('block_size', 1024)
        print(f"Using block_size from checkpoint: {block_size}")
    
    # Create model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model config: {model_args}")
    print(f"Checkpoint iter: {checkpoint.get('iter_num', 'unknown')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    # Load metadata for encoding/decoding
    meta_path = os.path.join('data', dataset, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"Vocab size: {vocab_size}")
    else:
        print(f"Warning: meta.pkl not found at {meta_path}")
        vocab_size = model.config.vocab_size
    
    # Data loading function
    data_dir = os.path.join('data', dataset)
    
    def get_batch(split='val'):
        """Get a batch from validation or test data."""
        if split == 'test':
            data_file = os.path.join(data_dir, 'test.bin')
            if not os.path.exists(data_file):
                print(f"Test data not found, using validation data instead")
                data_file = os.path.join(data_dir, 'val.bin')
        else:
            data_file = os.path.join(data_dir, 'val.bin')
        
        if not os.path.exists(data_file):
            print(f"Error: Data file not found at {data_file}")
            return None, None
        
        data = np.memmap(data_file, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if device == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y
    
    # Evaluation loop
    print(f"\nEvaluating on validation set ({eval_iters} batches)...")
    losses = torch.zeros(eval_iters)
    
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = get_batch('val')
            if X is None:
                break
            
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            
            if (k + 1) % 50 == 0:
                print(f"  Batch {k+1}/{eval_iters}: loss = {loss.item():.4f}")
    
    val_loss = losses.mean().item()
    
    # Calculate BPC (Bits Per Character)
    # BPC = loss * ln(2) where loss is in nats (natural log)
    bpc = val_loss * math.log(2)
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"{'='*50}")
    print(f"Validation Loss (nats): {val_loss:.4f}")
    print(f"Bits Per Character (BPC): {bpc:.4f}")
    print(f"Perplexity: {math.exp(val_loss):.2f}")
    print(f"{'='*50}")
    
    return val_loss, bpc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model and calculate BPC")
    
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to checkpoint (e.g., out-enwik8-baseline/ckpt.pt)'
    )
    parser.add_argument(
        '--dataset', type=str, default='enwik8',
        choices=['enwik8', 'shakespeare_char', 'openwebtext'],
        help='Dataset used for training'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        choices=['val', 'test'],
        help='Which split for evaluation'
    )
    parser.add_argument(
        '--eval_iters', type=int, default=200,
        help='Number of batches to evaluate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--block_size', type=int, default=None,
        help='Context length (uses checkpoint value if not specified)'
    )
    
    args = parser.parse_args()
    
    test(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=args.block_size
    )
