"""
Verify that a dataset has been properly prepared with .bin files.
Usage: python analysis/verify_dataset.py --dataset enwik8
"""

import os
import pickle
import argparse
import numpy as np

def verify_dataset(dataset_name):
    """
    Verify dataset integrity and print statistics.
    
    Args:
        dataset_name: Name of dataset (e.g., 'enwik8', 'shakespeare_char')
    """
    data_dir = os.path.join('data', dataset_name)
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Verifying dataset: {dataset_name}")
    print(f"{'='*60}")
    
    all_good = True
    
    # Check train.bin
    train_path = os.path.join(data_dir, 'train.bin')
    if os.path.exists(train_path):
        train_size = os.path.getsize(train_path)
        train_tokens = train_size // 2  # uint16 = 2 bytes
        print(f"✅ train.bin: {train_tokens:,} tokens ({train_size / 1e6:.2f} MB)")
    else:
        print(f"❌ train.bin not found")
        all_good = False
    
    # Check val.bin
    val_path = os.path.join(data_dir, 'val.bin')
    if os.path.exists(val_path):
        val_size = os.path.getsize(val_path)
        val_tokens = val_size // 2
        print(f"✅ val.bin: {val_tokens:,} tokens ({val_size / 1e6:.2f} MB)")
    else:
        print(f"❌ val.bin not found")
        all_good = False
    
    # Check test.bin
    test_path = os.path.join(data_dir, 'test.bin')
    if os.path.exists(test_path):
        test_size = os.path.getsize(test_path)
        test_tokens = test_size // 2
        print(f"✅ test.bin: {test_tokens:,} tokens ({test_size / 1e6:.2f} MB)")
    else:
        print(f"⚠️  test.bin not found (optional)")
    
    # Check meta.pkl
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            vocab_size = meta.get('vocab_size', 'unknown')
            print(f"✅ meta.pkl: vocab_size = {vocab_size}")
        except Exception as e:
            print(f"❌ meta.pkl corrupted: {e}")
            all_good = False
    else:
        print(f"❌ meta.pkl not found")
        all_good = False
    
    # Summary statistics
    if all_good:
        total_tokens = train_tokens + val_tokens
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Train/Val split: {100*train_tokens/total_tokens:.1f}% / {100*val_tokens/total_tokens:.1f}%")
        print(f"{'='*60}")
        
        # Training estimate
        print(f"\nTraining estimate (from config/train_enwik8_baseline_baby.py):")
        batch_size = 64
        block_size = 256
        max_iters = 5000
        tokens_per_iter = batch_size * block_size
        total_train_tokens = tokens_per_iter * max_iters
        epochs = total_train_tokens / train_tokens
        print(f"  Batch size: {batch_size}")
        print(f"  Block size: {block_size}")
        print(f"  Max iterations: {max_iters}")
        print(f"  Tokens per iteration: {tokens_per_iter:,}")
        print(f"  Total training tokens: {total_train_tokens:,}")
        print(f"  Epochs: {epochs:.2f}")
        print(f"{'='*60}\n")
        
        return True
    else:
        print(f"\n❌ Dataset is incomplete! Run:")
        print(f"   cd data/{dataset_name}")
        print(f"   python prepare.py")
        print(f"{'='*60}\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset preparation")
    parser.add_argument(
        '--dataset', type=str, default='enwik8',
        choices=['enwik8', 'shakespeare_char', 'openwebtext'],
        help='Dataset to verify'
    )
    
    args = parser.parse_args()
    success = verify_dataset(args.dataset)
    exit(0 if success else 1)
