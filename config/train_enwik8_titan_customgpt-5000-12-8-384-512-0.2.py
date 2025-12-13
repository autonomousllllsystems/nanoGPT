# Train enwik8 character-level model with Titan sparse attention
# Enables longer context via interleaved dense/sparse layers

eval_interval = 500
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
init_from = 'scratch'

wandb_log = True
wandb_project = 'enwik8-baseline'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512  # 2x longer context than baseline (256)

# Titan GPT - interleaved dense and sparse attention
model_type = 'titan'
sparse_mode = 'strided'  # 'strided' or 'window'
sparse_stride = 2        # attend to every 2nd position in sparse layers
sparse_window = 128      # alternative: sliding window size
sparse_layer_interval = 2  # alternate dense/sparse every 2 layers (0=dense, 1=sparse, 2=dense, ...)

# Model architecture
n_layer = 12
n_head = 8
n_embd = 384
dropout = 0.2

# Optimizer
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = max_iters
min_lr = 1e-5
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

warmup_iters = 100

# System
# device = 'cuda'
# compile = False  # disable for debugging
# dtype = 'float16'

out_dir = f'out-enwik8-titan-{max_iters}-{n_layer}-{n_head}-{n_embd}-{block_size}-{dropout}'
wandb_run_name = f'enwik8-titan-{max_iters}-{n_layer}-{n_head}-{n_embd}-{block_size}-{dropout}'


# ============================================================
# Final Test Set Evaluation
# ============================================================
# Test loss: 1.0622
# Test BPC:  1.5325
# Test BPB:  0.1916
# Test perplexity: 2.89