# train a miniature character-level shakespeare model with ALiBi attention
# good for debugging and playing on macbooks and such

eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 50 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

wandb_log = True # override via command line if you like
wandb_project = 'enwik8-baseline'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64  # Increased for better gradient estimates
block_size = 512 # Longer context helps ALiBi shine (it extrapolates well)

# ALiBi GPT model - uses Attention with Linear Biases instead of positional embeddings
model_type = 'alibi'  # 'standard' or 'alibi'
n_layer = 8  # Deeper model for better capacity
n_head = 8   # More heads for better representation
n_embd = 512 # Wider model for better capacity
dropout = 0.2  # Lower dropout - ALiBi is less prone to overfitting
bias = True  # Use bias for better expressiveness

learning_rate = 3e-4 # Optimal for Adam with this architecture
max_iters = 50000 # More iterations for convergence
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.95 # Standard Adam beta2 works better with larger models
weight_decay = 0.1  # Regularization helps generalization

warmup_iters = 100 # Longer warmup for stability

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
# dtype = 'float16' # removes the warning:  [0/0] Not enough SMs to use max_autotune_gemm mode --> nope

out_dir =        f'out-enwik8-baseline-customgpt-alibi-{max_iters}-{n_layer}-{n_head}-{n_embd}-{block_size}'
wandb_run_name = f'enwik8-baseline-customgpt-alibi-{max_iters}-{n_layer}-{n_head}-{n_embd}-{block_size}'