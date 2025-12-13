# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 50 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

wandb_log = True # override via command line if you like
wandb_project = 'enwik8-baseline'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256 # context of up to 256 previous characters

# # GPT 2 model :)
n_layer = 8
n_head = 6
n_embd = 384
dropout = 0.2



learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
# dtype = 'float16' # removes the warning:  [0/0] Not enough SMs to use max_autotune_gemm mode --> nope

out_dir =        f'out-enwik8-baseline-customgpt-{n_layer}-{n_head}-{n_embd}-{block_size}-{dropout}'
wandb_run_name = f'enwik8-baseline-customgpt-{n_layer}-{n_head}-{n_embd}-{block_size}-{dropout}'