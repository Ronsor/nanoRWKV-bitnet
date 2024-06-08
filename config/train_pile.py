
out_dir = 'out-pile'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

# wandb config
#wandb_log = False
#wandb_project = 'shakespeare-char'
#wandb_run_name = 'mini-gpt'

dataset = 'pile_tokenized'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 1024 # context of up to 1024 tokens

# approximately match Eagle-1.5B configuration
n_layer = 24
n_head = 32
n_embd = 2048
dropout = 0.0

learning_rate = 0.1 # BitNet b1.58 requires a high LR. tune this if it sucks.
max_iters = 300_000_000
lr_decay_iters = max_iters
min_lr = learning_rate / 10 # learning_rate / 10 usually
beta2 = 0.99

warmup_iters = 100
compile = True
