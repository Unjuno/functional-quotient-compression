# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 600000
lr_decay_iters = 600000

eval_interval = 1000
eval_iters = 200
log_interval = 10

weight_decay = 1e-1
