"""
Adapted from https://github.com/karpathy/nanoGPT.

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch_geometric.data import Data, Batch

from nn.models import Bandformer

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'bandforme_project'
wandb_run_name = 'bandformer' # 'run' + str(time.time())
# data
dataset = 'nm-6-cleaned-maxlen-30.pt'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 32 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
n_layer = 1
n_head = 1
n_embd = 192
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 100 # total number of training iterations
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 5e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# Configuration system: load from YAML and/or override from command line
from configurator import setup_config
config = setup_config(globals())
# -----------------------------------------------------------------------------

# reproducibility settings
seed = 1337 # base random seed (each DDP process gets seed + rank_offset)
deterministic = False # set to True for full reproducibility (may be slower, some ops still non-deterministic)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    # Ensure gradient_accumulation_steps is an integer
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
print(f"batch size: {batch_size}, gradient accumulation steps: {gradient_accumulation_steps}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Set random seeds for reproducibility
# Each process gets a different seed offset to ensure different random sequences
torch.manual_seed(seed + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed + seed_offset)  # sets seed for all CUDA devices
np.random.seed(seed + seed_offset)
random.seed(seed + seed_offset)

# Set deterministic behavior (optional, can slow down training)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: Some operations may still be non-deterministic even with these settings
    # Full reproducibility may require setting environment variable: CUBLAS_WORKSPACE_CONFIG=:4096:8
else:
    # Allow non-deterministic operations for better performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
_data_cache = {}

def get_batch(split):
    # Load from prepared train/val split files
    data_path = os.path.join('data', f'{split}.pt')
    cache_key = data_path
    
    if cache_key not in _data_cache:
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        _data_cache[cache_key] = data
    
    data_list = _data_cache[cache_key]
    split_size = len(data_list)
    
    # Randomly sample batch_size indices
    ix = torch.randint(0, split_size, (batch_size,))
    
    # Get batch samples
    batch_samples = [data_list[idx] for idx in ix.tolist()]
    
    batch_obj = Batch.from_data_list(batch_samples)
    
    # Efficiently extract and concatenate kpoints and bands
    # Each kpoints/bands tensor has shape (1, 128, 3) or (1, num_bands), so we cat along dim=0
    batch_obj.kpoints = torch.cat([data.kpoints for data in batch_samples], dim=0)
    batch_obj.bands = torch.cat([data.bands for data in batch_samples], dim=0)
    
    if device_type == 'cuda':
        batch_obj = batch_obj.pin_memory().to(device, non_blocking=True)
    else:
        batch_obj = batch_obj.to(device)
    
    del batch_samples
    return batch_obj


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init - Bandformer uses different parameters
model_args = dict(
    d_model=n_embd,
    enc_heads=n_head,
    enc_num_layers=n_layer,
    dec_heads=n_head,
    dec_num_layers=n_layer,
    dropout=dropout
)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new Bandformer model from scratch")
    model = Bandformer(**model_args)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    for k in ['d_model', 'enc_heads', 'enc_num_layers', 'dec_heads', 'dec_num_layers', 'dropout']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    model = Bandformer(**model_args)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer - use AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        maes = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = get_batch(split)
            with ctx:
                xr, xi = model(batch)
                target = batch.bands  # [batch_size, num_bands, num_kpoints]
                fft = torch.fft.rfft(target, norm='forward')
                yr, yi = fft.real, fft.imag
                
                loss1 = torch.nn.functional.mse_loss(xr, yr)
                loss2 = torch.nn.functional.mse_loss(xi, yi)
                loss = loss1 + loss2
            
            # Compute MAE outside autocast context (more efficient)
            # Convert to float32 for torch.complex (doesn't support bfloat16)
            with torch.no_grad():
                xfft = torch.complex(xr.float(), xi.float())
                x = torch.fft.irfft(xfft, dim=-1, norm='forward')
                mae = torch.nn.functional.l1_loss(x, target)
                
            losses[k] = loss.item()
            maes[k] = mae.item()
        out[f'{split}_loss'] = losses.mean()
        out[f'{split}_mae'] = maes.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
batch = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        metrics = estimate_loss()
        print(f"step {iter_num}: train loss {metrics['train_loss']:.4f}, train mae {metrics['train_mae']:.4f}, val loss {metrics['val_loss']:.4f}, val mae {metrics['val_mae']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": metrics['train_loss'],
                "train/mae": metrics['train_mae'],
                "val/loss": metrics['val_loss'],
                "val/mae": metrics['val_mae'],
                "lr": lr,
            })
        if metrics['val_loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = metrics['val_loss']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    mae = None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            xr, xi = model(batch)
            target = batch.bands  # [batch_size, num_bands, num_kpoints]
            fft = torch.fft.rfft(target, norm='forward')
            yr, yi = fft.real, fft.imag
            
            loss1 = torch.nn.functional.mse_loss(xr, yr)
            loss2 = torch.nn.functional.mse_loss(xi, yi)
            loss = loss1 + loss2
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        
        # Compute MAE only when needed for logging (outside autocast, on last micro_step)
        if iter_num % log_interval == 0 and master_process and micro_step == gradient_accumulation_steps - 1:
            with torch.no_grad():
                # Convert to float32 for torch.complex (doesn't support bfloat16)
                xfft = torch.complex(xr.float(), xi.float())
                x = torch.fft.irfft(xfft, dim=-1, norm='forward')
                mae = torch.nn.functional.l1_loss(x, target)
        
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        batch = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss and mae as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        maef = mae.item() if mae is not None else 0.0
        print(f"iter {iter_num}: loss {lossf:.4f}, mae {maef:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
