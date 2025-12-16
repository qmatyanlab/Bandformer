"""
Evaluation script for Bandformer.
Loads the best checkpoint from 'out/ckpt.pt' and evaluates on the validation set.
"""

import os
import torch
from torch_geometric.data import Batch
from nn.models import Bandformer
from contextlib import nullcontext

# -----------------------------------------------------------------------------
# Configuration
out_dir = 'out'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
eval_iters = 200 # Number of batches to evaluate
# -----------------------------------------------------------------------------

def main():
    # Setup device
    print(f"Using device: {device}")
    torch.manual_seed(1337)

    # Load checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    
    print(f"Model args: {model_args}")

    # Initialize model
    print("Initializing model...")
    model = Bandformer(**model_args)
    state_dict = checkpoint['model']

    # Fix state dict keys (remove _orig_mod prefix if present)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Data loading helper
    # We use a simple cache to avoid reloading the large pt file
    _data_cache = {}

    def get_batch(split):
        # Load from prepared train/val split files
        data_path = os.path.join('data', f'{split}.pt')
        cache_key = data_path
        
        if cache_key not in _data_cache:
            print(f"Loading data from {data_path}...")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            data = torch.load(data_path, map_location='cpu', weights_only=False)
            _data_cache[cache_key] = data
        
        data_list = _data_cache[cache_key]
        split_size = len(data_list)
        
        # Randomly sample batch_size indices
        ix = torch.randint(0, split_size, (batch_size,))
        batch_samples = [data_list[idx] for idx in ix.tolist()]
        
        batch_obj = Batch.from_data_list(batch_samples)
        # Efficiently extract and concatenate kpoints and bands
        batch_obj.kpoints = torch.cat([data.kpoints for data in batch_samples], dim=0)
        batch_obj.bands = torch.cat([data.bands for data in batch_samples], dim=0)
        
        if 'cuda' in device:
            batch_obj = batch_obj.pin_memory().to(device, non_blocking=True)
        else:
            batch_obj = batch_obj.to(device)
        
        return batch_obj

    # Evaluation loop
    print(f"Evaluating on 'val' split for {eval_iters} batches...")

    losses = []
    maes = []

    # Context for mixed precision
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    with torch.no_grad():
        for k in range(eval_iters):
            try:
                batch = get_batch('val')
            except FileNotFoundError as e:
                print(e)
                return

            with ctx:
                xr, xi = model(batch)
                target = batch.bands
                fft = torch.fft.rfft(target, norm='forward')
                yr, yi = fft.real, fft.imag
                
                loss1 = torch.nn.functional.mse_loss(xr, yr)
                loss2 = torch.nn.functional.mse_loss(xi, yi)
                loss = loss1 + loss2
            
            # Compute MAE
            # Convert to float32 for torch.complex (doesn't support bfloat16)
            xfft = torch.complex(xr.float(), xi.float())
            x = torch.fft.irfft(xfft, dim=-1, norm='forward')
            mae = torch.nn.functional.l1_loss(x, target)
            
            losses.append(loss.item())
            maes.append(mae.item())
            
            if (k + 1) % 20 == 0:
                print(f"Processed {k + 1}/{eval_iters} batches...")

    avg_loss = sum(losses) / len(losses)
    avg_mae = sum(maes) / len(maes)

    print("-" * 40)
    print(f"Validation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print("-" * 40)

if __name__ == '__main__':
    main()
