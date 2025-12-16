"""
Data preparation script for train/val split.

This script loads the original data file, splits it into train and validation sets,
and saves them in a format compatible with main.py.

Usage:
    python data/prepare.py
"""

import os
import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    # Configuration
    input_file = 'nm-6-cleaned-maxlen-30.pt'
    train_ratio = 0.9
    seed = 1337
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Determine paths
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = data_dir
    
    # Handle input file path
    input_path = os.path.join(data_dir, input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading data from: {input_path}")
    data = torch.load(input_path, map_location='cpu')
    
    if not isinstance(data, list):
        raise ValueError(f"Expected data to be a list, got {type(data)}")
    
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
        
    data_list = []
    for sample in data:
        data_list.append(Data.from_dict(sample))
    
    # Shuffle data for random split
    indices = torch.randperm(total_samples).tolist()
    
    # Calculate split sizes
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Split data
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    
    # Save train and val splits
    train_output_path = os.path.join(output_dir, 'train.pt')
    val_output_path = os.path.join(output_dir, 'val.pt')
    
    print(f"Saving train split to: {train_output_path}")
    torch.save(train_data, train_output_path)
    
    print(f"Saving val split to: {val_output_path}")
    torch.save(val_data, val_output_path)
    
    print("Data preparation complete!")
    print(f"Train file: {train_output_path} ({len(train_data)} samples)")
    print(f"Val file: {val_output_path} ({len(val_data)} samples)")
