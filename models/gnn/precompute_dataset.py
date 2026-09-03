import shutil

from models.gnn.dataset import get_gnn_datasets
import torch
import os

train_ds, val_ds, test_ds = get_gnn_datasets(num_samples=600, seed=42)

output_dir = "precomputed_dataset"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

for i, data in enumerate(train_ds):
    torch.save(data, f"{output_dir}/sample_train_{i}.pt")

for i, data in enumerate(val_ds):
    torch.save(data, f"{output_dir}/sample_val_{i}.pt")

for i, data in enumerate(test_ds):
    torch.save(data, f"{output_dir}/sample_test_{i}.pt")

print("Precomputation done: train/val/test splits created.")