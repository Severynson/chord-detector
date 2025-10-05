import numpy as np
import torch
from torch.utils.data import Dataset

def compute_class_weights(dset: Dataset, num_classes: int) -> torch.Tensor:
    # quick pass to count labels
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(min(len(dset), 2000)):  # cap to speed up
        _, y = dset[i]
        vals, c = np.unique(y.numpy(), return_counts=True)
        counts[vals] += c
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    w = torch.tensor(weights, dtype=torch.float32)
    return w

def accuracy_per_frame(logits: torch.Tensor, y: torch.Tensor) -> float:
    # logits: (B,T,C), y: (B,T)
    pred = logits.argmax(dim=-1)
    correct = (pred == y).float().sum().item()
    total = y.numel()
    return correct / max(total, 1)
