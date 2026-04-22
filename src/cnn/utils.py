import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    accs = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        accs.append(100.0 * correct_k / targets.size(0))
    return tuple(accs)