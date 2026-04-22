import torch
import torch.nn as nn
from .utils import topk_accuracy

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_top1 = total_top5 = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bsz = y.size(0)
        t1, t5 = topk_accuracy(logits, y, ks=(1, 5))
        total_loss += loss.item() * bsz
        total_top1 += t1 * bsz
        total_top5 += t5 * bsz
        n += bsz

    return total_loss / n, total_top1 / n, total_top5 / n

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_top1 = total_top5 = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = y.size(0)
        t1, t5 = topk_accuracy(logits, y, ks=(1, 5))
        total_loss += loss.item() * bsz
        total_top1 += t1 * bsz
        total_top5 += t5 * bsz
        n += bsz

    return total_loss / n, total_top1 / n, total_top5 / n