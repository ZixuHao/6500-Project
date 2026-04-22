import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ViT_B_16_Weights


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunConfig:
    data_root: str
    output_dir: str
    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    image_size: int = 224
    num_workers: int = 2
    seed: int = 42
    freeze_backbone: bool = False
    subset_train: int = 0
    subset_val: int = 0
    checkpoint_dir: str = ""
    resume: bool = True
    save_every: int = 1
    save_optimizer: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    weights = ViT_B_16_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def maybe_subset(dataset, n: int):
    if n and n < len(dataset):
        indices = list(range(len(dataset)))[:n]
        return torch.utils.data.Subset(dataset, indices)
    return dataset


def load_datasets(data_root: str, image_size: int, subset_train: int = 0, subset_val: int = 0):
    train_tf, eval_tf = build_transforms(image_size)
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "test"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected ImageFolder structure at {train_dir} and {val_dir}. "
            "Please prepare CUB split as data/CUB_200_2011/train/<class>/img.jpg and test/<class>/img.jpg"
        )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    train_ds = maybe_subset(train_ds, subset_train)
    val_ds = maybe_subset(val_ds, subset_val)
    return train_ds, val_ds


def create_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    weights = ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> Dict[str, float]:
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = {}
    for k in ks:
        out[f"top{k}"] = correct[:k].reshape(-1).float().sum(0).item() / targets.size(0)
    return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        metrics = topk_accuracy(logits.detach(), labels, ks=(1, 5))
        total_loss += loss.item() * bs
        top1_sum += metrics["top1"] * bs
        top5_sum += metrics["top5"] * bs
        total += bs

    return {
        "loss": total_loss / max(total, 1),
        "top1": top1_sum / max(total, 1),
        "top5": top5_sum / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names: List[str], save_mistakes_to: str = None):
    model.eval()
    total_loss = 0.0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    mistakes = []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        bs = labels.size(0)
        metrics = topk_accuracy(logits, labels, ks=(1, 5))

        total_loss += loss.item() * bs
        top1_sum += metrics["top1"] * bs
        top5_sum += metrics["top5"] * bs
        total += bs

        if save_mistakes_to is not None:
            top5_idx = probs.topk(5, dim=1).indices.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            for i in range(bs):
                if preds_cpu[i] != labels_cpu[i] and len(mistakes) < 50:
                    mistakes.append({
                        "batch": batch_idx,
                        "index_in_batch": i,
                        "true_class": class_names[labels_cpu[i]],
                        "pred_class": class_names[preds_cpu[i]],
                        "top5": [class_names[j] for j in top5_idx[i]],
                    })

    metrics = {
        "loss": total_loss / max(total, 1),
        "top1": top1_sum / max(total, 1),
        "top5": top5_sum / max(total, 1),
    }
    if save_mistakes_to is not None:
        with open(save_mistakes_to, "w", encoding="utf-8") as f:
            json.dump(mistakes, f, indent=2)
    return metrics


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_checkpoint_dir(cfg: RunConfig) -> str:
    if cfg.checkpoint_dir.strip():
        ckpt_dir = cfg.checkpoint_dir
    else:
        ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    return ckpt_dir


def get_paths(cfg: RunConfig) -> Dict[str, str]:
    ckpt_dir = resolve_checkpoint_dir(cfg)
    return {
        "checkpoint_dir": ckpt_dir,
        "last_ckpt": os.path.join(ckpt_dir, "last_checkpoint.pt"),
        "best_ckpt": os.path.join(ckpt_dir, "best_model.pt"),
        "history_json": os.path.join(cfg.output_dir, "metrics.json"),
        "mistakes_json": os.path.join(cfg.output_dir, "mistakes_preview.json"),
        "run_config_json": os.path.join(cfg.output_dir, "run_config.json"),
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_top1: float,
    history: List[dict],
    cfg: RunConfig,
    save_optimizer: bool = True,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_top1": best_val_top1,
        "history": history,
        "config": asdict(cfg),
    }
    if save_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def try_resume(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    resume: bool = True,
) -> Tuple[int, float, List[dict]]:
    if (not resume) or (not os.path.exists(ckpt_path)):
        return 0, -1.0, []

    print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_top1 = float(checkpoint.get("best_val_top1", -1.0))
    history = checkpoint.get("history", [])

    print(f"Resume start_epoch={start_epoch}, best_val_top1={best_val_top1:.4f}")
    return start_epoch, best_val_top1, history


def copy_if_different(src: str, dst: str) -> None:
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Check-In 3: ViT extension for CUB bird classification")
    parser.add_argument("--data-root", type=str, default="data/CUB_200_2011")
    parser.add_argument("--output-dir", type=str, default="results/vit_extension")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Directory for resume/best checkpoints. Put this in Google Drive for Colab persistence.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true", help="Ablation: train only the classification head")
    parser.add_argument("--subset-train", type=int, default=0, help="Use first N training images for quick smoke test")
    parser.add_argument("--subset-val", type=int, default=0, help="Use first N val images for quick smoke test")
    parser.add_argument("--save-every", type=int, default=1, help="Save last checkpoint every N epochs")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch even if a checkpoint exists")
    parser.add_argument("--no-save-optimizer", action="store_true", help="Do not store optimizer state in checkpoints")
    args = parser.parse_args()

    cfg = RunConfig(
        **vars(args),
        resume=(not args.no_resume),
        save_optimizer=(not args.no_save_optimizer),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # Remove helper-only argparse fields already converted into RunConfig fields
    cfg.resume = not args.no_resume
    cfg.save_optimizer = not args.no_save_optimizer

    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    paths = get_paths(cfg)
    save_json(paths["run_config_json"], asdict(cfg))

    train_ds, val_ds = load_datasets(cfg.data_root, cfg.image_size, cfg.subset_train, cfg.subset_val)
    class_names = train_ds.dataset.classes if isinstance(train_ds, torch.utils.data.Subset) else train_ds.classes

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    model = create_model(num_classes=len(class_names), freeze_backbone=cfg.freeze_backbone).to(cfg.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_val_top1, history = try_resume(
        paths["last_ckpt"],
        model,
        optimizer,
        cfg.device,
        resume=cfg.resume,
    )

    if start_epoch >= cfg.epochs:
        print(
            f"Checkpoint already completed epoch {start_epoch - 1}. "
            f"Requested total epochs={cfg.epochs}, so no further training is needed."
        )
    else:
        for epoch in range(start_epoch, cfg.epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                cfg.device,
                class_names,
                save_mistakes_to=paths["mistakes_json"] if epoch == cfg.epochs - 1 else None,
            )
            record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
            history.append(record)
            print(json.dumps(record))

            if (epoch + 1) % max(cfg.save_every, 1) == 0:
                save_checkpoint(
                    paths["last_ckpt"],
                    model,
                    optimizer,
                    epoch,
                    best_val_top1,
                    history,
                    cfg,
                    save_optimizer=cfg.save_optimizer,
                )

            if val_metrics["top1"] > best_val_top1:
                best_val_top1 = val_metrics["top1"]
                save_checkpoint(
                    paths["best_ckpt"],
                    model,
                    optimizer,
                    epoch,
                    best_val_top1,
                    history,
                    cfg,
                    save_optimizer=cfg.save_optimizer,
                )
                # Keep a plain state_dict copy in output_dir for compatibility with earlier workflow
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model_state_dict.pt"))
                print(f"New best checkpoint saved. top1={best_val_top1:.4f}")

        # Final save even if epochs is not divisible by save_every
        save_checkpoint(
            paths["last_ckpt"],
            model,
            optimizer,
            cfg.epochs - 1,
            best_val_top1,
            history,
            cfg,
            save_optimizer=cfg.save_optimizer,
        )

    summary = {
        "config": asdict(cfg),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_classes": len(class_names),
        "best_val_top1": best_val_top1,
        "history": history,
        "checkpoint_dir": paths["checkpoint_dir"],
        "last_checkpoint": paths["last_ckpt"],
        "best_checkpoint": paths["best_ckpt"],
    }
    save_json(paths["history_json"], summary)
    print(f"Saved metrics to {paths['history_json']}")
    print(f"Last checkpoint: {paths['last_ckpt']}")
    print(f"Best checkpoint: {paths['best_ckpt']}")


if __name__ == "__main__":
    main()
