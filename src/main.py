import time
import torch
import torch.nn as nn

from .config import Config
from .utils import set_seed, ensure_dir
from .data import build_loaders
from .models import build_model
from .engine import train_one_epoch, eval_one_epoch

def main():
    cfg = Config()

    # device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    set_seed(cfg.seed)
    ensure_dir(cfg.ckpt_dir)

    train_loader, test_loader = build_loaders(cfg)
    model = build_model(cfg).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_top1 = 0.0
    start = time.time()

    ckpt_path = f"{cfg.ckpt_dir}/{cfg.run_name}.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_t1, tr_t5 = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        te_loss, te_t1, te_t5 = eval_one_epoch(model, test_loader, criterion, cfg.device)

        if te_t1 > best_top1:
            best_top1 = te_t1
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

        print(
            f"[{cfg.run_name}] Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {tr_loss:.4f} top1 {tr_t1:.2f}% top5 {tr_t5:.2f}% | "
            f"test loss {te_loss:.4f} top1 {te_t1:.2f}% top5 {te_t5:.2f}% | "
            f"best {best_top1:.2f}%"
        )

    print(f"Done in {(time.time()-start)/60:.1f} min. Best top1: {best_top1:.2f}%")
    print(f"Saved best checkpoint to: {ckpt_path}")

if __name__ == "__main__":
    main()