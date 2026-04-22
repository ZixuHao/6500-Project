from dataclasses import dataclass

@dataclass
class Config:
    # data
    data_root: str = "data/CUB_200_2011"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4

    # training
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"

    # experiments
    finetune_backbone: bool = False
    use_bbox_crop: bool = False
    bbox_pad: float = 0.05

    # bookkeeping
    num_classes: int = 200
    ckpt_dir: str = "checkpoints"
    run_name: str = "resnet18_baseline"