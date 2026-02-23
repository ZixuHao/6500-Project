import os
from typing import Optional, Dict, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_cub_metadata(data_root: str) -> pd.DataFrame:
    """
    Returns df with:
      img_id, rel_path, label(0..199), is_train(0/1)
    """
    images = pd.read_csv(os.path.join(data_root, "images.txt"),
                         sep=r"\s+", names=["img_id", "rel_path"])
    labels = pd.read_csv(os.path.join(data_root, "image_class_labels.txt"),
                         sep=r"\s+", names=["img_id", "label_1based"])
    split = pd.read_csv(os.path.join(data_root, "train_test_split.txt"),
                        sep=r"\s+", names=["img_id", "is_train"])

    df = images.merge(labels, on="img_id").merge(split, on="img_id")
    df["label"] = df["label_1based"] - 1
    df.drop(columns=["label_1based"], inplace=True)
    return df

def load_bboxes(data_root: str) -> Dict[int, Tuple[float, float, float, float]]:
    """
    bounding_boxes.txt format:
      img_id x y width height
    """
    bb = pd.read_csv(os.path.join(data_root, "bounding_boxes.txt"),
                     sep=r"\s+", names=["img_id", "x", "y", "w", "h"])
    return {int(r.img_id): (float(r.x), float(r.y), float(r.w), float(r.h)) for _, r in bb.iterrows()}

def crop_with_bbox(img: Image.Image, bbox, pad: float = 0.0) -> Image.Image:
    """
    bbox: (x, y, w, h) in pixel coordinates
    pad: proportion of bbox size added as padding on each side
    """
    x, y, w, h = bbox
    W, H = img.size

    px = pad * w
    py = pad * h

    x1 = max(0, int(round(x - px)))
    y1 = max(0, int(round(y - py)))
    x2 = min(W, int(round(x + w + px)))
    y2 = min(H, int(round(y + h + py)))

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

class CUBDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transform=None,
        use_bbox_crop: bool = False,
        bbox_pad: float = 0.0,
        bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.use_bbox_crop = use_bbox_crop
        self.bbox_pad = bbox_pad
        self.bboxes = bboxes if bboxes is not None else {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = int(row["img_id"])
        img_path = os.path.join(self.data_root, "images", row["rel_path"])
        img = Image.open(img_path).convert("RGB")

        if self.use_bbox_crop:
            bbox = self.bboxes.get(img_id, None)
            if bbox is not None:
                img = crop_with_bbox(img, bbox, pad=self.bbox_pad)

        y = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, y

def build_transforms(image_size: int):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    test_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, test_tf

def build_loaders(cfg):
    df = load_cub_metadata(cfg.data_root)
    train_df = df[df["is_train"] == 1].copy()
    test_df = df[df["is_train"] == 0].copy()

    bboxes = load_bboxes(cfg.data_root) if cfg.use_bbox_crop else None
    train_tf, test_tf = build_transforms(cfg.image_size)

    train_ds = CUBDataset(train_df, cfg.data_root, transform=train_tf,
                          use_bbox_crop=cfg.use_bbox_crop, bbox_pad=cfg.bbox_pad, bboxes=bboxes)
    test_ds  = CUBDataset(test_df, cfg.data_root, transform=test_tf,
                          use_bbox_crop=cfg.use_bbox_crop, bbox_pad=cfg.bbox_pad, bboxes=bboxes)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, test_loader