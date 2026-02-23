import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes: int, finetune_backbone: bool):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not finetune_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model

def build_model(cfg):

    return build_resnet18(cfg.num_classes, cfg.finetune_backbone)