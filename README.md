# 6500-Project

## Dataset

### Name
CUB-200-2011 (Caltech-UCSD Birds 200)

### Source
Official website:
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

### Size

- 11,788 images

- 200 bird species

- ~60 images per class

### License
For research and educational use.

## Download Instructions

1. Visit the official website above

2. Download CUB_200_2011.tgz

3. Extract:

```{bash}
tar -xvf CUB_200_2011.tgz
```

4. Place the folder in:

```{bash}
project_root/data/CUB_200_2011/
```

Expected structure:

```
CUB_200_2011/
├── images/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
├── classes.txt
└── bounding_boxes.txt
```

## Task Definition
This project studies fine-grained visual recognition using CUB-200-2011.

We explore multiple computer vision formulations:

1. Image Classification
Predict bird species (200-way classification).

2. Object Localization (Bounding Box Prediction)
Use provided bounding boxes to study localization performance and IoU-based evaluation.

3. Classification with Localization Supervision
Compare full-image training vs cropped-to-bounding-box training.

## Success Criteria

### Classification

- Top-1 Accuracy

- Top-5 Accuracy

- Per-class accuracy

- Confusion matrix

### Localization

- IoU (Intersection over Union)

- Localization accuracy (IoU > 0.5)

## Scope
This project covers the following course concepts:

- CNN fundamentals (convolution, receptive field, pooling)

- Modern architectures (ResNet, DenseNet)

- Transfer learning (frozen vs fine-tuned backbones)

- Bounding box evaluation (IoU)

- Fine-grained classification challenges

- Error analysis and class confusion patterns

## Modality
Vision (RGB images)

## Baselines

---

### 1. Classical Baseline: HOG + Linear SVM

**Feature extractor:** HOG (Histogram of Oriented Gradients)  
**Classifier:** Linear SVM  

#### Configuration

- Image size: **128 × 128**
- Orientations: **9**
- Pixels per cell: **16 × 16**
- Cells per block: **2 × 2**

SVM:
- C: **1.0**
- Max iterations: **5000**

#### Dataset

- Train size: **5,994**
- Test size: **5,794**
- Bounding box crop: **Disabled**

#### Results

- **Top-1 Accuracy:** **2.11%**
- **Top-5 Accuracy:** **8.23%**

#### Analysis

- Performance is extremely low, only slightly above random guessing (~0.5% for 200 classes).
- HOG captures only local edge/gradient information and ignores color and high-level semantics.
- Fine-grained classification (bird species) requires subtle texture and shape differences that handcrafted features fail to model.
- Linear SVM further limits representation power compared to deep neural networks.

---

### 2. Deep Learning Baseline: ResNet-18 (ImageNet pretrained)

Backbone: torchvision resnet18  
Pretrained on ImageNet  
Final FC layer replaced with 200-way classifier  
End-to-end fine-tuning  

#### Training Setup

- Input resolution: **224 × 224**
- Data augmentation:
  - RandomResizedCrop (scale 0.7–1.0)
  - RandomHorizontalFlip
- Normalization: ImageNet mean/std
- Batch size: *(your cfg.batch_size, e.g. 64)*
- Optimizer: *(fill based on your config)*
- Epochs: **10**
- Bounding box crop: **Disabled**
- Training time (Colab GPU): ~2.8 minutes

#### Results

- Train Top-1: **81.95%**
- Test Top-1: **58.18%**
- Test Top-5: **85.50%**

**Best Top-1 Accuracy: 58.18%**

---

### Comparison Summary

| Method | Top-1 | Top-5 |
|--------|------|------|
| HOG + SVM | 2.11% | 8.23% |
| ResNet-18 | 58.18% | 85.50% |

#### Key Insight

- Deep learning dramatically outperforms classical methods.
- Learned hierarchical features are essential for fine-grained recognition.
- Classical pipelines fail due to lack of semantic understanding and invariance.

---

### Reproduce

```{bash}
python -m src.classical_baseline
python -m src.main
```
