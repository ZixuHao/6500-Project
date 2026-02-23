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