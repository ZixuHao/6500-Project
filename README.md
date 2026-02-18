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
We study a fine-grained image classification problem using the CUB-200-2011 dataset.
Given an RGB image of a bird, the goal is to predict its species label among 200 classes.

## Success Criteria
We evaluate performance using top-1 classification accuracy on the held-out test set.

## Scope
This project focuses on supervised classification using pretrained CNN backbones.
We do not address detection or segmentation in this semester.

## Modality
Vision (RGB images)