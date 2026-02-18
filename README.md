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