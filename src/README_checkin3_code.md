# Check-In 3 Code Additions

This folder contains a minimal advanced extension for the CUB bird classification project.

## Files
- `vit_extension.py`: fine-tune a pretrained ViT-B/16 on your CUB train/test split.
- `compare_results.py`: merge baseline metrics and ViT metrics into one comparison table.

## Expected dataset layout
Use ImageFolder format:

```text
data/CUB_200_2011/
  train/
    001.Black_footed_Albatross/
      img1.jpg
      ...
  test/
    001.Black_footed_Albatross/
      imgA.jpg
      ...
```

## Quick smoke test
```bash
python vit_extension.py \
  --data-root data/CUB_200_2011 \
  --output-dir results/vit_smoke \
  --epochs 1 \
  --batch-size 8 \
  --subset-train 256 \
  --subset-val 128
```

## Main run
```bash
python vit_extension.py \
  --data-root data/CUB_200_2011 \
  --output-dir results/vit_full \
  --epochs 5 \
  --batch-size 16 \
  --lr 3e-4
```

## Ablation run: freeze backbone
```bash
python vit_extension.py \
  --data-root data/CUB_200_2011 \
  --output-dir results/vit_frozen \
  --epochs 5 \
  --batch-size 16 \
  --freeze-backbone
```

## Build comparison table
Suppose you already have baseline result JSON files from earlier check-ins:

```bash
python compare_results.py \
  --baseline-json results/hog_svm/metrics.json results/resnet18/metrics.json \
  --baseline-labels "HOG+SVM" "ResNet18 baseline" \
  --vit-json results/vit_full/metrics.json \
  --vit-label "ViT-B/16 fine-tune" \
  --output-csv results/comparison_table.csv
```

## What this gives you for the report
- Advanced extension: ViT-B/16 fine-tuning.
- Fair comparison: same split, same top-1 / top-5 metrics.
- Ablation: frozen backbone vs full fine-tuning.
- Failure analysis artifact: `mistakes_preview.json`.
