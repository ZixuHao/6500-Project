import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src.data import load_cub_metadata, load_bboxes, crop_with_bbox


def parse_args():
    parser = argparse.ArgumentParser(description="Classical baseline: HOG + Linear SVM on CUB-200-2011")
    parser.add_argument("--data-root", type=str, default="data/CUB_200_2011")
    parser.add_argument("--output-dir", type=str, default="outputs/classical_hog_svm")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--pixels-per-cell", type=int, default=16)
    parser.add_argument("--cells-per-block", type=int, default=2)
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--use-bbox-crop", action="store_true")
    parser.add_argument("--bbox-pad", type=float, default=0.05)
    parser.add_argument("--num-qualitative", type=int, default=8)
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_hog_feature(
    img: Image.Image,
    image_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> np.ndarray:
    # grayscale classical baseline
    img_gray = img.convert("L")
    arr = np.asarray(img_gray).astype(np.float32) / 255.0
    arr = resize(
        arr,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    )

    feat = hog(
        arr,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def build_feature_matrix(
    df: pd.DataFrame,
    data_root: str,
    image_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
    use_bbox_crop: bool = False,
    bbox_pad: float = 0.0,
):
    bboxes = load_bboxes(data_root) if use_bbox_crop else None

    features = []
    labels = []
    paths = []
    img_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting HOG"):
        img_id = int(row["img_id"])
        rel_path = row["rel_path"]
        y = int(row["label"])

        img_path = os.path.join(data_root, "images", rel_path)
        img = load_image(img_path)

        if use_bbox_crop and bboxes is not None and img_id in bboxes:
            img = crop_with_bbox(img, bboxes[img_id], pad=bbox_pad)

        feat = image_to_hog_feature(
            img=img,
            image_size=image_size,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )

        features.append(feat)
        labels.append(y)
        paths.append(rel_path)
        img_ids.append(img_id)

    X = np.stack(features, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, paths, img_ids


def topk_accuracy_from_scores(scores: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(scores, axis=1)[:, -k:]
    hits = np.any(topk == y_true[:, None], axis=1)
    return float(hits.mean() * 100.0)


def save_confusion_matrix(cm: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (HOG + Linear SVM)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_grid(
    df_subset: pd.DataFrame,
    data_root: str,
    save_path: str,
    title: str,
    n: int = 8,
):
    if len(df_subset) == 0:
        return

    df_vis = df_subset.head(n).copy()
    cols = min(4, len(df_vis))
    rows = int(np.ceil(len(df_vis) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes[:, None]

    axes = axes.flatten()

    for ax in axes[len(df_vis):]:
        ax.axis("off")

    for ax, (_, row) in zip(axes, df_vis.iterrows()):
        img_path = os.path.join(data_root, "images", row["rel_path"])
        img = load_image(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"true={row['true_label']}\npred={row['pred_label']}",
            fontsize=9
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cub_metadata(args.data_root)
    train_df = df[df["is_train"] == 1].copy().reset_index(drop=True)
    test_df = df[df["is_train"] == 0].copy().reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Test size:  {len(test_df)}")

    X_train, y_train, train_paths, train_img_ids = build_feature_matrix(
        train_df,
        data_root=args.data_root,
        image_size=args.image_size,
        orientations=args.orientations,
        pixels_per_cell=args.pixels_per_cell,
        cells_per_block=args.cells_per_block,
        use_bbox_crop=args.use_bbox_crop,
        bbox_pad=args.bbox_pad,
    )

    X_test, y_test, test_paths, test_img_ids = build_feature_matrix(
        test_df,
        data_root=args.data_root,
        image_size=args.image_size,
        orientations=args.orientations,
        pixels_per_cell=args.pixels_per_cell,
        cells_per_block=args.cells_per_block,
        use_bbox_crop=args.use_bbox_crop,
        bbox_pad=args.bbox_pad,
    )

    print("Fitting Linear SVM...")
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(C=args.c, max_iter=args.max_iter, dual="auto")
    )
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    scores = clf.decision_function(X_test)

    top1 = accuracy_score(y_test, y_pred) * 100.0
    top5 = topk_accuracy_from_scores(scores, y_test, k=5)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "method": "HOG + LinearSVM",
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "image_size": args.image_size,
        "orientations": args.orientations,
        "pixels_per_cell": args.pixels_per_cell,
        "cells_per_block": args.cells_per_block,
        "C": args.c,
        "max_iter": args.max_iter,
        "use_bbox_crop": bool(args.use_bbox_crop),
        "bbox_pad": args.bbox_pad,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "img_id": test_img_ids,
        "rel_path": test_paths,
        "true_label": y_test,
        "pred_label": y_pred,
        "correct": (y_test == y_pred),
    })
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    save_confusion_matrix(cm, str(output_dir / "confusion_matrix.png"))

    correct_df = pred_df[pred_df["correct"]].sample(
        n=min(args.num_qualitative, pred_df["correct"].sum()),
        random_state=42
    ) if pred_df["correct"].sum() > 0 else pred_df.iloc[:0]

    error_df = pred_df[~pred_df["correct"]].sample(
        n=min(args.num_qualitative, (~pred_df["correct"]).sum()),
        random_state=42
    ) if (~pred_df["correct"]).sum() > 0 else pred_df.iloc[:0]

    save_qualitative_grid(
        correct_df,
        data_root=args.data_root,
        save_path=str(output_dir / "qualitative_correct.png"),
        title="Correct Predictions: HOG + Linear SVM",
        n=args.num_qualitative,
    )

    save_qualitative_grid(
        error_df,
        data_root=args.data_root,
        save_path=str(output_dir / "qualitative_errors.png"),
        title="Failure Cases: HOG + Linear SVM",
        n=args.num_qualitative,
    )

    print("\n=== Classical Baseline Results ===")
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()