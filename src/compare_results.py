import argparse
import json
from pathlib import Path

import pandas as pd


def load_metric(path: str, label: str):
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    history = payload.get('history', [])
    best = max((row['val']['top1'] for row in history), default=payload.get('best_val_top1', None))
    last = history[-1]['val'] if history else {}
    return {
        'model': label,
        'top1': round(float(last.get('top1', best or 0.0)) * 100, 2),
        'top5': round(float(last.get('top5', 0.0)) * 100, 2),
        'val_loss': round(float(last.get('loss', 0.0)), 4),
        'epochs': len(history),
        'notes': 'advanced extension' if 'vit' in label.lower() else 'baseline',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-json', type=str, nargs='*', default=[])
    parser.add_argument('--baseline-labels', type=str, nargs='*', default=[])
    parser.add_argument('--vit-json', type=str, required=True)
    parser.add_argument('--vit-label', type=str, default='ViT-B/16 fine-tune')
    parser.add_argument('--output-csv', type=str, default='results/comparison_table.csv')
    args = parser.parse_args()

    rows = []
    for path, label in zip(args.baseline_json, args.baseline_labels):
        rows.append(load_metric(path, label))
    rows.append(load_metric(args.vit_json, args.vit_label))

    df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_markdown(index=False))
    print(f'Saved comparison table to {out_path}')


if __name__ == '__main__':
    main()
