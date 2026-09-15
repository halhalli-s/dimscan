"""Compare a predicted AI1 object mask against a hand-made RGB silhouette mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _bbox_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred_box = _bbox(pred)
    gt_box = _bbox(gt)
    if pred_box is None or gt_box is None:
        return {
            "pred_bbox_xyxy": pred_box,
            "gt_bbox_xyxy": gt_box,
            "width_error_px": None,
            "height_error_px": None,
            "top_error_px": None,
            "left_error_px": None,
            "right_error_px": None,
            "bottom_error_px": None,
        }
    pred_width = pred_box[2] - pred_box[0]
    pred_height = pred_box[3] - pred_box[1]
    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]
    return {
        "pred_bbox_xyxy": pred_box,
        "gt_bbox_xyxy": gt_box,
        "width_error_px": int(pred_width - gt_width),
        "height_error_px": int(pred_height - gt_height),
        "top_error_px": int(pred_box[1] - gt_box[1]),
        "left_error_px": int(pred_box[0] - gt_box[0]),
        "right_error_px": int(pred_box[2] - gt_box[2]),
        "bottom_error_px": int(pred_box[3] - gt_box[3]),
    }


def _overlay(rgb: Image.Image, pred: np.ndarray, gt: np.ndarray) -> Image.Image:
    base = np.asarray(rgb.convert("RGB"), dtype=np.uint8).copy()
    false_negative = gt & ~pred
    false_positive = pred & ~gt
    true_positive = pred & gt
    base[true_positive] = np.asarray(base[true_positive] * 0.45 + np.asarray([64, 220, 100]) * 0.55, dtype=np.uint8)
    base[false_negative] = np.asarray(base[false_negative] * 0.35 + np.asarray([255, 80, 80]) * 0.65, dtype=np.uint8)
    base[false_positive] = np.asarray(base[false_positive] * 0.35 + np.asarray([70, 140, 255]) * 0.65, dtype=np.uint8)
    return Image.fromarray(base, mode="RGB")


def evaluate(*, rgb_path: Path, pred_mask_path: Path, gt_mask_path: Path, out_dir: Path) -> dict[str, Any]:
    rgb = Image.open(rgb_path).convert("RGB")
    pred = _read_mask(pred_mask_path, rgb.size)
    gt = _read_mask(gt_mask_path, rgb.size)
    intersection = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    pred_count = int(np.count_nonzero(pred))
    gt_count = int(np.count_nonzero(gt))
    false_negative_count = int(np.count_nonzero(gt & ~pred))
    false_positive_count = int(np.count_nonzero(pred & ~gt))
    payload: dict[str, Any] = {
        "rgb_path": str(rgb_path),
        "pred_mask_path": str(pred_mask_path),
        "gt_mask_path": str(gt_mask_path),
        "rgb_size": [int(rgb.size[0]), int(rgb.size[1])],
        "pred_foreground_pixels": pred_count,
        "gt_foreground_pixels": gt_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "false_negative_pixels": false_negative_count,
        "false_positive_pixels": false_positive_count,
        "iou": float(intersection / union) if union else 0.0,
        "precision": float(intersection / pred_count) if pred_count else 0.0,
        "recall": float(intersection / gt_count) if gt_count else 0.0,
        **_bbox_metrics(pred, gt),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _overlay(rgb, pred, gt).save(out_dir / "mask_validation_overlay.png", format="PNG")
    Image.fromarray((gt & ~pred).astype(np.uint8) * 255, mode="L").save(out_dir / "false_negative_mask.png", format="PNG")
    Image.fromarray((pred & ~gt).astype(np.uint8) * 255, mode="L").save(out_dir / "false_positive_mask.png", format="PNG")
    (out_dir / "mask_validation_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--gt-mask", required=True, type=Path)
    parser.add_argument("--pred-mask", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("testing_development/ai1_mask_validation/outputs"))
    args = parser.parse_args()
    pred_mask = args.pred_mask or args.rgb.parent / "object_mask.png"
    metrics = evaluate(rgb_path=args.rgb, pred_mask_path=pred_mask, gt_mask_path=args.gt_mask, out_dir=args.out_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
