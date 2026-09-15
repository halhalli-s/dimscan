# AI1 Mask Validation

Small offline helper for comparing a production `object_mask.png` against a hand-made ground-truth silhouette.

Example:

```bash
python3 testing_development/ai1_mask_validation/evaluate_mask.py \
  --rgb datasets/single/jobs/<job>/view_01/rgb.png \
  --pred-mask datasets/single/jobs/<job>/view_01/object_mask.png \
  --gt-mask /path/to/hand_mask.png \
  --out-dir testing_development/ai1_mask_validation/outputs/<job>
```

The tool writes IoU, precision, recall, false-negative/false-positive counts, bbox extent errors, and overlays.
