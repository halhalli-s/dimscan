# Object Axis Span Validation

Isolated hardware validation tool for checking raw RGB-camera X/Y/Z spans from
the clean object cloud. This does not modify production capture, segmentation,
geometry, features, AI2, GT, SKU policy, or the website.

## Run

From the project root:

```bash
python3 testing_development/object_axis_span_validation/capture_object_spans.py
```

Place one rigid object or plant inside the already calibrated production ROI
before running the command. A rectangular box or ruler-backed object is best for
manual validation because the X/Y/Z spans can be checked directly.

Start the same Python environment used for production capture first, for example:

```bash
source .venv/bin/activate
```

If the YOLOE model is not provided through `DIMSCAN_YOLOE_MODEL_PATH` or config,
pass it explicitly with `--model`.

Optional arguments:

```bash
python3 testing_development/object_axis_span_validation/capture_object_spans.py \
  --output-dir testing_development/object_axis_span_validation/outputs \
  --warmup-frames 5 \
  --model /absolute/path/to/yoloe.pt
```

Debug modes:

```bash
python3 testing_development/object_axis_span_validation/capture_object_spans.py \
  --mask /absolute/path/to/rgb_shape_mask.png

python3 testing_development/object_axis_span_validation/capture_object_spans.py \
  --no-segmentation
```

`--mask` must already match the RGB image shape. The tool fails on mismatched
masks instead of resizing them.

## Outputs

Each run writes a timestamped folder under `outputs/` containing:

- `rgb.png`
- `depth_raw.npy`
- `depth_aligned_to_rgb.npy`
- `object_mask.png`
- `object_mask_overlay_rgb.png`
- `depth_aligned_visualization.png`
- `valid_aligned_depth_overlay_rgb.png`
- `object_cloud_before_roi.ply`
- `object_cloud_roi_filtered.ply`
- `axis_span_report.json`
- `axis_span_report.txt`

The filtered cloud is the cloud used for span calculation.

## Coordinate Convention

The tool reports raw camera axes only:

- `X`: `(col - cx) * z / fx`, positive toward increasing image columns
- `Y`: `(row - cy) * z / fy`, positive toward increasing image rows
- `Z`: aligned depth converted to meters, positive away from the RGB camera

It reports:

- X min/max/span
- Y min/max/span
- Z min/max/span
- meters and inches using `39.3701` inches per meter

It does not assign semantic length, width, or height.

## Manual Checks

Run several captures while moving the object inside the ROI:

- Move the object right in the RGB image: reported X values should increase.
- Move the object down in the RGB image: reported Y values should increase under the current deprojection equation.
- Move the object farther from the camera: reported Z values should increase.

For dimensional validation, place a rigid box or ruler-visible object in the ROI
and compare the reported X/Y/Z spans against physical measurements. These are
camera-axis spans only, not package semantic length/width/height.

## Failure Reports

When a failure occurs after the output folder is created, the tool writes:

- `axis_span_report.json`
- `axis_span_report.txt`

The JSON contains `status: "failed"` and the failure reason.
