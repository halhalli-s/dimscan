# DimScan

DimScan is an end-to-end RGB-D perception and machine-learning system for
measuring irregular, deformable objects — such as nursery plants — in 3D
and predicting the shipping-box length, width, and height they should be
packed into.

## Why irregular objects are not rigid boxes

Standard dimensioning systems assume a rigid, closed object: a fixed
bounding box computed once from a single depth capture is enough.
Irregular, deformable objects like potted plants break that assumption.
A potted plant is an irregular, partially
transparent, non-rigid cluster of foliage around a rigid pot — leaves
overhang the pot at unpredictable angles, gaps between leaves let the
background show through, and the same plant can present a different
silhouette from view to view. Naively bounding "everything above the
table" produces boxes distorted by stray leaves, nearby fixtures, or
background clutter that RANSAC and a bounding box alone cannot tell apart
from the plant itself. DimScan's geometry pipeline exists specifically to
recover a trustworthy 3D shape for that kind of object before any
prediction happens.

## Pipeline overview

Each capture flows through the same sequence of stages, whether it is
being collected for training data or scanned for a live prediction:

1. **Orbbec RGB-D capture.** An Orbbec Gemini camera captures aligned RGB
   and depth frames along with device/stream intrinsics
   (`capture/orbbec_camera.py`, `capture/pointcloud.py`).
2. **Metric point-cloud generation.** Aligned depth is projected into the
   RGB camera frame using the calibrated intrinsics and a validated
   depth-to-meters scale, producing a full-scene point cloud in real-world
   units (`pipeline/object_extraction.py:full_scene_cloud_from_aligned_depth`).
3. **Calibrated 3D ROI.** A fixed, calibrated region of interest (depth
   range, lateral half-width, height bounds) crops the scene to the
   working volume in front of the camera, discarding points that can't
   physically belong to the object being scanned.
4. **Table/object isolation.** A RANSAC plane fit finds the table surface
   and splits the ROI into "table" and "above/below table" point sets.
   A short-lived, strictly validated per-camera-geometry cache lets
   repeated captures from the same fixed setup skip a full RANSAC refit
   when the cached plane still explains the new scene well.
5. **Voxelization and clustering.** The remaining points are voxelized and
   grouped into connected components (26-connected voxel neighborhoods),
   turning a point cloud into a small number of spatial clusters.
6. **Dominant-object extraction.** Clusters are scored by point count,
   penalized by lateral distance from the ROI center, and implausible
   clusters (outside the expected object zone, or with an unrealistic
   physical span) are rejected. The highest-scoring cluster becomes the
   primary object.
7. **Nearby-fragment recovery.** Smaller clusters close enough to the
   primary cluster's bounding box (and small enough relative to it) are
   merged back in, so a leaf or thin stem segmented into its own cluster
   isn't lost from the final shape.
8. **`object_cloud.ply` — the authoritative geometry.** The merged result
   is written once per view as `object_cloud.ply` and is the single
   source of truth for that view's geometry. Every debug artifact
   (rejected ROI points, the table plane, rejected clusters, a robust
   width-line visualization) is written alongside it for inspection, but
   none of them feed back into what `object_cloud.ply` contains.

Every stage above is geometry-only. No learned model touches
`object_cloud.ply`.

## AI1 (YOLOE): support and debug only

AI1 is a YOLOE-based segmentation model (`segmentation/yoloe_segmenter.py`)
used to detect and mask candidate plant/pot regions in the RGB frame. It
is used for **validation and debugging** — sanity-checking that the
geometry pipeline's dominant cluster overlaps a plausible plant/pot
detection, and producing human-inspectable masks. AI1 output is explicitly
flagged as non-authoritative throughout the codebase
(`ai1_role: "validation_only"`, `ai1_used_for_object_extraction: False` in
`pipeline/object_extraction.py`): AI1 can never resegment, veto, or modify
`object_cloud.ply`. If AI1 disagrees with the geometry pipeline, the
geometry pipeline still wins — AI1's disagreement is only surfaced as a
debug signal.

## Features and AI2

Once `object_cloud.ply` exists for a job, deterministic geometry and shape
features are extracted from it (`features/extractor.py`) — spans,
robust percentile-based extents, point counts, and (for group jobs)
composition features describing how multiple items relate to each other.
These features, plus safe SKU metadata, are the input to **AI2**: a
scikit-learn `RandomForestRegressor` pipeline (`ai2/train.py`,
`ai2/predict.py`) trained to predict the shipping box's length, width,
and height.

**Ground truth** is the box an operator actually used to ship the plant,
entered by hand after packing (`pipeline/ground_truth.py`). It is the
label AI2 is trained against — never a prediction, never inferred. A
prediction is written to its own file (`ai2_prediction.json`); it is
compared against ground truth for reporting, but the write path for
predictions and the write path for ground truth are separate, and a
prediction can never overwrite a recorded ground-truth box.

**Single-plant and group (multi-item) jobs are separate workflows and
separate models.** They have distinct dataset roots
(`datasets/single/jobs`, `datasets/group/jobs`), distinct feature schema
versions, and distinct trained AI2 models
(`models/packing/single/ai2_v1`, `models/packing/group/ai2_v1`). Group
jobs get additional deterministic composition features layered on top of
the same per-item geometry features.

**Prediction reuses the same geometry pipeline as collection.** A
prediction job runs through the identical capture → ROI → table removal →
clustering → `object_cloud.ply` → feature-extraction sequence used during
data collection, then loads the appropriate trained AI2 model for that
job type and scores it. There is no separate, lighter-weight "prediction
mode" geometry path to keep in sync with the training path — the pipeline
run in production has already been used to build training data.

## Current measured results

- Same-plant measurement variation: approximately **0.07–0.09 in** across
  repeated captures of the same plant.
- Collection-ready latency improved from roughly **20–25 s/view** to
  approximately **3.4–3.9 s/view** (largely from the table-plane cache
  avoiding a redundant full-resolution RANSAC refit per view).
- **Single-plant prediction works end-to-end**, from capture through a
  trained AI2 model to a predicted box.
- **Group data collection is implemented**; the group training dataset is
  still being expanded, so the group AI2 model should be treated as an
  early baseline rather than a validated production model.

## Architecture overview

```
capture/        Orbbec camera adapter, RGB-D → point-cloud helpers
geometry/       Scene/plant/pot geometry dataclasses and validated schemas
pipeline/       Job orchestration: capture, object extraction, features,
                ground truth, prediction, arrangement, session state
segmentation/   AI1 (YOLOE) segmentation — validation/debug only
features/       Deterministic feature extraction + schema/validators
ai2/            AI2 dataset construction, training, and inference
packing/        Rule-based box-fit suggestion from combined features
metadata/       SKU/item metadata parsing and lookup
app/            Flask app: routes, static UI, dev server entrypoint
scripts/        CLI entrypoints (e.g. AI2 training)
tests/          Automated tests (no hardware required)
testing_development/  Ad-hoc validation scratch work for pipeline stages
experiments/    Standalone experiment/collection scripts
utils/          Shared I/O, path, and logging helpers
```

## Key engineering decisions

- **Geometry is the source of truth, ML is downstream of it.** AI1 never
  controls `object_cloud.ply`; AI2 only ever consumes features derived
  from it. This keeps the expensive-to-validate part of the system (3D
  shape) decoupled from model retraining.
- **A calibrated ROI plus RANSAC plane removal, not a general-purpose
  scene segmenter, isolates the object.** For a fixed capture rig this is
  far more predictable and debuggable than relying on a learned
  segmenter to find "the object" in an arbitrary scene.
- **Dominant-cluster selection with bounded fragment merging**, instead
  of a single largest-connected-component rule, so a plant's disjoint
  leaves/stems are recovered without also pulling in unrelated background
  clusters.
- **A strictly validated, process-local table-plane cache**, only reused
  when its cached plane still explains the current scene's inlier ratio,
  dominant-side ratio, and orientation — otherwise it transparently falls
  back to a full RANSAC refit. This is what took per-view latency from
  ~20–25 s to ~3.4–3.9 s without weakening the plane-fit guarantees.
- **Single vs. group jobs are kept as separate schemas and separate
  models** rather than one shared model, since group composition
  features are fundamentally different from single-item geometry
  features.
- **Predictions are append-only artifacts, never label writes.** Ground
  truth and predictions live in different files with different write
  paths, so there is no code path by which a prediction can silently
  become a training label.

## Project structure

```
dimscan/
├── app/            Flask app (routes, templates, static UI, server entrypoint)
├── ai2/            AI2 dataset, train, predict
├── capture/         Orbbec camera + point-cloud capture
├── features/        Feature extraction, schema, validators
├── geometry/         Geometry dataclasses/schemas
├── metadata/         SKU/item metadata parsing and lookup (catalog data itself is not public)
├── packing/          Rule-based box-fit suggestion
├── pipeline/         Job orchestration and the object-extraction pipeline
├── scripts/          CLI entrypoints
├── segmentation/      AI1 / YOLOE segmentation (debug/validation only)
├── tests/             Automated tests
├── testing_development/  Scratch validation for individual pipeline stages
├── experiments/        Standalone capture/collection experiment scripts
├── utils/            Shared I/O/path/logging helpers
├── requirements.txt
├── .env.example
└── README.md
```

`datasets/`, `prediction_data/`, `data/`, `Log/`, `Dataset_backup*/`, and
`metadata/sku_catalog.json` are present locally but intentionally excluded
from version control — see **Data & privacy** below.

## Setup

```bash
cd /home/hinokami/dimscan
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Hardware / Orbbec SDK setup

Live camera capture requires `pyorbbecsdk`, Orbbec's own Python SDK. It is
**not published on PyPI** and is intentionally not in `requirements.txt`.
Install the platform-specific wheel or build from source from Orbbec's
official SDK distribution, matching your Python version and OS/arch. The
rest of DimScan (routes, feature extraction, AI2 training/inference,
tests) runs without it — `capture/orbbec_camera.py` lazily imports
`pyorbbecsdk` and raises a clear runtime error only when a live capture is
actually attempted without it installed.

## Run

```bash
cd /home/hinokami/dimscan
source .venv/bin/activate
export DIMSCAN_YOLOE_MODEL_NAME=yoloe-11s-seg.pt
python -m app.server
```

## AI2 training

```bash
# Single-plant model
python scripts/train_ai2_v1.py \
  --job-type single \
  --jobs-root datasets/single/jobs \
  --out models/packing/single/ai2_v1

# Group (multi-item) model
python scripts/train_ai2_v1.py \
  --job-type group \
  --jobs-root datasets/group/jobs \
  --out models/packing/group/ai2_v1
```

Both commands read completed jobs from the given jobs root (each job's
`combined_features.json` and `ground_truth.json`), fit a
`RandomForestRegressor` pipeline, and write `model.joblib`,
`feature_list.json`, `labels.json`, and `report.json` into `--out`.

## Known limitations

- The group AI2 model is trained on a small, still-growing dataset and is
  an early baseline, not a validated production model.
- The table-plane RANSAC and ROI bounds assume a fixed, calibrated
  capture rig; moving the camera or table requires recalibrating the ROI
  and re-validating the plane cache assumptions.
- AI1 (YOLOE) is validation/debug tooling only; its confidence scores are
  not currently used to gate or improve geometry extraction.
- Rigid-object edge cases (e.g. very sparse or very reflective foliage,
  extreme leaf overhang past the calibrated ROI) can still produce a
  degraded `object_cloud.ply`, since the pipeline has no fallback beyond
  its own validation thresholds.

## Future work

- Grow the group dataset enough to validate the group AI2 model the same
  way the single-plant model has been validated.
- Explore using AI1 detections as an active signal (not just a debug
  check) where geometry-only extraction is ambiguous.
- Expand automated test coverage for the fragment-merging and
  table-plane-cache decision boundaries.

## Data & privacy

`datasets/`, `prediction_data/`, `data/`, `Log/`, `Dataset_backup*/`, and
`metadata/sku_catalog.json` are excluded from this repository and always
will be. They contain real captured point clouds, depth arrays, and a
private reference catalog of plant/pot item metadata used to enrich
features. None of that is needed to read, build, or run the code in this
repository. Trained model binaries (`*.joblib`, `*.pt`, `*.onnx`, etc.)
are excluded for the same reason: they are derived from that private
data. Anyone cloning this repository can run the pipeline against their
own captures and train their own models using the commands above.
