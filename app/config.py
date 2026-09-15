"""Central configuration for the production DimScan project."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class DimScanConfig:
    """Project-level paths, schema versions, and naming constants."""

    project_name: str = "dimscan"
    default_units: str = "inches"
    feature_schema_version: str = "v1.1"
    group_feature_schema_version: str = "v1.2"
    geometry_schema_version: str = "v1.0"
    ground_truth_schema_version: str = "v1.0"

    dataset_root: Path = Path("datasets")
    single_jobs_dir: Path = Path("datasets/single/jobs")
    single_exports_dir: Path = Path("datasets/single/exports")
    group_jobs_dir: Path = Path("datasets/group/jobs")
    group_exports_dir: Path = Path("datasets/group/exports")

    models_root: Path = Path("models")
    segmentation_models_dir: Path = Path("models/segmentation")
    single_packing_models_dir: Path = Path("models/packing/single")
    group_packing_models_dir: Path = Path("models/packing/group")

    experiments_dir: Path = Path("experiments")
    tests_dir: Path = Path("tests")

    job_metadata_filename: str = "job_metadata.json"
    item_list_filename: str = "item_list.json"
    arrangement_filename: str = "arrangement.json"
    combined_features_filename: str = "combined_features.json"
    ground_truth_filename: str = "ground_truth.json"
    session_filename: str = "session.json"

    rgb_filename: str = "rgb.png"
    depth_filename: str = "depth.png"
    depth_raw_filename: str = "depth_raw.npy"
    depth_aligned_to_rgb_filename: str = "depth_aligned_to_rgb.npy"
    capture_meta_filename: str = "capture_meta.json"
    cloud_filename: str = "cloud.ply"
    geometry_filename: str = "geometry.json"
    features_filename: str = "features.json"
    quality_summary_filename: str = "quality_summary.json"

    segmented_dirname: str = "segmented"
    pot_cloud_filename: str = "pot.ply"
    leaf_cloud_filename: str = "leaf.ply"
    stem_cloud_filename: str = "stem.ply"
    flower_cloud_filename: str = "flower.ply"

    job_type_single: str = "single"
    job_type_group: str = "group"
    mode_data_collection: str = "data_collection"
    mode_prediction: str = "prediction"
    view_mode_single: str = "single_view"
    view_mode_two_view_rectangle: str = "two_view_rectangle"

    camera_name: str = "Orbbec Gemini 435Le"
    default_camera_pose: str = "side_bottom"

    segmentation_enabled: bool = True
    segmentation_backend: str = "yoloe"
    enable_ai1_validation: bool = True
    yoloe_model_path: str | None = None
    yoloe_model_name: str | None = None
    yoloe_confidence_threshold: float = 0.25
    pot_min_confidence: float = 0.40
    pot_min_object_overlap_ratio: float = 0.30
    pot_min_area_ratio_to_object: float = 0.03
    pot_max_area_ratio_to_object: float = 0.55
    pot_min_lower_position_score: float = 0.50
    pot_min_center_alignment_score: float = 0.35
    pot_min_mask_fill_ratio: float = 0.18
    pot_min_point_count: int = 500
    pot_min_depth_width_in: float = 1.5
    pot_max_depth_width_in: float = 30.0
    pot_min_depth_height_in: float = 1.0
    pot_max_depth_height_in: float = 30.0
    nursery_container_lookup: ClassVar[dict[str, dict[str, float]]] = {
        "1 qt": {"diameter_in": 4.0, "height_in": 4.0, "volume_qt": 1.0},
        "2 qt": {"diameter_in": 5.5, "height_in": 5.0, "volume_qt": 2.0},
        "3 qt": {"diameter_in": 6.5, "height_in": 6.0, "volume_qt": 3.0},
        "1 gal": {"diameter_in": 6.5, "height_in": 7.0, "volume_qt": 4.0},
        "2 gal": {"diameter_in": 8.5, "height_in": 8.5, "volume_qt": 8.0},
        "3 gal": {"diameter_in": 10.0, "height_in": 9.5, "volume_qt": 12.0},
        "5 gal": {"diameter_in": 12.0, "height_in": 11.0, "volume_qt": 20.0},
    }
    yoloe_prompts: ClassVar[list[str]] = [
        "potted plant",
        "plant",
        "plant pot",
        "flower pot",
        "planter",
        "leaves",
        "foliage",
        "table",
    ]

    roi_enabled: bool = True
    roi_coordinate_frame: str = "rgb_camera"
    roi_units: str = "meters"
    roi_min_depth_m: float = 0.25
    roi_max_depth_m: float = 1.3168
    roi_half_width_m: float = 0.5334
    roi_min_height_m: float | None = None
    roi_max_height_m: float | None = 1.0668
    roi_margin_m: float | None = 0.05
    table_plane_fit_voxel_size_m: float | None = None
    table_plane_fit_max_points: int | None = 50000


def get_config() -> DimScanConfig:
    """Return a new DimScan configuration instance."""
    return DimScanConfig()


def ensure_project_dirs(cfg: DimScanConfig) -> None:
    """Create standard project directories if they do not already exist."""
    dirs = (
        cfg.dataset_root,
        cfg.single_jobs_dir,
        cfg.single_exports_dir,
        cfg.group_jobs_dir,
        cfg.group_exports_dir,
        cfg.models_root,
        cfg.segmentation_models_dir,
        cfg.single_packing_models_dir,
        cfg.group_packing_models_dir,
        cfg.experiments_dir,
        cfg.tests_dir,
    )

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
