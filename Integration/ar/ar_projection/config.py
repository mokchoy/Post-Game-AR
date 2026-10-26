"""Single JSON source of defaults; strict typed validation before opening devices."""
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, get_type_hints, get_origin, get_args, Union
import json
import math


@dataclass(frozen=True)
class Camera:
    source_type: str
    index: int
    video_path: Optional[str]
    backend: str
    width: int
    height: int
    calibration_path: str
    undistort: bool
    demo_approx: bool
    timestamps_path: Optional[str] = None


@dataclass(frozen=True)
class Tags:
    enabled: bool
    family: str
    anchor_id: int
    size_m: float
    corner_refinement_window_px: int
    corner_sigma_px: float
    max_reprojection_rms_px: float
    min_side_px: float
    ambiguity_error_gap_px: float
    ambiguity_rotation_gap_deg: float


@dataclass(frozen=True)
class LK:
    enabled: bool
    window_size_px: int
    max_level: int
    max_iterations: int
    epsilon: float
    forward_backward_max_error_px: float


@dataclass(frozen=True)
class IdRevalidation:
    mode: str = "off"  # off, loss_clear, age_expire, descriptor_check
    max_age_frames: int = 50  # age_expire: clear an inherited id after this many unconfirmed frames
    max_descriptor_distance: int = 60  # descriptor_check: clear an inherited id if its live descriptor drifts past this Hamming distance from the map


@dataclass(frozen=True)
class GuidedMatching:
    enabled: bool = False
    search_radius_px: float = 12.0  # base window radius around the prior-pose projection
    max_hamming_distance: int = 60
    ratio_test: float = 0.75
    revalidate_existing: bool = False  # clear inherited ids whose landmark projects farther than search_radius_px from the feature under the prior pose
    geometry_margin: float = 1.0  # geometry_first: drop a pick if another window candidate is closer than margin x its pixel distance; 1.0 = off
    geometry_first: bool = False  # rank window candidates by pixel distance; descriptor only breaks ties and rejects outliers


@dataclass(frozen=True)
class Features:
    enabled: bool
    backend: str
    max_features: int
    grid_rows: int
    grid_cols: int
    scale_factor: float
    nlevels: int
    fast_threshold: int
    edge_threshold: int
    patch_size: int
    ratio_test: float
    max_hamming_distance: int
    mutual_matching: bool
    observation_sigma_px: float
    exclude_tag_regions: bool
    tag_mask_margin_px: int
    exclude_rectangles: list
    redetect_every_n_frames: int
    redetect_below_tracks: int
    lk: LK
    redetection_match_radius_px: float = 5.0
    id_revalidation: IdRevalidation = IdRevalidation()
    guided_matching: GuidedMatching = GuidedMatching()


@dataclass(frozen=True)
class Mapping:
    mode: str
    map_path: str
    max_keyframes: int
    max_landmarks: int
    min_ready_landmarks: int
    min_baseline_m: float
    min_parallax_deg: float
    min_confirming_views: int
    min_depth_m: float
    max_depth_m: float
    max_reprojection_rms_px: float
    refine_bootstrap_map: bool
    refinement_max_nfev: int
    refinement_time_budget_s: float = 5.0  # unused for stopping since refinement_max_evaluations replaced it (deterministic); kept for old configs
    refinement_retry_interval_s: float = 2.0
    refinement_max_evaluations: int = 2000  # deterministic residual-evaluation cap; replaces the old wall-clock deadline (non-reproducible under load)
    multiview_descriptor_bank: int = 0  # extra descriptors per landmark from other confirming views, beyond the primary; 0 = off (baseline)


@dataclass(frozen=True)
class Pose:
    mode: str
    min_landmark_inliers: int
    min_occupied_grid_cells: int
    ransac_iterations: int
    ransac_reprojection_threshold_px: float
    ransac_confidence: float
    robust_loss: str
    robust_f_scale: float
    max_nfev: int
    max_landmark_rms_px: float
    max_tag_rms_px: float
    max_linear_speed_m_s: float
    max_angular_speed_deg_s: float
    max_prediction_gap_s: float
    prior_grace_s: float = 0.0  # keep the last valid pose as a seed (never rendered) this long after failures; 0 = drop at once


@dataclass(frozen=True)
class Uncertainty:
    enabled: bool
    project_to_panel_corners: bool
    translation_difference_step_m: float
    rotation_difference_step_rad: float
    conditioning_translation_scale_m: float
    conditioning_rotation_scale_rad: float
    min_scaled_singular_value_ratio: float


@dataclass(frozen=True)
class Filter:
    enabled: bool
    mode: str
    measurement_weight: float


@dataclass(frozen=True)
class Panel:
    width_m: float
    height_m: float
    offset_y_m: float
    tilt_x_deg: float


@dataclass(frozen=True)
class Display:
    enabled: bool
    debug: bool
    draw_features: bool
    draw_uncertainty: bool


@dataclass(frozen=True)
class Logging:
    enabled: bool
    output_dir: str
    save_input_video: bool
    save_annotated_video: bool
    input_codec: str = "FFV1"
    association_dump: bool = False


@dataclass(frozen=True)
class BackgroundPrior:
    """Step 48b stored static-background depth map (160 x 120) and foreground detection. Off by default."""
    enabled: bool = False
    agree_tol: float = 0.15       # relative |Z_new - Z_stored| that counts as agreement
    near_ratio: float = 0.8       # foreground: Z < near_ratio * stored Z
    fg_slack: float = 0.1         # foreground hides pixels with Z < Z_panel + fg_slack (m)
    count_cap: int = 20
    min_count: int = 3            # stored pixels with count >= min_count are trusted
    open_px: int = 7              # opening kernel for the foreground mask


@dataclass(frozen=True)
class Occlusion:
    """Step 45 live depth occlusion (relative model + base-line reference). Off by default; old configs are unaffected."""
    enabled: bool = False
    model: str = "relative"
    repo: Optional[str] = None
    checkpoint: Optional[str] = None
    input_size: int = 252
    margin_floor_m: float = 0.10
    margin_fraction: float = 0.10
    line_reference: bool = True
    line_hold_frames: int = 30
    require_tracking_acceptance: bool = True
    reference: str = "line"       # "line" (Step 45 base-line fit) or "plane" (Step 48a table/floor-plane fit)
    plane_extent_m: list = field(default_factory=lambda: [-1.0, 1.0, -0.6, 0.9])   # world x_min, x_max, y_min, y_max on Z=0
    plane_min_span: float = 0.30  # inverse-depth span (p95-p5 over median) the inliers must cover
    plane_min_px: int = 3000      # stored-map pixels join the fit when the plane gives fewer than this
    background_prior: BackgroundPrior = field(default_factory=BackgroundPrior)


def _default_panels():
    # Viewer-centric layout at scale 1, yaw 0 (x right, y away, z up); see Instruction.md Step 50. toe_in_deg > 0 turns the
    # panel's outer end towards the viewer. "above" stacks a panel on the named one (+ gap_m).
    return [{"name": "game", "role": "game", "side": "center", "width_m": 0.5, "height_m": 0.28125, "toe_in_deg": 0.0, "above": None},
            {"name": "video", "role": "video", "side": "left", "width_m": 0.25, "height_m": 0.4444, "toe_in_deg": 5.0, "above": None},
            {"name": "stats", "role": "stats", "side": "right", "width_m": 0.25, "height_m": 0.1875, "toe_in_deg": 0.0, "above": None},
            {"name": "score", "role": "score", "side": "right", "width_m": 0.25, "height_m": 0.1875, "toe_in_deg": 0.0, "above": "stats"}]


@dataclass(frozen=True)
class Array:
    """Step 50 array of upright, world-anchored panels (replaces the single panel when enabled). Off by default."""
    enabled: bool = False
    gap_m: float = 0.05
    facing_yaw_deg: float = 180.0   # viewer-centric layout -> world: 180 puts the viewer on the +Y side of the tag (as in the live runs)
    yaw_step_deg: float = 15.0
    yaw_limit_deg: float = 60.0
    scale_step: float = 1.1
    scale_min: float = 0.5
    scale_max: float = 2.0
    near_plane_m: float = 0.05
    texture_long_px: int = 400
    video_fit: str = "contain"      # contain | cover | stretch, for video frames on a panel of another aspect ratio
    left_video: Optional[str] = None
    score: str = "1:0"
    panels: list = field(default_factory=_default_panels)
    games: list = field(default_factory=lambda: [{"name": "Game 1", "video": None}, {"name": "Game 2", "video": None}, {"name": "Game 3", "video": None}])
    players: list = field(default_factory=lambda: [{"title": "Person 1", "lines": ["Height: //", "Weight: //", "Rank: //"]},
                                                   {"title": "Person 2", "lines": ["Height: //", "Weight: //", "Rank: //"]}])


@dataclass(frozen=True)
class Gestures:
    """Step 50 gesture control from a second camera in its own thread. Off by default. Positive yaw = left panel nearer the viewer."""
    enabled: bool = False
    camera_index: int = 0
    gestures_dir: str = "../gestures"
    model: str = "ddnet"
    window_seconds: float = 3.0
    min_confidence: float = 0.70
    vote_window: int = 5
    required_votes: int = 4
    cooldown_s: float = 1.0
    actions: dict = field(default_factory=lambda: {
        "Swiping Left": ["game", -1], "Swiping Right": ["game", 1],
        "Turning Hand Clockwise": ["yaw", 1], "Turning Hand Counterclockwise": ["yaw", -1],
        "Zooming In With Full Hand": ["scale", 1], "Zooming Out With Full Hand": ["scale", -1],
        "Zooming In With Two Fingers": ["scale", 1], "Zooming Out With Two Fingers": ["scale", -1],
        "Thumb Up": ["play", 1], "Thumb Down": ["reset", 0], "Stop Sign": ["stop", 0]})


@dataclass(frozen=True)
class Calibration:
    columns: int
    rows: int
    square_size_m: float
    views: int


@dataclass(frozen=True)
class Config:
    schema_version: int
    seed: int
    camera: Camera
    tags: Tags
    features: Features
    mapping: Mapping
    pose: Pose
    uncertainty: Uncertainty
    filter: Filter
    panel: Panel
    display: Display
    logging: Logging
    calibration: Calibration
    occlusion: Occlusion = Occlusion()
    array: Array = Array()
    gestures: Gestures = Gestures()


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.json"


def typed(cls, data, prefix=""):
    if not isinstance(data, dict):
        raise ValueError(f"{prefix}: expected object")
    data = dict(data)
    for f in fields(cls):
        if f.name not in data and f.default is not MISSING:
            data[f.name] = f.default
    names = {f.name for f in fields(cls)}
    if set(data) != names:
        raise ValueError(f"{prefix}: unknown {sorted(set(data)-names)}, missing {sorted(names-set(data))}")
    values = {}
    for name, kind in get_type_hints(cls).items():
        value, label = data[name], prefix + name
        if hasattr(kind, "__dataclass_fields__"):
            values[name] = typed(kind, value, label + ".")
            continue
        if get_origin(kind) is Union:
            kind = next(t for t in get_args(kind) if t is not type(None))
            if value is None:
                values[name] = None
                continue
        valid = (type(value) in (int, float) if kind is float else type(value) is kind)
        if not valid or (type(value) in (float, int) and not math.isfinite(value)):
            raise ValueError(f"{label}: expected finite {kind.__name__}")
        values[name] = float(value) if kind is float else value
    return cls(**values)


def validate(c: Config) -> None:
    enums = {"camera.source_type": ("camera", "video"), "camera.backend": ("auto", "dshow", "msmf", "v4l2", "avfoundation", "ffmpeg"),
             "tags.family": ("APRILTAG_36h11",), "features.backend": ("orb",), "mapping.mode": ("off", "build", "localize"),
             "pose.mode": ("tag_only", "landmark_only", "joint"), "pose.robust_loss": ("linear", "huber", "soft_l1", "cauchy", "arctan"), "filter.mode": ("ema",),
             "logging.input_codec": ("FFV1", "HFYU"), "features.id_revalidation.mode": ("off", "loss_clear", "age_expire", "descriptor_check"), "occlusion.model": ("relative", "metric"), "occlusion.reference": ("line", "plane")}
    for label, choices in enums.items():
        obj = c
        for part in label.split("."):
            obj = getattr(obj, part)
        if obj not in choices:
            raise ValueError(f"{label}: supported values {choices}")
    nonnegative = {"index", "anchor_id", "fast_threshold", "edge_threshold", "tag_mask_margin_px", "redetect_below_tracks", "max_level", "offset_y_m", "ambiguity_error_gap_px", "ambiguity_rotation_gap_deg", "seed", "multiview_descriptor_bank", "max_age_frames", "max_descriptor_distance", "refinement_max_evaluations", "prior_grace_s", "camera_index", "cooldown_s"}
    signed = {"tilt_x_deg", "facing_yaw_deg"}
    def bounds(obj, prefix=""):
        for f in fields(obj):
            v = getattr(obj, f.name)
            if hasattr(v, "__dataclass_fields__"):
                bounds(v, prefix + f.name + ".")
            elif type(v) in (int, float) and f.name not in signed:
                if v < 0 or (v == 0 and f.name not in nonnegative):
                    raise ValueError(f"{prefix+f.name}: must be {'nonnegative' if f.name in nonnegative else 'positive'}")
    bounds(c)
    if c.schema_version != 1 or c.seed > 2**31-1:
        raise ValueError("schema_version must be 1; seed must fit signed int32")
    for label, value in (("features.ratio_test", c.features.ratio_test), ("pose.ransac_confidence", c.pose.ransac_confidence),
                        ("features.guided_matching.ratio_test", c.features.guided_matching.ratio_test)):
        if not 0 < value < 1:
            raise ValueError(f"{label}: must be between 0 and 1")
    if not 0 < c.filter.measurement_weight <= 1 or not 0 < c.uncertainty.min_scaled_singular_value_ratio < 1:
        raise ValueError("filter.measurement_weight or uncertainty.min_scaled_singular_value_ratio: invalid fraction")
    if (c.features.scale_factor <= 1 or c.features.max_hamming_distance > 256 or c.features.id_revalidation.max_descriptor_distance > 256
            or c.features.guided_matching.max_hamming_distance > 256):
        raise ValueError("features.scale_factor must exceed 1; max_hamming_distance fields <= 256")
    if c.features.patch_size < 3 or c.features.patch_size % 2 == 0 or c.features.lk.window_size_px % 2 == 0:
        raise ValueError("features.patch_size and features.lk.window_size_px must be odd")
    if c.mapping.min_depth_m >= c.mapping.max_depth_m:
        raise ValueError("mapping.min_depth_m must be less than max_depth_m")
    if c.mapping.min_confirming_views < 3 or c.mapping.min_confirming_views > c.mapping.max_keyframes or c.mapping.min_ready_landmarks > c.mapping.max_landmarks:
        raise ValueError("mapping support/capacity settings are impossible (at least three views)")
    if c.pose.min_landmark_inliers < 4 or c.pose.min_occupied_grid_cells > c.features.grid_rows*c.features.grid_cols:
        raise ValueError("pose landmark count/coverage settings are impossible")
    for i, rect in enumerate(c.features.exclude_rectangles):
        if not isinstance(rect, list) or len(rect) != 4 or any(type(v) is not int for v in rect) or min(rect[2:]) <= 0:
            raise ValueError(f"features.exclude_rectangles[{i}]: expected [x,y,positive width,positive height] integers")
    ex = c.occlusion.plane_extent_m
    if len(ex) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in ex) or ex[0] >= ex[1] or ex[2] >= ex[3]:
        raise ValueError("occlusion.plane_extent_m: expected [x_min, x_max, y_min, y_max] with min < max")
    a = c.array
    if a.video_fit not in ("contain", "cover", "stretch") or a.scale_min > 1 or a.scale_max < 1 or a.scale_step <= 1 or a.texture_long_px < 16:
        raise ValueError("array: video_fit contain/cover/stretch; scale_min <= 1 <= scale_max; scale_step > 1; texture_long_px >= 16")
    names = [p.get("name") for p in a.panels]
    if len(set(names)) != len(names) or sum(p.get("side") == "center" for p in a.panels) != 1 or not a.games:
        raise ValueError("array.panels: unique names and exactly one centre panel; array.games must not be empty")
    for p in a.panels:
        if p.get("side") not in ("center", "left", "right") or p.get("role") not in ("game", "video", "score", "stats") \
                or not (p.get("width_m", 0) > 0 and p.get("height_m", 0) > 0) or (p.get("above") not in names + [None]):
            raise ValueError(f"array.panels[{p.get('name')}]: side center/left/right, role game/video/score/stats, positive size, 'above' an existing name or null")
    for label, (action, _) in c.gestures.actions.items():
        if action not in ("game", "yaw", "scale", "play", "stop", "reset"):
            raise ValueError(f"gestures.actions[{label}]: unknown action {action}")
    if c.occlusion.background_prior.near_ratio >= 1:
        raise ValueError("occlusion.background_prior.near_ratio: must be below 1")
    if c.mapping.mode == "build" and (not c.tags.enabled or not c.features.enabled or c.camera.demo_approx):
        raise ValueError("mapping.mode=build requires tags, features and actual calibration")
    if c.pose.mode == "tag_only" and not c.tags.enabled:
        raise ValueError("pose.mode=tag_only requires tags.enabled")
    if c.pose.mode == "landmark_only" and (not c.features.enabled or c.mapping.mode != "localize"):
        raise ValueError("pose.mode=landmark_only requires features and mapping.mode=localize")
    if c.pose.mode == "joint" and not c.tags.enabled and (not c.features.enabled or c.mapping.mode != "localize"):
        raise ValueError("pose.mode=joint without tags requires features and a ready loaded map")
    if c.camera.demo_approx and (c.pose.mode != "tag_only" or c.mapping.mode != "off" or c.uncertainty.enabled):
        raise ValueError("camera.demo_approx requires tag_only, mapping off, uncertainty disabled")
    if c.camera.source_type == "video" and not c.camera.video_path:
        raise ValueError("camera.video_path is required for video")
    if c.camera.timestamps_path is not None and (c.camera.source_type != "video" or not c.camera.timestamps_path):
        raise ValueError("camera.timestamps_path requires a video source and nonempty JSONL path")
    if c.calibration.columns < 2 or c.calibration.rows < 2 or c.calibration.views < 8:
        raise ValueError("calibration requires >=2 inner corners per dimension and >=8 views")
    for label in ("camera.calibration_path", "mapping.map_path", "logging.output_dir"):
        section, name = label.split(".")
        if not getattr(getattr(c, section), name):
            raise ValueError(f"{label}: cannot be empty")


def load_config(filename=DEFAULT_PATH, overrides=None) -> Config:
    filename = Path(filename).resolve()
    data = json.loads(filename.read_text(encoding="utf-8-sig"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {x}")))
    data["camera"].setdefault("timestamps_path", None)
    data["features"].setdefault("redetection_match_radius_px", 5.0)
    data["features"].setdefault("id_revalidation", {"mode": "off", "max_age_frames": 50, "max_descriptor_distance": 60})
    data["features"].setdefault("guided_matching", {"enabled": False, "search_radius_px": 12.0, "max_hamming_distance": 60, "ratio_test": 0.75, "geometry_first": False, "revalidate_existing": False, "geometry_margin": 1.0})
    data["mapping"].setdefault("refinement_time_budget_s", 5.0)
    data["mapping"].setdefault("refinement_retry_interval_s", 2.0)
    data["mapping"].setdefault("refinement_max_evaluations", 2000)
    data["mapping"].setdefault("multiview_descriptor_bank", 0)
    data["logging"].setdefault("input_codec", "FFV1")
    data["logging"].setdefault("association_dump", False)
    data["occlusion"] = {**asdict(Occlusion()), **data.get("occlusion", {})}
    data["array"] = {**asdict(Array()), **data.get("array", {})}
    data["gestures"] = {**asdict(Gestures()), **data.get("gestures", {})}
    for label, value in (overrides or {}).items():
        section, name = label.split(".")
        if section not in data or name not in data[section]:
            raise ValueError(f"{label}: unknown override")
        data[section][name] = value
    c = typed(Config, data)
    validate(c)
    for section, name in (("camera", "video_path"), ("camera", "timestamps_path"), ("camera", "calibration_path"), ("mapping", "map_path"), ("logging", "output_dir"), ("gestures", "gestures_dir")):
        value = data[section][name]
        if value is not None:
            data[section][name] = str((filename.parent / value).resolve())
    return typed(Config, data)
