"""Owned records. Geometry is float64, descriptors uint8; pixels are processed pixels."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray  # (3,3), processed intrinsics
    distortion: np.ndarray
    size: tuple[int, int]
    identity: dict


@dataclass
class FramePacket:
    index: int
    timestamp: float  # source seconds; live monotonic acquisition, video index/FPS
    original: np.ndarray
    bgr: np.ndarray
    gray: np.ndarray
    camera: CameraModel


@dataclass
class TagObservation:
    id: int
    pixels: np.ndarray  # (4,2), TL TR BR BL
    world: np.ndarray  # (4,3), metres in anchor W
    candidates: list[np.ndarray] = field(default_factory=list)  # T_CW
    errors: list[float] = field(default_factory=list)
    ambiguous: bool = False
    valid: bool = False
    reason: Optional[str] = None


@dataclass
class FeatureObservation:
    track_id: int
    pixel: np.ndarray  # (2,)
    descriptor: Optional[np.ndarray] = None  # (32,), uint8
    landmark_id: Optional[int] = None
    flow_error: Optional[float] = None
    valid: bool = True


@dataclass
class LandmarkMap:
    points: np.ndarray  # (N,3) in W, metres
    descriptors: np.ndarray  # (N,32), uint8
    ids: np.ndarray  # (N,), unique int64
    support: np.ndarray  # (N,), independent keyframe counts
    rms: np.ndarray  # (N,), pixel RMS
    metadata: dict


@dataclass
class PoseEstimate:
    T_CW: Optional[np.ndarray] = None
    valid: bool = False
    source: str = "none"
    landmark_inliers: int = 0
    tag_inliers: int = 0
    landmark_rms: Optional[float] = None
    tag_rms: Optional[float] = None
    optimization_status: Optional[int] = None
    covariance: Optional[np.ndarray] = None  # [dt_camera,dr_camera], LEFT retraction
    covariance_reason: Optional[str] = None
    reason: Optional[str] = None
    covariance_ms: float = 0.0


@dataclass
class FrameResult:
    state: str
    raw_pose: PoseEstimate
    render_pose: Optional[np.ndarray]
    raw_panel: Optional[np.ndarray]
    render_panel: Optional[np.ndarray]
    pixel_covariance: Optional[np.ndarray]
    diagnostics: dict
    timings: dict
