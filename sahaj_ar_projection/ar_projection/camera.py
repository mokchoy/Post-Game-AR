"""Capture and one coherent preprocessing path; calibration NPZ remains compatible."""
from pathlib import Path
import hashlib
import sys
import time
import cv2
import numpy as np
from .types import CameraModel, FramePacket


def backend(name, video=False):
    if name == "auto":
        return cv2.CAP_ANY if video else (cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    names = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2, "avfoundation": cv2.CAP_AVFOUNDATION, "ffmpeg": cv2.CAP_FFMPEG}
    if name not in names:
        raise ValueError(f"camera.backend: unsupported {name}")
    return names[name]


def load_calibration(filename, size, demo_approx=False):
    w, h = size
    if demo_approx:
        f = w/(2*np.tan(np.deg2rad(60)/2))
        K, d = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]]), np.zeros(5)
        identity = {"calibration_sha256": None, "approximate": True}
    else:
        p = Path(filename)
        try:
            with np.load(p, allow_pickle=False) as data:
                K = np.asarray(data["camera_matrix"], np.float64).copy()
                d = np.asarray(data["dist_coeffs"], np.float64).reshape(-1)
                raw_size = data["image_size"]
                if raw_size.shape != (2,) or raw_size.dtype.kind not in "iu" or np.any(raw_size <= 0):
                    raise ValueError("image_size must be two positive integers")
                calibrated_size = tuple(map(int, raw_size))
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"camera.calibration_path: {p}: {exc}") from exc
        # Strict dimensions: live capture can change FOV even at identical aspect ratio.
        if calibrated_size != size:
            raise ValueError(f"calibration dimensions {calibrated_size} != captured {size}; recalibrate at this mode (no implicit resize)")
        identity = {"calibration_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "approximate": False}
    if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0
            or d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all()):
        raise ValueError("calibration: invalid finite OpenCV intrinsics/distortion")
    return K, d, identity


class Preprocessor:
    def __init__(self, cfg, size):
        K, d, identity = load_calibration(cfg.calibration_path, size, cfg.demo_approx)
        self.remap = None
        if cfg.undistort:
            # Preserve full extent and original K; no crop/new principal point.
            self.remap = cv2.initUndistortRectifyMap(K, d, None, K, size, cv2.CV_32FC1)
        identity.update({"size": list(size), "undistort": cfg.undistort, "K": K.tolist(), "distortion": d.tolist()})
        self.camera = CameraModel(K, np.zeros_like(d) if cfg.undistort else d, size, identity)

    def packet(self, index, timestamp, bgr):
        if (bgr.shape[1], bgr.shape[0]) != self.camera.size:
            raise ValueError("capture dimensions changed within a run")
        processed = cv2.remap(bgr, *self.remap, interpolation=cv2.INTER_LINEAR) if self.remap else bgr.copy()
        return FramePacket(index, timestamp, bgr, processed, cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), self.camera)


class Capture:
    def __init__(self, cfg):
        self.cfg = cfg
        video = cfg.source_type == "video"
        self.cap = cv2.VideoCapture(cfg.video_path if video else cfg.index, backend(cfg.backend, video))
        self.index = 0
        self.preprocessor = None
        if not self.cap.isOpened():
            self.close()
            raise ValueError("cannot open camera/video source")
        if not video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if video and (not np.isfinite(self.fps) or self.fps <= 0):
            self.close()
            raise ValueError("video has invalid source FPS; deterministic timestamps unavailable")
        self.timestamp_basis = "frame_index/source_fps" if video else "monotonic acquisition (not exposure)"

    def read(self):
        ok, bgr = self.cap.read()
        acquired = time.monotonic()
        if not ok:
            if self.index == 0 or self.cfg.source_type == "camera":
                raise ValueError("capture returned no frame")
            return None
        if self.preprocessor is None:
            self.preprocessor = Preprocessor(self.cfg, (bgr.shape[1], bgr.shape[0]))
        timestamp = self.index/self.fps if self.cfg.source_type == "video" else acquired
        packet = self.preprocessor.packet(self.index, timestamp, bgr)
        self.index += 1
        return packet

    def close(self):
        self.cap.release()
