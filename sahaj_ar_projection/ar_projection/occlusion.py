"""Optional, acceptance-gated metric RGB depth and diagnostic panel compositing.

Core geometry/evaluation use NumPy/OpenCV only. PyTorch is imported only when
the explicitly enabled model is constructed in the separate depth environment.
"""
import argparse
from collections import deque
import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from .diagnostics import write_json
from .experiments import fingerprint, percentile, summarize
from .geometry import depths, tag_points
from .rendering import visible_panel


@lru_cache(maxsize=4)
def camera_rays(size, intrinsics):
    w, h = size
    y, x = np.indices((h, w))
    return np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(np.array(intrinsics).reshape(3, 3)).T


_VIDEO = {"cap": None, "frame": None, "t0": 0., "n": 0}


def set_panel_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open panel video {path}")
    _VIDEO.update(cap=cap, frame=None, t0=time.monotonic(), n=0, fps=cap.get(cv2.CAP_PROP_FPS) or 25.)


def advance_panel_video():
    """Call once per frame: sets the panel texture to the video frame for wall-clock time (loops; skips frames when behind)."""
    cap = _VIDEO["cap"]
    if cap is None:
        return
    due = int((time.monotonic()-_VIDEO["t0"])*_VIDEO["fps"])
    while _VIDEO["n"] <= due or _VIDEO["frame"] is None:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _VIDEO["t0"], _VIDEO["n"] = time.monotonic(), 0
            due = 0
            continue
        _VIDEO["frame"], _VIDEO["n"] = frame, _VIDEO["n"]+1


def panel_layer(T, panel, camera, scene_depth=None, margin_m=.02, invalid_occludes=False, margin_fraction=0.):
    """Return BGR texture, visibility, and camera-Z metres (zero outside panel). margin_fraction > 0 makes the margin
    max(margin_m, margin_fraction * panel Z) per pixel."""
    w, h = camera.size
    mask = np.zeros((h, w), np.uint8)
    z = np.zeros((h, w), np.float64)
    pixels = visible_panel(T, panel, camera)
    if pixels is None:
        return np.zeros((h, w, 3), np.uint8), mask.astype(bool), z
    if np.any(camera.distortion != 0):
        raise ValueError("occlusion requires rectified camera pixels (camera.undistort=true)")
    xyz = panel@T[:3, :3].T+T[:3, 3]
    normal = np.cross(xyz[1]-xyz[0], xyz[3]-xyz[0])
    rays = camera_rays(camera.size, tuple(camera.K.ravel()))
    denominator = rays@normal
    cv2.fillConvexPoly(mask, np.rint(pixels).astype(np.int32), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = float(normal@xyz[0])/denominator
    valid = mask.astype(bool) & np.isfinite(distance) & (distance > 0)
    z[valid] = distance[valid]
    y, x = np.indices((96, 160))
    checks = ((x//16+y//16) % 2).astype(bool)
    texture = np.empty((96, 160, 3), np.uint8)
    texture[checks], texture[~checks] = (220, 180, 40), (245, 245, 245)
    if _VIDEO["frame"] is not None:  # stretched to the panel; panel is 16:9, video is not necessarily
        texture = cv2.flip(_VIDEO["frame"], 0)  # panel corner order puts the texture's top at the panel's bottom
    th, tw = texture.shape[:2]
    H = cv2.getPerspectiveTransform(np.array([[0, 0], [tw-1, 0], [tw-1, th-1], [0, th-1]], np.float32), pixels.astype(np.float32))
    color = cv2.warpPerspective(texture, H, (w, h))
    if scene_depth is not None:
        scene_depth = np.asarray(scene_depth)
        if scene_depth.shape != z.shape:
            raise ValueError("scene depth must match the rectified image dimensions")
        known = np.isfinite(scene_depth) & (scene_depth > 0)
        margin = np.maximum(margin_m, margin_fraction*z) if margin_fraction else margin_m
        ok = scene_depth+margin >= z
        valid &= (known & ok) if invalid_occludes else (~known | ok)
    return color, valid, z


def plane_depth(T_CW, point_world, normal_world, camera):
    """Analytic camera-Z of an INFINITE plane (through point_world, normal normal_world, both
    world-frame) at every pixel of the image -- an oracle depth for a flat surface (e.g. the
    table), unlike panel_layer's quad which is clipped to its own small polygon. Same ray-plane
    algebra as panel_layer, generalized from a quad's own corners to an arbitrary point+normal."""
    if np.any(camera.distortion != 0):
        raise ValueError("plane_depth requires rectified camera pixels (camera.undistort=true)")
    normal_c = normal_world@T_CW[:3, :3].T
    point_c = point_world@T_CW[:3, :3].T+T_CW[:3, 3]
    rays = camera_rays(camera.size, tuple(camera.K.ravel()))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = float(normal_c@point_c)/(rays@normal_c)
    valid = np.isfinite(z) & (z > 0)
    out = np.zeros(z.shape)
    out[valid] = z[valid]
    return out, valid


@dataclass(frozen=True)
class PlaneParams:
    """Step 37 dense table-plane correction settings. terms: 'full' = a+b*raw+c*row+d*col,
    'affine' = a+b*raw, 'noint' = b*raw+c*row+d*col (plane in image, no raw offset),
    'fixedb' = a+b0*raw+c*row+d*col with b0 (dZ/draw) supplied from the landmark-anchor fit, not fitted:
    on a flat table raw is nearly a plane in (row, col), so b is barely identifiable there (Step 37)."""
    terms: str = "full"
    z_min_m: float = .3
    z_max_m: float = 1.5
    min_grazing_deg: float = 10.
    max_height_m: float = .03       # map landmarks higher than this above Z=0 mark "object" pixels
    landmark_radius_px: int = 8
    panel_dilate_px: int = 15
    min_pixels: int = 5000
    min_fraction: float = .02
    max_mad_m: float = .05
    b_range: tuple = (.5, 2.)
    hold: int = 5
    max_fit_pixels: int = 20000


def _design(raw, rows, cols, terms):
    cols_ = {"full": (np.ones_like(raw), raw, rows, cols), "affine": (np.ones_like(raw), raw),
             "noint": (raw, rows, cols), "fixedb": (np.ones_like(raw), rows, cols)}[terms]
    return np.stack(cols_, axis=-1)


def huber_lstsq(A, y, iterations=30):
    """IRLS Huber linear fit; scale re-estimated each pass from the residual MAD."""
    x = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(iterations):
        r = y-A@x
        k = 1.345*max(1.4826*float(np.median(np.abs(r))), 1e-3)
        w = np.sqrt(np.where(np.abs(r) <= k, 1., k/np.maximum(np.abs(r), 1e-12)))
        x = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)[0]
    return x


class PlaneCorrector:
    """Per-frame fit of Z_true ~ f(raw, row, col) on table-plane pixels (oracle depth from the
    world Z=0 plane under the current pose), coefficients held as a median of the last N passing
    fits. Gate failure clears the hold (never reuse a stale fit) and returns None so the caller
    can fall back. The plane is a gated PRIOR: it assumes the tag lies flat on a large flat table."""
    def __init__(self, params=PlaneParams(), tag_size_m=.17):
        self.params, self.tag_size_m, self.held = params, tag_size_m, deque(maxlen=params.hold)
        self.rng = np.random.default_rng(0)

    def calibration_mask(self, T, camera, panel, raw, landmark_points=None):
        p = self.params
        w, h = camera.size
        oracle, valid = plane_depth(T, np.zeros(3), np.array([0., 0, 1]), camera)
        rays = camera_rays(camera.size, tuple(camera.K.ravel()))
        n_c = np.array([0., 0, 1])@T[:3, :3].T
        sin_angle = np.abs(rays@n_c)/np.linalg.norm(rays, axis=-1)
        mask = valid & (oracle >= p.z_min_m) & (oracle <= p.z_max_m) & (sin_angle >= np.sin(np.radians(p.min_grazing_deg)))
        mask &= np.isfinite(raw) & (raw > 0)
        k = np.ones((2*p.panel_dilate_px+1,)*2, np.uint8)
        for quad in (panel, tag_points(self.tag_size_m)):
            _, vis, z = panel_layer(T, quad, camera)
            mask &= ~cv2.dilate((z > 0).astype(np.uint8), k).astype(bool)
        if landmark_points is not None and len(landmark_points):
            high = landmark_points[landmark_points[:, 2] > p.max_height_m]
            if len(high):
                xyz = high@T[:3, :3].T+T[:3, 3]
                xyz = xyz[xyz[:, 2] > 0]
                uv = xyz@camera.K.T
                uv = uv[:, :2]/uv[:, 2:3]
                blocked = np.zeros((h, w), np.uint8)
                for u, v in uv:
                    if -w < u < 2*w and -h < v < 2*h:
                        cv2.circle(blocked, (int(round(u)), int(round(v))), p.landmark_radius_px, 1, -1)
                mask &= ~blocked.astype(bool)
        return mask, oracle

    def __call__(self, raw, T, panel, camera, landmark_points=None, b_prior=None):
        """Return (corrected depth or None, stats). b_prior: dZ/draw, required for terms='fixedb'."""
        p = self.params
        h, w = raw.shape
        mask, oracle = self.calibration_mask(T, camera, panel, raw, landmark_points)
        n = int(mask.sum())
        stats = {"plane_pixels": n, "plane_fraction": n/(h*w), "plane_terms": p.terms, "plane_gate": False}
        rows, cols = np.indices((h, w))
        rows, cols = rows/(h-1)*2-1, cols/(w-1)*2-1
        if n < max(p.min_pixels, p.min_fraction*h*w):
            self.held.clear()
            stats["plane_gate_reason"] = "too few plane pixels"
            return None, stats
        idx = np.flatnonzero(mask.ravel())
        if n > p.max_fit_pixels:
            idx = self.rng.choice(idx, p.max_fit_pixels, replace=False)
        fixed = p.terms == "fixedb"
        if fixed and b_prior is None:
            self.held.clear()
            stats["plane_gate_reason"] = "no b prior (anchor fit unavailable)"
            return None, stats
        offset = b_prior*raw if fixed else 0.  # part of the model that is not fitted
        A = _design(raw.ravel()[idx], rows.ravel()[idx], cols.ravel()[idx], p.terms)
        y = oracle.ravel()[idx]-(offset.ravel()[idx] if fixed else 0.)
        x = huber_lstsq(A, y)
        braw = b_prior if fixed else x[{"full": 1, "affine": 1, "noint": 0}[p.terms]]
        mad = float(np.median(np.abs(y-A@x)))
        P = np.stack((np.ones(len(idx)), rows.ravel()[idx], cols.ravel()[idx]), axis=-1)
        r = raw.ravel()[idx]
        r2 = 1-float(np.sum((r-P@np.linalg.lstsq(P, r, rcond=None)[0])**2)/max(np.sum((r-r.mean())**2), 1e-12))
        # r2 near 1 => raw is almost a plane in (row, col) on the table, so b is barely identifiable from table pixels alone
        stats.update(plane_coeffs=x.tolist(), plane_residual_mad_m=mad, plane_raw_planarity_r2=r2)
        if mad > p.max_mad_m or not p.b_range[0] <= braw <= p.b_range[1]:
            self.held.clear()
            stats["plane_gate_reason"] = "residual MAD or raw coefficient out of range"
            return None, stats
        self.held.append(x)
        xh = np.median(np.array(self.held), axis=0)
        corrected = np.einsum("hwk,k->hw", _design(raw, rows, cols, p.terms), xh)+(offset if fixed else 0.)
        corrected[~(np.isfinite(raw) & (raw > 0))] = 0
        resid = (corrected-oracle)[mask]
        h3 = np.arange(h)[:, None]*np.ones((1, w))
        stats.update(plane_gate=True, plane_held_coeffs=xh.tolist(), plane_margin_3sigma_m=float(3*1.4826*np.median(np.abs(resid-np.median(resid)))),
                     plane_bias_by_row_band_m={name: float(np.median(resid[(h3[mask] >= lo*h) & (h3[mask] < hi*h)])) if np.any((h3[mask] >= lo*h) & (h3[mask] < hi*h)) else None
                                               for name, lo, hi in (("top", 0, 1/3), ("middle", 1/3, 2/3), ("bottom", 2/3, 1.0001))})
        return corrected, stats


def tag_plane_scale(T_CW, raw_depth, tag_size_m, camera, erode_px=3):
    """Step 30 calibration: s = median(Z_true/Z_pred) over the tag's own plane, using the tag's
    known size for Z_true (ray/plane intersection, reusing panel_layer) and the model's raw output
    for Z_pred. A valid detected tag cannot itself be occluded, so no separate check is needed.
    Returns None if too few pixels survive erosion (blurred-edge margin)."""
    _, valid, z = panel_layer(T_CW, tag_points(tag_size_m), camera)
    eroded = cv2.erode(valid.astype(np.uint8), np.ones((2*erode_px+1, 2*erode_px+1), np.uint8)).astype(bool)
    eroded &= np.isfinite(raw_depth) & (raw_depth > 0)
    if not np.any(eroded):
        return None
    return float(np.median(z[eroded]/raw_depth[eroded]))


def require_tracking_acceptance(report_path, config):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("kind") != "tracking_acceptance" or report.get("passed") is not True:
        raise ValueError("depth is gated: first pass experiments assess with controlled, physical and soak recordings")
    if report.get("fingerprint") != fingerprint(config):
        raise ValueError("tracking acceptance report does not match this calibration, tag, map or image size")


class MetricDepth:
    """Author-provided DA-V2 Small Hypersim metric implementation, no training."""
    def __init__(self, repository, checkpoint, input_size=252):
        metric_path = (Path(repository)/"metric_depth").resolve()
        if not (metric_path/"depth_anything_v2"/"dpt.py").is_file():
            raise ValueError("--depth-repo must point to the Depth-Anything-V2 author checkout")
        if input_size < 14 or input_size % 14:
            raise ValueError("depth input size must be a positive multiple of 14")
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sys.path.insert(0, str(metric_path))
        module = importlib.import_module("depth_anything_v2.dpt")
        if not Path(module.__file__).resolve().is_relative_to(metric_path):
            raise ValueError("a different depth_anything_v2 module is already imported; use the isolated depth environment")
        self.model = module.DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model = self.model.to(self.device).eval()
        self.input_size = input_size

    def predict(self, bgr):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with self.torch.inference_mode():
            depth = self.model.infer_image(bgr, self.input_size)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        return depth, {"inference_ms": (time.perf_counter()-start)*1000, "input_size": self.input_size,
                       "device": self.device, "gpu_peak_allocated_mb": self.torch.cuda.max_memory_allocated()/2**20 if self.device == "cuda" else None}


class RelativeDepth:
    """Step 43: the author's NON-metric DepthAnythingV2 (relative, disparity-like: larger = nearer) with the relative
    checkpoint. Separate from MetricDepth; run in its own process (both import a package called depth_anything_v2)."""
    def __init__(self, repository, checkpoint, input_size=252):
        root = Path(repository).resolve()
        if not (root/"depth_anything_v2"/"dpt.py").is_file():
            raise ValueError("--depth-repo must point to the Depth-Anything-V2 author checkout")
        if input_size < 14 or input_size % 14:
            raise ValueError("depth input size must be a positive multiple of 14")
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sys.path.insert(0, str(root))
        module = importlib.import_module("depth_anything_v2.dpt")
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("a different depth_anything_v2 module is already imported; use a separate process")
        self.model = module.DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model = self.model.to(self.device).eval()
        self.input_size = input_size

    predict = MetricDepth.predict


def anchor_sample(raw, x, y, radius=2, edge_range=.5):
    """Patch-median raw value around (x,y): a single pixel is noisy and can land exactly on a
    depth edge (object boundary) where the anchor's pixel location is ambiguous between near and
    far. Returns None (reject this anchor) if the patch is out of bounds, has no valid depth, or
    spans more than edge_range raw units (likely straddling an edge).
    ponytail: edge_range=.5 is a first-guess threshold on this checkpoint's raw units (range
    ~1-19 off-saturation, Step 27b), not calibrated against labelled edges; revisit if Step 32
    labelling shows it's rejecting good anchors or keeping bad ones."""
    y0, y1 = max(0, y-radius), min(raw.shape[0], y+radius+1)
    x0, x1 = max(0, x-radius), min(raw.shape[1], x+radius+1)
    patch = raw[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0)
    if not np.any(valid):
        return None
    vals = patch[valid]
    if float(vals.max()-vals.min()) > edge_range:
        return None
    return float(np.median(vals))


HIDDEN_MARGINS = (.02, .03, .05, .1, .15, .2)


def hidden_by_margin(scene, z, margins=HIDDEN_MARGINS):
    on = z > 0
    known = np.isfinite(scene[on]) & (scene[on] > 0)
    return {str(m): float(np.mean(known & (scene[on]+m < z[on]))) for m in margins}


def collect_anchors(raw, result, landmark_map, features):
    """(true camera-Z, patch-median model output) pairs at mapped landmark features, as DepthPrototype uses them."""
    anchor_z, anchor_raw = [], []
    if landmark_map is not None and result.raw_pose.landmark_inliers >= 12:
        lookup = {int(id): i for i, id in enumerate(landmark_map.ids)}
        for f in features:
            if f.landmark_id in lookup:
                x, y = np.rint(f.pixel).astype(int)
                if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0]:
                    sampled = anchor_sample(raw, x, y)
                    if sampled is not None:
                        z = depths(result.raw_pose.T_CW, landmark_map.points[lookup[f.landmark_id]][None])[0]
                        if z > 0:
                            anchor_z.append(float(z))
                            anchor_raw.append(sampled)
    return anchor_z, anchor_raw


def fit_affine_huber(z, raw):
    """raw = a + b*z, Huber-robust so a handful of outlier anchors (e.g. an occluder sitting on
    a mapped landmark) can't drag the line toward them the way plain least squares would.
    f_scale (the inlier/outlier boundary) is set from the initial OLS fit's own residual spread
    (1.4826*MAD, the usual robust-std estimator) rather than a fixed constant, since the depth
    model's raw-unit scale varies by scene/frame."""
    b0, a0 = np.polyfit(z, raw, 1)
    scale = max(float(np.median(np.abs(raw-(a0+b0*z))))*1.4826, 1e-6)
    fit = least_squares(lambda p: (p[0]+p[1]*z-raw)/scale, [a0, b0], loss="huber", f_scale=1.0)
    return float(fit.x[0]), float(fit.x[1])


class DepthPrototype:
    """Affine (scale+offset) fit of raw model output to true anchor distance: real_m = (raw-a)/b.
    Step 31 found the model's raw output is not proportional to distance (no fit through the
    origin); a straight-line fit with an intercept cut the same anchors' fit error from a 0.256 m
    mean (proportional) to 0.065 m (affine) -- see Instruction.md Step 30/31 log, 2026-10-24."""
    def __init__(self, model, output=None, margin_m=.02, invalid_occludes=False, slope_sign=-1, plane_correction=None, depth_frames=(), export_images=True):
        """slope_sign: expected sign of b (raw vs. true distance). -1 (default) matches every
        checkpoint tested so far, where raw falls as distance rises (disparity-like, Step 27b
        hypothesis: the relative-depth checkpoint was loaded by mistake). A genuine metric
        checkpoint is expected to give b>0; pass slope_sign=+1 for that case."""
        self.model = model
        self.fits = deque(maxlen=5)
        self.margin_m, self.invalid_occludes, self.slope_sign = margin_m, invalid_occludes, slope_sign
        self.plane = PlaneCorrector(plane_correction) if plane_correction else None  # Step 37, default off
        self.export_images = export_images  # False: per-frame json (+ depth_frames arrays) only, no mask/image/depth dumps
        self.depth_frames = set(depth_frames)  # frame indices whose raw/scene depth is saved (diagnostics)
        self.output = Path(output) if output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)

    def render(self, packet, result, panel, landmark_map, features):
        stats = {"status": "tracking unavailable", "synchronized_frame_index": packet.index}
        result.diagnostics["occlusion"] = stats
        if not result.raw_pose.valid or result.render_pose is None:
            self.fits.clear()
            return None
        raw, inference = self.model.predict(packet.bgr)
        raw = np.asarray(raw, np.float64)
        if raw.shape != packet.gray.shape:
            raise ValueError("depth model output must match the processed frame")
        stats.update(inference)
        anchor_z, anchor_raw = collect_anchors(raw, result, landmark_map, features)
        stats["scale_anchor_count"] = len(anchor_z)
        if len(anchor_z) >= 12:
            anchor_z, anchor_raw = np.array(anchor_z), np.array(anchor_raw)
            a, b = fit_affine_huber(anchor_z, anchor_raw)
            predicted = (anchor_raw-a)/b if b*self.slope_sign > 0 else None
            spread = float(np.median(np.abs(predicted-anchor_z)/anchor_z)) if predicted is not None else None
            stats["fit_relative_mad"] = spread
            if predicted is not None and spread <= .25:
                self.fits.append((a, b))
            else:
                self.fits.clear()
        else:
            self.fits.clear()
        anchor_scene = None
        if self.fits:
            a = float(np.median([f[0] for f in self.fits]))
            b = float(np.median([f[1] for f in self.fits]))
            anchor_scene = (raw-a)/b
            stats.update(depth_fit_a=a, depth_fit_b=b, slope_sign=self.slope_sign)
        corrected = None
        if self.plane:
            corrected, plane_stats = self.plane(raw, result.render_pose, panel, packet.camera, landmark_map.points if landmark_map is not None else None,
                                                b_prior=1/b if self.fits else None)
            stats.update(plane_stats)
        scene = corrected if corrected is not None else anchor_scene
        if scene is None:
            stats["status"] = "unavailable: metric scale lacks consistent mapped anchors"
            color, visible, z = panel_layer(result.render_pose, panel, packet.camera)
        else:
            color, visible, z = panel_layer(result.render_pose, panel, packet.camera, scene,
                                            margin_m=self.margin_m, invalid_occludes=self.invalid_occludes)
            stats["status"] = "metric depth active (prototype)" if corrected is None else "metric depth active (plane-corrected)"
            if np.any(z > 0):
                on = z > 0
                stats["hidden_fraction_by_margin"] = hidden_by_margin(scene, z)
                stats["hidden_fraction"] = float(np.mean(~visible[on]))
                stats["panel_z_median_m"] = float(np.median(z[on]))
                if anchor_scene is not None and corrected is not None:
                    stats["anchor_hidden_by_margin"] = hidden_by_margin(anchor_scene, z)
        if packet.index in self.depth_frames and self.output:
            name = f"{packet.index:06d}"
            np.save(self.output/(name+"-raw.npy"), raw.astype(np.float32), allow_pickle=False)
            cv2.imwrite(str(self.output/(name+"-image.png")), packet.bgr)
            for tag, arr in (("scene-anchor", anchor_scene), ("scene-plane", corrected)):
                if arr is not None:
                    np.save(self.output/(f"{name}-{tag}.npy"), arr.astype(np.float32), allow_pickle=False)
        stats["valid_depth_fraction"] = float(np.mean(np.isfinite(raw[z > 0]) & (raw[z > 0] > 0))) if np.any(z > 0) else None
        if self.output:
            name = f"{packet.index:06d}"
            if self.export_images:
                cv2.imwrite(str(self.output/(name+".png")), visible.astype(np.uint8)*255)
                cv2.imwrite(str(self.output/(name+"-image.png")), packet.bgr)
                np.save(self.output/(name+"-virtual-depth.npy"), z.astype(np.float32), allow_pickle=False)
            write_json(self.output/(name+".json"), {**stats, "source_time_s": packet.timestamp, "K": packet.camera.K,
                                                    "render_T_CW": result.render_pose, "units": "camera_Z_metres"})
        return color, visible


# Step 45: production version of the Step 44 variant A+L (relative model, mode A alignment, base-line reference).
# Constants copied from scripts/step44_baseline.py.
_EPS = 1e-3                                   # guard on d - t in the relative model's units
_LINE_MIN_PX, _LINE_MIN_CLEAN, _LINE_MAX_REL_MAD, _LINE_SPAN_MIN = 40, .30, .10, .15


def _depth_from(d, s, t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d-t > _EPS, s/(d-t), np.inf)


def _line_samples(T, panel, camera):
    """In-image samples (pixels, Z) of the panel-plane/table-plane line through the panel's two base corners."""
    base = panel[np.argsort(panel[:, 2])[:2]]
    u = (base[1]-base[0])/np.linalg.norm(base[1]-base[0])
    pc = (base[0]+np.arange(-3, 3, .005)[:, None]*u)@T[:3, :3].T+T[:3, 3]
    pc = pc[pc[:, 2] > .1]
    uv = pc@camera.K.T
    uv = uv[:, :2]/uv[:, 2:3]
    w, h = camera.size
    ok = (uv[:, 0] >= 2) & (uv[:, 0] < w-2) & (uv[:, 1] >= 2) & (uv[:, 1] < h-2)
    return np.rint(uv[ok]).astype(int), pc[ok, 2]


def _wls2(x, y, w):
    """Weighted least squares y = a + b*x via the 2x2 normal equations (much cheaper than lstsq on tens of thousands of points)."""
    sw, sx, sy = w.sum(), (w*x).sum(), (w*y).sum()
    sxx, sxy = (w*x*x).sum(), (w*x*y).sum()
    det = sw*sxx-sx*sx
    if abs(det) < 1e-12*max(sw*sxx, 1e-12):
        return np.array([sy/max(sw, 1e-12), 0.])
    return np.array([(sxx*sy-sx*sxy)/det, (sw*sxy-sx*sy)/det])


def _robust_inverse_fit(iz, d):
    """Huber IRLS fit d = t + s*iz (scale from MAD), then drop residuals beyond 2.5 sigma and refit.
    Returns ((t, s), inlier mask, sigma). Reference: scripts/step48_analyze.py robust_fit."""
    def irls(x, y):
        w = np.ones_like(y)
        for _ in range(15):
            p = _wls2(x, y, w*w)                       # weights enter squared, as in the lstsq(A*w, y*w) form
            r = y-p[0]-p[1]*x
            sc = 1.4826*np.median(np.abs(r))+1e-6
            w = np.sqrt(np.minimum(1., 1.5*sc/np.maximum(np.abs(r), 1e-9)))
        return p, sc
    p, sc = irls(iz, d)
    inliers = np.abs(d-p[0]-p[1]*iz) < 2.5*sc
    if inliers.sum() >= 10:
        p, sc = irls(iz[inliers], d[inliers])
        inliers = np.abs(d-p[0]-p[1]*iz) < 2.5*sc
    return p, inliers, sc


def _plane_grid(T, camera, extent, step=2):
    """Camera-Z of the world Z=0 plane on every step-th pixel (NaN where invalid) and the mask of pixels inside the
    world box `extent` = [x_min, x_max, y_min, y_max] that are closer than 5 m. Computed on the subsampled grid only."""
    rays = camera_rays(camera.size, tuple(camera.K.ravel()))[::step, ::step]
    normal_c = np.array([0., 0., 1.])@T[:3, :3].T
    offset = float(normal_c@T[:3, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        zg = offset/(rays@normal_c)
    valid = np.isfinite(zg) & (zg > 0)
    zg = np.where(valid, zg, np.nan)
    world = (rays*np.nan_to_num(zg)[..., None]-T[:3, 3])@T[:3, :3]
    x0, x1, y0, y1 = extent
    inside = valid & (zg < 5) & (world[..., 0] > x0) & (world[..., 0] < x1) & (world[..., 1] > y0) & (world[..., 1] < y1)
    return zg, inside


class BackgroundPrior:
    """Step 48b: stored static-background depth (1/4 resolution) kept in the last pose's camera frame and warped to the
    current pose (forward splat, nearest wins). A pixel is stored only where new depth agrees; a near blob is never
    inserted and nothing nearer than the panel is stored inside the panel footprint. Foreground = nearer than the stored
    background by near_ratio. Reference: scripts/step48_prior.py."""
    S = 4
    NEAR_ABS_M, RESET_AFTER = .6, 90

    def __init__(self, cfg):
        self.cfg, self.key = cfg, None
        self.reset()

    def reset(self):
        self.P = self.C = self.T = None
        self.lost = 0

    def _setup(self, camera):
        key = (tuple(camera.size), tuple(camera.K.ravel()))
        if key != self.key:
            self.key = key
            w, h = camera.size
            self.h, self.w = h//self.S, w//self.S
            K = camera.K.copy()
            K[:2] /= self.S
            K[:2, 2] = (camera.K[:2, 2]+.5)/self.S-.5
            self.K = K
            y, x = np.indices((self.h, self.w))
            self.rays = np.stack((x, y, np.ones_like(x)), axis=-1).reshape(-1, 3)@np.linalg.inv(K).T
            self.reset()

    def skipped(self):
        self.lost += 1
        if self.lost > self.RESET_AFTER:
            self.reset()

    def warp(self, T, camera):
        """(stored depth, count) in the current camera frame; empty on the first frame."""
        self._setup(camera)
        self.lost = 0
        empty = np.full((self.h, self.w), np.nan), np.zeros((self.h, self.w))
        if self.P is None:
            return empty
        ok = np.isfinite(self.P).ravel()
        X = self.rays[ok]*self.P.ravel()[ok, None]
        Xw = (X-self.T[:3, 3])@self.T[:3, :3]
        Xc = Xw@T[:3, :3].T+T[:3, 3]
        good = Xc[:, 2] > .1
        uv = Xc[good]@self.K.T
        u, v = np.rint(uv[:, 0]/uv[:, 2]).astype(int), np.rint(uv[:, 1]/uv[:, 2]).astype(int)
        z, c = Xc[good, 2], self.C.ravel()[ok][good]
        inside = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        u, v, z, c = u[inside], v[inside], z[inside], c[inside]
        order = np.argsort(-z)                      # farthest first, so the nearest point is written last
        Pn, Cn = empty
        Pn[v[order], u[order]] = z[order]
        Cn[v[order], u[order]] = c[order]
        return Pn, Cn

    def trusted(self, Pw, Cw):
        return np.isfinite(Pw) & (Cw >= self.cfg.min_count)

    def foreground(self, scene, Pw, Cw, size):
        """Full-resolution foreground mask: scene nearer than near_ratio x stored background, opened."""
        ref = cv2.resize(np.where(self.trusted(Pw, Cw), Pw, 0).astype(np.float32), tuple(size), interpolation=cv2.INTER_NEAREST)
        near = (ref > 0) & (scene < self.cfg.near_ratio*ref)
        k = np.ones((self.cfg.open_px, self.cfg.open_px), np.uint8)
        return cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool), ref

    def update(self, Zs, Pw, Cw, zplane_low, panel_z_low, T):
        """Zs: low-resolution aligned depth. zplane_low: plane depth where the plane is in view (NaN elsewhere)."""
        c = self.cfg
        trusted = self.trusted(Pw, Cw)
        ref = np.where(trusted, Pw, zplane_low)
        with np.errstate(invalid="ignore"):
            blob = ((np.isfinite(ref) & (Zs < .8*np.nan_to_num(ref, nan=99.))) | (Zs < self.NEAR_ABS_M)).astype(np.uint8)
            blob = cv2.dilate(blob, np.ones((5, 5), np.uint8)).astype(bool)
            Pn, Cn = Pw.copy(), Cw.copy()
            agree = np.isfinite(Pw) & (np.abs(Zs-Pw) < c.agree_tol*Pw)
            Pn[agree] = (Pw[agree]*Cn[agree]+Zs[agree])/(Cn[agree]+1)
            Cn[agree] = np.minimum(Cn[agree]+1, c.count_cap)
            new = ~np.isfinite(Pw) & np.isfinite(Zs) & (Zs < 8) & ~blob & ~((panel_z_low > 0) & (Zs < panel_z_low))
            Pn[new], Cn[new] = Zs[new], 1
            far = np.isfinite(Pw) & (Zs > 1.15*Pw)
            Cn[far] -= 1
        Pn[Cn <= 0] = np.nan
        self.P, self.C, self.T = Pn, Cn, T.copy()


class RelativeLineOcclusion:
    """Relative-depth occlusion. Mode A: Huber fit d = t + s/Z on mapped-landmark anchors (>= 12, 5-frame median hold).
    Line reference (optional): refit to the expected depth along the panel base line, shift moved to the farther
    side; gated, held for line_hold_frames, then plain mode A. Hidden when Z_real < Z_panel - max(floor, fraction*Z_panel).
    Interface identical to DepthPrototype.render. `enabled` can be switched at run time (key o)."""
    def __init__(self, model, output=None, margin_floor_m=.10, margin_fraction=.10, line_reference=True, line_hold_frames=30,
                 checkpoint_path=None, tracking_acceptance="passed", reference="line", plane_extent_m=(-1., 1., -.6, .9),
                 plane_min_span=.30, plane_min_px=3000, prior=None):
        self.model, self.floor, self.fraction = model, margin_floor_m, margin_fraction
        self.line_reference, self.hold_frames = line_reference, line_hold_frames
        self.reference, self.extent, self.min_span, self.min_px = reference, list(plane_extent_m), plane_min_span, plane_min_px
        self.prior = BackgroundPrior(prior) if prior is not None and prior.enabled else None
        self.plane_held, self.plane_age = None, 0
        self.fits, self.line_fit_held, self.line_age = deque(maxlen=5), None, 0
        self.enabled, self.keep_view, self.view, self._line_uv = True, False, None, None
        self.output = Path(output) if output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
        self.run_metadata = {"model": "relative", "checkpoint": Path(checkpoint_path).name if checkpoint_path else None,
                             "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() if checkpoint_path else None,
                             "input_size": model.input_size, "margin_floor_m": margin_floor_m, "margin_fraction": margin_fraction,
                             "line_reference": line_reference, "line_hold_frames": line_hold_frames, "tracking_acceptance": tracking_acceptance,
                             "reference": reference, "plane_extent_m": list(plane_extent_m), "plane_min_span": plane_min_span, "plane_min_px": plane_min_px,
                             "background_prior": asdict(prior) if prior is not None else None}

    def _anchor_fit(self, d, result, landmark_map, features):
        z, raw = collect_anchors(d, result, landmark_map, features)
        if len(z) < 12:
            return None
        z, raw = np.array(z), np.array(raw)
        t, s = fit_affine_huber(1/z, raw)
        zh = _depth_from(raw, s, t)
        return (t, s) if s > 0 and np.all(np.isfinite(zh)) and np.median(np.abs(zh-z)/z) <= .25 else None

    def _line_fit(self, d, T, panel, camera, t, s):
        uv, zl = _line_samples(T, panel, camera)
        self._line_uv = uv
        info = {"line_in_px": int(len(zl))}
        if len(zl) < _LINE_MIN_PX:
            return None, {**info, "line_reason": "line outside image"}
        dl = cv2.medianBlur(d.astype(np.float32), 3).astype(np.float64)[uv[:, 1], uv[:, 0]]
        iz = 1/zl
        span = float((iz.max()-iz.min())/iz.mean())
        if span >= _LINE_SPAN_MIN:
            t2, s2 = huber_lstsq(np.stack((np.ones(len(iz)), iz), axis=-1), dl)
        else:
            t2, s2 = t, s
        t2 += float(np.percentile(dl-(t2+s2*iz), 30))
        if s2 <= 0:
            return None, {**info, "line_reason": "scale not positive"}
        zh = _depth_from(dl, s2, t2)
        clean = np.isfinite(zh) & (zh >= zl-np.maximum(self.floor, self.fraction*zl))
        if clean.mean() < _LINE_MIN_CLEAN:
            return None, {**info, "line_reason": "line covered"}
        rel = float(np.median(np.abs(zh[clean]-zl[clean])/zl[clean]))
        info.update(line_rel_mad=rel, line_profile_span=span)
        if rel > _LINE_MAX_REL_MAD:
            return None, {**info, "line_reason": "MAD too large"}
        return (t2, s2), info

    def _plane_reference(self, d, T, camera, result, landmark_map, features, zg, inside, extra):
        """Step 48a: fit d = t + s/Z on world-plane pixels in the box plus mapped anchors (plus stored-map pixels when the
        plane gives fewer than plane_min_px). Gate on inlier span / fraction / s > 0; hold, then fall back to the anchor fit."""
        before = time.perf_counter()
        info = {}
        iz, dv = 1/zg[inside], d[::2, ::2][inside]
        info["plane_px"] = int(len(iz))
        stride = max(1, len(iz)//8000)                  # the fit does not need every pixel; keeps it to a few ms
        iz, dv = iz[::stride], dv[::stride]
        if len(iz) < self.min_px and extra is not None:
            iz, dv = np.concatenate((iz, 1/extra[0])), np.concatenate((dv, extra[1]))
            info["plane_prior_px"] = int(len(extra[0]))
        anchor_z, anchor_raw = collect_anchors(d, result, landmark_map, features)
        if anchor_z:
            iz, dv = np.concatenate((iz, 1/np.array(anchor_z))), np.concatenate((dv, np.array(anchor_raw)))
        fit, reason = None, None
        if len(iz) < 200:
            reason = "plane outside view"
        else:
            (t, s), inl, sc = _robust_inverse_fit(iz, dv)
            ii = iz[inl]
            span = float((np.percentile(ii, 95)-np.percentile(ii, 5))/np.median(ii)) if inl.any() else 0.
            info.update(plane_inlier_fraction=float(inl.mean()), plane_span=span, plane_rel_residual=float(sc/s) if s > 0 else None,
                        plane_se_rel_s=float(sc/(np.sqrt(inl.sum())*max(np.std(ii), 1e-9)*s)) if s > 0 and inl.sum() > 2 else None)
            if not (np.isfinite(t) and np.isfinite(s)) or s <= 0:
                reason = "scale not positive"
            elif span < self.min_span:
                reason = "inverse-depth span too small"
            elif inl.mean() < .5:
                reason = "too few inliers"
            else:
                fit = (float(t), float(s))
        info["plane_fit_ms"] = (time.perf_counter()-before)*1000
        if fit:
            self.plane_held, self.plane_age = fit, 0
            return fit, "plane_fresh", info
        info["plane_reason"] = reason
        self.plane_age += 1
        if self.plane_held and self.plane_age <= self.hold_frames:
            return self.plane_held, "plane_held", info
        if self.fits:
            return tuple(np.median(np.array(self.fits), axis=0)), "plane_fallback", info
        return None, "unavailable", info

    def render(self, packet, result, panel, landmark_map, features):
        stats = {"status": "tracking unavailable", "alignment_source": "unavailable"}
        result.diagnostics["occlusion"] = stats
        self.view = None
        if not result.raw_pose.valid or result.render_pose is None:
            self.fits.clear()
            self.line_fit_held = self.plane_held = None
            if self.prior is not None:
                self.prior.skipped()
            return None
        d, inference = self.model.predict(packet.bgr)
        d = np.asarray(d, np.float64)
        stats.update(inference)
        T, camera = result.render_pose, packet.camera
        fit = self._anchor_fit(d, result, landmark_map, features)
        self.fits.append(fit) if fit else self.fits.clear()
        Pw = Cw = ds = zg = inside = None
        if self.prior is not None or self.reference == "plane":
            zg, inside = _plane_grid(T, camera, self.extent)
        if self.prior is not None:
            Pw, Cw = self.prior.warp(T, camera)
            ds = cv2.resize(d, (self.prior.w, self.prior.h), interpolation=cv2.INTER_AREA)
        if self.reference == "plane":
            extra = None
            if self.prior is not None:
                trusted = self.prior.trusted(Pw, Cw)
                extra = (Pw[trusted], ds[trusted])
            params, source, info = self._plane_reference(d, T, camera, result, landmark_map, features, zg, inside, extra)
            stats.update(info)
            if params is None:
                stats["status"] = "visibility unavailable"
                color, visible, z = panel_layer(T, panel, camera)
                return color, visible
        else:
            if not self.fits:
                stats["status"] = "visibility unavailable"
                color, visible, z = panel_layer(T, panel, camera)
                return color, visible
            t, s = np.median(np.array(self.fits), axis=0)
            params, source = (t, s), "anchor"
            if self.line_reference:
                line, info = self._line_fit(d, T, panel, camera, t, s)
                stats.update(info)
                if line:
                    self.line_fit_held, self.line_age, params, source = line, 0, line, "line_fresh"
                else:
                    self.line_age += 1
                    if self.line_fit_held and self.line_age <= self.hold_frames:
                        params, source = self.line_fit_held, "line_held"
                    else:
                        source = "line_fallback"
        scene = _depth_from(d, params[1], params[0])
        scene_known = np.where(np.isfinite(scene), scene, 1e6)
        color, visible, z = panel_layer(T, panel, camera, scene_known, margin_m=self.floor, margin_fraction=self.fraction)
        fg = None
        if self.prior is not None:
            before = time.perf_counter()
            fg, _ = self.prior.foreground(scene, Pw, Cw, camera.size)
            hide_fg = fg & (z > 0) & (scene < z+self.prior.cfg.fg_slack)
            _, visible_fg, _ = panel_layer(T, panel, camera, np.where(hide_fg, 1e-3, scene_known), margin_m=self.floor, margin_fraction=self.fraction)
            stats.update(prior_coverage=float(np.mean(self.prior.trusted(Pw, Cw))), foreground_fraction=float(fg.mean()),
                         hidden_by_foreground=float(np.mean(visible[z > 0] & ~visible_fg[z > 0])) if (z > 0).any() else 0.)
            visible = visible_fg
            if source.endswith("fresh"):
                Zs = _depth_from(ds, params[1], params[0])
                zplane = np.where(inside, zg, np.nan)[1::2, 1::2]
                self.prior.update(Zs, Pw, Cw, zplane, z[2::4, 2::4], T)
            stats["prior_ms"] = (time.perf_counter()-before)*1000
        on = z > 0
        if self.keep_view:
            self.view = {"scene": scene, "z": z, "visible": visible, "line_uv": self._line_uv if self.line_reference and self.reference == "line" else None}
            if self.prior is not None:
                self.view.update(fg=fg, prior=cv2.resize(np.nan_to_num(np.where(self.prior.trusted(Pw, Cw), Pw, 0)).astype(np.float32), tuple(camera.size), interpolation=cv2.INTER_NEAREST))
        stats.update(status="metric depth active (relative, "+source+")", alignment_source=source, fit=[float(params[0]), float(params[1])])
        if on.any():
            zp = float(np.median(z[on]))
            stats.update(panel_z_median_m=zp, margin_m=max(self.floor, self.fraction*zp), hidden_fraction=float(np.mean(~visible[on])))
        if self.output:
            name = f"{packet.index:06d}"
            cv2.imwrite(str(self.output/(name+"-image.png")), packet.bgr)
            np.save(self.output/(name+"-scene.npy"), np.where(np.isfinite(scene), scene, 99.).astype(np.float32), allow_pickle=False)
        return color, visible


def mask_metrics(prediction, truth, tolerance=2):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be matching 2D visibility masks")
    union = np.count_nonzero(prediction | truth)
    iou = np.count_nonzero(prediction & truth)/union if union else 1.0
    a = prediction & ~cv2.erode(prediction.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    b = truth & ~cv2.erode(truth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    near_a = cv2.dilate(a.astype(np.uint8), kernel).astype(bool)
    near_b = cv2.dilate(b.astype(np.uint8), kernel).astype(bool)
    precision = np.count_nonzero(a & near_b)/np.count_nonzero(a) if np.any(a) else float(not np.any(b))
    recall = np.count_nonzero(b & near_a)/np.count_nonzero(b) if np.any(b) else float(not np.any(a))
    return {"iou": float(iou), "boundary_f1_at_2px": 2*precision*recall/(precision+recall) if precision+recall else 0.0}


def evaluate(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scores, transitions, times = [], [], []
    previous = None
    conditions = {}
    for item in manifest["frames"]:
        prediction = cv2.imread(str(path.parent/item["prediction"]), cv2.IMREAD_GRAYSCALE)
        truth = cv2.imread(str(path.parent/item["truth"]), cv2.IMREAD_GRAYSCALE)
        if prediction is None or truth is None:
            raise ValueError("missing prediction or manually annotated visibility mask")
        prediction, truth = prediction >= 128, truth >= 128
        score = mask_metrics(prediction, truth)
        scores.append(score)
        conditions.setdefault(item["condition"], []).append(score["iou"])
        if previous is not None:
            old, a, b = previous
            if item["sequence"] == old["sequence"] and item["frame_index"] == old["frame_index"]+1:
                transitions.append(float(np.mean((prediction ^ a) ^ (truth ^ b))))
        previous = item, prediction, truth
        if item.get("inference_ms") is not None:
            times.append(item["inference_ms"])
    if not scores:
        raise ValueError("no annotated frames")
    mean_iou = float(np.mean([s["iou"] for s in scores]))
    report = {"method": manifest["method"], "input_size": manifest.get("input_size"),
            "frames": len(scores), "mean_visibility_iou": mean_iou,
            "mean_boundary_f1_at_2px": float(np.mean([s["boundary_f1_at_2px"] for s in scores])),
            "condition_iou": {k: float(np.mean(v)) for k, v in conditions.items()},
            "temporal_transition_error_fraction": float(np.mean(transitions)) if transitions else None,
            "temporal_note": "mask-transition disagreement against labelled transitions; not a pose-warped flicker measure",
            "inference_p95_ms": percentile(times, 95), "iou_target_met": mean_iou >= .85,
            "front_and_behind_annotated": "front" in conditions and "behind" in conditions,
            "deployment_note": "IoU alone does not establish live FPS/latency or metric correctness"}
    live = summarize(path.parent/manifest["live_run"]) if manifest.get("live_run") else None
    report["live_performance"] = live
    report["deployment_checks"] = {
        "visibility_iou_085": mean_iou >= .85,
        "front_and_behind_annotated": report["front_and_behind_annotated"],
        "matching_live_input_size": live is not None and live["depth_input_sizes"] == [manifest.get("input_size")],
        "live_20fps": live is not None and (live["displayed_fps"] or 0) >= 20,
        "live_read_to_display_100ms": live is not None and live["read_to_display_p95_ms"] is not None and live["read_to_display_p95_ms"] <= 100,
        "metric_scale_available_95_percent": live is not None and live["occlusion_active_fraction"] >= .95}
    report["passed"] = all(report["deployment_checks"].values())
    return report


def select_resolution(reports):
    qualified = [r for r in reports if r.get("passed") and r.get("input_size") in (252, 392, 518)]
    winner = max(qualified, key=lambda r: (r["input_size"], r["mean_visibility_iou"])) if qualified else None
    return {"selected_input_size": winner["input_size"] if winner else None,
            "selected_method": winner["method"] if winner else None,
            "note": "highest evaluated resolution passing both labelled quality and measured live performance; none means leave occlusion disabled"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate visibility PNGs from metric depth or author Implicit Depth inference")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("manifest", help="JSON with method and frames: prediction, truth, sequence, frame_index, condition")
    evaluator.add_argument("--output", required=True)
    selection = commands.add_parser("select")
    selection.add_argument("reports", nargs="+")
    selection.add_argument("--output", required=True)
    args = parser.parse_args()
    report = (evaluate(args.manifest) if args.command == "evaluate" else
              select_resolution([json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]))
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
