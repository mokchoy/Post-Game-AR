"""Optional, acceptance-gated metric RGB depth and diagnostic panel compositing.

Core geometry/evaluation use NumPy/OpenCV only. PyTorch is imported only when
the explicitly enabled model is constructed in the separate depth environment.
"""
import argparse
from collections import deque
from functools import lru_cache
import importlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .diagnostics import write_json
from .experiments import fingerprint, percentile, summarize
from .geometry import depths
from .rendering import visible_panel


@lru_cache(maxsize=4)
def camera_rays(size, intrinsics):
    w, h = size
    y, x = np.indices((h, w))
    return np.stack((x, y, np.ones_like(x)), axis=-1) @ np.linalg.inv(np.array(intrinsics).reshape(3, 3)).T


def panel_layer(T, panel, camera, scene_depth=None, margin_m=.02):
    """Return BGR texture, visibility, and camera-Z metres (zero outside panel)."""
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
    H = cv2.getPerspectiveTransform(np.array([[0, 0], [159, 0], [159, 95], [0, 95]], np.float32), pixels.astype(np.float32))
    color = cv2.warpPerspective(texture, H, (w, h))
    if scene_depth is not None:
        scene_depth = np.asarray(scene_depth)
        if scene_depth.shape != z.shape:
            raise ValueError("scene depth must match the rectified image dimensions")
        known = np.isfinite(scene_depth) & (scene_depth > 0)
        valid &= ~known | (scene_depth+margin_m >= z)
    return color, valid, z


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


class DepthPrototype:
    def __init__(self, model, output=None):
        self.model = model
        self.scales = deque(maxlen=5)
        self.output = Path(output) if output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)

    def render(self, packet, result, panel, landmark_map, features):
        stats = {"status": "tracking unavailable", "synchronized_frame_index": packet.index}
        result.diagnostics["occlusion"] = stats
        if not result.raw_pose.valid or result.render_pose is None:
            self.scales.clear()
            return None
        raw, inference = self.model.predict(packet.bgr)
        raw = np.asarray(raw, np.float64)
        if raw.shape != packet.gray.shape:
            raise ValueError("depth model output must match the processed frame")
        stats.update(inference)
        ratios = []
        if landmark_map is not None and result.raw_pose.landmark_inliers >= 12:
            lookup = {int(id): i for i, id in enumerate(landmark_map.ids)}
            for f in features:
                if f.landmark_id in lookup:
                    x, y = np.rint(f.pixel).astype(int)
                    if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0] and np.isfinite(raw[y, x]) and raw[y, x] > 0:
                        z = depths(result.raw_pose.T_CW, landmark_map.points[lookup[f.landmark_id]][None])[0]
                        if z > 0:
                            ratios.append(float(z/raw[y, x]))
        stats["scale_anchor_count"] = len(ratios)
        if len(ratios) >= 12:
            scale = float(np.median(ratios))
            spread = float(np.median(np.abs(np.asarray(ratios)-scale))/scale)
            stats["scale_relative_mad"] = spread
            if spread <= .25:
                self.scales.append(scale)
            else:
                self.scales.clear()
        else:
            self.scales.clear()
        if not self.scales:
            stats["status"] = "unavailable: metric scale lacks consistent mapped anchors"
            color, visible, z = panel_layer(result.render_pose, panel, packet.camera)
        else:
            scale = float(np.median(self.scales))
            color, visible, z = panel_layer(result.render_pose, panel, packet.camera, raw*scale)
            stats.update(status="metric depth active (prototype)", metric_scale=scale)
        stats["valid_depth_fraction"] = float(np.mean(np.isfinite(raw[z > 0]) & (raw[z > 0] > 0))) if np.any(z > 0) else None
        if self.output:
            name = f"{packet.index:06d}"
            cv2.imwrite(str(self.output/(name+".png")), visible.astype(np.uint8)*255)
            cv2.imwrite(str(self.output/(name+"-image.png")), packet.bgr)
            np.save(self.output/(name+"-virtual-depth.npy"), z.astype(np.float32), allow_pickle=False)
            write_json(self.output/(name+".json"), {**stats, "source_time_s": packet.timestamp, "K": packet.camera.K,
                                                    "render_T_CW": result.render_pose, "units": "camera_Z_metres"})
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
