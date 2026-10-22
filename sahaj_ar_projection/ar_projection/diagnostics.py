"""Finite JSONL for every frame, unique runs, conditional uncertainty labels."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import logging
import platform
import uuid
import cv2
import numpy as np
import scipy
from .geometry import camera_position, project


COVARIANCE_ASSUMPTIONS = {
    "label": "conditional_local_pose_covariance", "order": "[dt_x,dt_y,dt_z,dr_x,dr_y,dr_z]",
    "retraction": "Delta(dt,Rodrigues(dr)) @ T_CW; camera-side translation; not exact SE3 exponential",
    "conditional_on": "fixed calibration, fixed final map, assumed pixel noise, selected pose mode",
    "approximation": "inverse robust-weighted local information; per-coordinate rho-prime weights; not sandwich/posterior",
    "excluded": "calibration/map error, correlation, gross mismatch and multiple planar modes",
}


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite diagnostic value")
    return value


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_value(data), indent=2, allow_nan=False), encoding="utf-8")


def code_sha256():
    """Identity of the tracking code: SHA-256 over every ar_projection/*.py (names and bytes, sorted)."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()+path.read_bytes())
    return digest.hexdigest()


def association_rows(features, landmark_map, pose, camera, threshold_px):
    """One row per currently mapped observation; predicted pixel/inlier flag need a valid solver pose."""
    lookup = {int(id): p for id, p in zip(landmark_map.ids, landmark_map.points)} if landmark_map is not None else {}
    rows = []
    for f in features:
        if f.landmark_id is None:
            continue
        predicted, inlier = None, None
        if pose.valid and f.landmark_id in lookup:
            predicted = project(pose.T_CW, lookup[f.landmark_id][None], camera)[0]
            inlier = bool(np.linalg.norm(predicted-f.pixel) <= threshold_px)
        rows.append({"track_id": f.track_id, "landmark_id": int(f.landmark_id), "association_source": f.association_source,
                     "inherited": f.association_source in ("lk", "orb_redetection"),
                     "match_distance": f.match_distance, "match_ratio": f.match_ratio, "age_frames": f.age_frames,
                     "pixel": f.pixel, "predicted_pixel": predicted, "ransac_inlier": inlier})
    return rows


class RunLogger:
    def __init__(self, config, camera, timestamp_basis, map_metadata=None, source_fps=None):
        self.config = config
        self.path = None
        self.file = None
        self.assoc_file = None
        self.writers = {}
        self.source_fps = source_fps
        self.counts = Counter()
        self.times = []
        self.failure_start = None
        self.failure_durations = []
        self.last_time = None
        self.bootstrap = None
        if config.logging.enabled:
            self.path = Path(config.logging.output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8])
            self.path.mkdir(parents=True, exist_ok=False)
            write_json(self.path/"config.json", asdict(config))
            input_identity = {"type": config.camera.source_type, "index": config.camera.index, "path": config.camera.video_path}
            if config.camera.source_type == "video":
                stat = Path(config.camera.video_path).stat()
                input_identity.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            write_json(self.path/"metadata.json", {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                                                  "scipy": scipy.__version__, "calibration": camera.identity, "input": input_identity,
                                                  "actual_dimensions": camera.size, "timestamp_basis": timestamp_basis, "source_fps": source_fps,
                                                  "map": map_metadata, "covariance": COVARIANCE_ASSUMPTIONS,
                                                  "reproducibility": "seeded sequential replay; not bitwise across versions/backends", "code_sha256": code_sha256(),
                                                  "recording": f"{config.logging.input_codec} lossless input AVI; MJPG annotation; JSONL acquisition times authoritative"})
            self.file = (self.path/"frames.jsonl").open("x", encoding="utf-8")
            if config.logging.association_dump:
                self.assoc_file = (self.path/"assoc.jsonl").open("x", encoding="utf-8")

    def frame(self, packet, result):
        if self.file is None:
            return
        pose = result.raw_pose
        timestamp = packet.timestamp if np.isfinite(packet.timestamp) else None
        row = {"schema_version": 1, "frame_index": packet.index, "source_time_s": timestamp,
               "state": result.state, "actual_pose_source": pose.source, "raw_pose": asdict(pose), "render_T_CW": result.render_pose,
               "raw_camera_position_W": camera_position(pose.T_CW) if pose.valid else None,
               "render_camera_position_W": camera_position(result.render_pose) if result.render_pose is not None else None,
               "raw_panel_pixels": result.raw_panel, "render_panel_pixels": result.render_panel,
               "raw_panel_pixel_covariance": result.pixel_covariance, "render_covariance": None,
               "pixel_covariance_source": "raw_pose" if result.pixel_covariance is not None else None,
               "predicted_corner_rms_px": float(np.sqrt(np.mean(np.trace(result.pixel_covariance, axis1=1, axis2=2)))) if result.pixel_covariance is not None else None,
               "covariance_assumptions": COVARIANCE_ASSUMPTIONS,
               "diagnostics": result.diagnostics, "timings_ms": result.timings}
        self.file.write(json.dumps(json_value(row), allow_nan=False)+"\n")
        self.file.flush()
        if self.assoc_file is not None and result.association_observations is not None:
            self.assoc_file.write(json.dumps(json_value({"frame_index": packet.index, "source_time_s": timestamp,
                                                         "observations": result.association_observations}), allow_nan=False)+"\n")
            self.assoc_file.flush()
        self.counts[pose.source] += 1
        self.times.append(result.timings["total"])
        if result.diagnostics.get("mapping_mode") == "build":
            self.bootstrap = {key: result.diagnostics.get(key) for key in
                              ("bootstrap_reason", "bootstrap_refinement", "bootstrap_worker", "stable_landmarks", "keyframes", "map_ready")}
        if timestamp is not None:
            if not pose.valid and self.failure_start is None:
                self.failure_start = timestamp
            if pose.valid and self.failure_start is not None:
                self.failure_durations.append(max(0., timestamp-self.failure_start))
                self.failure_start = None
            self.last_time = timestamp

    def video(self, name, image):
        if self.path is None:
            return
        if name not in self.writers:
            fps = self.source_fps
            if fps is None or not np.isfinite(fps) or fps <= 0:
                fps = 30.
            codec = self.config.logging.input_codec if name == "input" else "MJPG"
            writer = cv2.VideoWriter(str(self.path/f"{name}.avi"), cv2.VideoWriter_fourcc(*codec), fps, (image.shape[1], image.shape[0]))
            if not writer.isOpened():
                writer.release()
                raise ValueError(f"cannot open {name} video writer")
            self.writers[name] = writer
        self.writers[name].write(image)

    def close(self, map_metadata=None, exit_reason="completed"):
        if self.file is None:
            return
        try:
            n = sum(self.counts.values())
            if self.failure_start is not None and self.last_time is not None:
                self.failure_durations.append(max(0., self.last_time-self.failure_start))
            summary = {"frames": n, "valid_pose_percent": 100*(n-self.counts["none"])/n if n else None,
                       "pose_source_counts": dict(self.counts), "failure_durations_s": self.failure_durations,
                       "median_processing_ms": float(np.median(self.times)) if n else None,
                       "p95_processing_ms": float(np.percentile(self.times, 95)) if n else None,
                       "throughput_fps": 1000/float(np.mean(self.times)) if n and np.mean(self.times) > 0 else None,
                       "throughput_note": "processing including video encoding, not capture-to-display latency",
                       "map": map_metadata, "bootstrap_at_last_frame": self.bootstrap, "exit_reason": exit_reason}
            write_json(self.path/"summary.json", summary)
            logging.info("Run saved to %s; valid poses %.1f%%", self.path, summary["valid_pose_percent"] or 0)
        finally:
            self.file.close()
            self.file = None
            if self.assoc_file is not None:
                self.assoc_file.close()
                self.assoc_file = None
            for writer in self.writers.values():
                writer.release()
