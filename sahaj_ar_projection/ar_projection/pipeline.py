"""Frame orchestration; observations precede annotation, current evidence drives state."""
from dataclasses import replace
import logging
import time
import cv2
import numpy as np
from .camera import Capture
from .diagnostics import RunLogger
from .features import FeatureTracker
from .geometry import project
from .landmarks import MapBuilder, associate, load_map, map_metadata, save_map
from .pose import PoseSolver
from .rendering import PoseSmoother, annotate, panel_covariance, panel_geometry, visible_panel
from .tags import TagDetector
from .types import FrameResult, PoseEstimate


class Pipeline:
    def __init__(self, config, camera):
        self.config = config
        self.camera = camera
        self.tags = TagDetector(config.tags)
        self.features = FeatureTracker(config.features)
        self.builder = MapBuilder(config, camera) if config.mapping.mode == "build" else None
        self.map = load_map(config.mapping.map_path, map_metadata(config, camera), config) if config.mapping.mode == "localize" else None
        self.solver = PoseSolver(config)
        self.smoother = PoseSmoother(config.filter)
        self.panel = panel_geometry(config.panel)
        self.previous = None
        self.previous_time = None
        self.state = "BOOTSTRAP"
        self.last_tags, self.last_features = [], []

    def reset(self):
        if self.builder is not None:
            self.builder = MapBuilder(self.config, self.camera)
            self.map = None
        self.previous, self.previous_time = None, None
        self.features.reset()
        self.smoother.reset()
        self.state = "BOOTSTRAP"

    def save(self, replace_existing=False):
        if self.map is None:
            raise ValueError("map not ready; bootstrap has not met quality criteria")
        save_map(self.config.mapping.map_path, self.map, replace_existing)

    def process(self, packet, tags=None, features=None):
        """Optional measured observations allow camera-free integration tests."""
        start = time.perf_counter()
        timings = {name: 0.0 for name in ("detection", "tracking", "mapping", "pose", "covariance", "rendering", "total")}
        if packet.camera.identity != self.camera.identity:
            raise ValueError("pipeline camera/preprocessing identity changed")
        old_state = self.state
        dt = packet.timestamp-self.previous_time if self.previous_time is not None else None
        invalid_time = not np.isfinite(packet.timestamp) or (dt is not None and dt <= 0)
        gap = invalid_time or (dt is not None and dt > self.config.pose.max_prediction_gap_s)
        if gap:
            self.previous = None
            self.smoother.reset()
            self.features.reset()
        before = time.perf_counter()
        observations = self.tags.detect(packet.gray, packet.camera) if tags is None else tags
        timings["detection"] = (time.perf_counter()-before)*1000
        before = time.perf_counter()
        if features is None:
            features = self.features.update(packet.gray, observations, packet.index, force_detect=old_state == "LOST" or gap)
            timings["detection"] += self.features.detection_ms
            timings["tracking"] = max(0., (time.perf_counter()-before)*1000-self.features.detection_ms)
        else:
            timings["tracking"] = (time.perf_counter()-before)*1000
        self.last_tags, self.last_features = observations, features
        before = time.perf_counter()
        if self.builder is not None and self.map is None and not invalid_time:
            anchor = next((t for t in observations if t.id == self.config.tags.anchor_id and t.valid and not t.ambiguous), None)
            if anchor is not None:
                self.builder.add(anchor.candidates[0], anchor, features)
                self.map = self.builder.map
        X, U = associate(features, self.map, self.config.features) if self.map is not None and self.config.features.enabled else (np.empty((0, 3)), np.empty((0, 2)))
        timings["mapping"] = (time.perf_counter()-before)*1000
        before = time.perf_counter()
        pose = PoseEstimate(reason="invalid/nonincreasing source timestamp") if invalid_time else self.solver.estimate(observations, X, U, packet.camera, self.previous, dt)
        elapsed = (time.perf_counter()-before)*1000
        timings["pose"] = max(0., elapsed-pose.covariance_ms)
        timings["covariance"] = pose.covariance_ms
        before = time.perf_counter()
        render_pose = raw_panel = render_panel = pixel_cov = None
        filter_reset = gap or not pose.valid
        if pose.valid:
            bootstrap = self.builder is not None and self.map is None
            self.state = "BOOTSTRAP" if bootstrap else "RELOCALIZING" if old_state == "LOST" or gap else "TRACKING"
            filter_reset = old_state == "LOST" or (old_state == "BOOTSTRAP" and self.state != "BOOTSTRAP") or gap
            if filter_reset:
                self.smoother.reset()
            render_pose = self.smoother.update(pose.T_CW)
            raw_panel = visible_panel(pose.T_CW, self.panel, packet.camera)
            render_panel = visible_panel(render_pose, self.panel, packet.camera)
            pixel_cov = panel_covariance(pose.T_CW, self.panel, packet.camera, pose.covariance, self.config.uncertainty)
            self.previous = pose.T_CW.copy()
            # Remove unsupported map associations rather than carrying them through LK.
            if self.map is not None:
                lookup = {int(id): p for id, p in zip(self.map.ids, self.map.points)}
                for f in features:
                    if f.landmark_id in lookup and np.linalg.norm(project(pose.T_CW, lookup[f.landmark_id][None], packet.camera)[0]-f.pixel) > self.config.pose.ransac_reprojection_threshold_px:
                        f.landmark_id = None
        else:
            self.state = "LOST"
            self.smoother.reset()
            self.previous = None
        self.previous_time = packet.timestamp if np.isfinite(packet.timestamp) else None
        timings["rendering"] = (time.perf_counter()-before)*1000
        timings["total"] = (time.perf_counter()-start)*1000
        diagnostics = {"tags": [{"id": t.id, "valid": t.valid, "ambiguous": t.ambiguous, "reason": t.reason, "candidate_rms_px": t.errors} for t in observations],
                       "tracked_features": len(features), "mapped_observations": len(X), "map_size": len(self.map.ids) if self.map is not None else 0,
                       "stable_landmarks": self.builder.candidate_count if self.builder else (len(self.map.ids) if self.map is not None else 0),
                       "landmark_target": self.config.mapping.min_ready_landmarks,
                       "map_ready": self.map is not None, "bootstrap_reason": self.builder.reason if self.builder else None,
                       "keyframes": len(self.builder.keyframes) if self.builder else 0,
                       "keyframe_limit": self.config.mapping.max_keyframes, "mapping_mode": self.config.mapping.mode,
                       "bootstrap_capacity_reached": self.builder is not None and self.map is None and len(self.builder.keyframes) >= self.config.mapping.max_keyframes,
                       "configured_pose_mode": self.config.pose.mode,
                       "timestamp_gap_reset": bool(gap), "raw_panel_visibility": "visible" if raw_panel is not None else "invalid, off-image or behind camera",
                       "filter_reset": bool(filter_reset)}
        return FrameResult(self.state, pose, render_pose, raw_panel, render_panel, pixel_cov, diagnostics, timings)


def run(config, save_when_ready=False, replace_map=False, max_frames=None):
    np.random.seed(config.seed)
    cv2.setRNGSeed(config.seed)
    capture = logger = pipeline = None
    display = config.display
    saved = False
    exit_reason = "completed"
    try:
        capture = Capture(config.camera)
        packet = capture.read()
        pipeline = Pipeline(config, packet.camera)
        logger = RunLogger(config, packet.camera, capture.timestamp_basis, pipeline.map.metadata if pipeline.map else None, capture.fps)
        while packet is not None:
            result = pipeline.process(packet)
            image = None
            if display.enabled or config.logging.save_annotated_video:
                before = time.perf_counter()
                image = annotate(packet, result, pipeline.last_tags, pipeline.last_features, display, config.camera.demo_approx)
                extra = (time.perf_counter()-before)*1000
                result.timings["rendering"] += extra
                result.timings["total"] += extra
            logger.frame(packet, result)
            if config.logging.save_input_video:
                logger.video("input", packet.original)
            if config.logging.save_annotated_video:
                logger.video("annotated", image)
            if save_when_ready and not saved and pipeline.map is not None:
                pipeline.save(replace_map)
                saved = True
                logging.info("Frozen map saved to %s", config.mapping.map_path)
            if display.enabled:
                cv2.imshow("AprilTag floating panel", image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("d"):
                    display = replace(display, debug=not display.debug)
                if key == ord("r"):
                    pipeline.reset()
                    saved = False
                if key == ord("s"):
                    try:
                        pipeline.save(replace_map)
                        logging.info("Map saved to %s", config.mapping.map_path)
                    except (ValueError, FileExistsError) as exc:
                        logging.warning("Map save refused: %s", exc)
            if max_frames is not None and packet.index+1 >= max_frames:
                break
            packet = capture.read()
        if save_when_ready and not saved:
            raise ValueError("--save-map requested but no ready map was produced; inspect bootstrap_reason in logs")
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    except (ValueError, OSError, cv2.error):
        exit_reason = "failed"
        raise
    finally:
        if capture is not None:
            capture.close()
        if logger is not None:
            logger.close(pipeline.map.metadata if pipeline is not None and pipeline.map else None, exit_reason)
        if display.enabled:
            cv2.destroyAllWindows()
