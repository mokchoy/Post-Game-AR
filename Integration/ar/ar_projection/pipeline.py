"""Frame orchestration; observations precede annotation, current evidence drives state."""
from dataclasses import replace
import logging
import time
import cv2
import numpy as np
from .camera import Capture
from .diagnostics import RunLogger, association_rows
from .features import FeatureTracker
from .geometry import project
from .landmarks import MapBuilder, associate, load_map, map_metadata, revalidate_ids, save_map
from .pose import PoseSolver
import queue
from .content import ArrayRuntime
from .gestures import GestureWorker
from .occlusion import advance_panel_video, panel_layer
from .filters import make_smoother
from .rendering import annotate, depth_pane, open_window, side_by_side, toggle_fullscreen, panel_covariance, panel_geometry, visible_panel
from .tags import TagDetector
from .types import FrameResult, PoseEstimate


class Pipeline:
    def __init__(self, config, camera, background_mapping=False):
        self.config = config
        self.camera = camera
        self.tags = TagDetector(config.tags)
        self.features = FeatureTracker(config.features)
        self.background_mapping = background_mapping
        self.builder = MapBuilder(config, camera, background_mapping) if config.mapping.mode == "build" else None
        self.map = load_map(config.mapping.map_path, map_metadata(config, camera), config) if config.mapping.mode == "localize" else None
        self.solver = PoseSolver(config)
        self.smoother = make_smoother(config.filter)
        self.panel = panel_geometry(config.panel)
        self.previous = None
        self.previous_time = None
        self.prior_time = None  # timestamp of the pose held in self.previous
        self.state = "BOOTSTRAP"
        self.last_tags, self.last_features = [], []
        self.map_quality = None

    def reset(self):
        if self.builder is not None:
            self.builder.close()
            self.builder = MapBuilder(self.config, self.camera, self.background_mapping)
            self.map = None
            self.map_quality = None
        self.previous, self.previous_time = None, None
        self.features.reset()
        self.smoother.reset()
        self.state = "BOOTSTRAP"

    def close(self):
        if self.builder is not None:
            self.builder.close()

    def save(self, replace_existing=False):
        if self.map is None:
            raise ValueError("map not ready; bootstrap has not met quality criteria")
        save_map(self.config.mapping.map_path, self.map, replace_existing)

    def process(self, packet, tags=None, features=None, withhold_tag=False):
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
        if invalid_time:
            self.features.reset()
        before = time.perf_counter()
        observations = self.tags.detect(packet.gray, packet.camera) if tags is None else tags
        timings["detection"] = (time.perf_counter()-before)*1000
        timings["tag_detection"] = timings["detection"]
        timings["orb_detection"] = 0.
        before = time.perf_counter()
        if features is None:
            features = self.features.update(packet.gray, observations, packet.index, force_detect=old_state == "LOST" or gap)
            feature_tracking = self.features.frame_stats
            timings["detection"] += self.features.detection_ms
            timings["orb_detection"] = self.features.detection_ms
            timings["tracking"] = max(0., (time.perf_counter()-before)*1000-self.features.detection_ms)
        else:
            feature_tracking = {"enabled": bool(self.config.features.enabled), "source": "provided"}
            timings["tracking"] = (time.perf_counter()-before)*1000
        self.last_tags, self.last_features = observations, features
        before = time.perf_counter()
        if self.builder is not None and self.map is None:
            self.builder.poll()
            if self.builder.map is None and not invalid_time:
                anchor = next((t for t in observations if t.id == self.config.tags.anchor_id and t.valid and not t.ambiguous), None)
                if anchor is not None:
                    self.builder.add(anchor.candidates[0], anchor, features)
                    self.builder.poll()
            self.map = self.builder.map
        associations = {"mapped_observations": 0, "reason": "map unavailable" if self.map is None else None}
        if self.map is not None and self.map_quality is None:
            self.map_quality = {"landmarks": len(self.map.ids), "support_min": int(np.min(self.map.support)),
                                "support_median": float(np.median(self.map.support)),
                                "reprojection_rms_median_px": float(np.median(self.map.rms)),
                                "world_extent_m": np.ptp(self.map.points, axis=0).tolist(),
                                "note": "map size/support are not proof of useful operating-view coverage"}
        revalidation = self.config.features.id_revalidation
        ids_cleared = revalidate_ids(features, self.map, revalidation, old_state == "LOST") if revalidation.mode != "off" else 0
        X, U = associate(features, self.map, self.config.features, associations, self.previous, packet.camera) if self.map is not None and self.config.features.enabled else (np.empty((0, 3)), np.empty((0, 2)))
        timings["mapping"] = (time.perf_counter()-before)*1000
        before = time.perf_counter()
        prior_age = packet.timestamp-self.prior_time if self.previous is not None and self.prior_time is not None else None
        pose = PoseEstimate(reason="invalid/nonincreasing source timestamp") if invalid_time else self.solver.estimate([] if withhold_tag else observations, X, U, packet.camera, self.previous, prior_age if self.previous is not None else dt)
        elapsed = (time.perf_counter()-before)*1000
        timings["pose"] = max(0., elapsed-pose.covariance_ms)
        timings["covariance"] = pose.covariance_ms
        association_observations = (association_rows(features, self.map, pose, packet.camera, self.config.pose.ransac_reprojection_threshold_px)
                                    if self.config.logging.association_dump else None)
        before = time.perf_counter()
        render_pose = raw_panel = render_panel = pixel_cov = None
        removed_associations = 0
        filter_reset = gap or not pose.valid
        if pose.valid:
            bootstrap = self.builder is not None and self.map is None
            self.state = "BOOTSTRAP" if bootstrap else "RELOCALIZING" if old_state == "LOST" or gap else "TRACKING"
            filter_reset = old_state == "LOST" or (old_state == "BOOTSTRAP" and self.state != "BOOTSTRAP") or gap
            if filter_reset:
                self.smoother.reset()
            render_pose = self.smoother.update(pose.T_CW, packet.timestamp, pose.covariance)
            raw_panel = visible_panel(pose.T_CW, self.panel, packet.camera)
            render_panel = visible_panel(render_pose, self.panel, packet.camera)
            pixel_cov = panel_covariance(pose.T_CW, self.panel, packet.camera, pose.covariance, self.config.uncertainty)
            self.previous = pose.T_CW.copy()
            self.prior_time = packet.timestamp
            # Remove unsupported map associations rather than carrying them through LK.
            if self.map is not None:
                lookup = {int(id): p for id, p in zip(self.map.ids, self.map.points)}
                for f in features:
                    if f.landmark_id in lookup:
                        if np.linalg.norm(project(pose.T_CW, lookup[f.landmark_id][None], packet.camera)[0]-f.pixel) > self.config.pose.ransac_reprojection_threshold_px:
                            f.landmark_id = None
                            removed_associations += 1
                        else:
                            f.frames_since_confirmed = 0
        else:
            self.state = "LOST"
            self.smoother.reset()
            if not (self.previous is not None and prior_age is not None and prior_age <= self.config.pose.prior_grace_s):
                self.previous = None
        self.previous_time = packet.timestamp if np.isfinite(packet.timestamp) else None
        timings["rendering"] = (time.perf_counter()-before)*1000
        timings["total"] = (time.perf_counter()-start)*1000
        anchor_tag = next((t for t in observations if t.id == self.config.tags.anchor_id), None)
        tag_log = {"candidates_detected": len(observations),
                   "anchor_min_side_px": float(np.min(np.linalg.norm(anchor_tag.pixels-np.roll(anchor_tag.pixels, 1, axis=0), axis=1))) if anchor_tag is not None else None,
                   "anchor_rms_px": anchor_tag.errors[0] if anchor_tag is not None and anchor_tag.errors else None,
                   "anchor_status": "not detected" if anchor_tag is None else (anchor_tag.reason or "ok")}
        diagnostics = {"tag_log": tag_log, "tags": [{"id": t.id, "valid": t.valid, "ambiguous": t.ambiguous, "reason": t.reason, "candidate_rms_px": t.errors} for t in observations],
                       "tracked_features": len(features), "mapped_observations": len(X), "map_size": len(self.map.ids) if self.map is not None else 0,
                       "stable_landmarks": self.builder.candidate_count if self.builder else (len(self.map.ids) if self.map is not None else 0),
                       "landmark_target": self.config.mapping.min_ready_landmarks,
                       "map_ready": self.map is not None, "bootstrap_reason": self.builder.reason if self.builder else None,
                       "bootstrap_refinement": dict(self.builder.refinement_stats) if self.builder else None,
                       "bootstrap_worker": self.builder.worker_stats() if self.builder else None,
                       "keyframes": len(self.builder.keyframes) if self.builder else 0,
                       "keyframe_limit": self.config.mapping.max_keyframes, "mapping_mode": self.config.mapping.mode,
                       "feature_tracking": feature_tracking,
                       "bootstrap_capacity_reached": self.builder is not None and self.map is None and not self.builder.pending and len(self.builder.keyframes) >= self.config.mapping.max_keyframes,
                       "configured_pose_mode": self.config.pose.mode,
                       "source_interval_s": dt if dt is None or np.isfinite(dt) else None,
                       "pose_prior_expired": bool(gap), "feature_tracker_reset": bool(invalid_time),
                       "reset_reason": "invalid timestamp" if invalid_time else "pose prior expired; LK retained, ORB forced" if gap else None,
                       "map_associations": associations, "pose_gates": pose.diagnostics,
                       "map_quality": self.map_quality,
                       "associations_removed_by_reprojection": removed_associations,
                       "ids_cleared_by_revalidation": ids_cleared, "prior_age_s": prior_age, "prior_used": prior_age is not None,
                       "tag_observations_withheld": bool(withhold_tag),
                       "timestamp_gap_reset": bool(gap), "raw_panel_visibility": "visible" if raw_panel is not None else "invalid, off-image or behind camera",
                       "filter_reset": bool(filter_reset)}
        diagnostics["panel_visibility"] = ("tracking lost" if not pose.valid else "visible" if render_panel is not None else
                                           "off-image or nonpositive depth")
        if withhold_tag:
            reference_config = replace(self.config, pose=replace(self.config.pose, mode="tag_only"),
                                       uncertainty=replace(self.config.uncertainty, enabled=False))
            reference = PoseSolver(reference_config).estimate(observations, [], [], packet.camera)
            reference_panel = visible_panel(reference.T_CW, self.panel, packet.camera) if reference.valid else None
            diagnostics["reference_panel_pixels"] = reference_panel
            diagnostics["tag_reference_valid"] = reference.valid
            diagnostics["tag_reference_note"] = "tag-reference agreement, not independent ground truth"
            diagnostics["tag_reference_rms_px"] = reference.tag_rms
            diagnostics["tag_reference_T_CW"] = reference.T_CW if reference.valid else None
        return FrameResult(self.state, pose, render_pose, raw_panel, render_panel, pixel_cov, diagnostics, timings, association_observations)


WINDOW = "AprilTag floating panel"


def run(config, save_when_ready=False, replace_map=False, max_frames=None, withhold_tag=False, withhold_tag_windows=None, occlusion=None,
        left_video=None, array_script=None):
    """withhold_tag_windows: optional [(start_s, end_s), ...]; the tag is withheld inside any
    window in addition to whenever withhold_tag is set. None (default) leaves behavior exactly
    as before this parameter existed. array_script: optional {frame_index: [(action, value), ...]} applied to the Step 50
    array (config.array.enabled) before that frame, for replays; left_video overrides array.left_video."""
    np.random.seed(config.seed)
    cv2.setRNGSeed(config.seed)
    capture = logger = pipeline = arr = worker = None
    display = config.display
    saved = False
    exit_reason = "completed"
    try:
        logging.info("Opening capture...")
        capture = Capture(config.camera)
        logging.info("Capture opened; waiting for the first frame...")
        packet = capture.read()
        logging.info("First frame %dx%d; initializing tracking...", *packet.camera.size)
        # Headless file replay remains sequential and reproducible. Live capture
        # and visible previews never run map construction/refinement inline.
        pipeline = Pipeline(config, packet.camera, background_mapping=config.camera.source_type == "camera" or display.enabled)
        logger = RunLogger(config, packet.camera, capture.timestamp_basis, pipeline.map.metadata if pipeline.map else None, capture.fps,
                           {"occlusion": occlusion.run_metadata} if getattr(occlusion, "run_metadata", None) else None)
        logging.info("Tracking started; input recording %s", config.logging.input_codec if config.logging.save_input_video else "disabled")
        side_view = gesture_pane = False
        if config.array.enabled:
            arr = ArrayRuntime(config, left_video)
            if config.gestures.enabled:
                events = queue.Queue()
                record = (logger.path is not None and config.camera.source_type == "camera"
                          and (config.logging.save_input_video or config.logging.save_annotated_video))
                worker = GestureWorker(config.gestures, events, logger.path/"gestures.jsonl" if logger.path is not None else None,
                                       logger.path if record else None)
                worker.start()
        if display.enabled:
            open_window(WINDOW)
            if worker is not None:                  # webcam_stable-style tuning, in the same window; key g hides the pane
                gesture_pane = worker.preview = True
                g = config.gestures
                for name, value, top in (("Threshold %", round(g.min_confidence*100), 100), ("Vote window", g.vote_window, 25), ("Required votes", g.required_votes, 25)):
                    cv2.createTrackbar(name, WINDOW, int(value), top, lambda _: None)
                    cv2.setTrackbarMin(name, WINDOW, 1)
        while packet is not None:
            withhold_this_frame = withhold_tag or (withhold_tag_windows is not None
                                                    and any(start <= packet.timestamp <= end for start, end in withhold_tag_windows))
            panelset = None
            if arr is not None:
                before = time.perf_counter()
                for action, value in (array_script or {}).get(packet.index, ()):
                    arr.do(action, value, "script")
                arr.drain(events if worker is not None else None)
                gesture_ms = (time.perf_counter()-before)*1000
                before = time.perf_counter()
                panelset = arr.panelset(packet.timestamp)
                pipeline.panel = panelset.centre_corners
                content_ms = (time.perf_counter()-before)*1000
            result = pipeline.process(packet, withhold_tag=withhold_this_frame)
            layer = None
            if occlusion is not None and not getattr(occlusion, "enabled", True):
                result.diagnostics["occlusion"] = {"status": "off (key o)", "alignment_source": "off"}
            if occlusion is not None and getattr(occlusion, "enabled", True):
                if arr is None:
                    advance_panel_video()
                before = time.perf_counter()
                layer = occlusion.render(packet, result, pipeline.panel if arr is None else panelset, pipeline.map, pipeline.last_features)
                extra = (time.perf_counter()-before)*1000
                result.timings["occlusion"] = extra
                result.timings["total"] += extra
            elif arr is not None and result.render_pose is not None:
                before = time.perf_counter()
                layer = panel_layer(result.render_pose, panelset, packet.camera)[:2]
                result.timings["array_render"] = (time.perf_counter()-before)*1000
                result.timings["total"] += result.timings["array_render"]
            if arr is not None:
                result.timings["gesture_queue"], result.timings["array_content"] = gesture_ms, content_ms
                result.timings["total"] += gesture_ms+content_ms
                ids = panelset.ids
                visible = {}
                if ids is not None:
                    n_all = np.bincount(ids[ids >= 0], minlength=len(panelset.names))
                    n_vis = np.bincount(ids[(ids >= 0) & layer[1]], minlength=len(panelset.names)) if layer is not None else None
                    visible = {n: (int(n_all[k]), float(n_vis[k]/n_all[k]) if n_vis is not None and n_all[k] else None) for k, n in enumerate(panelset.names)}
                result.diagnostics["array"] = {"yaw_deg": arr.state.yaw_deg, "scale": arr.state.scale, "game": arr.state.game, "playing": arr.state.playing,
                                               "actions": arr.take_log(), "video_frames": arr.content.frames, "panel_pixels": {n: v[0] for n, v in visible.items()},
                                               "panel_visible_fraction": {n: v[1] for n, v in visible.items()},
                                               "gesture_worker": None if worker is None else {"alive": worker.is_alive(), "error": worker.error, "loop_hz": worker.rate_hz}}
            image = shown = None
            if display.enabled or config.logging.save_annotated_video:
                before = time.perf_counter()
                image = annotate(packet, result, pipeline.last_tags, pipeline.last_features, display, config.camera.demo_approx, layer)
                extra = (time.perf_counter()-before)*1000
                result.timings["rendering"] += extra
                result.timings["total"] += extra
                if side_view and getattr(occlusion, "view", None):
                    shown = side_by_side(image, occlusion.view)
            before = time.perf_counter()
            if config.logging.save_input_video:
                logger.video("input", packet.original)
            if config.logging.save_annotated_video:
                logger.video("annotated", image)
                logger.video("rgb", packet.bgr)        # Step 52: rectified camera image, no overlay, frame-aligned with annotated
                if getattr(occlusion, "keep_view", False):
                    view = getattr(occlusion, "view", None)
                    logger.video("depth", depth_pane(view, image.shape))
                    # Step 52: this frame's aligned depth (shows the people), not the stored background used by the prior
                    logger.video("scene_depth", depth_pane(None if view is None else {**view, "prior": None}, image.shape))
            result.timings["recording"] = (time.perf_counter()-before)*1000
            result.timings["total"] += result.timings["recording"]
            if save_when_ready and not saved and pipeline.map is not None:
                pipeline.save(replace_map)
                saved = True
                logging.info("Frozen map saved to %s", config.mapping.map_path)
            key = None
            if display.enabled:
                shown = image if shown is None else shown
                if gesture_pane and worker.frame is not None:
                    pane = worker.frame
                    shown = np.hstack([shown, cv2.resize(pane, (round(pane.shape[1]*shown.shape[0]/pane.shape[0]), shown.shape[0]))])
                cv2.imshow(WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            result.timings["capture_read"] = packet.capture_read_ms
            result.timings["acquisition_to_display"] = ((now-packet.acquired_monotonic)*1000
                                                       if display.enabled and packet.acquired_monotonic is not None else None)
            result.timings["read_to_display"] = (packet.capture_read_ms+result.timings["acquisition_to_display"]
                                                 if result.timings["acquisition_to_display"] is not None else None)
            result.diagnostics["display_monotonic_s"] = now if display.enabled else None
            logger.frame(packet, result)
            if display.enabled:
                if key in (ord("q"), 27):
                    break
                if key == ord("d"):
                    display = replace(display, debug=not display.debug)
                if key == ord("f"):
                    toggle_fullscreen(WINDOW)
                if key == ord("v") and hasattr(occlusion, "keep_view"):
                    side_view = occlusion.keep_view = not occlusion.keep_view
                if key == ord("p") and logger.path is not None and shown is not None:
                    (logger.path/"sidebyside").mkdir(exist_ok=True)
                    cv2.imwrite(str(logger.path/"sidebyside"/f"{packet.index:06d}.png"), shown)
                if key == ord("o") and occlusion is not None and hasattr(occlusion, "enabled"):
                    occlusion.enabled = not occlusion.enabled
                    logging.info("Occlusion %s", "on" if occlusion.enabled else "off")
                if arr is not None:
                    arr.key(key)
                if worker is not None:
                    if key == ord("g"):
                        gesture_pane = worker.preview = not gesture_pane
                    if key in (ord("["), ord("]")):
                        cv2.setTrackbarPos("Threshold %", WINDOW, max(1, min(100, cv2.getTrackbarPos("Threshold %", WINDOW)+(5 if key == ord("]") else -5))))
                    window = max(1, cv2.getTrackbarPos("Vote window", WINDOW))
                    required = max(1, cv2.getTrackbarPos("Required votes", WINDOW))
                    if required > window:
                        required = window
                        cv2.setTrackbarPos("Required votes", WINDOW, required)
                    worker.request(cv2.getTrackbarPos("Threshold %", WINDOW)/100, window, required)
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
        logging.info("Interrupted; closing run" if logger is not None else "Interrupted before startup completed")
    except (ValueError, OSError, cv2.error):
        exit_reason = "failed"
        raise
    finally:
        if worker is not None:
            worker.stop()
            worker.join(timeout=3)
        if arr is not None:
            arr.close()
        if pipeline is not None:
            pipeline.close()
        if capture is not None:
            capture.close()
        if logger is not None:
            logger.close(pipeline.map.metadata if pipeline is not None and pipeline.map else None, exit_reason)
        if display.enabled:
            cv2.destroyAllWindows()
    return logger.path if logger is not None else None
