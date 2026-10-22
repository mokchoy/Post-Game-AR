"""Numerical equivalence, bounded jobs, and lossless recording regressions."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from scipy.optimize import least_squares
import ar_projection.landmarks as landmarks
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.diagnostics import RunLogger
from ar_projection.features import FeatureTracker
from ar_projection.geometry import project, retract, tag_points, transform
from ar_projection.landmarks import Keyframe, MapBuilder, map_metadata
from ar_projection.pipeline import Pipeline
from ar_projection.tags import TagDetector
from ar_projection.types import CameraModel, FeatureObservation, FramePacket, LandmarkMap


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        base = load_config(Path(__file__).parent/"fixtures"/"tracking-config.json")
        self.cfg = replace(base, mapping=replace(base.mapping, max_keyframes=4, max_landmarks=40, min_ready_landmarks=40),
                           filter=replace(base.filter, enabled=False), display=replace(base.display, enabled=False))
        self.cam = CameraModel(np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1]]), np.zeros(5), (640, 480), {})
        self.T = transform([2.8, .1, .05], [0, 0, .9])
        rng = np.random.default_rng(44)
        self.X = rng.uniform([-.2, -.15, -.08], [.2, .15, .08], (40, 3))
        self.desc = rng.integers(0, 256, (40, 32), dtype=np.uint8)

    def keyframes(self):
        frames = []
        for j in range(3):
            T = self.T.copy()
            T[0, 3] += j*.07
            tag = TagDetector(self.cfg.tags).observation(0, project(T, tag_points(self.cfg.tags.size_m), self.cam), self.cam)
            features = {i: FeatureObservation(i, u, d.copy()) for i, (u, d) in
                        enumerate(zip(project(T, self.X, self.cam), self.desc))}
            frames.append(Keyframe(T, tag, features))
        return frames

    def views(self, frames):
        return [[(j, frame.features[i]) for j, frame in enumerate(frames)] for i in range(len(self.X))]

    def ready_map(self):
        return LandmarkMap(self.X.copy(), self.desc.copy(), np.arange(40, dtype=np.int64),
                           np.full(40, 3, np.int64), np.zeros(40), map_metadata(self.cfg, self.cam))

    def packet(self, index):
        gray = np.zeros((480, 640), np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return FramePacket(index, index*.03, bgr, bgr.copy(), gray, self.cam)

    def test_duplicate_filter_preserves_identity_and_radius_boundary(self):
        cfg = replace(self.cfg.features, max_features=20, redetect_every_n_frames=1, redetection_match_radius_px=5.)
        tracker = FeatureTracker(cfg)
        old = [FeatureObservation(i, np.array(p, float), self.desc[i], 17 if i == 0 else None)
               for i, p in enumerate([(100, 100), (102, 100), (300, 300)])]
        detected = [FeatureObservation(-1, np.array(p, float), self.desc[i]) for i, p in
                    enumerate([(100, 100), (107, 100), (107.00001, 100), (305, 300), (350, 300)])]
        tracker.tracks = old
        tracker.next_id = len(old)
        with patch.object(tracker, "flow", return_value=old), patch.object(tracker, "detect", return_value=detected), \
                patch("ar_projection.features.match_descriptors", return_value=[(0, 0, 0.0, 0.0)]):
            result = tracker.update(np.zeros((480, 640), np.uint8), [], 1)
        retained = old[1:]
        expected = retained+[f for f in detected if f.track_id == 0 or
                             min(np.linalg.norm(f.pixel-g.pixel) for g in retained) > cfg.redetection_match_radius_px]
        expected.sort(key=lambda f: f.landmark_id is None)
        self.assertEqual([f.track_id for f in result], [f.track_id for f in expected])
        self.assertEqual(sum(f.landmark_id == 17 for f in result), 1)
        self.assertEqual(len(result), 5)

    def test_batched_bundle_residual_matches_scalar_projection(self):
        self.cam.distortion[:] = [.01, -.02, .001, -.001, .01]
        builder = MapBuilder(self.cfg, self.cam)
        builder.keyframes = self.keyframes()
        views = self.views(builder.keyframes)
        def check(fun, x0, **kwargs):
            delta = np.random.default_rng(2).normal(0, 1e-4, len(x0))
            Ts = [retract(k.T, delta[6*j:6*j+6]) for j, k in enumerate(builder.keyframes)]
            Ps = self.X+delta[18:].reshape(-1, 3)
            expected = [((project(T, k.tag.world, self.cam)-k.tag.pixels)/self.cfg.tags.corner_sigma_px).ravel()
                        for T, k in zip(Ts, builder.keyframes)]
            expected += [(project(Ts[j], Ps[i:i+1], self.cam)[0]-f.pixel)/self.cfg.features.observation_sigma_px
                         for i, obs in enumerate(views) for j, f in obs]
            with patch.object(landmarks, "project", wraps=project) as calls:
                actual = fun(delta)
                self.assertEqual(calls.call_count, 6)
            np.testing.assert_allclose(actual, np.concatenate(expected), atol=1e-10, rtol=1e-10)
            return least_squares(fun, x0, **kwargs)
        with patch.object(landmarks, "least_squares", side_effect=check):
            result = builder._refine(self.X.copy(), views)
        np.testing.assert_allclose(result, self.X, atol=1e-9)
        self.assertEqual(builder.refinement_stats["outcome"], "accepted")

    def test_refinement_deadline_cancellation_and_solver_failure_are_distinct(self):
        for outcome in ("evaluation_budget_exceeded", "cancelled", "solver_failed"):
            cfg = replace(self.cfg, mapping=replace(self.cfg.mapping, refinement_max_nfev=1,
                                                   refinement_max_evaluations=0 if outcome == "evaluation_budget_exceeded" else 1000))
            builder = MapBuilder(cfg, self.cam)
            builder.keyframes = self.keyframes()
            original = builder.keyframes[0].T.copy()
            if outcome == "cancelled":
                builder.close()
            self.assertIsNone(builder._refine(self.X+.005, self.views(builder.keyframes)))
            self.assertEqual(builder.refinement_stats["outcome"], outcome)
            np.testing.assert_array_equal(builder.keyframes[0].T, original)
            if outcome == "solver_failed":
                self.assertEqual(builder.refinement_stats["optimizer_status"], 0)
                self.assertEqual(builder.refinement_stats["nfev"], 1)
                self.assertIn("maximum number", builder.refinement_stats["message"])

    def test_background_job_does_not_block_frames_and_publishes_without_tag(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        ready = self.ready_map()
        def build(job):
            calls.append(job)
            started.set()
            if not release.wait(3):
                raise RuntimeError("test worker was not released")
            job.map = ready
        pipe = Pipeline(self.cfg, self.cam, background_mapping=True)
        pipe.builder.keyframes = self.keyframes()
        try:
            with patch.object(MapBuilder, "_build", build):
                pipe.builder.poll()
                self.assertTrue(started.wait(2))
                # Changing a worker pose must not change the live builder yet.
                original = pipe.builder.keyframes[0].T.copy()
                calls[0].keyframes[0].T[0, 3] += .001
                np.testing.assert_array_equal(pipe.builder.keyframes[0].T, original)
                T = self.T.copy()
                T[0, 3] += .21
                tag = TagDetector(self.cfg.tags).observation(0, project(T, tag_points(self.cfg.tags.size_m), self.cam), self.cam)
                features = [FeatureObservation(i, u, d.copy()) for i, (u, d) in
                            enumerate(zip(project(T, self.X, self.cam), self.desc))]
                result = pipe.process(self.packet(0), [tag], features)
                self.assertTrue(result.raw_pose.valid)
                self.assertFalse(result.diagnostics["bootstrap_capacity_reached"])
                self.assertEqual(len(pipe.builder.keyframes), 4)
                self.assertEqual(len(calls), 1)
                release.set()
                pipe.builder._future.result(timeout=2)
                result = pipe.process(self.packet(1), [], [])
                self.assertTrue(result.diagnostics["map_ready"])
                self.assertIs(pipe.map, ready)
                self.assertFalse(result.raw_pose.valid)  # publication is not pose evidence
        finally:
            release.set()
            pipe.close()

    def test_background_retries_need_new_keyframe_and_cooldown(self):
        calls = []
        def fail(job):
            calls.append(len(job.keyframes))
            job.refinement_stats = {"outcome": "solver_failed"}
        builder = MapBuilder(self.cfg, self.cam, background=True)
        builder.keyframes = self.keyframes()
        try:
            with patch.object(MapBuilder, "_build", fail), patch.object(landmarks.time, "monotonic", return_value=100.) as clock:
                builder.poll()
                builder._future.result(timeout=2)
                builder.keyframes.append(self.keyframes()[0])
                builder.poll()
                self.assertIsNone(builder._future)
                self.assertTrue(builder.pending)
                clock.return_value = 103.
                builder.poll()
                builder._future.result(timeout=2)
                builder.poll()
                clock.return_value = 110.
                builder.poll()
                self.assertEqual(calls, [3, 4])
                self.assertFalse(builder.pending)
        finally:
            builder.close()

    def test_real_background_build_publishes_only_after_quality_checks(self):
        builder = MapBuilder(self.cfg, self.cam, background=True)
        builder.keyframes = self.keyframes()
        try:
            builder.poll()
            builder._future.result(timeout=3)
            builder.poll()
            self.assertIsNotNone(builder.map, builder.reason)
            self.assertEqual(builder.refinement_stats["outcome"], "accepted")
            np.testing.assert_allclose(builder.map.points, self.X[builder.map.ids], atol=1e-8)
            self.assertTrue(np.all(builder.map.support == 3))
            self.assertTrue(np.all(builder.map.rms <= self.cfg.mapping.max_reprojection_rms_px))
        finally:
            builder.close()

    def test_reset_cancels_old_job_and_discards_its_result(self):
        started = threading.Event()
        ready = self.ready_map()
        def wait_for_reset(job):
            started.set()
            job._cancel.wait(3)
            job.map = ready
        pipe = Pipeline(self.cfg, self.cam, background_mapping=True)
        pipe.builder.keyframes = self.keyframes()
        try:
            with patch.object(MapBuilder, "_build", wait_for_reset):
                pipe.builder.poll()
                self.assertTrue(started.wait(2))
                old = pipe.builder
                future = old._future
                pipe.reset()
                future.result(timeout=2)
                self.assertTrue(old._cancel.is_set())
                self.assertIsNot(pipe.builder, old)
                pipe.process(self.packet(0), [], [])
                self.assertIsNone(pipe.map)
                self.assertFalse(pipe.builder.keyframes)
        finally:
            pipe.close()

    def test_huffyuv_recording_is_pixel_exact_with_original_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            np.savez(p/"camera.npz", camera_matrix=self.cam.K, dist_coeffs=self.cam.distortion, image_size=[640, 480])
            cfg = replace(self.cfg, camera=replace(self.cfg.camera, calibration_path=str(p/"camera.npz")),
                          mapping=replace(self.cfg.mapping, mode="off"),
                          logging=replace(self.cfg.logging, enabled=True, input_codec="HFYU", output_dir=str(p)))
            logger = RunLogger(cfg, self.cam, "synthetic", source_fps=30.)
            pipe = Pipeline(cfg, self.cam)
            frames = np.random.default_rng(8).integers(0, 256, (3, 480, 640, 3), dtype=np.uint8)
            times = [10., 10.03, 10.9]
            for i, (frame, timestamp) in enumerate(zip(frames, times)):
                packet = self.packet(i)
                packet.original, packet.timestamp = frame, timestamp
                logger.video("input", frame)
                logger.frame(packet, pipe.process(packet, [], []))
            logger.close()
            cfg = replace(cfg.camera, source_type="video", video_path=str(logger.path/"input.avi"),
                          timestamps_path=str(logger.path/"frames.jsonl"))
            capture = Capture(cfg)
            try:
                for frame, timestamp in zip(frames, times):
                    packet = capture.read()
                    np.testing.assert_array_equal(packet.original, frame)
                    self.assertEqual(packet.timestamp, timestamp)
                self.assertIsNone(capture.read())
            finally:
                capture.close()
            self.assertIn("HFYU", json.loads((logger.path/"metadata.json").read_text())["recording"])


if __name__ == "__main__":
    unittest.main()
