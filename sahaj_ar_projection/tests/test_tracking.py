"""Seeded checks against known generating geometry; no webcam or GUI required."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from ar_projection.camera import Capture, Preprocessor, backend, load_calibration
from ar_projection.config import DEFAULT_PATH, load_config
from ar_projection.diagnostics import RunLogger
from ar_projection.features import FeatureTracker, feature_mask, match_descriptors
from ar_projection.geometry import camera_position, inverse, panel_points, project, projection_jacobian, retract, tag_points, transform, triangulate
from ar_projection.landmarks import MapBuilder, associate, load_map, map_metadata, save_map
from ar_projection.pipeline import Pipeline, run
from ar_projection.pose import PoseSolver, covariance
from ar_projection.rendering import PoseSmoother, panel_covariance, visible_panel
from ar_projection.tags import TagDetector
from ar_projection.types import CameraModel, FeatureObservation, FramePacket, LandmarkMap, TagObservation


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config()
        self.rng = np.random.default_rng(7)
        cv2.setRNGSeed(7)
        self.cam = CameraModel(np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1]]), np.zeros(5), (640, 480),
                               {"calibration_sha256": "synthetic", "approximate": False, "size": [640, 480], "undistort": True})
        self.T = transform([2.8, .1, .05], [0, 0, .9])
        self.X = self.rng.uniform([-.28, -.20, -.10], [.28, .20, .10], (60, 3))
        self.desc = self.rng.integers(0, 256, (60, 32), dtype=np.uint8)
        self.tag = self.make_tag(self.T)

    def make_tag(self, T):
        return TagDetector(self.c.tags).observation(0, self.generate(T, tag_points(.1)), self.cam)

    def generate(self, T, X):
        # Independent pinhole generator, not the project's project helper.
        xyz = X@T[:3, :3].T+T[:3, 3]
        normalized = xyz[:, :2]/xyz[:, 2:]
        return normalized*[800, 800]+[320, 240]

    def features(self, T, mapped=False, id_offset=0):
        return [FeatureObservation(i+id_offset, p, d.copy(), i if mapped else None) for i, (p, d) in enumerate(zip(self.generate(T, self.X), self.desc))]

    def ready_map(self, config=None):
        return LandmarkMap(self.X.copy(), self.desc.copy(), np.arange(60, dtype=np.int64), np.full(60, 3, np.int64), np.zeros(60), map_metadata(config or self.c, self.cam))

    def packet(self, index, timestamp=None):
        bgr = np.zeros((480, 640, 3), np.uint8)
        return FramePacket(index, index/30 if timestamp is None else timestamp, bgr, bgr.copy(), bgr[:, :, 0].copy(), self.cam)

    def test_geometry_conventions_and_default_panel(self):
        np.testing.assert_allclose(inverse(self.T)@self.T, np.eye(4), atol=1e-12)
        C = camera_position(self.T)
        np.testing.assert_allclose(self.T[:3, :3]@C+self.T[:3, 3], 0, atol=1e-12)
        np.testing.assert_allclose(tag_points(.1), [[-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0], [-.05, -.05, 0]])
        panel = panel_points(.2, .12, .15, 90)
        np.testing.assert_allclose(panel, [[-.1, .15, 0], [.1, .15, 0], [.1, .15, .12], [-.1, .15, .12]], atol=1e-12)
        np.testing.assert_allclose(project(self.T, self.X, self.cam), self.generate(self.T, self.X), atol=1e-10)
        self.assertIsNone(visible_panel(transform([0, 0, 0], [0, 0, -1]), panel, self.cam))
        self.assertIsNone(visible_panel(transform([0, 0, 0], [10, 0, 1]), panel, self.cam))

    def test_nonzero_distortion_preprocessing_and_triangulation(self):
        d = np.array([.15, -.05, .002, -.003, .01])
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"calibration.npz"
            np.savez(file, camera_matrix=self.cam.K, dist_coeffs=d, image_size=[640, 480])
            camera = replace(self.c.camera, calibration_path=str(file))
            raw = Preprocessor(replace(camera, undistort=False), self.cam.size)
            undistorted = Preprocessor(camera, self.cam.size)
            pixels = cv2.projectPoints(self.X, cv2.Rodrigues(self.T[:3, :3])[0], self.T[:3, 3], self.cam.K, d)[0].reshape(-1, 2)
            np.testing.assert_allclose(project(self.T, self.X, raw.camera), pixels)
            clean = cv2.undistortPoints(pixels[:, None], self.cam.K, d, P=self.cam.K).reshape(-1, 2)
            np.testing.assert_allclose(project(self.T, self.X, undistorted.camera), clean, atol=1e-4)
            self.assertEqual(undistorted.packet(0, 0., np.zeros((480, 640, 3), np.uint8)).bgr.shape, (480, 640, 3))
            T2 = self.T.copy()
            T2[0, 3] += .08
            x = self.X[0]
            point = triangulate(self.T, T2, project(self.T, x[None], raw.camera)[0], project(T2, x[None], raw.camera)[0], raw.camera, self.c.mapping)
            np.testing.assert_allclose(point, x, atol=1e-6)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                load_calibration(file, (320, 240))

    def test_tag_validation_and_ambiguity(self):
        self.assertTrue(self.tag.valid)
        self.assertFalse(self.tag.ambiguous)
        np.testing.assert_allclose(self.tag.candidates[0], self.T, atol=1e-9)
        small = self.generate(transform([2.8, .1, .05], [0, 0, 5]), tag_points(.1))
        self.assertFalse(TagDetector(self.c.tags).observation(0, small, self.cam).valid)
        ambiguous = replace(self.tag, ambiguous=True)
        result = PoseSolver(self.c).estimate([ambiguous], [], [], self.cam)
        self.assertFalse(result.valid)
        self.assertIn("ambiguity", result.reason)
        builder = MapBuilder(self.c, self.cam)
        builder.add(self.T, ambiguous, self.features(self.T))
        self.assertEqual(len(builder.keyframes), 0)
        solved = PoseSolver(self.c).estimate([ambiguous], self.X, self.generate(self.T, self.X), self.cam)
        self.assertTrue(solved.valid)

    def test_translated_mapping_support_bounds_refinement_and_freeze(self):
        for refine in (False, True):
            config = replace(self.c, mapping=replace(self.c.mapping, max_landmarks=45, refine_bootstrap_map=refine))
            builder = MapBuilder(config, self.cam)
            for index in range(3):
                T = self.T.copy()
                T[0, 3] += index*.065
                builder.add(T, self.make_tag(T), self.features(T, id_offset=index*100))
                if index < 2:
                    self.assertIsNone(builder.map)
            self.assertIsNotNone(builder.map, builder.reason)
            self.assertEqual(len(builder.map.ids), 45)
            expected = self.X[builder.map.ids]
            np.testing.assert_allclose(builder.map.points, expected, atol=1e-7)
            self.assertTrue(np.all(builder.map.support >= 3))
            before = builder.map.points.copy()
            for _ in range(15):
                builder.add(self.T, self.tag, self.features(self.T))
            self.assertEqual(len(builder.keyframes), 3)
            np.testing.assert_array_equal(builder.map.points, before)
            features = self.features(self.T, id_offset=1000)
            X, U = associate(features, builder.map, config.features)
            self.assertEqual(len(X), 45)
            self.assertEqual(len({f.landmark_id for f in features if f.landmark_id is not None}), 45)

    def test_bootstrap_refinement_retries_after_new_keyframe(self):
        config = replace(self.c, mapping=replace(self.c.mapping, max_landmarks=45, min_ready_landmarks=45,
                                                  max_keyframes=5, refine_bootstrap_map=True))
        builder = MapBuilder(config, self.cam)
        attempts = []

        def fail_once(points, views):
            attempts.append(len(builder.keyframes))
            return None if len(attempts) == 1 else points

        with patch.object(builder, "_refine", side_effect=fail_once):
            for index in range(4):
                T = self.T.copy()
                T[0, 3] += index*.065
                builder.add(T, self.make_tag(T), self.features(T, id_offset=index*100))

        self.assertEqual(attempts, [3, 4])
        self.assertIsNotNone(builder.map, builder.reason)

    def test_bad_triangulation_and_incomplete_capacity(self):
        x = self.X[0]
        T2 = transform([.2, 0, 0], [0, 0, 0])@self.T  # rotate camera at same centre
        self.assertIsNone(triangulate(self.T, T2, self.generate(self.T, x[None])[0], self.generate(T2, x[None])[0], self.cam, self.c.mapping))
        T3 = self.T.copy()
        T3[0, 3] += .08
        behind = np.array([[0., 0, 2.]])
        self.assertIsNone(triangulate(self.T, T3, self.generate(self.T, behind)[0], self.generate(T3, behind)[0], self.cam, self.c.mapping))
        far = np.array([[0., 0, -50.]])
        self.assertIsNone(triangulate(self.T, T3, self.generate(self.T, far)[0], self.generate(T3, far)[0], self.cam, self.c.mapping))
        distant = (inverse(self.T)[:3, :3]@np.array([0., 0., 4.5])+camera_position(self.T))[None]
        self.assertIsNone(triangulate(self.T, T3, self.generate(self.T, distant)[0], self.generate(T3, distant)[0], self.cam, self.c.mapping))
        builder = MapBuilder(self.c, self.cam)
        for i in range(self.c.mapping.max_keyframes):
            T = self.T.copy()
            T[0, 3] += i*.05
            # Fixtures retain valid tag observations; empty features cannot create a map.
            tag = TagObservation(0, self.generate(T, tag_points(.1)), tag_points(.1), [T], [0.], False, True)
            builder.add(T, tag, [])
        self.assertEqual(len(builder.keyframes), self.c.mapping.max_keyframes)
        self.assertIsNone(builder.map)
        self.assertIn("capacity", builder.reason)

    def test_joint_outliers_ablation_quality_and_motion(self):
        U = self.generate(self.T, self.X)+self.rng.normal(0, .2, (60, 2))
        U[:18] = self.rng.uniform([0, 0], [640, 480], (18, 2))
        solver = PoseSolver(self.c)
        result = solver.estimate([self.tag], self.X, U, self.cam)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.source, "joint")
        self.assertLess(result.landmark_inliers, 50)
        self.assertLess(np.linalg.norm(camera_position(result.T_CW)-camera_position(self.T)), .004)
        self.assertLess(result.landmark_rms, 1)
        for mode, source in (("tag_only", "tag"), ("landmark_only", "landmark")):
            config = replace(self.c, pose=replace(self.c.pose, mode=mode))
            pose = PoseSolver(config).estimate([self.tag], self.X, U, self.cam)
            self.assertTrue(pose.valid, pose.reason)
            self.assertEqual(pose.source, source)
        absent = solver.estimate([], [], [], self.cam, self.T, .03)
        self.assertFalse(absent.valid)
        self.assertIsNone(absent.T_CW)
        wrong = replace(self.tag, pixels=self.tag.pixels+[40, 0])
        result = solver.estimate([wrong], self.X, self.generate(self.T, self.X), self.cam)
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "landmark")
        jump = self.T.copy()
        jump[0, 3] += .1
        gated = solver.estimate([], self.X, self.generate(jump, self.X), self.cam, self.T, .001)
        self.assertFalse(gated.valid)
        recovered = solver.estimate([], self.X, self.generate(jump, self.X), self.cam, self.T, 1.)
        self.assertTrue(recovered.valid)

    def test_covariance_noise_rank_and_pixel_monte_carlo(self):
        config = replace(self.c, pose=replace(self.c.pose, robust_loss="linear"))
        U = self.generate(self.T, self.X)
        cov1, reason = covariance(self.T, self.X, U, np.ones(60), self.cam, config)
        self.assertIsNone(reason)
        cov2, _ = covariance(self.T, self.X, U, np.full(60, 2.), self.cam, config)
        np.testing.assert_allclose(cov2, 4*cov1, rtol=1e-8)
        singular, reason = covariance(self.T, np.tile(self.X[0], (12, 1)), np.tile(U[0], (12, 1)), np.ones(12), self.cam, config)
        self.assertIsNone(singular)
        self.assertIn("rank", reason)
        panel = panel_points(.2, .12, .15, 90)
        predicted = panel_covariance(self.T, panel, self.cam, cov1, config.uncertainty)
        samples = self.rng.multivariate_normal(np.zeros(6), cov1, size=1500)
        pixels = np.array([project(retract(self.T, d), panel, self.cam) for d in samples])
        measured = np.array([np.cov(pixels[:, i].T) for i in range(4)])
        np.testing.assert_allclose(measured, predicted, rtol=.20, atol=.005)
        disabled = replace(config, uncertainty=replace(config.uncertainty, enabled=False))
        self.assertIsNone(covariance(self.T, self.X, U, np.ones(60), self.cam, disabled)[0])

    def test_map_safe_roundtrip_compatibility_and_no_overwrite(self):
        m = self.ready_map()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"map.npz"
            save_map(file, m)
            loaded = load_map(file, m.metadata, self.c)
            for name in ("points", "descriptors", "ids", "support", "rms"):
                np.testing.assert_array_equal(getattr(m, name), getattr(loaded, name))
            self.assertEqual(loaded.metadata["units"], "metres")
            with self.assertRaises(FileExistsError):
                save_map(file, m)
            save_map(file, m, replace=True)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_map(file, {**m.metadata, "tag_size_m": .2}, self.c)
            np.savez(file, points=np.array([{}], object), descriptors=m.descriptors, ids=m.ids, support=m.support, rms=m.rms,
                     metadata=np.array(json.dumps(m.metadata)))
            with self.assertRaisesRegex(ValueError, "Object arrays"):
                load_map(file, m.metadata, self.c)

    def test_strict_config_combinations_paths_and_types(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"config.json"
            original = json.loads(DEFAULT_PATH.read_text())
            invalid = [("features", "backend", "xfeat"), ("camera", "backend", "unknown"), ("pose", "mode", "unknown"),
                       ("tags", "size_m", -.1), ("features", "max_features", True), ("features", "enabled", False),
                       ("tags", "enabled", False), ("camera", "demo_approx", True), ("mapping", "min_confirming_views", 2),
                       ("features", "exclude_rectangles", [[0, 0, -1, 20]]), ("pose", "mode", "landmark_only"),
                       ("camera", "index", "0"), ("panel", "typo", 1), ("filter", "measurement_weight", float("nan"))]
            for section, key, value in invalid:
                data = json.loads(json.dumps(original))
                data[section][key] = value
                file.write_text(json.dumps(data))
                with self.assertRaises(ValueError, msg=f"{section}.{key}"):
                    load_config(file)
            file.write_text(json.dumps(original))
            config = load_config(file)
            self.assertEqual(config.mapping.map_path, str(Path(directory)/"data"/"landmark-map.npz"))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                backend("invalid")

    def test_features_lk_descriptor_only_masks_and_reciprocity(self):
        gray = self.rng.integers(0, 256, (480, 640), dtype=np.uint8)
        cfg = replace(self.c.features, max_features=120, redetect_below_tracks=20, exclude_rectangles=[[-5, -5, 25, 25], [900, 900, 10, 10]])
        mask = feature_mask(gray.shape, [self.tag], cfg)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[20, 20], 255)
        for enabled in (True, False):
            tracker = FeatureTracker(replace(cfg, lk=replace(cfg.lk, enabled=enabled)))
            a = tracker.update(gray, [self.tag], 0)
            b = tracker.update(gray, [self.tag], 1)
            self.assertLessEqual(len(b), 120)
            self.assertGreater(len(set(f.track_id for f in a)&set(f.track_id for f in b)), 40)
            self.assertEqual(len(b), len({f.track_id for f in b}))
            self.assertTrue(all(mask[int(f.pixel[1]), int(f.pixel[0])] for f in b))
        pairs = match_descriptors(self.desc, self.desc, cfg)
        self.assertEqual(len(pairs), 60)
        self.assertTrue(all(a == b for a, b in pairs))

    def test_occlusion_loss_recovery_and_headless_observation_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.c, mapping=replace(self.c.mapping, mode="localize", map_path=str(Path(directory)/"map.npz")),
                             logging=replace(self.c.logging, output_dir=str(Path(directory)/"runs")), display=replace(self.c.display, enabled=False),
                             filter=replace(self.c.filter, enabled=True))
            save_map(config.mapping.map_path, self.ready_map(config))
            pipe = Pipeline(config, self.cam)
            logger = RunLogger(config, self.cam, "synthetic frame_index/30", pipe.map.metadata)
            sources, states = [], []
            for i, (tags, features) in enumerate([([self.tag], self.features(self.T)), ([], self.features(self.T)), ([], []), ([self.tag], self.features(self.T))]):
                result = pipe.process(self.packet(i), tags, features)
                logger.frame(self.packet(i), result)
                sources.append(result.raw_pose.source)
                states.append(result.state)
                if i == 2:
                    self.assertIsNone(result.render_panel)
                    self.assertIsNone(result.render_pose)
                    self.assertIsNone(pipe.smoother.previous)
                if i == 3:
                    np.testing.assert_allclose(result.render_pose, result.raw_pose.T_CW, atol=1e-12)
            logger.close(pipe.map.metadata)
            self.assertEqual(sources, ["joint", "landmark", "none", "joint"])
            self.assertEqual(states, ["TRACKING", "TRACKING", "LOST", "RELOCALIZING"])
            rows = [json.loads(line, parse_constant=lambda s: self.fail(s)) for line in (logger.path/"frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertIsNone(rows[2]["raw_pose"]["T_CW"])
            self.assertTrue(all(r["render_covariance"] is None for r in rows))
            self.assertEqual(json.loads((logger.path/"summary.json").read_text())["valid_pose_percent"], 75.)
            # Loaded-map localisation is functional with tag detection disabled.
            tagless = replace(config, tags=replace(config.tags, enabled=False), pose=replace(config.pose, mode="landmark_only"))
            result = Pipeline(tagless, self.cam).process(self.packet(0), [], self.features(self.T))
            self.assertTrue(result.raw_pose.valid, result.raw_pose.reason)
            self.assertEqual(result.raw_pose.source, "landmark")

    def test_disabled_side_effects_and_timestamp_filter_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.c, mapping=replace(self.c.mapping, mode="off"), logging=replace(self.c.logging, enabled=False, output_dir=str(Path(directory)/"runs")),
                             display=replace(self.c.display, enabled=False), filter=replace(self.c.filter, enabled=True))
            with patch.object(cv2, "imshow", side_effect=AssertionError("GUI")), patch.object(cv2, "waitKey", side_effect=AssertionError("GUI")):
                pipe = Pipeline(config, self.cam)
                logger = RunLogger(config, self.cam, "synthetic")
                first = pipe.process(self.packet(0), [self.tag], [])
                logger.frame(self.packet(0), first)
                second = pipe.process(self.packet(1, timestamp=1.), [self.tag], [])
                self.assertTrue(second.diagnostics["timestamp_gap_reset"])
                np.testing.assert_allclose(second.raw_pose.T_CW, second.render_pose, atol=1e-12)
                invalid = pipe.process(self.packet(2, timestamp=1.), [self.tag], [])
                self.assertFalse(invalid.raw_pose.valid)
                logger.close()
                self.assertFalse(Path(config.logging.output_dir).exists())
        smoother = PoseSmoother(replace(self.c.filter, enabled=True))
        A, B = transform([0, 0, np.deg2rad(179)], [0, 0, 1]), transform([0, 0, np.deg2rad(-179)], [.1, 0, 1])
        smoother.update(A)
        blended = smoother.update(B)
        self.assertGreater(abs(cv2.Rodrigues(blended[:3, :3])[0][2, 0]), 3.)

    def test_full_headless_recorded_video_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            calibration = p/"calibration.npz"
            np.savez(calibration, camera_matrix=self.cam.K, dist_coeffs=np.zeros(5), image_size=[640, 480])
            video = p/"input.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
            self.assertTrue(writer.isOpened())
            marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 400)
            H = cv2.getPerspectiveTransform(np.array([[0, 0], [399, 0], [399, 399], [0, 399]], np.float32), self.tag.pixels.astype(np.float32))
            for present in (True, True, False, True):
                image = cv2.warpPerspective(marker, H, (640, 480), borderValue=255) if present else np.full((480, 640), 255, np.uint8)
                writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            writer.release()
            config = replace(self.c, camera=replace(self.c.camera, source_type="video", video_path=str(video), calibration_path=str(calibration)),
                             mapping=replace(self.c.mapping, mode="off"), pose=replace(self.c.pose, mode="tag_only"),
                             display=replace(self.c.display, enabled=False), logging=replace(self.c.logging, output_dir=str(p/"runs"), save_input_video=True, save_annotated_video=True))
            with patch.object(cv2, "imshow", side_effect=AssertionError("GUI")), patch.object(cv2, "waitKey", side_effect=AssertionError("GUI")), patch.object(cv2, "destroyAllWindows", side_effect=AssertionError("GUI")):
                run(config)
            rows = [json.loads(line) for line in next((p/"runs").glob("*/frames.jsonl")).read_text().splitlines()]
            self.assertEqual([r["source_time_s"] for r in rows], [0., .1, .2, .3])
            self.assertEqual([r["actual_pose_source"] for r in rows], ["tag", "tag", "none", "tag"])
            self.assertTrue(next((p/"runs").glob("*/input.avi")).exists())
            # Replay observations, not wall-clock-dependent timing, must repeat.
            run(replace(config, logging=replace(config.logging, save_input_video=False, save_annotated_video=False)))
            all_logs = list((p/"runs").glob("*/frames.jsonl"))
            again = [json.loads(line) for line in all_logs[1].read_text().splitlines()]
            for a, b in zip(rows, again):
                self.assertEqual(a["actual_pose_source"], b["actual_pose_source"])
                if a["raw_pose"]["T_CW"] is not None:
                    np.testing.assert_allclose(a["raw_pose"]["T_CW"], b["raw_pose"]["T_CW"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
