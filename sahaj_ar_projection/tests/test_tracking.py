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
from ar_projection.config import load_config
from ar_projection.diagnostics import RunLogger
from ar_projection.features import FeatureTracker, feature_mask, group_match_descriptors, match_descriptors
from ar_projection.geometry import camera_position, inverse, panel_points, project, projection_jacobian, retract, tag_points, transform, triangulate
from ar_projection.landmarks import MapBuilder, associate, guided_matches, load_map, map_metadata, save_map
from ar_projection.pipeline import Pipeline, run
from ar_projection.pose import PoseSolver, covariance
from ar_projection.rendering import PoseSmoother, annotate, panel_covariance, visible_panel
from ar_projection.tags import TagDetector
from ar_projection.types import CameraModel, FeatureObservation, FramePacket, LandmarkMap, TagObservation


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(Path(__file__).parent/"fixtures"/"tracking-config.json")
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

    def test_group_match_descriptors_avoids_flat_ratio_test_false_ambiguity(self):
        def with_distance(n_bits):
            byte_full, rem = divmod(n_bits, 8)
            d = np.zeros(32, np.uint8)
            d[:byte_full] = 0xFF
            if rem:
                d[byte_full] = (1 << rem)-1
            return d
        query = np.zeros((1, 32), np.uint8)
        same_landmark_view = with_distance(5)  # landmark 0, primary view: distance 5 from query
        same_landmark_other_view = with_distance(6)  # landmark 0, extra view: distance 6
        other_landmark = with_distance(20)  # a genuinely different landmark: distance 20
        train = np.stack([same_landmark_view, same_landmark_other_view, other_landmark])
        train_ids = np.array([0, 0, 1], np.int64)
        grouped = group_match_descriptors(query, train, train_ids, self.c.features)
        self.assertEqual(grouped, [(0, 0, 5.0, .25)])  # correct: best 5 vs the true second landmark's 20
        flat = match_descriptors(query, train, self.c.features)
        self.assertEqual(flat, [])  # the bug this fixes: 5 vs landmark 0's own second row (6) looks ambiguous

    def test_group_match_descriptors_rejects_non_reciprocal_landmark_match(self):
        def with_distance(n_bits):
            byte_full, rem = divmod(n_bits, 8)
            d = np.zeros(32, np.uint8)
            d[:byte_full] = 0xFF
            if rem:
                d[byte_full] = (1 << rem)-1
            return d
        train = np.stack([with_distance(0), with_distance(40)])  # landmark 0 close to query0; landmark 1 far from both
        train_ids = np.array([0, 1], np.int64)
        closer_query = np.zeros((1, 32), np.uint8)  # exact match to landmark 0
        farther_query = with_distance(10)  # also closest to landmark 0, but less close than closer_query
        cfg = replace(self.c.features, mutual_matching=True)
        both = group_match_descriptors(np.vstack([closer_query, farther_query]), train, train_ids, cfg)
        self.assertEqual([(q, id) for q, id, _, _ in both], [(0, 0)])  # query 1 loses landmark 0's reciprocity to query 0

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

    def test_multiview_descriptor_bank_roundtrips_and_old_format_still_loads(self):
        extra = np.zeros((60, 2, 32), np.uint8)
        extra[0, 0] = self.rng.integers(0, 256, 32, dtype=np.uint8)
        counts = np.zeros(60, np.int64)
        counts[0] = 1
        m = LandmarkMap(self.X.copy(), self.desc.copy(), np.arange(60, dtype=np.int64), np.full(60, 3, np.int64), np.zeros(60),
                        map_metadata(self.c, self.cam), extra, counts)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"map.npz"
            save_map(file, m)
            loaded = load_map(file, m.metadata, self.c)
            np.testing.assert_array_equal(loaded.extra_descriptors, extra)
            np.testing.assert_array_equal(loaded.extra_descriptor_counts, counts)
            old = self.ready_map()  # no extra fields: simulates a map saved before this feature existed
            file2 = Path(directory)/"old.npz"
            save_map(file2, old)
            loaded_old = load_map(file2, old.metadata, self.c)
            self.assertIsNone(loaded_old.extra_descriptors)
            self.assertIsNone(loaded_old.extra_descriptor_counts)

    def test_descriptor_bank_selects_most_angularly_separated_views(self):
        builder = MapBuilder(self.c, self.cam)
        point = self.X[0]

        class _KF:
            def __init__(self, T):
                self.T = T
        poses = [self.T]
        for dx in (.05, .2, .4):
            T = self.T.copy()
            T[0, 3] += dx
            poses.append(T)
        builder.keyframes = [_KF(T) for T in poses]
        obs = [(j, FeatureObservation(j, np.zeros(2), np.full(32, j+1, np.uint8))) for j in range(len(poses))]
        chosen, count = builder._descriptor_bank(point, obs, 2)
        self.assertEqual(count, 2)
        primary_dir = point-camera_position(poses[0])
        primary_dir = primary_dir/np.linalg.norm(primary_dir)
        angles = []
        for j, T in enumerate(poses[1:], start=1):
            direction = point-camera_position(T)
            direction = direction/np.linalg.norm(direction)
            angles.append((float(np.arccos(np.clip(primary_dir@direction, -1, 1))), j))
        expected_ids = sorted(angles, reverse=True)[:2]
        expected = np.array([np.full(32, j+1, np.uint8) for _, j in expected_ids])
        np.testing.assert_array_equal(sorted(chosen.tolist()), sorted(expected.tolist()))

    def test_associate_matches_via_extra_descriptor_bank_entry(self):
        extra = np.zeros((60, 1, 32), np.uint8)
        extra[5, 0] = self.rng.integers(0, 256, 32, dtype=np.uint8)
        counts = np.zeros(60, np.int64)
        counts[5] = 1
        m = LandmarkMap(self.X.copy(), self.desc.copy(), np.arange(60, dtype=np.int64), np.full(60, 3, np.int64), np.zeros(60),
                        map_metadata(self.c, self.cam), extra, counts)
        feature = FeatureObservation(999, self.generate(self.T, self.X[5:6])[0], extra[5, 0].copy())
        X, U = associate([feature], m, self.c.features)
        self.assertEqual(feature.landmark_id, 5)
        self.assertEqual(feature.association_source, "orb_map")
        self.assertEqual(len(X), 1)

    def test_guided_matches_prefers_descriptor_over_closer_pixel_and_respects_radius(self):
        m = LandmarkMap(self.X[:1].copy(), self.desc[:1].copy(), np.array([7], np.int64), np.array([3], np.int64),
                        np.zeros(1), map_metadata(self.c, self.cam))
        proj = self.generate(self.T, self.X[:1])[0]
        correct = FeatureObservation(0, proj+[5, 0], self.desc[0].copy())  # 5px away, correct descriptor
        wrong = FeatureObservation(1, proj+[1, 0], self.desc[5].copy())  # 1px away (closer!), wrong descriptor
        cfg = replace(self.c.features.guided_matching, search_radius_px=12.)
        matches = guided_matches([wrong, correct], m, self.T, self.cam, cfg, used=set())
        self.assertEqual(len(matches), 1)
        k, id, distance, ratio = matches[0]
        self.assertEqual(id, 7)
        self.assertIs([wrong, correct][k], correct)
        far = FeatureObservation(2, proj+[100, 0], self.desc[0].copy())  # correct descriptor but outside the radius
        self.assertEqual(guided_matches([far], m, self.T, self.cam, cfg, used=set()), [])

    def test_guided_matches_loose_limit_and_geometry_first(self):
        m = LandmarkMap(self.X[:1].copy(), self.desc[:1].copy(), np.array([7], np.int64), np.array([3], np.int64),
                        np.zeros(1), map_metadata(self.c, self.cam))
        proj = self.generate(self.T, self.X[:1])[0]
        near = FeatureObservation(0, proj+[1, 0], self.desc[5].copy())  # nearest in pixels, descriptor of another landmark
        base = replace(self.c.features.guided_matching, search_radius_px=12., max_hamming_distance=10)
        self.assertEqual(guided_matches([near], m, self.T, self.cam, base, used=set()), [])  # tight limit rejects it
        loose = replace(base, max_hamming_distance=256)
        self.assertEqual(guided_matches([near], m, self.T, self.cam, loose, used=set())[0][1], 7)  # loosened limit accepts
        far = FeatureObservation(1, proj+[8, 0], self.desc[0].copy())  # exact descriptor, but farther in pixels
        self.assertEqual(guided_matches([near, far], m, self.T, self.cam, loose, used=set())[0][0], 1)  # descriptor ranking
        geometry = replace(loose, geometry_first=True)
        self.assertEqual(guided_matches([near, far], m, self.T, self.cam, geometry, used=set())[0][0], 0)  # nearest pixel wins
        tie = FeatureObservation(2, proj+[-1, 0], self.desc[0].copy())  # same pixel distance as `near`, correct descriptor
        self.assertEqual(guided_matches([near, tie], m, self.T, self.cam, geometry, used=set())[0][0], 1)  # descriptor breaks the tie

    def test_geometry_margin_drops_ambiguous_picks_only(self):
        m = LandmarkMap(self.X[:1].copy(), self.desc[:1].copy(), np.array([7], np.int64), np.array([3], np.int64),
                        np.zeros(1), map_metadata(self.c, self.cam))
        proj = self.generate(self.T, self.X[:1])[0]
        near = FeatureObservation(0, proj+[2, 0], self.desc[0].copy())
        rival = FeatureObservation(1, proj+[0, 3], self.desc[0].copy())  # 3 px vs 2 px: ratio 1.5
        base = replace(self.c.features.guided_matching, search_radius_px=12., max_hamming_distance=256, geometry_first=True)
        self.assertEqual(len(guided_matches([near, rival], m, self.T, self.cam, base, used=set())), 1)  # margin off (default 1.0)
        self.assertEqual(guided_matches([near, rival], m, self.T, self.cam, replace(base, geometry_margin=2.0), used=set()), [])  # ambiguous
        self.assertEqual(len(guided_matches([near], m, self.T, self.cam, replace(base, geometry_margin=2.0), used=set())), 1)  # unique pick kept

    def test_guided_matches_one_to_one_per_feature_and_per_landmark(self):
        m = LandmarkMap(self.X[:2].copy(), self.desc[:2].copy(), np.array([7, 8], np.int64), np.array([3, 3], np.int64),
                        np.zeros(2), map_metadata(self.c, self.cam))
        proj = self.generate(self.T, self.X[:2])
        feature = FeatureObservation(0, proj[0], self.desc[0].copy())  # only candidate; near landmark 7's projection
        cfg = replace(self.c.features.guided_matching, search_radius_px=12.)
        matches = guided_matches([feature], m, self.T, self.cam, cfg, used=set())
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1], 7)  # landmark 8 found no available candidate once 7 claimed the only feature
        matches = guided_matches([feature], m, self.T, self.cam, cfg, used={7})  # 7 already claimed elsewhere this frame
        self.assertEqual(matches, [])

    def test_associate_uses_guided_matching_when_enabled_and_falls_back_globally(self):
        m = self.ready_map()
        guided_cfg = replace(self.c.features, guided_matching=replace(self.c.features.guided_matching, enabled=True, search_radius_px=12.))
        features = self.features(self.T)  # landmark_id=None for all, pixels/descriptors from self.X/self.desc via self.T
        X, U = associate(features, m, guided_cfg, prior_T_CW=self.T, camera=self.cam)
        self.assertEqual(len(X), 60)
        self.assertTrue(all(f.association_source == "guided" for f in features))

    def test_associate_guided_matching_default_off_ignores_prior(self):
        m = self.ready_map()
        features_with_prior = self.features(self.T)
        features_without_prior = self.features(self.T)
        X1, U1 = associate(features_with_prior, m, self.c.features, prior_T_CW=self.T, camera=self.cam)
        X2, U2 = associate(features_without_prior, m, self.c.features)
        self.assertTrue(all(f.association_source == "orb_map" for f in features_with_prior))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(U1, U2)

    def test_guided_revalidation_clears_wrong_inherited_id_and_rematches_it(self):
        m = self.ready_map()
        features = self.features(self.T)
        for f in features:
            f.association_source = "lk"
        wrong, right = features[0], features[1]
        wrong.landmark_id, right.landmark_id = int(m.ids[2]), int(m.ids[1])  # id 2 projects far from feature 0; id 1 is correct
        guided = replace(self.c.features.guided_matching, enabled=True, revalidate_existing=True)
        cfg = replace(self.c.features, guided_matching=guided)
        stats = {}
        associate(features, m, cfg, stats, prior_T_CW=self.T, camera=self.cam)
        self.assertEqual(stats["ids_cleared_by_guided_revalidation"], 1)
        self.assertEqual(right.landmark_id, int(m.ids[1]))  # correct inherited id kept
        self.assertEqual(wrong.landmark_id, int(m.ids[0]))  # cleared, then matched to its own landmark by the guided pass
        self.assertEqual(wrong.association_source, "guided")
        ids = [f.landmark_id for f in features if f.landmark_id is not None]
        self.assertEqual(len(ids), len(set(ids)))  # one-to-one
        off = replace(cfg, guided_matching=replace(guided, revalidate_existing=False))
        features = self.features(self.T)
        features[0].landmark_id, features[0].association_source = int(m.ids[2]), "lk"
        associate(features, m, off, {}, prior_T_CW=self.T, camera=self.cam)
        self.assertEqual(features[0].landmark_id, int(m.ids[2]))  # default off leaves the wrong id alone

    def test_strict_config_combinations_paths_and_types(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"config.json"
            original = json.loads((Path(__file__).parent/"fixtures"/"tracking-config.json").read_text())
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
            first_stats = tracker.frame_stats
            self.assertTrue(first_stats["orb"]["ran"])
            self.assertEqual(first_stats["orb"]["keypoints"], len(a))
            self.assertEqual(first_stats["orb"]["new_tracks"], len(a))
            b = tracker.update(gray, [self.tag], 1)
            stats = tracker.frame_stats
            self.assertLessEqual(len(b), 120)
            self.assertGreater(len(set(f.track_id for f in a)&set(f.track_id for f in b)), 40)
            self.assertEqual(len(b), len({f.track_id for f in b}))
            self.assertTrue(all(mask[int(f.pixel[1]), int(f.pixel[0])] for f in b))
            self.assertEqual(stats["tracks_after_update"], len(b))
            self.assertEqual(stats["lk"]["retained"]+stats["lk"]["dropped"], stats["lk"]["input"])
            self.assertEqual(stats["orb"]["ran"], not enabled)
            if enabled:
                self.assertEqual(stats["lk"]["retained"], len(b))
        pairs = match_descriptors(self.desc, self.desc, cfg)
        self.assertEqual(len(pairs), 60)
        self.assertTrue(all(a == b for a, b, _, _ in pairs))

    def test_feature_tracking_diagnostics_reach_frame_and_overlay(self):
        config = replace(self.c, mapping=replace(self.c.mapping, mode="off"),
                         display=replace(self.c.display, enabled=False))
        pipe = Pipeline(config, self.cam)
        gray = self.rng.integers(0, 256, (480, 640), dtype=np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        packet = FramePacket(0, 0., bgr, bgr.copy(), gray, self.cam)
        result = pipe.process(packet, tags=[])
        stats = result.diagnostics["feature_tracking"]
        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["orb"]["ran"])
        self.assertEqual(stats["tracks_after_update"], result.diagnostics["tracked_features"])
        self.assertEqual(annotate(packet, result, [], pipe.last_features, config.display).shape, bgr.shape)

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
