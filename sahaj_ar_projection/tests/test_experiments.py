"""Regression checks with explicit synthetic geometry; no cameras or ML weights."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from ar_projection.camera import Capture, Preprocessor
from ar_projection.config import BackgroundPrior as PriorConfig, IdRevalidation, load_config
from ar_projection.diagnostics import RunLogger, code_sha256
from ar_projection.experiments import compare, provenance, summarize, tracking_acceptance, window_metrics, window_sources
from ar_projection.features import FeatureTracker
from ar_projection.geometry import depths, panel_points, tag_points, transform
from ar_projection.landmarks import associate, map_metadata, revalidate_ids, save_map
from ar_projection.occlusion import DepthPrototype, anchor_sample, evaluate, mask_metrics, PlaneCorrector, PlaneParams, panel_layer, plane_depth, BackgroundPrior, RelativeLineOcclusion, require_tracking_acceptance, select_resolution, tag_plane_scale
from ar_projection.pipeline import Pipeline, run
from ar_projection.pose import PoseSolver
from ar_projection.rendering import side_by_side, visible_panel
from ar_projection.tags import TagDetector
from ar_projection.types import CameraModel, FeatureObservation, FramePacket, LandmarkMap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from score_associations import score  # noqa: E402
from oracle_diagnostic import classify_pair, frame_backward_matches, grid_quota_survivors, hamming_distances, nearest_keypoint  # noqa: E402
from descriptor_validity_check import (distractor_geometry_report, self_consistency_report,  # noqa: E402
                                       tight_radius_report, view_change_bin, view_change_report)


class ExperimentTests(unittest.TestCase):
    def test_windows_camera_default_is_set_before_cv2_and_preserves_override(self):
        code = ("import sys; sys.platform = 'win32'; import ar_projection; import os; "
                "assert 'cv2' not in sys.modules; "
                "print(os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'])")
        env = os.environ.copy()
        env.pop("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", None)
        root = Path(__file__).resolve().parents[1]
        for override, expected in ((None, "0"), ("1", "1")):
            if override is not None:
                env["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = override
            result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=10, check=True)
            self.assertEqual(result.stdout.strip(), expected)

    def setUp(self):
        base = load_config(Path(__file__).parent/"fixtures"/"tracking-config.json")
        self.cfg = replace(base, tags=replace(base.tags, size_m=.10),
                           mapping=replace(base.mapping, min_ready_landmarks=40, max_keyframes=8),
                           features=replace(base.features, exclude_tag_regions=True),
                           pose=replace(base.pose, mode="joint"), filter=replace(base.filter, enabled=False),
                           display=replace(base.display, enabled=False))
        self.rng = np.random.default_rng(9)
        self.cam = CameraModel(np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1]]), np.zeros(5), (640, 480), {"synthetic": True})
        self.T = transform([2.8, .1, .05], [0, 0, .9])
        self.X = self.rng.uniform([-.25, -.18, -.08], [.25, .18, .08], (60, 3))
        self.desc = self.rng.integers(0, 256, (60, 32), dtype=np.uint8)
        self.tag = TagDetector(self.cfg.tags).observation(0, self.pixels(tag_points(.1)), self.cam)

    def pixels(self, X, T=None):
        T = self.T if T is None else T
        xyz = X@T[:3, :3].T+T[:3, 3]
        return xyz[:, :2]/xyz[:, 2:]*[800, 800]+[320, 240]

    def features(self, T=None):
        return [FeatureObservation(i, p, d.copy()) for i, (p, d) in enumerate(zip(self.pixels(self.X, T), self.desc))]

    def packet(self, index, timestamp, gray=None):
        gray = np.zeros((480, 640), np.uint8) if gray is None else gray
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return FramePacket(index, timestamp, bgr, bgr.copy(), gray, self.cam)

    def config_with_map(self, directory):
        cfg = replace(self.cfg, mapping=replace(self.cfg.mapping, mode="localize", map_path=str(Path(directory)/"map.npz")))
        m = LandmarkMap(self.X, self.desc, np.arange(60, dtype=np.int64), np.full(60, 3, np.int64), np.zeros(60), map_metadata(cfg, self.cam))
        save_map(cfg.mapping.map_path, m)
        return cfg

    def test_translated_lk_retains_identity_and_measures_displacement(self):
        cfg = replace(self.cfg.features, max_features=160, redetect_every_n_frames=15, redetect_below_tracks=0)
        tracker = FeatureTracker(cfg)
        gray = self.rng.integers(0, 256, (480, 640), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), .8)
        old = {f.track_id: f.pixel.copy() for f in tracker.update(gray, [], 0)}
        moved = cv2.warpAffine(gray, np.array([[1, 0, 3], [0, 1, 2]], np.float32), (640, 480), borderMode=cv2.BORDER_REFLECT101)
        new = tracker.update(moved, [], 1)
        self.assertGreater(len(new), .8*len(old))
        np.testing.assert_allclose(np.median([f.pixel-old[f.track_id] for f in new], axis=0), [3, 2], atol=.2)
        self.assertTrue(all(f.age_frames == 2 for f in new))

    def test_old_configs_gain_only_explicit_optional_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            data = asdict(self.cfg)
            del data["features"]["redetection_match_radius_px"]
            del data["camera"]["timestamps_path"]
            path = Path(directory)/"config.json"
            path.write_text(json.dumps(data))
            cfg = load_config(path)
            self.assertEqual(cfg.features.redetection_match_radius_px, 5.)
            self.assertIsNone(cfg.camera.timestamps_path)
            data["camera"]["timestamps_path"] = "times.jsonl"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "requires a video"):
                load_config(path)

    def test_redetection_radius_is_independent_of_flow_consistency(self):
        cfg = replace(self.cfg.features, redetect_every_n_frames=1, redetect_below_tracks=0,
                      redetection_match_radius_px=5., lk=replace(self.cfg.features.lk, forward_backward_max_error_px=.1))
        tracker = FeatureTracker(cfg)
        old = [FeatureObservation(i, np.array([100.+i*30, 100.]), self.desc[i], i, age_frames=2) for i in range(3)]
        detected = [FeatureObservation(-1, f.pixel+[3, 0], f.descriptor.copy()) for f in old]
        tracker.tracks = old
        with patch.object(tracker, "flow", return_value=old), patch.object(tracker, "detect", return_value=detected):
            new = tracker.update(np.zeros((480, 640), np.uint8), [], 1)
        self.assertEqual([f.landmark_id for f in new], [0, 1, 2])
        self.assertEqual([f.track_id for f in new], [0, 1, 2])
        self.assertTrue(all(f.association_source == "orb_redetection" for f in new))

    def test_gap_expires_pose_prior_without_destroying_feature_tracks(self):
        cfg = replace(self.cfg, mapping=replace(self.cfg.mapping, mode="off"))
        pipe = Pipeline(cfg, self.cam)
        gray = self.rng.integers(0, 256, (480, 640), dtype=np.uint8)
        pipe.process(self.packet(0, 0., gray), [self.tag])
        ids = {f.track_id for f in pipe.last_features}
        result = pipe.process(self.packet(1, .6, gray), [self.tag])
        self.assertTrue(result.diagnostics["pose_prior_expired"])
        self.assertFalse(result.diagnostics["feature_tracker_reset"])
        self.assertGreater(len(ids & {f.track_id for f in pipe.last_features}), len(ids)*.7)
        invalid = pipe.process(self.packet(2, .6, gray), [self.tag])
        self.assertTrue(invalid.diagnostics["feature_tracker_reset"])
        self.assertFalse(invalid.raw_pose.valid)

    def test_map_associations_separate_lk_redetection_and_descriptor_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            features = self.features()
            features[0].landmark_id, features[0].association_source = 0, "lk"
            features[1].landmark_id, features[1].association_source = 1, "orb_redetection"
            stats = {}
            X, U = associate(features, pipe.map, cfg.features, stats)
            self.assertEqual(len(X), 60)
            self.assertEqual(stats["inherited_lk_associations"], 1)
            self.assertEqual(stats["redetected_orb_associations"], 1)
            self.assertEqual(stats["fresh_descriptor_associations"], 58)
            self.assertEqual(U.shape, (60, 2))

    def test_pose_reports_count_and_spatial_rejection(self):
        solver = PoseSolver(self.cfg)
        pose = solver.estimate([self.tag], self.X[:10], self.pixels(self.X[:10]), self.cam)
        self.assertEqual(pose.source, "tag")
        self.assertEqual(pose.diagnostics["landmark_gate_reason"], "insufficient mapped observations")
        small = self.X*.03
        small += [.04, .035, 0]
        pose = solver.estimate([self.tag], small, self.pixels(small), self.cam)
        self.assertEqual(pose.source, "tag")
        self.assertEqual(pose.diagnostics["landmark_gate_reason"], "insufficient image coverage")

    def test_withholding_masks_tags_but_never_passes_them_to_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            with patch.object(pipe.features, "update", return_value=self.features()) as update:
                result = pipe.process(self.packet(0, 0.), [self.tag], withhold_tag=True)
            self.assertEqual(update.call_args.args[1], [self.tag])
            self.assertTrue(cfg.features.exclude_tag_regions)
            self.assertEqual(result.raw_pose.source, "landmark")
            self.assertTrue(result.diagnostics["tag_reference_valid"])
            np.testing.assert_allclose(result.raw_panel, result.diagnostics["reference_panel_pixels"], atol=1e-5)

    def test_sustained_synthetic_tag_absence_and_loss_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(self.config_with_map(directory), self.cam)
            for i in range(31):
                result = pipe.process(self.packet(i, i*10.), [], self.features())
                self.assertTrue(result.raw_pose.valid, result.raw_pose.reason)
                self.assertEqual(result.raw_pose.source, "landmark")
            lost = pipe.process(self.packet(31, 301.), [], [])
            self.assertIsNone(lost.render_panel)
            self.assertEqual(lost.state, "LOST")
            recovered = pipe.process(self.packet(32, 301.03), [], self.features())
            self.assertEqual(recovered.state, "RELOCALIZING")

    def test_recorded_timestamps_preserve_real_gaps_and_reject_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            np.savez(p/"camera.npz", camera_matrix=self.cam.K, dist_coeffs=np.zeros(5), image_size=[640, 480])
            writer = cv2.VideoWriter(str(p/"input.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 30, (640, 480))
            self.assertTrue(writer.isOpened())
            image = self.rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(3):
                writer.write(image)
            writer.release()
            rows = [{"frame_index": i, "source_time_s": t} for i, t in enumerate([20., 20.03, 20.8])]
            (p/"times.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            camera = replace(self.cfg.camera, source_type="video", backend="msmf", video_path=str(p/"input.avi"),
                             calibration_path=str(p/"camera.npz"), timestamps_path=str(p/"times.jsonl"))
            capture = Capture(camera)
            try:
                packets = [capture.read() for _ in range(3)]
                self.assertEqual([p.timestamp for p in packets], [20., 20.03, 20.8])
                np.testing.assert_array_equal(packets[0].original, image)
                self.assertIsNone(capture.read())
            finally:
                capture.close()
            (p/"times.jsonl").write_text(json.dumps(rows[0]))
            capture = Capture(camera)
            try:
                capture.read()
                with self.assertRaisesRegex(ValueError, "more frames"):
                    capture.read()
            finally:
                capture.close()

    def test_panel_depth_is_metric_and_partial_panel_is_clipped(self):
        panel = np.array([[-.5, -.15, 0], [.15, -.15, 0], [.15, .15, 0], [-.5, .15, 0]])
        T = transform([0, 0, 0], [0, 0, 1.])
        self.assertIsNotNone(visible_panel(T, panel, self.cam))
        color, mask, z = panel_layer(T, panel, self.cam)
        self.assertTrue(np.any(mask[:, 0]))
        np.testing.assert_allclose(z[mask], 1.)
        scene = np.full((480, 640), 1.5)
        scene[:, :320] = .5
        _, visible, _ = panel_layer(T, panel, self.cam, scene)
        self.assertFalse(np.any(visible[:, :320]))
        self.assertTrue(np.any(visible[:, 320:]))
        self.assertEqual(color.dtype, np.uint8)

    def test_plane_depth_matches_panel_layer_on_the_same_plane_but_is_not_clipped(self):
        T = transform([0, 0, 0], [0, 0, 1.])
        z, valid = plane_depth(T, np.zeros(3), np.array([0., 0, 1]), self.cam)
        self.assertTrue(np.all(valid))  # whole image, not clipped to any polygon
        np.testing.assert_allclose(z[valid], 1.)
        panel = np.array([[-.15, -.15, 0], [.15, -.15, 0], [.15, .15, 0], [-.15, .15, 0]])
        T2 = transform([0, .6, 0], [0, 0, 1.])
        _, panel_mask, panel_z = panel_layer(T2, panel, self.cam)
        z2, valid2 = plane_depth(T2, np.zeros(3), np.array([0., 0, 1]), self.cam)
        np.testing.assert_allclose(z2[panel_mask], panel_z[panel_mask])  # same plane, same depth where panel_layer is also valid
        self.assertGreater(np.count_nonzero(valid2), np.count_nonzero(panel_mask))  # but not clipped to the panel's own quad

    def _tilted_table(self):
        R = cv2.Rodrigues(np.array([np.pi-.6, 0, 0]))[0]  # camera at (0,-.8,1) looks down at the Z=0 table
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, -R@np.array([0., -.8, 1.0])
        panel = np.array([[-.05, -.05, 0], [.05, -.05, 0], [.05, .05, 0], [-.05, .05, 0]])
        oracle, valid = plane_depth(T, np.zeros(3), np.array([0., 0, 1]), self.cam)
        return T, panel, oracle, valid

    def test_plane_correction_recovers_a_known_bias_and_ignores_objects(self):
        T, panel, oracle, valid = self._tilted_table()
        rows = (np.indices(oracle.shape)[0]/(oracle.shape[0]-1)*2-1)
        params = dict(z_min_m=.1, z_max_m=5., min_pixels=1000, max_mad_m=.02, b_range=(.5, 2.))
        table = valid.copy()
        table[300:380, 100:200] = False
        # full terms, table only: scale + offset + row tilt recovered (raw is a plane here, so only the sum is identifiable)
        raw = (oracle-.05-.08*rows)/1.2
        corrected, stats = PlaneCorrector(PlaneParams(terms="full", **params))(raw, T, panel, self.cam)
        self.assertTrue(stats["plane_gate"])
        self.assertLess(float(np.median(np.abs(corrected-oracle)[table])), .01)
        # affine terms with an object in front of the table: the fit ignores it and the object stays nearer
        raw = (oracle-.05)/1.2
        raw[300:380, 100:200] = .3
        corrected, stats = PlaneCorrector(PlaneParams(terms="affine", **params))(raw, T, panel, self.cam)
        self.assertTrue(stats["plane_gate"])
        self.assertLess(float(np.median(np.abs(corrected-oracle)[table])), .01)
        self.assertLess(float(corrected[340, 150]), float(oracle[340, 150])-.1)

    def test_plane_correction_fixed_slope_needs_a_prior_and_removes_the_row_bias(self):
        T, panel, oracle, valid = self._tilted_table()
        rows = (np.indices(oracle.shape)[0]/(oracle.shape[0]-1)*2-1)
        raw = (oracle-.05-.08*rows)/1.2
        raw[300:380, 100:200] = .3
        corrector = PlaneCorrector(PlaneParams(terms="fixedb", z_min_m=.1, z_max_m=5., min_pixels=1000, max_mad_m=.02))
        self.assertIsNone(corrector(raw, T, panel, self.cam)[0])
        corrected, stats = corrector(raw, T, panel, self.cam, b_prior=1.2)
        self.assertTrue(stats["plane_gate"])
        table = valid.copy()
        table[300:380, 100:200] = False
        self.assertLess(float(np.median(np.abs(corrected-oracle)[table])), .01)
        self.assertLess(float(corrected[340, 150]), float(oracle[340, 150])-.1)

    def test_plane_correction_gate_failure_returns_none_and_clears_the_hold(self):
        T, panel, oracle, _ = self._tilted_table()
        corrector = PlaneCorrector(PlaneParams(z_min_m=.1, z_max_m=5., min_pixels=1000))
        self.assertIsNotNone(corrector(oracle.copy(), T, panel, self.cam)[0])
        self.assertTrue(len(corrector.held))
        corrected, stats = PlaneCorrector(PlaneParams(z_min_m=9., z_max_m=10.))(oracle.copy(), T, panel, self.cam)
        self.assertIsNone(corrected)
        self.assertFalse(stats["plane_gate"])
        bad = PlaneCorrector(PlaneParams(z_min_m=.1, z_max_m=5., min_pixels=1000, max_mad_m=1e-9))
        bad.held.append(np.zeros(4))
        raw = np.random.default_rng(0).uniform(1, 2, oracle.shape)
        self.assertIsNone(bad(raw, T, panel, self.cam)[0])
        self.assertEqual(len(bad.held), 0)  # never reuse a stale fit

    def test_tilted_panel_depth_varies_along_image_rays(self):
        panel = np.array([[-.15, -.15, 0], [.15, -.15, 0], [.15, .15, 0], [-.15, .15, 0]])
        T = transform([0, .6, 0], [0, 0, 1.])
        _, mask, z = panel_layer(T, panel, self.cam)
        self.assertGreater(float(np.ptp(z[mask])), .1)
        y, x = np.where(mask)
        rays = np.stack(((x-320)/800, (y-240)/800, np.ones_like(x)), axis=-1)
        xyz = rays*z[mask, None]
        normal = T[:3, 2]
        np.testing.assert_allclose((xyz-T[:3, 3])@normal, 0, atol=1e-10)

    def test_tag_plane_scale_recovers_a_known_scale(self):
        T = transform([0, 0, 0], [0, 0, 1.])
        _, valid, z = panel_layer(T, tag_points(.17), self.cam)
        raw = np.zeros_like(z)
        raw[valid] = z[valid]/3.0  # model under-reports depth by a factor of 3
        self.assertAlmostEqual(tag_plane_scale(T, raw, .17, self.cam), 3.0, places=6)

    def test_tag_plane_scale_is_none_without_enough_valid_pixels(self):
        T = transform([0, 0, 0], [0, 0, 1.])
        self.assertIsNone(tag_plane_scale(T, np.zeros((480, 640)), .17, self.cam))

    def test_invalid_depth_is_visible_by_default_and_occluded_when_requested(self):
        panel = np.array([[-.15, -.15, 0], [.15, -.15, 0], [.15, .15, 0], [-.15, .15, 0]])
        T = transform([0, 0, 0], [0, 0, 1.])
        scene = np.full((480, 640), np.nan)  # depth unknown everywhere
        _, visible_default, _ = panel_layer(T, panel, self.cam, scene)
        _, visible_occludes, z = panel_layer(T, panel, self.cam, scene, invalid_occludes=True)
        self.assertTrue(np.array_equal(visible_default, z > 0))  # unknown depth: current policy keeps the panel visible
        self.assertFalse(np.any(visible_occludes))  # unknown depth treated as occluded when requested

    def test_mask_metrics_and_missing_acceptance_are_not_success(self):
        mask = np.zeros((80, 100), bool)
        mask[10:60, 20:70] = True
        self.assertEqual(mask_metrics(mask, mask), {"iou": 1., "boundary_f1_at_2px": 1.})
        self.assertEqual(mask_metrics(mask, ~mask)["iou"], 0.)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/"acceptance.json"
            report.write_text(json.dumps({"kind": "tracking_acceptance", "passed": False}))
            with self.assertRaisesRegex(ValueError, "gated"):
                require_tracking_acceptance(report, self.cfg)

    def test_comparison_replays_same_frames_and_cannot_pass_without_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            np.savez(p/"camera.npz", camera_matrix=self.cam.K, dist_coeffs=np.zeros(5), image_size=[640, 480])
            camera = replace(self.cfg.camera, calibration_path=str(p/"camera.npz"))
            cfg = replace(self.cfg, camera=camera, mapping=replace(self.cfg.mapping, mode="localize", map_path=str(p/"map.npz")),
                          logging=replace(self.cfg.logging, output_dir=str(p/"results"), enabled=True))
            processed = Preprocessor(camera, (640, 480)).camera
            m = LandmarkMap(self.X, self.desc, np.arange(60, dtype=np.int64), np.full(60, 3, np.int64), np.zeros(60), map_metadata(cfg, processed))
            save_map(cfg.mapping.map_path, m)
            recording = p/"recording"
            recording.mkdir()
            (recording/"config.json").write_text(json.dumps(asdict(cfg)))
            writer = cv2.VideoWriter(str(recording/"input.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 30, (640, 480))
            marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 400)
            H = cv2.getPerspectiveTransform(np.array([[0, 0], [399, 0], [399, 399], [0, 399]], np.float32), self.tag.pixels.astype(np.float32))
            image = cv2.cvtColor(cv2.warpPerspective(marker, H, (640, 480), borderValue=255), cv2.COLOR_GRAY2BGR)
            timestamps = [10., 10.03, 10.8, 10.83]
            for _ in timestamps:
                writer.write(image)
            writer.release()
            (recording/"frames.jsonl").write_text("\n".join(json.dumps({"frame_index": i, "source_time_s": t}) for i, t in enumerate(timestamps)))
            report = compare(recording, None, p/"comparison.json")
            self.assertIsNone(report["selected_landmark_variant"])
            self.assertEqual(len(report["variants"]), 5)
            for summary in report["variants"].values():
                self.assertEqual(summary["frames"], 4)
                self.assertIsNone(summary["displayed_fps"])
                self.assertEqual(summary["pose_prior_expirations"], 1)
            paths = [report["variants"][k]["run"] for k in ("lk_periodic", "joint", "joint")]
            assessment = tracking_acceptance(*paths)
            self.assertFalse(assessment["passed"])
            self.assertFalse(assessment["checks"]["live_20fps"])
            self.assertFalse(assessment["checks"]["physical_tag_hidden_120s"])

    def test_summarize_separates_startup_from_loss_and_ignores_withheld_tag_as_evidence(self):
        def row(t, supported, withheld=False, tag=True, gate="insufficient RANSAC inliers"):
            return {"source_time_s": t, "actual_pose_source": "landmark" if supported else "none",
                    "raw_pose": {"valid": supported, "tag_inliers": 0, "landmark_inliers": 12 if supported else 0},
                    "diagnostics": {"tag_observations_withheld": withheld, "tags": [{"id": 0, "valid": tag, "ambiguous": False}] if tag else [],
                                    "pose_gates": {"ransac_attempted": gate is None, "landmark_gate_reason": gate}}}
        rows = [row(0., False), row(1., False), row(2., True, withheld=True, tag=False),   # startup: 2 s to acquire
                row(3., False, withheld=True), row(4., False, withheld=True),                # loss; a visible but withheld tag is no evidence
                row(5., False, withheld=True, gate=None),                                    # landmark evidence returns here
                row(6., True, withheld=True)]
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/"config.json").write_text(json.dumps(asdict(self.cfg)))
            (p/"frames.jsonl").write_text(chr(10).join(json.dumps(r) for r in rows))
            report = summarize(p)
        self.assertTrue(report["acquired"])
        self.assertEqual(report["startup_acquisition_s"], 2.0)
        self.assertEqual(report["longest_loss_s"], 3.0)                        # first lost frame t=3 to t=6, startup excluded
        self.assertEqual(report["recovery_after_available_evidence_max_s"], 1.0)  # from t=5, not from the withheld tag at t=3

    def test_acceptance_provenance_records_tracking_config_and_code_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/"config.json").write_text(json.dumps(asdict(self.cfg)))
            (p/"metadata.json").write_text(json.dumps({"code_sha256": code_sha256()}))
            record = provenance(p)
            self.assertEqual(record["code_sha256"], record["current_code_sha256"])
            self.assertEqual(record["tracking_config"]["pose"], asdict(self.cfg.pose))
            self.assertNotIn("camera", record["tracking_config"])

    def test_withhold_tag_windows_restricts_withholding_to_configured_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            np.savez(p/"camera.npz", camera_matrix=self.cam.K, dist_coeffs=np.zeros(5), image_size=[640, 480])
            camera = replace(self.cfg.camera, calibration_path=str(p/"camera.npz"))
            cfg = replace(self.cfg, camera=camera, mapping=replace(self.cfg.mapping, mode="localize", map_path=str(p/"map.npz")),
                          pose=replace(self.cfg.pose, mode="joint"),
                          logging=replace(self.cfg.logging, output_dir=str(p/"results"), enabled=True))
            processed = Preprocessor(camera, (640, 480)).camera
            m = LandmarkMap(self.X, self.desc, np.arange(60, dtype=np.int64), np.full(60, 3, np.int64), np.zeros(60), map_metadata(cfg, processed))
            save_map(cfg.mapping.map_path, m)
            recording = p/"recording"
            recording.mkdir()
            writer = cv2.VideoWriter(str(recording/"input.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 30, (640, 480))
            marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 400)
            H = cv2.getPerspectiveTransform(np.array([[0, 0], [399, 0], [399, 399], [0, 399]], np.float32), self.tag.pixels.astype(np.float32))
            image = cv2.cvtColor(cv2.warpPerspective(marker, H, (640, 480), borderValue=255), cv2.COLOR_GRAY2BGR)
            timestamps = [10., 10.03, 10.8, 10.83]  # only 10.8/10.83 fall inside the configured window
            for _ in timestamps:
                writer.write(image)
            writer.release()
            (recording/"frames.jsonl").write_text("\n".join(json.dumps({"frame_index": i, "source_time_s": t}) for i, t in enumerate(timestamps)))
            replay_cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(recording/"input.avi"),
                                                     timestamps_path=str(recording/"frames.jsonl")))
            path = run(replay_cfg, withhold_tag_windows=[(10.5, 11.0)])
            rows = [json.loads(line) for line in (Path(path)/"frames.jsonl").read_text().splitlines()]
            withheld = [r["diagnostics"]["tag_observations_withheld"] for r in rows]
            self.assertEqual(withheld, [False, False, True, True])
            sources = [r["actual_pose_source"] for r in rows]
            self.assertTrue(all(x in ("tag", "joint") for x in sources[:2]))
            self.assertTrue(all(x not in ("tag", "joint") for x in sources[2:]))
            self.assertEqual(window_sources(path, [(10.5, 11.0)])["inside"].keys() & {"tag", "joint"}, set())

    def test_window_metrics_computes_valid_fraction_lost_time_run_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            panel = [[0., 0], [1., 0], [1., 1.], [0., 1.]]

            def row(t, valid, withheld):
                return {"source_time_s": t, "actual_pose_source": "landmark" if valid else "none",
                       "raw_pose": {"valid": valid, "landmark_inliers": 10 if valid else 0, "tag_inliers": 0},
                       "raw_panel_pixels": panel if valid else None,
                       "diagnostics": {"tag_observations_withheld": withheld,
                                      "reference_panel_pixels": panel if valid else None}}
            rows = [row(0., True, True), row(1., True, True), row(2., False, True), row(3., True, True),  # window [0,3]
                   row(4., True, False)]  # recovery, outside the window
            (p/"frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            metrics = window_metrics(p, [(0., 3.)])["0.0:3.0"]
            self.assertEqual(metrics["count"], 4)
            self.assertAlmostEqual(metrics["valid_fraction"], .75)
            self.assertEqual(metrics["time_to_first_lost_s"], 2.0)
            self.assertEqual(metrics["longest_supported_run_s"], 1.0)
            self.assertEqual(metrics["reference_samples"], 12)  # 3 supported frames x 4 panel corners, matching summarize()'s own convention
            self.assertAlmostEqual(metrics["reference_median_px"], 0.0)
            self.assertEqual(metrics["recovery_after_window_s"], 1.0)

    def test_depth_scale_failure_is_exposed_and_exports_failed_frames(self):
        class FixedDepth:
            def predict(self, image):
                return np.ones(image.shape[:2]), {"inference_ms": 1.}
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            result = pipe.process(self.packet(0, 0.), [self.tag], [])
            output = Path(directory)/"predictions"
            depth = DepthPrototype(FixedDepth(), output)
            layer = depth.render(self.packet(0, 0.), result, pipe.panel, pipe.map, [])
            self.assertIsNotNone(layer)
            self.assertIn("unavailable", result.diagnostics["occlusion"]["status"])
            self.assertTrue((output/"000000.json").exists())

    def test_depth_affine_fit_rejects_a_degenerate_non_decreasing_fit(self):
        class ConstantDepth:
            def predict(self, image):
                return np.ones(image.shape[:2]), {"inference_ms": 1.}  # raw doesn't vary with distance: unfittable
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(self.config_with_map(directory), self.cam)
            packet = self.packet(0, 0.)
            result = pipe.process(packet, [], self.features())
            prototype = DepthPrototype(ConstantDepth())
            prototype.render(packet, result, pipe.panel, pipe.map, pipe.last_features)
            stats = result.diagnostics["occlusion"]
            self.assertIn("unavailable", stats["status"])
            self.assertGreaterEqual(stats["scale_anchor_count"], 12)
            self.assertFalse(prototype.fits)

    def test_anchor_sample_patch_median_and_edge_rejection(self):
        raw = np.full((20, 20), 5.0)
        raw[8:12, 8:12] = 5.3  # a flat-ish patch, well within edge_range
        self.assertAlmostEqual(anchor_sample(raw, 10, 10), 5.3, places=6)
        edge = np.full((20, 20), 5.0)
        edge[:, 10:] = 50.0  # a sharp depth edge running right through (10, 10)
        self.assertIsNone(anchor_sample(edge, 10, 10))

    def test_depth_slope_sign_controls_which_fit_direction_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(self.config_with_map(directory), self.cam)
            packet = self.packet(0, 0.)
            result = pipe.process(packet, [], self.features())
            # Bake a POSITIVE slope this time (raw rises with true Z, the opposite of every
            # checkpoint seen so far) to check the sign gate in isolation.
            a0, b0 = 5.0, 3.0
            z_true = depths(result.raw_pose.T_CW, self.X)
            raw_at = {tuple(np.rint(p).astype(int)): a0+b0*z for p, z in zip(self.pixels(self.X), z_true)}

            class FixedDepth:
                def predict(self, image):
                    raw = np.full(image.shape[:2], 1.0)
                    for (x, y), v in raw_at.items():
                        if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0]:
                            raw[max(0, y-2):y+3, max(0, x-2):x+3] = v  # a flat patch, not a 1px spike: survives anchor_sample's edge rejection
                    return raw, {"inference_ms": 1.}

            default = DepthPrototype(FixedDepth())  # slope_sign=-1
            default.render(packet, result, pipe.panel, pipe.map, pipe.last_features)
            self.assertIn("unavailable", result.diagnostics["occlusion"]["status"])
            self.assertFalse(default.fits)

            positive = DepthPrototype(FixedDepth(), slope_sign=1)
            positive.render(packet, result, pipe.panel, pipe.map, pipe.last_features)
            stats = result.diagnostics["occlusion"]
            self.assertEqual(stats["status"], "metric depth active (prototype)")
            self.assertAlmostEqual(stats["depth_fit_a"], a0, places=3)
            self.assertAlmostEqual(stats["depth_fit_b"], b0, places=3)

    def test_depth_affine_fit_uses_current_geometric_anchors_and_stops_when_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = Pipeline(self.config_with_map(directory), self.cam)
            packet = self.packet(0, 0.)
            result = pipe.process(packet, [], self.features())
            # Bake a known affine relationship (raw = a0 + b0*true_Z) at each landmark's own
            # pixel, so the fit the production code recovers can be checked exactly.
            a0, b0 = 23.0, -5.0
            z_true = depths(result.raw_pose.T_CW, self.X)
            raw_at = {tuple(np.rint(p).astype(int)): a0+b0*z for p, z in zip(self.pixels(self.X), z_true)}

            class FixedDepth:
                def predict(self, image):
                    raw = np.full(image.shape[:2], 1.0)
                    for (x, y), v in raw_at.items():
                        if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0]:
                            raw[max(0, y-2):y+3, max(0, x-2):x+3] = v  # a flat patch, not a 1px spike: survives anchor_sample's edge rejection
                    return raw, {"inference_ms": 1.}

            prototype = DepthPrototype(FixedDepth())
            prototype.render(packet, result, pipe.panel, pipe.map, pipe.last_features)
            stats = result.diagnostics["occlusion"]
            self.assertEqual(stats["status"], "metric depth active (prototype)")
            self.assertGreaterEqual(stats["scale_anchor_count"], 12)
            self.assertAlmostEqual(stats["depth_fit_a"], a0, places=3)
            self.assertAlmostEqual(stats["depth_fit_b"], b0, places=3)
            lost = pipe.process(self.packet(1, .03), [], [])
            self.assertIsNone(prototype.render(self.packet(1, .03), lost, pipe.panel, pipe.map, []))
            self.assertFalse(prototype.fits)

    def test_prior_grace_keeps_seed_through_short_gaps_only_and_never_renders_it(self):
        for grace in (0.0, .05, .1):
            with tempfile.TemporaryDirectory() as directory:
                cfg = self.config_with_map(directory)
                pipe = Pipeline(replace(cfg, pose=replace(cfg.pose, prior_grace_s=grace)), self.cam)
                pipe.process(self.packet(0, 0.), [], self.features())
                short = pipe.process(self.packet(1, .03), [], [])
                long = pipe.process(self.packet(2, .06), [], [])
                self.assertEqual([short.diagnostics["prior_age_s"] is not None, long.diagnostics["prior_age_s"] is not None],
                                 [True, grace >= .03])  # age is read before this frame's keep/drop decision
                self.assertFalse(short.raw_pose.valid or long.raw_pose.valid)
                self.assertIsNone(short.render_pose)
                self.assertEqual(pipe.previous is not None, grace >= .06)

    def test_prior_grace_values_differ_over_a_long_failure(self):
        used = {}
        for grace in (.1, .5):
            with tempfile.TemporaryDirectory() as directory:
                cfg = self.config_with_map(directory)
                pipe = Pipeline(replace(cfg, pose=replace(cfg.pose, prior_grace_s=grace)), self.cam)
                pipe.process(self.packet(0, 0.), [], self.features())
                results = [pipe.process(self.packet(i, i*.05), [], []) for i in range(1, 7)]  # 0.3 s of failures, 0.05 s apart
                used[grace] = sum(r.diagnostics["prior_used"] for r in results)
                self.assertFalse(any(r.raw_pose.valid for r in results))
        # the seed is read before the keep/drop decision, so it is used one frame past the grace time
        self.assertEqual(used, {.1: 3, .5: 6})

    def test_visibility_evaluation_requires_live_performance_and_selects_highest_passing_size(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            mask = np.zeros((80, 100), np.uint8)
            mask[10:70, 10:90] = 255
            cv2.imwrite(str(p/"mask.png"), mask)
            manifest = {"method": "synthetic", "input_size": 252, "frames": [
                {"prediction": "mask.png", "truth": "mask.png", "sequence": "clip", "frame_index": i, "condition": condition}
                for i, condition in enumerate(("front", "behind"))]}
            (p/"manifest.json").write_text(json.dumps(manifest))
            report = evaluate(p/"manifest.json")
            self.assertTrue(report["iou_target_met"])
            self.assertTrue(report["front_and_behind_annotated"])
            self.assertFalse(report["passed"])
            self.assertIsNone(select_resolution([report])["selected_input_size"])
            reports = [{"passed": True, "input_size": n, "mean_visibility_iou": .9, "method": "test"} for n in (252, 392)]
            reports.append({"passed": False, "input_size": 518, "mean_visibility_iou": 1., "method": "test"})
            self.assertEqual(select_resolution(reports)["selected_input_size"], 392)

    def test_association_dump_disabled_leaves_frames_jsonl_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            paths = []
            for dump in (False, True):
                run_cfg = replace(cfg, logging=replace(cfg.logging, enabled=True, output_dir=str(Path(directory)/f"out-{dump}"),
                                                        association_dump=dump))
                pipe = Pipeline(run_cfg, self.cam)
                logger = RunLogger(run_cfg, self.cam, "test", pipe.map.metadata)
                result = pipe.process(self.packet(0, 0.), [self.tag], self.features())
                logger.frame(self.packet(0, 0.), result)
                logger.close()
                paths.append(logger.path)
            rows = [json.loads((p/"frames.jsonl").read_text()) for p in paths]
            for row in rows:
                del row["timings_ms"]  # wall-clock noise; not part of the dump's effect on frames.jsonl
                del row["raw_pose"]["covariance_ms"]
            self.assertEqual(rows[0], rows[1])
            self.assertFalse((paths[0]/"assoc.jsonl").exists())
            self.assertTrue((paths[1]/"assoc.jsonl").exists())

    def test_association_dump_schema_and_scorer_detects_known_wrong_association(self):
        with tempfile.TemporaryDirectory() as directory:
            self.cam.identity.update(K=self.cam.K.tolist(), distortion=self.cam.distortion.tolist())
            cfg = self.config_with_map(directory)
            run_cfg = replace(cfg, logging=replace(cfg.logging, enabled=True, output_dir=str(Path(directory)/"out"),
                                                    association_dump=True))
            pipe = Pipeline(run_cfg, self.cam)
            logger = RunLogger(run_cfg, self.cam, "test", pipe.map.metadata)
            features = self.features()
            features[50].landmark_id, features[50].association_source = 51, "lk"  # deliberately wrong identity
            packet = self.packet(0, 0.)
            result = pipe.process(packet, [self.tag], features, withhold_tag=True)
            self.assertTrue(result.diagnostics["tag_reference_valid"])
            self.assertLessEqual(result.diagnostics["tag_reference_rms_px"], 1.0)
            logger.frame(packet, result)
            logger.close()
            observations = json.loads((logger.path/"assoc.jsonl").read_text().splitlines()[0])["observations"]
            for key in ("track_id", "landmark_id", "association_source", "inherited", "match_distance",
                       "match_ratio", "age_frames", "pixel", "predicted_pixel", "ransac_inlier"):
                self.assertIn(key, observations[0])
            wrong = [o for o in observations if o["landmark_id"] == 51 and o["association_source"] == "lk"]
            self.assertEqual(len(wrong), 1)
            report = score(logger.path)
            self.assertEqual(report["scored_frames"], 1)
            self.assertAlmostEqual(report["precision_within_3px"], 1-1/len(observations), places=6)

    def test_id_revalidation_off_clears_nothing_regardless_of_state(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            features = self.features()
            features[0].landmark_id, features[0].association_source = 1, "lk"  # wrong on purpose
            cleared = revalidate_ids(features, pipe.map, cfg.features.id_revalidation, lost=True)
            self.assertEqual(cfg.features.id_revalidation.mode, "off")
            self.assertEqual(cleared, 0)
            self.assertEqual(features[0].landmark_id, 1)

    def test_id_revalidation_loss_clear_only_while_lost_then_rematches_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            mode = IdRevalidation(mode="loss_clear")
            features = self.features()
            features[0].landmark_id, features[0].association_source = 1, "lk"  # wrong: should be 0
            cleared = revalidate_ids(features, pipe.map, mode, lost=False)
            self.assertEqual(cleared, 0)
            self.assertEqual(features[0].landmark_id, 1)  # not lost: untouched
            cleared = revalidate_ids(features, pipe.map, mode, lost=True)
            self.assertEqual(cleared, 1)
            self.assertIsNone(features[0].landmark_id)
            X, U = associate(features, pipe.map, cfg.features)  # fresh descriptor matching recovers the correct id
            self.assertEqual(features[0].landmark_id, 0)
            self.assertEqual(len({f.landmark_id for f in features if f.landmark_id is not None}), len(X))  # ids stay one-to-one

    def test_id_revalidation_never_clears_a_fresh_orb_map_id(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            features = self.features()
            features[0].landmark_id, features[0].association_source = 1, "orb_map"  # fresh this frame, even if "wrong"
            for mode in (IdRevalidation(mode="loss_clear"), IdRevalidation(mode="age_expire", max_age_frames=0),
                        IdRevalidation(mode="descriptor_check", max_descriptor_distance=0)):
                features[0].frames_since_confirmed = 999
                cleared = revalidate_ids(features, pipe.map, mode, lost=True)
                self.assertEqual(cleared, 0)
                self.assertEqual(features[0].landmark_id, 1)

    def test_id_revalidation_age_expire_clears_only_past_the_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            mode = IdRevalidation(mode="age_expire", max_age_frames=10)
            features = self.features()
            features[0].landmark_id, features[0].association_source, features[0].frames_since_confirmed = 1, "lk", 5
            features[1].landmark_id, features[1].association_source, features[1].frames_since_confirmed = 2, "orb_redetection", 11
            cleared = revalidate_ids(features, pipe.map, mode, lost=False)
            self.assertEqual(cleared, 1)
            self.assertEqual(features[0].landmark_id, 1)  # under the threshold: a correctly-confirmed track keeps its id
            self.assertIsNone(features[1].landmark_id)

    def test_id_revalidation_descriptor_check_clears_only_drifted_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            mode = IdRevalidation(mode="descriptor_check", max_descriptor_distance=10)
            features = self.features()
            features[0].landmark_id, features[0].association_source = 0, "lk"  # unchanged descriptor: distance 0
            features[1].landmark_id, features[1].association_source = 2, "lk"  # claims id 2 but carries id 5's descriptor
            features[1].descriptor = self.desc[5].copy()
            cleared = revalidate_ids(features, pipe.map, mode, lost=False)
            self.assertEqual(cleared, 1)
            self.assertEqual(features[0].landmark_id, 0)
            self.assertIsNone(features[1].landmark_id)

    def test_oracle_nearest_keypoint_and_hit_radii(self):
        proj = np.array([100., 100.])
        keypoints = np.array([[101., 101.], [200., 200.]])
        i, distance = nearest_keypoint(proj, keypoints)
        self.assertEqual(i, 0)
        self.assertAlmostEqual(distance, np.sqrt(2), places=6)
        self.assertEqual(nearest_keypoint(proj, np.empty((0, 2))), (None, float("inf")))

    def test_oracle_grid_quota_survivors_keeps_highest_response_per_cell(self):
        cfg = replace(self.cfg.features, grid_rows=1, grid_cols=2, max_features=2)
        keys = [cv2.KeyPoint(10., 10., 1., response=5.), cv2.KeyPoint(20., 10., 1., response=9.),  # left cell: 9 wins
               cv2.KeyPoint(70., 10., 1., response=3.), cv2.KeyPoint(90., 10., 1., response=1.)]  # right cell: 3 wins
        survivors = grid_quota_survivors(keys, cfg, (100, 20))
        self.assertEqual(survivors, {1, 2})

    def test_oracle_classify_pair_accepts_true_match_and_flags_a_closer_distractor(self):
        def with_distance(n_bits):
            byte_full, rem = divmod(n_bits, 8)
            d = np.zeros(32, np.uint8)
            d[:byte_full] = 0xFF
            if rem:
                d[byte_full] = (1 << rem)-1
            return d
        cfg = self.cfg.features
        query = np.zeros(32, np.uint8)
        true_pair_close = np.stack([with_distance(0), with_distance(40)])  # true landmark 0 exact; landmark 1 far
        frame_descriptors = np.stack([query])
        backward = frame_backward_matches(true_pair_close, frame_descriptors, cfg)
        result = classify_pair(query, true_pair_close, 0, backward, 0, cfg)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "accepted")
        true_pair_beaten = np.stack([with_distance(20), with_distance(5)])  # landmark 1 is closer than true landmark 0
        backward = frame_backward_matches(true_pair_beaten, frame_descriptors, cfg)
        result = classify_pair(query, true_pair_beaten, 0, backward, 0, cfg)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "distractor_closer")
        self.assertEqual(result["rank"], 2)

    def test_oracle_hamming_distances_matches_known_bit_counts(self):
        def with_distance(n_bits):
            byte_full, rem = divmod(n_bits, 8)
            d = np.zeros(32, np.uint8)
            d[:byte_full] = 0xFF
            if rem:
                d[byte_full] = (1 << rem)-1
            return d
        query = np.zeros(32, np.uint8)
        train = np.stack([with_distance(5), with_distance(60)])
        np.testing.assert_allclose(hamming_distances(query, train)[0], [5., 60.])

    def test_descriptor_validity_tight_radius_report_separates_near_and_far_pairs(self):
        rows = [{"nearest_px": 1.0, "true_distance": 5.0, "accepted": True},
               {"nearest_px": 2.5, "true_distance": 60.0, "accepted": False},
               {"nearest_px": 8.0, "true_distance": 110.0, "accepted": False}]
        report = tight_radius_report(rows)
        self.assertEqual(report["2.0"]["count"], 1)
        self.assertEqual(report["2.0"]["median"], 5.0)
        self.assertEqual(report["5.0"]["count"], 2)
        self.assertEqual(report["10.0"]["count"], 3)
        self.assertEqual(report["2.0"]["fraction_under"]["30"], 1.0)

    def test_descriptor_validity_view_change_bin_boundaries(self):
        self.assertEqual(view_change_bin(0), "0-5")
        self.assertEqual(view_change_bin(5), "0-5")
        self.assertEqual(view_change_bin(5.1), "5-15")
        self.assertEqual(view_change_bin(30), "15-30")
        self.assertEqual(view_change_bin(90), "30+")

    def test_descriptor_validity_view_change_report_places_known_angle_in_right_bin(self):
        point = np.array([0., 0., 0.])
        source_T = np.eye(4)
        source_T[:3, 3] = [-1., 0, 0]  # identity rotation: camera_position(T) == -T[:3,3] == [1,0,0]
        provenance = {5: [(0, None, None)]}
        keyframe_poses = [source_T]
        same_view_row = {"landmark_id": 5, "nearest_px": 1., "true_distance": 10., "landmark_point": point.tolist(),
                         "camera_position": [1., 0, 0]}  # identical camera position: angle 0
        far_view_row = {"landmark_id": 5, "nearest_px": 1., "true_distance": 90., "landmark_point": point.tolist(),
                        "camera_position": [0., 1., 0]}  # 90 degrees away
        report = view_change_report([same_view_row, far_view_row], provenance, keyframe_poses)
        self.assertEqual(report["0-5"]["median_true_distance"], 10.)
        self.assertEqual(report["30+"]["median_true_distance"], 90.)

    def test_descriptor_validity_distractor_geometry_flags_near_duplicate_landmark(self):
        from types import SimpleNamespace
        fake_map = SimpleNamespace(ids=np.array([0, 1], np.int64), points=np.array([[0., 0, 0], [0.01, 0, 0]]),  # 1cm apart
                                   descriptors=np.stack([np.zeros(32, np.uint8), np.zeros(32, np.uint8)]))  # identical: zero diversity
        rows = [{"landmark_id": 0, "distractor_id": 1, "reason": "distractor_closer"}]
        report = distractor_geometry_report(rows, fake_map)
        self.assertEqual(report["fraction_within_metres"]["0.02"], 1.0)
        self.assertEqual(report["stored_descriptor_pairwise_hamming"]["median"], 0.0)
        self.assertEqual(report["stored_descriptor_pairwise_hamming"]["fraction_under_40"], 1.0)

    def test_descriptor_validity_self_consistency_zero_for_correct_large_for_wrong(self):
        image = np.zeros((200, 200), np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)  # a strong corner-rich shape for ORB
        keys, descriptors = cv2.ORB_create().detectAndCompute(image, None)
        self.assertGreater(len(keys), 0, "fixture image must produce at least one ORB keypoint")
        pixel = np.array(keys[0].pt)

        from types import SimpleNamespace
        correct_map = SimpleNamespace(ids=np.array([0], np.int64), descriptors=descriptors[:1])  # matches what's actually there
        correct = self_consistency_report(correct_map, {0: [(0, pixel, image)]})
        self.assertEqual(correct["source_view"]["median"], 0.0)

        wrong_map = SimpleNamespace(ids=np.array([0], np.int64), descriptors=np.stack([np.full(32, 0xAA, np.uint8)]))  # planted mismatch
        wrong = self_consistency_report(wrong_map, {0: [(0, pixel, image)]})
        self.assertGreater(wrong["source_view"]["median"], 50)

    def test_id_revalidation_default_leaves_pipeline_output_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config_with_map(directory)
            pipe = Pipeline(cfg, self.cam)
            result = pipe.process(self.packet(0, 0.), [self.tag], self.features())
            self.assertEqual(result.diagnostics["ids_cleared_by_revalidation"], 0)

    def test_step45_margin_fraction_default_and_scaling(self):
        T, panel, oracle, valid = self._tilted_table()
        scene = oracle-.15            # scene 15 cm nearer than the table panel
        _, v0, z = panel_layer(T, panel, self.cam, scene, margin_m=.10)
        _, v1, _ = panel_layer(T, panel, self.cam, scene, margin_m=.10, margin_fraction=0.)
        self.assertTrue(np.array_equal(v0, v1))                # default unchanged
        on = z > 0
        self.assertFalse(v0[on].any())                         # 15 cm > 10 cm floor: hidden
        _, v2, _ = panel_layer(T, panel, self.cam, scene, margin_m=.10, margin_fraction=.30)
        self.assertTrue(v2[on & (z*.30 >= .15)].all())          # fractional margin covers it where 30% of Z >= 15 cm

    def test_step45_config_default_off_and_partial_block(self):
        self.assertFalse(load_config("config-logitech.json").occlusion.enabled)
        with tempfile.TemporaryDirectory() as directory:
            raw = json.loads(Path("config-logitech.json").read_text())
            raw["occlusion"] = {"enabled": True, "line_reference": False}
            path = Path(directory)/"c.json"
            path.write_text(json.dumps(raw))
            block = load_config(path).occlusion
            self.assertTrue(block.enabled and not block.line_reference)
            self.assertEqual(block.input_size, 252)

    def test_step45_relative_line_occlusion_without_pose_draws_nothing(self):
        class Model:
            input_size = 252
        occ = RelativeLineOcclusion(Model())
        result = type("R", (), {"raw_pose": type("P", (), {"valid": False})(), "render_pose": None, "diagnostics": {}})()
        self.assertIsNone(occ.render(None, result, None, None, []))
        self.assertEqual(result.diagnostics["occlusion"]["alignment_source"], "unavailable")

    def test_step46_side_by_side_shape_and_far_config(self):
        T, panel, oracle, valid = self._tilted_table()
        _, visible, z = panel_layer(T, panel, self.cam, oracle-.3, margin_m=.1)
        view = {"scene": oracle-.3, "z": z, "visible": visible, "line_uv": np.array([[10, 10], [20, 20]])}
        out = side_by_side(np.zeros((480, 640, 3), np.uint8), view)
        self.assertEqual(out.shape, (360, 1440, 3))
        cfg = load_config("config-logitech-far.json")
        self.assertEqual((cfg.panel.width_m, cfg.panel.height_m, cfg.filter.enabled), (1.0, 0.5625, True))

    def test_step48_plane_fit_recovers_synthetic_t_s(self):
        T, panel, oracle, valid = self._tilted_table()
        t, s = -3.0, 5.0
        d = t+s/np.where(valid, oracle, 1.)
        d[200:300, 250:400] = d.max()                              # a near "hand" that must not drag the fit
        d += np.random.default_rng(1).normal(0, .01, d.shape)
        class Model:
            input_size = 252
        occ = RelativeLineOcclusion(Model(), reference="plane", plane_extent_m=(-3., 3., -3., 3.), plane_min_span=.2)
        zg, inside = __import__("ar_projection.occlusion", fromlist=["_plane_grid"])._plane_grid(T, self.cam, occ.extent)
        result = type("R", (), {"raw_pose": type("P", (), {"landmark_inliers": 0})()})()
        fit, source, info = occ._plane_reference(d, T, self.cam, result, None, [], zg, inside, None)
        self.assertEqual(source, "plane_fresh")
        self.assertAlmostEqual(fit[0], t, delta=.1)
        self.assertAlmostEqual(fit[1], s, delta=.1)
        self.assertGreaterEqual(info["plane_span"], .2)

    def test_step48_plane_fit_rejects_a_flat_depth_range(self):
        T, panel, oracle, valid = self._tilted_table()
        class Model:
            input_size = 252
        occ = RelativeLineOcclusion(Model(), reference="plane", plane_extent_m=(-.02, .02, -.02, .02), plane_min_span=.3)
        zg, inside = __import__("ar_projection.occlusion", fromlist=["_plane_grid"])._plane_grid(T, self.cam, occ.extent)
        result = type("R", (), {"raw_pose": type("P", (), {"landmark_inliers": 0})()})()
        fit, source, info = occ._plane_reference(-3+5/np.where(valid, oracle, 1.), T, self.cam, result, None, [], zg, inside, None)
        self.assertIsNone(fit)
        self.assertEqual(source, "unavailable")

    def test_step48_background_prior_identity_warp_and_never_stores_a_near_blob(self):
        prior = BackgroundPrior(PriorConfig(enabled=True))
        T = np.eye(4)
        h, w = 120, 160
        Zs = np.full((h, w), 3.)
        Zs[40:80, 60:100] = .5                                   # near blob, e.g. a hand
        Pw, Cw = prior.warp(T, self.cam)
        self.assertTrue(np.isnan(Pw).all())
        prior.update(Zs, Pw, Cw, np.full((h, w), np.nan), np.zeros((h, w)), T)
        self.assertTrue(np.isnan(prior.P[50:70, 70:90]).all())   # blob not stored
        self.assertAlmostEqual(float(np.nanmedian(prior.P)), 3.)
        Pw2, Cw2 = prior.warp(T, self.cam)
        self.assertTrue(np.array_equal(np.isfinite(Pw2), np.isfinite(prior.P)))
        self.assertTrue(np.allclose(Pw2[np.isfinite(Pw2)], prior.P[np.isfinite(prior.P)]))
        # a static occluder nearer than the panel inside its footprint is not stored either
        prior2 = BackgroundPrior(PriorConfig(enabled=True))
        prior2.warp(T, self.cam)
        Zs2 = np.full((h, w), 3.)
        Zs2[40:80, 60:100] = 1.0
        footprint = np.zeros((h, w)); footprint[20:100, 40:120] = 2.0
        prior2.update(Zs2, np.full((h, w), np.nan), np.zeros((h, w)), np.full((h, w), np.nan), footprint, T)
        self.assertTrue(np.isnan(prior2.P[45:75, 65:95]).all())

    def test_step48_defaults_keep_line_reference_and_prior_off(self):
        o = load_config("config-logitech-far.json").occlusion
        self.assertEqual((o.reference, o.background_prior.enabled), ("line", False))
        class Model:
            input_size = 252
        occ = RelativeLineOcclusion(Model())
        self.assertEqual((occ.reference, occ.prior), ("line", None))


if __name__ == "__main__":
    unittest.main()
