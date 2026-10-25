import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import classifiers
import inspect_probabilities
import jester_pipeline as jp
import landmarks as lm
import run_training_queue as queue
import training
import training_data as td
import training_reports as reports


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["No gesture", "Doing other things"] + [f"gesture {i:02d}" for i in range(25)]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_pose(wrist: tuple[float, float], size: float) -> np.ndarray:
    base = np.stack((np.linspace(0, 1, 21), np.tile(np.linspace(0, 1, 7), 3)), axis=1).astype(np.float32)
    return np.asarray(wrist, np.float32) + base * np.float32(size)


class SharedInputFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT)
        cls.root = Path(cls.temp.name)
        labels_dir = cls.root / "jester_labels"
        labels_dir.mkdir()
        (labels_dir / "labels.csv").write_text("".join(f"{label}\n" for label in CLASSES), encoding="utf-8")
        train, validation, test = [], [], []
        clip_id = 1
        for label in CLASSES:
            for _ in range(6):
                train.append(f"{clip_id};{label}\n")
                clip_id += 1
            validation.append(f"{clip_id};{label}\n")
            clip_id += 1
            test.append(f"{clip_id}\n")
            clip_id += 1
        (labels_dir / "train.csv").write_text("".join(train), encoding="utf-8")
        (labels_dir / "validation.csv").write_text("".join(validation), encoding="utf-8")
        (labels_dir / "test.csv").write_text("".join(test), encoding="utf-8")
        (labels_dir / "test-answers.csv").write_text("not used\n", encoding="utf-8")
        cls.cache_root = cls.root / "cache"
        paths = jp.cache_paths({"_cache_root": cls.cache_root})
        jp.claim_cache(cls.cache_root)
        paths["metadata"].mkdir(parents=True)
        folds = paths["metadata"] / "fold_assignments.csv"
        jp.generate_folds(labels_dir, folds, paths["fold_views"], 42)
        review_path = paths["metadata"] / "class_review.csv"
        jp.create_review_template(labels_dir, review_path)
        review = read_csv(review_path)
        with review_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(review[0]))
            writer.writeheader()
            for row in review:
                row["compatibility_decision"] = "retain"
                row["reason"] = "synthetic fixture"
                writer.writerow(row)

        classes, records = jp.read_labels(labels_dir)
        frames_root = paths["frames"]
        for cid, record in records.items():
            if record["official_split"] == "test":
                continue
            clip_dir = frames_root / cid
            clip_dir.mkdir(parents=True)
            for frame in range(1, 4):
                image = np.full((100, 176, 3), frame, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", image)
                if not ok:
                    raise AssertionError("fixture JPEG encode failed")
                (clip_dir / f"{frame:05d}.jpg").write_bytes(encoded.tobytes())
            scan = jp.scan_frame_dir(clip_dir)
            frame_marker = {"status": "complete", "frame_count": 3, "frame_set_sha256": jp.frame_set_hash(scan["entries"])}
            jp.write_json(clip_dir / "complete.json", frame_marker)
        manifest = paths["metadata"] / "clip_manifest.csv"
        jp.build_manifest(labels_dir, frames_root, manifest, folds, None, review_path)

        raw_root = paths["landmarks"] / "rtmlib"
        raw_root.mkdir(parents=True)
        for cid, record in records.items():
            if record["official_split"] == "test":
                continue
            all_missing = cid == "1"
            candidates = []
            offsets = [0]
            counts = []
            for frame in range(3):
                if all_missing or frame == 2:
                    current = []
                elif frame == 0:
                    current = [make_pose((0.10, 0.10), 0.10), make_pose((0.40, 0.40), 0.30)]
                else:
                    current = [make_pose((0.41, 0.40), 0.12), make_pose((0.82, 0.82), 0.10)]
                candidates.extend(current)
                counts.append(len(current))
                offsets.append(len(candidates))
            xy = np.asarray(candidates, dtype=np.float32).reshape(-1, 21, 2)
            metadata = {
                "clip_id": cid, "class_label": record["class_label"], "official_split": record["official_split"],
                "source_frame_set_sha256": frame_marker["frame_set_sha256"], "joints": list(lm.JOINTS),
                "frame_rate_fps": 12, "extractor": "rtmlib", "input_identity": f"fixture-{cid}",
            }
            arrays = {
                "frame_indices": np.asarray([1, 2, 3], dtype=np.int32),
                "frame_timestamps_seconds": np.asarray([0.0, 1 / 12, 2 / 12], dtype=np.float64),
                "source_frame_names": np.asarray(["00001.jpg", "00002.jpg", "00003.jpg"], dtype="U64"),
                "frame_valid": np.ones(3, dtype=np.bool_), "image_widths": np.full(3, 176, dtype=np.int32),
                "image_heights": np.full(3, 100, dtype=np.int32), "frame_hand_offsets": np.asarray(offsets, dtype=np.int64),
                "detected_hand_count": np.asarray(counts, dtype=np.int16), "hand_xy": xy,
                "joint_mask": np.isfinite(xy).all(axis=-1), "joint_scores": np.ones((len(xy), 21), np.float32),
                "detection_scores": np.full(len(xy), np.nan, np.float32), "handedness": np.full(len(xy), "", dtype="U16"),
                "handedness_scores": np.full(len(xy), np.nan, np.float32), "visibility": np.full((len(xy), 21), np.nan, np.float32),
                "presence": np.full((len(xy), 21), np.nan, np.float32), "metadata_json": np.asarray(json.dumps(metadata)),
            }
            npz = raw_root / f"{cid}.npz"
            lm._atomic_npz(npz, arrays)
            jp.write_json(raw_root / f"{cid}.complete.json", {
                "status": "complete", "input_identity": f"fixture-{cid}",
                "input_frame_set_sha256": frame_marker["frame_set_sha256"], "output_sha256": jp.sha256_file(npz),
            })
        cls.folds_before = jp.sha256_file(folds)
        (cls.root / "config.json").write_text(json.dumps({
            "cloud_root": str(cls.root), "local_cache_root": "cache", "cache_limit_gb": 2, "labels_dir": "jester_labels",
        }), encoding="utf-8")
        cls.training_config_path = cls.root / "training_config.json"
        cls.training_config_path.write_text(json.dumps({
            "pipeline_config": "config.json", "extractor": "rtmlib", "sequence_length": 8, "seed": 42,
            "epochs": 100, "patience": 10, "batch_size": 64, "learning_rate": 0.001,
            "gradient_clip_norm": 1.0, "device": "cpu",
        }), encoding="utf-8")
        cls.feature_cache_path = td.build_feature_cache(cls.training_config_path)
        cls.feature_cache = td.validate_feature_cache(cls.feature_cache_path)

    @classmethod
    def tearDownClass(cls):
        features = getattr(cls, "feature_cache", {}).get("features")
        if features is not None and getattr(features, "_mmap", None) is not None:
            features._mmap.close()
        cls.temp.cleanup()

    def test_content_addressed_cache_reuses_saved_folds_and_excludes_test(self):
        paths = jp.cache_paths({"_cache_root": self.cache_root})
        self.assertEqual(jp.sha256_file(paths["metadata"] / "fold_assignments.csv"), self.folds_before)
        index = self.feature_cache["index"]
        self.assertEqual(len(index), 27 * 7)
        self.assertTrue(all(row["official_split"] != "test" for row in index))
        self.assertEqual({row["compatibility_decision"] for row in index}, {"retain"})
        self.assertEqual(td.build_feature_cache(self.training_config_path), self.feature_cache_path)
        self.assertEqual(self.feature_cache["features"].shape, (189, 8, 43))
        self.assertNotIn("ids_file_sha256", self.feature_cache["metadata"]["identity_inputs"])

    def comparison_config(self):
        rows, seen = [], set()
        for row in self.feature_cache["index"]:
            key = (row["class_label"], row["official_split"], row["fold"])
            if key not in seen:
                rows.append({name: row[name] for name in ("clip_id", "class_label", "official_split")})
                seen.add(key)
        ids = self.root / "comparison_ids.csv"
        jp._write_csv(ids, tuple(rows[0]), rows)
        config = json.loads(self.training_config_path.read_text(encoding="utf-8"))
        config["ids_file"] = ids.name
        path = self.root / "comparison_config.json"
        jp.write_json(path, config)
        return path, ids, rows

    def test_clip_subset_ignores_unselected_raw_sources_and_freezes_selection(self):
        path, ids, rows = self.comparison_config()
        selected = {row["clip_id"] for row in rows}
        config = td.load_config(path)
        self.assertEqual(config["_ids_file"], ids.resolve())
        unselected = next(row["clip_id"] for row in self.feature_cache["index"] if row["clip_id"] not in selected)
        raw = self.cache_root / "landmarks" / "rtmlib" / f"{unselected}.npz"
        hidden = raw.with_suffix(".unused")
        manifest = self.cache_root / "metadata" / "clip_manifest.csv"
        original_manifest = manifest.read_bytes()
        raw.rename(hidden)
        subset = None
        try:
            manifest_rows = read_csv(manifest)
            for row in manifest_rows:
                if row["clip_id"] == unselected:
                    row["extraction_status"] = "not_extracted"
            jp._write_csv(manifest, tuple(manifest_rows[0]), manifest_rows)
            source = td.inspect_inputs(path, require_features=False)
            self.assertEqual(source["blockers"], [])
            self.assertEqual(source["source"]["retained_ids"], selected)
            subset_path = td.build_feature_cache(path)
            subset = td.validate_feature_cache(subset_path)
            self.assertEqual(subset["features"].shape, (108, 8, 43))
            self.assertEqual({row["clip_id"] for row in subset["index"]}, selected)
            self.assertNotEqual(subset["identity"], self.feature_cache["identity"])
            self.assertEqual(jp.sha256_file(config["_paths"]["metadata"] / "fold_assignments.csv"), self.folds_before)
            with patch.object(jp, "validate_frame_completion", side_effect=AssertionError("RGB reopened")):
                frozen = td.inspect_inputs(path)
            self.assertEqual(frozen["blockers"], [])
            frozen["feature_cache"]["features"]._mmap.close()
            ids.write_bytes(ids.read_bytes() + b"\n")
            stale = td.inspect_inputs(path)
            self.assertTrue(stale["blockers"])
            self.assertIsNone(stale["feature_cache"])
            rows.append(next({name: row[name] for name in ("clip_id", "class_label", "official_split")}
                             for row in self.feature_cache["index"] if row["clip_id"] == unselected))
            jp._write_csv(ids, tuple(rows[0]), rows)
            expanded = td.inspect_inputs(path, require_features=False)
            self.assertTrue(expanded["blockers"])
            self.assertIn(unselected, expanded["source"]["retained_ids"])
            self.assertNotEqual(td._source_settings(config, CLASSES)["ids_file_sha256"], subset["metadata"]["identity_inputs"]["ids_file_sha256"])
        finally:
            if subset is not None:
                subset["features"]._mmap.close()
            hidden.rename(raw)
            manifest.write_bytes(original_manifest)

    def test_clip_subset_rejects_invalid_ids_metadata_and_missing_fold_or_holdout(self):
        path, ids, rows = self.comparison_config()
        test_id = next(row["clip_id"] for row in read_csv(self.cache_root / "metadata" / "clip_manifest.csv") if row["official_split"] == "test")
        for extra, message in (
            ({"clip_id": rows[0]["clip_id"]}, "unique"),
            ({"clip_id": ""}, "nonempty"),
            ({"clip_id": "9999999"}, "unknown"),
            ({"clip_id": test_id}, "non-training/validation"),
        ):
            with self.subTest(extra=extra):
                jp._write_csv(ids, ("clip_id", "class_label", "official_split"), rows + [extra])
                with self.assertRaisesRegex(ValueError, message):
                    td.inspect_inputs(path, require_features=False)
        for key in ("class_label", "official_split"):
            changed = [rows[0] | {key: "wrong"}, *rows[1:]]
            jp._write_csv(ids, tuple(rows[0]), changed)
            with self.assertRaisesRegex(ValueError, "mismatched"):
                td.inspect_inputs(path, require_features=False)
        for split, message in (("train", "in fold"), ("validation", "lacks official validation")):
            removed = next(row for row in rows if row["official_split"] == split)
            jp._write_csv(ids, tuple(rows[0]), [row for row in rows if row != removed])
            with self.assertRaisesRegex(ValueError, message):
                td.inspect_inputs(path, require_features=False)
        ids.unlink()
        self.assertIn("missing clip ID selection", td.inspect_inputs(path)["blockers"][0])

    def test_clip_subset_rejects_excluded_classes(self):
        path, ids, rows = self.comparison_config()
        review = self.cache_root / "metadata" / "class_review.csv"
        original = review.read_bytes()
        try:
            classes = read_csv(review)
            classes[-1]["compatibility_decision"] = "exclude"
            jp._write_csv(review, tuple(classes[0]), classes)
            with self.assertRaisesRegex(ValueError, "excluded classes"):
                td.inspect_inputs(path, require_features=False)
        finally:
            review.write_bytes(original)

    def test_all_missing_clip_is_kept_with_zero_coordinates_and_flag(self):
        index = next(row for row in self.feature_cache["index"] if row["clip_id"] == "1")
        row = self.feature_cache["features"][int(index["feature_index"])]
        self.assertTrue(np.all(row[:, :42] == 0))
        self.assertTrue(np.all(row[:, 42] == 0))
        self.assertEqual(index["all_missing"], "true")
        self.assertEqual(index["valid_pose_frames"], "0")

    def test_training_uses_frozen_cache_without_opening_raw_frames(self):
        with patch.object(jp, "validate_frame_completion", side_effect=AssertionError("RGB reopened")), patch.object(lm, "_read_marker", side_effect=AssertionError("raw cache reopened")):
            result = td.inspect_inputs(self.training_config_path)
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["feature_cache"]["identity"], self.feature_cache["identity"])
        result["feature_cache"]["features"]._mmap.close()

    def test_frozen_cache_rejects_changed_source_metadata(self):
        review = self.cache_root / "metadata" / "class_review.csv"
        original = review.read_bytes()
        try:
            review.write_bytes(original + b"\n")
            with patch.object(jp, "validate_frame_completion", side_effect=AssertionError("RGB reopened")):
                result = td.inspect_inputs(self.training_config_path)
            self.assertTrue(result["blockers"])
            self.assertIsNone(result["feature_cache"])
        finally:
            review.write_bytes(original)

    def test_eleven_class_subset_gets_new_cache_mapping_and_experiment_without_changing_folds(self):
        config = td.load_config(self.training_config_path)
        paths = config["_paths"]
        review_path = paths["metadata"] / "class_review.csv"
        manifest_path = paths["metadata"] / "clip_manifest.csv"
        originals = {path: path.read_bytes() for path in (review_path, manifest_path)}
        before = queue._experiment_snapshot(config, {}, self.feature_cache, "subset-test")
        subset_cache = None
        try:
            rows = read_csv(review_path)
            for row in rows:
                row["compatibility_decision"] = "retain" if row["class_label"] in CLASSES[:11] else "exclude"
                row["reason"] = "Outside selected gesture set"
            jp._write_csv(review_path, tuple(rows[0]), rows)
            jp.build_manifest(self.root / "jester_labels", paths["frames"], manifest_path,
                              paths["metadata"] / "fold_assignments.csv", None, review_path)
            self.assertEqual(len(read_csv(manifest_path)), 27 * 8)
            stale = td.inspect_inputs(self.training_config_path)
            self.assertTrue(stale["blockers"])
            self.assertIsNone(stale["feature_cache"])
            subset_path = td.build_feature_cache(self.training_config_path)
            subset_cache = td.validate_feature_cache(subset_path)
            self.assertNotEqual(subset_cache["identity"], self.feature_cache["identity"])
            self.assertTrue(self.feature_cache_path.is_dir())
            self.assertEqual(subset_cache["metadata"]["class_order"], CLASSES[:11])
            self.assertEqual(subset_cache["features"].shape, (77, 8, 43))
            self.assertEqual({row["class_label"] for row in subset_cache["index"]}, set(CLASSES[:11]))
            self.assertFalse(any(row["official_split"] == "test" for row in subset_cache["index"]))
            self.assertEqual(jp.sha256_file(paths["metadata"] / "fold_assignments.csv"), self.folds_before)
            np.testing.assert_array_equal(training._class_ids(np.asarray(CLASSES[:11]), CLASSES[:11]), np.arange(11))
            after = queue._experiment_snapshot(config, {}, subset_cache, "subset-test")
            self.assertNotEqual(before["identity"], after["identity"])
            experiment_path = self.root / "subset-experiment.json"
            jp.write_json(experiment_path, before)
            with self.assertRaisesRegex(ValueError, "resume rejected"):
                queue._check_experiment(experiment_path, after)
            predictions = self.root / "subset-predictions.csv"
            labels = np.asarray([CLASSES[10]], dtype=object)
            training._write_predictions(predictions, ["fixture"], labels, {
                "probabilities": np.eye(11, dtype=np.float32)[[10]], "predicted": np.asarray([10]),
            }, CLASSES[:11], "gru", "subset-test")
            training._validate_predictions(predictions, ["fixture"], labels, CLASSES[:11], "gru", "subset-test")
            self.assertEqual(read_csv(predictions)[0]["predicted_class"], CLASSES[10])
        finally:
            if subset_cache is not None:
                subset_cache["features"]._mmap.close()
            for path, content in originals.items():
                path.write_bytes(content)

    def test_preparation_still_verifies_raw_sources(self):
        with patch.object(jp, "validate_frame_completion", side_effect=RuntimeError("raw source changed")):
            with self.assertRaisesRegex(RuntimeError, "raw source changed"):
                td.build_feature_cache(self.training_config_path)

    def test_frozen_cache_rejects_tampered_class_order(self):
        path = self.feature_cache_path / "metadata.json"
        original = path.read_bytes()
        try:
            metadata = json.loads(original)
            metadata["class_order"] = list(reversed(metadata["class_order"]))
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen identity"):
                td.validate_feature_cache(self.feature_cache_path)
        finally:
            path.write_bytes(original)

    def test_final_job_passes_integer_epoch_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            experiment = Path(scratch)
            frozen = {"identity": "test", "data_identity": self.feature_cache["identity"], "class_order": CLASSES,
                      "final_fit_epoch_budgets": {"gru": {"epochs": 2, "median_best_cv_epoch": 2, "fold_best_epochs": [1, 2, 3]}}}
            (experiment / "experiment.json").write_text(json.dumps(frozen), encoding="utf-8")
            weights = experiment / "weights.pt"
            torch.save({"model_state": classifiers.build_model("gru", len(CLASSES)).state_dict()}, weights)
            with patch.object(training, "_load_features", return_value=({"device": "cpu"}, {}, self.feature_cache, CLASSES)), patch.object(training, "_fit", return_value={"checkpoint": weights, "best_epoch": 2}) as fit, patch.object(training, "_prediction_eval"):
                training.run_notebook_job(str(self.training_config_path), str(experiment), json.dumps({"kind": "final", "model": "gru"}), "test")
            self.assertEqual(fit.call_args.args[-1], 2)

    def test_landmark_extraction_counts_cache_once_for_multiple_clips(self):
        ids = self.root / "two_clips.csv"
        ids.write_text("clip_id\n1\n2\n", encoding="utf-8")
        args = SimpleNamespace(extractor="mediapipe", ids_file=ids, model=Path("unused"))
        config = jp.load_config(self.root / "config.json")
        adapter = SimpleNamespace(detect=lambda *args: lm._empty_result(), close=lambda: None)
        package = {"extractor": "mediapipe", "joints": list(lm.JOINTS), "frame_rate_fps": 12}
        with patch.object(lm, "_package_info", return_value=package), patch.object(lm, "_adapter", return_value=adapter), patch.object(jp, "_cache_bytes", wraps=jp._cache_bytes) as scans:
            result = lm.extract(args, config)
        self.assertEqual(result["completed_new"], 2)
        self.assertEqual(scans.call_count, 1)

    def test_sampled_timeline_repeats_short_clip_and_preserves_original_times(self):
        indices = td._sample_indices(3, 8)
        np.testing.assert_array_equal(indices, [0, 0, 1, 1, 1, 1, 2, 2])
        audit = np.load(self.feature_cache_path / "sample_timestamps.npy", mmap_mode="r", allow_pickle=False)
        row = self.feature_cache["features"][0]
        self.assertEqual(row.shape, (8, 43))
        np.testing.assert_allclose(audit[0], indices / 12)
        audit._mmap.close()

    def test_candidate_selection_tracks_wrist_through_missing_frame(self):
        poses = np.asarray([
            make_pose((0.10, 0.10), 0.10), make_pose((0.40, 0.40), 0.30),
            make_pose((0.41, 0.40), 0.12),
            make_pose((0.80, 0.80), 0.10), make_pose((0.42, 0.40), 0.12),
        ])
        offsets = np.asarray([0, 2, 3, 3, 5])
        selected, valid, chosen, counts = td._select_candidates(poses, np.ones((5, 21), bool), offsets)
        np.testing.assert_array_equal(chosen, [1, 0, -1, 1])
        np.testing.assert_array_equal(counts, [2, 1, 0, 2])
        np.testing.assert_array_equal(valid, [True, True, False, True])
        np.testing.assert_array_equal(selected[2], np.zeros((21, 2), dtype=np.float32))

    def test_first_frame_and_nearest_wrist_ties_use_candidate_index(self):
        pose = make_pose((0.25, 0.25), 0.25)
        poses = np.asarray([pose, pose, make_pose((0.375, 0.25), 0.125), make_pose((0.125, 0.25), 0.125)])
        _, _, chosen, _ = td._select_candidates(poses, np.ones((4, 21), bool), np.asarray([0, 2, 4]))
        np.testing.assert_array_equal(chosen, [0, 0])


class TrainingUnitTests(unittest.TestCase):
    def test_temporal_sampling_and_feature_width(self):
        self.assertEqual(td.FEATURES, 43)
        np.testing.assert_array_equal(td._sample_indices(20, 8), [0, 3, 5, 8, 11, 14, 16, 19])
        with self.assertRaises(ValueError):
            td.load_config(ROOT / "training_config.json")
            # Default is valid; the direct shape contract rejects odd T.
            classifiers._validate_input(torch.zeros((1, 9, 43)))

    def test_batch_singleton_merges_without_dropping_samples(self):
        self.assertEqual(training.batch_sizes(65, 64), [63, 2])
        self.assertEqual(training.batch_sizes(129, 64), [64, 63, 2])
        self.assertEqual(sum(training.batch_sizes(65, 64)), 65)
        with self.assertRaises(ValueError):
            training.batch_sizes(1, 64)

    def test_recurrent_and_ddnet_shapes_gradients_and_logits(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            x = torch.rand(2, 8, 43)
            x[..., 42] = 1
            for count in (27, 11):
                for name in ("gru", "lstm", "ddnet"):
                    model = classifiers.build_model(name, count)
                    logits = model(x)
                    self.assertEqual(logits.shape, (2, count))
                    self.assertFalse(torch.allclose(logits.sum(dim=1), torch.ones(2)))
                    logits.square().mean().backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.set_num_threads(old_threads)

    def test_ddnet_stream_sizes_distances_and_missing_masks(self):
        x = torch.zeros(2, 8, 43)
        x[0, :, 42] = 1
        x[0, 2, 42] = 0
        x[0, :, 2] = 3
        x[0, :, 3] = 4
        x[1, :, :42] = torch.rand(8, 42)
        static, slow, fast = classifiers.ddnet_features(x)
        self.assertEqual(tuple(static.shape), (2, 8, 211))
        self.assertEqual(tuple(slow.shape), (2, 8, 43))
        self.assertEqual(tuple(fast.shape), (2, 4, 43))
        self.assertAlmostEqual(float(static[0, 0, 0]), 5.0, places=5)
        self.assertTrue(torch.all(static[1] == 0))
        self.assertTrue(torch.all(slow[1] == 0))
        self.assertTrue(torch.all(fast[1] == 0))
        for stream in (slow, fast):
            invalid = stream[..., -1] == 0
            self.assertTrue(torch.all(stream[..., :-1][invalid] == 0))

    def test_metric_background_rates_have_explicit_denominators(self):
        scores = training._metrics(np.asarray([0, 0, 2, 2, 2]), np.asarray([2, 0, 1, 2, 0]), ["No gesture", "Doing other things", "Swipe"])
        self.assertEqual(scores["background_to_gesture"], {"numerator": 1, "denominator": 2})
        self.assertEqual(scores["gesture_to_background"], {"numerator": 2, "denominator": 3})
        self.assertEqual(list(scores["per_class"]), ["No gesture", "Doing other things", "Swipe"])

    def test_worker_batches_preserve_order_labels_and_singleton_handling(self):
        features = np.arange(65 * 8 * 43, dtype=np.float32).reshape(65, 8, 43)
        positions = list(reversed(range(65)))
        labels = np.arange(65, dtype=np.int64) % 3
        batches = []
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            path = Path(scratch) / "features.npy"
            np.save(path, features)
            mapped = np.load(path, mmap_mode="r")
            loader = training._training_loader(mapped, positions, labels, batches, 4)
            del mapped
            before = torch.get_rng_state().clone()
            try:
                for seed in (42, 43):
                    order = torch.randperm(65, generator=torch.Generator().manual_seed(seed)).tolist()
                    batches[:] = [order[:63], order[63:]]
                    loaded = list(loader)
                    self.assertEqual([len(y) for x, y in loaded], training.batch_sizes(65))
                    torch.testing.assert_close(torch.cat([x for x, y in loaded]), torch.from_numpy(features[np.asarray(positions)[order]]))
                    torch.testing.assert_close(torch.cat([y for x, y in loaded]), torch.from_numpy(labels[order]))
                torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            finally:
                del loader
        for invalid in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "num_workers"):
                training._training_loader(features, positions, labels, batches, invalid)

    def test_checkpoint_resume_matches_uninterrupted_weights_optimizer_and_rng(self):
        torch.set_num_threads(1)
        cache = {"features": np.random.default_rng(4).normal(size=(4, 8, 43)).astype(np.float32)}
        positions = [0, 1, 2, 3]
        labels = np.asarray(["a", "b", "a", "b"], dtype=object)
        config = {"seed": 42, "batch_size": 64, "learning_rate": 0.001, "gradient_clip_norm": 1.0, "device": "cpu"}
        job = {"model": "gru", "kind": "final"}
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            root = Path(scratch)
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            a = training._fit("gru", positions, labels, cache, ["a", "b"], config, job, "same-id", uninterrupted, final_epochs=3)
            # Compare four spawned workers and resume against the zero-worker baseline.
            np.save(root / "features.npy", cache["features"])
            cache["features"] = np.load(root / "features.npy", mmap_mode="r")
            config["num_workers"] = 4
            original_save = training._atomic_torch_save
            interrupted_once = {"value": False}

            def interrupt_after_epoch(path, value):
                original_save(path, value)
                if path.name == "last.pt" and value["completed_epoch"] == 1 and not interrupted_once["value"]:
                    interrupted_once["value"] = True
                    raise KeyboardInterrupt("synthetic interruption")

            with patch.object(training, "_atomic_torch_save", side_effect=interrupt_after_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    training._fit("gru", positions, labels, cache, ["a", "b"], config, job, "same-id", resumed, final_epochs=3)
            b = training._fit("gru", positions, labels, cache, ["a", "b"], config, job, "same-id", resumed, final_epochs=3)
            checkpoint_a = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
            checkpoint_b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
            for key in checkpoint_a["model_state"]:
                torch.testing.assert_close(checkpoint_a["model_state"][key], checkpoint_b["model_state"][key], rtol=0, atol=0)
            self.assertEqual(checkpoint_a["history"], checkpoint_b["history"])
            self.assertEqual(checkpoint_a["completed_epoch"], checkpoint_b["completed_epoch"])
            self.assertEqual(checkpoint_a["optimizer_state"]["param_groups"], checkpoint_b["optimizer_state"]["param_groups"])
            for key, value in checkpoint_a["optimizer_state"]["state"].items():
                for name, tensor in value.items():
                    if torch.is_tensor(tensor):
                        torch.testing.assert_close(tensor, checkpoint_b["optimizer_state"]["state"][key][name], rtol=0, atol=0)
            torch.testing.assert_close(checkpoint_a["rng"]["torch_cpu"], checkpoint_b["rng"]["torch_cpu"], rtol=0, atol=0)
            torch.testing.assert_close(checkpoint_a["rng"]["shuffle"], checkpoint_b["rng"]["shuffle"], rtol=0, atol=0)
            self.assertEqual(a["training"]["completed_epochs"], 3)
            self.assertEqual(b["training"]["completed_epochs"], 3)
            del cache["features"]
            with patch.object(classifiers, "build_model", side_effect=AssertionError("completed training must be reused")):
                reused = training._fit("gru", positions, labels, cache, ["a", "b"], config, job, "same-id", resumed, final_epochs=3)
            self.assertEqual(reused["training"]["best_epoch"], 3)

    def test_cv_checkpoint_earliest_tie_and_patience(self):
        torch.set_num_threads(1)
        cache = {
            "features": np.random.default_rng(5).normal(size=(4, 8, 43)).astype(np.float32),
            "index": [
                {"feature_index": "0", "clip_id": "1", "class_label": "a", "official_split": "train", "fold": "1"},
                {"feature_index": "1", "clip_id": "2", "class_label": "b", "official_split": "train", "fold": "1"},
                {"feature_index": "2", "clip_id": "3", "class_label": "a", "official_split": "train", "fold": "2"},
                {"feature_index": "3", "clip_id": "4", "class_label": "b", "official_split": "train", "fold": "3"},
            ],
        }
        job = {"model": "gru", "kind": "cv", "fold": 1}
        scores = iter((0.2, 0.7, 0.7, 0.7))
        config = {"seed": 42, "epochs": 10, "patience": 2, "batch_size": 64, "learning_rate": 0.001, "gradient_clip_norm": 1.0, "device": "cpu"}
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch, patch.object(training, "_predict", side_effect=lambda *a, **k: (None, {"macro_f1": next(scores)})), patch.object(training, "_evaluation_loss", return_value=1.0):
            result = training._fit("gru", [2, 3], np.asarray(["a", "b"], object), cache, ["a", "b"], config, job, "cv-id", Path(scratch))
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["training"]["completed_epochs"], 4)
            self.assertFalse(result["training"]["reached_epoch_cap"])
            self.assertEqual([row["epoch"] for row in result["history"]], [1, 2, 3, 4])

    def test_training_config_enforces_common_even_t(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            root = Path(scratch)
            (root / "config.json").write_text(json.dumps({"cloud_root": str(root), "local_cache_root": "cache", "cache_limit_gb": 1}), encoding="utf-8")
            path = root / "training.json"
            path.write_text(json.dumps({"pipeline_config": "config.json", "sequence_length": 9}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "even"):
                td.load_config(path)


class QueueAndInspectionTests(unittest.TestCase):
    def test_exact_12_job_order_and_epoch_median(self):
        names = [training.job_name(job) for job in queue.job_plan()]
        self.assertEqual(names, [f"{model}_cv_fold_{fold}" for model in ("gru", "lstm", "ddnet") for fold in (1, 2, 3)] + [f"{model}_final" for model in ("gru", "lstm", "ddnet")])
        self.assertEqual(queue.median_epoch_budget([12, 4, 8]), 8)
        with self.assertRaises(ValueError):
            queue.median_epoch_budget([2, 3])
        self.assertNotEqual(queue._lock_path(Path("experiments/run.a")), queue._lock_path(Path("experiments/run.b")))

    def test_check_mode_lists_jobs_and_starts_no_kernel(self):
        with patch.object(queue.td, "inspect_inputs", return_value={"blockers": ["fixture not ready"], "feature_cache": None}), patch.object(queue.pm, "execute_notebook") as execute, contextlib.redirect_stdout(io.StringIO()):
            code = queue.main(["--config", "unused.json", "--experiment-id", "read-only"])
        self.assertEqual(code, 2)
        execute.assert_not_called()

    def test_ready_check_reserves_output_space_without_kernel(self):
        inspected = {"blockers": [], "config": {"_paths": {"root": Path("cache")}}, "feature_cache": {"index": []}}
        with patch.object(queue.td, "inspect_inputs", return_value=inspected), patch.object(queue, "_reserve_training_space", return_value=1024) as reserve, patch.object(queue.pm, "execute_notebook") as execute, contextlib.redirect_stdout(io.StringIO()):
            code = queue.main(["--config", "unused.json", "--experiment-id", "ready-check"])
        self.assertEqual(code, 0)
        reserve.assert_called_once_with(inspected)
        execute.assert_not_called()

    def test_training_space_estimate_covers_predictions_and_checkpoints(self):
        cache = {
            "index": [{"official_split": "train"}] * 6 + [{"official_split": "validation"}] * 2,
            "metadata": {"class_order": ["background", "gesture"]},
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            inspected = {
                "config": {"_paths": {"root": Path(scratch)}, "_pipeline": {"cache_limit_gb": 2}},
                "feature_cache": cache,
            }
            estimate = queue._estimated_training_output_bytes(cache)
            self.assertGreater(estimate, 128 * 1024**2)
            with patch.object(queue.td.jp, "ensure_capacity") as capacity:
                additional = queue._reserve_training_space(inspected, existing_output_bytes=1024)
            self.assertEqual(additional, estimate - 1024)
            capacity.assert_called_once_with(Path(scratch), 2, estimate - 1024)

    def test_compare_mode_reuses_completed_jobs_without_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            root = Path(scratch)
            experiment_dir = root / "training" / "experiments" / "saved-run"
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "experiment.json").write_text("{}", encoding="utf-8")
            config = {"_paths": {"root": root}}
            inspected = {"blockers": [], "config": config, "feature_cache": {}, "source": {}}
            with (
                patch.object(queue.td, "inspect_inputs", return_value=inspected),
                patch.object(queue, "_experiment_snapshot", return_value={}),
                patch.object(queue, "_check_experiment", return_value={"identity": "fixture", "final_fit_epoch_budgets": [4, 4, 4]}),
                patch.object(queue, "_cv_budgets", return_value=[4, 4, 4]),
                patch.object(queue.training, "validate_job_complete", return_value=True),
                patch.object(queue, "_comparison_done", return_value=False),
                patch.object(queue, "_write_comparison_done"),
                patch.object(queue.training_reports, "generate_reports") as generate,
                patch.object(queue.pm, "execute_notebook") as execute,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = queue.main(["--config", "unused.json", "--experiment-id", "saved-run", "--compare"])
            self.assertEqual(code, 0)
            generate.assert_called_once()
            execute.assert_not_called()

    def test_sequential_queue_stops_on_first_failure(self):
        jobs = queue.job_plan()[:3]
        seen = []
        with patch.object(queue.training, "validate_job_complete", return_value=False), contextlib.redirect_stdout(io.StringIO()):
            def fail(job):
                seen.append(training.job_name(job))
                raise RuntimeError("synthetic job failure")
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                queue.run_jobs(jobs, Path("unused"), {"identity": "x"}, fail)
        self.assertEqual(seen, ["gru_cv_fold_1"])

    def test_papermill_attempts_are_fresh_and_interrupt_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            experiment = Path(scratch)
            (experiment / "experiment.json").write_text(json.dumps({"identity": "identity", "class_order": CLASSES}), encoding="utf-8")
            job = {"model": "gru", "kind": "cv", "fold": 1}
            with patch.object(queue.training, "validate_job_complete", return_value=True), patch.object(queue.pm, "execute_notebook") as execute:
                def notebook_output(_source, output, **kwargs):
                    kwargs["stdout_file"].write("notebook stdout\n")
                    kwargs["stderr_file"].write("notebook stderr\n")
                    kwargs["stdout_file"].flush()
                    kwargs["stderr_file"].flush()
                    Path(output).write_text("mock notebook", encoding="utf-8")
                execute.side_effect = notebook_output
                queue._execute_job(experiment, ROOT / "training_config.json", "identity", job)
                self.assertEqual(execute.call_args.kwargs["kernel_name"], "python3")
                self.assertEqual(execute.call_args.kwargs["parameters"]["RUN_TRAINING"], True)
                self.assertTrue(execute.call_args.kwargs["stdout_file"].closed)
                self.assertEqual((experiment / "attempts/gru_cv_fold_1/attempt-0001.stdout.log").read_text(), "notebook stdout\n")
            with patch.object(queue.pm, "execute_notebook", side_effect=KeyboardInterrupt("synthetic interrupt")):
                with self.assertRaises(KeyboardInterrupt):
                    queue._execute_job(experiment, ROOT / "training_config.json", "identity", job)
            error = experiment / "attempts" / "gru_cv_fold_1" / "attempt-0002.error.txt"
            self.assertIn("KeyboardInterrupt", error.read_text(encoding="utf-8"))

    def test_completed_job_marker_hashes_required_artifacts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            experiment_dir = Path(scratch)
            experiment = {"identity": "experiment"}
            job = {"model": "gru", "kind": "cv", "fold": 1}
            identity = training._job_identity(experiment, job)
            job_dir = experiment_dir / "jobs" / training.job_name(job)
            job_dir.mkdir(parents=True)
            for name in ("last.pt", "best.pt", "history.json", "metrics.json", "predictions.csv"):
                (job_dir / name).write_text(name, encoding="utf-8")
            training._write_marker(job_dir / "training_complete.json", identity, ("last.pt", "history.json", "best.pt"))
            training._write_marker(job_dir / "evaluation_complete.json", identity, ("metrics.json", "predictions.csv"))
            self.assertTrue(training.validate_job_complete(experiment_dir, experiment, job))
            (job_dir / "predictions.csv").write_text("tampered", encoding="utf-8")
            self.assertFalse(training.validate_job_complete(experiment_dir, experiment, job))

    def test_resume_rejects_changed_experiment_fingerprint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            path = Path(scratch) / "experiment.json"
            path.write_text(json.dumps({"identity": "old"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                queue._check_experiment(path, {"identity": "new"})

    def test_inspection_reads_saved_probabilities_for_all_models_and_ties(self):
        classes = ["No gesture", "Doing other things", "Thumb Up"]
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            experiment = Path(scratch)
            identity = "run-a"
            (experiment / "experiment.json").write_text(json.dumps({"identity": identity, "class_order": classes}), encoding="utf-8")
            pred = {"true": np.asarray([2]), "predicted": np.asarray([0]), "probabilities": np.asarray([[0.5, 0.5, 0.0]])}
            for model in ("gru", "lstm", "ddnet"):
                job_dir = experiment / "jobs" / f"{model}_final"
                training._write_predictions(job_dir / "predictions.csv", ["55"], np.asarray(["Thumb Up"], object), pred, classes, model, identity)
            result = inspect_probabilities.inspect(experiment, "55")
            self.assertEqual([item["model"] for item in result], ["GRU", "LSTM", "DDNET"])
            self.assertEqual(result[0]["predicted"], "No gesture")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = inspect_probabilities.main(["--experiment", str(experiment), "--clip-id", "55"])
            self.assertEqual(code, 0)
            text = stdout.getvalue()
            self.assertIn("Thumb Up: 0.000000%", text)
            self.assertIn("No gesture: 50.000000%", text)
            self.assertEqual(text.count("No gesture: 50.000000%"), 3)

    def test_export_validation_rejects_missing_rows_and_probability_columns(self):
        classes = ["a", "b"]
        prediction = {"predicted": np.asarray([0]), "probabilities": np.asarray([[0.75, 0.25]])}
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            path = Path(scratch) / "predictions.csv"
            training._write_predictions(path, ["1"], np.asarray(["a"], object), prediction, classes, "gru", "id")
            training._validate_predictions(path, ["1"], np.asarray(["a"], object), classes, "gru", "id")
            path.write_text("clip_id,true_class,predicted_class,correct,model_id,config_id,prob__a\n1,a,a,true,gru,id,1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns"):
                training._validate_predictions(path, ["1"], np.asarray(["a"], object), classes, "gru", "id")
            with self.assertRaisesRegex(ValueError, "columns"):
                reports._read_prediction(path, classes)

    def test_training_disabled_papermill_smoke(self):
        import papermill
        import zmq

        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            output = Path(scratch) / "disabled.ipynb"
            context = None
            socket = None
            try:
                context = zmq.Context()
                socket = context.socket(zmq.REP)
                socket.bind_to_random_port("tcp://127.0.0.1")
            except zmq.ZMQError as exc:
                self.skipTest(f"local ZeroMQ sockets are unavailable: {exc}")
            finally:
                if socket is not None:
                    socket.close(0)
                if context is not None:
                    context.term()
            with queue.temporary_python_kernel():
                notebook = papermill.execute_notebook(str(ROOT / "train_classifier.ipynb"), str(output), kernel_name="python3", cwd=str(ROOT), progress_bar=False)
            rendered = "\n".join(str(item.get("text", "")) for cell in notebook.cells for item in cell.get("outputs", []))
            self.assertIn("RUN_TRAINING=False; no training started.", rendered)

    def test_temporary_kernel_uses_current_interpreter_and_restores_environment(self):
        import os
        import sys
        from jupyter_client.kernelspec import KernelSpecManager

        old_path, old_runtime = os.environ.get("JUPYTER_PATH"), os.environ.get("JUPYTER_RUNTIME_DIR")
        with queue.temporary_python_kernel():
            spec = KernelSpecManager().get_kernel_spec("python3")
            self.assertEqual(spec.argv[0], sys.executable)
            self.assertTrue(Path(os.environ["JUPYTER_RUNTIME_DIR"]).is_dir())
        self.assertEqual(os.environ.get("JUPYTER_PATH"), old_path)
        self.assertEqual(os.environ.get("JUPYTER_RUNTIME_DIR"), old_runtime)


if __name__ == "__main__":
    unittest.main()
