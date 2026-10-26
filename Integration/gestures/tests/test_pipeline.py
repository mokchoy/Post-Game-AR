import csv
import io
import json
import tempfile
import tarfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import jester_pipeline as jp
import landmarks as lm


CLASSES = ["No gesture", "Doing other things"] + [f"gesture {i:02d}" for i in range(25)]


def write_labels(root: Path, per_class: int = 6) -> Path:
    root.mkdir(parents=True)
    (root / "labels.csv").write_text("".join(f"{name}\n" for name in CLASSES), encoding="utf-8")
    train, validation, test = [], [], []
    clip = 1
    for label in CLASSES:
        for _ in range(per_class):
            train.append(f"{clip};{label}\n")
            clip += 1
        validation.append(f"{clip};{label}\n")
        clip += 1
        test.append(f"{clip}\n")
        clip += 1
    (root / "train.csv").write_text("".join(train), encoding="utf-8")
    (root / "validation.csv").write_text("".join(validation), encoding="utf-8")
    (root / "test.csv").write_text("".join(test), encoding="utf-8")
    (root / "test-answers.csv").write_text("this must not be read\n", encoding="utf-8")
    return root


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)
        self.labels = write_labels(self.root / "labels")

    def tearDown(self):
        self.temp.cleanup()

    def subset_review(self):
        config = {"_cache_root": self.root / "cache", "cache_limit_gb": 50}
        path = jp.cache_paths(config)["metadata"] / "class_review.csv"
        jp.create_review_template(self.labels, path)
        rows = read_csv(path)
        for row in rows:
            row["compatibility_decision"] = "retain" if row["class_label"] in CLASSES[:11] else "exclude"
            row["reason"] = "synthetic selection"
        jp._write_csv(path, tuple(rows[0]), rows)
        return config, path

    def test_official_splits_and_unlabeled_test_are_preserved(self):
        summary = jp.validate_labels(self.labels)
        self.assertEqual(summary["classes"], 27)
        self.assertFalse(summary["test_answers_read"])
        _, records = jp.read_labels(self.labels)
        self.assertEqual({r["official_split"] for r in records.values()}, {"train", "validation", "test"})
        self.assertTrue(all(not r["class_label"] for r in records.values() if r["official_split"] == "test"))
        self.assertEqual(len(records), len(set(records)))

    def test_folds_are_reproducible_balanced_and_disjoint(self):
        first = self.root / "folds.csv"
        views = self.root / "views"
        second = self.root / "folds-copy.csv"
        jp.generate_folds(self.labels, first, views, seed=42)
        jp.generate_folds(self.labels, second, self.root / "views-copy", seed=42)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        rows = read_csv(first)
        self.assertEqual(len(rows), 27 * 6)
        for label in CLASSES:
            counts = [sum(row["class_label"] == label and row["fold"] == str(k) for row in rows) for k in (1, 2, 3)]
            self.assertEqual(counts, [2, 2, 2])
        all_train = {row["clip_id"] for row in rows}
        for fold in (1, 2, 3):
            train = {row["clip_id"] for row in read_csv(views / f"fold_{fold}_train.csv")}
            dev = {row["clip_id"] for row in read_csv(views / f"fold_{fold}_development.csv")}
            self.assertFalse(train & dev)
            self.assertEqual(train | dev, all_train)
            self.assertEqual(dev, {row["clip_id"] for row in rows if row["fold"] == str(fold)})

    def test_pilot_selection_is_seeded_balanced_and_never_uses_other_splits(self):
        first = self.root / "pilot.csv"
        second = self.root / "pilot-copy.csv"
        one = jp.select_pilot(self.labels, first, per_class=2, seed=42)
        jp.select_pilot(self.labels, second, per_class=2, seed=42)
        self.assertEqual(one["selected_clips"], 54)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        rows = read_csv(first)
        self.assertTrue(all(row["official_split"] == "train" for row in rows))
        self.assertEqual(len({row["clip_id"] for row in rows}), len(rows))
        self.assertTrue(all(sum(row["class_label"] == label for row in rows) == 2 for label in CLASSES))
        with self.assertRaises(RuntimeError):
            jp.select_pilot(self.labels, self.root / "short.csv", per_class=7, seed=42)
        self.assertFalse((self.root / "short.csv").exists())

    def test_pilot_extraction_uses_declared_sample_count(self):
        labels = write_labels(self.root / "large-labels", per_class=500)
        config = {"_cache_root": self.root / "cache", "cache_limit_gb": 50}
        for count in (200, 500):
            path = self.root / f"pilot-{count}.csv"
            jp.select_pilot(labels, path, per_class=count)
            # Stop after selection validation, before opening any real archives.
            with patch.object(jp, "_archive_stage_identity", side_effect=RuntimeError("selection accepted")):
                with self.assertRaisesRegex(RuntimeError, "selection accepted"):
                    jp.extract_frames(config, labels, path, False)
        self.assertEqual(jp._parser().parse_args(["pilot"]).per_class, 500)
        rows = read_csv(path)
        cases = {
            "incomplete": rows[:-1],
            "duplicate": rows + [rows[0]],
            "mixed counts": [rows[0] | {"requested_per_class": "200"}] + rows[1:],
            "missing count": [row | {"requested_per_class": ""} for row in rows],
            "wrong split": [rows[0] | {"official_split": "validation"}] + rows[1:],
        }
        for name, invalid in cases.items():
            with self.subTest(name=name):
                jp._write_csv(path, tuple(rows[0]), invalid)
                with patch.object(jp, "_archive_stage_identity") as archive:
                    with self.assertRaises(ValueError):
                        jp.extract_frames(config, labels, path, False)
                    archive.assert_not_called()

    def test_subset_pilot_is_balanced_deterministic_and_ignores_excluded_shortfalls(self):
        _, review = self.subset_review()
        _, records = jp.read_labels(self.labels)
        kept_excluded = set()
        train = []
        for cid, row in records.items():
            label = row["class_label"]
            if row["official_split"] == "train" and (label in CLASSES[:11] or label not in kept_excluded):
                train.append(f"{cid};{label}\n")
                kept_excluded.add(label)
        (self.labels / "train.csv").write_text("".join(train), encoding="utf-8")
        first, second = self.root / "subset.csv", self.root / "subset-copy.csv"
        result = jp.select_pilot(self.labels, first, per_class=2, review_path=review)
        jp.select_pilot(self.labels, second, per_class=2, review_path=review)
        self.assertEqual(result["classes"], 11)
        self.assertEqual(result["selected_clips"], 22)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        rows = read_csv(first)
        self.assertEqual({row["class_label"] for row in rows}, set(CLASSES[:11]))
        self.assertTrue(all(sum(row["class_label"] == label for row in rows) == 2 for label in CLASSES[:11]))
        self.assertTrue(all(row["official_split"] == "train" for row in rows))
        with self.assertRaises(RuntimeError):
            jp.select_pilot(self.labels, self.root / "short-subset.csv", per_class=7, review_path=review)
        self.assertFalse((self.root / "short-subset.csv").exists())

    def test_subset_pilot_extraction_checks_active_counts_before_archive_access(self):
        config, review = self.subset_review()
        path = self.root / "subset.csv"
        jp.select_pilot(self.labels, path, per_class=2, review_path=review)
        rows = read_csv(path)
        expected = {row["clip_id"] for row in rows}
        # Both old all-class pilots and new subset pilots use the same active clips.
        legacy = self.root / "legacy.csv"
        jp.select_pilot(self.labels, legacy, per_class=2)
        for selection in (path, legacy):
            with (
                patch.object(jp, "_archive_stage_identity", return_value=([], {}, "fixture")),
                patch.object(jp, "_archive_stats", side_effect=RuntimeError("selection accepted")) as stats,
            ):
                with self.assertRaisesRegex(RuntimeError, "selection accepted"):
                    jp.extract_frames(config, self.labels, selection, False)
                self.assertEqual(stats.call_args.args[1], expected)
        for invalid in (rows[:-1], rows + [rows[0]], [rows[0] | {"requested_per_class": "3"}] + rows[1:]):
            jp._write_csv(path, tuple(rows[0]), invalid)
            with patch.object(jp, "_archive_stage_identity") as archive:
                with self.assertRaises(ValueError):
                    jp.extract_frames(config, self.labels, path, False)
                archive.assert_not_called()

    def test_subset_filters_explicit_frame_and_landmark_ids_and_rejects_unlabeled_or_empty(self):
        config, review_path = self.subset_review()
        classes, records = jp.read_labels(self.labels)
        review = read_csv(review_path)
        # Preparation must still allow pending and flagged classes during review.
        review[2]["compatibility_decision"] = "pending"
        review[3]["compatibility_decision"] = "flag"
        jp._write_csv(review_path, tuple(review[0]), review)
        rows = [row for row in records.values() if row["class_label"]]
        path = self.root / "ids.csv"
        jp._write_csv(path, ("clip_id", "class_label", "official_split"), rows)
        expected = [row["clip_id"] for row in rows if row["class_label"] in classes[:11]]
        self.assertEqual(lm._selected_ids(path, config, records), expected)
        with (
            patch.object(jp, "_archive_stage_identity", return_value=([], {}, "fixture")),
            patch.object(jp, "_archive_stats", side_effect=RuntimeError("selection accepted")) as stats,
        ):
            with self.assertRaisesRegex(RuntimeError, "selection accepted"):
                jp.extract_frames(config, self.labels, None, False, path)
            self.assertEqual(stats.call_args.args[1], set(expected))
        for invalid in (
            [row for row in rows if row["class_label"] not in classes[:11]],
            [{"clip_id": next(cid for cid, row in records.items() if row["official_split"] == "test")}],
            [rows[0] | {"class_label": "wrong label"}],
            rows + [rows[0]],
        ):
            jp._write_csv(path, ("clip_id", "class_label", "official_split"), invalid)
            with patch.object(jp, "_archive_stage_identity") as archive:
                with self.assertRaises(ValueError):
                    jp.extract_frames(config, self.labels, None, False, path)
                archive.assert_not_called()
            with self.assertRaises(ValueError):
                lm._selected_ids(path, config, records)
        with patch.object(jp, "_archive_stage_identity") as archive:
            with self.assertRaisesRegex(ValueError, "--all"):
                jp.extract_frames(config, self.labels, None, True)
            archive.assert_not_called()

    def test_frame_promotion_retries_windows_locks_and_preserves_destination(self):
        partial, final = self.root / "partial", self.root / "final"
        partial.mkdir()
        locked = PermissionError("locked")
        locked.winerror = 5
        rename = jp.os.rename
        with patch.object(jp.os, "rename", side_effect=[locked, locked, None]) as move, patch.object(jp.time, "sleep") as sleep:
            jp._promote_frame_clip(partial, final)
            self.assertEqual(move.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
        with patch.object(jp.os, "rename", side_effect=locked) as move, patch.object(jp.time, "sleep"):
            with self.assertRaises(PermissionError):
                jp._promote_frame_clip(partial, final)
            self.assertEqual(move.call_count, 4)
        self.assertTrue(partial.is_dir())
        final.mkdir()
        with patch.object(jp.os, "rename", wraps=rename) as move:
            with self.assertRaises(FileExistsError):
                jp._promote_frame_clip(partial, final)
            move.assert_not_called()
        self.assertTrue(partial.is_dir())
        self.assertTrue(final.is_dir())

    def test_frame_resume_recovers_staging_reports_locks_and_skips_archive_extraction(self):
        config = {"_cache_root": self.root / "cache", "cache_limit_gb": 1}
        jp.claim_cache(config["_cache_root"])
        frames = jp.cache_paths(config)["frames"]
        staging = jp._staging_root(frames)
        ids = self.root / "ids.csv"
        jp._write_csv(ids, ("clip_id",), [{"clip_id": cid} for cid in ("1", "2", "3")])
        ok, encoded = cv2.imencode(".jpg", np.zeros((100, 176, 3), np.uint8))
        self.assertTrue(ok)
        for cid in ("1", "2", "3"):
            clip = (frames if cid == "3" else staging) / cid
            clip.mkdir()
            (clip / "00001.jpg").write_bytes(encoded.tobytes())
            scan = jp.scan_frame_dir(clip)
            jp.write_json(clip / "complete.json", {"status": "complete", "archive_identity": "archive-a", "frame_count": 1, "frame_set_sha256": jp.frame_set_hash(scan["entries"])})
        stats = {cid: {"frame_count": 1, "max_index": 1, "indices": [1], "bytes": len(encoded)} for cid in ("1", "2", "3")}
        promote = jp._promote_frame_clip

        def locked_clip(partial, final):
            if partial.name == "1":
                raise PermissionError("synthetic lock")
            promote(partial, final)

        with patch.object(jp, "_archive_stage_identity", return_value=([], {}, "archive-a")), patch.object(jp, "_archive_stats", return_value=stats), patch.object(jp, "_open_tar", side_effect=AssertionError("must not re-extract valid staging")), patch.object(jp, "ensure_capacity") as capacity:
            with patch.object(jp, "_promote_frame_clip", side_effect=locked_clip):
                with self.assertRaisesRegex(RuntimeError, "1 frame extraction failure"):
                    jp.extract_frames(config, self.labels, None, False, ids)
            self.assertTrue((staging / "1" / "complete.json").is_file())
            self.assertTrue((frames / "2" / "complete.json").is_file())
            failures = read_csv(jp.cache_paths(config)["metadata"] / "frame_extraction_failures.csv")
            self.assertEqual([row["clip_id"] for row in failures], ["1"])
            self.assertIn("staged clip preserved", failures[0]["reason"])
            result = jp.extract_frames(config, self.labels, None, False, ids)
            self.assertEqual(result["recovered_staged"], 1)
            self.assertEqual(result["already_complete"], 2)
            self.assertEqual(result["newly_extracted"], 0)
            self.assertEqual(result["frame_payload_bytes"], 0)
            self.assertEqual(capacity.call_args.args[2], 4096)
            self.assertTrue(jp.validate_frame_completion(frames / "1", "archive-a", 1)[0])

    def test_frame_reextraction_replaces_incompatible_staging_and_preserves_failed_promotion(self):
        config = {"_cache_root": self.root / "cache", "cache_limit_gb": 1}
        jp.claim_cache(config["_cache_root"])
        frames = jp.cache_paths(config)["frames"]
        partial = jp._staging_root(frames) / "1"
        partial.mkdir()
        ok, encoded = cv2.imencode(".jpg", np.zeros((100, 176, 3), np.uint8))
        self.assertTrue(ok)
        payload = encoded.tobytes()
        (partial / "00001.jpg").write_bytes(payload)
        jp.write_json(partial / "complete.json", {"status": "complete", "archive_identity": "old", "frame_count": 1, "frame_set_sha256": jp.frame_set_hash(jp.scan_frame_dir(partial)["entries"])})
        ids = self.root / "ids.csv"
        jp._write_csv(ids, ("clip_id",), [{"clip_id": "1"}])
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            member = tarfile.TarInfo("1/00001.jpg")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        buffer.seek(0)
        stats = {"1": {"frame_count": 1, "max_index": 1, "indices": [1], "bytes": len(payload)}}
        with patch.object(jp, "_archive_stage_identity", return_value=([], {}, "new")), patch.object(jp, "_archive_stats", return_value=stats):
            with patch.object(jp, "_open_tar", return_value=(buffer, tarfile.open(fileobj=buffer, mode="r|"))), patch.object(jp, "_promote_frame_clip", side_effect=PermissionError("locked")):
                with self.assertRaisesRegex(RuntimeError, "1 frame extraction failure"):
                    jp.extract_frames(config, self.labels, None, False, ids)
            self.assertTrue(jp.validate_frame_completion(partial, "new", 1)[0])
            self.assertFalse((frames / "1").exists())
            with patch.object(jp, "_open_tar", side_effect=AssertionError("must reuse newly validated staging")):
                result = jp.extract_frames(config, self.labels, None, False, ids)
            self.assertEqual(result["recovered_staged"], 1)

    def test_incompatible_generated_output_is_preserved(self):
        output = self.root / "stable.csv"
        jp._write_csv(output, ("id",), [{"id": "old"}], preserve_existing=True)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            jp._write_csv(output, ("id",), [{"id": "new"}], preserve_existing=True)
        self.assertEqual(output.read_bytes(), original)

    def test_cache_limit_stops_without_deleting_existing_output(self):
        cache = self.root / "cache"
        cache.mkdir()
        completed = cache / "result.bin"
        completed.write_bytes(b"completed")
        with self.assertRaises(RuntimeError):
            jp.ensure_capacity(cache, 1, 1024**3 + 1, free_bytes=2 * 1024**3)
        self.assertEqual(completed.read_bytes(), b"completed")
        with self.assertRaises(RuntimeError):
            jp.ensure_capacity(cache, 1, 100, free_bytes=10)
        self.assertEqual(completed.read_bytes(), b"completed")

    def test_incremental_capacity_keeps_limit_and_free_space_guards(self):
        with patch.object(jp, "_cache_bytes", side_effect=AssertionError("cache rescanned")):
            jp.ensure_capacity(self.root, 1, 100, used_bytes=200, free_bytes=1000)
            with self.assertRaisesRegex(RuntimeError, "managed cache limit"):
                jp.ensure_capacity(self.root, 1, 100, used_bytes=1024**3, free_bytes=1000)
            with self.assertRaisesRegex(RuntimeError, "insufficient free disk"):
                jp.ensure_capacity(self.root, 1, 100, used_bytes=0, free_bytes=10)

    def test_multipart_tar_reader_streams_numeric_members(self):
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode="w:gz") as archive:
            for name, data in (("1/00002.jpg", b"two"), ("1/00010.jpg", b"ten")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        payload = content.getvalue()
        split = len(payload) // 2
        part_a, part_b = self.root / "part-a", self.root / "part-b"
        part_a.write_bytes(payload[:split])
        part_b.write_bytes(payload[split:])
        reader, archive = jp._open_tar([part_a, part_b])
        try:
            rows = [(jp._tar_frame(member), archive.extractfile(member).read()) for member in archive]
        finally:
            archive.close()
            reader.close()
        self.assertEqual([row[0][1] for row in rows], [2, 10])
        self.assertEqual([row[1] for row in rows], [b"two", b"ten"])

    def test_numeric_frame_scan_records_gaps_and_corruption(self):
        clip = self.root / "clip"
        clip.mkdir()
        image = np.zeros((100, 176, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        (clip / "1.jpg").write_bytes(encoded.tobytes())
        (clip / "2.jpg").write_bytes(b"not a jpeg")
        (clip / "10.jpg").write_bytes(encoded.tobytes())
        scan = jp.scan_frame_dir(clip)
        self.assertEqual([row["index"] for row in scan["entries"]], list(range(1, 11)))
        self.assertEqual(scan["corrupt"], [2])
        self.assertEqual(scan["missing"], list(range(3, 10)))

    def test_landmark_cache_keeps_every_frame_and_explicit_no_detection(self):
        clip = self.root / "frames" / "7"
        clip.mkdir(parents=True)
        image = np.zeros((100, 176, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        (clip / "00001.jpg").write_bytes(encoded.tobytes())
        (clip / "00002.jpg").write_bytes(encoded.tobytes())
        scan = jp.scan_frame_dir(clip)
        jp.write_json(clip / "complete.json", {"status": "complete", "frame_count": 2, "frame_set_sha256": jp.frame_set_hash(scan["entries"])})

        class FakeAdapter:
            def detect(self, image, timestamp):
                if timestamp == 0:
                    return lm._empty_result()
                points = np.zeros((1, 21, 2), dtype=np.float32)
                points[0, 0] = (0.25, 0.75)
                return {"xy": points, "joint_scores": np.ones((1, 21), np.float32), "handedness": [""], "handedness_scores": [np.nan], "detection_scores": [np.nan], "visibility": lm._nan_joints(1), "presence": lm._nan_joints(1)}

            def close(self):
                pass

        with patch.object(lm, "_adapter", return_value=FakeAdapter()):
            arrays, _ = lm._extract_clip("7", clip, "rtmlib", {"detector": Path("det"), "pose": Path("pose")}, {"extractor": "rtmlib", "joints": list(lm.JOINTS), "frame_rate_fps": 12})
        self.assertEqual(arrays["frame_indices"].tolist(), [1, 2])
        self.assertEqual(arrays["frame_timestamps_seconds"].tolist(), [0.0, 1 / 12])
        self.assertEqual(arrays["detected_hand_count"].tolist(), [0, 1])
        self.assertEqual(arrays["frame_hand_offsets"].tolist(), [0, 0, 1])
        self.assertEqual(arrays["hand_xy"].shape, (1, 21, 2))
        with tempfile.TemporaryDirectory(dir=self.root) as scratch:
            cache_path = Path(scratch) / "landmarks.npz"
            lm._atomic_npz(cache_path, arrays)
            with np.load(cache_path, allow_pickle=False) as cache:
                self.assertTrue(lm._valid_cache(cache))
        self.assertEqual(tuple(lm.JOINTS[:6]), ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp"))

    def test_pixel_coordinates_normalize_without_centering_or_mirroring(self):
        coords = np.asarray([[[200.0, 25.0], [400.0, 50.0]]], dtype=np.float32)
        normalized = lm.normalize_pixel_xy(coords, 400, 100)
        np.testing.assert_allclose(normalized, [[[0.5, 0.25], [1.0, 0.5]]])
        np.testing.assert_array_equal(coords, [[[200.0, 25.0], [400.0, 50.0]]])

    def test_rtmlib_components_use_supported_defaults_and_preserve_no_detection(self):
        detector = Mock(side_effect=[[], [[0, 0, 10, 10, 0.9]]])
        pose = Mock()
        points = np.zeros((1, 21, 2), dtype=np.float32)
        points[0, 0] = (100, 50)
        pose.return_value = (points, np.ones((1, 21), dtype=np.float32))
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch("rtmlib.RTMDet", return_value=detector) as detector_api, patch("rtmlib.RTMPose", return_value=pose) as pose_api:
            adapter = lm.RTMLibHand(Path("detector.onnx"), Path("pose.onnx"))
            empty = adapter.detect(image, 0)
            detected = adapter.detect(image, 83)

        detector_api.assert_called_once_with("detector.onnx", model_input_size=(320, 320), det_mode=None, score_thr=0.3, nms_thr=0.45, backend="onnxruntime", device="cpu")
        pose_api.assert_called_once_with("pose.onnx", model_input_size=(256, 256), to_openpose=False, backend="onnxruntime", device="cpu")
        pose.assert_called_once_with(image, bboxes=[[0, 0, 10, 10, 0.9]])
        self.assertEqual(empty["xy"].shape, (0, 21, 2))
        np.testing.assert_allclose(detected["xy"][0, 0], [0.5, 0.5])
        self.assertEqual(detected["joint_scores"].shape, (1, 21))

    def test_rtmlib_cuda_requires_provider_and_rejects_cpu_fallback(self):
        detector, pose = Mock(), Mock()
        detector.session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        pose.session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider unavailable"):
                lm.RTMLibHand(Path("detector.onnx"), Path("pose.onnx"), "cuda")
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider"]), patch("rtmlib.RTMDet", return_value=detector) as det_api, patch("rtmlib.RTMPose", return_value=pose) as pose_api:
            lm.RTMLibHand(Path("detector.onnx"), Path("pose.onnx"), "cuda")
            self.assertEqual(det_api.call_args.kwargs["device"], "cuda")
            self.assertEqual(pose_api.call_args.kwargs["device"], "cuda")
            pose.session.get_providers.return_value = ["CPUExecutionProvider"]
            with self.assertRaisesRegex(RuntimeError, "refusing silent CPU fallback"):
                lm.RTMLibHand(Path("detector.onnx"), Path("pose.onnx"), "cuda")

    def test_gpu_extraction_identity_differs_from_cpu_and_mediapipe_rejects_cuda(self):
        with patch.object(lm, "_model_info", return_value={"sha256": "same"}):
            cpu = lm._package_info("rtmlib", {"detector": Path("det"), "pose": Path("pose")})
            gpu = lm._package_info("rtmlib", {"detector": Path("det"), "pose": Path("pose")}, "cuda")
        self.assertEqual(cpu["runtime_backend"], "onnxruntime-cpu")
        self.assertEqual(gpu["runtime_backend"], "onnxruntime-cuda")
        self.assertEqual(gpu["settings"]["device"], "cuda")
        self.assertEqual(lm.SETTINGS["rtmlib"]["device"], "cpu")
        self.assertNotEqual(cpu, gpu)
        with self.assertRaises(ValueError):
            lm._package_info("mediapipe", {}, "cuda")
        parsed = jp._parser().parse_args(["extract-landmarks", "--extractor", "rtmlib", "--device", "cuda"])
        self.assertEqual(parsed.device, "cuda")
        with self.assertRaises(SystemExit):
            jp._parser().parse_args(["benchmark", "--device", "cuda", "--model", "task", "--rtm-det-model", "det", "--rtm-pose-model", "pose"])

    def test_mediapipe_optional_metadata_can_be_none(self):
        point = SimpleNamespace(x=0.25, y=0.75, visibility=None, presence=None)
        category = SimpleNamespace(category_name="Left", score=None)
        fake = lm.MediaPipeHand.__new__(lm.MediaPipeHand)
        fake.mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=1), Image=lambda **kwargs: kwargs)
        fake.task = SimpleNamespace(detect_for_video=lambda image, timestamp: SimpleNamespace(hand_landmarks=[[point] * 21], handedness=[[category]]))

        result = fake.detect(np.zeros((100, 200, 3), dtype=np.uint8), 0)

        self.assertEqual(result["xy"].shape, (1, 21, 2))
        np.testing.assert_allclose(result["xy"][0, 0], [0.25, 0.75])
        self.assertEqual(result["handedness"], ["Left"])
        self.assertTrue(np.isnan(result["handedness_scores"][0]))
        self.assertTrue(np.isnan(result["visibility"]).all())
        self.assertTrue(np.isnan(result["presence"]).all())

    def test_cache_markers_reject_partial_and_invalidate_on_identity_change(self):
        path = self.root / "7.npz"
        marker = self.root / "7.complete.json"
        metadata = {"input_identity": "identity-a", "source_frame_set_sha256": "frame-a", "package_key": "model-a", "joints": list(lm.JOINTS), "frame_rate_fps": 12}
        arrays = {
            "frame_indices": np.asarray([1]), "frame_timestamps_seconds": np.asarray([0.0]),
            "source_frame_names": np.asarray(["00001.jpg"]), "frame_valid": np.asarray([True]),
            "image_widths": np.asarray([176]), "image_heights": np.asarray([100]),
            "frame_hand_offsets": np.asarray([0, 0]), "detected_hand_count": np.asarray([0]),
            "hand_xy": np.empty((0, 21, 2)), "joint_mask": np.empty((0, 21), bool),
            "joint_scores": np.empty((0, 21)), "detection_scores": np.empty((0,)),
            "handedness": np.empty((0,), dtype="U16"), "handedness_scores": np.empty((0,)),
            "visibility": np.empty((0, 21)), "presence": np.empty((0, 21)),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
        lm._atomic_npz(path, arrays)
        self.assertFalse(lm._read_marker(path, marker, "identity-a"))
        self.assertTrue(lm._recover_unmarked(path, marker, "identity-a", "frame-a", "model-a"))
        self.assertTrue(lm._read_marker(path, marker, "identity-a"))
        self.assertFalse(lm._read_marker(path, marker, "identity-b"))

    def test_manifest_does_not_assign_a_class_to_unlabeled_test_ids(self):
        output = self.root / "manifest.csv"
        frames = self.root / "cache" / "frames"
        jp.build_manifest(self.labels, frames, output, None, None, None)
        rows = read_csv(output)
        test_rows = [row for row in rows if row["official_split"] == "test"]
        self.assertEqual(len(test_rows), 27)
        self.assertTrue(all(row["class_label"] == "" and row["compatibility_decision"] == "not_applicable" for row in test_rows))

    def test_selected_manifest_refresh_scans_only_selected_clips_and_preserves_other_statuses(self):
        output = self.root / "manifest.csv"
        frames = self.root / "cache" / "frames"
        jp.build_manifest(self.labels, frames, output, None, None, None)
        before = read_csv(output)
        before[1]["frame_count"] = "17"
        before[1]["extraction_status"] = "partial"
        jp._write_csv(output, tuple(before[0]), before)
        (frames / "1").mkdir(parents=True)
        (frames / "2").mkdir()
        ids = self.root / "selected.csv"
        ids.write_text("clip_id\n1\n", encoding="utf-8")
        scan = {"valid_frame_count": 3, "integrity_status": "ok"}
        with patch.object(jp, "validate_frame_completion", return_value=(True, {"frame_count": 3}, scan)) as validate:
            jp.build_manifest(self.labels, frames, output, None, None, None, ids)
        validate.assert_called_once_with(frames / "1")
        after = read_csv(output)
        self.assertEqual(after[0]["frame_count"], "3")
        self.assertEqual(after[0]["extraction_status"], "complete")
        self.assertEqual(after[1:], before[1:])

    def test_bundle_contains_completed_frames_and_preserves_local_outputs(self):
        cache_root = self.root / "cache"
        jp.claim_cache(cache_root)
        clip = cache_root / "frames" / "1"
        clip.mkdir(parents=True)
        image = np.zeros((100, 176, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        frame = clip / "00001.jpg"
        frame.write_bytes(encoded.tobytes())
        scan = jp.scan_frame_dir(clip)
        jp.write_json(clip / "complete.json", {"status": "complete", "frame_count": 1, "frame_set_sha256": jp.frame_set_hash(scan["entries"])})
        config = {"_cache_root": cache_root, "_cloud_root": self.root / "drive", "cache_limit_gb": 1}
        result = jp._copy_completed_bundle(config, "all", False)
        bundle = Path(result["bundle"])
        with zipfile.ZipFile(bundle) as archive:
            self.assertIn("frames/1/00001.jpg", archive.namelist())
            self.assertIn("frames/1/complete.json", archive.namelist())
        self.assertTrue(frame.is_file())
        self.assertTrue(bundle.is_file())

    def test_extractor_comparison_rejects_different_clip_sets(self):
        cache_root = self.root / "cache"
        config = {"_cache_root": cache_root, "_cloud_root": self.root / "drive", "cache_limit_gb": 1}
        jp.claim_cache(cache_root)
        reports = cache_root / "reports"
        for name, digest in (("rtmlib", "a"), ("mediapipe", "b")):
            folder = reports / name
            folder.mkdir(parents=True)
            jp.write_json(folder / "summary.json", {"clips_reviewed": 1, "complete_clips": 1, "failed_clips": 0, "clip_ids_sha256": digest})
        with self.assertRaises(ValueError):
            lm.compare(config)
        jp.write_json(reports / "mediapipe" / "summary.json", {"clips_reviewed": 1, "complete_clips": 1, "failed_clips": 0, "clip_ids_sha256": "a"})
        result = lm.compare(config)
        self.assertTrue(result["reports_available"]["rtmlib"])
        self.assertTrue((reports / "extractor_comparison.md").is_file())


if __name__ == "__main__":
    unittest.main()
