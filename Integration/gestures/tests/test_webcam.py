import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import classifiers
import training_data
import webcam


def pose(wrist: tuple[float, float], size: float) -> np.ndarray:
    base = np.stack((np.linspace(0, 1, 21), np.tile(np.linspace(0, 1, 7), 3)), axis=1)
    return np.asarray(wrist, dtype=np.float32) + base.astype(np.float32) * np.float32(size)


def job_identity(experiment_identity: str, model_name: str) -> str:
    payload = json.dumps(
        {"experiment_identity": experiment_identity, "job": {"kind": "final", "model": model_name}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class WebcamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_preprocessing_matches_training_helpers_and_keeps_sequence_length(self):
        first = pose((0.1, 0.1), 0.1)
        larger = pose((0.7, 0.1), 0.3)
        frames = [
            np.stack((first, larger)),
            np.empty((0, 21, 2), dtype=np.float32),
            np.stack((pose((0.12, 0.1), 0.1), pose((0.72, 0.1), 0.3))),
            np.stack((pose((0.14, 0.1), 0.1), pose((0.74, 0.1), 0.3))),
        ]
        features, poses, valid = webcam.preprocess_candidates(frames, 8)
        all_xy = np.concatenate(frames)
        counts = np.asarray([len(frame) for frame in frames], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        expected_poses, expected_valid, _, _ = training_data._select_candidates(
            all_xy, np.isfinite(all_xy).all(axis=2), offsets
        )
        indices = training_data._sample_indices(len(expected_poses), 8)
        expected = np.zeros((8, 43), dtype=np.float32)
        expected[:, :42] = expected_poses[indices].reshape(8, 42)
        expected[:, 42] = expected_valid[indices].astype(np.float32)
        self.assertEqual(features.shape, (8, 43))
        np.testing.assert_array_equal(poses, expected_poses)
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(features, expected)
        np.testing.assert_array_equal(poses[0], larger)
        np.testing.assert_array_equal(poses[2], frames[2][1])
        self.assertFalse(valid[1])
        np.testing.assert_array_equal(poses[1], np.zeros((21, 2), dtype=np.float32))
        self.assertEqual(float(features[indices == 1, 42].sum()), 0.0)

    def _write_experiment(self, model_name="gru", state=None):
        experiment_dir = self.root / "experiment"
        job_dir = experiment_dir / "jobs" / f"{model_name}_final"
        job_dir.mkdir(parents=True)
        identity = hashlib.sha256(b"synthetic experiment").hexdigest()
        classes = ["gesture Z", "gesture A"]
        experiment = {
            "schema_version": 1,
            "experiment_id": "synthetic",
            "identity": identity,
            "data_identity": hashlib.sha256(b"synthetic data").hexdigest(),
            "class_order": classes,
            "settings": {"extractor": "mediapipe", "sequence_length": 8},
        }
        (experiment_dir / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
        for name, content in (("last.pt", b"synthetic last"), ("history.json", b"{}")):
            (job_dir / name).write_bytes(content)
        model = classifiers.build_model(model_name, len(classes))
        if state is None:
            state = model.state_dict()
        identity_for_job = job_identity(identity, model_name)
        torch.save({"identity": identity_for_job, "model_state": state, "epochs": 1}, job_dir / "final.pt")
        hashes = {
            name: hashlib.sha256((job_dir / name).read_bytes()).hexdigest()
            for name in ("last.pt", "history.json", "final.pt")
        }
        (job_dir / "training_complete.json").write_text(json.dumps({
            "status": "complete", "identity": identity_for_job, "sha256": hashes,
        }), encoding="utf-8")
        return experiment_dir, job_dir, classes, identity_for_job

    def test_checkpoint_validates_identity_hash_strict_state_and_saved_order(self):
        experiment_dir, _, classes, _ = self._write_experiment()
        with patch.object(webcam.torch, "load", wraps=torch.load) as load:
            loaded = webcam.load_model(experiment_dir, "gru")
        self.assertEqual(loaded["class_order"], classes)
        self.assertEqual(loaded["sequence_length"], 8)
        self.assertFalse(loaded["model"].training)
        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        self.assertIs(load.call_args.kwargs["weights_only"], True)

    def test_checkpoint_rejects_marker_identity_and_hash_mismatch(self):
        experiment_dir, job_dir, _, _ = self._write_experiment()
        marker_path = job_dir / "training_complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["identity"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "marker identity"):
            webcam.load_model(experiment_dir, "gru")
        marker["identity"] = job_identity(hashlib.sha256(b"synthetic experiment").hexdigest(), "gru")
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        with (job_dir / "final.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            webcam.load_model(experiment_dir, "gru")

    def test_checkpoint_rejects_wrong_state_dict_strictly(self):
        experiment_dir, _, _, _ = self._write_experiment(state={"unexpected": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "state does not match"):
            webcam.load_model(experiment_dir, "gru")

    def test_checkpoint_rejects_payload_identity_mismatch(self):
        experiment_dir, job_dir, _, _ = self._write_experiment()
        checkpoint_path = job_dir / "final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["identity"] = "0" * 64
        torch.save(checkpoint, checkpoint_path)
        marker_path = job_dir / "training_complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["sha256"]["final.pt"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint job/experiment identity"):
            webcam.load_model(experiment_dir, "gru")

    def test_warmup_prediction_cadence_and_missing_hand_clear(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, values):
                self.calls += 1
                assert not torch.is_grad_enabled()
                assert values.device.type == "cpu"
                return torch.tensor([[0.0, 2.0]])

        model = FakeModel().eval()
        loaded = {
            "identity": "experiment",
            "class_order": ["saved second", "saved first"],
            "sequence_length": 8,
            "model_name": "gru",
            "model": model,
        }
        session = webcam.LiveSession(self.root, loaded, 3.0)
        hand = np.stack([pose((0.2, 0.2), 0.1)])
        session.add_sample(0.0, hand)
        self.assertFalse(session.warmed_up(2.99))
        self.assertFalse(session.predict_if_due(2.99))
        session.add_sample(2.95, hand)
        self.assertTrue(session.warmed_up(3.0))
        self.assertTrue(session.predict_if_due(3.0))
        self.assertEqual(session.prediction[0], "saved first")
        self.assertFalse(session.predict_if_due(3.199))
        self.assertTrue(session.predict_if_due(3.2))
        self.assertEqual(model.calls, 2)
        session.add_sample(3.21, np.empty((0, 21, 2), dtype=np.float32))
        self.assertFalse(session.current_valid)
        self.assertIsNone(session.prediction)
        self.assertFalse(session.predict_if_due(3.5))
        self.assertEqual(model.calls, 2)

    def test_warmup_accepts_slower_sampling_and_recovers_after_capture_gap(self):
        class FakeModel(torch.nn.Module):
            def forward(self, values):
                return torch.tensor([[0.0, 1.0]])

        hand = np.stack([pose((0.2, 0.2), 0.1)])
        for interval in (1 / 12 + 0.000001, 0.100001, 0.125001):
            with self.subTest(interval=interval):
                loaded = {
                    "identity": "experiment", "class_order": ["a", "b"],
                    "sequence_length": 8, "model_name": "gru", "model": FakeModel(),
                }
                session = webcam.LiveSession(self.root, loaded, 3.0)
                for index in range(math.ceil(3.0 / interval) + 1):
                    timestamp = index * interval
                    session.add_sample(timestamp, hand)
                    if timestamp < 3.0:
                        self.assertFalse(session.predict_if_due(timestamp))
                self.assertTrue(session.predict_if_due(timestamp))

                resumed_at = timestamp + 6.0
                session.add_sample(resumed_at, hand)
                self.assertFalse(session.predict_if_due(resumed_at))
                session.add_sample(resumed_at + interval, hand)
                self.assertFalse(session.predict_if_due(resumed_at + interval))
                for index in range(2, math.ceil(3.0 / interval) + 1):
                    timestamp = resumed_at + index * interval
                    session.add_sample(timestamp, hand)
                self.assertTrue(session.predict_if_due(timestamp))

    def test_failed_model_switch_retains_active_model_and_records_error(self):
        model = torch.nn.Linear(1, 1)
        loaded = {
            "identity": "experiment", "class_order": ["a", "b"],
            "sequence_length": 8, "model_name": "gru", "model": model,
        }
        session = webcam.LiveSession(self.root, loaded, 3.0)
        with patch.object(webcam, "load_model", side_effect=ValueError("bad checkpoint hash")):
            self.assertFalse(session.switch_model("lstm"))
        self.assertIs(session.model, model)
        self.assertEqual(session.model_name, "gru")
        self.assertIn("bad checkpoint hash", session.switch_error)

    def test_successful_model_switch_keeps_global_prediction_cadence(self):
        class FakeModel(torch.nn.Module):
            def forward(self, values):
                return torch.tensor([[1.0, 0.0]])

        old_model, new_model = FakeModel(), FakeModel()
        loaded = {
            "identity": "experiment", "class_order": ["a", "b"],
            "sequence_length": 8, "model_name": "gru", "model": old_model,
        }
        session = webcam.LiveSession(self.root, loaded, 3.0)
        hand = np.stack([pose((0.2, 0.2), 0.1)])
        session.add_sample(0.0, hand)
        session.add_sample(2.95, hand)
        self.assertTrue(session.predict_if_due(3.0))
        with patch.object(webcam, "load_model", return_value=loaded | {
                "model_name": "lstm", "model": new_model}):
            self.assertTrue(session.switch_model("lstm"))
        self.assertIs(session.model, new_model)
        self.assertFalse(session.predict_if_due(3.1))
        self.assertTrue(session.predict_if_due(3.2))

    def test_camera_failures_still_release_resources(self):
        hand_model = self.root / "hand_landmarker.task"
        hand_model.write_bytes(b"synthetic model placeholder")
        args = SimpleNamespace(experiment=self.root, window_seconds=3.0, camera=7, hand_model=hand_model)
        loaded = {
            "identity": "experiment", "class_order": ["a", "b"],
            "sequence_length": 8, "model_name": "gru", "model": torch.nn.Linear(1, 1),
        }

        unopened = Mock()
        unopened.isOpened.return_value = False
        with patch.object(webcam.cv2, "VideoCapture", return_value=unopened), \
                patch.object(webcam.cv2, "destroyAllWindows") as destroy:
            with self.assertRaisesRegex(RuntimeError, "could not open camera 7"):
                webcam.run_camera(args, loaded)
        unopened.release.assert_called_once()
        destroy.assert_called_once()

        capture, detector = Mock(), Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch.object(webcam.cv2, "VideoCapture", return_value=capture), \
                patch.object(webcam.landmarks, "MediaPipeHand", return_value=detector), \
                patch.object(webcam.cv2, "destroyAllWindows") as destroy:
            with self.assertRaisesRegex(RuntimeError, "failed to read"):
                webcam.run_camera(args, loaded)
        capture.release.assert_called_once()
        detector.close.assert_called_once()
        destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
