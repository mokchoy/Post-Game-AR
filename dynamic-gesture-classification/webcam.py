"""CPU-only live webcam inference for a completed Jester experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import classifiers
import landmarks
import training_data


MODELS = {"1": "gru", "2": "lstm", "3": "ddnet"}
SAMPLE_INTERVAL = 1.0 / 12.0
PREDICTION_INTERVAL = 1.0 / 5.0
DEFAULT_HAND_MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_identity(experiment_identity: str, model_name: str) -> str:
    job = {"kind": "final", "model": model_name}
    payload = json.dumps(
        {"experiment_identity": experiment_identity, "job": job},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_model(experiment_dir: Path, model_name: str) -> dict:
    if model_name not in {"gru", "lstm", "ddnet"}:
        raise ValueError(f"unknown model: {model_name}")
    experiment_dir = Path(experiment_dir).resolve()
    metadata_path = experiment_dir / "experiment.json"
    try:
        experiment = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read experiment metadata {metadata_path}: {exc}") from exc
    if not isinstance(experiment, dict) or experiment.get("schema_version") != 1:
        raise ValueError("unsupported or invalid experiment metadata")
    identity = experiment.get("identity")
    if not _valid_digest(identity) or not _valid_digest(experiment.get("data_identity")):
        raise ValueError("experiment identity is missing or invalid")
    classes = experiment.get("class_order")
    if (not isinstance(classes, list) or len(classes) < 2
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("experiment class_order must contain unique nonempty class names")
    settings = experiment.get("settings")
    if not isinstance(settings, dict) or settings.get("extractor") != "mediapipe":
        raise ValueError("live webcam supports MediaPipe experiments only")
    sequence_length = settings.get("sequence_length")
    if type(sequence_length) is not int or sequence_length < 8 or sequence_length % 2:
        raise ValueError("experiment sequence_length must be even and at least 8")

    job_dir = experiment_dir / "jobs" / f"{model_name}_final"
    marker_path = job_dir / "training_complete.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read training completion marker {marker_path}: {exc}") from exc
    if not isinstance(marker, dict):
        raise ValueError("training completion marker must be a JSON object")
    job_identity = _job_identity(identity, model_name)
    if marker.get("status") != "complete" or marker.get("identity") != job_identity:
        raise ValueError(f"training completion marker identity mismatch for {model_name}_final")
    hashes = marker.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError("training completion marker has no file hashes")
    for name in ("last.pt", "history.json", "final.pt"):
        path = job_dir / name
        if not path.is_file() or not _valid_digest(hashes.get(name)) or hashes[name] != _sha256(path):
            raise ValueError(f"training completion marker hash mismatch or missing file: {name}")

    checkpoint_path = job_dir / "final.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load CPU checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("identity") != job_identity:
        raise ValueError("checkpoint job/experiment identity mismatch")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state")
    model = classifiers.build_model(model_name, len(classes)).cpu()
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"checkpoint state does not match {model_name}: {exc}") from exc
    model.eval()
    return {
        "experiment_dir": experiment_dir,
        "identity": identity,
        "class_order": classes,
        "sequence_length": sequence_length,
        "model_name": model_name,
        "model": model,
    }


def _candidate_array(value: np.ndarray) -> np.ndarray:
    if value is None or np.asarray(value).size == 0:
        return np.empty((0, 21, 2), dtype=np.float32)
    candidates = np.asarray(value, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[1:] != (21, 2):
        raise ValueError(f"MediaPipe candidates must have shape (hands, 21, 2), got {candidates.shape}")
    return candidates.copy()


def _select_window(frame_candidates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not frame_candidates:
        raise ValueError("cannot preprocess an empty rolling window")
    counts = np.asarray([len(frame) for frame in frame_candidates], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    xy = np.concatenate(frame_candidates, axis=0) if offsets[-1] else np.empty((0, 21, 2), dtype=np.float32)
    joint_mask = np.isfinite(xy).all(axis=2)
    poses, valid, _, _ = training_data._select_candidates(xy, joint_mask, offsets)
    return poses, valid


def preprocess_candidates(frame_candidates: list[np.ndarray], sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the training candidate selection and uniform sampling to one window."""
    poses, valid = _select_window(frame_candidates)
    sample = training_data._sample_indices(len(poses), sequence_length)
    features = np.zeros((sequence_length, 43), dtype=np.float32)
    features[:, :42] = poses[sample].reshape(sequence_length, 42)
    features[:, 42] = valid[sample].astype(np.float32)
    return features, poses, valid


class LiveSession:
    def __init__(self, experiment_dir: Path, loaded: dict, window_seconds: float):
        self.experiment_dir = Path(experiment_dir).resolve()
        self.identity = loaded["identity"]
        self.class_order = loaded["class_order"]
        self.sequence_length = loaded["sequence_length"]
        self.window_seconds = window_seconds
        self.model_name = loaded["model_name"]
        self.model = loaded["model"]
        self.samples: deque[tuple[float, np.ndarray]] = deque()
        self.started_at: float | None = None
        self.last_prediction_at = -math.inf
        self.current_pose = np.zeros((21, 2), dtype=np.float32)
        self.current_valid = False
        self.prediction: tuple[str, float] | None = None
        self.switch_error: str | None = None

    def add_sample(self, timestamp: float, candidates: np.ndarray) -> None:
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError("sample timestamps must increase monotonically")
        if self.started_at is None:
            self.started_at = timestamp
        self.samples.append((timestamp, _candidate_array(candidates)))
        cutoff = timestamp - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        poses, valid = _select_window([frame for _, frame in self.samples])
        self.current_pose = poses[-1]
        self.current_valid = bool(valid[-1])
        if not self.current_valid:
            self.prediction = None

    def warmed_up(self, timestamp: float) -> bool:
        if self.started_at is None or timestamp - self.started_at < self.window_seconds:
            return False
        if self.window_seconds <= SAMPLE_INTERVAL:
            return bool(self.samples)
        if len(self.samples) < 2:
            return False
        span = self.samples[-1][0] - self.samples[0][0]
        sample_interval = max(SAMPLE_INTERVAL, span / (len(self.samples) - 1))
        return span >= self.window_seconds - sample_interval

    def predict_if_due(self, timestamp: float) -> bool:
        if (not self.warmed_up(timestamp) or not self.current_valid
                or timestamp - self.last_prediction_at < PREDICTION_INTERVAL):
            return False
        features, _, _ = preprocess_candidates(
            [frame for _, frame in self.samples], self.sequence_length
        )
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features[None, ...]).to("cpu"))
            if not isinstance(logits, torch.Tensor) or logits.shape != (1, len(self.class_order)):
                raise ValueError("model returned logits with an unexpected shape")
            probabilities = torch.softmax(logits, dim=1)[0]
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("model returned non-finite probabilities")
        index = int(torch.argmax(probabilities).item())
        self.prediction = (self.class_order[index], float(probabilities[index].item()))
        self.last_prediction_at = timestamp
        return True

    def switch_model(self, model_name: str) -> bool:
        if model_name == self.model_name:
            self.switch_error = None
            return True
        try:
            loaded = load_model(self.experiment_dir, model_name)
            if (loaded["identity"] != self.identity or loaded["class_order"] != self.class_order
                    or loaded["sequence_length"] != self.sequence_length):
                raise ValueError("experiment identity or frozen settings changed during the session")
        except Exception as exc:
            self.switch_error = str(exc)
            return False
        self.model_name, self.model = loaded["model_name"], loaded["model"]
        self.prediction = None
        self.switch_error = None
        return True


def _draw(frame: np.ndarray, session: LiveSession, fps: float, timestamp: float) -> None:
    height, width = frame.shape[:2]
    if session.current_valid:
        points = [
            (round(float(x) * (width - 1)), round(float(y) * (height - 1)))
            for x, y in session.current_pose
        ]
        for start, end in landmarks.HAND_EDGES:
            cv2.line(frame, points[start], points[end], (0, 220, 0), 2)
        for point in points:
            cv2.circle(frame, point, 3, (40, 80, 255), -1)

    lines = [f"Model: {session.model_name.upper()}", f"Processing FPS: {fps:.1f}"]
    if not session.warmed_up(timestamp):
        lines.append("Warming up")
    elif session.current_valid:
        if session.prediction is None:
            lines.append("Prediction: --")
        else:
            label, probability = session.prediction
            lines.append(f"Prediction: {label} ({probability:.3f})")
    if not session.current_valid:
        lines.append("No hand detected")
    if session.switch_error:
        lines.append(f"Model switch failed: {session.switch_error[:90]}")
    for index, line in enumerate(lines):
        color = (0, 0, 255) if line.startswith("Model switch failed") else (255, 255, 255)
        cv2.putText(frame, line, (16, 30 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def run_camera(args, initial_model: dict) -> None:
    capture = detector = None
    try:
        if not args.hand_model.is_file():
            raise FileNotFoundError(f"MediaPipe hand model not found: {args.hand_model}")
        capture = cv2.VideoCapture(args.camera)
        if not capture.isOpened():
            raise RuntimeError(f"could not open camera {args.camera}")
        try:
            detector = landmarks.MediaPipeHand(args.hand_model)
        except Exception as exc:
            raise RuntimeError(f"could not initialize MediaPipe hand model: {exc}") from exc

        session = LiveSession(args.experiment, initial_model, args.window_seconds)
        video_started = time.monotonic()
        last_sample_at: float | None = None
        last_video_timestamp_ms = -1
        while True:
            loop_started = time.monotonic()
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read a frame from camera {args.camera}")
            now = time.monotonic()
            if last_sample_at is None or now - last_sample_at >= SAMPLE_INTERVAL:
                video_timestamp_ms = max(int((now - video_started) * 1000), last_video_timestamp_ms + 1)
                result = detector.detect(frame, video_timestamp_ms)
                session.add_sample(now, result["xy"])
                last_sample_at = now
                last_video_timestamp_ms = video_timestamp_ms
            session.predict_if_due(now)
            fps = 1.0 / max(time.monotonic() - loop_started, 1e-9)
            _draw(frame, session, fps, now)
            cv2.imshow("Dynamic gesture recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if chr(key) in MODELS:
                session.switch_model(MODELS[chr(key)])
    finally:
        try:
            if capture is not None:
                capture.release()
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="completed experiment directory")
    parser.add_argument("--model", choices=("gru", "lstm", "ddnet"), default="gru")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    args = parser.parse_args(argv)
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error("--window-seconds must be finite and greater than zero")
    try:
        loaded = load_model(args.experiment, args.model)
        run_camera(args, loaded)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
