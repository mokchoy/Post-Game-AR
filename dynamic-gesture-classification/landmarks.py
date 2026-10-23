"""Optional RTMLib and MediaPipe extraction, reporting, and CPU benchmarking."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np

import jester_pipeline as jp


FPS = 12
JOINTS = jp.JOINTS
HAND_EDGES = tuple((start + offset, start + offset + 1) for start in (1, 5, 9, 13, 17) for offset in range(3)) + tuple((0, start) for start in (1, 5, 9, 13, 17))
SETTINGS = {
    "rtmlib": {"det_mode": None, "det_input": [320, 320], "det_score_threshold": 0.3, "det_nms_threshold": 0.45, "pose_input": [256, 256], "to_openpose": False, "device": "cpu", "backend": "onnxruntime"},
    "mediapipe": {"running_mode": "VIDEO", "num_hands": 2, "min_hand_detection_confidence": 0.5, "min_hand_presence_confidence": 0.5, "min_tracking_confidence": 0.5, "delegate": "CPU"},
}


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _model_info(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"local model file not found: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": jp.sha256_file(path)}


def _package_info(extractor: str, models: dict[str, Path]) -> dict:
    return {
        "extractor": extractor,
        "package_version": _pkg_version("rtmlib" if extractor == "rtmlib" else "mediapipe"),
        "runtime_version": _pkg_version("onnxruntime" if extractor == "rtmlib" else "mediapipe"),
        "runtime_backend": "onnxruntime-cpu" if extractor == "rtmlib" else "mediapipe-cpu",
        "models": {name: _model_info(path) for name, path in models.items()},
        "settings": SETTINGS[extractor],
        "joints": list(JOINTS),
        "frame_rate_fps": FPS,
        "coordinates": "image-normalized x,y; origin top-left; no mirroring or wrist/scale normalization",
    }


class RTMLibHand:
    def __init__(self, detector_model: Path, pose_model: Path):
        from rtmlib import RTMDet, RTMPose

        self.detector = RTMDet(str(detector_model), model_input_size=(320, 320), det_mode=None, score_thr=0.3, nms_thr=0.45, backend="onnxruntime", device="cpu")
        self.pose = RTMPose(str(pose_model), model_input_size=(256, 256), to_openpose=False, backend="onnxruntime", device="cpu")

    def close(self) -> None:
        return None

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> dict:
        height, width = bgr.shape[:2]
        boxes = self.detector(bgr)
        if boxes is None or np.asarray(boxes).size == 0:
            return _empty_result()
        points, scores = self.pose(bgr, bboxes=boxes)
        points = np.asarray(points, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if points.ndim == 2:
            points = points[None, ...]
        if scores.ndim == 1:
            scores = scores[None, ...]
        if points.ndim != 3 or points.shape[1:] != (21, 2) or scores.shape != points.shape[:2]:
            raise ValueError(f"RTMLib returned unexpected hand shape: points={points.shape}, scores={scores.shape}")
        xy = normalize_pixel_xy(points, width, height)
        # RTMLib's public RTMDet API returns boxes without detection scores or handedness.
        return {"xy": xy, "joint_scores": scores, "handedness": [""] * len(points), "handedness_scores": [np.nan] * len(points), "detection_scores": [np.nan] * len(points), "visibility": _nan_joints(len(points)), "presence": _nan_joints(len(points))}


class MediaPipeHand:
    def __init__(self, model: Path):
        import mediapipe as mp

        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.task = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self.task.close()

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> dict:
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.task.detect_for_video(image, timestamp_ms)
        hands = result.hand_landmarks
        if not hands:
            return _empty_result()
        xy = np.asarray([[[p.x, p.y] for p in hand] for hand in hands], dtype=np.float32)
        if xy.shape[1:] != (21, 2):
            raise ValueError(f"MediaPipe returned unexpected hand shape: {xy.shape}")
        handedness, handedness_scores = [], []
        for rows in result.handedness:
            category = rows[0] if rows else None
            handedness.append((getattr(category, "category_name", "") or "") if category else "")
            handedness_scores.append(_optional_float(getattr(category, "score", None)) if category else np.nan)
        visibility = _nan_joints(len(hands))
        presence = _nan_joints(len(hands))
        for hi, hand in enumerate(hands):
            for ji, point in enumerate(hand):
                visibility[hi, ji] = _optional_float(getattr(point, "visibility", None))
                presence[hi, ji] = _optional_float(getattr(point, "presence", None))
        # MediaPipe coordinates are already normalized to original image width and height.
        return {"xy": xy, "joint_scores": _nan_joints(len(hands)), "handedness": handedness, "handedness_scores": handedness_scores, "detection_scores": [np.nan] * len(hands), "visibility": visibility, "presence": presence}


def _nan_joints(hands: int) -> np.ndarray:
    return np.full((hands, 21), np.nan, dtype=np.float32)


def _optional_float(value) -> float:
    return float(value) if value is not None else np.nan


def _empty_result() -> dict:
    return {"xy": np.empty((0, 21, 2), np.float32), "joint_scores": np.empty((0, 21), np.float32), "handedness": [], "handedness_scores": [], "detection_scores": [], "visibility": np.empty((0, 21), np.float32), "presence": np.empty((0, 21), np.float32)}


def normalize_pixel_xy(points: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32).copy()
    if xy.shape[-1] != 2 or width <= 0 or height <= 0:
        raise ValueError("coordinates must end in x,y and image dimensions must be positive")
    xy[..., 0] /= width
    xy[..., 1] /= height
    return xy


def _adapter(extractor: str, models: dict[str, Path]):
    if extractor == "rtmlib":
        return RTMLibHand(models["detector"], models["pose"])
    return MediaPipeHand(models["task"])


def _selected_ids(path: Path | None, config: dict, records: dict[str, dict], training_only: bool = False) -> list[str]:
    default = jp.cache_paths(config)["metadata"] / "pilot_ids.csv"
    source = path or default
    if not source.is_file():
        raise FileNotFoundError(f"clip ID list not found: {source}; create it with the explicit pilot command")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row.get("clip_id", "").strip() for row in rows]
    if not ids or any(not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"clip ID list must contain unique clip_id values: {source}")
    unknown = set(ids) - records.keys()
    non_train = [cid for cid in ids if cid in records and records[cid]["official_split"] != "train"] if training_only else []
    mismatched = [
        row.get("clip_id", "") for row in rows
        if row.get("clip_id") in records
        and ((row.get("official_split") and row["official_split"] != records[row["clip_id"]]["official_split"])
             or (row.get("class_label") and row["class_label"] != records[row["clip_id"]]["class_label"]))
    ]
    if unknown or mismatched or non_train:
        raise ValueError(f"inputs contain unknown or mismatched clip metadata: unknown={sorted(unknown)[:5]}, mismatched={mismatched[:5]}, non-training={non_train[:5]}")
    return ids


def _ids_hash(ids: list[str]) -> str:
    stable = sorted(ids, key=jp._numeric_key)
    return hashlib.sha256("\n".join(stable).encode("utf-8")).hexdigest()


def _read_marker(npz_path: Path, marker_path: Path, identity: str) -> bool:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("input_identity") != identity:
            return False
        if marker.get("output_sha256") != jp.sha256_file(npz_path):
            return False
        with np.load(npz_path, allow_pickle=False) as data:
            return _valid_cache(data)
    except Exception:
        return False


def _valid_cache(data) -> bool:
    required = {
        "frame_indices", "frame_timestamps_seconds", "source_frame_names", "frame_valid",
        "image_widths", "image_heights", "frame_hand_offsets", "detected_hand_count",
        "hand_xy", "joint_mask", "joint_scores", "detection_scores", "handedness",
        "handedness_scores", "visibility", "presence", "metadata_json",
    }
    if not required.issubset(data.files if hasattr(data, "files") else data.keys()):
        return False
    frame_count = len(data["frame_indices"])
    hand_count = len(data["hand_xy"])
    offsets = data["frame_hand_offsets"]
    counts = data["detected_hand_count"]
    if (
        data["frame_indices"].shape != (frame_count,)
        or data["frame_timestamps_seconds"].shape != (frame_count,)
        or data["source_frame_names"].shape != (frame_count,)
        or data["frame_valid"].shape != (frame_count,)
        or data["image_widths"].shape != (frame_count,)
        or data["image_heights"].shape != (frame_count,)
        or counts.shape != (frame_count,)
        or offsets.shape != (frame_count + 1,)
        or data["hand_xy"].shape != (hand_count, 21, 2)
        or data["joint_mask"].shape != (hand_count, 21)
        or data["joint_scores"].shape != (hand_count, 21)
        or data["visibility"].shape != (hand_count, 21)
        or data["presence"].shape != (hand_count, 21)
        or any(data[key].shape != (hand_count,) for key in ("detection_scores", "handedness", "handedness_scores"))
        or not frame_count
        or int(offsets[0]) != 0
        or int(offsets[-1]) != hand_count
        or not np.array_equal(np.diff(offsets), counts)
        or np.any(np.diff(data["frame_indices"]) <= 0)
        or not np.allclose(data["frame_timestamps_seconds"], (data["frame_indices"] - 1) / FPS)
        or np.any(counts < 0)
    ):
        return False
    try:
        metadata = json.loads(str(data["metadata_json"].item()))
        return metadata.get("joints") == list(JOINTS) and metadata.get("frame_rate_fps") == FPS
    except (ValueError, json.JSONDecodeError):
        return False


def _recover_unmarked(npz_path: Path, marker_path: Path, identity: str, frame_identity: str, package_key: str) -> bool:
    """Recover a verified NPZ left between the atomic data and marker writes."""
    if marker_path.exists() or not npz_path.is_file():
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            valid = (
                metadata.get("input_identity") == identity
                and metadata.get("source_frame_set_sha256") == frame_identity
                and metadata.get("package_key") == package_key
                and _valid_cache(data)
            )
        if not valid:
            return False
        jp.write_json(marker_path, {
            "status": "complete", "input_identity": identity,
            "input_frame_set_sha256": frame_identity, "package_key": package_key,
            "output_sha256": jp.sha256_file(npz_path), "elapsed_seconds": 0,
            "recovered_after_interruption": True, "created_utc": jp.utc_now(),
        })
        return _read_marker(npz_path, marker_path, identity)
    except Exception:
        return False


def _atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = str(path) + ".partial"
    try:
        with open(temp_name, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temp_name, allow_pickle=False) as check:
            if not _valid_cache(check):
                raise ValueError("temporary landmark cache failed validation")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _extract_clip(clip_id: str, clip_dir: Path, extractor: str, models: dict[str, Path], package_info: dict, adapter=None) -> tuple[dict, float]:
    valid, frame_marker, scan = jp.validate_frame_completion(clip_dir)
    if not valid:
        raise RuntimeError(f"source frames are incomplete or corrupt: {scan.get('integrity_status', 'invalid marker')}")
    entries = scan["entries"]
    started = time.perf_counter()
    owns_adapter = adapter is None
    if owns_adapter:
        adapter = _adapter(extractor, models)
    offsets = [0]
    xy_parts, joint_score_parts = [], []
    handedness, handedness_scores, detection_scores, visibility_parts, presence_parts = [], [], [], [], []
    frame_valid, widths, heights, names, detected_counts = [], [], [], [], []
    try:
        for entry in entries:
            frame_valid.append(bool(entry["valid"]))
            widths.append(int(entry["width"]))
            heights.append(int(entry["height"]))
            names.append(entry["filename"] or "")
            result = _empty_result()
            if entry["valid"]:
                raw = np.fromfile(entry["path"], dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
                if image is None:
                    frame_valid[-1] = False
                else:
                    timestamp_ms = (entry["index"] - 1) * 1000 // FPS
                    result = adapter.detect(image, timestamp_ms)
            count = len(result["xy"])
            detected_counts.append(count)
            if count:
                xy_parts.append(result["xy"])
                joint_score_parts.append(result["joint_scores"])
                handedness.extend(result["handedness"])
                handedness_scores.extend(result["handedness_scores"])
                detection_scores.extend(result["detection_scores"])
                visibility_parts.append(result["visibility"])
                presence_parts.append(result["presence"])
            offsets.append(offsets[-1] + count)
    finally:
        if owns_adapter:
            adapter.close()
    hands = offsets[-1]
    xy = np.concatenate(xy_parts, axis=0) if xy_parts else np.empty((0, 21, 2), np.float32)
    joint_scores = np.concatenate(joint_score_parts, axis=0) if joint_score_parts else np.empty((0, 21), np.float32)
    visibility = np.concatenate(visibility_parts, axis=0) if visibility_parts else np.empty((0, 21), np.float32)
    presence = np.concatenate(presence_parts, axis=0) if presence_parts else np.empty((0, 21), np.float32)
    frame_indices = np.asarray([entry["index"] for entry in entries], dtype=np.int32)
    timestamp_seconds = (frame_indices.astype(np.float64) - 1) / FPS
    metadata = package_info | {
        "clip_id": clip_id,
        "source_frame_set_sha256": frame_marker["frame_set_sha256"],
        "source_frame_count": len(entries),
        "runtime_backend": "onnxruntime-cpu" if extractor == "rtmlib" else "mediapipe-cpu",
        "candidate_storage": "concatenated hand candidates; frame_hand_offsets maps candidates to source frames",
        "score_note": "RTMLib joint scores and MediaPipe handedness scores have different meanings and are not comparable.",
    }
    arrays = {
        "frame_indices": frame_indices,
        "frame_timestamps_seconds": timestamp_seconds,
        "source_frame_names": np.asarray(names, dtype="U64"),
        "frame_valid": np.asarray(frame_valid, dtype=np.bool_),
        "image_widths": np.asarray(widths, dtype=np.int32),
        "image_heights": np.asarray(heights, dtype=np.int32),
        "frame_hand_offsets": np.asarray(offsets, dtype=np.int64),
        "detected_hand_count": np.asarray(detected_counts, dtype=np.int16),
        "hand_xy": xy.astype(np.float32),
        "joint_mask": np.isfinite(xy).all(axis=-1),
        "joint_scores": joint_scores.astype(np.float32),
        "detection_scores": np.asarray(detection_scores, dtype=np.float32),
        "handedness": np.asarray(handedness, dtype="U16"),
        "handedness_scores": np.asarray(handedness_scores, dtype=np.float32),
        "visibility": visibility.astype(np.float32),
        "presence": presence.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    }
    if xy.shape != (hands, 21, 2) or arrays["joint_scores"].shape != (hands, 21) or offsets[-1] != hands:
        raise AssertionError("candidate and frame-offset arrays do not align")
    return arrays, time.perf_counter() - started


def extract(args, config: dict) -> dict:
    labels_dir = config["_labels_dir"]
    classes, records = jp.read_labels(labels_dir)
    ids = _selected_ids(args.ids_file, config, records)
    frames_root = jp.cache_paths(config)["frames"]
    if args.extractor == "rtmlib":
        models = {"detector": args.rtm_det_model, "pose": args.rtm_pose_model}
    else:
        models = {"task": args.model}
    if any(path is None for path in models.values()):
        raise ValueError("local model paths are required; downloads never start automatically")
    package_info = _package_info(args.extractor, models)
    package_key = hashlib.sha256(json.dumps(package_info, sort_keys=True).encode("utf-8")).hexdigest()
    cache_root = jp.cache_paths(config)["root"]
    # shortcut: accounting assumes one cache writer; add shared reservations before parallel extraction.
    used_bytes = jp._cache_bytes(cache_root)
    report_reserve = len(ids) * 1024 + 1024 * 1024
    jp.ensure_capacity(cache_root, config["cache_limit_gb"], report_reserve, used_bytes=used_bytes)
    used_bytes += report_reserve
    jp.claim_cache(jp.cache_paths(config)["root"])
    output_root = jp.cache_paths(config)["landmarks"] / args.extractor
    owner = output_root / ".jester_pipeline_owner.json"
    if output_root.exists() and any(output_root.iterdir()) and not owner.is_file():
        raise RuntimeError(f"refusing to use non-owned landmark directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not owner.exists():
        jp.write_json(owner, {"owner": "jester-data-pipeline", "extractor": args.extractor})
    for clip_id in ids:
        partial = output_root / f"{clip_id}.npz.partial"
        if partial.exists() and not (output_root / f"{clip_id}.npz").exists():
            partial.unlink()
    elapsed = []
    failures = []
    reused = 0
    written = 0
    shared_adapter = None
    try:
        for clip_id in ids:
            clip_dir = frames_root / clip_id
            npz_path = output_root / f"{clip_id}.npz"
            marker_path = output_root / f"{clip_id}.complete.json"
            frame_ok, frame_marker, _ = jp.validate_frame_completion(clip_dir)
            identity_data = {"clip_id": clip_id, "class_label": records[clip_id]["class_label"], "official_split": records[clip_id]["official_split"], "frame_set_sha256": frame_marker.get("frame_set_sha256"), "package_key": package_key}
            identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode("utf-8")).hexdigest()
            if npz_path.exists() or marker_path.exists():
                if _read_marker(npz_path, marker_path, identity):
                    reused += 1
                    continue
                if _recover_unmarked(npz_path, marker_path, identity, frame_marker.get("frame_set_sha256", ""), package_key):
                    used_bytes += marker_path.stat().st_size
                    reused += 1
                    continue
                failures.append({"clip_id": clip_id, "reason": "existing landmark output is partial or incompatible; preserved"})
                continue
            if not frame_ok:
                failures.append({"clip_id": clip_id, "reason": "source frame output is incomplete, missing, or corrupt"})
                continue
            try:
                adapter = None
                if args.extractor == "rtmlib":
                    if shared_adapter is None:
                        shared_adapter = _adapter(args.extractor, models)
                    adapter = shared_adapter
                arrays, seconds = _extract_clip(clip_id, clip_dir, args.extractor, models, package_info, adapter)
                metadata = json.loads(str(arrays["metadata_json"].item()))
                metadata.update({"input_identity": identity, "package_key": package_key, "class_label": records[clip_id]["class_label"], "official_split": records[clip_id]["official_split"]})
                arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
                # Candidate counts vary; reserve for the exact uncompressed output before writing it.
                estimate = sum(value.nbytes for value in arrays.values()) + 1024 * 1024
                jp.ensure_capacity(cache_root, config["cache_limit_gb"], estimate, used_bytes=used_bytes)
                _atomic_npz(npz_path, arrays)
                marker = {
                    "status": "complete",
                    "input_identity": identity,
                    "input_frame_set_sha256": frame_marker["frame_set_sha256"],
                    "package_key": package_key,
                    "output_sha256": jp.sha256_file(npz_path),
                    "elapsed_seconds": seconds,
                    "created_utc": jp.utc_now(),
                }
                jp.write_json(marker_path, marker)
                if not _read_marker(npz_path, marker_path, identity):
                    raise IOError("completed landmark output failed read-back validation")
                elapsed.append(seconds)
                written += 1
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append({"clip_id": clip_id, "reason": reason})
                if "managed cache limit would be exceeded" in str(exc) or "insufficient free disk space" in str(exc):
                    break
            finally:
                # Count files committed before a failure as well as successful writes.
                used_bytes += sum(path.stat().st_size for path in (npz_path, marker_path) if path.is_file())
    finally:
        if shared_adapter is not None:
            shared_adapter.close()
    report_root = jp.cache_paths(config)["reports"] / args.extractor
    report_root.mkdir(parents=True, exist_ok=True)
    jp._write_csv(report_root / "landmark_failures.csv", ("clip_id", "reason"), failures)
    summary = {
        "extractor": args.extractor,
        "requested_clips": len(ids),
        "clip_ids_sha256": _ids_hash(ids),
        "completed_new": written,
        "reused_valid": reused,
        "failures": len(failures),
        "elapsed_processing_seconds": round(sum(elapsed), 4),
        "archive_and_cloud_transfer_included": False,
        "package_key": package_key,
        "created_utc": jp.utc_now(),
    }
    jp.write_json(report_root / "extraction_run.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} landmark extraction failure(s); see {report_root / 'landmark_failures.csv'}")
    return summary


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _runs(indices: list[int]) -> list[str]:
    if not indices:
        return []
    runs = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            runs.append(f"{start}-{previous}" if start != previous else str(start))
            start = value
        previous = value
    runs.append(f"{start}-{previous}" if start != previous else str(start))
    return runs


def _frame_gaps(scan: dict) -> list[int]:
    return sorted(scan.get("missing", []))


def _draw_sheet(clip_id: str, clip_dir: Path, cache: dict | None, destination: Path, reason: str = "", cache_root: Path | None = None, limit_gb: float = 50) -> bool:
    scan = jp.scan_frame_dir(clip_dir)
    valid = [entry for entry in scan["entries"] if entry["valid"] and entry["path"]]
    if not valid:
        return False
    positions = np.linspace(0, len(valid) - 1, min(5, len(valid)), dtype=int).tolist()
    selected = [valid[index] for index in positions]
    tiles = []
    for entry in selected:
        raw = np.fromfile(entry["path"], dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            continue
        if cache is not None:
            frame_slot = int(entry["index"]) - 1
            offsets = cache["frame_hand_offsets"]
            if frame_slot + 1 < len(offsets):
                for hand_index in range(int(offsets[frame_slot]), int(offsets[frame_slot + 1])):
                    color = ((20, 220, 20), (255, 180, 20), (220, 60, 220))[hand_index % 3]
                    points = []
                    for point_index, (x, y) in enumerate(cache["hand_xy"][hand_index]):
                        if cache["joint_mask"][hand_index, point_index]:
                            points.append((int(round(float(x) * image.shape[1])), int(round(float(y) * image.shape[0]))))
                        else:
                            points.append(None)
                    for first, second in HAND_EDGES:
                        if points[first] is not None and points[second] is not None:
                            cv2.line(image, points[first], points[second], color, 1, cv2.LINE_AA)
                    for point in points:
                        if point is not None:
                            cv2.circle(image, point, 2, color, -1)
        scale = max(2, min(5, 600 // max(image.shape[1], 1)))
        image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
        caption = f"frame {entry['index']} ({(entry['index'] - 1) / FPS:.2f}s)"
        if reason:
            caption += f" | {reason[:45]}"
        cv2.putText(image, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if reason else (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    if not tiles:
        return False
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    canvas = np.zeros((height * len(tiles), width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        canvas[index * height:index * height + tile.shape[0], :tile.shape[1]] = tile
    if cache_root is not None:
        jp.ensure_capacity(cache_root, limit_gb, canvas.nbytes + 1024 * 1024)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.stem + ".partial" + destination.suffix)
    if not cv2.imwrite(str(temp_path), canvas):
        raise IOError(f"could not write overlay: {destination}")
    os.replace(temp_path, destination)
    return True


def report(args, config: dict) -> dict:
    labels_dir = config["_labels_dir"]
    classes, records = jp.read_labels(labels_dir)
    ids = _selected_ids(args.ids_file, config, records, training_only=True)
    review_path = jp.cache_paths(config)["metadata"] / "class_review.csv"
    review = jp.read_class_review(classes, review_path if review_path.exists() else None)
    paths = jp.cache_paths(config)
    feature_root = paths["landmarks"] / args.extractor
    report_root = paths["reports"] / args.extractor
    overlay_root = report_root / "overlays"
    jp.ensure_capacity(paths["root"], config["cache_limit_gb"], len(ids) * 512 + 4 * 1024 * 1024)
    jp.claim_cache(paths["root"])
    report_root.mkdir(parents=True, exist_ok=True)
    by_class = defaultdict(lambda: {"selected": 0, "valid_frames": 0, "detected_frames": 0, "multi_hand_frames": 0, "detected_hands": 0, "failed_clips": 0, "elapsed_seconds": [], "landmark_bytes": 0, "wrist_steps": []})
    clip_rows, failures = [], []
    sample_ids = {
        label: sorted(
            (cid for cid in ids if records[cid]["class_label"] == label),
            key=lambda cid: jp._rank(42, "overlay", label, cid),
        )[:5]
        for label in classes
    }
    preview_data = {}
    flagged = []
    logged_failures = {}
    failure_path = report_root / "landmark_failures.csv"
    if failure_path.is_file():
        with failure_path.open("r", encoding="utf-8-sig", newline="") as stream:
            logged_failures = {row["clip_id"]: row["reason"] for row in csv.DictReader(stream)}
    frame_failures = {}
    frame_failure_path = paths["metadata"] / "frame_extraction_failures.csv"
    if frame_failure_path.is_file():
        with frame_failure_path.open("r", encoding="utf-8-sig", newline="") as stream:
            frame_failures = {row["clip_id"]: row["reason"] for row in csv.DictReader(stream)}
    for clip_id in ids:
        record = records[clip_id]
        label = record["class_label"]
        stats = by_class[label]
        stats["selected"] += 1
        clip_dir = paths["frames"] / clip_id
        if not clip_dir.is_dir() and (paths["frames"] / ".staging" / clip_id).is_dir():
            clip_dir = paths["frames"] / ".staging" / clip_id
        frame_ok, frame_marker, scan = jp.validate_frame_completion(clip_dir)
        if not frame_ok and clip_dir.is_dir() and scan.get("integrity_status") == "partial":
            scan = jp.scan_frame_dir(clip_dir)
        cache_path = feature_root / f"{clip_id}.npz"
        marker_path = feature_root / f"{clip_id}.complete.json"
        cache = None
        failure = ""
        if cache_path.exists() and marker_path.exists():
            try:
                cache = _load_npz(cache_path)
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                metadata = json.loads(str(cache["metadata_json"].item()))
                if (
                    marker.get("status") != "complete"
                    or marker.get("output_sha256") != jp.sha256_file(cache_path)
                    or not _valid_cache(cache)
                    or metadata.get("clip_id") != clip_id
                    or metadata.get("class_label") != label
                    or metadata.get("official_split") != record["official_split"]
                    or metadata.get("source_frame_set_sha256") != frame_marker.get("frame_set_sha256")
                    or marker.get("package_key") != metadata.get("package_key")
                ):
                    raise ValueError("landmark cache marker/hash mismatch")
            except Exception as exc:
                failure = f"invalid landmark cache: {exc}"
                cache = None
        else:
            failure = "landmark cache not complete"
        gaps = _frame_gaps(scan)
        corrupt = scan.get("corrupt", [])
        if cache is None or not frame_ok:
            reason = frame_failures.get(clip_id) or logged_failures.get(clip_id) or (failure if "invalid landmark cache" in failure else "")
            status = "failed" if reason or (clip_dir.exists() and not frame_ok) else "not_extracted"
            if status == "failed":
                stats["failed_clips"] += 1
                failures.append({"clip_id": clip_id, "class_label": label, "reason": reason or "source frames incomplete or corrupt", "missing_frame_gaps": ";".join(_runs(gaps)), "corrupt_frame_indices": ";".join(map(str, corrupt))})
            if clip_dir.is_dir() and status == "failed":
                flagged.append((clip_id, None, clip_dir, reason or failure or "source frames incomplete"))
            preview_data[clip_id] = (None, clip_dir, reason or failure or "not extracted")
            clip_rows.append({"clip_id": clip_id, "class_label": label, "valid_frames": scan.get("valid_frame_count", 0), "detected_frames": 0, "multiple_hand_frames": 0, "missing_frame_gaps": ";".join(_runs(gaps)), "corrupt_frame_indices": ";".join(map(str, corrupt)), "status": status})
            continue
        detected = np.asarray(cache["detected_hand_count"])
        valid_mask = np.asarray(cache["frame_valid"], dtype=bool)
        valid_count = int(valid_mask.sum())
        detected_count = int(((detected > 0) & valid_mask).sum())
        multi_count = int(((detected > 1) & valid_mask).sum())
        stats["detected_hands"] += int(detected[valid_mask].sum())
        wrists = []
        for frame_i in range(len(detected)):
            if valid_mask[frame_i] and int(detected[frame_i]) == 1:
                hand_index = int(cache["frame_hand_offsets"][frame_i])
                if cache["joint_mask"][hand_index, 0]:
                    wrists.append((frame_i, cache["hand_xy"][hand_index, 0]))
        stats["wrist_steps"].extend(float(np.linalg.norm(point - previous)) for (previous_i, previous), (frame_i, point) in zip(wrists, wrists[1:]) if frame_i == previous_i + 1)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        stats["valid_frames"] += valid_count
        stats["detected_frames"] += detected_count
        stats["multi_hand_frames"] += multi_count
        stats["elapsed_seconds"].append(float(marker.get("elapsed_seconds", 0)))
        stats["landmark_bytes"] += cache_path.stat().st_size
        preview_data[clip_id] = (cache, clip_dir, "")
        clip_rows.append({"clip_id": clip_id, "class_label": label, "valid_frames": valid_count, "detected_frames": detected_count, "multiple_hand_frames": multi_count, "missing_frame_gaps": ";".join(_runs(gaps)), "corrupt_frame_indices": ";".join(map(str, corrupt)), "status": "complete"})
    previewed = set()
    for label in classes:
        for clip_id in sample_ids[label]:
            cache, clip_dir, reason = preview_data.get(clip_id, (None, paths["frames"] / clip_id, "not extracted"))
            if clip_dir.is_dir() and _draw_sheet(clip_id, clip_dir, cache, overlay_root / label.replace("/", "_") / f"{clip_id}.jpg", reason, paths["root"], config["cache_limit_gb"]):
                previewed.add(clip_id)
    for clip_id, cache, clip_dir, reason in flagged:
        if _draw_sheet(clip_id, clip_dir, cache, overlay_root / "flagged_failures" / f"{clip_id}.jpg", reason, paths["root"], config["cache_limit_gb"]):
            previewed.add(clip_id)
    class_rows = []
    for label in classes:
        stats = by_class[label]
        denominator = stats["valid_frames"]
        class_rows.append({
            "class_label": label,
            "selected_clips": stats["selected"],
            "complete_extraction_failures": stats["failed_clips"],
            "valid_frames": denominator,
            "frames_with_detection": stats["detected_frames"],
            "detection_coverage": f"{stats['detected_frames'] / denominator:.6f}" if denominator else "",
            "frames_with_multiple_hands": stats["multi_hand_frames"],
            "detected_hand_candidates": stats["detected_hands"],
            "single_hand_wrist_step_median_norm": f"{statistics.median(stats['wrist_steps']):.6f}" if stats["wrist_steps"] else "",
            "single_hand_wrist_step_p95_norm": f"{_percentile95(stats['wrist_steps']):.6f}" if stats["wrist_steps"] else "",
            "landmark_cache_bytes": stats["landmark_bytes"],
            "compatibility_decision": review[label]["compatibility_decision"],
            "compatibility_reason": review[label]["reason"],
            "instability_notes": review[label]["instability_notes"],
            "possible_hand_switching_notes": review[label]["possible_hand_switching_notes"],
            "reviewer_notes": review[label]["reviewer_notes"],
        })
    jp._write_csv(report_root / "clip_quality.csv", ("clip_id", "class_label", "valid_frames", "detected_frames", "multiple_hand_frames", "missing_frame_gaps", "corrupt_frame_indices", "status"), clip_rows)
    jp._write_csv(report_root / "class_quality.csv", tuple(class_rows[0].keys()), class_rows)
    jp._write_csv(
        report_root / "class_review.csv",
        ("class_label", "compatibility_decision", "reason", "instability_notes", "possible_hand_switching_notes", "reviewer_notes"),
        [{"class_label": label, **review[label]} for label in classes],
    )
    jp._write_csv(report_root / "report_failures.csv", ("clip_id", "class_label", "reason", "missing_frame_gaps", "corrupt_frame_indices"), failures)
    # Raw files stay in the local cache; reports summarize space rather than copying frame trees.
    all_frames_bytes = sum(p.stat().st_size for p in paths["frames"].rglob("*") if p.is_file() and not p.name.endswith(".partial")) if paths["frames"].exists() else 0
    current_feature_bytes = sum(p.stat().st_size for p in feature_root.glob("*.npz")) if feature_root.exists() else 0
    complete_times = [t for item in by_class.values() for t in item["elapsed_seconds"]]
    avg_seconds = statistics.mean(complete_times) if complete_times else 0.0
    valid_clip_count = sum(1 for row in clip_rows if row["status"] == "complete")
    selected_frames = sum(row["valid_frames"] for row in clip_rows if row["status"] == "complete")
    selected_frame_bytes = sum(p.stat().st_size for cid in ids if (paths["frames"] / cid).is_dir() for p in (paths["frames"] / cid).glob("*.jpg"))
    avg_frame_bytes = selected_frame_bytes / max(1, selected_frames)
    selected_feature_bytes = sum((feature_root / f"{row['clip_id']}.npz").stat().st_size for row in clip_rows if row["status"] == "complete")
    avg_feature_bytes = selected_feature_bytes / max(1, valid_clip_count)
    estimate = {
        "basis": "observed selected clips only; rough linear extrapolation, not a measured full extraction",
        "official_clips": len(records),
        "observed_average_seconds_per_clip": round(avg_seconds, 4),
        "estimated_seconds_for_all_official_clips": round(avg_seconds * len(records), 2) if complete_times else None,
        "observed_average_frame_bytes": round(avg_frame_bytes, 2) if selected_frames else None,
        "estimated_frame_bytes_for_all_clips": round(avg_frame_bytes * len(records)) if selected_frames else None,
        "observed_average_landmark_bytes_per_clip": round(avg_feature_bytes, 2) if valid_clip_count else None,
        "estimated_landmark_bytes_for_all_clips": round(avg_feature_bytes * len(records)) if valid_clip_count else None,
        "current_frames_bytes": all_frames_bytes,
        "current_landmark_npz_bytes": current_feature_bytes,
        "current_managed_cache_bytes": jp._cache_bytes(paths["root"]),
    }
    summary = {"extractor": args.extractor, "clips_reviewed": len(ids), "clip_ids_sha256": _ids_hash(ids), "complete_clips": valid_clip_count, "failed_clips": len(failures), "overlay_clips": len(previewed), "low_coverage_is_not_failure": True, "jitter_is_not_ground_truth_accuracy": True, "estimates": estimate, "created_utc": jp.utc_now()}
    jp.write_json(report_root / "summary.json", summary)
    _atomic_text(report_root / "summary.md",
        f"# {args.extractor} extraction report\n\n"
        f"- Clips reviewed: {len(ids)}; completed landmark caches: {valid_clip_count}; failures: {len(failures)}.\n"
        f"- Detection coverage is reported per class; low coverage on background clips is not automatically an extraction failure.\n"
        f"- Raw coordinates retain image-space translation, scale, and direction. Jitter or coordinate changes are diagnostics, not landmark-accuracy evidence.\n"
        f"- Full-dataset time and storage values are pilot-based linear estimates; transfer time is excluded.\n",
    )
    return summary


def compare(config: dict) -> dict:
    root = jp.cache_paths(config)["reports"]
    jp.ensure_capacity(jp.cache_paths(config)["root"], config["cache_limit_gb"], 4 * 1024 * 1024)
    jp.claim_cache(jp.cache_paths(config)["root"])
    summaries = {}
    for extractor in ("rtmlib", "mediapipe"):
        path = root / extractor / "summary.json"
        summaries[extractor] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if all(summaries.values()) and summaries["rtmlib"].get("clip_ids_sha256") != summaries["mediapipe"].get("clip_ids_sha256"):
        raise ValueError("extractor reports use different clip ID sets; comparison stopped")
    lines = ["# RTMLib and MediaPipe comparison", "", "Both reports must use the same pilot IDs. The score scales are extractor-specific and are not compared.", "", "| Extractor | Clips | Complete | Failures | Median latency (ms) | p95 latency (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for extractor, summary in summaries.items():
        benchmark = root / "benchmark" / f"{extractor}.json"
        timing = json.loads(benchmark.read_text(encoding="utf-8")) if benchmark.is_file() else {}
        lines.append(f"| {extractor} | {summary['clips_reviewed'] if summary else 'not reported'} | {summary['complete_clips'] if summary else '—'} | {summary['failed_clips'] if summary else '—'} | {timing.get('median_inference_ms', 'not benchmarked')} | {timing.get('p95_inference_ms', 'not benchmarked')} |")
    lines += ["", "This is an extraction and CPU-latency comparison only. It does not estimate classifier performance or continuous recognition reliability.", ""]
    root.mkdir(parents=True, exist_ok=True)
    (root / "extractor_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"reports_available": {key: value is not None for key, value in summaries.items()}, "path": str(root / "extractor_comparison.md"), "created_utc": jp.utc_now()}
    jp.write_json(root / "extractor_comparison.json", result)
    return result


def _percentile95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def benchmark(args, config: dict) -> dict:
    classes, records = jp.read_labels(config["_labels_dir"])
    ids = _selected_ids(args.ids_file, config, records, training_only=True)
    paths_by_extractor = {
        "rtmlib": {"detector": args.rtm_det_model, "pose": args.rtm_pose_model},
        "mediapipe": {"task": args.model},
    }
    jp.ensure_capacity(jp.cache_paths(config)["root"], config["cache_limit_gb"], 4 * 1024 * 1024)
    jp.claim_cache(jp.cache_paths(config)["root"])
    results = {}
    for extractor, models in paths_by_extractor.items():
        info = _package_info(extractor, models)
        images = []
        for clip_id in ids:
            scan = jp.scan_frame_dir(jp.cache_paths(config)["frames"] / clip_id)
            for entry in scan["entries"]:
                if entry["valid"] and entry["path"]:
                    images.append((clip_id, entry))
        if not images:
            raise RuntimeError("benchmark found no valid source frames in the supplied clips")
        # One untimed frame on a disposable instance warms model/session setup without changing timed clip state.
        warm_path = images[0][1]["path"]
        warm_raw = np.fromfile(warm_path, dtype=np.uint8)
        warm_image = cv2.imdecode(warm_raw, cv2.IMREAD_COLOR)
        warm = _adapter(extractor, models)
        try:
            warm.detect(warm_image, 0)
        finally:
            warm.close()
        read_ms, decode_ms, inference_ms = [], [], []
        active_clip, adapter = None, None
        try:
            for clip_id, entry in images:
                if clip_id != active_clip:
                    if extractor == "mediapipe" and adapter is not None:
                        adapter.close()
                        adapter = None
                    if adapter is None:
                        adapter = _adapter(extractor, models)
                    active_clip = clip_id
                start = time.perf_counter()
                raw = np.fromfile(entry["path"], dtype=np.uint8)
                read_ms.append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
                decode_ms.append((time.perf_counter() - start) * 1000)
                if image is None:
                    continue
                start = time.perf_counter()
                adapter.detect(image, (entry["index"] - 1) * 1000 // FPS)
                inference_ms.append((time.perf_counter() - start) * 1000)
        finally:
            if adapter is not None:
                adapter.close()
        if not inference_ms:
            raise RuntimeError(f"benchmark decoding failed for all frames for {extractor}")
        results[extractor] = {
            "extractor": extractor,
            "clips": ids,
            "frame_count": len(inference_ms),
            "median_decode_ms": round(statistics.median(decode_ms), 4),
            "p95_decode_ms": round(_percentile95(decode_ms), 4),
            "median_file_read_ms": round(statistics.median(read_ms), 4),
            "p95_file_read_ms": round(_percentile95(read_ms), 4),
            "median_inference_ms": round(statistics.median(inference_ms), 4),
            "p95_inference_ms": round(_percentile95(inference_ms), 4),
            "inference_boundary": "includes RGB conversion where applicable, preprocessing, detection and landmark inference; excludes model/session loading, warm-up, image decode, file reads, and cloud transfer",
            "warmup_policy": "one valid frame on a disposable extractor instance per backend, excluded from timings",
            "hardware": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
            "thread_settings": {key: os.environ.get(key, "unset") for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "opencv_threads": cv2.getNumThreads(),
            "onnxruntime_threads": {"intra_op_num_threads": "runtime default", "inter_op_num_threads": "runtime default"} if extractor == "rtmlib" else None,
            "mediapipe_threads": "runtime default; installed HandLandmarkerOptions exposes no thread-count setting" if extractor == "mediapipe" else None,
            "runtime_backend": info["runtime_backend"] if "runtime_backend" in info else ("onnxruntime-cpu" if extractor == "rtmlib" else "mediapipe-cpu"),
            "extractor_version": info["package_version"],
            "runtime_version": _pkg_version("onnxruntime" if extractor == "rtmlib" else "mediapipe"),
            "settings": SETTINGS[extractor],
            "models": info["models"],
            "cloud_transfer_included": False,
            "created_utc": jp.utc_now(),
        }
    benchmark_root = jp.cache_paths(config)["reports"] / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    for extractor, result in results.items():
        jp.write_json(benchmark_root / f"{extractor}.json", result)
    (benchmark_root / "comparison.md").write_text(
        "# CPU extraction benchmark\n\n"
        "The measured inference boundary includes color conversion where needed, preprocessing, detection, and landmark inference. File read and image decode are timed separately.\n\n"
        "| Extractor | Read median / p95 ms | Decode median / p95 ms | Inference median / p95 ms |\n|---|---:|---:|---:|\n" +
        "\n".join(f"| {name} | {row['median_file_read_ms']} / {row['p95_file_read_ms']} | {row['median_decode_ms']} / {row['p95_decode_ms']} | {row['median_inference_ms']} / {row['p95_inference_ms']} |" for name, row in results.items()) +
        "\n\nOne untimed warm-up frame was run on a disposable backend instance. The benchmark does not measure cloud transfer, classifier performance, or continuous recognition reliability.\n",
        encoding="utf-8",
    )
    return {"extractors": list(results), "identical_clip_ids": True, "frame_count_per_extractor": {k: v["frame_count"] for k, v in results.items()}, "report": str(benchmark_root / "comparison.md")}


def run_command(args, config: dict) -> dict:
    if args.command == "extract-landmarks":
        return extract(args, config)
    if args.command == "report":
        return report(args, config)
    if args.command == "benchmark":
        return benchmark(args, config)
    raise ValueError(f"unsupported landmarks command: {args.command}")
