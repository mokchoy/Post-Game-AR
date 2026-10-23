"""Shared, audit-friendly input cache for the three Jester classifiers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import jester_pipeline as jp
import landmarks as lm


HERE = Path(__file__).resolve().parent
FEATURES = 43
BACKGROUND = {"No gesture", "Doing other things"}
PREPROCESSING = "jester-hand-v1:largest-first-nearest-wrist;uniform-nearest;xy-joint-major+valid"


def load_config(path: Path) -> dict:
    path = path.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    t = int(value.get("sequence_length", 32))
    if t < 8 or t % 2:
        raise ValueError("sequence_length must be even and at least 8")
    for key, expected in (("seed", 42), ("batch_size", 64), ("learning_rate", 0.001), ("gradient_clip_norm", 1.0), ("patience", 10), ("epochs", 100)):
        if value.get(key, expected) != expected:
            raise ValueError(f"{key} is frozen at {expected}")
    extractor = value.get("extractor")
    workers = value.get("num_workers", 0)
    if type(workers) is not int or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if extractor not in (None, "rtmlib", "mediapipe"):
        raise ValueError("extractor must be null, 'rtmlib', or 'mediapipe'")
    pipeline_path = Path(value.get("pipeline_config", "config.json"))
    if not pipeline_path.is_absolute():
        pipeline_path = path.parent / pipeline_path
    pipeline = jp.load_config(pipeline_path)
    return value | {
        "_training_config_path": path,
        "_pipeline": pipeline,
        "_paths": jp.cache_paths(pipeline),
        "_sequence_length": t,
        "_pipeline_config_path": pipeline_path.resolve(),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdecimal() else (2**63, value)


def _sha256(path: Path) -> str:
    return jp.sha256_file(path)


def _class_review(path: Path) -> tuple[list[str], list[str], dict[str, dict]]:
    rows = _read_csv(path)
    classes = [row.get("class_label", "") for row in rows]
    if len(classes) != 27 or len(set(classes)) != 27 or "" in classes:
        raise ValueError("class_review.csv must contain each official class exactly once")
    review = {row["class_label"]: row for row in rows}
    for name in BACKGROUND:
        if review.get(name, {}).get("compatibility_decision") != "retain":
            raise ValueError(f"required background class must be retained: {name}")
    bad = [name for name in classes if review[name].get("compatibility_decision") not in {"pending", "retain", "flag", "exclude"}]
    if bad:
        raise ValueError("class review has invalid decisions: " + ", ".join(bad))
    retained = [name for name in classes if review[name]["compatibility_decision"] in {"retain", "flag"}]
    return classes, retained, review


def _source_settings(config: dict, retained: list[str]) -> dict:
    paths = config["_paths"]
    return {
        "preprocessing": PREPROCESSING,
        "sequence_length": config["_sequence_length"],
        "extractor": config["extractor"],
        "manifest_sha256": _sha256(paths["metadata"] / "clip_manifest.csv"),
        "fold_assignments_sha256": _sha256(paths["metadata"] / "fold_assignments.csv"),
        "class_review_sha256": _sha256(paths["metadata"] / "class_review.csv"),
        "retained_classes": retained,
    }


def _read_source(config: dict, verify_raw: bool = True) -> dict:
    paths = config["_paths"]
    blockers: list[str] = []
    for path, name in (
        (paths["metadata"] / "clip_manifest.csv", "clip manifest"),
        (paths["metadata"] / "fold_assignments.csv", "saved fold assignments"),
        (paths["metadata"] / "class_review.csv", "class review"),
    ):
        if not path.is_file():
            blockers.append(f"missing {name}: {path}")
    if blockers:
        return {"blockers": blockers, "paths": paths}

    manifest_rows = _read_csv(paths["metadata"] / "clip_manifest.csv")
    manifest = {row["clip_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("clip manifest has duplicate clip IDs")
    classes, retained, review = _class_review(paths["metadata"] / "class_review.csv")
    train_ids = {cid for cid, row in manifest.items() if row.get("official_split") == "train"}
    validation_ids = {cid for cid, row in manifest.items() if row.get("official_split") == "validation"}
    test_ids = {cid for cid, row in manifest.items() if row.get("official_split") == "test"}
    if not train_ids or not validation_ids or test_ids & (train_ids | validation_ids):
        raise ValueError("manifest official split membership is invalid")
    manifest_classes = {row.get("class_label", "") for row in manifest.values() if row.get("official_split") in {"train", "validation"}}
    if manifest_classes != set(classes):
        raise ValueError("class review labels do not match the labeled manifest classes")

    fold_rows = _read_csv(paths["metadata"] / "fold_assignments.csv")
    fold_by_id: dict[str, str] = {}
    for row in fold_rows:
        cid, fold = row.get("clip_id", ""), row.get("fold", "")
        if cid in fold_by_id or cid not in train_ids or fold not in {"1", "2", "3"}:
            raise ValueError(f"invalid saved fold assignment: {row}")
        if row.get("official_split") != "train" or row.get("class_label") != manifest[cid].get("class_label"):
            raise ValueError(f"saved fold assignment disagrees with manifest: {cid}")
        fold_by_id[cid] = fold
    if set(fold_by_id) != train_ids:
        raise ValueError("saved fold assignments must cover every official training clip exactly once")
    for cid, fold in fold_by_id.items():
        if manifest[cid].get("cv_fold") != fold:
            raise ValueError(f"manifest CV fold differs from saved assignment for clip {cid}")
    for label in classes:
        counts = [sum(manifest[cid]["class_label"] == label and fold_by_id[cid] == fold for cid in train_ids) for fold in ("1", "2", "3")]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"saved CV assignments are not class-stratified for {label!r}: {counts}")

    # Existing fold views are checked against, never used to regenerate, assignments.
    for fold in ("1", "2", "3"):
        for role, expected in (
            ("train", {cid for cid, value in fold_by_id.items() if value != fold}),
            ("development", {cid for cid, value in fold_by_id.items() if value == fold}),
        ):
            path = paths["fold_views"] / f"fold_{fold}_{role}.csv"
            if path.exists() and {r.get("clip_id") for r in _read_csv(path)} != expected:
                raise ValueError(f"saved fold view disagrees with fold_assignments.csv: {path}")

    pending = [name for name in classes if review[name].get("compatibility_decision") == "pending"]
    if pending:
        blockers.append(f"class review pending for {len(pending)} classes: {', '.join(pending)}")
    if config.get("extractor") is None:
        blockers.append("extractor is unset; freeze RTMLib or MediaPipe after review")

    retained_ids = {
        cid for cid, row in manifest.items()
        if row.get("official_split") in {"train", "validation"} and row.get("class_label") in retained
    }
    if not retained_ids:
        blockers.append("no retained labeled training or validation clips")
    incomplete_frames = []
    for cid in sorted(retained_ids, key=_numeric_key):
        row = manifest[cid]
        if row.get("extraction_status") != "complete" or row.get("integrity_status") not in {"complete", "ok"}:
            incomplete_frames.append(cid)
    if incomplete_frames:
        blockers.append(f"{len(incomplete_frames)} retained clips lack complete frames; first IDs: {', '.join(incomplete_frames[:5])}")

    raw_root = paths["landmarks"] / str(config.get("extractor") or "")
    raw_hashes: dict[str, str] = {}
    raw_missing: list[str] = []
    if verify_raw and config.get("extractor"):
        for cid in sorted(retained_ids, key=_numeric_key):
            npz = raw_root / f"{cid}.npz"
            marker_path = raw_root / f"{cid}.complete.json"
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                frame_ok, frame_marker, _ = jp.validate_frame_completion(
                    paths["frames"] / cid, expected_count=int(manifest[cid].get("frame_count") or 0)
                )
                if not frame_ok:
                    raise ValueError("source frames are incomplete or changed")
                digest = _sha256(npz)
                if marker.get("output_sha256") != digest or not lm._read_marker(npz, marker_path, marker.get("input_identity", "")):
                    raise ValueError("cache marker or digest mismatch")
                with np.load(npz, allow_pickle=False) as data:
                    info = json.loads(str(data["metadata_json"].item()))
                    if info.get("clip_id") != cid:
                        raise ValueError("cache clip ID mismatch")
                    if info.get("class_label") != manifest[cid].get("class_label") or info.get("official_split") != manifest[cid].get("official_split"):
                        raise ValueError("cache class or official split differs from manifest")
                    if info.get("source_frame_set_sha256") != frame_marker.get("frame_set_sha256") or marker.get("input_frame_set_sha256") != frame_marker.get("frame_set_sha256"):
                        raise ValueError("cache source-frame identity mismatch")
                    if len(data["frame_indices"]) != int(manifest[cid].get("frame_count") or 0):
                        raise ValueError("cache frame count differs from clip manifest")
                    indices = data["frame_indices"]
                    if not np.array_equal(indices, np.arange(1, len(indices) + 1, dtype=indices.dtype)):
                        raise ValueError("source frame gap; repair extraction before training")
                raw_hashes[cid] = digest + ":" + str(marker.get("input_identity", ""))
            except Exception as exc:
                raw_missing.append(f"{cid} ({exc})")
        if raw_missing:
            blockers.append(f"{len(raw_missing)} raw landmark caches missing or invalid; first: " + ", ".join(raw_missing[:5]))

    identity = None
    if verify_raw and config.get("extractor") and not raw_missing:
        identity_input = _source_settings(config, retained) | {"clip_raw_identity": raw_hashes}
        identity = hashlib.sha256(json.dumps(identity_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "blockers": blockers, "paths": paths, "manifest": manifest, "fold_by_id": fold_by_id,
        "classes": classes, "retained_classes": retained, "review": review,
        "train_ids": train_ids, "validation_ids": validation_ids, "test_ids": test_ids,
        "retained_ids": retained_ids, "raw_root": raw_root, "raw_hashes": raw_hashes,
        "identity": identity, "identity_input": identity_input if identity else None,
    }


def validate_feature_cache(path: Path, identity: str | None = None) -> dict:
    marker_path = path / "complete.json"
    metadata_path = path / "metadata.json"
    features_path = path / "features.npy"
    index_path = path / "clip_index.csv"
    audit_paths = [path / f"{name}.npy" for name in (
        "audit_offsets", "audit_frame_indices", "selected_candidate_indices", "candidate_counts", "ambiguous_frames", "sample_frame_indices", "sample_timestamps"
    )]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or (identity and marker.get("identity") != identity):
        raise ValueError(f"shared feature cache identity/status mismatch: {path}")
    frozen_inputs = metadata["identity_inputs"]
    frozen_identity = hashlib.sha256(json.dumps(frozen_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (
        frozen_identity != marker.get("identity") or metadata.get("identity") != frozen_identity
        or metadata.get("class_order") != frozen_inputs.get("retained_classes")
        or metadata.get("sequence_length") != frozen_inputs.get("sequence_length")
    ):
        raise ValueError("shared feature metadata differs from its frozen identity")
    expected_files = [features_path, index_path, *audit_paths]
    if any(not p.is_file() for p in expected_files):
        raise ValueError(f"shared feature cache is incomplete: {path}")
    hashes = marker.get("sha256", {})
    for file in expected_files:
        if hashes.get(file.name) != _sha256(file):
            raise ValueError(f"shared feature file hash mismatch: {file.name}")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.dtype != np.float32 or features.ndim != 3 or features.shape[1:] != (metadata["sequence_length"], FEATURES):
        raise ValueError("shared feature matrix has incompatible dtype or shape")
    index = _read_csv(index_path)
    if len(index) != features.shape[0] or [int(row["feature_index"]) for row in index] != list(range(len(index))):
        raise ValueError("shared feature index does not cover the matrix")
    if len({row["clip_id"] for row in index}) != len(index):
        raise ValueError("shared feature index contains duplicate clip IDs")
    offsets = np.load(audit_paths[0], mmap_mode="r", allow_pickle=False)
    frame_indices = np.load(audit_paths[1], mmap_mode="r", allow_pickle=False)
    selected = np.load(audit_paths[2], mmap_mode="r", allow_pickle=False)
    counts = np.load(audit_paths[3], mmap_mode="r", allow_pickle=False)
    ambiguous = np.load(audit_paths[4], mmap_mode="r", allow_pickle=False)
    samples = np.load(audit_paths[5], mmap_mode="r", allow_pickle=False)
    timestamps = np.load(audit_paths[6], mmap_mode="r", allow_pickle=False)
    if (
        offsets.shape != (len(index) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 1)
        or offsets[-1] != len(frame_indices) or selected.shape != frame_indices.shape
        or counts.shape != frame_indices.shape or ambiguous.shape != frame_indices.shape
        or samples.shape != (len(index), features.shape[1]) or timestamps.shape != samples.shape
        or not np.array_equal(ambiguous, counts > 1)
    ):
        raise ValueError("shared feature audit arrays do not align")
    del offsets, frame_indices, selected, counts, ambiguous, samples, timestamps
    return {"path": path, "metadata": metadata, "index": index, "features": features, "identity": marker["identity"]}


def inspect_inputs(config_path: Path, require_features: bool = True) -> dict:
    config = load_config(config_path)
    source = _read_source(config, verify_raw=not require_features)
    blockers = list(source.get("blockers", []))
    feature_cache = None
    if require_features:
        if not blockers:
            expected = _source_settings(config, source["retained_classes"])
            matches = []
            input_root = config["_paths"]["root"] / "training" / "inputs"
            for metadata_path in sorted(input_root.glob("*/metadata.json")):
                if not (metadata_path.parent / "complete.json").is_file():
                    continue
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                frozen = metadata.get("identity_inputs", {})
                if {key: value for key, value in frozen.items() if key != "clip_raw_identity"} == expected:
                    matches.append(metadata_path.parent)
            if len(matches) != 1:
                blockers.append("expected exactly one frozen shared feature cache matching current metadata; run prepare_training_inputs.py --build")
            else:
                try:
                    feature_cache = validate_feature_cache(matches[0], matches[0].name)
                    source["identity"] = feature_cache["identity"]
                    source["identity_input"] = feature_cache["metadata"]["identity_inputs"]
                except Exception as exc:
                    blockers.append(f"shared feature cache invalid: {exc}")
        return {"config": config, "source": source, "feature_cache": feature_cache, "blockers": blockers}
    if source.get("identity"):
        path = config["_paths"]["root"] / "training" / "inputs" / source["identity"]
        if path.exists():
            try:
                feature_cache = validate_feature_cache(path, source["identity"])
            except Exception as exc:
                blockers.append(f"shared feature cache invalid: {exc}")
        elif require_features:
            blockers.append(f"shared feature cache missing; run prepare_training_inputs.py --build --config {config['_training_config_path']}")
    elif require_features:
        blockers.append("shared feature cache cannot be identified until source data and extractor are ready")
    return {"config": config, "source": source, "feature_cache": feature_cache, "blockers": blockers}


def _select_candidates(xy: np.ndarray, joint_mask: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = len(offsets) - 1
    selected = np.full((frames, 21, 2), 0, dtype=np.float32)
    valid = np.zeros(frames, dtype=np.bool_)
    selected_indices = np.full(frames, -1, dtype=np.int16)
    counts = np.diff(offsets).astype(np.int16)
    previous_wrist = None
    for frame in range(frames):
        start, end = int(offsets[frame]), int(offsets[frame + 1])
        options = [i for i in range(start, end) if joint_mask[i].all() and np.isfinite(xy[i]).all()]
        if not options:
            continue
        if previous_wrist is None:
            def rank(candidate: int) -> tuple[float, int]:
                points = xy[candidate]
                area = float(np.ptp(points[:, 0]) * np.ptp(points[:, 1]))
                return (-area, candidate - start)
            chosen = min(options, key=rank)
        else:
            chosen = min(options, key=lambda candidate: (float(np.sum((xy[candidate, 0] - previous_wrist) ** 2)), candidate - start))
        pose = np.asarray(xy[chosen], dtype=np.float32)
        selected[frame] = pose
        valid[frame] = True
        selected_indices[frame] = chosen - start
        previous_wrist = pose[0].copy()
    return selected, valid, selected_indices, counts


def _sample_indices(frame_count: int, length: int) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("clip has no source frames")
    return np.floor(np.linspace(0, frame_count - 1, num=length) + 0.5).astype(np.int32)


def _raw_pose(path: Path, clip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        metadata = json.loads(str(raw["metadata_json"].item()))
        if metadata.get("clip_id") != clip_id:
            raise ValueError(f"raw landmark clip ID mismatch for {clip_id}")
        poses, valid, selected, counts = _select_candidates(raw["hand_xy"], raw["joint_mask"], raw["frame_hand_offsets"])
        indices = raw["frame_indices"].astype(np.int32, copy=True)
        timestamps = raw["frame_timestamps_seconds"].astype(np.float64, copy=True)
    return poses, valid, selected, counts, np.column_stack((indices, timestamps))


def build_feature_cache(config_path: Path) -> Path:
    inspected = inspect_inputs(config_path, require_features=False)
    blockers = inspected["blockers"]
    if blockers:
        raise RuntimeError("input preparation blocked:\n- " + "\n- ".join(blockers))
    config, source = inspected["config"], inspected["source"]
    if source.get("identity") is None:
        raise RuntimeError("cannot build shared inputs without a frozen extractor and complete source artifacts")
    root = config["_paths"]["root"]
    input_root = root / "training" / "inputs"
    final = input_root / source["identity"]
    if final.exists():
        cached = validate_feature_cache(final, source["identity"])
        cached["features"]._mmap.close()
        return final

    ids = sorted(source["retained_ids"], key=_numeric_key)
    frame_counts: list[int] = []
    for cid in ids:
        with np.load(source["raw_root"] / f"{cid}.npz", allow_pickle=False) as raw:
            frame_counts.append(len(raw["frame_indices"]))
    audit_frames = sum(frame_counts)
    t = config["_sequence_length"]
    estimated = len(ids) * t * FEATURES * 4 + audit_frames * 9 + (len(ids) + 1) * 8 + len(ids) * 400 + 1024**2
    jp.ensure_capacity(root, config["_pipeline"]["cache_limit_gb"], estimated)
    input_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{source['identity'][:12]}-", suffix=".partial", dir=input_root))
    try:
        matrix_path = temp / "features.npy"
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(ids), t, FEATURES))
        audit_offsets = np.lib.format.open_memmap(temp / "audit_offsets.npy", mode="w+", dtype=np.int64, shape=(len(ids) + 1,))
        audit_frames_path = np.lib.format.open_memmap(temp / "audit_frame_indices.npy", mode="w+", dtype=np.int32, shape=(audit_frames,))
        audit_selected = np.lib.format.open_memmap(temp / "selected_candidate_indices.npy", mode="w+", dtype=np.int16, shape=(audit_frames,))
        audit_counts = np.lib.format.open_memmap(temp / "candidate_counts.npy", mode="w+", dtype=np.int16, shape=(audit_frames,))
        audit_ambiguous = np.lib.format.open_memmap(temp / "ambiguous_frames.npy", mode="w+", dtype=np.bool_, shape=(audit_frames,))
        sample_frames = np.lib.format.open_memmap(temp / "sample_frame_indices.npy", mode="w+", dtype=np.int32, shape=(len(ids), t))
        sample_times = np.lib.format.open_memmap(temp / "sample_timestamps.npy", mode="w+", dtype=np.float64, shape=(len(ids), t))
        audit_offsets[0] = 0
        index_rows = []
        audit_cursor = 0
        for feature_i, cid in enumerate(ids):
            pose, valid, selected, counts, source_timeline = _raw_pose(source["raw_root"] / f"{cid}.npz", cid)
            if len(pose) != frame_counts[feature_i] or len(pose) != int(source["manifest"][cid].get("frame_count") or 0):
                raise ValueError(f"raw frame count changed while preparing clip {cid}")
            sample = _sample_indices(len(pose), t)
            row = np.zeros((t, FEATURES), dtype=np.float32)
            row[:, :42] = pose[sample].reshape(t, 42)
            row[:, 42] = valid[sample].astype(np.float32)
            matrix[feature_i] = row
            end = audit_cursor + len(pose)
            audit_frames_path[audit_cursor:end] = source_timeline[:, 0].astype(np.int32)
            audit_selected[audit_cursor:end] = selected
            audit_counts[audit_cursor:end] = counts
            audit_ambiguous[audit_cursor:end] = counts > 1
            audit_offsets[feature_i + 1] = end
            audit_cursor = end
            sample_frames[feature_i] = source_timeline[sample, 0].astype(np.int32)
            sample_times[feature_i] = source_timeline[sample, 1]
            frame_count = len(valid)
            valid_count = int(valid.sum())
            index_rows.append({
                "clip_id": cid,
                "feature_index": feature_i,
                "class_label": source["manifest"][cid]["class_label"],
                "official_split": source["manifest"][cid]["official_split"],
                "compatibility_decision": source["review"][source["manifest"][cid]["class_label"]]["compatibility_decision"],
                "compatibility_reason": source["review"][source["manifest"][cid]["class_label"]].get("reason", ""),
                "fold": source["fold_by_id"].get(cid, ""),
                "source_frame_count": frame_count,
                "detected_candidate_frames": int((counts > 0).sum()),
                "valid_pose_frames": valid_count,
                "multi_hand_frames": int((counts > 1).sum()),
                "detection_coverage": (valid_count / frame_count) if frame_count else 0.0,
                "all_missing": str(valid_count == 0).lower(),
                "audit_offset": int(audit_offsets[feature_i]),
            })
        for array in (matrix, audit_offsets, audit_frames_path, audit_selected, audit_counts, audit_ambiguous, sample_frames, sample_times):
            array.flush()
            array._mmap.close()
        del array, matrix, audit_offsets, audit_frames_path, audit_selected, audit_counts, audit_ambiguous, sample_frames, sample_times
        with (temp / "clip_index.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = list(index_rows[0])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(index_rows)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {
            "schema_version": 1,
            "identity": source["identity"],
            "identity_inputs": source["identity_input"],
            "preprocessing": PREPROCESSING,
            "sequence_length": t,
            "features": FEATURES,
            "feature_order": "21 joints in official order; each x,y; detection-validity flag last",
            "class_order": source["retained_classes"],
            "official_test_used": False,
            "clip_count": len(ids),
            "audit_frame_count": audit_frames,
        }
        (temp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        files = [temp / name for name in (
            "features.npy", "clip_index.csv", "audit_offsets.npy", "audit_frame_indices.npy", "selected_candidate_indices.npy", "candidate_counts.npy", "ambiguous_frames.npy", "sample_frame_indices.npy", "sample_timestamps.npy"
        )]
        completion = {"status": "complete", "identity": source["identity"], "sha256": {p.name: _sha256(p) for p in files}}
        (temp / "complete.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
        validated = validate_feature_cache(temp, source["identity"])
        validated["features"]._mmap.close()
        try:
            temp.rename(final)
        except FileExistsError:
            validate_feature_cache(final, source["identity"])
            shutil.rmtree(temp)
        return final
    except BaseException:
        for name in ("matrix", "audit_offsets", "audit_frames_path", "audit_selected", "audit_counts", "audit_ambiguous", "sample_frames", "sample_times", "array"):
            mapped = locals().get(name)
            try:
                mapped._mmap.close()
            except (AttributeError, OSError):
                pass
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise
