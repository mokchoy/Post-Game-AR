"""Small, explicit Jester dataset preparation pipeline. No training code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
ARCHIVE_NAMES = ("20bnjester-v1-00", "20bnjester-v1-01", "20bnjester-v1-02")
SPLITS = ("train", "validation", "test")
BACKGROUNDS = {"No gesture", "Doing other things"}
JOINTS = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cloud_root", "local_cache_root", "cache_limit_gb"):
        if key not in config:
            raise ValueError(f"config is missing {key!r}")
    if float(config["cache_limit_gb"]) <= 0:
        raise ValueError("cache_limit_gb must be greater than zero")
    config["_config_path"] = str(path.resolve())
    config["_cloud_root"] = Path(config["cloud_root"])
    local = Path(config["local_cache_root"])
    config["_cache_root"] = (path.parent / local).resolve() if not local.is_absolute() else local
    config["_labels_dir"] = Path(config.get("labels_dir", "jester_labels"))
    if not config["_labels_dir"].is_absolute():
        config["_labels_dir"] = config["_cloud_root"] / config["_labels_dir"]
    return config


def cache_paths(config: dict) -> dict[str, Path]:
    root = config["_cache_root"]
    return {
        "root": root,
        "archives": root / "archives",
        "frames": root / "frames",
        "metadata": root / "metadata",
        "fold_views": root / "metadata" / "folds",
        "landmarks": root / "landmarks",
        "reports": root / "reports",
        "backups": root / "backups",
    }


def _cache_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
        for name in files:
            path = Path(base) / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def ensure_capacity(root: Path, limit_gb: float, extra_bytes: int, free_bytes: int | None = None, *, used_bytes: int | None = None) -> None:
    used = _cache_bytes(root) if used_bytes is None else used_bytes
    limit = int(float(limit_gb) * 1024**3)
    available = _free_bytes(root) if free_bytes is None else free_bytes
    if used + extra_bytes > limit:
        raise RuntimeError(
            f"managed cache limit would be exceeded: {used + extra_bytes:,} > {limit:,} bytes"
        )
    if available < extra_bytes:
        raise RuntimeError(f"insufficient free disk space: need {extra_bytes:,}, have {available:,} bytes")


def claim_cache(root: Path) -> None:
    marker = root / ".jester_pipeline_owner.json"
    if root.exists() and not marker.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"cache root contains existing files but is not marked as pipeline-owned: {root}; "
            "choose an empty local_cache_root"
        )
    root.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        owner = json.loads(marker.read_text(encoding="utf-8"))
        if owner.get("owner") != "jester-data-pipeline":
            raise RuntimeError(f"cache ownership marker is incompatible: {marker}")
    else:
        write_json(marker, {"owner": "jester-data-pipeline", "created_utc": utc_now()})


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_key(value: str) -> tuple[int, str]:
    return int(value), value


def _rank(seed: int, purpose: str, label: str, clip_id: str) -> str:
    value = f"{seed}\0{purpose}\0{label}\0{clip_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _read_rows(path: Path, delimiter: str) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream, delimiter=delimiter))


def read_labels(labels_dir: Path) -> tuple[list[str], dict[str, dict]]:
    label_rows = _read_rows(labels_dir / "labels.csv", ";")
    if not label_rows or any(len(row) != 1 or not row[0].strip() for row in label_rows):
        raise ValueError("labels.csv must contain one class name per row and no header")
    classes = [row[0].strip() for row in label_rows]
    if len(classes) != 27 or len(classes) != len(set(classes)):
        raise ValueError("labels.csv must contain the 27 unique official Jester classes")

    records: dict[str, dict] = {}
    split_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = _read_rows(labels_dir / f"{split}.csv", ";")
        ids: set[str] = set()
        for line, row in enumerate(rows, 1):
            expected_fields = 1 if split == "test" else 2
            if len(row) != expected_fields:
                raise ValueError(f"{split}.csv row {line}: expected {expected_fields} field(s), got {len(row)}")
            clip_id = row[0]
            if not clip_id.isdecimal():
                raise ValueError(f"{split}.csv row {line}: nonnumeric clip ID {clip_id!r}")
            if clip_id in ids:
                raise ValueError(f"{split}.csv row {line}: duplicate clip ID {clip_id}")
            ids.add(clip_id)
            label = row[1] if split != "test" else ""
            if split != "test" and label not in classes:
                raise ValueError(f"{split}.csv row {line}: unknown class {label!r}")
            records[clip_id] = {
                "clip_id": clip_id,
                "class_label": label,
                "official_split": split,
                "test_label_status": "unlabeled" if split == "test" else "labeled",
            }
        split_ids[split] = ids

    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_ids[first] & split_ids[second]
        if overlap:
            raise ValueError(f"official splits overlap between {first} and {second}: {sorted(overlap, key=_numeric_key)[:5]}")
    train_classes = {r["class_label"] for r in records.values() if r["official_split"] == "train"}
    val_classes = {r["class_label"] for r in records.values() if r["official_split"] == "validation"}
    if train_classes != set(classes) or val_classes != set(classes):
        raise ValueError("train.csv and validation.csv must each contain all 27 official classes")
    return classes, records


def validate_labels(labels_dir: Path) -> dict:
    classes, records = read_labels(labels_dir)
    summary = {
        "classes": len(classes),
        "clips_by_split": {
            split: sum(row["official_split"] == split for row in records.values()) for split in SPLITS
        },
        "train_clips_by_class": {
            label: sum(
                row["official_split"] == "train" and row["class_label"] == label
                for row in records.values()
            )
            for label in classes
        },
        "test_answers_read": False,
    }
    return summary


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict], preserve_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        if preserve_existing and path.exists():
            if sha256_file(Path(temp_name)) != sha256_file(path):
                raise FileExistsError(f"generated output differs from existing file; preserved it: {path}")
            os.unlink(temp_name)
        else:
            os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def select_pilot(labels_dir: Path, output: Path, per_class: int = 500, seed: int = 42, review_path: Path | None = None) -> dict:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    classes, records = read_labels(labels_dir)
    review = read_class_review(classes, review_path)
    classes = [label for label in classes if review[label]["compatibility_decision"] != "exclude"]
    by_class: dict[str, list[str]] = defaultdict(list)
    for clip_id, row in records.items():
        if row["official_split"] == "train":
            by_class[row["class_label"]].append(clip_id)
    shortfalls = {label: len(by_class[label]) for label in classes if len(by_class[label]) < per_class}
    if shortfalls:
        summary = {
            "status": "insufficient_training_examples",
            "seed": seed,
            "requested_per_class": per_class,
            "expected_total": per_class * len(classes),
            "insufficient_classes": shortfalls,
            "created_utc": utc_now(),
        }
        write_json(output.with_suffix(".report.json"), summary)
        raise RuntimeError(f"pilot not written: insufficient classes {shortfalls}")
    rows = []
    for label in classes:
        selected = sorted(by_class[label], key=lambda cid: (_rank(seed, "pilot", label, cid), _numeric_key(cid)))[:per_class]
        rows.extend({
            "clip_id": clip_id,
            "class_label": label,
            "official_split": "train",
            "pilot_member": "true",
            "seed": seed,
            "requested_per_class": per_class,
        } for clip_id in selected)
    if len(rows) != per_class * len(classes) or len({r["clip_id"] for r in rows}) != len(rows):
        raise AssertionError("pilot selection count or uniqueness invariant failed")
    _write_csv(output, ("clip_id", "class_label", "official_split", "pilot_member", "seed", "requested_per_class"), rows, preserve_existing=True)
    summary = {
        "status": "complete",
        "seed": seed,
        "requested_per_class": per_class,
        "classes": len(classes),
        "selected_clips": len(rows),
        "official_split": "train",
        "created_utc": utc_now(),
    }
    write_json(output.with_suffix(".report.json"), summary)
    return summary


def generate_folds(labels_dir: Path, output: Path, views_dir: Path, seed: int = 42) -> dict:
    classes, records = read_labels(labels_dir)
    by_class: dict[str, list[str]] = defaultdict(list)
    for clip_id, row in records.items():
        if row["official_split"] == "train":
            by_class[row["class_label"]].append(clip_id)
    assignments: dict[str, int] = {}
    rows = []
    for label in classes:
        ranked = sorted(by_class[label], key=lambda cid: (_rank(seed, "fold", label, cid), _numeric_key(cid)))
        for index, clip_id in enumerate(ranked):
            fold = index % 3 + 1
            assignments[clip_id] = fold
            rows.append({
                "clip_id": clip_id,
                "class_label": label,
                "official_split": "train",
                "fold": fold,
                "seed": seed,
            })
    if len(assignments) != sum(r["official_split"] == "train" for r in records.values()):
        raise AssertionError("fold assignment did not cover every official training clip")
    _write_csv(output, ("clip_id", "class_label", "official_split", "fold", "seed"), rows, preserve_existing=True)
    views_dir.mkdir(parents=True, exist_ok=True)
    for held_out in (1, 2, 3):
        train_rows = [r | {"role": "train"} for r in rows if r["fold"] != held_out]
        dev_rows = [r | {"role": "development"} for r in rows if r["fold"] == held_out]
        _write_csv(views_dir / f"fold_{held_out}_train.csv", ("clip_id", "class_label", "official_split", "fold", "seed", "role"), train_rows, preserve_existing=True)
        _write_csv(views_dir / f"fold_{held_out}_development.csv", ("clip_id", "class_label", "official_split", "fold", "seed", "role"), dev_rows, preserve_existing=True)
        if {r["clip_id"] for r in train_rows} & {r["clip_id"] for r in dev_rows}:
            raise AssertionError(f"fold {held_out} has train/development leakage")
        if len(train_rows) + len(dev_rows) != len(rows):
            raise AssertionError(f"fold {held_out} does not cover the training set")
    return {
        "seed": seed,
        "folds": 3,
        "training_clips": len(rows),
        "development_role": "one held-out fold",
        "official_validation_included": False,
        "official_test_included": False,
        "created_utc": utc_now(),
    }


def create_review_template(labels_dir: Path, output: Path) -> None:
    classes, _ = read_labels(labels_dir)
    rows = []
    for label in classes:
        required = label in BACKGROUNDS
        rows.append({
            "class_label": label,
            "compatibility_decision": "retain" if required else "pending",
            "reason": "required background class" if required else "",
            "instability_notes": "",
            "possible_hand_switching_notes": "",
            "reviewer_notes": "",
        })
    _write_csv(
        output,
        ("class_label", "compatibility_decision", "reason", "instability_notes", "possible_hand_switching_notes", "reviewer_notes"),
        rows,
        preserve_existing=True,
    )


def read_clip_ids(path: Path, records: dict[str, dict]) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [(row.get("clip_id") or "").strip() for row in rows]
    if not ids or any(not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("clip ID selection must contain unique, nonempty clip_id values")
    unknown = set(ids) - records.keys()
    if unknown:
        raise ValueError(f"selection contains unknown clip IDs: {sorted(unknown)[:5]}")
    for cid, row in zip(ids, rows):
        if any(row.get(key) and row[key] != records[cid].get(key) for key in ("class_label", "official_split")):
            raise ValueError(f"selection has mismatched split/class metadata: {cid}")
    return ids


def filter_preparation_ids(ids: list[str], records: dict[str, dict], review: dict[str, dict]) -> list[str]:
    excluded = {label for label, row in review.items() if row["compatibility_decision"] == "exclude"}
    if excluded and any(not records[cid]["class_label"] for cid in ids):
        raise ValueError("unlabeled test clips cannot be filtered by gesture; use labeled train/validation IDs")
    selected = [cid for cid in ids if records[cid]["class_label"] not in excluded]
    if not selected:
        raise ValueError("selection contains no non-excluded clip IDs")
    return selected


def read_class_review(classes: list[str], path: Path | None) -> dict[str, dict]:
    defaults = {
        label: {
            "compatibility_decision": "retain" if label in BACKGROUNDS else "pending",
            "reason": "required background class" if label in BACKGROUNDS else "",
            "instability_notes": "",
            "possible_hand_switching_notes": "",
            "reviewer_notes": "",
        }
        for label in classes
    }
    if path is None or not path.exists():
        return defaults
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_label = {row.get("class_label", ""): row for row in rows}
    if set(by_label) != set(classes) or len(rows) != len(classes):
        raise ValueError("class review must contain exactly one row for every official class")
    valid = {"pending", "retain", "flag", "exclude"}
    result = {}
    for label in classes:
        row = by_label[label]
        decision = row.get("compatibility_decision", "").strip()
        if decision not in valid:
            raise ValueError(f"class review has invalid decision {decision!r} for {label!r}")
        reason = row.get("reason", "").strip()
        if decision == "exclude" and not reason:
            raise ValueError(f"excluded class {label!r} needs a reason")
        if label in BACKGROUNDS and decision != "retain":
            raise ValueError(f"{label!r} is a required background class and must be retained")
        result[label] = {
            "compatibility_decision": decision,
            "reason": reason,
            "instability_notes": row.get("instability_notes", "").strip(),
            "possible_hand_switching_notes": row.get("possible_hand_switching_notes", "").strip(),
            "reviewer_notes": row.get("reviewer_notes", "").strip(),
        }
    return result


_FRAME_NAME = re.compile(r"^([0-9]+)\.jpg$", re.IGNORECASE)


def scan_frame_dir(clip_dir: Path, decode: bool = True) -> dict:
    import cv2
    import numpy as np

    by_index: dict[int, list[Path]] = defaultdict(list)
    invalid_names = []
    for path in clip_dir.iterdir():
        if not path.is_file() or path.name == "complete.json":
            continue
        match = _FRAME_NAME.fullmatch(path.name)
        if match:
            by_index[int(match.group(1))].append(path)
        else:
            invalid_names.append(path.name)
    duplicates = sorted(index for index, paths in by_index.items() if len(paths) > 1)
    invalid_indices = sorted(index for index in by_index if index < 1)
    max_index = max(by_index, default=0)
    entries = []
    missing = []
    corrupt = []
    heights = set()
    for index in range(1, max_index + 1):
        paths = by_index.get(index, [])
        if not paths:
            missing.append(index)
            entries.append({"index": index, "filename": "", "path": None, "valid": False, "width": 0, "height": 0})
            continue
        path = paths[0]
        width = height = 0
        valid = True
        if decode:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            valid = image is not None
            if valid:
                height, width = image.shape[:2]
                heights.add(height)
        if not valid:
            corrupt.append(index)
        entries.append({
            "index": index,
            "filename": path.name,
            "path": path,
            "valid": valid,
            "width": width,
            "height": height,
        })
    issues = []
    if not by_index:
        issues.append("no_frames")
    if missing:
        issues.append("missing_frames")
    if corrupt:
        issues.append("corrupt_frames")
    if duplicates:
        issues.append("duplicate_frame_indices")
    if invalid_indices:
        issues.append("invalid_frame_indices")
    if invalid_names:
        issues.append("invalid_frame_names")
    if any(height != 100 for height in heights):
        issues.append("unexpected_height")
    return {
        "entries": entries,
        "frame_count": max_index,
        "valid_frame_count": sum(entry["valid"] for entry in entries),
        "missing": missing,
        "corrupt": corrupt,
        "duplicates": duplicates,
        "invalid_indices": invalid_indices,
        "invalid_names": sorted(invalid_names),
        "heights": sorted(heights),
        "integrity_status": "ok" if not issues else ";".join(issues),
    }


def frame_set_hash(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        if not entry["valid"] or entry["path"] is None:
            raise ValueError("cannot hash an incomplete frame sequence")
        path = entry["path"]
        frame_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{entry['index']}\0{path.name}\0{size}\0{frame_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def validate_frame_completion(clip_dir: Path, archive_identity: str | None = None, expected_count: int | None = None) -> tuple[bool, dict, dict]:
    marker_path = clip_dir / "complete.json"
    if not marker_path.is_file():
        return False, {}, {"integrity_status": "partial"}
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        scan = scan_frame_dir(clip_dir)
        if (
            scan["frame_count"] < 1
            or scan["missing"]
            or scan["corrupt"]
            or scan["duplicates"]
            or scan["invalid_indices"]
            or scan["invalid_names"]
            or scan["valid_frame_count"] != marker.get("frame_count")
        ):
            return False, marker, scan
        if archive_identity is not None and marker.get("archive_identity") != archive_identity:
            return False, marker, scan
        if expected_count is not None and marker.get("frame_count") != expected_count:
            return False, marker, scan
        if frame_set_hash(scan["entries"]) != marker.get("frame_set_sha256"):
            return False, marker, scan
        return True, marker, scan
    except (OSError, ValueError, json.JSONDecodeError):
        return False, {}, {"integrity_status": "invalid_completion_marker"}


def build_manifest(labels_dir: Path, frames_root: Path, output: Path, folds_path: Path | None, pilot_path: Path | None, review_path: Path | None, ids_path: Path | None = None) -> dict:
    classes, records = read_labels(labels_dir)
    review = read_class_review(classes, review_path)
    selected = set(filter_preparation_ids(read_clip_ids(ids_path, records), records, review)) if ids_path is not None else None
    previous = {}
    if selected is not None and output.exists():
        with output.open("r", encoding="utf-8-sig", newline="") as stream:
            previous = {row["clip_id"]: row for row in csv.DictReader(stream)}
    folds = {}
    if folds_path and folds_path.exists():
        with folds_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("official_split") != "train" or row.get("fold") not in {"1", "2", "3"}:
                    raise ValueError(f"invalid fold assignment row: {row}")
                if row["clip_id"] not in records or records[row["clip_id"]]["official_split"] != "train" or row["class_label"] != records[row["clip_id"]]["class_label"] or row["clip_id"] in folds:
                    raise ValueError(f"duplicate, unknown, or mismatched fold assignment: {row}")
                folds[row["clip_id"]] = row["fold"]
        expected_train = {cid for cid, row in records.items() if row["official_split"] == "train"}
        if set(folds) != expected_train:
            raise ValueError("fold assignment file must cover every official training clip exactly once")
    pilot = set()
    if pilot_path and pilot_path.exists():
        with pilot_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("official_split") != "train":
                    raise ValueError("pilot membership must contain official training clips only")
                if row.get("clip_id") not in records or records[row["clip_id"]]["official_split"] != "train" or row.get("class_label") != records[row["clip_id"]]["class_label"]:
                    raise ValueError(f"unknown or mismatched pilot member: {row}")
                if row["clip_id"] in pilot:
                    raise ValueError(f"duplicate pilot clip ID: {row['clip_id']}")
                pilot.add(row["clip_id"])

    fields = (
        "clip_id", "class_label", "official_split", "test_label_status", "frame_dir", "frame_count", "valid_frame_count",
        "missing_frame_indices", "corrupt_frame_indices", "integrity_status", "extraction_status",
        "duplicate_frame_indices", "invalid_frame_indices", "invalid_frame_names",
        "cv_fold", "pilot_member", "compatibility_decision", "compatibility_reason",
    )
    rows = []
    for clip_id in sorted(records, key=_numeric_key):
        record = records[clip_id]
        clip_dir = frames_root / clip_id
        if selected is not None and clip_id not in selected and clip_id in previous:
            row = previous[clip_id] | record
        elif (selected is None or clip_id in selected) and clip_dir.is_dir():
            complete, marker, scan = validate_frame_completion(clip_dir)
            status = "complete" if complete else "partial"
            rel = os.path.relpath(clip_dir, config_root_for(frames_root))
            row = {
                **record,
                "frame_dir": rel,
                "frame_count": marker.get("frame_count", scan.get("frame_count", 0)),
                "valid_frame_count": scan.get("valid_frame_count", 0),
                "missing_frame_indices": ";".join(map(str, scan.get("missing", []))),
                "corrupt_frame_indices": ";".join(map(str, scan.get("corrupt", []))),
                "integrity_status": scan.get("integrity_status", "unknown"),
                "extraction_status": status,
                "duplicate_frame_indices": ";".join(map(str, scan.get("duplicates", []))),
                "invalid_frame_indices": ";".join(map(str, scan.get("invalid_indices", []))),
                "invalid_frame_names": ";".join(scan.get("invalid_names", [])),
            }
        else:
            row = {
                **record,
                "frame_dir": os.path.relpath(clip_dir, config_root_for(frames_root)),
                "frame_count": 0,
                "valid_frame_count": 0,
                "missing_frame_indices": "",
                "corrupt_frame_indices": "",
                "integrity_status": "not_extracted",
                "extraction_status": "not_extracted",
                "duplicate_frame_indices": "",
                "invalid_frame_indices": "",
                "invalid_frame_names": "",
            }
        row.update({
            "cv_fold": folds.get(clip_id, ""),
            "pilot_member": "true" if clip_id in pilot else "false",
            "compatibility_decision": review[record["class_label"]]["compatibility_decision"] if record["class_label"] else "not_applicable",
            "compatibility_reason": review[record["class_label"]]["reason"] if record["class_label"] else "unlabeled official test split",
        })
        rows.append(row)
    _write_csv(output, fields, rows)
    return {"clips": len(rows), "classes": len(classes), "created_utc": utc_now()}


def config_root_for(frames_root: Path) -> Path:
    # Manifest paths stay relative to the configured cache root.
    return frames_root.parent


class ConcatenatedReader:
    """Read multipart gzip/tar parts as one stream without building a second archive."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.index = 0
        self.current = None
        self.closed = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            return b""
        if size is None or size < 0:
            chunks = []
            while True:
                block = self.read(4 * 1024 * 1024)
                if not block:
                    return b"".join(chunks)
                chunks.append(block)
        result = bytearray()
        while len(result) < size:
            if self.current is None:
                if self.index >= len(self.paths):
                    break
                self.current = self.paths[self.index].open("rb")
                self.index += 1
            block = self.current.read(size - len(result))
            if block:
                result.extend(block)
            else:
                self.current.close()
                self.current = None
        return bytes(result)

    def close(self) -> None:
        self.closed = True
        if self.current is not None:
            self.current.close()
            self.current = None


def _tar_frame(member: tarfile.TarInfo) -> tuple[str, int, str] | None:
    if not member.isfile():
        return None
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {member.name!r}")
    if len(path.parts) < 2:
        return None
    clip_part, frame_name = path.parts[-2], path.parts[-1]
    match = _FRAME_NAME.fullmatch(frame_name)
    if not clip_part.isdecimal() or not match:
        return None
    return clip_part, int(match.group(1)), frame_name


def _open_tar(parts: list[Path]):
    reader = ConcatenatedReader(parts)
    try:
        archive = tarfile.open(fileobj=reader, mode="r|gz")
        return reader, archive
    except BaseException:
        reader.close()
        raise


def _archive_stage_identity(config: dict) -> tuple[list[Path], dict, str]:
    paths = cache_paths(config)
    report_path = paths["metadata"] / "stage_report.json"
    if not report_path.is_file():
        raise RuntimeError("local staged archives are missing; run the explicit stage command first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parts = []
    identities = []
    for name in ARCHIVE_NAMES:
        item = report.get("parts", {}).get(name)
        path = paths["archives"] / name
        if not item or not path.is_file():
            raise RuntimeError(f"staged archive part is missing: {name}")
        stat = path.stat()
        if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise RuntimeError(f"staged archive changed after hashing; restage {name}")
        parts.append(path)
        identities.append({"name": name, "size_bytes": stat.st_size, "sha256": item["sha256"]})
    archive_identity = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return parts, report, archive_identity


def _archive_stats(parts: list[Path], selected: set[str]) -> dict[str, dict]:
    counts: dict[str, dict] = {}
    reader, archive = _open_tar(parts)
    try:
        for member in archive:
            item = _tar_frame(member)
            if item is None:
                continue
            clip_id, frame_index, _ = item
            if clip_id not in selected:
                continue
            stats = counts.setdefault(clip_id, {"frame_count": 0, "max_index": 0, "bytes": 0, "indices": []})
            stats["frame_count"] += 1
            stats["max_index"] = max(stats["max_index"], frame_index)
            stats["bytes"] += member.size
            stats["indices"].append(frame_index)
    finally:
        archive.close()
        reader.close()
    return counts


def _staging_root(frames_root: Path) -> Path:
    root = frames_root / ".staging"
    if root.resolve().parent != frames_root.resolve():
        raise RuntimeError(f"staging directory escapes the frame cache: {root}")
    owner = root / ".jester_pipeline_owner.json"
    if root.exists() and not owner.is_file():
        raise RuntimeError(f"refusing to clean unmarked staging directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not owner.exists():
        write_json(owner, {"owner": "jester-data-pipeline"})
    return root


def _promote_frame_clip(partial: Path, final: Path) -> None:
    for attempt in range(4):
        if final.exists():
            raise FileExistsError(f"output path appeared during extraction; preserved existing output: {final}")
        try:
            os.rename(partial, final)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 3:
                raise
            time.sleep(0.25 * (attempt + 1))


def extract_frames(config: dict, labels_dir: Path, pilot_path: Path | None, extract_all: bool, ids_path: Path | None = None) -> dict:
    if sum((pilot_path is not None, bool(extract_all), ids_path is not None)) != 1:
        raise ValueError("choose exactly one of --pilot, --ids-file, or --all")
    classes, records = read_labels(labels_dir)
    paths = cache_paths(config)
    review = read_class_review(classes, paths["metadata"] / "class_review.csv")
    active_classes = [label for label in classes if review[label]["compatibility_decision"] != "exclude"]
    if extract_all:
        if len(active_classes) != len(classes):
            raise ValueError("--all is unavailable with excluded classes; use --pilot or --ids-file with labeled train/validation IDs")
        selected = set(records)
    elif pilot_path is not None:
        if not pilot_path or not pilot_path.is_file():
            raise FileNotFoundError(f"pilot selection is required before frame extraction: {pilot_path}")
        selected_rows = []
        requested_counts = set()
        with pilot_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    requested_counts.add(int(row.get("requested_per_class", "")))
                except ValueError:
                    raise ValueError("pilot rows must declare an integer requested_per_class") from None
                if row.get("official_split") != "train":
                    raise ValueError("pilot frame extraction accepts training clips only")
                clip_id = row.get("clip_id", "")
                if clip_id not in records or records[clip_id]["official_split"] != "train" or row.get("class_label") != records[clip_id]["class_label"]:
                    raise ValueError(f"pilot row has an unknown clip or wrong class label: {row}")
                selected_rows.append(clip_id)
        if len(selected_rows) != len(set(selected_rows)):
            raise ValueError("pilot selection contains duplicate clip IDs")
        selected_rows = filter_preparation_ids(selected_rows, records, review)
        counts = {label: sum(records[cid]["class_label"] == label for cid in selected_rows) for label in active_classes}
        if len(requested_counts) != 1 or next(iter(requested_counts)) < 1:
            raise ValueError("pilot rows must declare one positive requested_per_class")
        per_class = next(iter(requested_counts))
        if any(count != per_class for count in counts.values()):
            raise ValueError(f"pilot selection must contain exactly {per_class} clips per non-excluded class: {counts}")
        selected = set(selected_rows)
        if not selected:
            raise ValueError("pilot selection contains no clip IDs")
    else:
        if not ids_path or not ids_path.is_file():
            raise FileNotFoundError(f"clip ID selection not found: {ids_path}")
        with ids_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        selected_rows = [row.get("clip_id", "").strip() for row in rows]
        if not selected_rows or any(not cid for cid in selected_rows) or len(selected_rows) != len(set(selected_rows)):
            raise ValueError("clip ID selection must contain unique clip_id values")
        unknown = set(selected_rows) - records.keys()
        mismatched = [
            row for row in rows if row.get("clip_id") in records
            and ((row.get("official_split") and row["official_split"] != records[row["clip_id"]]["official_split"])
                 or (row.get("class_label") and row["class_label"] != records[row["clip_id"]]["class_label"]))
        ]
        if unknown or mismatched:
            raise ValueError(f"selection has unknown IDs or mismatched split/class metadata: unknown={sorted(unknown)[:5]}, mismatched={mismatched[:3]}")
        selected = set(filter_preparation_ids(selected_rows, records, review))
    unknown = selected - records.keys()
    if unknown:
        raise ValueError(f"selection contains unknown clip IDs: {sorted(unknown, key=_numeric_key)[:5]}")
    if pilot_path is not None and any(records[cid]["official_split"] != "train" for cid in selected):
        raise ValueError("pilot selection includes non-training data")

    parts, stage_report, archive_identity = _archive_stage_identity(config)
    stats = _archive_stats(parts, selected)
    missing = selected - stats.keys()
    if missing:
        failures = [{"clip_id": cid, "reason": "clip ID has no JPEG frames in the archive"} for cid in sorted(missing, key=_numeric_key)]
        failure_path = paths["metadata"] / "frame_extraction_failures.csv"
        ensure_capacity(paths["root"], config["cache_limit_gb"], len(failures) * 256 + 4096)
        claim_cache(paths["root"])
        _write_csv(failure_path, ("clip_id", "reason"), failures)
        raise RuntimeError(f"{len(failures)} selected clip ID(s) are absent from the archive; see {failure_path}")

    completed = set()
    ready = set()
    required_bytes = 0
    failures = []
    archive_invalid = set()
    frames_root = paths["frames"]
    # Check staging ownership before trusting or modifying any staged clip.
    staging = frames_root / ".staging"
    if staging.exists() and not (staging / ".jester_pipeline_owner.json").is_file():
        raise RuntimeError(f"refusing to reuse unmarked staging directory: {staging}")
    for clip_id in selected:
        final = paths["frames"] / clip_id
        if final.exists():
            valid, _, _ = validate_frame_completion(final, archive_identity, stats[clip_id]["frame_count"])
            if not valid:
                raise RuntimeError(f"existing frame output is incomplete or incompatible and was preserved: {final}")
            completed.add(clip_id)
        else:
            expected = stats[clip_id]
            expected_indices = sorted(expected["indices"])
            if expected_indices != list(range(1, expected["max_index"] + 1)):
                missing_indices = sorted(set(range(1, expected["max_index"] + 1)) - set(expected_indices))
                duplicate_indices = sorted(index for index in set(expected_indices) if expected_indices.count(index) > 1)
                failures.append({"clip_id": clip_id, "reason": f"archive frame indices missing={missing_indices[:50]} duplicate={duplicate_indices[:50]}"})
                archive_invalid.add(clip_id)
            else:
                partial = staging / clip_id
                valid, _, _ = validate_frame_completion(partial, archive_identity, expected["frame_count"])
                if valid:
                    ready.add(clip_id)
                else:
                    required_bytes += expected["bytes"]
    pending = selected - completed - archive_invalid - ready
    ensure_capacity(paths["root"], config["cache_limit_gb"], required_bytes + 4096 * max(1, len(pending)))
    claim_cache(paths["root"])
    frames_root.mkdir(parents=True, exist_ok=True)
    staging = _staging_root(frames_root)
    for clip_id in pending | ready:
        partial = staging / clip_id
        if clip_id in ready:
            try:
                _promote_frame_clip(partial, frames_root / clip_id)
            except OSError as exc:
                failures.append({"clip_id": clip_id, "reason": f"staged clip preserved; promotion failed: {type(exc).__name__}: {exc}"})
            continue
        if partial.resolve().parent != staging.resolve() or partial.is_symlink():
            raise RuntimeError(f"staged clip escapes the staging directory: {partial}")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()

    started = time.perf_counter()
    if pending:
        reader, archive = _open_tar(parts)
        try:
            for member in archive:
                item = _tar_frame(member)
                if item is None:
                    continue
                clip_id, frame_index, frame_name = item
                if clip_id not in pending:
                    continue
                target = staging / clip_id / frame_name
                source = archive.extractfile(member)
                if source is None:
                    failures.append({"clip_id": clip_id, "reason": f"archive member could not be read: {member.name}"})
                    continue
                try:
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                except Exception as exc:
                    failures.append({"clip_id": clip_id, "reason": f"{member.name}: {type(exc).__name__}: {exc}"})
        except Exception as exc:
            failures.append({"clip_id": "", "reason": f"archive stream failed: {type(exc).__name__}: {exc}"})
        finally:
            archive.close()
            reader.close()
    elapsed = time.perf_counter() - started

    for clip_id in pending:
        partial = staging / clip_id
        expected_count = stats[clip_id]["frame_count"]
        if any(row["clip_id"] == clip_id for row in failures):
            continue
        if not partial.is_dir():
            failures.append({"clip_id": clip_id, "reason": "no extracted frame directory"})
            continue
        scan = scan_frame_dir(partial)
        expected_indices = list(range(1, expected_count + 1))
        actual_indices = [entry["index"] for entry in scan["entries"] if entry["valid"]]
        if (
            actual_indices != expected_indices
            or scan["missing"]
            or scan["corrupt"]
            or scan["duplicates"]
            or scan["invalid_indices"]
            or scan["invalid_names"]
        ):
            failures.append({
                "clip_id": clip_id,
                "reason": f"frame validation failed: {scan['integrity_status']}; missing={scan['missing'][:50]} corrupt={scan['corrupt'][:50]} duplicate={scan['duplicates'][:50]}; expected {expected_count} contiguous frames",
            })
            continue
        marker = {
            "status": "complete",
            "archive_identity": archive_identity,
            "frame_count": expected_count,
            "frame_rate_fps": 12,
            "expected_height_px": 100,
            "height_warnings": [e["height"] for e in scan["entries"] if e["height"] != 100],
            "frame_set_sha256": frame_set_hash(scan["entries"]),
            "created_utc": utc_now(),
        }
        write_json(partial / "complete.json", marker)
        final = frames_root / clip_id
        try:
            _promote_frame_clip(partial, final)
        except OSError as exc:
            failures.append({"clip_id": clip_id, "reason": f"staged clip preserved; promotion failed: {type(exc).__name__}: {exc}"})

    failure_path = paths["metadata"] / "frame_extraction_failures.csv"
    _write_csv(failure_path, ("clip_id", "reason"), failures)
    if failures:
        raise RuntimeError(f"{len(failures)} frame extraction failure(s); see {failure_path}")
    return {
        "selected_clips": len(selected),
        "already_complete": len(completed),
        "recovered_staged": len(ready),
        "newly_extracted": len(pending) - sum(1 for row in failures if row["clip_id"] in pending),
        "frame_payload_bytes": required_bytes,
        "extraction_seconds_excludes_cloud_transfer": elapsed,
        "archive_identity": archive_identity,
        "created_utc": utc_now(),
    }


def archive_inventory(config: dict) -> dict:
    root = config["_cloud_root"]
    parts = {}
    for name in ARCHIVE_NAMES:
        path = root / name
        if not path.is_file():
            parts[name] = {"status": "missing"}
        else:
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                parts[name] = {"status": "changing_during_hash"}
            else:
                parts[name] = {
                    "status": "present",
                    "size_bytes": after.st_size,
                    "sha256": digest,
                    "mtime_ns": after.st_mtime_ns,
                }
    result = {
        "cloud_root": str(root),
        "parts": parts,
        "complete": all(item["status"] == "present" for item in parts.values()),
        "checked_utc": utc_now(),
    }
    paths = cache_paths(config)
    ensure_capacity(paths["root"], config["cache_limit_gb"], 4096)
    claim_cache(paths["root"])
    write_json(paths["metadata"] / "archive_inventory.json", result)
    return result


def stage_archives(config: dict) -> dict:
    cloud = config["_cloud_root"]
    local = cache_paths(config)
    sources = {name: cloud / name for name in ARCHIVE_NAMES}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"all official parts must be present in cloud_root before staging: {', '.join(missing)}"
        )
    current = {}
    need = 0
    for name, source in sources.items():
        target = local["archives"] / name
        source_stat = source.stat()
        current[name] = {"source": source, "target": target, "size": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns}
        if not target.is_file():
            need += source.stat().st_size
    ensure_capacity(local["root"], config["cache_limit_gb"], need + 1024 * 1024)
    claim_cache(local["root"])
    local["archives"].mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    parts = {}
    for name, item in current.items():
        source, target, size = item["source"], item["target"], item["size"]
        if target.exists():
            source_digest = sha256_file(source)
            source_after = source.stat()
            if (source_after.st_size, source_after.st_mtime_ns) != (size, item["source_mtime_ns"]):
                raise RuntimeError(f"cloud source changed while checking {name}; wait for Drive sync and retry")
            if target.stat().st_size != size or sha256_file(target) != source_digest:
                raise RuntimeError(f"staged part conflicts with cloud original; preserved both: {target}")
            digest = source_digest
        else:
            partial = target.with_name(target.name + ".partial")
            digest = hashlib.sha256()
            copied = 0
            try:
                with source.open("rb") as incoming, partial.open("wb") as outgoing:
                    for block in iter(lambda: incoming.read(4 * 1024 * 1024), b""):
                        outgoing.write(block)
                        digest.update(block)
                        copied += len(block)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                if copied != size:
                    raise IOError(f"copy size changed for {name}: expected {size}, got {copied}")
                source_after = source.stat()
                if (source_after.st_size, source_after.st_mtime_ns) != (size, item["source_mtime_ns"]):
                    raise RuntimeError(f"cloud source changed during staging {name}; local partial removed, retry after Drive sync")
                os.replace(partial, target)
            except BaseException:
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
                raise
            digest = digest.hexdigest()
        parts[name] = {
            "size_bytes": size,
            "sha256": digest,
            "cloud_source_mtime_ns": item["source_mtime_ns"],
            "mtime_ns": target.stat().st_mtime_ns,
        }
    result = {
        "cloud_root": str(cloud),
        "local_archive_root": str(local["archives"]),
        "parts": parts,
        "transfer_seconds": time.perf_counter() - started,
        "created_utc": utc_now(),
    }
    write_json(local["metadata"] / "stage_report.json", result)
    return result


def _copy_completed_bundle(config: dict, extractor: str, cloud_copy: bool) -> dict:
    paths = cache_paths(config)
    files = []
    extractors = ("rtmlib", "mediapipe") if extractor == "all" else (extractor,)
    from landmarks import _valid_cache
    import numpy as np
    for name in extractors:
        feature_root = paths["landmarks"] / name
        if not feature_root.is_dir():
            continue
        for npz_path in feature_root.glob("*.npz"):
            marker_path = feature_root / f"{npz_path.stem}.complete.json"
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                with np.load(npz_path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata_json"].item()))
                    valid = (
                        marker.get("status") == "complete"
                        and marker.get("output_sha256") == sha256_file(npz_path)
                        and marker.get("input_identity") == metadata.get("input_identity")
                        and marker.get("package_key") == metadata.get("package_key")
                        and _valid_cache(data)
                    )
                if valid:
                    files.extend((npz_path, marker_path))
            except Exception:
                continue
    if extractor == "all" and paths["frames"].is_dir():
        for clip_dir in paths["frames"].iterdir():
            if not clip_dir.is_dir() or not clip_dir.name.isdecimal():
                continue
            complete, _, _ = validate_frame_completion(clip_dir)
            if complete:
                files.extend(path for path in clip_dir.iterdir() if path.is_file() and (path.suffix.lower() == ".jpg" or path.name == "complete.json"))
    report_roots = (paths["reports"],) if extractor == "all" else (paths["reports"] / extractor,)
    for base in (paths["metadata"], *report_roots):
        if base.exists():
            files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in {".csv", ".json"})
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no completed outputs to bundle for {extractor}")
    total = sum(path.stat().st_size for path in files)
    overhead = 1024 * len(files) + 1024 * 1024
    ensure_capacity(paths["root"], config["cache_limit_gb"], total + overhead)
    claim_cache(paths["root"])
    paths["backups"].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = paths["backups"] / f"jester_{extractor}_{stamp}.zip"
    partial = bundle.with_suffix(".zip.partial")
    if bundle.exists() or partial.exists():
        raise FileExistsError(f"bundle output already exists; preserved: {bundle if bundle.exists() else partial}")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in files:
                archive.write(path, path.relative_to(paths["root"]).as_posix())
        with zipfile.ZipFile(partial, "r") as archive:
            bad = archive.testzip()
            if bad:
                raise IOError(f"bundle validation failed at {bad}")
        os.replace(partial, bundle)
    except BaseException:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    digest = sha256_file(bundle)
    result = {
        "bundle": str(bundle),
        "size_bytes": bundle.stat().st_size,
        "sha256": digest,
        "source_file_count": len(files),
        "cloud_copy_requested": cloud_copy,
        "cloud_copy_verified_by_local_readback": False,
        "created_utc": utc_now(),
    }
    if cloud_copy:
        target = config["_cloud_root"] / bundle.name
        if target.exists():
            raise FileExistsError(f"cloud bundle target already exists; preserved: {target}")
        target_partial = target.with_name(target.name + ".partial")
        if target_partial.exists():
            raise FileExistsError(f"partial cloud bundle exists; preserved: {target_partial}")
        shutil.copy2(bundle, target_partial)
        if target_partial.stat().st_size != bundle.stat().st_size or sha256_file(target_partial) != digest:
            raise IOError(f"cloud-path copy did not verify; local bundle preserved at {bundle}")
        os.replace(target_partial, target)
        result["cloud_copy"] = str(target)
        result["cloud_copy_verified_by_local_readback"] = True
        result["cloud_sync_confirmed"] = False
        result["note"] = "Drive sync confirmation is external; local bundle and source outputs were preserved."
    write_json(paths["backups"] / f"{bundle.stem}.receipt.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Jester data and landmarks; no model training.")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("validate-labels", help="validate official split and label files")
    p.add_argument("--labels-dir", type=Path)

    commands.add_parser("archive-inventory", help="record package sizes and SHA-256 hashes")
    commands.add_parser("stage-archives", help="copy original archive parts to local staging")

    p = commands.add_parser("folds", help="write three stratified training fold assignments and manifest views")
    p.add_argument("--labels-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)

    p = commands.add_parser("pilot", help="select training clips per class without replacement (default: 500)")
    p.add_argument("--labels-dir", type=Path)
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p = commands.add_parser("review-template", help="create the editable class compatibility review table")
    p.add_argument("--labels-dir", type=Path)

    p = commands.add_parser("manifest", help="write one clip manifest with split, fold, pilot, and frame status")
    p.add_argument("--labels-dir", type=Path)
    p.add_argument("--ids-file", type=Path, help="refresh frames for selected IDs only; preserve other saved frame statuses")

    p = commands.add_parser("extract-frames", help="selectively extract frames from staged multipart archives")
    p.add_argument("--labels-dir", type=Path)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--pilot", action="store_true", help="extract only IDs in metadata/pilot_ids.csv")
    choice.add_argument("--ids-file", type=Path, help="extract an explicit clip_id CSV selection")
    choice.add_argument("--all", action="store_true", help="explicitly extract every official clip")

    p = commands.add_parser("extract-landmarks", help="extract a selected clip list using one supported backend")
    p.add_argument("--extractor", choices=("rtmlib", "mediapipe"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="RTMLib extraction device; benchmark stays CPU")
    p.add_argument("--ids-file", type=Path)
    p.add_argument("--model", type=Path, help="MediaPipe Hand Landmarker .task model")
    p.add_argument("--rtm-det-model", type=Path, help="local RTMDet hand ONNX file")
    p.add_argument("--rtm-pose-model", type=Path, help="local RTMPose hand ONNX file")

    p = commands.add_parser("report", help="write coverage, failures, review table, and sample overlays")
    p.add_argument("--extractor", choices=("rtmlib", "mediapipe"), required=True)
    p.add_argument("--ids-file", type=Path)

    p = commands.add_parser("benchmark", help="CPU benchmark both extractors on the same explicit clip list")
    p.add_argument("--ids-file", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rtm-det-model", type=Path, required=True)
    p.add_argument("--rtm-pose-model", type=Path, required=True)

    commands.add_parser("compare", help="write a concise RTMLib versus MediaPipe report")

    p = commands.add_parser("bundle", help="bundle completed outputs for Drive without deleting local copies")
    p.add_argument("--extractor", choices=("rtmlib", "mediapipe", "all"), required=True)
    p.add_argument("--cloud-copy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = cache_paths(config)
        labels_dir = getattr(args, "labels_dir", None) or config["_labels_dir"]
        if args.command == "validate-labels":
            result = validate_labels(labels_dir)
        elif args.command == "archive-inventory":
            result = archive_inventory(config)
        elif args.command == "stage-archives":
            result = stage_archives(config)
        elif args.command == "folds":
            ensure_capacity(paths["root"], config["cache_limit_gb"], 32 * 1024 * 1024)
            claim_cache(paths["root"])
            result = generate_folds(
                labels_dir, paths["metadata"] / "fold_assignments.csv",
                paths["fold_views"], args.seed,
            )
        elif args.command == "pilot":
            ensure_capacity(paths["root"], config["cache_limit_gb"], 1024 * 1024)
            claim_cache(paths["root"])
            result = select_pilot(
                labels_dir, paths["metadata"] / "pilot_ids.csv", args.per_class, args.seed,
                paths["metadata"] / "class_review.csv",
            )
        elif args.command == "review-template":
            ensure_capacity(paths["root"], config["cache_limit_gb"], 1024 * 1024)
            claim_cache(paths["root"])
            review_path = paths["metadata"] / "class_review.csv"
            create_review_template(labels_dir, review_path)
            result = {"path": str(review_path), "created_utc": utc_now()}
        elif args.command == "manifest":
            ensure_capacity(paths["root"], config["cache_limit_gb"], 4 * 1024 * 1024)
            claim_cache(paths["root"])
            result = build_manifest(
                labels_dir, paths["frames"], paths["metadata"] / "clip_manifest.csv",
                paths["metadata"] / "fold_assignments.csv",
                paths["metadata"] / "pilot_ids.csv",
                paths["metadata"] / "class_review.csv",
                args.ids_file,
            )
        elif args.command == "extract-frames":
            result = extract_frames(
                config, labels_dir,
                paths["metadata"] / "pilot_ids.csv" if args.pilot else None,
                args.all,
                args.ids_file,
            )
        elif args.command in {"extract-landmarks", "report", "benchmark"}:
            from landmarks import run_command
            result = run_command(args, config)
        elif args.command == "compare":
            from landmarks import compare
            result = compare(config)
        elif args.command == "bundle":
            result = _copy_completed_bundle(config, args.extractor, args.cloud_copy)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
