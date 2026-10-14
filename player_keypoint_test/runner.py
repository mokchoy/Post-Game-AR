from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
os.environ["TEMP"] = str(ROOT / ".tmp")
os.environ["TMP"] = str(ROOT / ".tmp")
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
(ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
(ROOT / ".ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
import yaml

import point_model as pm
sys.path.insert(0, str(REPO / "monotrack_test"))
from reconstruction import (  # noqa: E402
    COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, load_court_geometry,
)
from run_projection import court_world_lines  # noqa: E402

SOURCE = REPO / "cleaned_rallies" / "set-1"
HOMOGRAPHY = REPO / "set-1" / "homography.csv"
CONFIG_DEFAULT = ROOT / "config.yaml"
WEIGHTS = ROOT / "weights" / "yolo11n.pt"
PREP = ROOT / "data" / "candidate"
SPLIT_FILE = ROOT / "splits" / "split.json"
MANIFEST_VERSION = 2
SPLIT_VERSION = 2
EXCLUDED_MATCHES = {
    "CHOU_Tien_Chen_Jonatan_CHRISTIE_Sudirman_Cup_2019_Quarter-finals",
    "CHOU_Tien_Chen_NG_Ka_Long_Angus_Sudirman_Cup_2019_Group_Stage",
}
REFERENCE_WIDTH, REFERENCE_HEIGHT = 1280, 720
COURT_HALF_Y_M = 6.7
KEY_FIELDS = ("dataset", "match", "game", "original_rally_id")
REVIEW_COLORS = {"A": (40, 150, 235), "B": (210, 95, 30)}
PREFERRED_REVIEW_CLIP = (
    SOURCE / "Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final" /
    "set1" / "rally_002.mp4"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(row: dict) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")).strip() for field in KEY_FIELDS)


def frame_key(row: dict) -> tuple[str, str, str, str, int]:
    return (*stable_key(row), int(row["clip_frame_num"]))


def safe_id(key: tuple) -> str:
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def load_ultralytics():
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("System Python needs the pinned Ultralytics dependency from requirements.txt") from error
    if ultralytics.__version__ != "8.4.175":
        raise RuntimeError(f"Ultralytics 8.4.175 is required; found {ultralytics.__version__}")
    import ultralytics.utils as utils
    (ROOT / ".ultralytics").mkdir(parents=True, exist_ok=True)
    utils.SETTINGS.update({
        "datasets_dir": str(ROOT / "data"), "weights_dir": str(WEIGHTS.parent),
        "runs_dir": str(ROOT / "runs"),
    })
    utils.WEIGHTS_DIR = WEIGHTS.parent
    return ultralytics, YOLO


def load_config(path: str | None = None, folds_override: int | None = None) -> dict:
    config_path = Path(path).expanduser().resolve() if path else CONFIG_DEFAULT
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"cv", "model", "train"}:
        raise ValueError("Config must contain exactly cv, model, and train sections")
    if any(not isinstance(config[key], dict) for key in ("cv", "model", "train")):
        raise ValueError("cv, model, and train settings must be YAML mappings")
    allowed = {
        "cv": {"folds", "seed"},
        "model": {"input_width", "input_height", "gaussian_sigma"},
        "train": {"epochs", "patience", "batch", "optimizer", "lr0", "weight_decay",
                  "freeze_backbone_epochs", "horizontal_flip_probability", "seed", "device",
                  "workers", "amp"},
    }
    for section, keys in allowed.items():
        unknown = set(config[section]) - keys
        missing = keys - set(config[section])
        if unknown or missing:
            raise ValueError(f"Invalid {section} keys; missing={sorted(missing)}, unsupported={sorted(unknown)}")
    if folds_override is not None:
        config["cv"]["folds"] = folds_override
    for section, key, lower in (("cv", "folds", 2), ("cv", "seed", 0),
                                ("model", "input_width", 32), ("model", "input_height", 32),
                                ("train", "epochs", 1), ("train", "patience", 0),
                                ("train", "batch", 1), ("train", "freeze_backbone_epochs", 0),
                                ("train", "seed", 0), ("train", "workers", 0)):
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{section}.{key} must be an integer >= {lower}")
    for key in ("input_width", "input_height"):
        if config["model"][key] % 4:
            raise ValueError(f"model.{key} must be divisible by 4 for the stride-4 heatmap")
    sigma = config["model"]["gaussian_sigma"]
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("model.gaussian_sigma must be finite and positive")
    train = config["train"]
    if train["optimizer"] != "AdamW":
        raise ValueError("train.optimizer currently supports only AdamW")
    for key, minimum, inclusive in (("lr0", 0, False), ("weight_decay", 0, True)):
        value = train[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"train.{key} must be finite")
        if value < minimum or (not inclusive and value == minimum):
            raise ValueError(f"train.{key} must be {'nonnegative' if inclusive else 'positive'}")
    flip = train["horizontal_flip_probability"]
    if isinstance(flip, bool) or not isinstance(flip, (int, float)) or not math.isfinite(flip) or not 0 <= flip <= 1:
        raise ValueError("train.horizontal_flip_probability must be in [0, 1]")
    if not isinstance(train["amp"], bool):
        raise ValueError("train.amp must be true or false")
    if isinstance(train["device"], bool) or not isinstance(train["device"], (str, int)):
        raise ValueError("train.device must be 'cpu', a CUDA index, or a CUDA device string")
    device = str(train["device"]).strip().lower()
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {train['device']!r} was requested, but CUDA is unavailable")
        index = 0 if device in {"cuda", "cuda:0"} else int(device.removeprefix("cuda:"))
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
    if train["freeze_backbone_epochs"] > train["epochs"]:
        raise ValueError("train.freeze_backbone_epochs cannot exceed train.epochs")
    config["_path"] = str(config_path)
    return config


def resolved_settings(config: dict) -> dict:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def load_saved_config(run_dir: Path) -> dict:
    path = run_dir / "resolved_settings.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Saved settings are missing from {run_dir}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"cv", "model", "train"}:
        raise ValueError(f"Incompatible or obsolete saved pose settings: {path}")
    return load_config(str(path))


def import_geometry(match: str):
    return load_court_geometry(HOMOGRAPHY, match)


def parse_int(value, field: str) -> int:
    if value is None or not str(value).strip():
        raise ValueError(f"{field} is missing")
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{field} is not a finite integer")
    return int(number)


def parse_float(value, field: str) -> float:
    if value is None or not str(value).strip():
        raise ValueError(f"{field} is missing")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is nonfinite")
    return number


def flaw_is_set(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "1.0", "true", "yes", "flaw"}


def align_references(annotation: dict) -> dict[str, tuple[float, float]]:
    hitter = str(annotation.get("player", "")).strip().upper()
    if hitter not in {"A", "B"}:
        raise ValueError("unknown player identity")
    player = (parse_float(annotation.get("player_location_x"), "player_location_x"),
              parse_float(annotation.get("player_location_y"), "player_location_y"))
    opponent = (parse_float(annotation.get("opponent_location_x"), "opponent_location_x"),
                parse_float(annotation.get("opponent_location_y"), "opponent_location_y"))
    return {"A": player if hitter == "A" else opponent,
            "B": opponent if hitter == "A" else player}


def native_point(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    return pm.pixel_center_scale(point, REFERENCE_WIDTH, REFERENCE_HEIGHT, width, height)


def reference_point(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    return pm.pixel_center_scale(point, width, height, REFERENCE_WIDTH, REFERENCE_HEIGHT)


def court_half(geometry, point_native: tuple[float, float], width: int, height: int) -> str:
    world = geometry.annotation_floor_to_world(reference_point(point_native, width, height))
    if not np.isfinite(world).all():
        raise ValueError("proxy floor point maps to nonfinite court coordinates")
    return "far" if float(world[1]) < COURT_HALF_Y_M else "near"


def checked_clip_path(match_dir: Path, filename: str) -> Path:
    if not filename:
        raise ValueError("rally clip filename is missing")
    path = (match_dir / filename.replace("/", os.sep)).resolve()
    if not path.is_relative_to(match_dir.resolve()):
        raise ValueError("rally clip path escapes its match folder")
    return path


def cleanup_index(path: Path) -> dict[tuple[str, str, str, str], list[dict]]:
    indexed = defaultdict(list)
    if path.is_file():
        for row in read_csv(path):
            indexed[stable_key(row)].append(row)
    return indexed


def inspect_rally(match_dir: Path, inventory: dict, cleanup_rows: list[dict],
                  geometries: dict, exclusions: list[dict]) -> dict:
    key, rally_id = stable_key(inventory), safe_id(stable_key(inventory))

    def exclude(reason: str, detail: str = "", frame: int | None = None) -> None:
        exclusions.append({**dict(zip(KEY_FIELDS, key)),
                           "clip_frame_num": "" if frame is None else frame,
                           "reason": reason, "detail": detail})

    reasons = []
    if len(cleanup_rows) != 1:
        reasons.append("missing_cleanup_record" if not cleanup_rows else "ambiguous_cleanup_records")
        exclude(reasons[-1], f"record_count={len(cleanup_rows)}")
    cleanup = cleanup_rows[0] if len(cleanup_rows) == 1 else {}
    for field, code in (("review_flags", "rally_inventory_review_flags"),):
        value = str(inventory.get(field, "") or "").strip()
        if value:
            reasons.append(code)
            exclude(code, value)
    cleanup_flags = str(cleanup.get("review_flags", "") or "").strip()
    if cleanup_flags:
        reasons.append("cleanup_review_flags")
        exclude(reasons[-1], cleanup_flags)
    outcome = str(cleanup.get("outcome", "") or "").strip().lower()
    if outcome == "retained-internal-cut":
        reasons.append("retained_internal_cut")
        exclude(reasons[-1], outcome)
    elif outcome.startswith("review-retained"):
        reasons.append("cleanup_review_outcome")
        exclude(reasons[-1], outcome)

    try:
        clip_path = checked_clip_path(match_dir, str(inventory.get("filename", "")))
        annotations_path = clip_path.with_suffix(".csv")
        if not annotations_path.resolve().is_relative_to(match_dir.resolve()):
            raise ValueError("annotation path escapes its match folder")
    except Exception as error:
        reasons.append("invalid_rally_clip_path")
        exclude(reasons[-1], str(error))
        return {"key": key, "id": rally_id, "status": "excluded", "reasons": reasons,
                "annotations": [], "frames": [], "clip_path": ""}
    if not clip_path.is_file():
        reasons.append("missing_cleaned_clip")
        exclude(reasons[-1], str(clip_path))
    if not annotations_path.is_file():
        reasons.append("missing_cleaned_annotations")
        exclude(reasons[-1], str(annotations_path))
    if reasons:
        return {"key": key, "id": rally_id, "status": "excluded", "reasons": reasons,
                "annotations": [], "frames": [], "clip_path": str(clip_path)}

    annotations = read_csv(annotations_path)
    if not annotations:
        reasons.append("empty_cleaned_annotations")
        exclude(reasons[-1])
    try:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise ValueError("OpenCV could not open the cleaned clip")
        fps, width, height, frame_count = (
            float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        cap.release()
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError(f"invalid metadata fps={fps}, width={width}, height={height}, frames={frame_count}")
    except Exception as error:
        reasons.append("failed_video_metadata_or_decode")
        exclude(reasons[-1], str(error))
        return {"key": key, "id": rally_id, "status": "excluded", "reasons": reasons,
                "annotations": annotations, "frames": [], "clip_path": str(clip_path)}
    try:
        if key[1] not in geometries:
            geometries[key[1]] = import_geometry(key[1])
        geometry = geometries[key[1]]
    except Exception as error:
        reasons.append("invalid_court_geometry")
        exclude(reasons[-1], str(error))
        return {"key": key, "id": rally_id, "status": "excluded", "reasons": reasons,
                "annotations": annotations, "frames": [], "clip_path": str(clip_path)}
    try:
        annotation_fps = parse_float(inventory.get("annotation_fps"), "annotation_fps")
        time_offset = parse_float(inventory.get("time_offset"), "time_offset")
        clean_start = parse_float(inventory.get("start_seconds"), "start_seconds")
        first_hit, last_hit = parse_int(inventory.get("first_hit_frame"), "first_hit_frame"), \
                              parse_int(inventory.get("last_hit_frame"), "last_hit_frame")
        if annotation_fps <= 0 or first_hit < 0 or last_hit < first_hit:
            raise ValueError("invalid annotation FPS or first/last frame bounds")
    except Exception as error:
        reasons.append("invalid_rally_timing_metadata")
        exclude(reasons[-1], str(error))
        return {"key": key, "id": rally_id, "status": "excluded", "reasons": reasons,
                "annotations": annotations, "frames": [], "clip_path": str(clip_path)}

    frames, previous_frame, source_frames = [], None, []
    sides = {"A": set(), "B": set()}
    for row in annotations:
        frame = None
        try:
            frame = parse_int(row.get("clip_frame_num"), "clip_frame_num")
            if frame < 0 or frame >= frame_count:
                raise ValueError(f"clip frame {frame} outside [0, {frame_count})")
            if previous_frame is not None and frame <= previous_frame:
                kind = "duplicate" if frame == previous_frame else "decreasing"
                raise ValueError(f"{kind} clip_frame_num {frame} after {previous_frame}")
            previous_frame = frame
            clip_time = parse_float(row.get("clip_time_seconds"), "clip_time_seconds")
            if not math.isclose(clip_time, frame / fps, rel_tol=0, abs_tol=0.00001):
                raise ValueError(f"clip_time_seconds mismatch at frame {frame}")
            original_frame = parse_int(row.get("frame_num"), "frame_num")
            source_time = parse_float(row.get("source_time_seconds"), "source_time_seconds")
            if not math.isclose(source_time, original_frame / annotation_fps + time_offset,
                                rel_tol=0, abs_tol=0.00001):
                raise ValueError(f"source_time_seconds mismatch at clip frame {frame}")
            if math.floor((source_time - clean_start) * fps + 0.5) != frame:
                raise ValueError(f"source-time mapping disagrees at clip frame {frame}")
            source_frames.append(original_frame)
            references = align_references(row)
            for player, point in references.items():
                if not 0 <= point[0] < REFERENCE_WIDTH or not 0 <= point[1] < REFERENCE_HEIGHT:
                    raise ValueError(f"{player} reference coordinate outside 1280x720: {point}")
                sides[player].add(court_half(geometry, native_point(point, width, height), width, height))
            frames.append({"row": row, "clip_frame_num": frame, "clip_time_seconds": clip_time,
                           "source_time_seconds": source_time, "frame_num": original_frame,
                           "references": references})
        except Exception as error:
            reason = str(error)
            code = "invalid_frame_or_time_correspondence" if any(
                term in reason.lower() for term in ("frame", "time", "fps", "bounds")
            ) else "invalid_reference_or_identity"
            reasons.append(code)
            exclude(code, reason, frame)
        player = str(row.get("player", "") or "").strip().upper()
        if player not in {"A", "B"} and "unknown_player_identity" not in reasons:
            reasons.append("unknown_player_identity")
            exclude("unknown_player_identity", f"player={row.get('player', '')}", frame)
        if flaw_is_set(row.get("flaw", "")) and "dataset_flaw_flag" not in reasons:
            reasons.append("dataset_flaw_flag")
            exclude("dataset_flaw_flag", f"flaw={row.get('flaw', '')}", frame)
    if source_frames and (min(source_frames) != first_hit or max(source_frames) != last_hit):
        reasons.append("inventory_annotation_frame_bounds_mismatch")
        exclude(reasons[-1], f"inventory={first_hit}:{last_hit}, annotations={min(source_frames)}:{max(source_frames)}")
    if len(sides["A"]) > 1 or len(sides["B"]) > 1:
        reasons.append("identity_changes_court_half")
        exclude(reasons[-1], f"A={sorted(sides['A'])}, B={sorted(sides['B'])}")
    if len(sides["A"]) == len(sides["B"]) == 1 and sides["A"] == sides["B"]:
        reasons.append("same_court_half")
        exclude(reasons[-1], f"A={sorted(sides['A'])}, B={sorted(sides['B'])}")
    if not frames and not reasons:
        reasons.append("no_valid_annotation_frames")
        exclude(reasons[-1])
    return {
        "key": key, "id": rally_id, "status": "excluded" if reasons else "automatic_pass",
        "reasons": sorted(set(reasons)), "annotations": annotations, "frames": frames,
        "clip_path": str(clip_path), "fps": fps, "width": width, "height": height,
        "frame_count": frame_count,
        "A_half": next(iter(sides["A"])) if len(sides["A"]) == 1 else "",
        "B_half": next(iter(sides["B"])) if len(sides["B"]) == 1 else "",
    }


def enumerate_rallies(exclusions: list[dict]) -> list[tuple[Path, dict, dict]]:
    result = []
    for match_dir in sorted(SOURCE.iterdir()) if SOURCE.is_dir() else []:
        if not match_dir.is_dir():
            continue
        if match_dir.name in EXCLUDED_MATCHES:
            exclusions.append({"dataset": "set-1", "match": match_dir.name, "game": "",
                               "original_rally_id": "", "clip_frame_num": "",
                               "reason": "permanent_match_exclusion", "detail": "experiment specification"})
            continue
        inventory_path, cleanup_path = match_dir / "rallies.csv", match_dir / "cleanup_report.csv"
        if not inventory_path.is_file():
            exclusions.append({"dataset": "set-1", "match": match_dir.name, "game": "",
                               "original_rally_id": "", "clip_frame_num": "",
                               "reason": "missing_rally_inventory", "detail": str(inventory_path)})
            continue
        cleanup_rows = cleanup_index(cleanup_path)
        for row in read_csv(inventory_path):
            result.append((match_dir, row, {"cleanup_rows": cleanup_rows.get(stable_key(row), [])}))
    return result


def frame_image_id(rally: dict, frame: dict) -> str:
    return f"{rally['id']}_f{frame['clip_frame_num']:06d}"


def review_sample_rows(manifest: list[dict]) -> list[dict]:
    rallies = defaultdict(list)
    for row in manifest:
        rallies[stable_key(row)].append(row)
    by_match = defaultdict(list)
    for key, rows in rallies.items():
        rows.sort(key=lambda row: int(row["clip_frame_num"]))
        by_match[key[1]].append((key, rows))
    samples = []
    for match in sorted(by_match):
        choices = sorted(by_match[match], key=lambda pair: (
            min(float(row["source_time_seconds"]) for row in pair[1]), pair[0]
        ))
        count = min(5, len(choices))
        indices = [0] if count == 1 else [round(i * (len(choices) - 1) / (count - 1)) for i in range(count)]
        for index in indices:
            key, frames = choices[index]
            row = frames[len(frames) // 2]
            samples.append({
                **dict(zip(KEY_FIELDS, key)), "rally_id": row["rally_id"],
                "clip_frame_num": row["clip_frame_num"], "image_path": row["image_path"],
                "A_ref_x": row["A_native_x"], "A_ref_y": row["A_native_y"],
                "B_ref_x": row["B_native_x"], "B_ref_y": row["B_native_y"],
                "A_half": row["A_half"], "B_half": row["B_half"],
            })
    return samples


def prepare_command(config: dict) -> int:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Cleaned set-1 footage folder not found: {SOURCE}")
    if (PREP / "frozen_manifest.jsonl").exists():
        raise RuntimeError("A frozen dataset already exists; preserving it")
    if (PREP / "manifest.jsonl").exists():
        old = read_jsonl(PREP / "manifest.jsonl")
        if old and any(row.get("manifest_version") != MANIFEST_VERSION for row in old):
            raise RuntimeError("Obsolete pose manifest found; preserve it and use a fresh point manifest directory")
        raise FileExistsError(f"Preparation already exists; preserving it at {PREP}")
    PREP.mkdir(parents=True, exist_ok=True)
    exclusions, outcomes, manifest = [], [], []
    candidates = enumerate_rallies(exclusions)
    geometries, screened = {}, []
    for index, (match_dir, inventory, extras) in enumerate(candidates, 1):
        screened.append(inspect_rally(match_dir, inventory, extras["cleanup_rows"], geometries, exclusions))
        if index % 100 == 0 or index == len(candidates):
            print(f"Screened {index}/{len(candidates)} rallies", flush=True)
    automatic = [rally for rally in screened if rally["status"] == "automatic_pass"]
    for rally in screened:
        if rally["status"] != "automatic_pass":
            outcomes.append({"key": rally["key"], "id": rally["id"], "status": "excluded",
                             "reasons": rally["reasons"], "clip_path": rally["clip_path"]})
            continue
        cap = cv2.VideoCapture(rally["clip_path"])
        decoded, error = [], ""
        for item in rally["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, item["clip_frame_num"])
            ok, image = cap.read()
            if not ok or image is None or image.shape[:2] != (rally["height"], rally["width"]):
                error = f"failed to decode exact annotated clip frame {item['clip_frame_num']}"
                break
            decoded.append(image)
        cap.release()
        if error:
            exclusions.append({**dict(zip(KEY_FIELDS, rally["key"])), "clip_frame_num": "",
                               "reason": "failed_frame_decode", "detail": error})
            outcomes.append({"key": rally["key"], "id": rally["id"], "status": "excluded",
                             "reasons": ["failed_frame_decode"], "clip_path": rally["clip_path"]})
            continue
        rally_rows, written, write_error = [], [], ""
        for item, image in zip(rally["frames"], decoded, strict=True):
            image_path = PREP / "images" / f"{frame_image_id(rally, item)}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                write_error = f"could not write {image_path}"
                break
            written.append(image_path)
            native = {player: native_point(item["references"][player], rally["width"], rally["height"])
                      for player in ("A", "B")}
            rally_rows.append({
                "manifest_version": MANIFEST_VERSION, **dict(zip(KEY_FIELDS, rally["key"])),
                "rally_id": rally["id"], "clip_path": rally["clip_path"],
                "source_csv": item["row"].get("source_csv", ""),
                "clip_frame_num": item["clip_frame_num"], "clip_time_seconds": item["clip_time_seconds"],
                "source_time_seconds": item["source_time_seconds"], "frame_num": item["frame_num"],
                "fps": rally["fps"], "width": rally["width"], "height": rally["height"],
                "image_path": str(image_path.resolve()),
                "A_ref_x": item["references"]["A"][0], "A_ref_y": item["references"]["A"][1],
                "B_ref_x": item["references"]["B"][0], "B_ref_y": item["references"]["B"][1],
                "A_native_x": native["A"][0], "A_native_y": native["A"][1],
                "B_native_x": native["B"][0], "B_native_y": native["B"][1],
                "A_half": rally["A_half"], "B_half": rally["B_half"],
                "hitter_player": str(item["row"].get("player", "")).strip().upper(),
                "automatic_check_status": "passed", "sampled_review_status": "not_sampled",
            })
        if write_error:
            for path in written:
                if path.resolve().is_relative_to(PREP.resolve()):
                    path.unlink(missing_ok=True)
            exclusions.append({**dict(zip(KEY_FIELDS, rally["key"])), "clip_frame_num": "",
                               "reason": "prepared_image_write_failed", "detail": write_error})
            outcomes.append({"key": rally["key"], "id": rally["id"], "status": "excluded",
                             "reasons": ["prepared_image_write_failed"], "clip_path": rally["clip_path"]})
            continue
        manifest.extend(rally_rows)
        outcomes.append({"key": rally["key"], "id": rally["id"], "status": "automatic_pass",
                         "reasons": [], "clip_path": rally["clip_path"],
                         "A_half": rally["A_half"], "B_half": rally["B_half"],
                         "frame_count": len(rally_rows)})
    samples = review_sample_rows(manifest)
    sample_keys = {frame_key(sample) for sample in samples}
    for row in manifest:
        if frame_key(row) in sample_keys:
            row["sampled_review_status"] = "pending"
    write_jsonl(PREP / "manifest.jsonl", manifest)
    write_jsonl(PREP / "rallies.jsonl", outcomes)
    write_csv(PREP / "exclusions.csv", exclusions, [*KEY_FIELDS, "clip_frame_num", "reason", "detail"])
    state = {"manifest_version": MANIFEST_VERSION, "created_at": utc_now(), "samples": {
        safe_id(frame_key(sample)): {"status": "pending", "sample": sample} for sample in samples
    }}
    write_json(PREP / "review_state.json", state)
    write_csv(PREP / "review_samples.csv", samples, [*KEY_FIELDS, "rally_id", "clip_frame_num",
              "image_path", "A_ref_x", "A_ref_y", "B_ref_x", "B_ref_y", "A_half", "B_half"])
    auto_outcomes = [row for row in outcomes if row["status"] == "automatic_pass"]
    summary = {
        "manifest_version": MANIFEST_VERSION, "model_kind": pm.MODEL_KIND,
        "created_at": utc_now(), "source": str(SOURCE),
        "permanently_excluded_matches": sorted(EXCLUDED_MATCHES),
        "matches_in_source": sum(path.is_dir() for path in SOURCE.iterdir()),
        "matches_after_permanent_exclusions": sum(
            path.is_dir() and path.name not in EXCLUDED_MATCHES for path in SOURCE.iterdir()
        ),
        "matches_with_inventories": len({item[1]["match"] for item in candidates}),
        "input_rallies": len(candidates),
        "preliminary_rallies": len(automatic),
        "preliminary_annotated_frames": sum(len(row["frames"]) for row in automatic),
        "automatic_rallies_after_decode": len(auto_outcomes),
        "automatic_annotated_frames": len(manifest),
        "matches_for_sampled_review": len({row["match"] for row in manifest}),
        "sampled_review_frames": len(samples), "excluded_rallies": len(outcomes) - len(auto_outcomes),
        "exclusion_rows": len(exclusions),
        "manifest_sha256": sha256_file(PREP / "manifest.jsonl"),
        "dataset_identity": dataset_identity(manifest),
    }
    write_json(PREP / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Exclusion reasons: {PREP / 'exclusions.csv'}")
    return 0


def draw_reference_sample(image: np.ndarray, sample: dict) -> np.ndarray:
    image = image.copy()
    for player in ("A", "B"):
        color = REVIEW_COLORS[player]
        point = (round(float(sample[f"{player}_ref_x"])), round(float(sample[f"{player}_ref_y"])))
        cv2.circle(image, point, 9, color, 2, cv2.LINE_AA)
        cv2.putText(image, player, (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    header = (f"{sample['match']} | game {sample['game']} | rally {sample['original_rally_id']} | "
              f"frame {sample['clip_frame_num']}")
    cv2.putText(image, header[:150], (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def apply_review_status(state: dict) -> None:
    manifest_path = PREP / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    status_by_frame = {
        frame_key(entry["sample"]): entry["status"]
        for entry in state["samples"].values()
    }
    for row in rows:
        row["sampled_review_status"] = status_by_frame.get(frame_key(row), "not_sampled")
    write_jsonl(manifest_path, rows)


def review_command() -> int:
    state_path = PREP / "review_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("Run prepare before review")
    if SPLIT_FILE.exists():
        raise RuntimeError("The accepted dataset is frozen; sampled review cannot be changed")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("manifest_version") != MANIFEST_VERSION:
        raise RuntimeError("Obsolete pose review state; it cannot be used for point-only preparation")
    window = "Point-only A/B sample review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for entry in state["samples"].values():
            if entry["status"] != "pending":
                continue
            sample = entry["sample"]
            if any(other["status"] == "rejected_match" and other["sample"]["match"] == sample["match"]
                   for other in state["samples"].values()):
                entry["status"] = "rejected_match"
                continue
            image = cv2.imread(sample["image_path"])
            if image is None:
                raise RuntimeError(f"Review image is missing or unreadable: {sample['image_path']}")
            image = draw_reference_sample(image, sample)
            cv2.putText(image, "A accept  R reject rally  M reject match  S save/quit  Q quit",
                        (18, image.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, image)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("a"), ord("A")):
                entry["status"] = "accepted"
            elif key in (ord("r"), ord("R")):
                entry["status"] = "rejected_rally"
            elif key in (ord("m"), ord("M")):
                for other in state["samples"].values():
                    if other["sample"]["match"] == sample["match"]:
                        other["status"] = "rejected_match"
            elif key in (ord("s"), ord("S"), ord("q"), ord("Q"), 27):
                break
            else:
                continue
            entry["updated_at"] = utc_now()
            state["updated_at"] = utc_now()
            write_json(state_path, state)
    finally:
        cv2.destroyWindow(window)
    state["updated_at"] = utc_now()
    state["complete"] = all(row["status"] != "pending" for row in state["samples"].values())
    write_json(state_path, state)
    apply_review_status(state)
    counts = defaultdict(int)
    for row in state["samples"].values():
        counts[row["status"]] += 1
    print(json.dumps({"review_state": str(state_path), "complete": state["complete"],
                      "sample_status_counts": dict(counts)}, indent=2))
    return 0


def review_exclusions(state: dict) -> list[dict]:
    result, seen_matches = [], set()
    for entry in state["samples"].values():
        sample, status = entry["sample"], entry["status"]
        if status == "rejected_rally":
            result.append({**{field: sample[field] for field in KEY_FIELDS},
                           "clip_frame_num": sample["clip_frame_num"],
                           "reason": "manual_sample_rejected_rally", "detail": "sample review rejected rally"})
        elif status == "rejected_match" and sample["match"] not in seen_matches:
            seen_matches.add(sample["match"])
            result.append({"dataset": sample["dataset"], "match": sample["match"], "game": "",
                           "original_rally_id": "", "clip_frame_num": "",
                           "reason": "manual_sample_rejected_match", "detail": "sample review rejected match"})
    return result


def validate_manifest(rows: list[dict], require_images: bool = True) -> None:
    if not rows:
        raise ValueError("Point manifest is empty")
    for row in rows:
        if row.get("manifest_version") != MANIFEST_VERSION or row.get("dataset") != "set-1":
            raise ValueError("Obsolete pose manifest or data outside cleaned_rallies/set-1")
        if row.get("automatic_check_status") != "passed":
            raise ValueError(f"Manifest row did not pass source checks: {frame_key(row)}")
        if row.get("A_half") not in pm.CHANNELS or row.get("B_half") not in pm.CHANNELS \
                or row["A_half"] == row["B_half"]:
            raise ValueError(f"Invalid A/B far/near mapping in {stable_key(row)}")
        width, height = int(row["width"]), int(row["height"])
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image dimensions in {frame_key(row)}")
        for player in ("A", "B"):
            x, y = float(row[f"{player}_native_x"]), float(row[f"{player}_native_y"])
            if not np.isfinite([x, y]).all() or not 0 <= x < width or not 0 <= y < height:
                raise ValueError(f"Invalid {player} point in {frame_key(row)}")
        if require_images:
            image_path = Path(row["image_path"]).resolve()
            if not image_path.is_relative_to(PREP.resolve()) or not image_path.is_file():
                raise ValueError(f"Prepared image is missing or outside point dataset: {image_path}")


def accepted_manifest() -> tuple[list[dict], dict, list[dict]]:
    state_path, manifest_path = PREP / "review_state.json", PREP / "manifest.jsonl"
    if not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("Run point-only prepare and complete sampled review before split")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("manifest_version") != MANIFEST_VERSION:
        raise RuntimeError("Obsolete pose review state; reprepare with the point-only pipeline")
    pending = sum(row["status"] == "pending" for row in state["samples"].values())
    if pending:
        raise RuntimeError(f"Sample review has {pending} pending rallies; use review to finish or resume")
    rejected_matches = {entry["sample"]["match"] for entry in state["samples"].values()
                        if entry["status"] == "rejected_match"}
    rejected_rallies = {stable_key(entry["sample"]) for entry in state["samples"].values()
                        if entry["status"] == "rejected_rally"}
    accepted = [row for row in read_jsonl(manifest_path) if row["match"] not in rejected_matches
                and stable_key(row) not in rejected_rallies]
    validate_manifest(accepted)
    return accepted, state, review_exclusions(state)


def dataset_identity(rows: list[dict]) -> str:
    fields = (*KEY_FIELDS, "manifest_version", "clip_frame_num", "frame_num", "clip_time_seconds",
              "source_time_seconds", "clip_path", "fps", "width", "height", "image_path",
              "A_ref_x", "A_ref_y", "B_ref_x", "B_ref_y", "A_native_x", "A_native_y",
              "B_native_x", "B_native_y", "A_half", "B_half")
    values = [[row[field] for field in fields] for row in sorted(
        rows, key=lambda item: (*stable_key(item), int(item["clip_frame_num"]))
    )]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def grouped_folds(matches: list[str], folds: int, seed: int) -> dict[str, int]:
    if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= len(matches):
        raise ValueError(f"Requested {folds} folds; require 2 <= folds <= {len(matches)} development matches")
    shuffled = sorted(set(matches))
    if len(shuffled) != len(matches):
        raise ValueError("Development match list contains duplicates")
    random.Random(seed).shuffle(shuffled)
    return {match: index % folds for index, match in enumerate(shuffled)}


def fixed_test_matches(matches: list[str]) -> list[str]:
    if len(matches) < 3:
        raise ValueError("At least three accepted matches are required for a 20% test and grouped CV")
    shuffled = sorted(matches)
    random.Random(42).shuffle(shuffled)
    count = max(1, int(math.floor(len(shuffled) * 0.20 + 0.5)))
    if count >= len(shuffled):
        raise ValueError("The fixed 20% test reservation leaves no development match")
    return sorted(shuffled[:count])


def split_command(config: dict) -> int:
    rows, review_state, manual = accepted_manifest()
    matches = sorted({row["match"] for row in rows})
    development = None
    if SPLIT_FILE.exists():
        saved = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        if saved.get("split_version") != SPLIT_VERSION:
            raise RuntimeError("Obsolete pose split found; preserving it rather than changing held-out membership")
        if saved.get("dataset_identity") != dataset_identity(rows):
            raise RuntimeError("A different accepted point dataset is already frozen; refusing to replace its split")
        held_out, development = saved["held_out_matches"], saved["development_matches"]
        if set(held_out) | set(development) != set(matches) or set(held_out) & set(development):
            raise RuntimeError("Saved whole-match assignments do not match the accepted dataset")
    else:
        held_out = fixed_test_matches(matches)
        development = sorted(set(matches) - set(held_out))
        saved = {"split_version": SPLIT_VERSION, "created_at": utc_now(),
                 "dataset_identity": dataset_identity(rows), "held_out_seed": 42,
                 "held_out_matches": held_out, "development_matches": development, "cv_runs": {}}
    folds, seed = int(config["cv"]["folds"]), int(config["cv"]["seed"])
    assignments = grouped_folds(development, folds, seed)
    cv_key = f"folds-{folds}-seed-{seed}"
    saved.setdefault("cv_runs", {})[cv_key] = {"folds": folds, "seed": seed, "match_to_fold": assignments}
    saved["updated_at"] = utc_now()
    write_json(SPLIT_FILE, saved)
    frozen = PREP / "frozen_manifest.jsonl"
    if frozen.exists():
        existing = read_jsonl(frozen)
        validate_manifest(existing)
        if dataset_identity(existing) != saved["dataset_identity"]:
            raise RuntimeError("Existing frozen manifest differs from the accepted point dataset")
    else:
        write_jsonl(frozen, rows)
    write_csv(PREP / "review_exclusions.csv", manual,
              [*KEY_FIELDS, "clip_frame_num", "reason", "detail"])
    print(json.dumps({"split_file": str(SPLIT_FILE), "accepted_matches": len(matches),
                      "held_out_matches": len(held_out), "development_matches": len(development),
                      "held_out_match_names": held_out, "cv_assignment_key": cv_key,
                      "fold_sizes": {str(i + 1): sum(value == i for value in assignments.values())
                                     for i in range(folds)}, "dataset_identity": saved["dataset_identity"]}, indent=2))
    return 0


def run_dependencies() -> dict:
    ultralytics, _ = load_ultralytics()
    return {
        "python": sys.version.split()[0], "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
        "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda,
        "opencv": cv2.__version__, "numpy": np.__version__, "pyyaml": yaml.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def ensure_pretrained_weights() -> Path:
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if WEIGHTS.is_file() and WEIGHTS.stat().st_size > 1_000_000:
        return WEIGHTS
    load_ultralytics()
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(
            str(WEIGHTS), repo="ultralytics/assets", release="v8.4.0"
        )).resolve()
        if downloaded != WEIGHTS.resolve() and downloaded.is_file():
            shutil.copy2(downloaded, WEIGHTS)
    except Exception as error:
        raise RuntimeError(f"Required official yolo11n.pt checkpoint is unavailable: {error}") from error
    if not WEIGHTS.is_file() or WEIGHTS.stat().st_size <= 1_000_000:
        raise RuntimeError(f"Required official yolo11n.pt checkpoint is missing or incomplete: {WEIGHTS}")
    return WEIGHTS


def dependency_snapshot() -> dict:
    return {**run_dependencies(), "pretrained_checkpoint": str(WEIGHTS.resolve()),
            "pretrained_checkpoint_sha256": sha256_file(WEIGHTS), "model_contract": pm.model_contract()}


def resolve_device(config: dict) -> torch.device:
    value = str(config["train"]["device"]).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    index = 0 if value == "cuda" else int(value.removeprefix("cuda:"))
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} was requested but is unavailable")
    return torch.device(f"cuda:{index}")


def resolved_train_loader(rows: list[dict], config: dict, training: bool):
    from torch.utils.data import DataLoader

    settings = config["model"]
    dataset = pm.FrameDataset(
        rows, settings["input_width"], settings["input_height"], settings["gaussian_sigma"],
        config["train"]["horizontal_flip_probability"] if training else 0.0,
    )
    generator = torch.Generator().manual_seed(config["train"]["seed"])
    return DataLoader(dataset, batch_size=config["train"]["batch"], shuffle=training,
                      num_workers=config["train"]["workers"], generator=generator if training else None,
                      pin_memory=resolve_device(config).type == "cuda")


def build_optimizer(model, train: dict):
    if train["optimizer"] != "AdamW":
        raise ValueError(f"Unsupported optimizer: {train['optimizer']}")
    return torch.optim.AdamW(model.parameters(), lr=float(train["lr0"]),
                             weight_decay=float(train["weight_decay"]))


def error_summary(rows: list[dict]) -> dict:
    expected = len(rows)
    available = [row for row in rows if row["available"]]

    def stats(key: str) -> dict:
        values = [float(row[key]) for row in available if row.get(key) is not None]
        return {"count": len(values), "mean": None if not values else float(np.mean(values)),
                "median": None if not values else float(np.median(values)),
                "p95": None if not values else float(np.percentile(values, 95))}

    def success(threshold: float) -> dict:
        count = sum(row["available"] and row.get("error_m") is not None and row["error_m"] <= threshold
                    for row in rows)
        return {"count": int(count), "denominator": expected,
                "rate": None if not expected else count / expected}

    return {"expected_predictions": expected, "available_predictions": len(available),
            "missing_predictions": expected - len(available),
            "coverage": None if not expected else len(available) / expected,
            "pixel_error": stats("error_px"), "metre_error": stats("error_m"),
            "within_0_5_m": success(0.5), "within_1_m": success(1.0)}


def projected_world(geometry, point_native, width: int, height: int):
    point_reference = reference_point(point_native, width, height)
    world = geometry.annotation_floor_to_world(point_reference)
    return None if not np.isfinite(world).all() else tuple(float(value) for value in world)


def score_model_rows(model, rows: list[dict], config: dict, device: torch.device):
    if not rows:
        return [], []
    loader = resolved_train_loader(rows, config, training=False)
    width, height = config["model"]["input_width"], config["model"]["input_height"]
    positions, errors, offset = [], [], 0
    geometries = {}
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            batch_rows = rows[offset:offset + len(images)]
            for row, output in zip(batch_rows, logits, strict=True):
                match = row["match"]
                if match not in geometries:
                    geometries[match] = import_geometry(match)
                geometry = geometries[match]
                native_width, native_height = int(row["width"]), int(row["height"])
                decoded = pm.decode_heatmaps(output, width, height, native_width, native_height)
                by_half = {"far": decoded[pm.CHANNELS["far"]], "near": decoded[pm.CHANNELS["near"]]}
                for player in ("A", "B"):
                    half = row[f"{player}_half"]
                    prediction = by_half[half]
                    x, y = prediction["x_native"], prediction["y_native"]
                    world = projected_world(geometry, (x, y), native_width, native_height)
                    ref_x, ref_y = float(row[f"{player}_native_x"]), float(row[f"{player}_native_y"])
                    ref_world = projected_world(geometry, (ref_x, ref_y), native_width, native_height)
                    available = world is not None and ref_world is not None
                    status = "available" if available else "invalid_court_projection"
                    error_px = math.hypot(x - ref_x, y - ref_y) if available else None
                    error_m = float(np.linalg.norm(np.asarray(world) - np.asarray(ref_world))) if available else None
                    positions.append({
                        **{field: row[field] for field in KEY_FIELDS}, "clip_frame_num": row["clip_frame_num"],
                        "player": player, "prediction_status": status, "available": available,
                        "x_native": x, "y_native": y,
                        "x_court_m": None if world is None else world[0],
                        "y_court_m": None if world is None else world[1],
                        "heatmap_score": prediction["heatmap_score"], "court_half": half,
                        "reference_x_native": ref_x, "reference_y_native": ref_y,
                    })
                    errors.append({**{field: row[field] for field in KEY_FIELDS},
                                   "clip_frame_num": row["clip_frame_num"], "player": player,
                                   "prediction_status": status, "available": available,
                                   "error_px": error_px, "error_m": error_m})
            offset += len(images)
    return positions, errors


def predict_and_score(weights: str | Path, rows: list[dict], config: dict,
                      output_dir: Path | None = None):
    model, checkpoint = pm.load_checkpoint(Path(weights))
    if checkpoint.get("model_contract") != pm.model_contract():
        raise RuntimeError("Checkpoint contract is not the point-only heatmap model")
    positions, errors = score_model_rows(model, rows, config, resolve_device(config))
    return positions, errors


def write_evaluation(output_dir: Path, positions: list[dict], errors: list[dict],
                     folds: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "player_positions.csv", positions, [
        *KEY_FIELDS, "clip_frame_num", "player", "prediction_status", "available",
        "x_native", "y_native", "x_court_m", "y_court_m", "heatmap_score", "court_half",
        "reference_x_native", "reference_y_native", "fold",
    ])
    write_csv(output_dir / "evaluation_results.csv", errors, [
        *KEY_FIELDS, "clip_frame_num", "player", "prediction_status", "available",
        "error_px", "error_m", "fold",
    ])
    metrics = {
        "combined": error_summary(errors),
        "per_player": {p: error_summary([row for row in errors if row["player"] == p]) for p in ("A", "B")},
        "per_match": {m: error_summary([row for row in errors if row["match"] == m])
                      for m in sorted({row["match"] for row in errors})},
    }
    if folds is not None:
        metrics["folds"] = {str(i): error_summary([row for row in errors if row.get("fold") == i])
                             for i in range(1, folds + 1)}
        if set(metrics["folds"]) != {str(i) for i in range(1, folds + 1)}:
            raise RuntimeError(f"Expected exactly {folds} fold summaries")
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def ensure_cv_assignment(config: dict, split: dict) -> str:
    folds, seed = int(config["cv"]["folds"]), int(config["cv"]["seed"])
    key = f"folds-{folds}-seed-{seed}"
    runs = split.setdefault("cv_runs", {})
    expected = grouped_folds(split["development_matches"], folds, seed)
    if key in runs and runs[key].get("match_to_fold") != expected:
        raise RuntimeError(f"Saved assignment {key} is incompatible; refusing to replace it")
    if key not in runs:
        runs[key] = {"folds": folds, "seed": seed, "match_to_fold": expected}
        split["updated_at"] = utc_now()
        write_json(SPLIT_FILE, split)
    return key


def cv_split_rows(rows: list[dict], split: dict, assignments: dict, fold_index: int):
    val_matches = {match for match, value in assignments.items() if value == fold_index}
    train = [row for row in rows if row["match"] in split["development_matches"]
             and row["match"] not in val_matches]
    val = [row for row in rows if row["match"] in val_matches]
    if not train or not val:
        raise ValueError(f"Fold {fold_index + 1} has an empty train or validation set")
    train_matches, val_rows_matches = {r["match"] for r in train}, {r["match"] for r in val}
    if train_matches & val_rows_matches or train_matches & set(split["held_out_matches"]):
        raise AssertionError("Match leakage between grouped CV partitions")
    return train, val


def run_training(train_rows: list[dict], val_rows: list[dict] | None, output_dir: Path,
                 config: dict, init_seed: int, epochs: int, *, final: bool = False) -> dict:
    weights = ensure_pretrained_weights()
    model, transfer = pm.initialize_from_yolo11n(weights, init_seed)
    if not transfer["complete_transfer"]:
        raise RuntimeError("Incomplete YOLO11n backbone transfer")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "weight_transfer.json", transfer)
    write_json(output_dir / "model_contract.json", pm.model_contract())
    device = resolve_device(config)
    model.to(device)
    loader = resolved_train_loader(train_rows, config, training=True)
    optimizer = build_optimizer(model, config["train"])
    use_amp = bool(config["train"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, best_epoch, bad_epochs, history = math.inf, None, 0, []
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    for epoch in range(1, epochs + 1):
        freeze_backbone = epoch <= config["train"]["freeze_backbone_epochs"]
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(not freeze_backbone)
        model.train()
        losses = []
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = pm.spatial_cross_entropy(model(images), targets)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().float().item()))
        selection = None
        if val_rows is not None:
            _, val_errors = score_model_rows(model, val_rows, config, device)
            metric = error_summary(val_errors)
            invalid = metric["missing_predictions"] > 0 or metric["metre_error"]["count"] != len(val_errors)
            score = math.inf if invalid or metric["metre_error"]["mean"] is None else metric["metre_error"]["mean"]
            selection = {"mean_court_error_m": None if not math.isfinite(score) else score,
                         "score_is_infinite": not math.isfinite(score),
                         "invalid_predictions": metric["missing_predictions"],
                         "validation_metrics": metric}
            improved = best_epoch is None or score < best_score
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                pm.save_checkpoint(best_path, model, resolved_settings(config), {
                    "epoch": epoch, "selected_epoch_count": epoch, "selection": selection,
                    "init_seed": init_seed, "pretrained_sha256": transfer["source_checkpoint_sha256"],
                })
            else:
                bad_epochs += 1
        last_hash = pm.save_checkpoint(last_path, model, resolved_settings(config), {
            "epoch": epoch, "init_seed": init_seed,
            "pretrained_sha256": transfer["source_checkpoint_sha256"], "final_training": final,
        })
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "selection": selection})
        if val_rows is not None and not (best_epoch == epoch):
            patience = config["train"]["patience"]
            if bad_epochs > 0 and bad_epochs >= patience:
                break
    write_json(output_dir / "epochs.json", history)
    if val_rows is not None and best_epoch is None:
        raise RuntimeError("Validation did not select a checkpoint")
    if final and history[-1]["epoch"] != epochs:
        raise RuntimeError("Final training ended before its fixed median epoch budget")
    return {
        "selected_epoch_count": epochs if final else best_epoch,
        "epochs_completed": history[-1]["epoch"], "best_checkpoint": str(best_path.resolve()) if val_rows is not None else None,
        "last_checkpoint": str(last_path.resolve()),
        "best_sha256": sha256_file(best_path) if best_path.is_file() else None,
        "last_sha256": last_hash, "selection": history[best_epoch - 1]["selection"] if val_rows else None,
        "init_seed": init_seed, "weight_transfer": transfer,
    }


def cv_run_identity(config: dict, split: dict, assignments: dict) -> str:
    payload = {"settings": resolved_settings(config), "dataset_identity": split["dataset_identity"],
               "held_out_matches": split["held_out_matches"], "match_to_fold": assignments,
               "pretrained_sha256": sha256_file(WEIGHTS)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"cv_{config['cv']['folds']}f_s{config['cv']['seed']}_{digest}"


def train_cv_command(config: dict) -> int:
    ensure_pretrained_weights()
    frozen_path = PREP / "frozen_manifest.jsonl"
    if not SPLIT_FILE.is_file() or not frozen_path.is_file():
        raise FileNotFoundError("Complete sampled review and freeze the point dataset before train-cv")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split.get("split_version") != SPLIT_VERSION:
        raise RuntimeError("Obsolete pose split; point-only training requires a version-2 split")
    rows = read_jsonl(frozen_path)
    validate_manifest(rows)
    if dataset_identity(rows) != split["dataset_identity"]:
        raise RuntimeError("Frozen point manifest differs from its split identity")
    key = ensure_cv_assignment(config, split)
    assignments = split["cv_runs"][key]["match_to_fold"]
    if set(assignments) != set(split["development_matches"]):
        raise RuntimeError("CV assignment does not cover every development match")
    if set(assignments.values()) != set(range(config["cv"]["folds"])):
        raise RuntimeError("Saved assignment does not cover exactly N folds")
    run_id = cv_run_identity(config, split, assignments)
    run_dir = ROOT / "runs" / run_id
    summary_path = run_dir / "cv_summary.json"
    if summary_path.is_file():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if completed.get("status") == "completed" and len(completed.get("fold_summaries", {})) == config["cv"]["folds"]:
            print(json.dumps({"run_id": run_id, "status": "already_completed"}, indent=2))
            return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = resolved_settings(config)
    settings_path = run_dir / "resolved_settings.yaml"
    if settings_path.exists() and yaml.safe_load(settings_path.read_text(encoding="utf-8")) != settings:
        raise RuntimeError(f"Run settings differ from resolved config: {run_dir}")
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "dependencies.json", dependency_snapshot())
    write_json(run_dir / "fold_assignments.json", {
        "split_version": SPLIT_VERSION, "cv_key": key, "folds": config["cv"]["folds"],
        "seed": config["cv"]["seed"], "match_to_fold": assignments,
        "held_out_matches": split["held_out_matches"], "development_matches": split["development_matches"],
    })
    summaries = {}
    for fold_index in range(config["cv"]["folds"]):
        fold = fold_index + 1
        fold_dir = run_dir / f"fold_{fold:02d}"
        fold_meta_path = fold_dir / "fold.json"
        if fold_meta_path.is_file():
            old = json.loads(fold_meta_path.read_text(encoding="utf-8"))
            if old.get("status") == "completed" and Path(old["best_checkpoint"]).is_file():
                if sha256_file(Path(old["best_checkpoint"])) != old.get("best_sha256"):
                    raise RuntimeError(f"Completed fold checkpoint hash changed: {fold_dir}")
                summaries[str(fold)] = old
                continue
            raise RuntimeError(f"Incomplete fold output exists; preserving it: {fold_dir}")
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise RuntimeError(f"Partial fold output exists; preserving it: {fold_dir}")
        train_rows, val_rows = cv_split_rows(rows, split, assignments, fold_index)
        fold_dir.mkdir(parents=True, exist_ok=True)
        init_seed = int(config["train"]["seed"]) + fold
        result = run_training(train_rows, val_rows, fold_dir, config, init_seed,
                              config["train"]["epochs"])
        summary = {
            "status": "completed", "fold": fold, "completed_at": utc_now(),
            "train_matches": sorted({row["match"] for row in train_rows}),
            "validation_matches": sorted({row["match"] for row in val_rows}),
            "train_frames": len(train_rows), "validation_frames": len(val_rows), **result,
        }
        write_json(fold_meta_path, summary)
        summaries[str(fold)] = summary
    if len(summaries) != config["cv"]["folds"]:
        raise RuntimeError(f"Expected exactly {config['cv']['folds']} completed folds")
    positions, errors = [], []
    for fold in range(1, config["cv"]["folds"] + 1):
        summary = summaries[str(fold)]
        val_matches = set(summary["validation_matches"])
        val_rows = [row for row in rows if row["match"] in val_matches]
        fold_positions, fold_errors = predict_and_score(summary["best_checkpoint"], val_rows, config)
        for row in fold_positions:
            row["fold"] = fold
        for row in fold_errors:
            row["fold"] = fold
        positions.extend(fold_positions)
        errors.extend(fold_errors)
    expected = sum(2 for row in rows if row["match"] in split["development_matches"])
    if len(errors) != expected or {row["fold"] for row in errors} != set(range(1, config["cv"]["folds"] + 1)):
        raise RuntimeError("OOF aggregation did not contain all development annotations and exactly N folds")
    metrics = write_evaluation(run_dir / "oof", positions, errors, folds=config["cv"]["folds"])
    write_json(summary_path, {"status": "completed", "run_id": run_id, "completed_at": utc_now(),
                              "folds": config["cv"]["folds"], "fold_summaries": summaries,
                              "selected_epoch_counts": [summaries[str(f)]["selected_epoch_count"]
                                                         for f in range(1, config["cv"]["folds"] + 1)],
                              "oof_metrics": str((run_dir / "oof" / "metrics.json").resolve()),
                              "metrics": metrics})
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir),
                      "completed_folds": config["cv"]["folds"],
                      "selected_epoch_counts": [summaries[str(f)]["selected_epoch_count"]
                                                for f in range(1, config["cv"]["folds"] + 1)],
                      "oof_metrics": metrics["combined"]}, indent=2))
    return 0


def train_final_command(cv_run_id: str) -> int:
    if not cv_run_id or Path(cv_run_id).name != cv_run_id:
        raise ValueError("CV run id must be a single folder name")
    cv_dir = ROOT / "runs" / cv_run_id
    summary_path = cv_dir / "cv_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Completed CV run not found: {cv_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = int(summary.get("folds", 0))
    if summary.get("status") != "completed" or len(summary.get("fold_summaries", {})) != folds:
        raise RuntimeError("Final training requires exactly N completed CV folds")
    config = load_saved_config(cv_dir)
    selected = summary["selected_epoch_counts"]
    if len(selected) != config["cv"]["folds"]:
        raise RuntimeError("CV selected epoch counts do not match its saved fold count")
    epochs = int(math.floor(statistics.median(selected) + 0.5))
    final_id, final_dir = f"final_{cv_run_id}", ROOT / "runs" / f"final_{cv_run_id}"
    final_path = final_dir / "final.json"
    if final_path.is_file():
        print(final_path.read_text(encoding="utf-8"))
        return 0
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    rows = read_jsonl(PREP / "frozen_manifest.jsonl")
    validate_manifest(rows)
    if dataset_identity(rows) != split["dataset_identity"]:
        raise RuntimeError("Frozen point dataset changed since CV")
    train_rows = [row for row in rows if row["match"] in split["development_matches"]]
    if {row["match"] for row in train_rows} & set(split["held_out_matches"]):
        raise AssertionError("Held-out match leaked into final training")
    if final_dir.exists() and any(final_dir.iterdir()):
        raise RuntimeError(f"Final output exists and will not be overwritten: {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=True)
    result = run_training(train_rows, None, final_dir / "training", config,
                          int(config["train"]["seed"]) + 10001, epochs, final=True)
    final_info = {
        "status": "completed", "run_id": final_id, "model_kind": pm.MODEL_KIND,
        "completed_at": utc_now(), "source_cv_run": cv_run_id, "epochs": epochs,
        "selected_epoch_counts": selected, "last_checkpoint": result["last_checkpoint"],
        "last_sha256": result["last_sha256"], "checkpoint_hashes": {
            "pretrained_sha256": result["weight_transfer"]["source_checkpoint_sha256"],
            "final_sha256": result["last_sha256"],
        }, "held_out_matches": split["held_out_matches"],
        "development_matches": split["development_matches"], "dataset_identity": split["dataset_identity"],
        "training_frames": len(train_rows), "test_used_for_training": False,
    }
    write_json(final_dir / "resolved_config.json", resolved_settings(config))
    (final_dir / "resolved_settings.yaml").write_text(yaml.safe_dump(resolved_settings(config), sort_keys=False),
                                                      encoding="utf-8")
    write_json(final_dir / "dependencies.json", dependency_snapshot())
    write_json(final_path, final_info)
    print(json.dumps(final_info, indent=2))
    return 0


def final_context(final_run_id: str) -> tuple[dict, dict, Path]:
    if not final_run_id or Path(final_run_id).name != final_run_id:
        raise ValueError("Final run id must be a single folder name")
    final_dir = ROOT / "runs" / final_run_id
    info_path = final_dir / "final.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Completed point-only final run metadata is missing: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("status") != "completed" or info.get("run_id") != final_run_id \
            or info.get("model_kind") != pm.MODEL_KIND:
        raise RuntimeError("Final run is incomplete or uses an obsolete pose checkpoint")
    config = load_saved_config(final_dir)
    checkpoint = Path(info["last_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(ROOT.resolve()) or not checkpoint.is_file():
        raise FileNotFoundError(f"Custom final checkpoint is missing or outside the project: {checkpoint}")
    digest = sha256_file(checkpoint)
    if digest != info.get("last_sha256"):
        raise RuntimeError("Final checkpoint SHA256 does not match final.json")
    return info, config, checkpoint


def load_frozen_split() -> tuple[dict, list[dict]]:
    if not SPLIT_FILE.is_file() or not (PREP / "frozen_manifest.jsonl").is_file():
        raise FileNotFoundError("Frozen point manifest and held-out split are required")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split.get("split_version") != SPLIT_VERSION:
        raise RuntimeError("Obsolete pose split; point-only workflow requires split version 2")
    rows = read_jsonl(PREP / "frozen_manifest.jsonl")
    validate_manifest(rows)
    if dataset_identity(rows) != split["dataset_identity"]:
        raise RuntimeError("Frozen point manifest does not match the saved split identity")
    return split, rows


def evaluate_test_command(final_run_id: str) -> int:
    info, config, checkpoint = final_context(final_run_id)
    output_dir = ROOT / "runs" / final_run_id / "test_evaluation"
    if (output_dir / "metrics.json").exists():
        raise RuntimeError("The full held-out test set was already evaluated; refusing to overwrite or rerun")
    split, rows = load_frozen_split()
    if set(info["held_out_matches"]) != set(split["held_out_matches"]):
        raise RuntimeError("Final run held-out matches differ from the frozen split")
    test_rows = [row for row in rows if row["match"] in split["held_out_matches"]]
    if {row["match"] for row in test_rows} != set(split["held_out_matches"]):
        raise RuntimeError("Held-out point manifest is incomplete")
    positions, errors = predict_and_score(checkpoint, test_rows, config)
    metrics = write_evaluation(output_dir, positions, errors)
    write_json(output_dir / "test_evaluation.json", {
        "model_kind": pm.MODEL_KIND, "created_at": utc_now(), "weights_sha256": sha256_file(checkpoint),
        "dataset_identity": split["dataset_identity"], "held_out_matches": split["held_out_matches"],
        "frames": len(test_rows), "metrics": metrics,
    })
    print(json.dumps({"output_dir": str(output_dir), "held_out_matches": split["held_out_matches"],
                      "metrics": metrics["combined"]}, indent=2))
    return 0


def annotations_by_frame(rows: list[dict], geometry, width: int, height: int) -> dict:
    annotations = {}
    for row in rows:
        frame = int(row["clip_frame_num"])
        if frame in annotations:
            raise ValueError(f"Duplicate annotation frame {frame} in {stable_key(row)}")
        refs = {}
        for player in ("A", "B"):
            image = (float(row[f"{player}_native_x"]), float(row[f"{player}_native_y"]))
            world = projected_world(geometry, image, width, height)
            refs[player] = {"image": image, "world": world}
        annotations[frame] = refs
    return annotations


def court_panel(height: int, predicted: dict, refs: dict, previous: dict,
                frame_number: int, halves: dict) -> np.ndarray:
    panel_width = 600
    panel = np.full((height, panel_width, 3), 248, dtype=np.uint8)
    scale = min((panel_width - 70) / COURT_WIDTH_M, (height - 190) / COURT_LENGTH_M)
    court_width, court_height = int(COURT_WIDTH_M * scale), int(COURT_LENGTH_M * scale)
    left, top = (panel_width - court_width) // 2, (height - court_height) // 2

    def pixel(point):
        return int(round(left + point[0] * scale)), int(round(top + (COURT_LENGTH_M - point[1]) * scale))

    cv2.putText(panel, "COURT (metres)", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
    for start, end in court_world_lines():
        cv2.line(panel, pixel(start), pixel(end), (85, 85, 85),
                 2 if start[1] == end[1] == NET_Y_M else 1)
    for player, point in predicted.items():
        if player in previous:
            cv2.line(panel, pixel(previous[player]), pixel(point), REVIEW_COLORS[player], 3, cv2.LINE_AA)
        cv2.circle(panel, pixel(point), 7, REVIEW_COLORS[player], -1, cv2.LINE_AA)
        cv2.putText(panel, player, (pixel(point)[0] + 9, pixel(point)[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, REVIEW_COLORS[player], 2)
    for player, point in refs.items():
        if point is not None:
            cv2.drawMarker(panel, pixel(point), REVIEW_COLORS[player], cv2.MARKER_CROSS, 18, 2)
    cv2.putText(panel, f"A {halves['A']}", (24, height - 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, REVIEW_COLORS["A"], 2)
    cv2.putText(panel, f"B {halves['B']}", (150, height - 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, REVIEW_COLORS["B"], 2)
    if refs:
        cv2.putText(panel, f"reference frame {frame_number}", (24, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)
    return panel


def verify_review_video(path: Path, expected_frames: int, fps: float,
                        dimensions: tuple[int, int]) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen rendered video: {path}")
    info = {"fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_frames": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))}
    decoded = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
    finally:
        cap.release()
    if (decoded != expected_frames or (info["width"], info["height"]) != dimensions or
            not math.isclose(info["fps"], fps, rel_tol=0, abs_tol=1e-3) or
            (info["reported_frames"] > 0 and info["reported_frames"] != expected_frames)):
        raise RuntimeError(f"Rendered video verification failed: {info}, decoded={decoded}")
    return {**info, "decoded_frames": decoded}


def render_review_rally(rally: dict, output_dir: Path, model, geometry,
                        config: dict, device: torch.device) -> dict:
    rows = sorted(rally["rows"], key=lambda row: int(row["clip_frame_num"]))
    template = rows[0]
    width, height, fps = int(template["width"]), int(template["height"]), float(template["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS in {rally['key']}")
    halves = {p: template[f"{p}_half"] for p in ("A", "B")}
    if {halves["A"], halves["B"]} != {"far", "near"}:
        raise ValueError(f"Invalid frozen A/B mapping in {rally['key']}")
    annotations = annotations_by_frame(rows, geometry, width, height)
    output_dir.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(rally["clip_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open selected rally: {rally['clip_path']}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_width, source_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if ((source_width, source_height) != (width, height) or
            not math.isclose(source_fps, fps, rel_tol=0, abs_tol=1e-5)):
        cap.release()
        raise RuntimeError("Video metadata differs from frozen provenance")
    video_path = output_dir / "review.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
                             (width + 600, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create review video: {video_path}")
    positions, errors, previous = [], [], {}
    frame_index = 0
    input_width, input_height = config["model"]["input_width"], config["model"]["input_height"]
    model.eval()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tensor = pm.image_tensor(frame, input_width, input_height).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)[0]
            decoded = pm.decode_heatmaps(logits, input_width, input_height, width, height)
            channels = {"far": decoded[pm.CHANNELS["far"]], "near": decoded[pm.CHANNELS["near"]]}
            court_points, references = {}, {}
            for player in ("A", "B"):
                half, prediction = halves[player], channels[halves[player]]
                x, y = prediction["x_native"], prediction["y_native"]
                world = projected_world(geometry, (x, y), width, height)
                if world is not None:
                    court_points[player] = world
                color = REVIEW_COLORS[player]
                point_px = (int(round(x)), int(round(y)))
                cv2.circle(frame, point_px, 8, color, -1, cv2.LINE_AA)
                cv2.putText(frame, f"{player} heatmap score {prediction['heatmap_score']:.4f}",
                            (max(10, point_px[0] + 10), max(24, point_px[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
                status = "available" if world is not None else "invalid_court_projection"
                position = {
                    **{field: template[field] for field in KEY_FIELDS}, "clip_frame_num": frame_index,
                    "clip_time_seconds": frame_index / source_fps, "player": player,
                    "prediction_status": status, "available": world is not None,
                    "x_native": x, "y_native": y,
                    "x_court_m": None if world is None else world[0],
                    "y_court_m": None if world is None else world[1],
                    "heatmap_score": prediction["heatmap_score"], "court_half": half,
                    "reference_x_native": None, "reference_y_native": None,
                }
                positions.append(position)
                if frame_index in annotations:
                    reference = annotations[frame_index][player]
                    references[player] = reference["world"]
                    ref_x, ref_y = reference["image"]
                    position["reference_x_native"], position["reference_y_native"] = ref_x, ref_y
                    error_px = math.hypot(x - ref_x, y - ref_y) if world is not None and reference["world"] else None
                    error_m = (float(np.linalg.norm(np.asarray(world) - np.asarray(reference["world"])))
                               if world is not None and reference["world"] else None)
                    errors.append({
                        **{field: template[field] for field in KEY_FIELDS}, "clip_frame_num": frame_index,
                        "clip_time_seconds": frame_index / source_fps, "player": player,
                        "prediction_status": status if error_m is not None else "invalid_court_projection",
                        "available": error_m is not None, "error_px": error_px, "error_m": error_m,
                    })
            cv2.putText(frame, f"frame {frame_index}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (20, 20, 20), 3)
            refs_for_frame = references if frame_index in annotations else {}
            panel = court_panel(height, court_points, refs_for_frame, previous, frame_index, halves)
            writer.write(np.hstack((frame, panel)))
            previous = court_points
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    if frame_index == 0 or (expected > 0 and frame_index != expected):
        raise RuntimeError(f"Decoded {frame_index} frames; source metadata reported {expected}")
    if any(frame >= frame_index for frame in annotations):
        raise RuntimeError(f"Annotated frame lies outside decoded clip: {rally['key']}")
    if len(positions) != frame_index * 2 or len(errors) != len(annotations) * 2:
        raise RuntimeError("Review exports do not align with decoded frames and annotation timestamps")
    write_csv(output_dir / "player_positions.csv", positions, [
        *KEY_FIELDS, "clip_frame_num", "clip_time_seconds", "player", "prediction_status", "available",
        "x_native", "y_native", "x_court_m", "y_court_m", "heatmap_score", "court_half",
        "reference_x_native", "reference_y_native",
    ])
    write_csv(output_dir / "shot_errors.csv", errors, [
        *KEY_FIELDS, "clip_frame_num", "clip_time_seconds", "player", "prediction_status",
        "available", "error_px", "error_m",
    ])
    metrics = {
        "diagnostic_scope": "selected-rally diagnostics; not the full held-out test score",
        "combined": error_summary(errors),
        "per_player": {p: error_summary([row for row in errors if row["player"] == p]) for p in ("A", "B")},
        "annotated_frames": sorted(annotations), "decoded_frames": frame_index,
        "source_fps": source_fps, "source_dimensions": [width, height],
        "review_dimensions": [width + 600, height],
    }
    write_json(output_dir / "metrics.json", metrics)
    video_info = verify_review_video(video_path, frame_index, source_fps, (width + 600, height))
    return {"key": list(rally["key"]), "directory": str(output_dir), "video": str(video_path),
            "frames": frame_index, "video_info": video_info, "metrics": metrics}


def review_rallies(rows: list[dict], held_out: set[str], seed: int = 42) -> list[dict]:
    if any(row.get("dataset") != "set-1" for row in rows):
        raise ValueError("Frozen manifest contains rows outside cleaned_rallies/set-1")
    groups = defaultdict(list)
    for row in rows:
        groups[stable_key(row)].append(row)
    by_match, preferred_key = defaultdict(list), None
    preferred_path = os.path.normcase(str(PREFERRED_REVIEW_CLIP.resolve()))
    for key, rally_rows in sorted(groups.items()):
        if key[0] != "set-1" or key[1] not in held_out:
            continue
        rally_rows.sort(key=lambda row: int(row["clip_frame_num"]))
        frames = [int(row["clip_frame_num"]) for row in rally_rows]
        paths = {os.path.normcase(str(Path(row["clip_path"]).resolve())) for row in rally_rows}
        mappings = {(row.get("A_half"), row.get("B_half")) for row in rally_rows}
        metadata = {(int(row["width"]), int(row["height"]), float(row["fps"])) for row in rally_rows}
        if not frames or min(frames) < 0 or len(frames) != len(set(frames)):
            raise ValueError(f"Invalid or duplicate frozen frame indices for {key}")
        if len(paths) != 1 or len(mappings) != 1 or len(metadata) != 1:
            raise ValueError(f"Provenance or A/B mapping changes within {key}")
        halves = next(iter(mappings))
        if set(halves) != {"far", "near"}:
            raise ValueError(f"Invalid A/B court-half mapping in {key}")
        width, height, fps = next(iter(metadata))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video metadata in {key}")
        for row in rally_rows:
            for player in ("A", "B"):
                x, y = float(row[f"{player}_native_x"]), float(row[f"{player}_native_y"])
                if not np.isfinite([x, y]).all() or not 0 <= x < width or not 0 <= y < height:
                    raise ValueError(f"Invalid {player} point in frozen rally {key}")
        clip = next(iter(paths))
        if clip == preferred_path:
            preferred_key = key
        by_match[key[1]].append({"key": key, "clip_path": clip, "rows": rally_rows})
    matches = sorted(by_match)
    if not matches:
        return []
    rng = random.Random(seed)
    if preferred_key is not None:
        selected_matches = [preferred_key[1]]
        remaining = [match for match in matches if match != preferred_key[1]]
        rng.shuffle(remaining)
        selected_matches.extend(remaining[:2])
    else:
        rng.shuffle(matches)
        selected_matches = matches[:3]
    selected = []
    for match in selected_matches:
        options = sorted(by_match[match], key=lambda rally: rally["key"])
        preferred = next((rally for rally in options if rally["key"] == preferred_key), None)
        selected.append(preferred if preferred is not None else rng.choice(options))
    return selected


def review_test_context(final_run_id: str) -> tuple[dict, list[str]]:
    context, blockers = {"final_dir": ROOT / "runs" / final_run_id, "selected": []}, []
    try:
        info, config, checkpoint = final_context(final_run_id)
        context.update(final_info=info, config=config, checkpoint=checkpoint,
                       checkpoint_hash=sha256_file(checkpoint))
        split, rows = load_frozen_split()
        context.update(split=split, rows=rows, dataset_identity=split["dataset_identity"])
        if info.get("test_used_for_training") is not False:
            blockers.append("final run does not confirm held-out exclusion from training")
        if set(info.get("held_out_matches", [])) != set(split["held_out_matches"]):
            blockers.append("final run held-out assignments differ from the frozen split")
        if set(info.get("development_matches", [])) != set(split["development_matches"]):
            blockers.append("final run development assignments differ from the frozen split")
        selected = review_rallies(rows, set(split["held_out_matches"]), seed=42)
        if len({rally["key"][1] for rally in selected}) != len(selected):
            blockers.append("review selection contains multiple rallies from one match")
        if any(rally["key"][1] not in set(split["held_out_matches"]) for rally in selected):
            blockers.append("review selection contains a development match")
        for rally in selected:
            if not Path(rally["clip_path"]).is_file():
                blockers.append(f"selected held-out clip is missing: {rally['clip_path']}")
            import_geometry(rally["key"][1])
        if not selected:
            blockers.append("frozen held-out dataset contains no eligible rallies")
        context["selected"] = selected
    except Exception as error:
        blockers.append(str(error))
    return context, blockers


def review_test_command(final_run_id: str, check_only: bool = False) -> int:
    context, blockers = review_test_context(final_run_id)
    if not blockers:
        try:
            _, checkpoint = pm.load_checkpoint(context["checkpoint"])
            if checkpoint.get("model_kind") != pm.MODEL_KIND:
                raise RuntimeError("checkpoint is not the custom point-only model")
        except Exception as error:
            blockers.append(f"custom point checkpoint preflight failed: {error}")
    selected = context.get("selected", [])
    report = {
        "ready": not blockers, "run_id": final_run_id,
        "dataset_identity": context.get("dataset_identity"),
        "selected_rallies": [{"key": list(item["key"]), "clip_path": item["clip_path"]} for item in selected],
        "selected_match_count": len({item["key"][1] for item in selected}),
        "shortfall_from_three": max(0, 3 - len(selected)), "blockers": blockers,
    }
    if check_only or blockers:
        print(json.dumps(report, indent=2))
        return 0 if not blockers else 2
    final_dir = context["final_dir"]
    output_root = final_dir / "selected_rally_reviews"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"review_test_{stamp}"
    suffix = 2
    while output_dir.exists():
        output_dir = output_root / f"review_test_{stamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True)
    settings = resolved_settings(context["config"])
    write_json(output_dir / "selection.json", {
        "diagnostic_scope": "selected-rally diagnostics; not the full held-out test score",
        "seed": 42, "final_run_id": final_run_id, "checkpoint": str(context["checkpoint"]),
        "checkpoint_sha256": context["checkpoint_hash"], "dataset_identity": context["dataset_identity"],
        "held_out_matches": sorted(context["split"]["held_out_matches"]),
        "resolved_settings": settings,
        "selected_rallies": [{"key": list(item["key"]), "clip_path": item["clip_path"]} for item in selected],
        "shortfall_from_three": max(0, 3 - len(selected)), "selection_saved_before_inference": True,
    })
    model, _ = pm.load_checkpoint(context["checkpoint"])
    device = resolve_device(context["config"])
    model.to(device).eval()
    rendered = []
    for rally in selected:
        rendered.append(render_review_rally(
            rally, output_dir / safe_id(rally["key"]), model,
            import_geometry(rally["key"][1]), context["config"], device,
        ))
    write_json(output_dir / "render_summary.json", {
        "final_run_id": final_run_id, "checkpoint_sha256": context["checkpoint_hash"],
        "dataset_identity": context["dataset_identity"], "selected_rallies": rendered,
    })
    print(json.dumps({"review_directory": str(output_dir), **report,
                      "rendered": [{"key": row["key"], "video": row["video"],
                                    "frames": row["frames"], "metrics": row["metrics"]["combined"]}
                                   for row in rendered]}, indent=2))
    return 0


def overlay_command(final_run_id: str, match: str, game: int, rally_id: int) -> int:
    info, config, checkpoint = final_context(final_run_id)
    split, rows = load_frozen_split()
    key = ("set-1", match, str(game), str(rally_id))
    selected_rows = [row for row in rows if stable_key(row) == key]
    if not selected_rows:
        raise ValueError(f"Rally is not part of the accepted frozen point dataset: {key}")
    rally = {"key": key, "clip_path": selected_rows[0]["clip_path"], "rows": selected_rows}
    model, _ = pm.load_checkpoint(checkpoint)
    model.to(resolve_device(config)).eval()
    output = ROOT / "runs" / final_run_id / "overlays" / safe_id(key)
    result = render_review_rally(rally, output, model, import_geometry(match), config,
                                 resolve_device(config))
    print(json.dumps({"final_run_id": final_run_id, "held_out": match in split["held_out_matches"],
                      "checkpoint_sha256": info["last_sha256"], "overlay": result}, indent=2))
    return 0


def run_model_checks(config: dict) -> dict:
    weights = ensure_pretrained_weights()
    device = resolve_device(config)
    model, transfer = pm.initialize_from_yolo11n(weights, 901)
    contract = pm.one_keypoint_output(model, 160, 128, "cpu")
    second, second_transfer = pm.initialize_from_yolo11n(weights, 902)
    if not torch.equal(next(model.backbone.parameters()), next(second.backbone.parameters())):
        raise AssertionError("Fold initialization did not preserve the same pretrained backbone")
    if pm.tensor_hash(pm.head_state_dict(model)) == pm.tensor_hash(pm.head_state_dict(second)):
        raise AssertionError("Independent initialization reused the same random heatmap head")
    model.to(device).train()
    optimizer = build_optimizer(model, config["train"])
    inputs = torch.rand(2, 3, 128, 160, device=device)
    targets = pm.heatmap_target([(72, 48), (124, 82)], 32, 40,
                                config["model"]["gaussian_sigma"]).unsqueeze(0).repeat(2, 1, 1, 1).to(device)
    before = next(model.heatmaps.parameters()).detach().clone()
    use_amp = config["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        loss = pm.spatial_cross_entropy(model(inputs), targets)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    if model.heatmaps.weight.grad is None or model.backbone[0].conv.weight.grad is None:
        raise AssertionError("Gradient did not reach both the custom head and YOLO11n backbone")
    if torch.equal(before, model.heatmaps.weight.detach()):
        raise AssertionError("Synthetic optimization did not update the custom head")
    checks = ROOT / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    temp = checks / "checkpoint_roundtrip.pt"
    checkpoint_hash = pm.save_checkpoint(temp, model, resolved_settings(config), {"synthetic": True})
    reloaded, payload = pm.load_checkpoint(temp)
    if pm.tensor_hash(reloaded.state_dict()) != pm.tensor_hash(model.state_dict()):
        raise AssertionError("Checkpoint reload changed model parameters")
    temp.unlink(missing_ok=True)
    summary = {
        "status": "passed", "created_at": utc_now(), "system_python": sys.executable,
        "dependencies": dependency_snapshot(), "device": str(device), "amp_effective": use_amp,
        "transfer": transfer, "second_independent_initialization": second_transfer,
        "model_output": contract, "synthetic_optimization_loss": float(loss.detach().float().item()),
        "head_gradient": True, "backbone_gradient": True,
        "checkpoint_roundtrip_sha256": checkpoint_hash,
        "checkpoint_kind": payload["model_kind"], "resolved_settings": resolved_settings(config),
    }
    write_json(checks / "check_summary.json", summary)
    return summary


def check_command(config: dict) -> int:
    print(json.dumps(run_model_checks(config), indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Point-only YOLO11n player-localisation experiment")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "split", "train-cv"):
        command = commands.add_parser(name)
        command.add_argument("--config", help="YAML settings path")
        if name in {"split", "train-cv"}:
            command.add_argument("--folds", type=int, help="Override YAML CV folds")
    commands.add_parser("review")
    final = commands.add_parser("train-final")
    final.add_argument("--run-id", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--split", choices=("test",), required=True)
    overlay = commands.add_parser("overlay")
    overlay.add_argument("--run-id", required=True)
    overlay.add_argument("--match", required=True)
    overlay.add_argument("--game", type=int, required=True)
    overlay.add_argument("--rally", type=int, required=True)
    test_review = commands.add_parser("review-test")
    test_review.add_argument("--run-id", required=True)
    test_review.add_argument("--check", action="store_true", help="Preflight only; does not run training smoke")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "review":
            return review_command()
        if args.command == "train-final":
            return train_final_command(args.run_id)
        if args.command == "evaluate":
            return evaluate_test_command(args.run_id)
        if args.command == "overlay":
            return overlay_command(args.run_id, args.match, args.game, args.rally)
        if args.command == "review-test":
            return review_test_command(args.run_id, args.check)
        config = load_config(args.config, getattr(args, "folds", None))
        if args.command == "check":
            return check_command(config)
        if args.command == "prepare":
            return prepare_command(config)
        if args.command == "split":
            return split_command(config)
        if args.command == "train-cv":
            return train_cv_command(config)
        raise ValueError(f"Unknown command: {args.command}")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
