from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
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
import yaml

sys.path.insert(0, str(REPO / "monotrack_test"))
from reconstruction import load_court_geometry  # noqa: E402

SOURCE = REPO / "cleaned_rallies" / "set-1"
HOMOGRAPHY = REPO / "set-1" / "homography.csv"
CONFIG_DEFAULT = ROOT / "config.yaml"
WEIGHTS = ROOT / "weights" / "yolo11n-pose.pt"
MODEL_YAML = ROOT / "model" / "yolo11n-pose-kpt1.yaml"
PREP = ROOT / "data" / "candidate"
SPLIT_FILE = ROOT / "splits" / "split.json"
EXCLUDED_MATCHES = {
    "CHOU_Tien_Chen_Jonatan_CHRISTIE_Sudirman_Cup_2019_Quarter-finals",
    "CHOU_Tien_Chen_NG_Ka_Long_Angus_Sudirman_Cup_2019_Group_Stage",
}
REFERENCE_WIDTH, REFERENCE_HEIGHT = 1280, 720
COURT_HALF_Y_M = 6.7
PREP_IMGSZ = 640
BOX_BATCH = 16
KEY_FIELDS = ("dataset", "match", "game", "original_rally_id")


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
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except ImportError as error:
        raise RuntimeError(
            "Install the pinned dependency with the project Python: "
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from error
    if ultralytics.__version__ != "8.4.175":
        raise RuntimeError(
            f"Ultralytics 8.4.175 is required; found {ultralytics.__version__}"
        )
    import ultralytics.utils as utils
    (ROOT / "weights").mkdir(parents=True, exist_ok=True)
    utils.SETTINGS.update({
        "datasets_dir": str(ROOT / "data"),
        "weights_dir": str(ROOT / "weights"),
        "runs_dir": str(ROOT / "runs"),
    })
    utils.WEIGHTS_DIR = ROOT / "weights"
    return torch, ultralytics, YOLO, DEFAULT_CFG_DICT


def load_config(path: str | None = None, folds_override: int | None = None) -> dict:
    config_path = Path(path).expanduser().resolve() if path else CONFIG_DEFAULT
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"cv", "preparation", "inference", "train"}:
        raise ValueError("Config must contain exactly cv, preparation, inference, and train sections")
    for section in ("cv", "preparation", "inference", "train"):
        if not isinstance(config[section], dict):
            raise ValueError(f"Config section {section!r} must be a mapping")
    if folds_override is not None:
        config["cv"]["folds"] = folds_override

    allowed_cv = {"folds", "seed"}
    allowed_preparation = {"confidence"}
    allowed_inference = {"confidence", "keypoint_confidence", "iou"}
    for section, allowed in (
        ("cv", allowed_cv), ("preparation", allowed_preparation), ("inference", allowed_inference)
    ):
        unknown = set(config[section]) - allowed
        if unknown:
            raise ValueError(f"Unsupported {section} settings: {sorted(unknown)}")
    folds = config["cv"].get("folds")
    seed = config["cv"].get("seed")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("cv.folds must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("cv.seed must be a nonnegative integer")

    for section, key in (
        ("preparation", "confidence"),
        ("inference", "confidence"),
        ("inference", "keypoint_confidence"),
    ):
        value = config[section].get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{section}.{key} must be finite and in [0, 1]")
    iou = config["inference"].get("iou")
    if not isinstance(iou, (int, float)) or not math.isfinite(iou) or not 0 < iou <= 1:
        raise ValueError("inference.iou must be finite and in (0, 1]")

    torch, _, _, native_args = load_ultralytics()
    train = config["train"]
    controlled = {
        "data", "model", "project", "name", "task", "mode", "resume", "pretrained",
        "save_dir", "exist_ok", "cfg", "weights",
    }
    unknown = set(train) - set(native_args)
    if unknown:
        raise ValueError(f"Unknown native Ultralytics training arguments: {sorted(unknown)}")
    invalid = set(train) & controlled
    if invalid:
        raise ValueError(f"Pipeline-controlled training arguments cannot be overridden: {sorted(invalid)}")
    for key in ("epochs", "patience", "batch", "imgsz", "workers", "close_mosaic"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"train.{key} must be an integer")
    if train["epochs"] < 1 or train["patience"] < 0 or train["batch"] == 0:
        raise ValueError("epochs must be positive, patience nonnegative, and batch nonzero")
    if train["batch"] < -1 or train["workers"] < 0 or train["close_mosaic"] < 0:
        raise ValueError("batch must be -1 or positive; workers and close_mosaic must be nonnegative")
    if train["imgsz"] < 32 or train["imgsz"] % 32:
        raise ValueError("train.imgsz must be a positive multiple of 32")
    if not isinstance(train.get("optimizer"), str) or not train["optimizer"].strip():
        raise ValueError("train.optimizer must be a nonempty string")
    for key in ("lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
                "warmup_bias_lr", "box", "cls", "dfl", "pose", "kobj"):
        if key not in train:
            continue
        value = train[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"train.{key} must be finite")
        if key in {"momentum", "warmup_momentum"} and not 0 <= value <= 1:
            raise ValueError(f"train.{key} must be in [0, 1]")
        if key not in {"momentum", "warmup_momentum"} and value < 0:
            raise ValueError(f"train.{key} must be nonnegative")
    for key in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "bgr",
                "erasing"):
        if key not in train:
            continue
        value = train[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"train.{key} must be finite and nonnegative")
        if key in {"hsv_h", "hsv_s", "hsv_v", "flipud", "fliplr", "mosaic", "mixup",
                   "copy_paste", "bgr", "erasing"} and value > 1:
            raise ValueError(f"train.{key} must be in [0, 1]")
    if not isinstance(train.get("amp"), bool):
        raise ValueError("train.amp must be true or false")
    if "seed" in train and (isinstance(train["seed"], bool) or not isinstance(train["seed"], int)
                            or train["seed"] < 0):
        raise ValueError("train.seed must be a nonnegative integer")
    freeze = train.get("freeze")
    if freeze is not None and not isinstance(freeze, (int, list)):
        raise ValueError("train.freeze must be an integer, a list of layer indices, or null")
    if not torch.cuda.is_available() and str(train.get("device", "")).strip() not in {"", "cpu"}:
        raise ValueError("The configured device requires CUDA, but CUDA is unavailable")
    config["_path"] = str(config_path)
    return config


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
    text = str(value or "").strip().lower()
    return text in {"1", "1.0", "true", "yes", "flaw"}


def align_references(annotation: dict) -> dict[str, tuple[float, float]]:
    hitter = str(annotation.get("player", "")).strip().upper()
    if hitter not in {"A", "B"}:
        raise ValueError("unknown player identity")
    player = (
        parse_float(annotation.get("player_location_x"), "player_location_x"),
        parse_float(annotation.get("player_location_y"), "player_location_y"),
    )
    opponent = (
        parse_float(annotation.get("opponent_location_x"), "opponent_location_x"),
        parse_float(annotation.get("opponent_location_y"), "opponent_location_y"),
    )
    return {"A": player if hitter == "A" else opponent, "B": opponent if hitter == "A" else player}


def native_point(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    return point[0] * width / REFERENCE_WIDTH, point[1] * height / REFERENCE_HEIGHT


def court_half(geometry, point_native: tuple[float, float], width: int, height: int) -> str:
    reference_point = (point_native[0] * REFERENCE_WIDTH / width,
                       point_native[1] * REFERENCE_HEIGHT / height)
    world = geometry.annotation_floor_to_world(reference_point)
    if not np.isfinite(world).all():
        raise ValueError("proxy floor point maps to nonfinite court coordinates")
    return "far" if float(world[1]) < COURT_HALF_Y_M else "near"


def checked_clip_path(match_dir: Path, filename: str) -> Path:
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


def inspect_rally(match_dir: Path, inventory: dict, cleanup_rows: list[dict], geometries: dict,
                  exclusions: list[dict]) -> dict:
    key = stable_key(inventory)
    rally_id = safe_id(key)

    def exclude(reason: str, detail: str = "", frame: int | None = None) -> None:
        exclusions.append({
            "dataset": key[0], "match": key[1], "game": key[2], "original_rally_id": key[3],
            "clip_frame_num": "" if frame is None else frame, "reason": reason, "detail": detail,
        })

    reasons = []
    if len(cleanup_rows) != 1:
        reasons.append("missing_cleanup_record" if not cleanup_rows else "ambiguous_cleanup_records")
        exclude(reasons[-1], f"record_count={len(cleanup_rows)}")
    cleanup = cleanup_rows[0] if len(cleanup_rows) == 1 else {}
    flags = str(inventory.get("review_flags", "") or "").strip()
    if flags:
        reasons.append("rally_inventory_review_flags")
        exclude(reasons[-1], flags)
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

    annotations_path = checked_clip_path(match_dir, str(inventory.get("filename", ""))).with_suffix(".csv")
    clip_path = checked_clip_path(match_dir, str(inventory.get("filename", "")))
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
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        first_hit = parse_int(inventory.get("first_hit_frame"), "first_hit_frame")
        last_hit = parse_int(inventory.get("last_hit_frame"), "last_hit_frame")
        if annotation_fps <= 0 or first_hit < 0 or last_hit < first_hit:
            raise ValueError("invalid annotation FPS or first/last frame bounds")
    except Exception as error:
        reasons.append("invalid_rally_timing_metadata")
        exclude(reasons[-1], str(error))
        return {"key": key, "id": rally_id, "status": "excluded", "reasons": reasons,
                "annotations": annotations, "frames": [], "clip_path": str(clip_path)}

    frames = []
    previous_frame = None
    sides = {"A": set(), "B": set()}
    source_frames = []
    for row in annotations:
        frame = None
        try:
            frame = parse_int(row.get("clip_frame_num"), "clip_frame_num")
            if frame < 0 or frame >= frame_count:
                raise ValueError(f"clip frame {frame} outside [0, {frame_count})")
            if previous_frame is not None and frame == previous_frame:
                raise ValueError(f"duplicate clip_frame_num {frame}")
            if previous_frame is not None and frame < previous_frame:
                raise ValueError(f"decreasing clip_frame_num {frame} after {previous_frame}")
            previous_frame = frame
            clip_time = parse_float(row.get("clip_time_seconds"), "clip_time_seconds")
            if not math.isclose(clip_time, frame / fps, rel_tol=0, abs_tol=0.00001):
                raise ValueError(f"clip_time_seconds mismatch at frame {frame}")
            original_frame = parse_int(row.get("frame_num"), "frame_num")
            source_time = parse_float(row.get("source_time_seconds"), "source_time_seconds")
            expected_source = original_frame / annotation_fps + time_offset
            if not math.isclose(source_time, expected_source, rel_tol=0, abs_tol=0.00001):
                raise ValueError(f"source_time_seconds mismatch at clip frame {frame}")
            mapped = math.floor((source_time - clean_start) * fps + 0.5)
            if mapped != frame:
                raise ValueError(f"source-time mapping gives frame {mapped}, label says {frame}")
            source_frames.append(original_frame)
            references = align_references(row)
            for player, point in references.items():
                x, y = point
                if not 0 <= x < REFERENCE_WIDTH or not 0 <= y < REFERENCE_HEIGHT:
                    raise ValueError(f"{player} reference coordinate outside 1280x720: {point}")
                native = native_point(point, width, height)
                side = court_half(geometry, native, width, height)
                sides[player].add(side)
            frames.append({"row": row, "clip_frame_num": frame, "clip_time_seconds": clip_time,
                           "source_time_seconds": source_time, "frame_num": original_frame,
                           "references": references})
        except Exception as error:
            reason = str(error)
            code = "invalid_frame_or_time_correspondence" if (
                any(token in reason for token in ("frame", "time", "FPS", "bounds"))
            ) else "invalid_reference_or_identity"
            reasons.append(code)
            exclude(code, reason, frame)
        player = str(row.get("player", "") or "").strip().upper()
        if player not in {"A", "B"}:
            if "unknown_player_identity" not in reasons:
                reasons.append("unknown_player_identity")
                exclude("unknown_player_identity", f"player={row.get('player', '')}", frame)
        if flaw_is_set(row.get("flaw", "")) and "dataset_flaw_flag" not in reasons:
            reasons.append("dataset_flaw_flag")
            exclude("dataset_flaw_flag", f"flaw={row.get('flaw', '')}", frame)

    if source_frames:
        if min(source_frames) != first_hit or max(source_frames) != last_hit:
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
        "key": key, "id": rally_id, "status": "excluded" if reasons else "pre_box_pass",
        "reasons": sorted(set(reasons)), "annotations": annotations, "frames": frames,
        "clip_path": str(clip_path), "fps": fps, "width": width, "height": height,
        "frame_count": frame_count, "A_half": next(iter(sides["A"])) if len(sides["A"]) == 1 else "",
        "B_half": next(iter(sides["B"])) if len(sides["B"]) == 1 else "",
        "geometry": geometry,
    }


def enumerate_rallies(exclusions: list[dict]) -> list[tuple[Path, dict, dict]]:
    result = []
    for match_dir in sorted(SOURCE.iterdir()) if SOURCE.is_dir() else []:
        if not match_dir.is_dir():
            continue
        if match_dir.name in EXCLUDED_MATCHES:
            exclusions.append({
                "dataset": "set-1", "match": match_dir.name, "game": "",
                "original_rally_id": "", "clip_frame_num": "",
                "reason": "permanent_match_exclusion", "detail": "excluded by experiment specification",
            })
            continue
        inventory_path = match_dir / "rallies.csv"
        report_path = match_dir / "cleanup_report.csv"
        if not inventory_path.is_file():
            exclusions.append({
                "dataset": "set-1", "match": match_dir.name, "game": "",
                "original_rally_id": "", "clip_frame_num": "",
                "reason": "missing_rally_inventory", "detail": str(inventory_path),
            })
            continue
        report_map = cleanup_index(report_path)
        for record in read_csv(inventory_path):
            result.append((match_dir, record, {"cleanup_rows": report_map.get(stable_key(record), [])}))
    return result


def box_associations(result, points: dict[str, tuple[float, float]], confidence: float) -> tuple[dict, list[str]]:
    boxes_obj = result.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return {}, ["missing_person_boxes"]
    boxes = boxes_obj.xyxy.detach().cpu().numpy()
    scores = boxes_obj.conf.detach().cpu().numpy()
    classes = boxes_obj.cls.detach().cpu().numpy().astype(int)
    eligible = [(index, box) for index, (box, score, cls) in enumerate(zip(boxes, scores, classes))
                if cls == 0 and score >= confidence and np.isfinite(box).all()]
    found, reasons = {}, []
    for player, point in points.items():
        matches = [(index, box) for index, box in eligible
                   if box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]]
        if len(matches) != 1:
            reasons.append(f"{player}_box_association_count_{len(matches)}")
        else:
            found[player] = tuple(float(v) for v in matches[0][1])
            found[player + "_box_index"] = matches[0][0]
    if len(found) == 4 and found["A_box_index"] == found["B_box_index"]:
        reasons.append("players_associated_to_same_box")
    return found, reasons


def ensure_weights() -> Path:
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.is_file():
        _, _, _, _ = load_ultralytics()
        from ultralytics.utils.downloads import attempt_download_asset
        downloaded = Path(attempt_download_asset(
            str(WEIGHTS), repo="ultralytics/assets", release="v8.4.0"
        ))
        if downloaded.resolve() != WEIGHTS.resolve() and downloaded.is_file():
            shutil.copy2(downloaded, WEIGHTS)
    if not WEIGHTS.is_file() or WEIGHTS.stat().st_size < 1_000_000:
        raise RuntimeError(f"Official yolo11n-pose.pt checkpoint is missing or incomplete: {WEIGHTS}")
    return WEIGHTS


def ensure_model_yaml() -> Path:
    if MODEL_YAML.exists():
        text = MODEL_YAML.read_text(encoding="utf-8")
        if "nc: 1" in text and "kpt_shape: [1, 3]" in text and "scale: n" in text:
            return MODEL_YAML
        raise RuntimeError(f"Custom model YAML already exists but does not match this experiment: {MODEL_YAML}")
    _, ultralytics, _, _ = load_ultralytics()
    source = Path(ultralytics.__file__).parent / "cfg" / "models" / "11" / "yolo11-pose.yaml"
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^nc:\s*\d+.*$", "nc: 1", text, count=1)
    text = re.sub(r"(?m)^kpt_shape:\s*\[[^\]]+\].*$", "kpt_shape: [1, 3]", text, count=1)
    text = text.replace("kpt_shape: [1, 3]", "kpt_shape: [1, 3]\nscale: n", 1)
    if "nc: 1" not in text or "kpt_shape: [1, 3]" not in text or "scale: n" not in text:
        raise RuntimeError("Could not derive the one-keypoint YOLO11 nano architecture")
    MODEL_YAML.parent.mkdir(parents=True, exist_ok=True)
    MODEL_YAML.write_text(text, encoding="utf-8")
    return MODEL_YAML


def init_custom_model(seed: int | None = None):
    torch, _, YOLO, _ = load_ultralytics()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    ensure_weights()
    model = YOLO(str(ensure_model_yaml()), task="pose")
    model.load(str(WEIGHTS))
    head = model.model.model[-1]
    if tuple(head.kpt_shape) != (1, 3) or int(head.nc) != 1:
        raise RuntimeError(f"Custom model contract is wrong: nc={head.nc}, kpt_shape={head.kpt_shape}")
    return model


def transfer_report(model) -> dict:
    torch, _, _, _ = load_ultralytics()
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    source_model = checkpoint.get("ema") or checkpoint.get("model")
    if hasattr(source_model, "float"):
        source_model = source_model.float()
    if source_model is None or not hasattr(source_model, "state_dict"):
        raise RuntimeError("Official checkpoint has no loadable model state")
    source_state = source_model.state_dict()
    target_state = model.model.state_dict()
    compatible = sorted(
        key for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    )
    transferred = [key for key in compatible if torch.equal(source_state[key].cpu(), target_state[key].cpu())]
    incompatible = sorted(
        key for key, value in source_state.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    )
    return {
        "source_checkpoint": str(WEIGHTS), "source_sha256": sha256_file(WEIGHTS),
        "source_state_tensors": len(source_state), "target_state_tensors": len(target_state),
        "compatible_tensors": len(compatible), "verified_transferred_tensors": len(transferred),
        "verified_transferred_key_sha256": hashlib.sha256(
            "\n".join(transferred).encode("utf-8")
        ).hexdigest(),
        "shape_incompatible_tensor_count": len(incompatible),
        "shape_incompatible_keys": incompatible,
        "custom_head": {"nc": int(model.model.model[-1].nc),
                        "kpt_shape": list(model.model.model[-1].kpt_shape)},
    }


def frame_image_id(rally: dict, frame: dict) -> str:
    return f"{rally['id']}_f{frame['clip_frame_num']:06d}"


def prepare_command(config: dict) -> int:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Cleaned set-1 footage folder not found: {SOURCE}")
    if (PREP / "summary.json").exists():
        raise FileExistsError(f"Preparation already exists; preserving it at {PREP}")
    PREP.mkdir(parents=True, exist_ok=True)
    exclusions = []
    candidates = enumerate_rallies(exclusions)
    geometries = {}
    screened = []
    for candidate_index, (match_dir, inventory, extras) in enumerate(candidates, 1):
        screened.append(inspect_rally(
            match_dir, inventory, extras["cleanup_rows"], geometries, exclusions
        ))
        if candidate_index % 100 == 0 or candidate_index == len(candidates):
            print(f"Screened {candidate_index}/{len(candidates)} rallies", flush=True)
    pre_box = [row for row in screened if row["status"] == "pre_box_pass"]
    pre_box_frame_count = sum(len(row["frames"]) for row in pre_box)
    print(f"Annotation screen passed {len(pre_box)} rallies / {pre_box_frame_count} frames; loading box model", flush=True)
    torch, _, YOLO, _ = load_ultralytics()
    ensure_weights()
    proposal_model = YOLO(str(WEIGHTS), task="pose")
    prep_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Box model ready on {prep_device}; checking {len(pre_box)} rallies", flush=True)
    manifest, outcomes = [], []
    for rally in screened:
        if rally["status"] != "pre_box_pass":
            outcomes.append({
                "key": rally["key"], "id": rally["id"], "status": "excluded",
                "reasons": rally["reasons"], "clip_path": rally["clip_path"],
            })

    def flush_box_group(group: list[dict]) -> None:
        if not group:
            return
        images = [image for state in group for image in state["decoded"]]
        entries = [(state, frame_index) for state in group
                   for frame_index in range(len(state["decoded"]))]
        results = proposal_model.predict(
            source=images, conf=float(config["preparation"]["confidence"]),
            iou=float(config["inference"]["iou"]), imgsz=PREP_IMGSZ, device=prep_device,
            classes=[0], batch=BOX_BATCH, verbose=False, stream=False,
        )
        if len(results) != len(entries):
            raise RuntimeError(f"Box model returned {len(results)} results for {len(entries)} frames")
        for result, (state, frame_index) in zip(results, entries, strict=True):
            rally = state["rally"]
            item = rally["frames"][frame_index]
            points = {
                player: native_point(item["references"][player], rally["width"], rally["height"])
                for player in ("A", "B")
            }
            found, reasons = box_associations(result, points, float(config["preparation"]["confidence"]))
            state["associations"].append(found)
            state["frame_reasons"].extend((item["clip_frame_num"], reason) for reason in reasons)

        for state in group:
            rally, decoded = state["rally"], state["decoded"]
            if state["frame_reasons"]:
                rally_reasons = sorted({reason for _, reason in state["frame_reasons"]})
                for frame_number, reason in state["frame_reasons"]:
                    exclusions.append({
                        **dict(zip(KEY_FIELDS, rally["key"])), "clip_frame_num": frame_number,
                        "reason": reason, "detail": "pretrained person-box association failed",
                    })
                outcomes.append({
                    "key": rally["key"], "id": rally["id"], "status": "excluded",
                    "reasons": rally_reasons, "clip_path": rally["clip_path"],
                })
                continue

            image_dir, label_dir = PREP / "images", PREP / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            rally_manifest, write_failed = [], ""
            for item, image, found in zip(rally["frames"], decoded, state["associations"], strict=True):
                image_id = frame_image_id(rally, item)
                image_path, label_path = image_dir / f"{image_id}.jpg", label_dir / f"{image_id}.txt"
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    write_failed = f"could not write {image_path}"
                    break
                label_lines = []
                for player in ("A", "B"):
                    x1, y1, x2, y2 = found[player]
                    point_x, point_y = native_point(item["references"][player], rally["width"], rally["height"])
                    values = ((x1 + x2) / (2 * rally["width"]), (y1 + y2) / (2 * rally["height"]),
                              (x2 - x1) / rally["width"], (y2 - y1) / rally["height"],
                              point_x / rally["width"], point_y / rally["height"], 2)
                    label_lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                rally_manifest.append({
                    **dict(zip(KEY_FIELDS, rally["key"])),
                    "rally_id": rally["id"], "clip_path": rally["clip_path"],
                    "source_csv": item["row"].get("source_csv", ""),
                    "clip_frame_num": item["clip_frame_num"], "clip_time_seconds": item["clip_time_seconds"],
                    "source_time_seconds": item["source_time_seconds"], "frame_num": item["frame_num"],
                    "fps": rally["fps"], "width": rally["width"], "height": rally["height"],
                    "image_path": str(image_path.resolve()), "label_path": str(label_path.resolve()),
                    "A_ref_x": item["references"]["A"][0], "A_ref_y": item["references"]["A"][1],
                    "B_ref_x": item["references"]["B"][0], "B_ref_y": item["references"]["B"][1],
                    "A_native_x": native_point(item["references"]["A"], rally["width"], rally["height"])[0],
                    "A_native_y": native_point(item["references"]["A"], rally["width"], rally["height"])[1],
                    "B_native_x": native_point(item["references"]["B"], rally["width"], rally["height"])[0],
                    "B_native_y": native_point(item["references"]["B"], rally["width"], rally["height"])[1],
                    "A_half": rally["A_half"], "B_half": rally["B_half"],
                    "A_box": list(found["A"]), "B_box": list(found["B"]),
                    "hitter_player": str(item["row"].get("player", "")).strip().upper(),
                    "automatic_check_status": "passed",
                })
            if write_failed:
                for row in rally_manifest:
                    Path(row["image_path"]).unlink(missing_ok=True)
                    Path(row["label_path"]).unlink(missing_ok=True)
                exclusions.append({
                    **dict(zip(KEY_FIELDS, rally["key"])), "clip_frame_num": "",
                    "reason": "prepared_image_write_failed", "detail": write_failed,
                })
                outcomes.append({"key": rally["key"], "id": rally["id"], "status": "excluded",
                                 "reasons": ["prepared_image_write_failed"], "clip_path": rally["clip_path"]})
                continue
            manifest.extend(rally_manifest)
            outcomes.append({
                "key": rally["key"], "id": rally["id"], "status": "automatic_pass",
                "reasons": [], "clip_path": rally["clip_path"], "A_half": rally["A_half"],
                "B_half": rally["B_half"], "frame_count": len(rally_manifest),
            })

    group, group_frames = [], 0
    for candidate_index, rally in enumerate(pre_box, 1):
        if candidate_index == 1 or candidate_index % 100 == 0:
            print(f"Decoding/inferencing {candidate_index}/{len(pre_box)} eligible rallies; images written={len(manifest)}",
                  flush=True)
        cap = cv2.VideoCapture(rally["clip_path"])
        decoded, decode_error = [], ""
        for item in rally["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, item["clip_frame_num"])
            ok, image = cap.read()
            if not ok or image is None or image.shape[:2] != (rally["height"], rally["width"]):
                decode_error = f"failed to decode exact clip frame {item['clip_frame_num']}"
                break
            decoded.append(image)
        cap.release()
        if decode_error:
            exclusions.append({
                **dict(zip(KEY_FIELDS, rally["key"])), "clip_frame_num": "",
                "reason": "failed_frame_decode", "detail": decode_error,
            })
            outcomes.append({"key": rally["key"], "id": rally["id"], "status": "excluded",
                             "reasons": ["failed_frame_decode"], "clip_path": rally["clip_path"]})
            continue
        if group and group_frames + len(decoded) > 128:
            flush_box_group(group)
            group, group_frames = [], 0
        group.append({"rally": rally, "decoded": decoded, "associations": [], "frame_reasons": []})
        group_frames += len(decoded)
        if group_frames >= 128:
            flush_box_group(group)
            group, group_frames = [], 0
    flush_box_group(group)

    passed_by_match = defaultdict(list)
    for row in outcomes:
        if row["status"] == "automatic_pass":
            passed_by_match[row["key"][1]].append(row)
    blocked_matches = set()
    for match, rallies in passed_by_match.items():
        if len(rallies) < 5:
            blocked_matches.add(match)
            exclusions.append({
                "dataset": "set-1", "match": match, "game": "", "original_rally_id": "",
                "clip_frame_num": "", "reason": "fewer_than_five_auto_pass_rallies_for_review",
                "detail": f"auto_pass_rallies={len(rallies)}",
            })
    if blocked_matches:
        for row in outcomes:
            if row["key"][1] in blocked_matches and row["status"] == "automatic_pass":
                row["status"] = "excluded"
                row["reasons"] = ["match_has_fewer_than_five_review_rallies"]
        for row in manifest:
            if row["match"] in blocked_matches:
                exclusions.append({
                    **{field: row[field] for field in KEY_FIELDS},
                    "clip_frame_num": row["clip_frame_num"],
                    "reason": "match_has_fewer_than_five_review_rallies",
                    "detail": "no five distinct automatic-pass rallies available for sampled review",
                })
                Path(row["image_path"]).unlink(missing_ok=True)
                Path(row["label_path"]).unlink(missing_ok=True)
        manifest = [row for row in manifest if row["match"] not in blocked_matches]

    manifest_path = PREP / "manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    write_jsonl(PREP / "rallies.jsonl", outcomes)
    exclusion_fields = [*KEY_FIELDS, "clip_frame_num", "reason", "detail"]
    write_csv(PREP / "exclusions.csv", exclusions, exclusion_fields)
    sample_rows = []
    selected_keys = set()
    samples_by_match = defaultdict(list)
    rally_manifest_groups = defaultdict(list)
    for row in manifest:
        rally_manifest_groups[tuple(row[field] for field in KEY_FIELDS)].append(row)
    for match in sorted({row["match"] for row in manifest}):
        match_rallies = []
        for key, rows in rally_manifest_groups.items():
            if key[1] == match:
                match_rallies.append((key, sorted(rows, key=lambda item: int(item["clip_frame_num"]))))
        match_rallies.sort(key=lambda pair: (int(pair[0][2]), int(pair[0][3])))
        choices = [int(round(i * (len(match_rallies) - 1) / 4)) for i in range(5)]
        for choice in choices:
            key, rows = match_rallies[choice]
            selected = rows[len(rows) // 2]
            sample = {
                **dict(zip(KEY_FIELDS, key)), "rally_id": selected["rally_id"],
                "clip_frame_num": selected["clip_frame_num"], "image_path": selected["image_path"],
                "A_box": selected["A_box"], "B_box": selected["B_box"],
                "A_ref_x": selected["A_native_x"], "A_ref_y": selected["A_native_y"],
                "B_ref_x": selected["B_native_x"], "B_ref_y": selected["B_native_y"],
            }
            sample_rows.append(sample)
            selected_keys.add((*key, int(selected["clip_frame_num"])))
            samples_by_match[match].append(sample)
    sample_fields = [*KEY_FIELDS, "rally_id", "clip_frame_num", "image_path",
                     "A_box", "B_box", "A_ref_x", "A_ref_y", "B_ref_x", "B_ref_y"]
    write_csv(PREP / "review_samples.csv", sample_rows, sample_fields)
    write_json(PREP / "review_state.json", {
        "created_at": utc_now(), "samples": {
            safe_id(tuple(sample[field] for field in (*KEY_FIELDS, "clip_frame_num"))):
                {"status": "pending", "sample": sample}
            for sample in sample_rows
        },
    })
    for row in manifest:
        row["review_sample_status"] = (
            "pending" if (*tuple(row[field] for field in KEY_FIELDS), int(row["clip_frame_num"])) in selected_keys
            else "not_sampled"
        )
    write_jsonl(manifest_path, manifest)
    auto_rallies = [row for row in outcomes if row["status"] == "automatic_pass"]
    summary = {
        "created_at": utc_now(), "source": str(SOURCE), "permanently_excluded_matches": sorted(EXCLUDED_MATCHES),
        "matches_in_source": sum(path.is_dir() for path in SOURCE.iterdir()),
        "matches_after_permanent_exclusions": sum(
            path.is_dir() and path.name not in EXCLUDED_MATCHES for path in SOURCE.iterdir()
        ),
        "matches_with_rally_inventories": len({item[1]["match"] for item in candidates}),
        "input_rallies_after_permanent_exclusions": len(candidates),
        "pre_box_rallies": len(pre_box), "pre_box_annotated_frames": pre_box_frame_count,
        "automatic_rallies_after_box_checks": len(auto_rallies),
        "automatic_annotated_frames_after_box_checks": len(manifest),
        "matches_for_review": len(samples_by_match), "review_sample_count": len(sample_rows),
        "automatic_box_confidence": config["preparation"]["confidence"],
        "pretrained_weights_sha256": sha256_file(WEIGHTS),
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(PREP / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Excluded reasons: {len(exclusions)} rows; see {PREP / 'exclusions.csv'}")
    return 0


def review_command() -> int:
    state_path = PREP / "review_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("Run prepare before review")
    if SPLIT_FILE.exists():
        raise RuntimeError("The accepted dataset is already frozen; review state cannot be changed")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    samples = state["samples"]
    window = "YOLO player-keypoint sample review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for sample_id, entry in samples.items():
            if entry["status"] != "pending":
                continue
            sample = entry["sample"]
            if any(
                other["status"] == "rejected_match"
                and other["sample"]["match"] == sample["match"]
                for other in samples.values()
            ):
                entry["status"] = "rejected_match"
                write_json(state_path, state)
                continue
            image = cv2.imread(sample["image_path"])
            if image is None:
                raise RuntimeError(f"Review image is missing or unreadable: {sample['image_path']}")
            for player, color, box_key, x_key, y_key in (
                ("A", (255, 80, 30), "A_box", "A_ref_x", "A_ref_y"),
                ("B", (30, 100, 255), "B_box", "B_ref_x", "B_ref_y"),
            ):
                x1, y1, x2, y2 = [int(round(value)) for value in sample[box_key]]
                point = (int(round(sample[x_key])), int(round(sample[y_key])))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.circle(image, point, 8, color, -1)
                cv2.putText(image, player, (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                cv2.putText(image, f"{player} reference", (point[0] + 8, point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            header = (
                f"{sample['match']} | game {sample['game']} | rally {sample['original_rally_id']} | "
                f"frame {sample['clip_frame_num']}"
            )
            cv2.putText(image, header[:150], (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
                entry["status"] = "rejected_match"
                for other in samples.values():
                    if other["sample"]["match"] == sample["match"]:
                        other["status"] = "rejected_match"
            elif key in (ord("s"), ord("S"), ord("q"), ord("Q"), 27):
                break
            else:
                continue
            entry["updated_at"] = utc_now()
            write_json(state_path, state)
    finally:
        cv2.destroyWindow(window)
    state["updated_at"] = utc_now()
    state["complete"] = all(row["status"] != "pending" for row in samples.values())
    write_json(state_path, state)
    counts = defaultdict(int)
    for row in samples.values():
        counts[row["status"]] += 1
    print(json.dumps({"review_state": str(state_path), "complete": state["complete"],
                      "sample_status_counts": dict(counts)}, indent=2))
    return 0


def review_exclusions(state: dict) -> list[dict]:
    rows = []
    seen_matches = set()
    for entry in state["samples"].values():
        sample, status = entry["sample"], entry["status"]
        if status == "rejected_rally":
            rows.append({
                **{field: sample[field] for field in KEY_FIELDS},
                "clip_frame_num": sample["clip_frame_num"],
                "reason": "manual_sample_rejected_rally",
                "detail": "sample review rejected the entire rally",
            })
        elif status == "rejected_match" and sample["match"] not in seen_matches:
            seen_matches.add(sample["match"])
            rows.append({
                "dataset": sample["dataset"], "match": sample["match"], "game": "",
                "original_rally_id": "", "clip_frame_num": "",
                "reason": "manual_sample_rejected_match",
                "detail": "sample review rejected the entire match",
            })
    return rows


def accepted_manifest() -> tuple[list[dict], dict, list[dict]]:
    state_path = PREP / "review_state.json"
    manifest_path = PREP / "manifest.jsonl"
    if not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("Run prepare and complete sampled review before split")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not all(row["status"] != "pending" for row in state["samples"].values()):
        pending = sum(row["status"] == "pending" for row in state["samples"].values())
        raise RuntimeError(f"Sample review has {pending} pending frames; use review to finish or resume")
    rejected_matches = {
        row["sample"]["match"] for row in state["samples"].values()
        if row["status"] == "rejected_match"
    }
    rejected_rallies = {
        stable_key(row["sample"]) for row in state["samples"].values()
        if row["status"] == "rejected_rally"
    }
    accepted = [
        row for row in read_jsonl(manifest_path)
        if row["match"] not in rejected_matches
        and stable_key(row) not in rejected_rallies
    ]
    if not accepted:
        raise ValueError("Sample review left no accepted annotated frames")
    return accepted, state, review_exclusions(state)


def grouped_folds(matches: list[str], folds: int, seed: int) -> dict[str, int]:
    if folds < 2 or folds > len(matches):
        raise ValueError(f"Requested {folds} folds; require 2 <= folds <= {len(matches)} development matches")
    shuffled = sorted(matches)
    random.Random(seed).shuffle(shuffled)
    return {match: index % folds for index, match in enumerate(shuffled)}


def dataset_identity(rows: list[dict]) -> str:
    values = [
        [row[field] for field in (*KEY_FIELDS, "clip_frame_num", "image_path", "label_path")]
        for row in sorted(rows, key=lambda item: (*stable_key(item), int(item["clip_frame_num"])))
    ]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def fixed_test_matches(matches: list[str]) -> list[str]:
    if len(matches) < 3:
        raise ValueError("At least three accepted matches are required for a 20% test and grouped CV")
    shuffled = sorted(matches)
    random.Random(42).shuffle(shuffled)
    test_count = max(1, int(math.floor(len(shuffled) * 0.20 + 0.5)))
    if test_count >= len(shuffled):
        raise ValueError("The fixed 20% test reservation leaves no development match")
    return sorted(shuffled[:test_count])


def split_command(config: dict) -> int:
    rows, review_state, manual_exclusions = accepted_manifest()
    matches = sorted({row["match"] for row in rows})
    test_matches = fixed_test_matches(matches)
    development = sorted(set(matches) - set(test_matches))
    folds = int(config["cv"]["folds"])
    fold_seed = int(config["cv"]["seed"])
    assignments = grouped_folds(development, folds, fold_seed)
    identity = dataset_identity(rows)
    if SPLIT_FILE.exists():
        saved = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        if saved["dataset_identity"] != identity:
            raise RuntimeError("A different accepted dataset is already frozen; refusing to replace its split")
        test_matches = saved["held_out_matches"]
        development = saved["development_matches"]
        assignments = grouped_folds(development, folds, fold_seed)
    else:
        saved = {
            "created_at": utc_now(), "preparation_summary_sha256": sha256_file(PREP / "summary.json"),
            "dataset_identity": identity, "review_state_sha256": sha256_file(PREP / "review_state.json"),
            "held_out_seed": 42, "held_out_matches": test_matches,
            "development_matches": development, "cv_runs": {},
        }
    cv_key = f"folds-{folds}-seed-{fold_seed}"
    saved.setdefault("cv_runs", {})[cv_key] = {
        "folds": folds, "seed": fold_seed, "match_to_fold": assignments,
    }
    saved["updated_at"] = utc_now()
    write_json(SPLIT_FILE, saved)
    frozen_path = PREP / "frozen_manifest.jsonl"
    if not frozen_path.exists():
        write_jsonl(frozen_path, rows)
    write_csv(PREP / "review_exclusions.csv", manual_exclusions,
              [*KEY_FIELDS, "clip_frame_num", "reason", "detail"])
    print(json.dumps({
        "split_file": str(SPLIT_FILE), "accepted_matches": len(matches),
        "held_out_matches": len(test_matches), "development_matches": len(development),
        "held_out_match_names": test_matches, "cv_assignment_key": cv_key,
        "fold_sizes": {str(fold + 1): sum(value == fold for value in assignments.values())
                       for fold in range(folds)},
        "dataset_identity": identity,
    }, indent=2))
    return 0


def run_dependencies() -> dict:
    torch, ultralytics, _, _ = load_ultralytics()
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception as error:
        raise RuntimeError(f"CUDA-compatible torchvision import failed: {error}") from error
    return {
        "python": sys.version.split()[0], "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "torchvision": torchvision_version,
        "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda,
        "opencv": cv2.__version__, "numpy": np.__version__, "pyyaml": yaml.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def train_with_local_amp_check(model, args: dict) -> dict:
    torch, _, _, _ = load_ultralytics()
    import ultralytics.engine.trainer as trainer_module
    from ultralytics.utils.torch_utils import autocast

    state = {"method": "local_custom_model", "passed": None}
    original_check = trainer_module.check_amp

    def local_check(current_model):
        was_training = current_model.training
        try:
            parameter = next(current_model.parameters())
            device = parameter.device
            if device.type != "cuda":
                state["passed"] = False
                return False
            image_size = max(256, int(current_model.stride.max() * 4))
            image = torch.rand((1, 3, image_size, image_size), device=device)
            current_model.eval()
            with torch.no_grad():
                fp32 = current_model(image)
                with autocast(enabled=True, device=device.type):
                    mixed = current_model(image)

            def leaves(value):
                if torch.is_tensor(value):
                    return [value]
                if isinstance(value, dict):
                    return [tensor for child in value.values() for tensor in leaves(child)]
                if isinstance(value, (list, tuple)):
                    return [tensor for child in value for tensor in leaves(child)]
                return []

            full_tensors, mixed_tensors = leaves(fp32), leaves(mixed)
            if len(full_tensors) != len(mixed_tensors):
                state["passed"] = False
            else:
                diffs = [
                    float((a.float() - b.float()).abs().max().item())
                    for a, b in zip(full_tensors, mixed_tensors, strict=True)
                    if a.numel()
                ]
                state["max_abs_error"] = max(diffs, default=0.0)
                state["passed"] = all(
                    a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=0.5)
                    for a, b in zip(full_tensors, mixed_tensors, strict=True)
                )
            return state["passed"]
        except Exception as error:
            state["error"] = str(error)
            state["passed"] = False
            return False
        finally:
            current_model.train(was_training)

    # The pinned Ultralytics AMP probe loads an unrelated yolo26n checkpoint; probe this model instead.
    trainer_module.check_amp = local_check
    try:
        model.train(**args)
    finally:
        trainer_module.check_amp = original_check
    state["effective_amp"] = bool(model.trainer.amp)
    return state


def build_training_args(train: dict, data_yaml: Path, project: Path, name: str) -> dict:
    return {
        **train, "data": str(data_yaml.resolve()), "project": str(project.resolve()),
        "name": name, "task": "pose", "exist_ok": False, "save": True, "pretrained": False,
    }


def model_contract(model) -> dict:
    head = model.model.model[-1]
    result = {"classes": int(head.nc), "keypoints": int(head.kpt_shape[0]),
              "keypoint_dimensions": int(head.kpt_shape[1])}
    if result != {"classes": 1, "keypoints": 1, "keypoint_dimensions": 3}:
        raise RuntimeError(f"Expected one class and one three-dimensional keypoint; found {result}")
    return result


def assert_prediction_contract(model, device) -> dict:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[120:610, 210:425] = (210, 210, 210)
    results = model.predict(source=image, imgsz=640, conf=0.000001, iou=0.7,
                            device=device, max_det=30, verbose=False)
    result = results[0]
    points = result.keypoints
    if points is None or points.data.ndim != 3 or points.data.shape[1] != 1:
        shape = None if points is None else tuple(points.data.shape)
        raise RuntimeError(f"Prediction output must contain one keypoint per detection; shape={shape}")
    return {"detections": len(result.boxes), "keypoints_per_detection": int(points.data.shape[1]),
            "output_shape": list(points.data.shape)}


def dependency_snapshot() -> dict:
    versions = run_dependencies()
    versions["pose_checkpoint_sha256"] = sha256_file(WEIGHTS)
    versions["model_yaml_sha256"] = sha256_file(ensure_model_yaml())
    return versions


def write_pose_data_yaml(directory: Path, train_rows: list[dict], val_rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    train_txt, val_txt = directory / "train.txt", directory / "val.txt"
    train_txt.write_text("\n".join(row["image_path"] for row in train_rows) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(row["image_path"] for row in val_rows) + "\n", encoding="utf-8")
    data_yaml = directory / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(PREP.resolve()), "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()), "names": {0: "player"},
        "kpt_shape": [1, 3], "flip_idx": [0],
    }, sort_keys=False), encoding="utf-8")
    return data_yaml


def create_smoke_data() -> Path:
    root = ROOT / "smoke" / "dataset"
    for split in ("train", "val"):
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[20:112, 20:54] = (40, 150, 240)
        image[20:112, 74:108] = (240, 130, 30)
        image_path = image_dir / "sample.jpg"
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not create smoke image {image_path}")
        (label_dir / "sample.txt").write_text(
            "0 0.28906250 0.51562500 0.26562500 0.71875000 0.28906250 0.82031250 2\n"
            "0 0.71093750 0.51562500 0.26562500 0.71875000 0.71093750 0.82031250 2\n",
            encoding="utf-8",
        )
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train", "val": "images/val",
        "names": {0: "player"}, "kpt_shape": [1, 3], "flip_idx": [0],
    }, sort_keys=False), encoding="utf-8")
    return data


def check_command(config: dict) -> int:
    summary_path = ROOT / "smoke" / "smoke_summary.json"
    if summary_path.is_file():
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(old_summary.get("amp_check"), dict) \
                and old_summary["amp_check"].get("method") == "local_custom_model":
            print(json.dumps(old_summary, indent=2))
            return 0
    torch, _, _, _ = load_ultralytics()
    weight_path = ensure_weights()
    model = init_custom_model(seed=0)
    transfer = transfer_report(model)
    contract = model_contract(model)
    device = 0 if torch.cuda.is_available() else "cpu"
    prediction = assert_prediction_contract(model, device)

    data_yaml = create_smoke_data()
    smoke_dir = ROOT / "smoke" / "train"
    args = dict(config["train"])
    args.update({"epochs": 1, "patience": 1, "batch": 2, "imgsz": 64,
                 "workers": 0, "mosaic": 0.0, "close_mosaic": 0,
                 "device": device, "project": str(smoke_dir.resolve()),
                 "name": "tiny_local_amp", "data": str(data_yaml.resolve()),
                 "task": "pose", "exist_ok": False, "save": True, "pretrained": False})
    amp_validation = train_with_local_amp_check(model, args)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Tiny smoke training did not produce best.pt")
    _, _, YOLO, _ = load_ultralytics()
    trained = YOLO(str(best), task="pose")
    trained_contract = model_contract(trained)
    trained_prediction = assert_prediction_contract(trained, device)
    summary = {
        "created_at": utc_now(), "status": "passed", "dependencies": dependency_snapshot(),
        "checkpoint": str(weight_path), "checkpoint_sha256": sha256_file(weight_path),
        "weight_transfer": transfer, "custom_model_contract": contract,
        "initialized_prediction": prediction, "smoke_checkpoint": str(best.resolve()),
        "smoke_checkpoint_sha256": sha256_file(best), "trained_model_contract": trained_contract,
        "trained_prediction": trained_prediction,
        "amp_check": amp_validation,
        "synthetic_training": {"epochs": 1, "images_each_split": 1, "instances_per_image": 2},
    }
    write_json(ROOT / "smoke" / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def cv_run_identity(config: dict, split_data: dict) -> tuple[str, str]:
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    cv_key = f"folds-{config['cv']['folds']}-seed-{config['cv']['seed']}"
    run_input = {"settings": resolved, "dataset_identity": split_data["dataset_identity"],
                 "fold_assignment": split_data["cv_runs"][cv_key]["match_to_fold"],
                 "checkpoint_sha256": sha256_file(WEIGHTS)}
    digest = hashlib.sha256(json.dumps(run_input, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"cv_{config['cv']['folds']}f_s{config['cv']['seed']}_{digest}", cv_key


def ensure_cv_assignment(config: dict, split_data: dict) -> str:
    folds, seed = int(config["cv"]["folds"]), int(config["cv"]["seed"])
    cv_key = f"folds-{folds}-seed-{seed}"
    cv_runs = split_data.setdefault("cv_runs", {})
    if cv_key not in cv_runs:
        assignment = grouped_folds(split_data["development_matches"], folds, seed)
        cv_runs[cv_key] = {"folds": folds, "seed": seed, "match_to_fold": assignment}
        split_data["updated_at"] = utc_now()
        write_json(SPLIT_FILE, split_data)
    return cv_key


def cv_split_rows(rows: list[dict], split_data: dict, assignments: dict, fold_index: int):
    validation_matches = {match for match, fold in assignments.items() if fold == fold_index}
    train_rows = [row for row in rows if row["match"] in split_data["development_matches"]
                  and row["match"] not in validation_matches]
    val_rows = [row for row in rows if row["match"] in validation_matches]
    if not train_rows or not val_rows:
        raise ValueError(f"Fold {fold_index + 1} has an empty train or validation split")
    if {row["match"] for row in train_rows} & {row["match"] for row in val_rows}:
        raise AssertionError("A match leaked between grouped train and validation")
    if set(row["match"] for row in train_rows) & set(split_data["held_out_matches"]):
        raise AssertionError("A held-out test match leaked into CV training")
    return train_rows, val_rows


def selected_epoch(checkpoint: Path) -> int:
    torch, _, _, _ = load_ultralytics()
    value = torch.load(checkpoint, map_location="cpu", weights_only=False).get("epoch", -1)
    if isinstance(value, torch.Tensor):
        value = int(value.item())
    return max(1, int(value) + 1)


def train_cv_command(config: dict) -> int:
    if not SPLIT_FILE.is_file() or not (PREP / "frozen_manifest.jsonl").is_file():
        raise FileNotFoundError("Complete review and run split before train-cv")
    split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    ensure_cv_assignment(config, split_data)
    cv_id, cv_key = cv_run_identity(config, split_data)
    assignments = split_data["cv_runs"][cv_key]["match_to_fold"]
    folds = int(config["cv"]["folds"])
    if set(assignments.values()) != set(range(folds)):
        raise RuntimeError("Saved grouped fold assignments do not cover the requested folds")
    rows = read_jsonl(PREP / "frozen_manifest.jsonl")
    if dataset_identity(rows) != split_data["dataset_identity"]:
        raise RuntimeError("Frozen manifest no longer matches the saved split identity")
    run_dir = ROOT / "runs" / cv_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    settings_path = run_dir / "resolved_settings.yaml"
    if settings_path.exists() and yaml.safe_load(settings_path.read_text(encoding="utf-8")) != resolved:
        raise RuntimeError(f"Run folder settings differ from the loaded config: {run_dir}")
    settings_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "fold_assignments.json", {
        "cv_key": cv_key, "folds": folds, "seed": config["cv"]["seed"],
        "match_to_fold": assignments, "held_out_matches": split_data["held_out_matches"],
        "development_matches": split_data["development_matches"],
    })
    write_json(run_dir / "dependencies.json", dependency_snapshot())
    write_json(run_dir / "checkpoint_hashes.json", {
        "initial_yolo11n_pose_sha256": sha256_file(WEIGHTS),
    })
    fold_summaries = {}
    for fold_index in range(folds):
        fold_dir = run_dir / f"fold_{fold_index + 1:02d}"
        metadata_path = fold_dir / "fold.json"
        if metadata_path.is_file():
            old = json.loads(metadata_path.read_text(encoding="utf-8"))
            if old.get("status") == "completed":
                fold_summaries[str(fold_index + 1)] = old
                continue
            raise RuntimeError(f"Fold output exists but is incomplete; preserving it: {fold_dir}")
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise RuntimeError(f"Fold output exists and will not be overwritten: {fold_dir}")
        train_rows, val_rows = cv_split_rows(rows, split_data, assignments, fold_index)
        data_yaml = write_pose_data_yaml(fold_dir / "dataset", train_rows, val_rows)
        init_seed = int(config["train"].get("seed", 42)) + fold_index + 1
        model = init_custom_model(seed=init_seed)
        report = transfer_report(model)
        write_json(fold_dir / "weight_transfer.json", report)
        write_json(metadata_path, {
            "status": "running", "fold": fold_index + 1, "started_at": utc_now(),
            "init_seed": init_seed, "train_matches": sorted({row["match"] for row in train_rows}),
            "validation_matches": sorted({row["match"] for row in val_rows}),
            "train_frames": len(train_rows), "validation_frames": len(val_rows),
            "transfer_sha256": report["source_sha256"],
        })
        args = build_training_args(config["train"], data_yaml, fold_dir, "train")
        amp_validation = train_with_local_amp_check(model, args)
        save_dir = Path(model.trainer.save_dir)
        best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(f"Fold {fold_index + 1} did not produce both best.pt and last.pt")
        epoch_count = selected_epoch(best)
        completed = {
            "status": "completed", "fold": fold_index + 1, "completed_at": utc_now(),
            "init_seed": init_seed, "selected_epoch_count": epoch_count,
            "train_matches": sorted({row["match"] for row in train_rows}),
            "validation_matches": sorted({row["match"] for row in val_rows}),
            "train_frames": len(train_rows), "validation_frames": len(val_rows),
            "data_yaml": str(data_yaml.resolve()), "best_checkpoint": str(best.resolve()),
            "last_checkpoint": str(last.resolve()), "best_sha256": sha256_file(best),
            "last_sha256": sha256_file(last), "amp_check": amp_validation,
        }
        write_json(metadata_path, completed)
        write_json(run_dir / "checkpoint_hashes.json", {
            **json.loads((run_dir / "checkpoint_hashes.json").read_text(encoding="utf-8")),
            f"fold_{fold_index + 1:02d}_best_sha256": completed["best_sha256"],
            f"fold_{fold_index + 1:02d}_last_sha256": completed["last_sha256"],
        })
        fold_summaries[str(fold_index + 1)] = completed

    if len(fold_summaries) != folds or any(row["status"] != "completed" for row in fold_summaries.values()):
        raise RuntimeError(f"Expected exactly {folds} completed folds; found {len(fold_summaries)}")
    oof_dir = run_dir / "oof"
    positions, errors = [], []
    for fold_index in range(folds):
        fold_num = str(fold_index + 1)
        fold_meta = fold_summaries[fold_num]
        validation_matches = set(fold_meta["validation_matches"])
        fold_rows = [row for row in rows if row["match"] in validation_matches]
        fold_positions, fold_errors = predict_and_score(
            fold_meta["best_checkpoint"], fold_rows, config, oof_dir / f"fold_{fold_num}"
        )
        for row in fold_positions:
            row["fold"] = fold_index + 1
        for row in fold_errors:
            row["fold"] = fold_index + 1
        positions.extend(fold_positions)
        errors.extend(fold_errors)
    if len({row["fold"] for row in errors}) != folds:
        raise RuntimeError("OOF aggregation did not include exactly N completed folds")
    write_evaluation(oof_dir, positions, errors, folds=folds)
    write_json(run_dir / "cv_summary.json", {
        "status": "completed", "run_id": cv_id, "completed_at": utc_now(),
        "folds": folds, "fold_summaries": fold_summaries,
        "selected_epoch_counts": [fold_summaries[str(i + 1)]["selected_epoch_count"]
                                  for i in range(folds)],
        "oof_metrics": str((oof_dir / "metrics.json").resolve()),
    })
    print(json.dumps({"run_id": cv_id, "run_dir": str(run_dir),
                      "completed_folds": len(fold_summaries),
                      "selected_epoch_counts": [fold_summaries[str(i + 1)]["selected_epoch_count"]
                                                for i in range(folds)],
                      "oof_metrics": str(oof_dir / "metrics.json")}, indent=2))
    return 0


def assign_predictions(result, row: dict, geometry, config: dict) -> tuple[dict, dict]:
    choices = {"far": [], "near": []}
    boxes_obj, points_obj = result.boxes, result.keypoints
    if boxes_obj is None or points_obj is None:
        return {}, {"far": "missing_detection", "near": "missing_detection"}
    boxes = boxes_obj.xyxy.detach().cpu().numpy()
    scores = boxes_obj.conf.detach().cpu().numpy()
    classes = boxes_obj.cls.detach().cpu().numpy().astype(int)
    points = points_obj.data.detach().cpu().numpy()
    conf = float(config["inference"]["confidence"])
    keypoint_conf = float(config["inference"]["keypoint_confidence"])
    height, width = int(row["height"]), int(row["width"])
    for index, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        if cls != 0 or score < conf or index >= len(points) or points.shape[1] != 1:
            continue
        point_x, point_y, point_conf = [float(value) for value in points[index, 0, :3]]
        if (not np.isfinite([point_x, point_y, point_conf]).all()
                or point_conf < keypoint_conf or not 0 <= point_x < width
                or not 0 <= point_y < height):
            continue
        side = court_half(geometry, (point_x, point_y), width, height)
        choices[side].append({
            "point": (point_x, point_y), "box": tuple(float(value) for value in box),
            "box_confidence": float(score), "keypoint_confidence": point_conf,
        })
    selected, status = {}, {}
    for side in ("far", "near"):
        if len(choices[side]) == 1:
            selected[side] = choices[side][0]
            status[side] = "available"
        elif len(choices[side]) > 1:
            status[side] = "ambiguous_multiple_detections"
        else:
            status[side] = "missing_detection"
    return selected, status


def predict_and_score(weights: str | Path, rows: list[dict], config: dict,
                      output_dir: Path) -> tuple[list[dict], list[dict]]:
    torch, _, YOLO, _ = load_ultralytics()
    model = YOLO(str(weights), task="pose")
    model_contract(model)
    train_cfg = config["train"]
    device = train_cfg.get("device", 0) if torch.cuda.is_available() else "cpu"
    batch_size = train_cfg["batch"] if train_cfg["batch"] > 0 else 8
    positions, errors = [], []
    geometries = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        results = model.predict(
            source=[row["image_path"] for row in batch], imgsz=train_cfg["imgsz"],
            conf=float(config["inference"]["confidence"]), iou=float(config["inference"]["iou"]),
            device=device, batch=batch_size, classes=[0], verbose=False, stream=False,
        )
        if len(results) != len(batch):
            raise RuntimeError(f"Model returned {len(results)} results for {len(batch)} frames")
        for row, result in zip(batch, results, strict=True):
            if row["match"] not in geometries:
                geometries[row["match"]] = import_geometry(row["match"])
            geometry = geometries[row["match"]]
            selected, side_status = assign_predictions(result, row, geometry, config)
            for player in ("A", "B"):
                wanted_half = row[f"{player}_half"]
                candidate = selected.get(wanted_half)
                status = side_status[wanted_half]
                ref_x = float(row[f"{player}_native_x"])
                ref_y = float(row[f"{player}_native_y"])
                error_px = error_m = None
                pred_x = pred_y = box_conf = point_conf = None
                box = None
                if candidate is not None:
                    pred_x, pred_y = candidate["point"]
                    box = candidate["box"]
                    box_conf = candidate["box_confidence"]
                    point_conf = candidate["keypoint_confidence"]
                    error_px = float(math.hypot(pred_x - ref_x, pred_y - ref_y))
                    ref_world = geometry.annotation_floor_to_world(
                        (ref_x * REFERENCE_WIDTH / int(row["width"]),
                         ref_y * REFERENCE_HEIGHT / int(row["height"]))
                    )
                    pred_world = geometry.annotation_floor_to_world(
                        (pred_x * REFERENCE_WIDTH / int(row["width"]),
                         pred_y * REFERENCE_HEIGHT / int(row["height"]))
                    )
                    if not np.isfinite(ref_world).all() or not np.isfinite(pred_world).all():
                        candidate = None
                        status = "invalid_court_projection"
                        error_px = None
                        pred_x = pred_y = box_conf = point_conf = None
                    else:
                        error_m = float(np.linalg.norm(pred_world - ref_world))
                positions.append({
                    **{field: row[field] for field in KEY_FIELDS},
                    "clip_frame_num": row["clip_frame_num"], "player": player,
                    "prediction_status": status, "available": candidate is not None,
                    "x_native": pred_x, "y_native": pred_y,
                    "box_x1": None if box is None or candidate is None else box[0],
                    "box_y1": None if box is None or candidate is None else box[1],
                    "box_x2": None if box is None or candidate is None else box[2],
                    "box_y2": None if box is None or candidate is None else box[3],
                    "box_confidence": box_conf, "keypoint_confidence": point_conf,
                    "court_half": wanted_half, "reference_x_native": ref_x,
                    "reference_y_native": ref_y,
                })
                errors.append({
                    **{field: row[field] for field in KEY_FIELDS},
                    "clip_frame_num": row["clip_frame_num"], "player": player,
                    "prediction_status": status, "available": candidate is not None,
                    "error_px": error_px, "error_m": error_m,
                })
    return positions, errors


def error_summary(rows: list[dict]) -> dict:
    expected = len(rows)
    available = [row for row in rows if row["available"]]
    pixel = [float(row["error_px"]) for row in available if row["error_px"] is not None]
    metres = [float(row["error_m"]) for row in available if row["error_m"] is not None]

    def stats(values: list[float]) -> dict:
        return {
            "count": len(values),
            "mean": None if not values else float(np.mean(values)),
            "median": None if not values else float(np.median(values)),
            "p95": None if not values else float(np.percentile(values, 95)),
        }

    def success(threshold: float) -> dict:
        count = sum(row["available"] and row["error_m"] is not None and row["error_m"] <= threshold
                    for row in rows)
        return {"count": int(count), "denominator": expected,
                "rate": None if not expected else count / expected}

    return {
        "expected_predictions": expected, "available_predictions": len(available),
        "missing_predictions": expected - len(available),
        "coverage": None if not expected else len(available) / expected,
        "pixel_error": stats(pixel), "metre_error": stats(metres),
        "within_0_5_m": success(0.5), "within_1_m": success(1.0),
    }


def write_evaluation(output_dir: Path, positions: list[dict], errors: list[dict],
                     folds: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    position_fields = [
        *KEY_FIELDS, "clip_frame_num", "player", "prediction_status", "available",
        "x_native", "y_native", "box_x1", "box_y1", "box_x2", "box_y2",
        "box_confidence", "keypoint_confidence", "court_half",
        "reference_x_native", "reference_y_native", "fold",
    ]
    error_fields = [*KEY_FIELDS, "clip_frame_num", "player", "prediction_status",
                    "available", "error_px", "error_m", "fold"]
    write_csv(output_dir / "player_positions.csv", positions, position_fields)
    write_csv(output_dir / "evaluation_results.csv", errors, error_fields)
    metrics = {
        "combined": error_summary(errors),
        "per_player": {
            player: error_summary([row for row in errors if row["player"] == player])
            for player in ("A", "B")
        },
        "per_match": {
            match: error_summary([row for row in errors if row["match"] == match])
            for match in sorted({row["match"] for row in errors})
        },
    }
    if folds is not None:
        metrics["folds"] = {
            str(fold): error_summary([row for row in errors if row.get("fold") == fold])
            for fold in range(1, folds + 1)
        }
        if set(metrics["folds"]) != {str(i) for i in range(1, folds + 1)}:
            raise RuntimeError(f"Expected exactly {folds} fold summaries")
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def load_saved_config(run_dir: Path) -> dict:
    path = run_dir / "resolved_settings.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Saved settings are missing from {run_dir}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not {"cv", "preparation", "inference", "train"} <= set(config):
        raise ValueError(f"Invalid saved settings in {path}")
    return config


def train_final_command(cv_run_id: str) -> int:
    cv_dir = ROOT / "runs" / cv_run_id
    cv_summary_path = cv_dir / "cv_summary.json"
    if not cv_summary_path.is_file():
        raise FileNotFoundError(f"Completed CV run not found: {cv_run_id}")
    cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    if cv_summary.get("status") != "completed":
        raise RuntimeError("Final training requires a completed CV run")
    final_id = f"final_{cv_run_id}"
    final_dir = ROOT / "runs" / final_id
    if (final_dir / "final.json").is_file():
        print((final_dir / "final.json").read_text(encoding="utf-8"))
        return 0
    if final_dir.exists() and any(final_dir.iterdir()):
        raise RuntimeError(f"Final output exists and will not be overwritten: {final_dir}")

    config = load_saved_config(cv_dir)
    selected = cv_summary["selected_epoch_counts"]
    epochs = int(math.floor(statistics.median(selected) + 0.5))
    split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    rows = read_jsonl(PREP / "frozen_manifest.jsonl")
    dev_rows = [row for row in rows if row["match"] in split_data["development_matches"]]
    if set(row["match"] for row in dev_rows) & set(split_data["held_out_matches"]):
        raise AssertionError("Held-out matches leaked into final training")
    final_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = write_pose_data_yaml(final_dir / "dataset", dev_rows, dev_rows)
    train_cfg = dict(config["train"])
    train_cfg["epochs"] = epochs
    # Keep the fixed median epoch budget; patience must not stop this final-epoch run early.
    train_cfg["patience"] = epochs + 1
    torch, _, _, _ = load_ultralytics()
    model = init_custom_model(seed=int(train_cfg.get("seed", 42)) + 10001)
    report = transfer_report(model)
    write_json(final_dir / "weight_transfer.json", report)
    resolved = {key: value for key, value in config.items()}
    resolved["train"] = train_cfg
    resolved["final_training"] = {
        "source_cv_run": cv_run_id, "selected_epoch_counts": selected,
        "rounded_median_epochs": epochs, "patience_override_to_finish_epoch_budget": epochs + 1,
    }
    (final_dir / "resolved_settings.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    write_json(final_dir / "dependencies.json", dependency_snapshot())
    args = build_training_args(train_cfg, data_yaml, final_dir, "train")
    amp_validation = train_with_local_amp_check(model, args)
    save_dir = Path(model.trainer.save_dir)
    best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
    if not last.is_file() or selected_epoch(last) != epochs:
        raise RuntimeError("Final training did not produce last.pt at the requested median epoch count")
    write_json(final_dir / "final.json", {
        "status": "completed", "run_id": final_id, "source_cv_run": cv_run_id,
        "completed_at": utc_now(), "epochs": epochs,
        "selected_epoch_counts": selected,
        "best_checkpoint": str(best.resolve()) if best.exists() else None,
        "last_checkpoint": str(last.resolve()), "last_sha256": sha256_file(last),
        "amp_check": amp_validation,
        "held_out_matches": split_data["held_out_matches"],
        "development_matches": split_data["development_matches"],
        "training_frames": len(dev_rows), "test_used_for_training": False,
    })
    print((final_dir / "final.json").read_text(encoding="utf-8"))
    return 0


def evaluate_test_command(final_run_id: str) -> int:
    final_dir = ROOT / "runs" / final_run_id
    final_path = final_dir / "final.json"
    if not final_path.is_file():
        raise FileNotFoundError(f"Completed final training run not found: {final_run_id}")
    final_info = json.loads(final_path.read_text(encoding="utf-8"))
    if final_info.get("status") != "completed":
        raise RuntimeError("Test evaluation requires a completed final model")
    output_dir = final_dir / "test_evaluation"
    if (output_dir / "metrics.json").exists():
        raise RuntimeError("The held-out test set has already been evaluated for this final run")
    config = load_saved_config(final_dir)
    split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    held_out = set(split_data["held_out_matches"])
    rows = [row for row in read_jsonl(PREP / "frozen_manifest.jsonl") if row["match"] in held_out]
    if not rows or set(row["match"] for row in rows) != held_out:
        raise RuntimeError("Held-out test manifest is incomplete")
    positions, errors = predict_and_score(final_info["last_checkpoint"], rows, config, output_dir)
    metrics = write_evaluation(output_dir, positions, errors)
    write_json(output_dir / "test_evaluation.json", {
        "created_at": utc_now(), "weights_sha256": final_info["last_sha256"],
        "held_out_matches": sorted(held_out), "frames": len(rows), "metrics": metrics,
    })
    print(json.dumps({"output_dir": str(output_dir), "held_out_matches": sorted(held_out),
                      "metrics": metrics["combined"]}, indent=2))
    return 0


def overlay_command(final_run_id: str, match: str, game: int, rally_id: int) -> int:
    final_dir = ROOT / "runs" / final_run_id
    final_path = final_dir / "final.json"
    if not final_path.is_file():
        raise FileNotFoundError(f"Completed final model not found: {final_run_id}")
    final_info = json.loads(final_path.read_text(encoding="utf-8"))
    config = load_saved_config(final_dir)
    key = ("set-1", match, str(game), str(rally_id))
    outcomes = {tuple(row["key"]): row for row in read_jsonl(PREP / "rallies.jsonl")}
    outcome = outcomes.get(key)
    if not outcome or outcome["status"] != "automatic_pass":
        raise ValueError(f"Rally did not pass automatic checks: {key}")
    frames = [row for row in read_jsonl(PREP / "manifest.jsonl") if stable_key(row) == key]
    if not frames:
        raise ValueError(f"No extracted frame provenance for rally: {key}")
    if not any(stable_key(row) == key for row in read_jsonl(PREP / "frozen_manifest.jsonl")):
        raise ValueError(f"Rally is not part of the frozen accepted dataset: {key}")
    frames.sort(key=lambda row: int(row["clip_frame_num"]))
    template = frames[0]
    clip_path = Path(template["clip_path"])
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open selected rally clip: {clip_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir = final_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_id(key)}.mp4"
    positions_path = output_dir / f"{safe_id(key)}_player_positions.csv"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create overlay video: {output_path}")
    torch, _, YOLO, _ = load_ultralytics()
    model = YOLO(final_info["last_checkpoint"], task="pose")
    model_contract(model)
    device = config["train"].get("device", 0) if torch.cuda.is_available() else "cpu"
    geometry = import_geometry(match)
    positions = []
    frame_index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            result = model.predict(
                source=image, imgsz=config["train"]["imgsz"],
                conf=float(config["inference"]["confidence"]), iou=float(config["inference"]["iou"]),
                device=device, classes=[0], verbose=False,
            )[0]
            marker = {**template, "width": width, "height": height}
            selected, statuses = assign_predictions(result, marker, geometry, config)
            for player, half, color in (
                ("A", template["A_half"], (255, 80, 30)),
                ("B", template["B_half"], (30, 100, 255)),
            ):
                candidate = selected.get(half)
                status = statuses[half]
                box = candidate["box"] if candidate else None
                point = candidate["point"] if candidate else None
                if candidate:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    px, py = [int(round(v)) for v in point]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(image, (px, py), 8, color, -1)
                    cv2.putText(image, f"{player} {candidate['box_confidence']:.2f}/"
                                f"{candidate['keypoint_confidence']:.2f}",
                                (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, f"{player}: {status}", (18, 35 + 28 * (player == "B")),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                positions.append({
                    "dataset": key[0], "match": match, "game": game,
                    "original_rally_id": rally_id, "clip_frame_num": frame_index,
                    "clip_time_seconds": frame_index / fps, "player": player,
                    "prediction_status": status, "available": candidate is not None,
                    "x_native": None if point is None else point[0],
                    "y_native": None if point is None else point[1],
                    "box": "" if box is None else json.dumps(box),
                    "box_confidence": None if candidate is None else candidate["box_confidence"],
                    "keypoint_confidence": None if candidate is None else candidate["keypoint_confidence"],
                    "court_half": half,
                })
            writer.write(image)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    write_csv(positions_path, positions, [
        *KEY_FIELDS, "clip_frame_num", "clip_time_seconds", "player", "prediction_status",
        "available", "x_native", "y_native", "box", "box_confidence",
        "keypoint_confidence", "court_half",
    ])
    print(json.dumps({"overlay_video": str(output_path), "player_positions": str(positions_path),
                      "frames": frame_index, "fps": fps, "width": width, "height": height}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Small grouped-CV YOLO11 player-keypoint experiment")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "split", "train-cv"):
        command = commands.add_parser(name)
        command.add_argument("--config", help="YAML config path")
        if name in {"split", "train-cv"}:
            command.add_argument("--folds", type=int, help="Explicit CV-fold override")
    commands.add_parser("review")
    final = commands.add_parser("train-final")
    final.add_argument("--run-id", required=True, help="Completed CV run id")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-id", required=True, help="Completed final run id")
    evaluate.add_argument("--split", choices=("test",), required=True)
    overlay = commands.add_parser("overlay")
    overlay.add_argument("--run-id", required=True, help="Completed final run id")
    overlay.add_argument("--match", required=True, help="Exact cleaned match folder name")
    overlay.add_argument("--game", type=int, required=True)
    overlay.add_argument("--rally", type=int, required=True, help="Original rally id")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "review":
            return review_command()
        if args.command == "train-final":
            return train_final_command(args.run_id)
        if args.command == "evaluate":
            if args.split != "test":
                raise ValueError("Only the fixed held-out test split is evaluated by this command")
            return evaluate_test_command(args.run_id)
        if args.command == "overlay":
            return overlay_command(args.run_id, args.match, args.game, args.rally)
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
