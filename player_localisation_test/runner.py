from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "monotrack_test"))
from reconstruction import (  # noqa: E402
    ANNOTATION_TO_VIDEO,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    load_court_geometry,
)
from run_projection import court_world_lines, load_annotations, video_metadata  # noqa: E402


VIDEO = REPO_ROOT / "cleaned_rallies/set-1/Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final/set1/rally_002.mp4"
ANNOTATIONS = VIDEO.with_suffix(".csv")
HOMOGRAPHY = REPO_ROOT / "set-1/homography.csv"
CHECKPOINT = ROOT / "checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
MODEL_NAME = "torchvision.models.detection.fasterrcnn_resnet50_fpn"
WEIGHTS_NAME = "FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
CONFIDENCE_THRESHOLD = 0.3
MAX_COURT_MARGIN_M = 1.0
EXPECTED_VIDEO = (1920, 1080, 25.0, 790)
PLAYER_ROWS = ("A", "B")
POSITION_FIELDS = (
    "clip_frame_num", "clip_time_seconds", "player", "available", "court_half",
    "score", "box_x1_px", "box_y1_px", "box_x2_px", "box_y2_px",
    "bottom_center_x_px", "bottom_center_y_px", "court_x_m", "court_y_m",
)
SHOT_FIELDS = (
    "clip_frame_num", "clip_time_seconds", "player", "prediction_status",
    "score", "predicted_x_px", "predicted_y_px", "reference_x_annotation_px",
    "reference_y_annotation_px", "reference_x_px", "reference_y_px",
    "predicted_x_m", "predicted_y_m", "reference_x_m", "reference_y_m",
    "error_px", "error_m",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path = CHECKPOINT) -> str:
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Checkpoint must be inside the pilot folder: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"Required local checkpoint is missing: {path}")
    digest = sha256(path)
    if not digest.startswith("258fb6c6"):
        raise ValueError(f"Checkpoint SHA-256 must start with 258fb6c6; found {digest}: {path}")
    return digest


def prepare_inputs() -> dict:
    for path in (VIDEO, ANNOTATIONS, HOMOGRAPHY):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = video_metadata(VIDEO)
    expected_w, expected_h, expected_fps, expected_count = EXPECTED_VIDEO
    if (metadata["width"], metadata["height"], metadata["frame_count"]) != (
        expected_w, expected_h, expected_count
    ) or not math.isclose(metadata["fps"], expected_fps, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Expected {expected_w}x{expected_h}, {expected_fps:g} fps, "
                         f"{expected_count} frames; found {metadata}")
    hits = load_annotations(ANNOTATIONS, metadata["frame_count"], metadata["fps"])
    if len(hits) != 29:
        raise ValueError(f"Expected 29 annotated shot rows; found {len(hits)}")
    for number, hit in enumerate(hits, start=1):
        expected_seconds = hit["clip_frame_num"] / metadata["fps"]
        if abs(expected_seconds - hit["clip_time_seconds"]) > 1e-6:
            raise ValueError(f"Shot row {number}: clip_time_seconds is not the rounded frame/FPS time")
        for suffix in ("x", "y"):
            for prefix in ("player_location", "opponent_location"):
                value = hit[f"{prefix}_{suffix}"]
                limit = 1280 if suffix == "x" else 720
                if not 0 <= value <= limit:
                    raise ValueError(f"Shot row {number}: {prefix}_{suffix} is outside 1280x720")
    geometry = load_court_geometry(HOMOGRAPHY, VIDEO.parent.parent.name)
    if geometry.row_id != "3":
        raise ValueError(f"Expected Korea Open homography row ID 3; found {geometry.row_id}")
    return {"metadata": metadata, "hits": hits, "geometry": geometry}


def align_hit_locations(hit: dict) -> dict[str, tuple[float, float]]:
    hitter = hit["player"]
    opponent = "B" if hitter == "A" else "A"
    return {
        hitter: (float(hit["player_location_x"]), float(hit["player_location_y"])),
        opponent: (float(hit["opponent_location_x"]), float(hit["opponent_location_y"])),
    }


def outside_distance_m(x: float, y: float) -> float:
    dx = max(0.0, -x, x - COURT_WIDTH_M)
    dy = max(0.0, -y, y - COURT_LENGTH_M)
    return math.hypot(dx, dy)


def select_candidates(
    detections: dict,
    geometry,
    frame_size: tuple[int, int] = (1920, 1080),
    threshold: float = CONFIDENCE_THRESHOLD,
    *, project=None,
) -> dict[str, dict | None]:
    width, height = frame_size
    selected: dict[str, dict | None] = {"far": None, "near": None}
    boxes = np.asarray(detections["boxes"])
    scores = np.asarray(detections["scores"])
    labels = np.asarray(detections["labels"])
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or len(boxes) != len(scores) or len(boxes) != len(labels):
        return selected
    for box, score, label in zip(boxes, scores, labels, strict=True):
        if int(label) != 1 or not np.isfinite(score) or float(score) < threshold:
            continue
        x1, y1, x2, y2 = map(float, box)
        if (not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1
                or x1 < 0 or y1 < 0 or x2 > width or y2 > height):
            continue
        bottom = ((x1 + x2) / 2.0, y2)
        try:
            court = np.asarray(project(bottom) if project else geometry.native_floor_to_world(bottom),
                               dtype=np.float64).reshape(2)
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue
        if not np.isfinite(court).all() or outside_distance_m(*court) > MAX_COURT_MARGIN_M:
            continue
        half = "far" if court[1] < NET_Y_M else "near"
        candidate = {
            "half": half, "score": float(score), "box": (x1, y1, x2, y2),
            "bottom": bottom, "court": (float(court[0]), float(court[1])),
        }
        if selected[half] is None or candidate["score"] > selected[half]["score"]:
            selected[half] = candidate
    return selected


def frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32).div_(255.0)


def inference(model, frame: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        result = model([frame_tensor(frame, device)])[0]
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def load_pretrained_model(device: torch.device, checkpoint_hash: str | None = None):
    if checkpoint_hash is None:
        validate_checkpoint()
    torch.hub.set_dir(str(ROOT))
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, progress=False
    )
    return model.to(device).eval()


def check_torchvision_ops(device: torch.device) -> None:
    boxes = torch.tensor([[0, 0, 4, 4], [1, 1, 5, 5]], dtype=torch.float32, device=device)
    keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.3)
    roi = torchvision.ops.roi_align(
        torch.ones((1, 1, 8, 8), device=device),
        [torch.tensor([[0, 0, 4, 4]], dtype=torch.float32, device=device)],
        output_size=(2, 2),
    )
    if keep.tolist() != [0] or tuple(roi.shape) != (1, 1, 2, 2):
        raise RuntimeError("Torchvision NMS or ROIAlign returned an unexpected result")


def fresh_output_dir(root: Path = ROOT, stamp: str | None = None) -> Path:
    root = root.resolve()
    parent = (root / "outputs").resolve()
    try:
        parent.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Output path escapes pilot folder: {parent}") from error
    parent.mkdir(parents=True, exist_ok=True)
    stem = f"run_{stamp or datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"
    suffix = 1
    while True:
        name = stem if suffix == 1 else f"{stem}_{suffix:02d}"
        target = parent / name
        try:
            target.mkdir()
            return target.resolve()
        except FileExistsError:
            suffix += 1


def make_shot_references(hits: list[dict], geometry) -> tuple[list[dict], dict[int, dict]]:
    rows, by_frame = [], {}
    for hit in hits:
        locations = align_hit_locations(hit)
        by_frame[hit["clip_frame_num"]] = {}
        for player in PLAYER_ROWS:
            annotation_xy = locations[player]
            world = np.asarray(geometry.annotation_floor_to_world(annotation_xy), dtype=np.float64)
            if not np.isfinite(world).all():
                raise ValueError(f"Non-finite court reference at frame {hit['clip_frame_num']} for {player}")
            by_frame[hit["clip_frame_num"]][player] = {
                "annotation": annotation_xy,
                "native": tuple(float(v * ANNOTATION_TO_VIDEO) for v in annotation_xy),
                "world": (float(world[0]), float(world[1])),
            }
            rows.append({"clip_frame_num": hit["clip_frame_num"], "clip_time_seconds": hit["clip_time_seconds"],
                         "player": player, "reference_annotation": annotation_xy,
                         "reference_native": by_frame[hit["clip_frame_num"]][player]["native"],
                         "reference_world": tuple(map(float, world))})
    return rows, by_frame


def shot_error_rows(references: list[dict], positions_by_key: dict[tuple[int, str], dict]) -> list[dict]:
    rows = []
    for reference in references:
        frame, player = reference["clip_frame_num"], reference["player"]
        position = positions_by_key[(frame, player)]
        available = position["available"]
        predicted_px = position.get("bottom")
        predicted_m = position.get("court")
        ref_px, ref_m = reference["reference_native"], reference["reference_world"]
        error_px = math.dist(predicted_px, ref_px) if available else None
        error_m = math.dist(predicted_m, ref_m) if available else None
        rows.append({
            "clip_frame_num": frame, "clip_time_seconds": reference["clip_time_seconds"],
            "player": player, "prediction_status": "available" if available else "missing",
            "score": position.get("score"),
            "predicted_x_px": predicted_px[0] if available else None,
            "predicted_y_px": predicted_px[1] if available else None,
            "reference_x_annotation_px": reference["reference_annotation"][0],
            "reference_y_annotation_px": reference["reference_annotation"][1],
            "reference_x_px": ref_px[0], "reference_y_px": ref_px[1],
            "predicted_x_m": predicted_m[0] if available else None,
            "predicted_y_m": predicted_m[1] if available else None,
            "reference_x_m": ref_m[0], "reference_y_m": ref_m[1],
            "error_px": error_px, "error_m": error_m,
        })
    return rows


def _error_stats(rows: list[dict], field: str) -> dict:
    values = np.asarray([row[field] for row in rows if row[field] is not None], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "euclidean_rmse": None}
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "euclidean_rmse": float(np.sqrt(np.mean(values ** 2))),
    }


def metrics_summary(rows: list[dict]) -> dict:
    def summarize(group: list[dict]) -> dict:
        eligible = len(group)
        available = sum(row["prediction_status"] == "available" for row in group)
        within = {}
        for limit, key in ((0.5, "within_0_5_m"), (1.0, "within_1_m")):
            count = sum(row["error_m"] is not None and row["error_m"] <= limit for row in group)
            within[key] = {"count": count, "denominator": eligible,
                           "percentage": 100.0 * count / eligible if eligible else None}
        return {
            "eligible_references": eligible, "available_predictions": available,
            "missing_predictions": eligible - available,
            "coverage_percentage": 100.0 * available / eligible if eligible else None,
            "pixel_error": _error_stats(group, "error_px"),
            "metre_error": _error_stats(group, "error_m"), **within,
        }

    return {"combined": summarize(rows), "per_player": {
        player: summarize([row for row in rows if row["player"] == player]) for player in PLAYER_ROWS
    }}


def court_panel(frame_h: int, panel_w: int, current: dict, references: dict,
                previous: dict, frame_number: int) -> np.ndarray:
    panel = np.full((frame_h, panel_w, 3), 248, dtype=np.uint8)
    scale = min((panel_w - 70) / COURT_WIDTH_M, (frame_h - 190) / COURT_LENGTH_M)
    court_w, court_h = int(COURT_WIDTH_M * scale), int(COURT_LENGTH_M * scale)
    left, top = (panel_w - court_w) // 2, (frame_h - court_h) // 2

    def pixel(point):
        x, y = point
        return (int(round(left + x * scale)), int(round(top + (COURT_LENGTH_M - y) * scale)))

    cv2.putText(panel, "COURT (metres)", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
    for start, end in court_world_lines():
        cv2.line(panel, pixel(start), pixel(end), (85, 85, 85), 2 if start[1] == end[1] == NET_Y_M else 1)
    colors = {"A": (40, 150, 235), "B": (210, 95, 30)}
    for player, location in current.items():
        if player in previous:
            cv2.line(panel, pixel(previous[player]), pixel(location), colors[player], 3, cv2.LINE_AA)
        cv2.circle(panel, pixel(location), 7, colors[player], -1, cv2.LINE_AA)
    for player, reference in references.items():
        cv2.drawMarker(panel, pixel(reference["world"]), colors[player], cv2.MARKER_CROSS, 18, 2)
        cv2.putText(panel, f"{player} ref", (pixel(reference["world"])[0] + 7, pixel(reference["world"])[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[player], 1, cv2.LINE_AA)
    cv2.putText(panel, "A far", (24, frame_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["A"], 2)
    cv2.putText(panel, "B near", (130, frame_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["B"], 2)
    if references:
        cv2.putText(panel, f"shot references | frame {frame_number}", (24, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)
    return panel


def render_frame(frame: np.ndarray, selected: dict, references: dict, previous: dict,
                 frame_number: int) -> tuple[np.ndarray, dict]:
    source = frame.copy()
    colors = {"A": (40, 150, 235), "B": (210, 95, 30)}
    current = {}
    for player, half in (("A", "far"), ("B", "near")):
        candidate = selected[half]
        if candidate is None:
            continue
        current[player] = candidate["court"]
        x1, y1, x2, y2 = map(lambda x: int(round(x)), candidate["box"])
        cx, cy = map(lambda x: int(round(x)), candidate["bottom"])
        cv2.rectangle(source, (x1, y1), (x2, y2), colors[player], 2)
        cv2.circle(source, (cx, cy), 6, colors[player], -1)
        cv2.putText(source, f"{player} {candidate['score']:.2f}", (x1, max(22, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[player], 2, cv2.LINE_AA)
    panel = court_panel(frame.shape[0], 600, current, references, previous, frame_number)
    cv2.putText(source, f"frame {frame_number}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
    return np.hstack((source, panel)), current


def verify_review(path: Path, expected_frames: int, expected_fps: float) -> dict:
    metadata = video_metadata(path)
    if metadata["frame_count"] != expected_frames or not math.isclose(
        metadata["fps"], expected_fps, rel_tol=0, abs_tol=1e-3
    ):
        raise RuntimeError(f"Review video metadata mismatch: {metadata}")
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    if count != expected_frames:
        raise RuntimeError(f"Review video decoded {count} frames; expected {expected_frames}")
    return {**metadata, "decoded_frames": count}


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def provenance(inputs: dict, checkpoint_hash: str, device: torch.device) -> dict:
    metadata, geometry = inputs["metadata"], inputs["geometry"]
    def file_info(path: Path, include_hash: bool) -> dict:
        stat = path.stat()
        info = {"path": str(path.resolve()), "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
        if include_hash:
            info["sha256"] = sha256(path)
        return info

    return {
        "model": {"architecture": MODEL_NAME, "weights": WEIGHTS_NAME,
                  "weights_url": FasterRCNN_ResNet50_FPN_Weights.COCO_V1.url,
                  "relationship_to_monotrack": "Same Faster R-CNN detector family; this pilot uses the separate official COCO_V1 checkpoint."},
        "checkpoint": {"path": str(CHECKPOINT.resolve()), "sha256": checkpoint_hash,
                       "required_sha256_prefix": "258fb6c6"},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "opencv": cv2.__version__,
                    "numpy": np.__version__, "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()},
        "input_provenance": {
            "video": {**file_info(VIDEO, False), **metadata},
            "annotations": {**file_info(ANNOTATIONS, True), "shot_rows": len(inputs["hits"]),
                            "coordinate_size": [1280, 720]},
            "homography_file": file_info(HOMOGRAPHY, True),
        },
        "homography": {"row_id": geometry.row_id, "video": geometry.row_video,
                       "annotation_to_dataset": geometry.homography_annotation_to_dataset.tolist(),
                       "dataset_to_metric": geometry.dataset_to_metric.tolist(),
                       "mapped_dataset_bounds": list(geometry.mapped_bounds),
                       "coordinate_units": "metres after dataset_to_metric"},
        "fixed_settings": {
            "person_coco_label": 1, "confidence_threshold": CONFIDENCE_THRESHOLD,
            "bottom_centre": "((x1+x2)/2, y2) in native video pixels",
            "annotation_to_video_scale": ANNOTATION_TO_VIDEO,
            "court_margin_m": MAX_COURT_MARGIN_M, "court_half_boundary_y_m": NET_Y_M,
            "far_identity": "A", "near_identity": "B", "smoothing": False,
            "interpolation": False, "clamping": False,
            "pixel_error_reference_units": "native video pixels; annotation references multiplied by 1.5",
        },
    }


def check_command() -> int:
    try:
        inputs = prepare_inputs()
        print(f"PASS inputs/timing: {len(inputs['hits'])} shots, {2 * len(inputs['hits'])} references; "
              f"{inputs['metadata']['frame_count']} frames at {inputs['metadata']['fps']:g} fps")
        geometry = inputs["geometry"]
        print(f"PASS homography: row {geometry.row_id} ({geometry.row_video}); dataset-to-metre transform loaded")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        check_torchvision_ops(device)
        print(f"PASS Torchvision NMS/ROIAlign on {device}")
        if not CHECKPOINT.is_file():
            print(f"MISSING checkpoint: {CHECKPOINT}")
            print("Real inference is pending; weights were not downloaded.")
            return 2
        digest = validate_checkpoint()
        model = load_pretrained_model(device, digest)
        capture = cv2.VideoCapture(str(VIDEO))
        try:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not decode frame 0 for the pretrained one-frame inference check")
            prediction = inference(model, frame, device)
        finally:
            capture.release()
        print(f"PASS checkpoint SHA-256 {digest}; official {WEIGHTS_NAME} model loaded; "
              f"one-frame inference returned {len(prediction['boxes'])} boxes on {device}")
        return 0
    except Exception as error:
        print(f"FAIL check: {type(error).__name__}: {error}")
        return 1


def run_command() -> int:
    inputs = prepare_inputs()
    checkpoint_hash = validate_checkpoint()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(device, checkpoint_hash)
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not decode the first video frame")
        selected = select_candidates(inference(model, frame, device), inputs["geometry"])
        output_dir = fresh_output_dir()
        review_path = output_dir / "review.mp4"
        writer = cv2.VideoWriter(str(review_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 inputs["metadata"]["fps"], (frame.shape[1] + 600, frame.shape[0]))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create review video: {review_path}")
        references, refs_by_frame = make_shot_references(inputs["hits"], inputs["geometry"])
        positions, positions_by_key = [], {}
        previous, frame_number = {}, 0
        while True:
            frame_time = frame_number / inputs["metadata"]["fps"]
            current = {}
            for player, half in (("A", "far"), ("B", "near")):
                candidate = selected[half]
                record = {"clip_frame_num": frame_number, "clip_time_seconds": frame_time,
                          "player": player, "available": candidate is not None,
                          "court_half": half}
                if candidate is not None:
                    record.update({"score": candidate["score"], "box": candidate["box"],
                                   "bottom": candidate["bottom"], "court": candidate["court"]})
                    current[player] = candidate["court"]
                else:
                    record.update({"score": None, "box": None, "bottom": None, "court": None})
                positions.append(record)
                positions_by_key[(frame_number, player)] = record
            rendered, previous = render_frame(frame, selected, refs_by_frame.get(frame_number, {}),
                                               previous, frame_number)
            writer.write(rendered)
            frame_number += 1
            ok, frame = capture.read()
            if not ok:
                break
            selected = select_candidates(inference(model, frame, device), inputs["geometry"])
        if frame_number != inputs["metadata"]["frame_count"]:
            raise RuntimeError(f"Decoded {frame_number} source frames; expected {inputs['metadata']['frame_count']}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if len(positions) != EXPECTED_VIDEO[3] * 2:
        raise RuntimeError(f"Expected 1,580 position rows; found {len(positions)}")
    shot_rows = shot_error_rows(references, positions_by_key)
    if len(shot_rows) != 58:
        raise RuntimeError(f"Expected 58 shot evaluation rows; found {len(shot_rows)}")
    review_info = verify_review(review_path, EXPECTED_VIDEO[3], EXPECTED_VIDEO[2])
    position_rows = []
    for row in positions:
        box = row.get("box") or (None,) * 4
        bottom = row.get("bottom") or (None, None)
        court = row.get("court") or (None, None)
        position_rows.append({
            "clip_frame_num": row["clip_frame_num"], "clip_time_seconds": row["clip_time_seconds"],
            "player": row["player"], "available": row["available"], "court_half": row["court_half"],
            "score": row["score"], "box_x1_px": box[0], "box_y1_px": box[1],
            "box_x2_px": box[2], "box_y2_px": box[3],
            "bottom_center_x_px": bottom[0], "bottom_center_y_px": bottom[1],
            "court_x_m": court[0], "court_y_m": court[1],
        })
    _write_csv(output_dir / "player_positions.csv", position_rows, POSITION_FIELDS)
    _write_csv(output_dir / "shot_errors.csv", shot_rows, SHOT_FIELDS)
    metrics = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "provenance": provenance(inputs, checkpoint_hash, device),
               "counts": {"decoded_frames": frame_number, "position_rows": len(positions),
                          "shot_references": len(shot_rows), "review_video": review_info},
               **metrics_summary(shot_rows)}
    with (output_dir / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Run complete: {output_dir}")
    print(json.dumps({"combined": metrics["combined"], "per_player": metrics["per_player"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrained Faster R-CNN court-plane player localization pilot")
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    if args.command == "check":
        return check_command()
    try:
        return run_command()
    except Exception as error:
        print(f"FAIL run: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
