"""Manual calibration and one-rally TrackNet/MonoTrack projection runner."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from reconstruction import (
    ANNOTATION_TO_VIDEO,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    POST_TOP_M,
    SOURCE_COMMIT,
    CourtGeometry,
    fit_camera_projection,
    frame_time_seconds,
    intersect_camera_ray_y,
    join_piecewise,
    load_court_geometry,
    project_point,
    reconstruct_flight,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_VIDEO = REPO_ROOT / "cleaned_rallies/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals/set1/rally_015.mp4"
DEFAULT_HITS = DEFAULT_VIDEO.with_suffix(".csv")
DEFAULT_HOMOGRAPHY = REPO_ROOT / "set-1/homography.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "TrackNetv2-pretrained/track.pt"
EXPECTED_VIDEO = (1920, 1080, 25.0, 524)
OUTPUT_SIZE = (1440, 720)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def inside_test(path: Path) -> Path:
    path = path.resolve()
    try:
        path.relative_to(HERE)
    except ValueError as error:
        raise ValueError(f"All new outputs must stay inside {HERE}: {path}") from error
    return path


def ensure_output_dir(value: str | Path) -> Path:
    output_dir = inside_test(repo_path(value))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def reserve_output_files(paths: list[Path]) -> list[Path]:
    created = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb"):
                pass
            created.append(path)
    except FileExistsError as error:
        for created_path in created:
            created_path.unlink(missing_ok=True)
        raise FileExistsError(f"Refusing to overwrite output: {error.filename or path}") from error
    except Exception:
        for created_path in created:
            created_path.unlink(missing_ok=True)
        raise
    return created


def video_metadata(video: Path) -> dict:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def load_annotations(csv_path: Path, frame_count: int, fps: float) -> list[dict]:
    required = {
        "player", "clip_frame_num", "clip_time_seconds",
        "player_location_x", "player_location_y",
        "opponent_location_x", "opponent_location_y",
    }
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Rally CSV is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if len(rows) != 18:
        raise ValueError(f"This one-rally test expects 18 annotated hits; found {len(rows)}")
    hits = []
    for row_number, row in enumerate(rows, start=1):
        frame = int(row["clip_frame_num"])
        seconds = float(row["clip_time_seconds"])
        if not 0 <= frame < frame_count:
            raise ValueError(f"Hit row {row_number} points outside the video: frame {frame}")
        if abs(frame_time_seconds(frame, fps) - seconds) > 0.01:
            raise ValueError(
                f"Hit row {row_number} timing disagrees: frame {frame} at {fps:g} fps is "
                f"{frame_time_seconds(frame, fps):.6f}s, CSV says {seconds:.6f}s"
            )
        if row["player"] not in ("A", "B"):
            raise ValueError(f"Hit row {row_number} has unknown player identity {row['player']!r}")
        for key in ("player_location_x", "player_location_y", "opponent_location_x", "opponent_location_y"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Hit row {row_number} has a non-finite {key}")
        row["clip_frame_num"] = frame
        row["clip_time_seconds"] = seconds
        hits.append(row)
    frames = [hit["clip_frame_num"] for hit in hits]
    if frames != sorted(set(frames)):
        raise ValueError("Hit rows must have unique, increasing clip_frame_num values")
    return hits


def prepare_inputs(video_value: str | Path, hits_value: str | Path, homography_value: str | Path):
    video, hits_csv, homography_csv = map(repo_path, (video_value, hits_value, homography_value))
    for path in (video, hits_csv, homography_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = video_metadata(video)
    observed = (metadata["width"], metadata["height"], metadata["fps"], metadata["frame_count"])
    if observed != EXPECTED_VIDEO:
        raise ValueError(f"Expected rally metadata {EXPECTED_VIDEO}; found {observed}")
    hits = load_annotations(hits_csv, metadata["frame_count"], metadata["fps"])
    geometry = load_court_geometry(homography_csv, video.parent.parent.name)
    return video, hits_csv, homography_csv, metadata, hits, geometry


def court_world_lines() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    x0, x1 = 0.0, COURT_WIDTH_M
    y0, y1 = 0.0, COURT_LENGTH_M
    lines = [((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))]
    for x in (0.46, COURT_WIDTH_M - 0.46):
        lines.append(((x, 0.0), (x, COURT_LENGTH_M)))
    for y in (1.98, COURT_LENGTH_M - 1.98, 0.76, COURT_LENGTH_M - 0.76, NET_Y_M):
        lines.append(((0.0, y), (COURT_WIDTH_M, y)))
    return lines


def draw_calibration_preview(
    frame: np.ndarray,
    geometry: CourtGeometry,
    selection: str | None,
    points: dict,
    frame_index: int,
    preview_scale: float,
) -> np.ndarray:
    image = frame.copy()
    corners = np.rint(geometry.corners_annotation * ANNOTATION_TO_VIDEO).astype(int)
    for index, point in enumerate(corners):
        cv2.circle(image, tuple(point), 9, (0, 80, 255), -1)
        cv2.putText(image, f"C{index + 1}", tuple(point + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
    for line in court_world_lines():
        p0 = geometry.world_to_annotation_floor(line[0]) * ANNOTATION_TO_VIDEO
        p1 = geometry.world_to_annotation_floor(line[1]) * ANNOTATION_TO_VIDEO
        cv2.line(image, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), (255, 180, 0), 2)
    for key, item in points.items():
        if item is None:
            continue
        pixel = tuple(np.rint(item["pixel_native"]).astype(int))
        color = (0, 255, 0) if key in ("left_post_top", "right_post_top") else (255, 0, 255)
        cv2.drawMarker(image, pixel, color, cv2.MARKER_CROSS, 18, 3)
        cv2.putText(image, key.replace("_", " "), (pixel[0] + 9, pixel[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    status = f"frame {frame_index}/{EXPECTED_VIDEO[3] - 1} | selected: {selection or 'none'} | preview scale {preview_scale:.4f}"
    help_line = "1/2: post tips  3: net contact  4: floor contact  a/d: frame  A/D: 25 frames  u: unknown event  q: save/quit"
    cv2.rectangle(image, (0, 0), (image.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(image, status, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, help_line, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    preview = cv2.resize(image, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    return preview


def pick_calibration(video: Path, geometry: CourtGeometry, calibration_path: Path) -> dict:
    metadata = video_metadata(video)
    max_preview_width, max_preview_height = 1280, 720
    preview_scale = min(max_preview_width / metadata["width"], max_preview_height / metadata["height"], 1.0)
    preview_width = int(round(metadata["width"] * preview_scale))
    preview_height = int(round(metadata["height"] * preview_scale))
    scale_x, scale_y = preview_width / metadata["width"], preview_height / metadata["height"]
    points: dict[str, dict | None] = {
        "left_post_top": None,
        "right_post_top": None,
        "net_contact": None,
        "floor_contact": None,
    }
    state = {"frame": 0, "rendered_frame": -1, "image": None, "selection": None}
    capture = cv2.VideoCapture(str(video))
    window = "MonoTrack one-rally calibration"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or state["selection"] is None:
            return
        name = state["selection"]
        if name in ("net_contact", "floor_contact"):
            points[name] = None
        else:
            points[name] = {
                "frame": int(state["frame"]),
                "pixel_native": [float(x / scale_x), float(y / scale_y)],
            }

    cv2.setMouseCallback(window, on_mouse)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        while True:
            if state["rendered_frame"] != state["frame"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, state["frame"])
                ok, source_frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {state['frame']}")
                state["image"] = source_frame
                state["rendered_frame"] = state["frame"]
            preview = draw_calibration_preview(
                state["image"], geometry, state["selection"], points, state["frame"], preview_scale
            )
            cv2.imshow(window, preview)
            key = cv2.waitKeyEx(20)
            key = key & 0xFF if key >= 0 else -1
            if key in (ord("q"), 27):
                break
            if key in (ord("a"), ord("A")):
                state["frame"] = max(0, state["frame"] - (25 if key == ord("A") else 1))
            elif key in (ord("d"), ord("D")):
                state["frame"] = min(metadata["frame_count"] - 1, state["frame"] + (25 if key == ord("D") else 1))
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                state["selection"] = {
                    ord("1"): "left_post_top",
                    ord("2"): "right_post_top",
                    ord("3"): "net_contact",
                    ord("4"): "floor_contact",
                }[key]
            elif key == ord("u") and state["selection"] in ("net_contact", "floor_contact"):
                points[state["selection"]] = None
    finally:
        capture.release()
        cv2.destroyWindow(window)

    if points["left_post_top"] is None or points["right_post_top"] is None:
        raise ValueError("Calibration was not saved: both net-post top tips are required")
    corners_native = geometry.corners_annotation * ANNOTATION_TO_VIDEO
    pole_tips = np.asarray(
        [points["left_post_top"]["pixel_native"], points["right_post_top"]["pixel_native"]],
        dtype=np.float64,
    )
    projection, reprojection_rmse = fit_camera_projection(corners_native, pole_tips)
    terminal_events = {}
    for name in ("net_contact", "floor_contact"):
        item = points[name]
        terminal_events[name] = (
            {"status": "manual", "frame": int(item["frame"]), "pixel_native": item["pixel_native"]}
            if item is not None else {"status": "unidentified", "frame": None, "pixel_native": None}
        )
    document = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": str(video),
        "video_metadata": metadata,
        "geometry": {
            "homography_row_id": geometry.row_id,
            "homography_row_video": geometry.row_video,
            "homography_annotation_to_dataset": geometry.homography_annotation_to_dataset.tolist(),
            "dataset_to_metric": geometry.dataset_to_metric.tolist(),
            "corners_annotation": geometry.corners_annotation.tolist(),
            "mapped_corners_dataset": geometry.mapped_corners.tolist(),
            "mapped_bounds_dataset": list(geometry.mapped_bounds),
        },
        "scaling": {
            "annotation_to_native_video": ANNOTATION_TO_VIDEO,
            "preview_size": [preview_width, preview_height],
            "native_size": [metadata["width"], metadata["height"]],
            "preview_to_native": [1.0 / scale_x, 1.0 / scale_y],
        },
        "camera_calibration": {
            "method": "MonoTrack six-point linear projective calibration adapted for native pixels",
            "world_points_m": [
                [0.0, 0.0, 0.0], [COURT_WIDTH_M, 0.0, 0.0],
                [0.0, COURT_LENGTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M, 0.0],
                [0.0, NET_Y_M, POST_TOP_M], [COURT_WIDTH_M, NET_Y_M, POST_TOP_M],
            ],
            "corners_native": corners_native.tolist(),
            "left_post_top": points["left_post_top"],
            "right_post_top": points["right_post_top"],
            "projection_matrix": projection.tolist(),
            "calibration_reprojection_rmse_px": reprojection_rmse,
        },
        "terminal_events": terminal_events,
        "provenance": {
            "repository": "https://github.com/jhwang7628/monotrack",
            "commit": SOURCE_COMMIT,
        },
    }
    reserve_output_files([calibration_path])
    try:
        with calibration_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except Exception:
        calibration_path.unlink(missing_ok=True)
        raise
    return document


def load_calibration(path: Path, geometry: CourtGeometry, metadata: dict) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("format_version") != 1:
        raise ValueError("Unsupported calibration file version")
    saved_geometry = document.get("geometry", {})
    if saved_geometry.get("homography_row_video") != geometry.row_video:
        raise ValueError("Calibration file belongs to a different homography row")
    if not np.allclose(saved_geometry.get("homography_annotation_to_dataset"), geometry.homography_annotation_to_dataset, atol=1e-10):
        raise ValueError("Calibration file homography does not match the supplied CSV")
    if not np.allclose(saved_geometry.get("corners_annotation"), geometry.corners_annotation, atol=1e-8):
        raise ValueError("Calibration file court corners do not match the supplied CSV")
    if document.get("video_metadata") != metadata:
        raise ValueError("Calibration file video metadata does not match the input video")
    if document.get("scaling", {}).get("annotation_to_native_video") != ANNOTATION_TO_VIDEO:
        raise ValueError("Calibration file uses an unexpected annotation-to-video scale")
    for name in ("left_post_top", "right_post_top"):
        point = document.get("camera_calibration", {}).get(name, {}).get("pixel_native")
        pixel = np.asarray(point, dtype=np.float64)
        if (pixel.shape != (2,) or not np.isfinite(pixel).all()
                or not 0 <= pixel[0] < metadata["width"] or not 0 <= pixel[1] < metadata["height"]):
            raise ValueError(f"Calibration file has an invalid {name} selection")
    projection = np.asarray(document.get("camera_calibration", {}).get("projection_matrix"), dtype=np.float64)
    if projection.shape != (3, 4) or not np.isfinite(projection).all() or abs(np.linalg.det(projection[:, :3])) < 1e-12:
        raise ValueError("Calibration file has an invalid camera projection matrix")
    for name in ("net_contact", "floor_contact"):
        event = document.get("terminal_events", {}).get(name)
        if not isinstance(event, dict):
            raise ValueError(f"Calibration file is missing terminal event metadata: {name}")
        if event.get("status") == "manual":
            pixel = np.asarray(event.get("pixel_native"), dtype=np.float64)
            frame = event.get("frame")
            if (pixel.shape != (2,) or not np.isfinite(pixel).all()
                    or not 0 <= pixel[0] < metadata["width"] or not 0 <= pixel[1] < metadata["height"]
                    or not 0 <= int(frame) < metadata["frame_count"]):
                raise ValueError(f"Calibration file has an invalid {name} selection")
    return document


def load_tracker(checkpoint: Path):
    import torch
    import torchvision

    tracker_dir = REPO_ROOT / "TrackNetv2-pretrained"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TrackNet checkpoint not found: {checkpoint}")
    if str(tracker_dir) not in sys.path:
        sys.path.insert(0, str(tracker_dir))
    spec = importlib.util.spec_from_file_location("monotrack_test_track_video", tracker_dir / "track_video.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the repository's TrackNet video helper")
    tracker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tracker.load_model(checkpoint, device)
    return model, device, tracker, torchvision.transforms.ToTensor()


def track_shuttle(video: Path, expected_count: int, checkpoint: Path) -> tuple[list[tuple[int, int] | None], str]:
    import torch

    model, device, tracker, to_tensor = load_tracker(checkpoint)
    capture = cv2.VideoCapture(str(video))
    observations: list[tuple[int, int] | None] = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for TrackNet: {video}")
        while True:
            frames = []
            for _ in range(3):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            actual_count = len(frames)
            frames.extend([frames[-1]] * (3 - actual_count))
            batch = tracker.preprocess_group(frames, device, to_tensor)
            with torch.inference_mode():
                prediction = model(batch)
            if tuple(prediction.shape) != (1, 3, 288, 512):
                raise RuntimeError(f"Unexpected TrackNet output shape: {tuple(prediction.shape)}")
            masks = (prediction[0, :actual_count].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            for mask in masks:
                visible, x, y = tracker.get_shuttle_position(mask)
                observations.append((int(x * EXPECTED_VIDEO[0] / 512), int(y * EXPECTED_VIDEO[1] / 288)) if visible else None)
        if len(observations) != expected_count:
            raise RuntimeError(f"TrackNet decoded {len(observations)} frames; expected {expected_count}")
    finally:
        capture.release()
    return observations, str(device)


def player_references(hits: list[dict], geometry: CourtGeometry) -> list[dict]:
    result = []
    for row in hits:
        hitter_native = np.asarray([row["player_location_x"], row["player_location_y"]]) * ANNOTATION_TO_VIDEO
        opponent_native = np.asarray([row["opponent_location_x"], row["opponent_location_y"]]) * ANNOTATION_TO_VIDEO
        hitter_world = geometry.native_floor_to_world(hitter_native)
        opponent_world = geometry.native_floor_to_world(opponent_native)
        result.append(
            {
                "frame": row["clip_frame_num"],
                "hitter": row["player"],
                "A": hitter_world if row["player"] == "A" else opponent_world,
                "B": opponent_world if row["player"] == "A" else hitter_world,
            }
        )
    return result


def summarize_hit_observations(hits: list[dict], observations: list[tuple[int, int] | None], radius: int = 2) -> list[dict]:
    summary = []
    for hit in hits:
        hit_frame = hit["clip_frame_num"]
        candidates = [
            (abs(frame - hit_frame), frame, observations[frame])
            for frame in range(max(0, hit_frame - radius), min(len(observations), hit_frame + radius + 1))
            if observations[frame] is not None
        ]
        if candidates:
            gap, frame, pixel = min(candidates, key=lambda item: (item[0], item[1]))
            summary.append({
                "hit_frame": hit_frame,
                "hitter": hit["player"],
                "nearest_visible_track_frame_within_2": frame,
                "frame_offset": int(gap),
                "pixel_native": list(pixel),
            })
        else:
            summary.append({
                "hit_frame": hit_frame,
                "hitter": hit["player"],
                "nearest_visible_track_frame_within_2": None,
                "frame_offset": None,
                "pixel_native": None,
            })
    return summary


def fit_hit_flights(
    hits: list[dict], players: list[dict], observations: list[tuple[int, int] | None], projection: np.ndarray, fps: float
):
    coordinates = np.full((len(observations), 3), np.nan, dtype=np.float64)
    phases = [""] * len(observations)
    segment_ids = [-1] * len(observations)
    paths = []
    diagnostics = []
    start_fixed = None
    last_path = None
    for index in range(len(hits) - 1):
        start_frame = hits[index]["clip_frame_num"]
        end_frame = hits[index + 1]["clip_frame_num"]
        start_ref = players[index][hits[index]["player"]]
        receive_ref = players[index + 1][hits[index + 1]["player"]]
        path, detail = reconstruct_flight(
            start_frame=start_frame,
            end_frame=end_frame,
            observations=observations,
            projection=projection,
            start_reference_xy=start_ref,
            receive_reference_xy=receive_ref,
            fps=fps,
            start_fixed=start_fixed,
            phase=f"hit_{index + 1}_to_hit_{index + 2}",
        )
        diagnostics.append(detail)
        if path is None:
            start_fixed = None
            last_path = None
            continue
        coordinates[start_frame:end_frame] = path[:-1]
        for frame in range(start_frame, end_frame):
            phases[frame] = detail["phase"]
            segment_ids[frame] = index
        paths.append({"start_frame": start_frame, "end_frame": end_frame, "trajectory": path, "phase": detail["phase"]})
        last_path = path
        observed = observations[end_frame]
        reprojection_error = None
        if observed is not None:
            reprojection_error = float(np.linalg.norm(project_point(projection, path[-1]) - observed))
        detail["hit_endpoint_reprojection_error_px"] = reprojection_error
        # MonoTrack carries a hit point forward only when its image reprojection is close.
        start_fixed = path[-1].copy() if reprojection_error is not None and reprojection_error < 40.0 else None
    return coordinates, phases, segment_ids, paths, diagnostics, last_path


def fit_terminal_tail(
    calibration: dict,
    geometry: CourtGeometry,
    hits: list[dict],
    observations: list[tuple[int, int] | None],
    projection: np.ndarray,
    fps: float,
    last_path: np.ndarray | None,
):
    details = {"extension": "piecewise flight assumptions beyond MonoTrack hit-to-hit reconstruction", "phases": []}
    terminal_events = calibration["terminal_events"]
    net_event = terminal_events["net_contact"]
    floor_event = terminal_events["floor_contact"]
    net_point = None
    floor_point = None
    if net_event.get("status") == "manual":
        try:
            net_point = intersect_camera_ray_y(projection, net_event["pixel_native"], NET_Y_M)
            if not 0.0 <= net_point[2] <= POST_TOP_M:
                raise ValueError(f"Net-plane intersection height {net_point[2]:.3f} m is outside 0..{POST_TOP_M:.2f} m")
            details["net_contact_world_m"] = net_point.tolist()
            details["net_contact_status"] = "available"
        except (ValueError, np.linalg.LinAlgError) as error:
            details["net_contact_status"] = "unidentifiable"
            details["net_contact_reason"] = str(error)
    else:
        details["net_contact_status"] = "unidentifiable"
        details["net_contact_reason"] = "manual_net_contact_not_identified"
    if floor_event.get("status") == "manual":
        try:
            floor_xy = geometry.native_floor_to_world(floor_event["pixel_native"])
            floor_point = np.asarray([floor_xy[0], floor_xy[1], 0.0], dtype=np.float64)
            details["floor_contact_world_m"] = floor_point.tolist()
            details["floor_contact_status"] = "available"
            details["floor_contact_inside_court"] = bool(
                0.0 <= floor_point[0] <= COURT_WIDTH_M and 0.0 <= floor_point[1] <= COURT_LENGTH_M
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            details["floor_contact_status"] = "unidentifiable"
            details["floor_contact_reason"] = str(error)
    else:
        details["floor_contact_status"] = "unidentifiable"
        details["floor_contact_reason"] = "manual_floor_contact_not_identified"

    last_hit_frame = hits[-1]["clip_frame_num"]
    net_frame = int(net_event["frame"]) if net_event.get("status") == "manual" else None
    floor_frame = int(floor_event["frame"]) if floor_event.get("status") == "manual" else None
    net_frame_valid = net_frame is not None and last_hit_frame < net_frame < len(observations)
    floor_order_valid = (
        net_frame_valid and floor_point is not None and floor_frame is not None
        and net_frame < floor_frame < len(observations)
    )
    details["event_order_status"] = "available" if floor_order_valid else "unidentifiable"
    if not floor_order_valid:
        details["event_order_reason"] = "expected last hit < net contact < floor contact < frame count"

    paths = []
    if net_point is None:
        details["phases"].append({"phase": "last_hit_to_net_contact", "status": "unidentifiable", "reason": "net_contact_position_unavailable"})
    elif last_path is None:
        details["phases"].append({"phase": "last_hit_to_net_contact", "status": "unidentifiable", "reason": "last_hit_3d_position_unavailable"})
    elif not net_frame_valid:
        details["phases"].append({"phase": "last_hit_to_net_contact", "status": "unidentifiable", "reason": "net_contact_must_follow_last_hit"})
    else:
        start = last_path[-1]
        first, detail = reconstruct_flight(
            start_frame=last_hit_frame,
            end_frame=net_frame,
            observations=observations,
            projection=projection,
            start_reference_xy=start[:2],
            receive_reference_xy=net_point[:2],
            fps=fps,
            start_fixed=start,
            end_fixed=net_point,
            phase="last_hit_to_net_contact",
        )
        details["phases"].append(detail)
        if first is not None:
            paths.append({"start_frame": last_hit_frame, "end_frame": net_frame, "trajectory": first, "phase": detail["phase"]})

    if net_point is None or floor_point is None:
        details["phases"].append({"phase": "net_contact_to_floor_contact", "status": "unidentifiable", "reason": "terminal_contact_position_unavailable"})
    elif not floor_order_valid:
        details["phases"].append({"phase": "net_contact_to_floor_contact", "status": "unidentifiable", "reason": details.get("event_order_reason", "terminal_event_order_unavailable")})
    else:
        second, detail = reconstruct_flight(
            start_frame=net_frame,
            end_frame=floor_frame,
            observations=observations,
            projection=projection,
            start_reference_xy=net_point[:2],
            receive_reference_xy=floor_point[:2],
            fps=fps,
            start_fixed=net_point,
            end_fixed=floor_point,
            phase="net_contact_to_floor_contact",
        )
        details["phases"].append(detail)
        if second is not None:
            paths.append({"start_frame": net_frame, "end_frame": floor_frame, "trajectory": second, "phase": detail["phase"]})
            first_path = next((path["trajectory"] for path in paths if path["phase"] == "last_hit_to_net_contact"), None)
            if first_path is not None:
                join_piecewise(first_path, second)
    details["piecewise_contact_position_shared"] = bool(
        net_point is not None
        and any(item["phase"] == "last_hit_to_net_contact" for item in paths)
        and any(item["phase"] == "net_contact_to_floor_contact" for item in paths)
    )
    return paths, details, floor_point if floor_point is not None else None


def assign_paths(
    coordinates: np.ndarray,
    phases: list[str],
    segment_ids: list[int],
    paths: list[dict],
    tail: bool = False,
) -> None:
    for index, item in enumerate(paths):
        start, end, trajectory = item["start_frame"], item["end_frame"], item["trajectory"]
        if end <= start:
            continue
        if tail:
            coordinates[start:end + 1] = trajectory
            frame_range = range(start, end + 1)
        else:
            coordinates[start:end] = trajectory[:-1]
            frame_range = range(start, end)
        for frame in frame_range:
            phases[frame] = item["phase"]
            segment_ids[frame] = -2 if tail else index


def hold_terminal_marker(coordinates: np.ndarray, phases: list[str], segment_ids: list[int], floor_frame: int, floor_point: np.ndarray) -> None:
    coordinates[floor_frame:] = np.asarray(floor_point, dtype=np.float64)
    for frame in range(floor_frame, len(phases)):
        phases[frame] = "terminal_hold_measured_floor"
        segment_ids[frame] = -3


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames,start_time,duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def write_tracked_csv(path: Path, observations: list[tuple[int, int] | None]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "time_seconds", "visibility", "x_native_px", "y_native_px"))
        for frame, point in enumerate(observations):
            writer.writerow((frame, f"{frame_time_seconds(frame, EXPECTED_VIDEO[2]):.6f}", int(point is not None), *(point if point is not None else ("", ""))))


def write_coordinates_csv(
    path: Path,
    coordinates: np.ndarray,
    phases: list[str],
    segment_ids: list[int],
    observations: list[tuple[int, int] | None],
    projection: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "frame", "time_seconds", "phase", "segment_index", "track_visible",
            "observed_x_native_px", "observed_y_native_px", "x_m", "y_m", "z_m",
            "reprojected_x_native_px", "reprojected_y_native_px", "reprojection_error_px",
        ))
        for frame, position in enumerate(coordinates):
            observed = observations[frame]
            if np.isfinite(position).all():
                projected = project_point(projection, position)
                reprojection_error = float(np.linalg.norm(projected - observed)) if observed is not None else ""
                world = [f"{float(value):.6f}" for value in position]
                reprojection = [f"{float(value):.3f}" for value in projected]
            else:
                world, reprojection, reprojection_error = ["", "", ""], ["", ""], ""
            writer.writerow((
                frame, f"{frame_time_seconds(frame, EXPECTED_VIDEO[2]):.6f}", phases[frame], segment_ids[frame],
                int(observed is not None), *(observed if observed is not None else ("", "")),
                *world, *reprojection, f"{reprojection_error:.3f}" if isinstance(reprojection_error, float) else "",
            ))


def panel_point(xy: np.ndarray, court_rect: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, width, height = court_rect
    return (
        int(round(left + float(xy[0]) / COURT_WIDTH_M * width)),
        int(round(top + float(xy[1]) / COURT_LENGTH_M * height)),
    )


def render_frame(
    source_frame: np.ndarray,
    frame_index: int,
    observations: list[tuple[int, int] | None],
    coordinates: np.ndarray,
    phases: list[str],
    paths: list[dict],
    players_by_hit: list[dict],
    projection: np.ndarray,
) -> np.ndarray:
    canvas = np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), dtype=np.uint8)
    panel_width = OUTPUT_SIZE[0] // 2
    cv2.putText(canvas, "TRACKNET OBSERVATION / 3D REPROJECTION", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
    shown_h = 405
    shown_w = 720
    top = 80
    shown = cv2.resize(source_frame, (shown_w, shown_h), interpolation=cv2.INTER_AREA)
    canvas[top:top + shown_h, :shown_w] = shown
    observed = observations[frame_index]
    if observed is not None:
        point = (int(round(observed[0] * shown_w / EXPECTED_VIDEO[0])), top + int(round(observed[1] * shown_h / EXPECTED_VIDEO[1])))
        cv2.circle(canvas, point, 7, (0, 220, 255), 2)
    if np.isfinite(coordinates[frame_index]).all():
        projected = project_point(projection, coordinates[frame_index])
        if np.isfinite(projected).all():
            point = (int(round(projected[0] * shown_w / EXPECTED_VIDEO[0])), top + int(round(projected[1] * shown_h / EXPECTED_VIDEO[1])))
            cv2.drawMarker(canvas, point, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 3)
    cv2.putText(canvas, "yellow: TrackNet pixel     magenta: reprojected estimate", (18, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
    if np.isfinite(coordinates[frame_index]).all():
        position = coordinates[frame_index]
        state = "terminal marker hold" if phases[frame_index] == "terminal_hold_measured_floor" else phases[frame_index]
        if "net_contact" in state:
            state = f"extension (piecewise): {state}"
        text = f"{state} | x={position[0]:.2f} m  y={position[1]:.2f} m  z={position[2]:.2f} m"
    else:
        text = "3D estimate unavailable at this frame"
    cv2.putText(canvas, text[:88], (18, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(canvas, f"frame {frame_index:03d} / 523   t={frame_index / EXPECTED_VIDEO[2]:.2f}s", (18, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    x0 = panel_width
    cv2.putText(canvas, "TOP-DOWN COURT  |  metres", (x0 + 18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    court_height = 520
    court_width = int(round(court_height * COURT_WIDTH_M / COURT_LENGTH_M))
    court_rect = (x0 + (panel_width - court_width) // 2, 100, court_width, court_height)
    left, court_top, width, height = court_rect
    for a, b in court_world_lines():
        color = (100, 180, 100) if a[1] != NET_Y_M else (255, 180, 0)
        cv2.line(canvas, panel_point(np.asarray(a), court_rect), panel_point(np.asarray(b), court_rect), color, 1 if a[1] != NET_Y_M else 2)
    cv2.putText(canvas, "A", (left - 20, court_top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 80, 255), 2)
    cv2.putText(canvas, "B", (left - 20, court_top + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)

    active_players = None
    for item in players_by_hit:
        if item["frame"] <= frame_index:
            active_players = item
        else:
            break
    if active_players is not None:
        for player, color in (("A", (30, 80, 255)), ("B", (255, 120, 0))):
            point = panel_point(active_players[player], court_rect)
            cv2.circle(canvas, point, 9, color, -1)
            cv2.putText(canvas, player, (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for item in paths:
        if frame_index < item["start_frame"]:
            continue
        stop = min(frame_index, item["end_frame"])
        end_index = stop - item["start_frame"]
        if end_index < 1:
            continue
        points = [panel_point(point[:2], court_rect) for point in item["trajectory"][:end_index + 1]]
        color = (200, 0, 200) if "net_contact" in item["phase"] else (0, 220, 220)
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, color, 2)
    current = coordinates[frame_index]
    if np.isfinite(current).all():
        point = panel_point(current[:2], court_rect)
        cv2.circle(canvas, point, 8, (255, 255, 255), 2)
        cv2.putText(canvas, f"z={current[2]:.2f} m", (x0 + 18, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    else:
        cv2.putText(canvas, "no available estimate", (x0 + 18, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.line(canvas, (panel_width, 0), (panel_width, OUTPUT_SIZE[1]), (90, 90, 90), 1)
    return canvas


def render_h264_with_audio(
    video: Path,
    output: Path,
    output_dir: Path,
    observations: list[tuple[int, int] | None],
    coordinates: np.ndarray,
    phases: list[str],
    paths: list[dict],
    players: list[dict],
    projection: np.ndarray,
    frame_count: int,
) -> dict:
    if not shutil_which("ffmpeg") or not shutil_which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required to encode H.264 and retain source audio")
    source_probe = ffprobe(video)
    source_has_audio = any(stream.get("codec_type") == "audio" for stream in source_probe.get("streams", []))
    capture = cv2.VideoCapture(str(video))
    ffmpeg = shutil_which("ffmpeg")
    temporary = tempfile.TemporaryDirectory(prefix="projection-", dir=output_dir)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for rendering: {video}")
        silent = Path(temporary.name) / "silent_video.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "1440x720",
            "-framerate", "25", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-frames:v", str(frame_count), "-movflags", "+faststart", str(silent),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame_index in range(frame_count):
                ok, source_frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode source frame {frame_index} while rendering")
                composed = render_frame(
                    source_frame, frame_index, observations, coordinates, phases, paths, players, projection
                )
                process.stdin.write(composed.tobytes())
            process.stdin.close()
            error_output = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"H.264 video encoding failed: {error_output.strip()}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        mux = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(video),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy",
            "-t", f"{frame_count / EXPECTED_VIDEO[2]:.6f}", "-movflags", "+faststart", str(output),
        ]
        result = subprocess.run(mux, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Could not mux source audio: {result.stderr.strip()}")
        output_probe = ffprobe(output)
        stream = next((item for item in output_probe.get("streams", []) if item.get("codec_type") == "video"), None)
        if stream is None:
            raise RuntimeError("Encoded output has no video stream")
        actual_frames = int(stream.get("nb_read_frames", stream.get("nb_frames", -1)))
        rate_numerator, rate_denominator = stream["r_frame_rate"].split("/", 1)
        actual_fps = float(rate_numerator) / float(rate_denominator)
        duration = float(output_probe["format"]["duration"])
        audio_stream = next((item for item in output_probe.get("streams", []) if item.get("codec_type") == "audio"), None)
        output_has_audio = audio_stream is not None
        if (stream.get("codec_name") != "h264" or stream.get("width") != 1440 or stream.get("height") != 720
                or actual_frames != frame_count or abs(actual_fps - 25.0) > 1e-6
                or abs(duration - frame_count / 25.0) > 0.02 or (source_has_audio and not output_has_audio)):
            raise RuntimeError(f"Encoded output failed metadata/synchronization checks: {output_probe}")
        audio_start_delta = None
        audio_duration_delta = None
        if audio_stream is not None:
            audio_start_delta = abs(float(audio_stream.get("start_time", 0.0)) - float(stream.get("start_time", 0.0)))
            audio_duration_delta = abs(float(audio_stream.get("duration", duration)) - float(stream.get("duration", duration)))
            if audio_start_delta > 0.04 or (source_has_audio and audio_duration_delta > 0.04):
                raise RuntimeError("Output audio and video synchronization differs by more than one frame")
        return {
            "codec": stream["codec_name"],
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(actual_fps),
            "frames": actual_frames,
            "duration_seconds": duration,
            "source_audio_present": source_has_audio,
            "output_audio_present": output_has_audio,
            "audio_video_start_delta_seconds": audio_start_delta,
            "audio_video_duration_delta_seconds": audio_duration_delta,
        }
    finally:
        capture.release()
        temporary.cleanup()


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def run_projection(args) -> None:
    video, hits_csv, homography_csv, metadata, hits, geometry = prepare_inputs(
        args.video, args.rally_csv, args.homography_csv
    )
    calibration_path = inside_test(repo_path(args.calibration))
    output_dir = ensure_output_dir(args.output_dir)
    calibration = load_calibration(calibration_path, geometry, metadata)
    projection = np.asarray(calibration["camera_calibration"]["projection_matrix"], dtype=np.float64)
    stem = video.stem
    outputs = [
        output_dir / f"{stem}_tracked_pixels.csv",
        output_dir / f"{stem}_trajectory_3d.csv",
        output_dir / f"{stem}_diagnostics.json",
        output_dir / f"{stem}_projection.mp4",
    ]
    created = reserve_output_files(outputs)
    complete = False
    try:
        observations, device = track_shuttle(video, metadata["frame_count"], repo_path(args.checkpoint))
        players = player_references(hits, geometry)
        coordinates, phases, segment_ids, paths, hit_diagnostics, last_path = fit_hit_flights(
            hits, players, observations, projection, metadata["fps"]
        )
        tail_paths, tail_diagnostics, floor_point = fit_terminal_tail(
            calibration, geometry, hits, observations, projection, metadata["fps"], last_path
        )
        assign_paths(coordinates, phases, segment_ids, tail_paths, tail=True)
        floor_event = calibration["terminal_events"]["floor_contact"]
        net_event = calibration["terminal_events"]["net_contact"]
        floor_frame = int(floor_event["frame"]) if floor_event.get("status") == "manual" else None
        net_frame = int(net_event["frame"]) if net_event.get("status") == "manual" else None
        if (floor_point is not None and floor_frame is not None and floor_frame > hits[-1]["clip_frame_num"]
                and (net_frame is None or floor_frame > net_frame)):
            hold_terminal_marker(coordinates, phases, segment_ids, floor_frame, floor_point)
        all_paths = paths + tail_paths
        write_tracked_csv(outputs[0], observations)
        write_coordinates_csv(outputs[1], coordinates, phases, segment_ids, observations, projection)
        output_meta = render_h264_with_audio(
            video, outputs[3], output_dir, observations, coordinates, phases,
            all_paths, players, projection, metadata["frame_count"],
        )
        diagnostics = {
            "input_video": str(video),
            "rally_csv": str(hits_csv),
            "homography_csv": str(homography_csv),
            "homography_row_id": geometry.row_id,
            "input_metadata": metadata,
            "annotated_hits": len(hits),
            "hit_to_hit_segments_attempted": len(hit_diagnostics),
            "hit_to_hit_segments_available": sum(item["status"] == "available" for item in hit_diagnostics),
            "hit_track_neighborhood": summarize_hit_observations(hits, observations),
            "tracker": {"script": str(REPO_ROOT / "TrackNetv2-pretrained/track_video.py"), "checkpoint": str(repo_path(args.checkpoint)), "device": device},
            "frame_timing": {"authoritative": "clip_frame_num", "fps": metadata["fps"], "fr_adjust": 1.0, "frame_time": "frame / fps"},
            "camera_calibration_rmse_px": calibration["camera_calibration"]["calibration_reprojection_rmse_px"],
            "hit_to_hit_segments": hit_diagnostics,
            "terminal_extension": tail_diagnostics,
            "output_video": output_meta,
            "interpretation": "Player locations are soft fit references, so agreement with them is not independent accuracy evidence. A plausible top-down path does not verify reconstructed height.",
            "mono_track_source": {"repository": "https://github.com/jhwang7628/monotrack", "commit": SOURCE_COMMIT, "license": "Adobe Research License; noncommercial research only"},
        }
        with outputs[2].open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(diagnostics, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        complete = True
    finally:
        if not complete:
            for path in created:
                path.unlink(missing_ok=True)
    print(f"Tracked pixels: {outputs[0]}")
    print(f"3D coordinates: {outputs[1]}")
    print(f"Diagnostics: {outputs[2]}")
    print(f"Projection video: {outputs[3]}")
    print(f"Video checks: {output_meta}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("calibrate", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
        sub.add_argument("--rally-csv", type=Path, default=DEFAULT_HITS)
        sub.add_argument("--homography-csv", type=Path, default=DEFAULT_HOMOGRAPHY)
        sub.add_argument("--calibration", type=Path, required=True, help="Output JSON in calibrate mode; input JSON in run mode")
        sub.add_argument("--output-dir", type=Path, default=HERE)
        if name == "run":
            sub.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "calibrate":
            video, hits_csv, homography_csv, metadata, hits, geometry = prepare_inputs(
                args.video, args.rally_csv, args.homography_csv
            )
            output_dir = ensure_output_dir(args.output_dir)
            calibration_path = inside_test(repo_path(args.calibration))
            if calibration_path.exists():
                raise FileExistsError(f"Refusing to overwrite calibration: {calibration_path}")
            print(f"Video verified: {metadata}; {len(hits)} hit rows; mapped court bounds {geometry.mapped_bounds}")
            document = pick_calibration(video, geometry, calibration_path)
            print(f"Calibration saved: {calibration_path}")
            print(f"Camera fit reprojection RMSE: {document['camera_calibration']['calibration_reprojection_rmse_px']:.2f}px")
        else:
            calibration_path = inside_test(repo_path(args.calibration))
            args.output_dir = ensure_output_dir(args.output_dir)
            if not calibration_path.is_file():
                raise FileNotFoundError(f"Calibration JSON not found: {calibration_path}")
            args.calibration = calibration_path
            run_projection(args)
    except Exception as error:
        print(f"run_projection: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
