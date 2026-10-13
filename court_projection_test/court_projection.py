from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MONOTRACK_DIR = REPO_ROOT / "monotrack_test"
if str(MONOTRACK_DIR) not in sys.path:
    sys.path.insert(0, str(MONOTRACK_DIR))

# Reuse monotrack's rally/homography validation, court transforms, and safe output reservation.
from reconstruction import (  # noqa: E402
    ANNOTATION_TO_VIDEO,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    CourtGeometry,
)
from run_projection import (  # noqa: E402
    court_world_lines,
    draw_stroke_endpoints,
    ffprobe,
    prepare_inputs,
    player_references,
    reserve_output_files,
)


DEFAULT_VIDEO = REPO_ROOT / "cleaned_rallies/set-1/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals/set1/rally_015.mp4"
DEFAULT_RALLY_CSV = DEFAULT_VIDEO.with_suffix(".csv")
DEFAULT_HOMOGRAPHY = REPO_ROOT / "set-1/homography.csv"
DEFAULT_TRACKED_CSV = REPO_ROOT / "monotrack_test/run_001/rally_015_tracked_pixels.csv"
DEFAULT_CALIBRATION = HERE / "calibration.json"
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
PREVIEW_SIZE = (1280, 720)
OUTPUT_SIZE = (1440, 720)
MIN_INTERSECTION_ANGLE_DEG = 1.0
COURT_MARGIN_M = 2.0
MAX_ONE_PIXEL_SHIFT_M = 1.0
MAX_SHORT_GAP_FRAMES = 5
INTERPOLATED_COLOR = (0, 165, 255)  # OpenCV BGR orange.

POINT_KEYS = {
    ord("1"): "left_post_bottom",
    ord("2"): "left_post_top",
    ord("3"): "right_post_bottom",
    ord("4"): "right_post_top",
}


@dataclass(frozen=True)
class Stroke:
    index: int
    identifier: str
    start_frame: int
    end_frame_exclusive: int
    a_annotation: np.ndarray
    b_annotation: np.ndarray
    a_native: np.ndarray
    b_native: np.ndarray
    ab_line: np.ndarray | None
    line_status: str
    a_source: str
    b_source: str


def resolved_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def project_output_path(value: str | Path) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (HERE / path).resolve()
    try:
        path.relative_to(HERE)
    except ValueError as error:
        raise ValueError(f"Outputs must stay inside {HERE}: {path}") from error
    return path


def normalize_line(line: np.ndarray) -> np.ndarray | None:
    line = np.asarray(line, dtype=np.float64)
    if line.shape != (3,) or not np.isfinite(line).all():
        return None
    scale = float(np.linalg.norm(line[:2]))
    return None if scale < 1e-12 else line / scale


def normalize_vanishing_point(point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("The post selections do not define a finite homogeneous vanishing point")
    direction_scale = float(np.linalg.norm(point[:2]))
    if direction_scale < 1e-12 and abs(point[2]) < 1e-12:
        raise ValueError("The post selections do not define a unique vertical vanishing point")
    if abs(point[2]) <= 1e-10 * max(direction_scale, 1e-12):
        point = point / max(direction_scale, 1e-12)
        point[2] = 0.0
        first = point[0] if abs(point[0]) > 1e-12 else point[1]
        return -point if first < 0 else point
    point = point / point[2]
    point[2] = 1.0
    return point


def vertical_vanishing_point(points: dict) -> np.ndarray:
    """Return [u,v,1] for a finite point or normalized [du,dv,0] at infinity."""
    try:
        lb = np.asarray(points["left_post_bottom"]["pixel_native_px"], dtype=np.float64)
        lt = np.asarray(points["left_post_top"]["pixel_native_px"], dtype=np.float64)
        rb = np.asarray(points["right_post_bottom"]["pixel_native_px"], dtype=np.float64)
        rt = np.asarray(points["right_post_top"]["pixel_native_px"], dtype=np.float64)
    except (KeyError, TypeError) as error:
        raise ValueError("All four manual post selections are required") from error
    selections = (lb, lt, rb, rt)
    if any(point.shape != (2,) or not np.isfinite(point).all() for point in selections):
        raise ValueError("Post selections must be finite native video coordinates")
    if np.linalg.norm(lb - lt) < 1e-6 or np.linalg.norm(rb - rt) < 1e-6:
        raise ValueError("A post bottom and top selection must be distinct")
    left_line = normalize_line(np.cross(np.append(lb, 1.0), np.append(lt, 1.0)))
    right_line = normalize_line(np.cross(np.append(rb, 1.0), np.append(rt, 1.0)))
    if left_line is None or right_line is None:
        raise ValueError("A post selection does not define an image line")
    return normalize_vanishing_point(np.cross(left_line, right_line))


def line_through_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray | None, str]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None, "degenerate_ab_line"
    if float(np.linalg.norm(a - b)) < 1e-6:
        return None, "degenerate_ab_line"
    line = normalize_line(np.cross(np.append(a, 1.0), np.append(b, 1.0)))
    return (line, "available") if line is not None else (None, "degenerate_ab_line")


def intersect_floor_pixel(
    observed_native: tuple[float, float] | np.ndarray,
    ab_line: np.ndarray | None,
    vanishing_point: np.ndarray,
    line_status: str = "available",
) -> tuple[np.ndarray | None, float | None, str]:
    """Intersect a vertical image line with the projected AB floor line."""
    if ab_line is None:
        return None, None, line_status
    pixel = np.asarray(observed_native, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        return None, None, "invalid_observation"
    vp = normalize_vanishing_point(vanishing_point)
    ray_line = normalize_line(np.cross(np.append(pixel, 1.0), vp))
    if ray_line is None:
        return None, None, "degenerate_vertical_line"
    floor_line = normalize_line(ab_line)
    if floor_line is None:
        return None, None, "degenerate_ab_line"
    cosine = float(np.clip(abs(np.dot(floor_line[:2], ray_line[:2])), 0.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    if angle < MIN_INTERSECTION_ANGLE_DEG:
        return None, angle, "intersection_angle_below_1_degree"
    intersection = np.cross(floor_line, ray_line)
    if not np.isfinite(intersection).all():
        return None, angle, "nonfinite_intersection"
    if abs(intersection[2]) <= 1e-12 * max(float(np.linalg.norm(intersection[:2])), 1.0):
        return None, angle, "intersection_at_infinity"
    floor_pixel = intersection[:2] / intersection[2]
    if not np.isfinite(floor_pixel).all():
        return None, angle, "nonfinite_intersection"
    return floor_pixel, angle, "available"


def prepare_strokes(hits: list[dict], frame_count: int) -> list[Stroke]:
    if len(hits) != 18:
        raise ValueError(f"Expected 18 hit rows; found {len(hits)}")
    strokes = []
    for index, hit in enumerate(hits):
        start = int(hit["clip_frame_num"])
        end = int(hits[index + 1]["clip_frame_num"]) if index < len(hits) - 1 else frame_count
        a_prefix = "player_location" if index == 0 else "hit"
        try:
            a = np.asarray([float(hit[f"{a_prefix}_x"]), float(hit[f"{a_prefix}_y"])], dtype=np.float64)
            b = np.asarray([float(hit["landing_x"]), float(hit["landing_y"])], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Stroke {index + 1} needs numeric {a_prefix}_x/y and landing_x/y endpoints") from error
        a_source = "first_row_player_location" if index == 0 else "current_row_hit_x_y"
        # shortcut: the final net-fault plane ends at landing_x/y; refine it with a measured net-contact point.
        b_source = "final_row_landing_x_y" if index == len(hits) - 1 else "current_row_landing_x_y"
        if not np.isfinite(a).all() or not np.isfinite(b).all() or end <= start:
            raise ValueError(f"Invalid AB endpoints or frame interval for stroke {index + 1}")
        a_native, b_native = a * ANNOTATION_TO_VIDEO, b * ANNOTATION_TO_VIDEO
        line, line_status = line_through_points(a_native, b_native)
        strokes.append(Stroke(
            index=index,
            identifier=f"stroke_{index + 1:02d}",
            start_frame=start,
            end_frame_exclusive=end,
            a_annotation=a,
            b_annotation=b,
            a_native=a_native,
            b_native=b_native,
            ab_line=line,
            line_status=line_status,
            a_source=a_source,
            b_source=b_source,
        ))
    if strokes[0].start_frame < 0 or strokes[-1].end_frame_exclusive > frame_count:
        raise ValueError("Stroke frame intervals fall outside the video")
    return strokes


def strokes_by_frame(strokes: list[Stroke], frame_count: int) -> list[Stroke | None]:
    result: list[Stroke | None] = [None] * frame_count
    for stroke in strokes:
        for frame in range(stroke.start_frame, stroke.end_frame_exclusive):
            if result[frame] is not None:
                raise ValueError(f"Overlapping stroke intervals at frame {frame}")
            result[frame] = stroke
    return result


def load_tracked_detections(path: Path, metadata: dict) -> list[tuple[float, float] | None]:
    required = {"frame", "time_seconds", "visibility", "x_native_px", "y_native_px"}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Tracked-pixel CSV is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if len(rows) != metadata["frame_count"]:
        raise ValueError(f"Expected {metadata['frame_count']} tracker rows; found {len(rows)}")
    observations = []
    for index, row in enumerate(rows):
        try:
            frame = int(row["frame"])
            seconds = float(row["time_seconds"])
            visible = int(row["visibility"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid tracker row {index + 2}") from error
        if frame != index or abs(seconds - frame / metadata["fps"]) > 1e-5 or visible not in (0, 1):
            raise ValueError(f"Tracker frame/time/visibility mismatch at row {index + 2}")
        if not visible:
            observations.append(None)
            continue
        try:
            x, y = float(row["x_native_px"]), float(row["y_native_px"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Visible tracker row {index + 2} has no shuttle pixel") from error
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < metadata["width"] and 0 <= y < metadata["height"]):
            raise ValueError(f"Visible tracker row {index + 2} has an invalid native shuttle pixel")
        observations.append((x, y))
    return observations


def geometry_document(geometry: CourtGeometry) -> dict:
    return {
        "homography_row_id": geometry.row_id,
        "homography_row_video": geometry.row_video,
        "homography_annotation_to_dataset": geometry.homography_annotation_to_dataset.tolist(),
        "dataset_to_metric": geometry.dataset_to_metric.tolist(),
        "corners_annotation": geometry.corners_annotation.tolist(),
        "mapped_bounds_dataset": list(geometry.mapped_bounds),
    }


def file_provenance(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def calibration_document(
    video: Path, rally_csv: Path, homography_csv: Path, metadata: dict,
    geometry: CourtGeometry, points: dict,
) -> dict:
    vp = vertical_vanishing_point(points)
    finite = bool(vp[2] != 0.0)
    return {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_method": "manual_opencv_clicks",
        "source": {
            "video": str(video.resolve()),
            "rally_csv": str(rally_csv.resolve()),
            "homography_csv": str(homography_csv.resolve()),
            "video_metadata": metadata,
        },
        "geometry": geometry_document(geometry),
        "scaling": {
            "annotation_to_native_video": ANNOTATION_TO_VIDEO,
            "preview_size": list(PREVIEW_SIZE),
            "native_size": [metadata["width"], metadata["height"]],
            "preview_to_native": [metadata["width"] / PREVIEW_SIZE[0], metadata["height"] / PREVIEW_SIZE[1]],
        },
        "post_points": points,
        "homogeneous_vertical_vanishing_point_native": vp.tolist(),
        "vertical_vanishing_point_kind": "finite" if finite else "at_infinity",
    }


def write_json_exclusive(path: Path, document: dict) -> None:
    created = reserve_output_files([path])
    try:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except Exception:
        for item in created:
            item.unlink(missing_ok=True)
        raise


def load_calibration(path: Path, video: Path, metadata: dict, geometry: CourtGeometry) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("format_version") != 1 or document.get("selection_method") != "manual_opencv_clicks":
        raise ValueError("Unsupported calibration file or calibration was not recorded as manual selections")
    source = document.get("source", {})
    if Path(source.get("video", "")).resolve() != video.resolve() or source.get("video_metadata") != metadata:
        raise ValueError("Calibration source video or metadata does not match the input")
    saved_geometry = document.get("geometry", {})
    if saved_geometry.get("homography_row_video") != geometry.row_video:
        raise ValueError("Calibration file belongs to a different homography row")
    for key, expected in (
        ("homography_annotation_to_dataset", geometry.homography_annotation_to_dataset),
        ("dataset_to_metric", geometry.dataset_to_metric),
        ("corners_annotation", geometry.corners_annotation),
    ):
        if not np.allclose(saved_geometry.get(key), expected, atol=1e-10):
            raise ValueError(f"Calibration geometry does not match homography field {key}")
    scaling = document.get("scaling", {})
    if (scaling.get("annotation_to_native_video") != ANNOTATION_TO_VIDEO
            or scaling.get("preview_size") != list(PREVIEW_SIZE)
            or scaling.get("native_size") != [metadata["width"], metadata["height"]]):
        raise ValueError("Calibration uses unexpected annotation, preview, or native-video scaling")
    points = document.get("post_points", {})
    for name in POINT_KEYS.values():
        item = points.get(name, {})
        pixel = np.asarray(item.get("pixel_native_px"), dtype=np.float64)
        frame = item.get("frame_index")
        if (pixel.shape != (2,) or not np.isfinite(pixel).all()
                or not 0 <= pixel[0] < metadata["width"] or not 0 <= pixel[1] < metadata["height"]
                or not isinstance(frame, int) or not 0 <= frame < metadata["frame_count"]):
            raise ValueError(f"Calibration is missing a valid manual selection: {name}")
    vp = vertical_vanishing_point(points)
    saved_vp = np.asarray(document.get("homogeneous_vertical_vanishing_point_native"), dtype=np.float64)
    expected_kind = "finite" if vp[2] != 0.0 else "at_infinity"
    if (saved_vp.shape != (3,) or not np.allclose(saved_vp, vp, atol=1e-7)
            or document.get("vertical_vanishing_point_kind") != expected_kind):
        raise ValueError("Saved vertical vanishing point does not match the four manual selections")
    return document


def draw_calibration_preview(frame: np.ndarray, geometry: CourtGeometry, selected: str | None, points: dict,
                             frame_index: int, error: str) -> np.ndarray:
    image = frame.copy()
    corners = np.rint(geometry.corners_annotation * ANNOTATION_TO_VIDEO).astype(int)
    for index, point in enumerate(corners):
        cv2.circle(image, tuple(point), 8, (0, 80, 255), -1)
        cv2.putText(image, f"C{index + 1}", tuple(point + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
    for a, b in court_world_lines():
        p0 = geometry.world_to_annotation_floor(a) * ANNOTATION_TO_VIDEO
        p1 = geometry.world_to_annotation_floor(b) * ANNOTATION_TO_VIDEO
        cv2.line(image, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), (255, 150, 0), 2)
    colors = {
        "left_post_bottom": (255, 0, 255), "left_post_top": (0, 255, 0),
        "right_post_bottom": (255, 0, 255), "right_post_top": (0, 255, 0),
    }
    for name, item in points.items():
        if item is None:
            continue
        px = np.rint(np.asarray(item["pixel_native_px"])).astype(int)
        color = colors[name]
        cv2.drawMarker(image, tuple(px), color, cv2.MARKER_CROSS, 20, 3)
        cv2.putText(image, name.replace("_", " "), (px[0] + 8, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 78), (0, 0, 0), -1)
    cv2.putText(image, f"frame {frame_index} | selected: {selected or 'none'}", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(image, "1 left bottom  2 left top  3 right bottom  4 right top  |  click to record", (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(image, "a/d: step  A/D: 25 frames  s: save  q/Esc: cancel", (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    if error:
        cv2.putText(image, error[:110], (15, image.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return cv2.resize(image, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)


def calibrate(video: Path, rally_csv: Path, homography_csv: Path, calibration_path: Path,
              metadata: dict, geometry: CourtGeometry) -> bool:
    if calibration_path.exists():
        raise FileExistsError(f"Refusing to overwrite calibration: {calibration_path}")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    points: dict[str, dict | None] = {name: None for name in POINT_KEYS.values()}
    state = {"frame": 0, "loaded": -1, "image": None, "selected": None, "error": ""}
    scale_x, scale_y = PREVIEW_SIZE[0] / metadata["width"], PREVIEW_SIZE[1] / metadata["height"]
    capture = cv2.VideoCapture(str(video))
    window = "Court-plane shuttle calibration"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or state["selected"] is None:
            return
        points[state["selected"]] = {
            "frame_index": int(state["frame"]),
            "pixel_native_px": [float(x / scale_x), float(y / scale_y)],
        }
        state["error"] = ""

    cv2.setMouseCallback(window, on_mouse)
    saved = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open calibration video: {video}")
        while True:
            if state["loaded"] != state["frame"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, state["frame"])
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read calibration frame {state['frame']}")
                state["image"] = frame
                state["loaded"] = state["frame"]
            preview = draw_calibration_preview(
                state["image"], geometry, state["selected"], points, state["frame"], state["error"]
            )
            cv2.imshow(window, preview)
            key = cv2.waitKeyEx(25)
            key = key & 0xFF if key >= 0 else -1
            if key in (ord("q"), 27):
                break
            if key in (ord("a"), ord("A")):
                state["frame"] = max(0, state["frame"] - (25 if key == ord("A") else 1))
            elif key in (ord("d"), ord("D")):
                state["frame"] = min(metadata["frame_count"] - 1, state["frame"] + (25 if key == ord("D") else 1))
            elif key in POINT_KEYS:
                state["selected"] = POINT_KEYS[key]
                state["error"] = ""
            elif key == ord("s"):
                missing = [name for name, item in points.items() if item is None]
                if missing:
                    state["error"] = "Select all four post points before saving: " + ", ".join(missing)
                    continue
                try:
                    document = calibration_document(video, rally_csv, homography_csv, metadata, geometry, points)
                    write_json_exclusive(calibration_path, document)
                except (ValueError, FileExistsError) as error:
                    state["error"] = str(error)
                    continue
                saved = True
                break
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    return saved


def project_observations(observations: list[tuple[float, float] | None], strokes: list[Stroke],
                         vanishing_point: np.ndarray, geometry: CourtGeometry, fps: float,
                         player_locations: list[dict]) -> list[dict]:
    by_frame = strokes_by_frame(strokes, len(observations))
    horizon = geometry.annotation_to_metric[2] * [1 / ANNOTATION_TO_VIDEO, 1 / ANNOTATION_TO_VIDEO, 1]
    centre_native = geometry.corners_annotation.mean(axis=0) * ANNOTATION_TO_VIDEO
    # Homographies have arbitrary sign/scale; use the court centre to select the visible floor side.
    horizon = horizon / float(horizon @ np.append(centre_native, 1.0))
    results = []
    for frame, observed in enumerate(observations):
        stroke = by_frame[frame]
        row = {
            "frame": frame,
            "time_seconds": frame / fps,
            "stroke": stroke,
            "player_locations": None if stroke is None else {
                "A": player_locations[stroke.index]["A"],
                "B": player_locations[stroke.index]["B"],
            },
            "observed": observed,
            "endpoints": {} if stroke is None else player_locations[stroke.index].get("endpoints", {}),
            "estimated": None,
            "floor_pixel": None,
            "court_xy": None,
            "angle_deg": None,
        }
        if observed is None:
            row["status"] = "missing_detection"
        elif stroke is None:
            row["status"] = "before_first_stroke"
        else:
            floor_pixel, angle, status = intersect_floor_pixel(
                observed, stroke.ab_line, vanishing_point, stroke.line_status
            )
            row["angle_deg"] = angle
            if status == "available":
                try:
                    if float(horizon @ np.append(floor_pixel, 1.0)) <= 1e-9:
                        status = "floor_point_at_or_behind_horizon"
                    else:
                        court_xy = geometry.native_floor_to_world(floor_pixel)
                        if not np.isfinite(court_xy).all():
                            status = "nonfinite_floor_transform"
                        elif not (-COURT_MARGIN_M <= court_xy[0] <= COURT_WIDTH_M + COURT_MARGIN_M
                                  and -COURT_MARGIN_M <= court_xy[1] <= COURT_LENGTH_M + COURT_MARGIN_M):
                            status = "implausible_court_position"
                        else:
                            for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                perturbed_floor, _, perturbed_status = intersect_floor_pixel(
                                    np.asarray(observed) + offset, stroke.ab_line, vanishing_point, stroke.line_status
                                )
                                if (perturbed_status != "available"
                                        or float(horizon @ np.append(perturbed_floor, 1.0)) <= 1e-9):
                                    status = "unstable_floor_projection"
                                    break
                                perturbed_xy = geometry.native_floor_to_world(perturbed_floor)
                                if (not np.isfinite(perturbed_xy).all()
                                        or np.linalg.norm(perturbed_xy - court_xy) > MAX_ONE_PIXEL_SHIFT_M):
                                    status = "unstable_floor_projection"
                                    break
                except (ValueError, np.linalg.LinAlgError, cv2.error):
                    status = "nonfinite_floor_transform"
                if status == "available":
                    row["floor_pixel"] = floor_pixel
                    row["court_xy"] = court_xy
            row["status"] = status
        row["position_source"] = "tracked" if row["status"] == "available" else "unavailable"
        results.append(row)
    return results


def fill_short_gaps(rows: list[dict], strokes: list[Stroke], vanishing_point: np.ndarray,
                    geometry: CourtGeometry, fps: float, player_locations: list[dict]) -> tuple[list[dict], dict]:
    """Test bracketed missing runs using only the original tracked anchors; accept whole gaps."""
    candidates = [None] * len(rows)
    gaps = []
    counts = {"enabled": True, "max_gap_frames": MAX_SHORT_GAP_FRAMES,
              "eligible_gaps": 0, "accepted_gaps": 0, "rejected_gaps": 0, "skipped_gaps": 0}
    start = 0
    while start < len(rows):
        if rows[start]["status"] != "missing_detection":
            start += 1
            continue
        end = start + 1
        while end < len(rows) and rows[end]["status"] == "missing_detection":
            end += 1
        stroke = rows[start]["stroke"]
        if (end - start > MAX_SHORT_GAP_FRAMES or start == 0 or end == len(rows)
                or stroke is None
                or any(row["stroke"] is not stroke for row in rows[start - 1:end + 1])
                or any(rows[i]["position_source"] != "tracked" for i in (start - 1, end))):
            counts["skipped_gaps"] += 1
        else:
            times = [rows[start - 1]["time_seconds"], rows[end]["time_seconds"]]
            anchors = np.asarray([rows[start - 1]["observed"], rows[end]["observed"]])
            query = [row["time_seconds"] for row in rows[start:end]]
            pixels = np.column_stack([np.interp(query, times, anchors[:, axis]) for axis in (0, 1)])
            candidates[start:end] = [tuple(pixel) for pixel in pixels]
            gaps.append((start, end))
        start = end
    counts["eligible_gaps"] = len(gaps)
    projected = project_observations(candidates, strokes, vanishing_point, geometry, fps, player_locations)
    filled = [dict(row) for row in rows]
    for start, end in gaps:
        if all(row["status"] == "available" for row in projected[start:end]):
            for frame in range(start, end):
                filled[frame] = {**projected[frame], "observed": None,
                                 "estimated": candidates[frame], "position_source": "interpolated"}
            counts["accepted_gaps"] += 1
        else:
            counts["rejected_gaps"] += 1
    return filled, counts


def validate_short_gaps(observations: list[tuple[float, float] | None], strokes: list[Stroke],
                        vanishing_point: np.ndarray, geometry: CourtGeometry, fps: float,
                        player_locations: list[dict]) -> dict:
    """Independently mask the first bracketed valid window per stroke/length before reconstruction."""
    reference = project_observations(observations, strokes, vanishing_point, geometry, fps, player_locations)
    trials, summaries = [], {}
    for length in (1, 3, 5):
        native_errors, court_errors, ab_errors = [], [], []
        attempted = accepted = 0
        for stroke in strokes[:-1]:  # The final landing time is unknown.
            start = next((frame for frame in range(stroke.start_frame + 1, stroke.end_frame_exclusive - length)
                          if all(row["position_source"] == "tracked"
                                 for row in reference[frame - 1:frame + length + 1])), None)
            trial = {"stroke_id": stroke.identifier, "window_frames": length,
                     "start_frame": start, "accepted": False}
            trials.append(trial)
            if start is None:
                continue
            attempted += 1
            masked = list(observations)
            masked[start:start + length] = [None] * length
            masked_rows = project_observations(masked, strokes, vanishing_point, geometry, fps, player_locations)
            reconstructed, _ = fill_short_gaps(masked_rows, strokes, vanishing_point, geometry, fps, player_locations)
            trial["accepted"] = all(row["position_source"] == "interpolated"
                                    for row in reconstructed[start:start + length])
            if not trial["accepted"]:
                continue
            accepted += 1
            a_world = geometry.native_floor_to_world(stroke.a_native)
            b_world = geometry.native_floor_to_world(stroke.b_native)
            for frame in range(start, start + length):
                native_errors.append(float(np.linalg.norm(
                    np.asarray(reconstructed[frame]["estimated"]) - observations[frame])))
                court_errors.append(float(np.linalg.norm(
                    reconstructed[frame]["court_xy"] - reference[frame]["court_xy"])))
                fraction = ((reference[frame]["time_seconds"] - stroke.start_frame / fps)
                            / ((stroke.end_frame_exclusive - stroke.start_frame) / fps))
                ab = a_world + fraction * (b_world - a_world)
                ab_errors.append(float(np.linalg.norm(ab - reference[frame]["court_xy"])))
        metrics = lambda errors: {
            "mean_euclidean_error": float(np.mean(errors)) if errors else None,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        }
        summaries[str(length)] = {
            "attempted_windows": attempted, "accepted_windows": accepted,
            "no_eligible_window_strokes": len(strokes) - 1 - attempted,
            "accepted_frames": len(native_errors), "native_pixels": metrics(native_errors),
            "interpolated_court_m": metrics(court_errors), "endpoint_ab_court_m": metrics(ab_errors),
        }
    return {
        "reference": "Held-out TrackNet pixels and their projections, not independent ground truth",
        "selection": "First interior valid window per stroke and length, with valid tracked anchors",
        "baseline": "Constant speed in court metres between existing AB endpoints at contact frame/fps times",
        "excluded_stroke_id": strokes[-1].identifier,
        "by_window_frames": summaries, "trials": trials,
    }


def print_gap_validation(validation: dict) -> None:
    for length, summary in validation["by_window_frames"].items():
        def description(metrics: dict) -> str:
            if metrics["rmse"] is None:
                return "mean/RMSE n/a"
            return f"mean {metrics['mean_euclidean_error']:.4f}, RMSE {metrics['rmse']:.4f}"
        print(f"Held-out {length}-frame windows: attempted {summary['attempted_windows']}, "
              f"accepted {summary['accepted_windows']} ({summary['accepted_frames']} frames); "
              f"native px {description(summary['native_pixels'])}; "
              f"court m interpolation {description(summary['interpolated_court_m'])}; "
              f"AB {description(summary['endpoint_ab_court_m'])}")


def write_ground_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "frame", "time_seconds", "stroke_id", "observed_shuttle_x_native_px", "observed_shuttle_y_native_px",
            "inferred_floor_x_native_px", "inferred_floor_y_native_px", "court_x_m", "court_y_m", "status",
            "estimated_shuttle_x_native_px", "estimated_shuttle_y_native_px", "position_source",
        ))
        for row in rows:
            observed = row["observed"]
            floor = row["floor_pixel"]
            court = row["court_xy"]
            estimated = row["estimated"]
            observed_fields = tuple(f"{value:.3f}" for value in observed) if observed is not None else ("", "")
            floor_fields = tuple(f"{value:.3f}" for value in floor) if floor is not None else ("", "")
            court_fields = tuple(f"{value:.6f}" for value in court) if court is not None else ("", "")
            estimated_fields = tuple(f"{value:.3f}" for value in estimated) if estimated is not None else ("", "")
            writer.writerow((
                row["frame"], f"{row['time_seconds']:.6f}", row["stroke"].identifier if row["stroke"] else "",
                *observed_fields, *floor_fields, *court_fields,
                row["status"], *estimated_fields, row["position_source"],
            ))


def trail_edges(rows: list[dict]) -> list[tuple[int, int, np.ndarray, np.ndarray, bool]]:
    edges = []
    previous = None
    for row in rows:
        if row["status"] != "available" or row["stroke"] is None:
            previous = None
            continue
        current = (row["frame"], row["stroke"].index, row["court_xy"], row["position_source"] == "interpolated")
        if previous is not None and current[0] == previous[0] + 1 and current[1] == previous[1]:
            edges.append((current[0], current[1], previous[2], current[2], previous[3] or current[3]))
        previous = current
    return edges


def court_panel_point(xy: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int] | None:
    if not np.isfinite(xy).all():
        return None
    left, top, width, height = rect
    px = left + float(xy[0]) / COURT_WIDTH_M * width
    py = top + float(xy[1]) / COURT_LENGTH_M * height
    if not math.isfinite(px) or not math.isfinite(py) or abs(px) > 1_000_000 or abs(py) > 1_000_000:
        return None
    return int(round(px)), int(round(py))


def stroke_color(index: int) -> tuple[int, int, int]:
    return (70 + (index * 73) % 180, 70 + (index * 131) % 180, 70 + (index * 191) % 180)


def render_frame(source_frame: np.ndarray, row: dict, trail: list, court_lines: list) -> np.ndarray:
    canvas = np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), dtype=np.uint8)
    video_left, video_top, video_width, video_height = 0, 52, 960, 540
    frame = cv2.resize(source_frame, (video_width, video_height), interpolation=cv2.INTER_AREA)
    stroke = row["stroke"]
    if stroke is not None:
        scale = np.asarray([video_width / FRAME_WIDTH, video_height / FRAME_HEIGHT])
        a = tuple(np.rint(stroke.a_native * scale).astype(int))
        b = tuple(np.rint(stroke.b_native * scale).astype(int))
        cv2.line(frame, a, b, (0, 70, 255), 2)
        cv2.circle(frame, a, 6, (0, 70, 255), -1)
        cv2.circle(frame, b, 6, (255, 120, 0), -1)
        cv2.putText(frame, "A", (a[0] + 6, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 70, 255), 2)
        cv2.putText(frame, "B", (b[0] + 6, b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)
    interpolated = row["position_source"] == "interpolated"
    pixel = row["estimated"] if interpolated else row["observed"]
    if pixel is not None:
        point = tuple(np.rint(np.asarray(pixel) * [video_width / FRAME_WIDTH, video_height / FRAME_HEIGHT]).astype(int))
        cv2.circle(frame, point, 7, INTERPOLATED_COLOR if interpolated else (0, 255, 255), -1)
    if row["floor_pixel"] is not None:
        point = row["floor_pixel"] * [video_width / FRAME_WIDTH, video_height / FRAME_HEIGHT]
        if np.isfinite(point).all() and np.max(np.abs(point)) <= 1_000_000:
            cv2.drawMarker(frame, tuple(np.rint(point).astype(int)), INTERPOLATED_COLOR if interpolated else (0, 255, 0), cv2.MARKER_TILTED_CROSS, 18, 3)
    canvas[video_top:video_top + video_height, video_left:video_left + video_width] = frame
    stroke_label = stroke.identifier if stroke is not None else "before first hit"
    cv2.putText(canvas, f"SOURCE  |  frame {row['frame']:03d}/523  {row['time_seconds']:.2f}s  |  {stroke_label}",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
    cv2.putText(canvas, "yellow: observed shuttle   green: inferred floor point   red/blue: stroke A/B", (16, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(canvas, f"projection status: {row['status']}", (16, 652), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if row["status"] == "available" else (180, 180, 180), 1)
    cv2.putText(canvas, "orange: interpolated shuttle, floor point and touching trail", (16, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, INTERPOLATED_COLOR, 1)

    x0 = 960
    cv2.line(canvas, (x0, 0), (x0, OUTPUT_SIZE[1]), (90, 90, 90), 1)
    cv2.putText(canvas, "TOP-DOWN COURT  |  metres", (x0 + 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
    court_height = 570
    court_width = int(round(court_height * COURT_WIDTH_M / COURT_LENGTH_M))
    rect = (x0 + (480 - court_width) // 2, 70, court_width, court_height)
    plot = canvas[52:641, x0:]
    plot_rect = (rect[0] - x0, rect[1] - 52, rect[2], rect[3])
    for a, b in court_lines:
        color = (255, 160, 0) if a[1] == NET_Y_M else (90, 180, 90)
        pa, pb = court_panel_point(np.asarray(a), plot_rect), court_panel_point(np.asarray(b), plot_rect)
        if pa is not None and pb is not None:
            cv2.line(plot, pa, pb, color, 2 if a[1] == NET_Y_M else 1)
    for end_frame, stroke_index, a, b, touches_fill in trail:
        if end_frame > row["frame"]:
            continue
        pa, pb = court_panel_point(a, plot_rect), court_panel_point(b, plot_rect)
        if pa is not None and pb is not None:
            cv2.line(plot, pa, pb, INTERPOLATED_COLOR if touches_fill else stroke_color(stroke_index), 2)
    player_marker_colors = {"A": (0, 70, 255), "B": (255, 120, 0)}
    for identity, xy in (row["player_locations"] or {}).items():
        point = court_panel_point(xy, plot_rect)
        if point is not None:
            cv2.circle(plot, point, 9, (255, 255, 255), -1)
            cv2.circle(plot, point, 6, player_marker_colors[identity], -1)
            label_point = (point[0] + 10, point[1] - 9)
            cv2.putText(plot, identity, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(plot, identity, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        player_marker_colors[identity], 1, cv2.LINE_AA)
    draw_stroke_endpoints(plot, row["endpoints"], plot_rect)
    if row["court_xy"] is not None:
        point = court_panel_point(row["court_xy"], plot_rect)
        if point is not None:
            cv2.circle(plot, point, 7, INTERPOLATED_COLOR if interpolated else (255, 255, 255), -1)
    if row["court_xy"] is not None:
        xy_text = f"x={row['court_xy'][0]:.2f} m  y={row['court_xy'][1]:.2f} m"
    else:
        xy_text = "projected position unavailable"
    cv2.putText(canvas, "A/B: annotated player locations at current hit", (x0 + 16, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.putText(canvas, xy_text, (x0 + 16, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    return canvas


def render_h264_with_audio(video: Path, output: Path, rows: list[dict], fps: float, frame_count: int) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required to create H.264 output and retain source audio")
    source_probe = ffprobe(video)
    source_audio = any(stream.get("codec_type") == "audio" for stream in source_probe.get("streams", []))
    capture = cv2.VideoCapture(str(video))
    court_lines = court_world_lines()
    edges = trail_edges(rows)
    with tempfile.TemporaryDirectory(prefix="court-projection-", dir=output.parent) as temp_name:
        silent = Path(temp_name) / "silent.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", "1440x720", "-framerate", f"{fps:g}", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-frames:v", str(frame_count),
            "-movflags", "+faststart", str(silent),
        ]
        process = None
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video for rendering: {video}")
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for index, row in enumerate(rows):
                ok, source_frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode source frame {index} while rendering")
                process.stdin.write(render_frame(source_frame, row, edges, court_lines).tobytes())
            process.stdin.close()
            error_output = process.stderr.read().decode("utf-8", errors="replace")
            if process.wait():
                raise RuntimeError(f"H.264 encoding failed: {error_output.strip()}")
        finally:
            capture.release()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
        mux = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(video),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy",
            "-t", f"{frame_count / fps:.6f}", "-movflags", "+faststart", str(output),
        ]
        result = subprocess.run(mux, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Could not mux source audio: {result.stderr.strip()}")
    output_probe = ffprobe(output)
    stream = next((item for item in output_probe.get("streams", []) if item.get("codec_type") == "video"), None)
    audio = next((item for item in output_probe.get("streams", []) if item.get("codec_type") == "audio"), None)
    if stream is None:
        raise RuntimeError("Encoded output has no video stream")
    actual_frames = int(stream.get("nb_read_frames", stream.get("nb_frames", -1)))
    rate_n, rate_d = stream["r_frame_rate"].split("/", 1)
    actual_fps = float(rate_n) / float(rate_d)
    duration = float(output_probe["format"]["duration"])
    start_delta = None if audio is None else abs(float(audio.get("start_time", 0)) - float(stream.get("start_time", 0)))
    duration_delta = None if audio is None else abs(float(audio.get("duration", duration)) - float(stream.get("duration", duration)))
    if (stream.get("codec_name") != "h264" or stream.get("width") != OUTPUT_SIZE[0] or stream.get("height") != OUTPUT_SIZE[1]
            or actual_frames != frame_count or abs(actual_fps - fps) > 1e-6
            or abs(duration - frame_count / fps) > 0.02 or (source_audio and audio is None)
            or (audio is not None and (start_delta > 0.04 or (source_audio and duration_delta > 0.04)))):
        raise RuntimeError(f"Encoded output failed frame-rate/audio synchronization checks: {output_probe}")
    return {
        "codec": stream["codec_name"], "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": actual_fps, "frames": actual_frames, "duration_seconds": duration,
        "source_audio_present": source_audio, "output_audio_present": audio is not None,
        "audio_video_start_delta_seconds": start_delta, "audio_video_duration_delta_seconds": duration_delta,
    }


def run(video: Path, rally_csv: Path, homography_csv: Path, tracked_csv: Path, calibration_path: Path,
        output_dir: Path, fill_short_gaps_enabled: bool = False) -> list[Path]:
    video, rally_csv, homography_csv, metadata, hits, geometry = prepare_inputs(video, rally_csv, homography_csv)
    tracked_csv = resolved_repo_path(tracked_csv)
    if not tracked_csv.is_file():
        raise FileNotFoundError(tracked_csv)
    calibration_path = project_output_path(calibration_path)
    output_dir = project_output_path(output_dir)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    calibration = load_calibration(calibration_path, video, metadata, geometry)
    observations = load_tracked_detections(tracked_csv, metadata)
    strokes = prepare_strokes(hits, metadata["frame_count"])
    vp = np.asarray(calibration["homogeneous_vertical_vanishing_point_native"], dtype=np.float64)
    players = player_references(hits, geometry)
    rows = project_observations(
        observations, strokes, vp, geometry, metadata["fps"], players,
    )
    gap_counts = {"enabled": False, "max_gap_frames": MAX_SHORT_GAP_FRAMES,
                  "eligible_gaps": 0, "accepted_gaps": 0, "rejected_gaps": 0, "skipped_gaps": 0}
    validation = None
    if fill_short_gaps_enabled:
        rows, gap_counts = fill_short_gaps(rows, strokes, vp, geometry, metadata["fps"], players)
        validation = validate_short_gaps(observations, strokes, vp, geometry, metadata["fps"], players)
        print_gap_validation(validation)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    outputs = [
        output_dir / f"{stem}_ground_trajectory.csv",
        output_dir / f"{stem}_diagnostics.json",
        output_dir / f"{stem}_court_projection.mp4",
    ]
    created = reserve_output_files(outputs)
    complete = False
    try:
        write_ground_csv(outputs[0], rows)
        output_video = render_h264_with_audio(video, outputs[2], rows, metadata["fps"], metadata["frame_count"])
        status_counts = Counter(row["status"] for row in rows)
        per_stroke = {}
        for stroke in strokes:
            subset = [row for row in rows if row["stroke"] is stroke]
            per_stroke[stroke.identifier] = {
                "start_frame_inclusive": stroke.start_frame,
                "end_frame_exclusive": stroke.end_frame_exclusive,
                "A_annotation": stroke.a_annotation.tolist(),
                "B_annotation": stroke.b_annotation.tolist(),
                "A_source": stroke.a_source,
                "B_source": stroke.b_source,
                "player_locations_court_m": {
                    identity: point.tolist() for identity, point in rows[stroke.start_frame]["player_locations"].items()
                },
                "AB_line_status": stroke.line_status,
                "visible_detections": sum(row["observed"] is not None for row in subset),
                "available_projections": sum(row["status"] == "available" for row in subset),
                "tracked_projections": sum(row["position_source"] == "tracked" for row in subset),
                "interpolated_projections": sum(row["position_source"] == "interpolated" for row in subset),
                "status_counts": dict(Counter(row["status"] for row in subset)),
            }
        diagnostics = {
            "inputs": {
                "video": file_provenance(video), "rally_csv": file_provenance(rally_csv),
                "tracked_detections_csv": file_provenance(tracked_csv),
                "homography_csv": file_provenance(homography_csv), "homography_row_id": geometry.row_id,
                "homography_row_video": geometry.row_video, "video_metadata": metadata,
            },
            "calibration": {
                "file": file_provenance(calibration_path), "selection_method": calibration["selection_method"],
                "post_points": calibration["post_points"],
                "homogeneous_vertical_vanishing_point_native": calibration["homogeneous_vertical_vanishing_point_native"],
                "vertical_vanishing_point_kind": calibration["vertical_vanishing_point_kind"],
            },
            "scaling": {"annotation_to_native_video": ANNOTATION_TO_VIDEO, "video_size": [metadata["width"], metadata["height"]]},
            "frame_timing": {"fps": metadata["fps"], "time_seconds": "frame / fps"},
            "projection_limits": {
                "court_margin_m": COURT_MARGIN_M, "max_one_pixel_shift_m": MAX_ONE_PIXEL_SHIFT_M,
            },
            "availability": {
                "frames": len(rows), "available": status_counts.get("available", 0),
                "tracked_projections": sum(row["position_source"] == "tracked" for row in rows),
                "interpolated_projections": sum(row["position_source"] == "interpolated" for row in rows),
                "status_counts": dict(status_counts), "strokes": per_stroke,
            },
            "short_gap_filling": gap_counts,
            "held_out_validation": validation,
            "approximation_warnings": [
                "Hit/landing annotations are approximate court footprints; only the first serve uses the hitter's player location as its starting footprint.",
                "Each shuttle footprint assumes the shuttle lies in the vertical plane through that stroke's fixed AB floor line.",
                "The final net-fault flight uses one approximate vertical plane from the last hit_x/y to landing_x/y and assumes the shuttle drops roughly beneath its net-contact position.",
                "No landing time is estimated for the final stroke. Optional interpolation fills only short missing runs within a stroke; unavailable frames and stroke boundaries break trails.",
                "Held-out TrackNet detections and projected positions are reference measurements, not independent ground truth for court footprints.",
            ],
            "output_video": output_video,
        }
        with outputs[1].open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(diagnostics, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        complete = True
    finally:
        if not complete:
            for path in created:
                path.unlink(missing_ok=True)
    print(f"Ground trajectory: {outputs[0]}")
    print(f"Diagnostics: {outputs[1]}")
    print(f"Projection video: {outputs[2]}")
    print(f"Available frames: {status_counts.get('available', 0)} / {len(rows)}; status counts: {dict(status_counts)}")
    print(f"Tracked projections: {diagnostics['availability']['tracked_projections']}; "
          f"interpolated projections: {diagnostics['availability']['interpolated_projections']}; "
          f"short gaps: {gap_counts}")
    print(f"Video checks: {output_video}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
        sub.add_argument("--rally-csv", type=Path, default=DEFAULT_RALLY_CSV)
        sub.add_argument("--homography-csv", type=Path, default=DEFAULT_HOMOGRAPHY)
        sub.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        if command == "run":
            sub.add_argument("--fill-short-gaps", action="store_true",
                             help="Fill up to five missing frames within a stroke and run held-out validation")
            sub.add_argument("--output-dir", type=Path, default=HERE)
            sub.add_argument("--tracked-csv", type=Path, default=DEFAULT_TRACKED_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        video, rally_csv, homography_csv, metadata, hits, geometry = prepare_inputs(
            args.video, args.rally_csv, args.homography_csv
        )
        calibration_path = project_output_path(args.calibration)
        if args.command == "calibrate":
            if calibration_path.exists():
                raise FileExistsError(f"Refusing to overwrite calibration: {calibration_path}")
            print(f"Verified {len(hits)} hit rows and {metadata['frame_count']} frames at {metadata['fps']:g} fps.")
            if not calibrate(video, rally_csv, homography_csv, calibration_path, metadata, geometry):
                print("Calibration cancelled; no file written.")
                return 1
            print(f"Calibration saved: {calibration_path}")
        else:
            output_dir = project_output_path(args.output_dir)
            run(video, rally_csv, homography_csv, args.tracked_csv, calibration_path, output_dir, args.fill_short_gaps)
    except Exception as error:
        print(f"court_projection: failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
