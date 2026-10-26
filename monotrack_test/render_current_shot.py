"""Keep the rally video and show only the current shot on its top-down court panel."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np

from run_projection import (
    HERE,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    court_world_lines,
    ffprobe,
    load_annotations,
    load_court_geometry,
    load_player_positions,
    panel_point,
    repo_path,
    reserve_output_files,
    shutil_which,
)


BATCH = HERE / "ten_clips_001"
SIZE = (1440, 720)
PANEL_WIDTH = SIZE[0] // 2
COURT_HEIGHT = 520
COURT_WIDTH = round(COURT_HEIGHT * COURT_WIDTH_M / COURT_LENGTH_M)
COURT_RECT = ((PANEL_WIDTH - COURT_WIDTH) // 2, 90, COURT_WIDTH, COURT_HEIGHT)
TRAIL_COLOR = (0, 220, 220)


def load_coordinates(path: Path, frame_count: int, fps: float) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", "time_seconds", "x_m", "y_m"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing trajectory columns: {path}")
        rows = list(reader)
    if len(rows) != frame_count:
        raise ValueError(f"Trajectory frame count disagrees with diagnostics: {path}")
    coordinates = np.full((frame_count, 2), np.nan)
    for frame, row in enumerate(rows):
        seconds = float(row["time_seconds"])
        if (int(row["frame"]) != frame or not math.isfinite(seconds)
                or abs(seconds - frame / fps) > 1e-5):
            raise ValueError(f"Invalid trajectory frame/timing at frame {frame}: {path}")
        values = [row["x_m"], row["y_m"]]
        if values == ["", ""]:
            continue
        point = np.asarray(values, dtype=float)
        if not np.isfinite(point).all():
            raise ValueError(f"Invalid trajectory coordinates at frame {frame}: {path}")
        coordinates[frame] = point
    return coordinates


def load_saved_projection(path: Path) -> tuple[Path, dict, list[int], np.ndarray, list[dict]]:
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    metadata = saved["input_metadata"]
    count, fps = metadata["frame_count"], float(metadata["fps"])
    if not isinstance(count, int) or count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid saved video metadata: {path}")
    video = repo_path(saved["input_video"])
    hits = load_annotations(repo_path(saved["rally_csv"]), count, fps, require_player_positions=False)
    geometry = load_court_geometry(repo_path(saved["homography_csv"]), video.parent.parent.name)
    source = saved["player_positions"]
    if source.get("source") != "keypoint_smoothed":
        raise ValueError(f"Expected saved smoothed player tracks: {path}")
    tracks, provenance = load_player_positions(source["csv_path"], video, metadata, hits, geometry)
    if provenance["csv_sha256"] != source["csv_sha256"]:
        raise ValueError(f"Saved player CSV changed since projection: {path}")
    stem = path.name.removesuffix("_diagnostics.json")
    projection_video = path.with_name(f"{stem}_projection.mp4")
    if not projection_video.is_file():
        raise FileNotFoundError(projection_video)
    coordinates = load_coordinates(path.with_name(f"{stem}_trajectory_3d.csv"), count, fps)
    return projection_video, metadata, [hit["clip_frame_num"] for hit in hits], coordinates, tracks


def render_frame(source_frame: np.ndarray, frame: int, fps: float, hit_frames: list[int],
                 coordinates: np.ndarray, players: dict) -> np.ndarray:
    if source_frame.shape != (SIZE[1], SIZE[0], 3):
        raise ValueError(f"Expected a {SIZE[0]}x{SIZE[1]} saved projection frame")
    canvas = np.zeros((SIZE[1], PANEL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(canvas, "TOP-DOWN COURT | CURRENT SHOT", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    for a, b in court_world_lines():
        net = a[1] == NET_Y_M
        cv2.line(canvas, panel_point(a, COURT_RECT), panel_point(b, COURT_RECT),
                 (255, 180, 0) if net else (100, 180, 100), 2 if net else 1)
    shot = bisect_right(hit_frames, frame) - 1
    if shot >= 0:
        points = coordinates[hit_frames[shot]:frame + 1]
        for a, b in zip(points, points[1:]):
            if np.isfinite(a).all() and np.isfinite(b).all():
                cv2.line(canvas, panel_point(a, COURT_RECT), panel_point(b, COURT_RECT), TRAIL_COLOR, 2)
        current = coordinates[frame]
        if np.isfinite(current).all():
            cv2.circle(canvas, panel_point(current, COURT_RECT), 8, (255, 255, 255), 2)
    for player, color in (("A", (30, 80, 255)), ("B", (255, 120, 0))):
        point = players[player]
        if np.isfinite(point).all():
            pixel = panel_point(point, COURT_RECT)
            cv2.circle(canvas, pixel, 9, color, -1)
            cv2.putText(canvas, player, (pixel[0] + 12, pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    status = f"Shot {shot + 1}" if shot >= 0 else "Before first hit"
    cv2.putText(canvas, f"{status} | frame {frame}/{len(coordinates) - 1} | {frame / fps:.2f}s",
                (18, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "Yellow: current shot | White: shuttle | A/B: players", (18, 695),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
    composed = source_frame.copy()
    composed[:, PANEL_WIDTH:] = canvas
    return composed


def write_video(output: Path, video: Path, metadata: dict, hit_frames: list[int],
                coordinates: np.ndarray, tracks: list[dict]) -> None:
    reserve_output_files([output])
    complete = False
    capture = None
    try:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open saved projection: {video}")
        if (int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(coordinates)
                or abs(capture.get(cv2.CAP_PROP_FPS) - metadata["fps"]) > 1e-6):
            raise ValueError(f"Saved projection frame count/FPS disagrees with diagnostics: {video}")
        command = [
            shutil_which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{SIZE[0]}x{SIZE[1]}",
            "-framerate", str(metadata["fps"]), "-i", "pipe:0", "-i", str(video),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-frames:v", str(len(coordinates)), "-t", f"{len(coordinates) / metadata['fps']:.6f}",
            "-movflags", "+faststart", str(output),
        ]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                try:
                    for frame in range(len(coordinates)):
                        ok, source_frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"Could not decode saved projection frame {frame}: {video}")
                        image = render_frame(source_frame, frame, metadata["fps"], hit_frames, coordinates, tracks[frame])
                        process.stdin.write(image.tobytes())
                except BrokenPipeError:
                    pass
                finally:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if process.wait():
                    errors.seek(0)
                    raise RuntimeError(f"Video encoding failed: {errors.read().decode(errors='replace')}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        probe = ffprobe(output)
        stream = next(item for item in probe["streams"] if item["codec_type"] == "video")
        numerator, denominator = map(float, stream["r_frame_rate"].split("/"))
        if (stream["codec_name"] != "h264" or (stream["width"], stream["height"]) != SIZE
                or int(stream["nb_read_frames"]) != len(coordinates)
                or abs(numerator / denominator - metadata["fps"]) > 1e-6):
            raise RuntimeError(f"Encoded video metadata mismatch: {output}")
        complete = True
    finally:
        if capture is not None:
            capture.release()
        if not complete:
            output.unlink(missing_ok=True)


def main() -> int:
    try:
        if not shutil_which("ffmpeg") or not shutil_which("ffprobe"):
            raise RuntimeError("ffmpeg and ffprobe are required")
        diagnostics = sorted(BATCH.glob("*/*_diagnostics.json"))
        if not diagnostics:
            raise FileNotFoundError(f"No saved projections found in {BATCH}")
        jobs = []
        for path in diagnostics:
            stem = path.name.removesuffix("_diagnostics.json")
            output = path.with_name(f"{stem}_projection_current_shot.mp4")
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite output: {output}")
            jobs.append((output, load_saved_projection(path)))
        for output, data in jobs:
            write_video(output, *data)
            print(f"Saved: {output}")
    except Exception as error:
        print(f"render_current_shot: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
