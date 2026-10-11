"""Reuse annotation-led raw rally clips and trim only outside their hit scene."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = Path(
    r"G:\My Drive\Match_Downloads\F ｜ MS ｜ Kento MOMOTA (JPN) [1] vs. CHOU Tien Chen (TPE) [2] ｜ BWF 2019.mp4"
)
DEFAULT_ANNOTATIONS = HERE.parent / "set-1" / "Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals"
DEFAULT_RAW_ROOT = HERE.parent / "trimmed_rallies"
DEFAULT_CLEANED_ROOT = HERE.parent / "cleaned_rallies"
SCENEDETECT_VERSION = "0.7.1"
SHORT_ENDING_TAIL_SECONDS = 1.0
SUMMARY_FIELDS = (
    "rally_id", "start_seconds", "end_seconds", "filename", "dataset", "match",
    "game", "original_rally_id", "source_csv", "first_hit_frame", "last_hit_frame",
    "annotation_fps", "time_offset", "export_start_seconds", "export_end_seconds",
    "review_flags",
)
LABEL_FIELDS = ("source_time_seconds", "clip_time_seconds", "clip_frame_num", "clip_timestamp")
REPORT_FIELDS = (
    "dataset", "match", "game", "original_rally_id", "raw_path", "cleaned_path",
    "scene_boundaries_zero_based_end_exclusive", "removed_leading_frames",
    "removed_leading_seconds", "removed_trailing_frames", "removed_trailing_seconds",
    "outcome", "review_flags",
)
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]| +(?=[A-Za-z0-9._-]))*$")


def executable(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} was not found on PATH")
    return path


def run_json(command, description):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{description}: {result.stderr.strip() or result.returncode}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description}: FFprobe returned invalid JSON") from exc


def probe_video(video):
    info = run_json(
        [executable("ffprobe"), "-v", "error", "-show_entries",
         "format=duration,start_time:stream=codec_type,width,height,avg_frame_rate,r_frame_rate,start_time",
         "-of", "json", str(video)],
        f"FFprobe could not read {video}",
    )
    try:
        stream = next(item for item in info["streams"] if item.get("codec_type") == "video")
        rate_text = stream.get("avg_frame_rate") or stream["r_frame_rate"]
        rate = Fraction(rate_text)
        avg_rate = Fraction(stream.get("avg_frame_rate", rate_text))
        nominal_rate = Fraction(stream.get("r_frame_rate", rate_text))
        video_start = float(stream.get("start_time", info["format"].get("start_time", 0)) or 0)
        duration = float(info["format"]["duration"])
        if rate <= 0 or duration <= 0:
            raise ValueError("nonpositive duration or frame rate")
        return {
            "width": int(stream["width"]), "height": int(stream["height"]),
            "duration": duration, "rate": rate, "avg_rate": avg_rate,
            "nominal_rate": nominal_rate, "video_start_time": video_start,
            "has_audio": any(item.get("codec_type") == "audio" for item in info["streams"]),
        }
    except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"FFprobe returned incomplete source metadata: {exc}") from exc


def frame_timestamps(video, start=None, duration=None):
    command = [executable("ffprobe"), "-v", "error", "-select_streams", "v:0"]
    if start is not None:
        command += ["-read_intervals", f"{start:.6f}%+{duration:.6f}"]
    command += ["-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)]
    info = run_json(command, f"FFprobe frame read failed for {video}")
    try:
        return [float(frame["best_effort_timestamp_time"]) for frame in info.get("frames", [])
                if frame.get("best_effort_timestamp_time") is not None]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"FFprobe returned invalid frame timestamps for {video}: {exc}") from exc


def verify_cfr(timestamps, rate, description):
    if len(timestamps) < 2:
        raise RuntimeError(f"cannot verify constant frame rate for {description}: fewer than two frames")
    period = 1.0 / float(rate)
    tolerance = max(0.0001, period * 0.002)
    if any(abs(later - earlier - period) > tolerance for earlier, later in zip(timestamps, timestamps[1:])):
        raise RuntimeError(f"{description} is not constant-frame-rate at {float(rate):g} fps")


def source_interval_origin(video, metadata, start, end):
    if metadata["avg_rate"] != metadata["nominal_rate"]:
        raise RuntimeError("annotation labels require a constant-frame-rate source")
    absolute_start = metadata["video_start_time"] + start
    tolerance = 0.0001
    period = 1.0 / float(metadata["rate"])
    for extra in (0.2, 5.2, 10.2, 20.2, 40.2):
        query_duration = end - start + extra
        if query_duration > 60.0:
            continue
        timestamps = frame_timestamps(video, absolute_start, query_duration)
        relative = [value - metadata["video_start_time"] for value in timestamps]
        inside = [value for value in relative if start - tolerance <= value < end - tolerance]
        if inside and inside[-1] >= end - period - tolerance:
            break
    else:
        raise RuntimeError(f"could not verify source frames in {start:.3f}-{end:.3f}s")
    if not inside:
        raise RuntimeError(f"FFprobe found no source frames in {start:.6f}-{end:.6f}s")
    if inside[0] - start > period + tolerance:
        raise RuntimeError("first source frame is more than one frame after the requested start")
    verify_cfr(inside, metadata["rate"], f"source interval {start:.3f}-{end:.3f}s")
    return inside[0], len(inside)


def probe_clip(path, source_metadata):
    info = run_json(
        [executable("ffprobe"), "-v", "error", "-count_frames", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,start_time,duration,nb_read_frames",
         "-of", "json", str(path)],
        f"FFprobe could not verify {path}",
    )
    try:
        video = next(stream for stream in info["streams"] if stream.get("codec_type") == "video")
        audio = [stream for stream in info["streams"] if stream.get("codec_type") == "audio"]
        fps, nominal_fps = Fraction(video["avg_frame_rate"]), Fraction(video["r_frame_rate"])
        count = int(video["nb_read_frames"])
        duration = float(info["format"]["duration"])
        video_start = float(video.get("start_time", 0) or 0)
        timestamps = frame_timestamps(path)
        if count != len(timestamps) or not timestamps:
            raise RuntimeError(f"FFprobe frame-count disagreement for {path}: {count} vs {len(timestamps)}")
        if abs(timestamps[0] - video_start) > 0.001:
            raise RuntimeError(f"first output frame timestamp disagrees with stream origin: {path}")
        verify_cfr([value - timestamps[0] for value in timestamps], fps, str(path))
        if fps != source_metadata["rate"] or nominal_fps != fps:
            raise RuntimeError(f"frame rate changed or is inconsistent in {path}: {fps}/{nominal_fps}")
        if (int(video["width"]) != source_metadata["width"] or
                int(video["height"]) != source_metadata["height"]):
            raise RuntimeError(f"resolution changed in {path}")
        if video.get("codec_name") != "h264":
            raise RuntimeError(f"expected H.264 video in {path}, found {video.get('codec_name')}")
        if video_start > 0.001:
            raise RuntimeError(f"output video does not start at local time zero: {path}")
        if source_metadata["has_audio"]:
            aac = next((stream for stream in audio if stream.get("codec_name") == "aac"), None)
            if aac is None:
                raise RuntimeError(f"output is missing AAC audio: {path}")
            audio_start = float(aac.get("start_time", 0) or 0)
            audio_duration = float(aac.get("duration", duration))
            tolerance = max(0.08, 1.0 / float(fps))
            if abs(audio_start - video_start) > tolerance or abs(audio_duration - duration) > tolerance:
                raise RuntimeError(f"audio/video stream origins or durations disagree in {path}")
        return {"fps": fps, "frames": count, "duration": duration, "start": timestamps[0],
                "has_audio": bool(audio), "audio_codec": audio[0].get("codec_name") if audio else "none"}
    except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"FFprobe returned incomplete clip metadata for {path}: {exc}") from exc


def validate_component(value, label):
    if not SAFE_COMPONENT.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"{label} must be a safe single path component: {value!r}")
    return value


def parse_frame(value, path, row_number):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} row {row_number}: frame_num must be a nonnegative integer") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{path} row {row_number}: frame_num must be a nonnegative integer")
    return int(numeric)


def parse_clip_frame(value, identity):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{identity}: clip_frame_num is not an integer") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{identity}: clip_frame_num is not a nonnegative integer")
    return int(numeric)


def parse_coarse_time(value):
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3) or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid whole-second annotation time {value!r}")
    values = [int(part) for part in parts]
    if values[-1] >= 60 or (len(values) == 3 and values[-2] >= 60):
        raise ValueError(f"invalid whole-second annotation time {value!r}")
    return values[0] * 60 + values[1] if len(values) == 2 else values[0] * 3600 + values[1] * 60 + values[2]


def timestamp(seconds):
    whole, millis = divmod(round(seconds * 1000), 1000)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def canonical_filename(record):
    return f"set{int(record['game'])}/rally_{int(record['original_rally_id']):03d}.mp4"


def add_flag(record, flag):
    flags = [item for item in str(record.get("review_flags", "")).split(";") if item]
    if flag not in flags:
        flags.append(flag)
    record["review_flags"] = ";".join(flags)


def read_annotations(folder, annotation_fps, time_offset, pre_padding, post_padding, video_duration):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"annotation folder does not exist: {folder}")
    if not math.isfinite(annotation_fps) or annotation_fps <= 0:
        raise ValueError("--annotation-fps must be finite and greater than zero")
    if not math.isfinite(time_offset) or min(pre_padding, post_padding) < 0:
        raise ValueError("offset must be finite and padding must be nonnegative")
    if not all(math.isfinite(value) for value in (pre_padding, post_padding)):
        raise ValueError("padding must be finite")
    dataset = validate_component(folder.parent.name, "dataset")
    match = validate_component(folder.name, "match")
    csv_files = []
    for path in folder.glob("set*.csv"):
        found = re.fullmatch(r"set([1-9]\d*)\.csv", path.name, re.IGNORECASE)
        if found:
            csv_files.append((int(found.group(1)), path))
    if not csv_files:
        raise ValueError(f"no set<N>.csv files found in {folder}")

    groups = {}
    for game, path in sorted(csv_files):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"rally", "time", "frame_num"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
            for row_number, row in enumerate(reader, 2):
                rally_id = str(row.get("rally", "")).strip()
                if not rally_id.isdigit() or int(rally_id) < 1:
                    raise ValueError(f"{path} row {row_number}: rally must be a positive original rally ID")
                frame = parse_frame(row.get("frame_num"), path, row_number)
                coarse_seconds = parse_coarse_time(row.get("time", ""))
                identity = (dataset, match, str(game), rally_id)
                group = groups.setdefault(identity, {
                    "dataset": dataset, "match": match, "game": str(game), "original_rally_id": rally_id,
                    "source_csv": path.name, "rows": [], "frames": [], "coarse": [],
                })
                group["rows"].append(dict(row))
                group["frames"].append(frame)
                group["coarse"].append((row_number, coarse_seconds, frame))
    if not groups:
        raise ValueError(f"annotation files contain no rally hits: {folder}")

    records = []
    for identity, group in groups.items():
        hit_times = [frame / annotation_fps + time_offset for frame in group["frames"]]
        if any(not math.isfinite(value) or value < 0 or value > video_duration for value in hit_times):
            raise ValueError(f"annotation hits for {group['source_csv']} rally {group['original_rally_id']} are outside the source video")
        first_hit, last_hit = min(hit_times), max(hit_times)
        start, end = max(0.0, first_hit - pre_padding), min(video_duration, last_hit + post_padding)
        if end <= start:
            raise ValueError(f"empty padded interval for game {group['game']} rally {group['original_rally_id']}")
        record = {
            **{key: group[key] for key in ("dataset", "match", "game", "original_rally_id", "source_csv")},
            "rally_id": group["original_rally_id"], "start_seconds": start, "end_seconds": end,
            "filename": "", "first_hit_frame": min(group["frames"]), "last_hit_frame": max(group["frames"]),
            "annotation_fps": annotation_fps, "time_offset": time_offset,
            "export_start_seconds": start, "export_end_seconds": end, "review_flags": "",
            "_rows": group["rows"], "_frames": group["frames"], "_identity": identity,
            "_first_hit_time": first_hit, "_last_hit_time": last_hit,
        }
        if start == 0 and first_hit - pre_padding < 0:
            add_flag(record, "start-clamped-to-video")
        if end == video_duration and last_hit + post_padding > video_duration:
            add_flag(record, "end-clamped-to-video")
        for row_number, coarse_seconds, frame in group["coarse"]:
            discrepancy = frame / annotation_fps + time_offset - coarse_seconds
            if abs(discrepancy) > 1.0:
                add_flag(record, "coarse-time-discrepancy")
                print(f"Warning: {group['source_csv']} row {row_number} rally {group['original_rally_id']} coarse time differs by {discrepancy:+.3f}s", file=sys.stderr)
        record["filename"] = canonical_filename(record)
        records.append(record)

    by_game = {}
    for record in records:
        by_game.setdefault((record["dataset"], record["match"], record["game"]), []).append(record)
    for game_records in by_game.values():
        for index, first in enumerate(game_records):
            for second in game_records[index + 1:]:
                if first["start_seconds"] < second["end_seconds"] and second["start_seconds"] < first["end_seconds"]:
                    add_flag(first, "padding-overlap")
                    add_flag(second, "padding-overlap")
                if first["_first_hit_time"] <= second["_last_hit_time"] and second["_first_hit_time"] <= first["_last_hit_time"]:
                    add_flag(first, "hit-span-overlap")
                    add_flag(second, "hit-span-overlap")
    return sorted(records, key=lambda item: (item["start_seconds"], int(item["game"]), int(item["original_rally_id"])))


def select_records(records, start, end):
    if (start is None) != (end is None):
        raise ValueError("--start and --end must be supplied together")
    if start is None:
        return records
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("bounded source times must be finite, nonnegative, and start before end")
    selected = [record for record in records
                if record["_first_hit_time"] < end and record["_last_hit_time"] >= start]
    if not selected:
        raise ValueError(f"no annotation hit spans overlap {start:.3f}-{end:.3f}s")
    print(f"Selected {len(selected)} rally/rallies with hit spans overlapping {start:.3f}-{end:.3f}s; full padding is retained")
    return selected


def record_path(root, record):
    return Path(root) / record["dataset"] / record["match"] / Path(record["filename"])


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return reader.fieldnames, list(reader)


def identity_of(row):
    return (str(row.get("dataset", "")), str(row.get("match", "")),
            str(row.get("game", "")), str(row.get("original_rally_id", "")))


def summary_map(path):
    if not Path(path).is_file():
        return {}
    _, rows = read_csv_rows(path)
    result = {}
    for row in rows:
        identity = identity_of(row)
        if identity in result:
            raise ValueError(f"duplicate rally identity in {path}: {identity}")
        result[identity] = row
    return result


def row_number(value, label):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} is not an integer")
    return int(numeric)


def check_summary(record, saved):
    if saved is None:
        raise ValueError("match-level rallies.csv is missing this identity")
    for field in ("dataset", "match", "game", "original_rally_id", "source_csv"):
        if str(saved.get(field, "")) != str(record[field]):
            raise ValueError(f"rallies.csv {field} mismatch: {saved.get(field)!r} != {record[field]!r}")
    if str(saved.get("filename", "")) != record["filename"]:
        raise ValueError("rallies.csv filename mismatch")
    for field in ("start_seconds", "end_seconds", "export_start_seconds", "export_end_seconds"):
        try:
            value = float(saved[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"rallies.csv has invalid {field}") from exc
        if not math.isclose(value, record[field], rel_tol=0, abs_tol=0.000002):
            raise ValueError(f"rallies.csv {field} mismatch: {value:.6f} != {record[field]:.6f}")


def verify_raw(record, raw_path, summary_rows, video, source_metadata):
    identity = f"game {record['game']} rally {record['original_rally_id']}"
    companion = raw_path.with_suffix(".csv")
    if not raw_path.is_file() or not companion.is_file():
        raise ValueError(f"{identity}: raw clip/companion CSV is missing")
    check_summary(record, summary_rows.get(record["_identity"]))
    raw_metadata = probe_clip(raw_path, source_metadata)
    origin, source_frames = source_interval_origin(
        video, source_metadata, record["start_seconds"], record["end_seconds"]
    )
    if raw_metadata["frames"] != source_frames:
        raise ValueError(f"{identity}: raw clip has {raw_metadata['frames']} frames; source interval has {source_frames}")

    fields, rows = read_csv_rows(companion)
    original_fields = list(record["_rows"][0])
    missing = [field for field in original_fields + list(LABEL_FIELDS) if field not in fields]
    if missing:
        raise ValueError(f"{identity}: raw companion CSV lacks columns {', '.join(missing)}")
    if len(rows) != len(record["_rows"]):
        raise ValueError(f"{identity}: raw companion CSV has {len(rows)} annotation rows; source has {len(record['_rows'])}")
    fps = float(raw_metadata["fps"])
    for number, (source_row, raw_row) in enumerate(zip(record["_rows"], rows), 2):
        for field, value in source_row.items():
            if raw_row.get(field, "") != value:
                raise ValueError(f"{identity} row {number}: original annotation column {field!r} changed")
        frame = parse_clip_frame(raw_row.get("clip_frame_num"), identity)
        if frame >= raw_metadata["frames"]:
            raise ValueError(f"{identity} row {number}: clip_frame_num is outside the raw video")
        source_time = parse_frame(source_row["frame_num"], companion, number) / record["annotation_fps"] + record["time_offset"]
        try:
            saved_source_time = float(raw_row["source_time_seconds"])
            clip_seconds = float(raw_row["clip_time_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{identity} row {number}: invalid companion timestamps") from exc
        if not math.isclose(saved_source_time, source_time, rel_tol=0, abs_tol=0.000002):
            raise ValueError(f"{identity} row {number}: source_time_seconds mismatch")
        mapped = math.floor((source_time - origin) * fps + 0.5)
        if frame != mapped:
            raise ValueError(f"{identity} row {number}: clip_frame_num {frame} does not match PTS-derived frame {mapped}")
        if not math.isclose(clip_seconds, frame / fps, rel_tol=0, abs_tol=0.000002):
            raise ValueError(f"{identity} row {number}: clip_time_seconds mismatch")
        if raw_row.get("clip_timestamp") != timestamp(frame / fps):
            raise ValueError(f"{identity} row {number}: clip_timestamp mismatch")
    return {"path": raw_path, "csv": companion, "metadata": raw_metadata,
            "source_origin": origin, "rows": rows, "fields": fields}


def write_annotation_csv(path, record, raw_rows, raw_fields, leading_frames, output_fps, output_frames):
    fields = list(raw_fields) + [field for field in LABEL_FIELDS if field not in raw_fields]
    prepared = []
    for index, row in enumerate(raw_rows, 2):
        raw_frame = parse_clip_frame(row.get("clip_frame_num"), record["filename"])
        cleaned_frame = raw_frame - leading_frames
        if not 0 <= cleaned_frame < output_frames:
            raise ValueError(f"{record['filename']} row {index}: hit maps outside cleaned video")
        labeled = dict(row)
        labeled["clip_frame_num"] = str(cleaned_frame)
        seconds = cleaned_frame / output_fps
        labeled["clip_time_seconds"] = f"{seconds:.6f}"
        labeled["clip_timestamp"] = timestamp(seconds)
        source_time = float(labeled["source_time_seconds"])
        mapped = math.floor((source_time - record["_clean_source_origin"]) * output_fps + 0.5)
        if mapped != cleaned_frame:
            raise ValueError(f"{record['filename']} row {index}: PTS-derived source mapping {mapped} differs from cleaned label {cleaned_frame}")
        prepared.append(labeled)
    write_csv(path, fields, prepared, overwrite=True)


def write_csv(path, fields, rows, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def summary_row(record, start=None, end=None, flags=None):
    start = record["start_seconds"] if start is None else start
    end = record["end_seconds"] if end is None else end
    row = {field: record.get(field, "") for field in SUMMARY_FIELDS}
    row.update({
        "rally_id": record["original_rally_id"], "start_seconds": f"{start:.6f}",
        "end_seconds": f"{end:.6f}", "filename": record["filename"],
        "export_start_seconds": f"{start:.6f}", "export_end_seconds": f"{end:.6f}",
        "annotation_fps": f"{record['annotation_fps']:.9g}",
        "time_offset": f"{record['time_offset']:.9g}",
        "review_flags": record.get("review_flags", "") if flags is None else flags,
    })
    return row


def merge_rows(existing_path, new_rows):
    existing = {}
    if Path(existing_path).is_file():
        _, rows = read_csv_rows(existing_path)
        existing = {identity_of(row): row for row in rows}
    existing.update({identity_of(row): row for row in new_rows})
    return sorted(existing.values(), key=lambda row: (row["dataset"], row["match"], int(row["game"]), int(row["original_rally_id"])))


def write_summary(folder, records, overwrite):
    path = Path(folder) / "rallies.csv"
    rows = [summary_row(record) for record in records]
    if path.exists() and overwrite:
        rows = merge_rows(path, rows)
    write_csv(path, SUMMARY_FIELDS, rows, overwrite=overwrite)


def export_raw(record, video, destination, source_metadata, overwrite=False):
    start, end = record["start_seconds"], record["end_seconds"]
    label_path = destination.with_suffix(".csv")
    if not overwrite and (destination.exists() or label_path.exists()):
        raise FileExistsError(f"refusing to overwrite raw output for {destination}; pass --overwrite")
    origin, source_frames = source_interval_origin(video, source_metadata, start, end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp.mp4", dir=destination.parent)
    os.close(handle)
    temp = Path(temp_name)
    handle, label_temp_name = tempfile.mkstemp(prefix=f".{label_path.name}.", suffix=".tmp.csv", dir=destination.parent)
    os.close(handle)
    label_temp = Path(label_temp_name)
    command = [executable("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{start:.6f}", "-i", str(video), "-t", f"{end-start:.6f}",
               "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(source_metadata["rate"]),
               "-fps_mode", "cfr", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temp)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"FFmpeg raw export failed for {destination.name}: {result.stderr.strip() or result.returncode}")
        clip_metadata = probe_clip(temp, source_metadata)
        if clip_metadata["frames"] != source_frames:
            raise RuntimeError(f"raw export frame count {clip_metadata['frames']} != PTS-verified source count {source_frames}")
        record["_raw_origin"] = origin
        label_rows = []
        for row in record["_rows"]:
            frame = parse_frame(row["frame_num"], destination, 0)
            source_time = frame / record["annotation_fps"] + record["time_offset"]
            clip_frame = math.floor((source_time - origin) * float(clip_metadata["fps"]) + 0.5)
            if not 0 <= clip_frame < clip_metadata["frames"]:
                raise ValueError(f"annotation hit maps outside exported raw clip {destination}")
            seconds = clip_frame / float(clip_metadata["fps"])
            label_rows.append({**row, "source_time_seconds": f"{source_time:.6f}",
                               "clip_time_seconds": f"{seconds:.6f}", "clip_frame_num": str(clip_frame),
                               "clip_timestamp": timestamp(seconds)})
        fields = list(record["_rows"][0]) + [field for field in LABEL_FIELDS if field not in record["_rows"][0]]
        label_temp.unlink()
        write_csv(label_temp, fields, label_rows)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite raw clip {destination}")
        if label_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite raw labels {label_path}")
        os.replace(temp, destination)
        os.replace(label_temp, label_path)
    except BaseException:
        temp.unlink(missing_ok=True)
        label_temp.unlink(missing_ok=True)
        raise
    return origin, clip_metadata


def load_scene_detector():
    try:
        import scenedetect
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector
    except ImportError as exc:
        raise RuntimeError("PySceneDetect is required; install trim_rallies/requirements-v2.txt") from exc
    version = getattr(scenedetect, "__version__", "")
    if version != SCENEDETECT_VERSION:
        raise RuntimeError(f"expected scenedetect=={SCENEDETECT_VERSION}, found {version or 'unknown'}")
    return SceneManager, open_video, AdaptiveDetector


def detect_scenes(path, frame_count):
    SceneManager, open_video, AdaptiveDetector = load_scene_detector()
    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector())
    try:
        manager.detect_scenes(video=video)
        found = [(scene[0].frame_num, scene[1].frame_num) for scene in manager.get_scene_list(start_in_scene=True)]
    finally:
        video.capture.release()
    found = [(start, min(end, frame_count)) for start, end in found if start < frame_count and end > 0]
    if not found:
        return [(0, frame_count)]
    normalized = []
    cursor = 0
    for start, end in found:
        if start > cursor:
            normalized.append((cursor, start))
        if start < cursor:
            start = cursor
        if end > start:
            normalized.append((start, end))
            cursor = end
    if cursor < frame_count:
        normalized.append((cursor, frame_count))
    if not normalized:
        return [(0, frame_count)]
    if normalized[0][0] != 0 or normalized[-1][1] != frame_count or any(a[1] != b[0] for a, b in zip(normalized, normalized[1:])):
        raise RuntimeError(f"scene detector returned incomplete frame coverage for {path}")
    return normalized


def containing_scene(scenes, hit_frames, frame_count):
    hits = [parse_clip_frame(value, "annotation") for value in hit_frames]
    if not hits or any(frame >= frame_count for frame in hits):
        raise ValueError("annotation frame mapping is outside the raw clip")
    matches = [scene for scene in scenes if all(scene[0] <= frame < scene[1] for frame in hits)]
    return matches[0] if len(matches) == 1 else None


def adjusted_labels(raw_rows, leading_frames, fps, frame_count):
    result = []
    for row in raw_rows:
        raw_frame = parse_clip_frame(row.get("clip_frame_num"), "annotation")
        cleaned_frame = raw_frame - leading_frames
        if not 0 <= cleaned_frame < frame_count:
            raise ValueError("adjusted annotation frame is outside the cleaned clip")
        result.append((cleaned_frame, cleaned_frame / fps, timestamp(cleaned_frame / fps)))
    return result


def copy_atomic(source, destination, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=destination.suffix, dir=destination.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        shutil.copy2(source, temp)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
        os.replace(temp, destination)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def export_clean_interval(video, destination, start, end, source_metadata, overwrite=False):
    origin, source_frames = source_interval_origin(video, source_metadata, start, end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp.mp4", dir=destination.parent)
    os.close(handle)
    temp = Path(temp_name)
    command = [executable("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{start:.6f}", "-i", str(video), "-t", f"{end-start:.6f}",
               "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(source_metadata["rate"]),
               "-fps_mode", "cfr", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temp)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"FFmpeg cleanup export failed for {destination.name}: {result.stderr.strip() or result.returncode}")
        clip_metadata = probe_clip(temp, source_metadata)
        if clip_metadata["frames"] != source_frames:
            raise RuntimeError(f"clean export frame count {clip_metadata['frames']} != PTS-verified source count {source_frames}")
        if abs(clip_metadata["start"]) > 0.001:
            raise RuntimeError(f"clean export first frame is not at local zero: {destination.name}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
        os.replace(temp, destination)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return origin, clip_metadata


def scene_text(scenes):
    return ";".join(f"{start}:{end}" for start, end in scenes)


def flag_short_ending_tail(record, hit_frames, leading_frames, output_frames, fps):
    if hit_frames and output_frames - 1 - (max(hit_frames) - leading_frames) < SHORT_ENDING_TAIL_SECONDS * fps:
        add_flag(record, "short-post-hit-tail")


def clean_record(record, raw_info, clean_root, source_video, source_metadata, overwrite=False):
    raw_path = raw_info["path"]
    clean_path = record_path(clean_root, record)
    raw_meta = raw_info["metadata"]
    report = {field: "" for field in REPORT_FIELDS}
    report.update({
        "dataset": record["dataset"], "match": record["match"], "game": record["game"],
        "original_rally_id": record["original_rally_id"], "raw_path": str(raw_path.resolve()),
        "cleaned_path": str(clean_path.resolve()), "removed_leading_frames": 0,
        "removed_leading_seconds": "0.000000", "removed_trailing_frames": 0,
        "removed_trailing_seconds": "0.000000",
    })
    raw_hit_frames = []
    try:
        raw_hit_frames = [parse_clip_frame(row.get("clip_frame_num"), record["filename"]) for row in raw_info["rows"]]
        scenes = detect_scenes(raw_path, raw_meta["frames"])
        report["scene_boundaries_zero_based_end_exclusive"] = scene_text(scenes)
        selected = containing_scene(scenes, raw_hit_frames, raw_meta["frames"])
    except Exception as exc:
        add_flag(record, "scene-detection-or-label-error")
        flag_short_ending_tail(record, raw_hit_frames, 0, raw_meta["frames"], float(raw_meta["fps"]))
        report.update({"outcome": "review-retained-full", "review_flags": f"{record['review_flags']};scene-error:{exc}"})
        copy_atomic(raw_path, clean_path, overwrite=overwrite)
        copy_atomic(raw_path.with_suffix(".csv"), clean_path.with_suffix(".csv"), overwrite=overwrite)
        return report, summary_row(record), True

    if selected is None:
        add_flag(record, "internal-cut-crosses-hit-span")
        flag_short_ending_tail(record, raw_hit_frames, 0, raw_meta["frames"], fps=float(raw_meta["fps"]))
        copy_atomic(raw_path, clean_path, overwrite=overwrite)
        copy_atomic(raw_path.with_suffix(".csv"), clean_path.with_suffix(".csv"), overwrite=overwrite)
        report.update({"outcome": "retained-internal-cut", "review_flags": record["review_flags"]})
        return report, summary_row(record), True

    leading = selected[0]
    trailing = raw_meta["frames"] - selected[1]
    fps = float(raw_meta["fps"])
    if leading == 0 and trailing == 0:
        flag_short_ending_tail(record, raw_hit_frames, 0, raw_meta["frames"], fps)
        copy_atomic(raw_path, clean_path, overwrite=overwrite)
        copy_atomic(raw_path.with_suffix(".csv"), clean_path.with_suffix(".csv"), overwrite=overwrite)
        report.update({"outcome": "copied-unchanged", "review_flags": record.get("review_flags", "")})
        return report, summary_row(record), True

    clean_start = raw_info["source_origin"] + leading / fps
    clean_end = raw_info["source_origin"] + (raw_meta["frames"] - trailing) / fps
    clean_origin, clean_meta = export_clean_interval(source_video, clean_path, clean_start, clean_end,
                                                     source_metadata, overwrite=overwrite)
    expected_origin = raw_info["source_origin"] + leading / fps
    if not math.isclose(clean_origin, expected_origin, rel_tol=0, abs_tol=max(0.0001, 0.002 / fps)):
        raise RuntimeError(f"clean export source frame origin {clean_origin:.6f}s != expected scene origin {expected_origin:.6f}s")
    if clean_meta["frames"] != raw_meta["frames"] - leading - trailing:
        raise RuntimeError(f"clean export frame count does not match selected scene for {record['filename']}")
    record["_clean_source_origin"] = clean_origin
    flag_short_ending_tail(record, raw_hit_frames, leading, clean_meta["frames"], float(clean_meta["fps"]))
    write_annotation_csv(clean_path.with_suffix(".csv"), record, raw_info["rows"], raw_info["fields"],
                         leading, float(clean_meta["fps"]), clean_meta["frames"])
    report.update({
        "removed_leading_frames": leading, "removed_leading_seconds": f"{leading/fps:.6f}",
        "removed_trailing_frames": trailing, "removed_trailing_seconds": f"{trailing/fps:.6f}",
        "outcome": "trimmed-reencoded", "review_flags": record.get("review_flags", ""),
    })
    return report, summary_row(record, clean_origin, clean_origin + clean_meta["frames"] / float(clean_meta["fps"])), True


def raw_output_state(folder, records):
    summary_path = Path(folder) / "rallies.csv"
    any_output = summary_path.exists()
    all_absent = not summary_path.exists()
    for record in records:
        path = record_path(folder, record)
        present = path.exists() or path.with_suffix(".csv").exists()
        any_output = any_output or present
        all_absent = all_absent and not present
    return any_output, all_absent


def export_new_raw(records, raw_root, video, source_metadata):
    for record in records:
        destination = record_path(raw_root, record)
        if destination.exists() or destination.with_suffix(".csv").exists():
            raise FileExistsError(f"raw output collision: {destination}; refusing to overwrite")
        export_raw(record, video, destination, source_metadata)
    folder = Path(raw_root) / records[0]["dataset"] / records[0]["match"]
    write_summary(folder, records, overwrite=False)


def preflight_clean(clean_folder, records, overwrite):
    if overwrite:
        return
    expected = [record_path(clean_folder.parent.parent, record) for record in records]
    collisions = [path for clip in expected for path in (clip, clip.with_suffix(".csv")) if path.exists()]
    for name in ("rallies.csv", "cleanup_report.csv"):
        if (Path(clean_folder) / name).exists():
            collisions.append(Path(clean_folder) / name)
    if collisions:
        raise FileExistsError(f"refusing to overwrite {collisions[0]}; pass --overwrite")


def self_test():
    assert validate_component("Sudirman Cup 2019", "match") == "Sudirman Cup 2019"
    assert validate_component("Sudirman  Cup 2019", "match") == "Sudirman  Cup 2019"
    for value in ("..", "folder/name", "folder\\name", "bad:name", " leading", "trailing "):
        try:
            validate_component(value, "match")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path component was accepted: {value!r}")
    assert containing_scene([(0, 20)], [0, 10, 19], 20) == (0, 20)  # no cuts
    scenes = [(0, 10), (10, 50), (50, 60)]
    assert containing_scene(scenes, [20, 49], 60) == (10, 50)  # leading and trailing cuts
    assert containing_scene(scenes, [9, 20], 60) is None  # internal cut divides hits
    assert containing_scene(scenes, [10], 60) == (10, 50)  # boundary starts the following scene
    assert containing_scene(scenes, [49], 60) == (10, 50)
    assert containing_scene(scenes, [50], 60) == (50, 60)  # exclusive end belongs to next scene
    adjusted = adjusted_labels([{"clip_frame_num": "25"}], 10, 25.0, 40)
    assert adjusted == [(15, 0.6, "00:00:00.600")]  # 10 leading frames shift 25 to 15
    assert adjusted_labels([{"clip_frame_num": "25"}], 0, 25.0, 30)[0][0] == 25  # trailing-only trim
    try:
        parse_clip_frame("bad", "test rally")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid frame label was accepted")
    try:
        containing_scene([(0, 10)], [10], 10)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range hit frame was accepted")
    try:
        adjusted_labels([{"clip_frame_num": "0"}], 1, 25.0, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("leading trim that removes a hit was accepted")
    ending = {"review_flags": ""}
    flag_short_ending_tail(ending, [281], 0, 289, 25.0)
    assert ending["review_flags"] == "short-post-hit-tail"
    print("Self-test passed: no cuts, edge cuts, internal cuts, single/boundary hits, timestamp shifts, trailing-only trim, and invalid frame mappings.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--annotation-folder", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--annotation-fps", type=float, default=25.0)
    parser.add_argument("--time-offset", type=float, default=0.0)
    parser.add_argument("--pre-padding", type=float, default=1.0)
    parser.add_argument("--post-padding", type=float, default=3.0)
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cleaned-output-root", type=Path, default=DEFAULT_CLEANED_ROOT)
    parser.add_argument("--start", type=float, help="bounded source start; select rally hit spans overlapping the range")
    parser.add_argument("--end", type=float, help="bounded source end; selected rallies keep their full padding")
    parser.add_argument("--overwrite", action="store_true", help="replace generated cleaned files for selected identities; raw clips are never overwritten")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    video = args.video.expanduser()
    if not video.is_file():
        raise FileNotFoundError(f"source video does not exist or is unreadable: {video}")
    # Check the pinned dependency before creating any outputs.
    load_scene_detector()
    source_metadata = probe_video(video)
    all_records = read_annotations(args.annotation_folder, args.annotation_fps, args.time_offset,
                                   args.pre_padding, args.post_padding, source_metadata["duration"])
    records = select_records(all_records, args.start, args.end)
    raw_folder = args.raw_output_root.expanduser() / all_records[0]["dataset"] / all_records[0]["match"]
    clean_folder = args.cleaned_output_root.expanduser() / all_records[0]["dataset"] / all_records[0]["match"]
    raw_summary = summary_map(raw_folder / "rallies.csv")
    any_raw, all_absent = raw_output_state(raw_folder, records)
    if not any_raw or all_absent:
        export_new_raw(records, args.raw_output_root.expanduser(), video, source_metadata)
        raw_summary = summary_map(raw_folder / "rallies.csv")
        print(f"Exported {len(records)} new raw clips to {raw_folder}")

    # Match-level identity inventory is checked against all annotation groups before clips are reused.
    expected_ids = {record["_identity"] for record in all_records}
    if raw_summary and set(raw_summary) != expected_ids and not args.overwrite:
        raise ValueError(f"raw rallies.csv identity mismatch: expected {len(expected_ids)}, found {len(raw_summary)}; existing clips were preserved")
    clips_on_disk = set()
    malformed_clips = []
    for path in raw_folder.glob("set*/rally_*.mp4"):
        try:
            clips_on_disk.add((all_records[0]["dataset"], all_records[0]["match"],
                               str(int(path.parent.name[3:])), str(int(path.stem[6:]))))
        except ValueError:
            malformed_clips.append(path)
    inventory_warnings = []
    if raw_summary and set(raw_summary) != expected_ids:
        inventory_warnings.append(f"summary identities expected {len(expected_ids)}, found {len(raw_summary)}")
    if clips_on_disk and clips_on_disk != expected_ids:
        inventory_warnings.append(f"clip identities expected {len(expected_ids)}, found {len(clips_on_disk)}")
    if malformed_clips:
        inventory_warnings.append(f"malformed raw clip name: {malformed_clips[0]}")
    for warning in inventory_warnings:
        print(f"RAW INVENTORY MISMATCH: {warning}; existing clips are preserved", file=sys.stderr)

    preflight_clean(clean_folder, records, args.overwrite)
    clean_folder.mkdir(parents=True, exist_ok=True)
    reports, clean_rows = [], []
    for warning in inventory_warnings:
        mismatch_report = {field: "" for field in REPORT_FIELDS}
        mismatch_report.update({
            "dataset": all_records[0]["dataset"], "match": all_records[0]["match"],
            "outcome": "raw-identity-inventory-mismatch", "review_flags": warning,
        })
        reports.append(mismatch_report)
    valid = 0
    for index, record in enumerate(records, 1):
        print(f"[{index}/{len(records)}] game {record['game']} rally {record['original_rally_id']}", flush=True)
        raw_path = record_path(args.raw_output_root.expanduser(), record)
        try:
            raw_info = verify_raw(record, raw_path, raw_summary, video, source_metadata)
        except Exception as exc:
            report = {field: "" for field in REPORT_FIELDS}
            report.update({
                "dataset": record["dataset"], "match": record["match"], "game": record["game"],
                "original_rally_id": record["original_rally_id"], "raw_path": str(raw_path.resolve()),
                "cleaned_path": str(record_path(args.cleaned_output_root.expanduser(), record).resolve()),
                "outcome": "raw-mismatch-skipped", "review_flags": f"raw-mismatch:{exc}",
            })
            reports.append(report)
            print(f"  RAW MISMATCH: {exc}", file=sys.stderr)
            continue
        valid += 1
        try:
            report, row, _ = clean_record(record, raw_info, args.cleaned_output_root.expanduser(),
                                          video, source_metadata, overwrite=args.overwrite)
            reports.append(report)
            clean_rows.append(row)
        except Exception as exc:
            report = {field: "" for field in REPORT_FIELDS}
            report.update({
                "dataset": record["dataset"], "match": record["match"], "game": record["game"],
                "original_rally_id": record["original_rally_id"], "raw_path": str(raw_path.resolve()),
                "cleaned_path": str(record_path(args.cleaned_output_root.expanduser(), record).resolve()),
                "outcome": "clean-export-failed", "review_flags": f"clean-export-error:{exc}",
            })
            reports.append(report)
            failed_path = record_path(args.cleaned_output_root.expanduser(), record)
            failed_path.unlink(missing_ok=True)
            failed_path.with_suffix(".csv").unlink(missing_ok=True)
            print(f"  CLEAN EXPORT FAILED: {exc}", file=sys.stderr)

    report_path = clean_folder / "cleanup_report.csv"
    if report_path.exists() and args.overwrite:
        reports = merge_rows(report_path, reports)
    write_csv(report_path, REPORT_FIELDS, reports, overwrite=args.overwrite)
    processed_count = len(clean_rows)
    clean_summary_path = clean_folder / "rallies.csv"
    if clean_summary_path.exists() and args.overwrite:
        clean_rows = merge_rows(clean_summary_path, clean_rows)
    write_csv(clean_summary_path, SUMMARY_FIELDS, clean_rows, overwrite=args.overwrite)
    print(f"Verified/reused {valid}/{len(records)} raw clips; cleaned outputs: {clean_folder}")
    print(f"Cleanup report: {report_path}")
    return 0 if processed_count == len(records) and not inventory_warnings else 2


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
