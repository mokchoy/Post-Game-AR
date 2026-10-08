"""Create reviewable candidate badminton rally clips from a match video."""

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

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = Path(
    r"G:\My Drive\Match_Downloads\F ｜ MS ｜ Kento MOMOTA (JPN) [1] vs. CHOU Tien Chen (TPE) [2] ｜ BWF 2019.mp4"
)
SAMPLE_FPS = 10
SMOOTHING_SAMPLES = 5
BATCH_SIZE = 16
CSV_FIELDS = ("rally_id", "start_seconds", "end_seconds", "filename")
ANNOTATION_FIELDS = CSV_FIELDS + (
    "dataset", "match", "game", "original_rally_id", "source_csv",
    "first_hit_frame", "last_hit_frame", "annotation_fps", "time_offset",
    "export_start_seconds", "export_end_seconds", "review_flags",
)
ANNOTATION_ID_FIELDS = ("dataset", "match", "game", "original_rally_id")
LABEL_FIELDS = ("source_time_seconds", "clip_time_seconds", "clip_frame_num")
GENERATED_CLIP = re.compile(r"^rally_\d{3,}\.mp4$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SACNN(nn.Module):
    """Reference SA-CNN: three conv/pool/BN/ReLU blocks and a 2-class head."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(32 * 27 * 27, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.pool(self.conv1(x))))
        x = self.relu(self.bn2(self.pool(self.conv2(x))))
        x = self.relu(self.bn3(self.pool(self.conv3(x))))
        return self.relu(self.l1(torch.flatten(x, 1)))


class FiveSampleSmoother:
    """Centered five-sample majority vote; timestamps stay with their samples."""

    def __init__(self):
        self.samples = []
        self.emitted = 0

    def _emit_ready(self, eof=False):
        output = []
        while self.emitted < len(self.samples):
            center = self.emitted
            if not eof and center + SMOOTHING_SAMPLES // 2 >= len(self.samples):
                break
            votes = []
            for index in range(center - 2, center + 3):
                index = min(max(index, 0), len(self.samples) - 1)
                votes.append(self.samples[index][0])
            label = int(sum(votes) >= 3)
            output.append((label, self.samples[center][1]))
            self.emitted += 1
        return output

    def push(self, label, timestamp):
        self.samples.append((int(label), float(timestamp)))
        return self._emit_ready()

    def flush(self):
        return self._emit_ready(eof=True)


class Segmenter:
    def __init__(self):
        self.intervals = []
        self.active_start = None
        self.last_timestamp = None

    def push(self, label, timestamp):
        timestamp = float(timestamp)
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("prediction timestamps must be nondecreasing")
        self.last_timestamp = timestamp
        if label not in (0, 1):
            raise ValueError("prediction label must be 0 or 1")
        if label == 1 and self.active_start is None:
            self.active_start = timestamp
        elif label == 0 and self.active_start is not None:
            self.intervals.append((self.active_start, timestamp))
            self.active_start = None

    def finish(self, eof_timestamp):
        eof_timestamp = float(eof_timestamp)
        if self.last_timestamp is not None and eof_timestamp < self.last_timestamp:
            raise ValueError("EOF timestamp precedes the last prediction")
        if self.active_start is not None:
            self.intervals.append((self.active_start, eof_timestamp))
            self.active_start = None
        return self.intervals


def validate_intervals(rows, video_duration, allow_annotation_overlap=False, require_annotation=False):
    """Validate intervals while retaining all identity and review metadata."""
    cleaned, seen_ids, previous_start, previous_end = [], set(), -1.0, -1.0
    for row_number, row in enumerate(rows, 2):
        record = dict(row)
        has_identity = any(record.get(field) not in (None, "") for field in ANNOTATION_ID_FIELDS)
        if require_annotation and not has_identity:
            raise ValueError(f"CSV row {row_number}: annotation identity fields are required")
        if has_identity:
            if not all(str(record.get(field, "")).strip() for field in ANNOTATION_ID_FIELDS):
                raise ValueError(f"CSV row {row_number}: annotation identity fields are incomplete")
            for field in ANNOTATION_ID_FIELDS[:2]:
                validate_component(str(record[field]).strip(), field)
            game = str(record["game"]).strip()
            original_id = str(record["original_rally_id"]).strip()
            if not game.isdigit() or int(game) < 1 or not original_id.isdigit() or int(original_id) < 1:
                raise ValueError(f"CSV row {row_number}: game and original_rally_id must be positive integers")
            rally_id = str(record.get("rally_id", "")).strip()
            if rally_id and rally_id != original_id:
                raise ValueError(f"CSV row {row_number}: rally_id must retain original_rally_id")
            identity = tuple(str(record[field]).strip() for field in ANNOTATION_ID_FIELDS)
        else:
            rally_id = str(record.get("rally_id", "")).strip()
            if not rally_id:
                raise ValueError(f"CSV row {row_number}: rally_id is blank")
            identity = (rally_id,)
        if identity in seen_ids:
            raise ValueError(f"CSV row {row_number}: rally identity is duplicated")
        seen_ids.add(identity)
        try:
            start = float(record["start_seconds"])
            end = float(record["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"CSV row {row_number}: start_seconds/end_seconds must be numbers") from exc
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"CSV row {row_number}: times must be finite")
        if start < 0 or end <= start or end > video_duration:
            raise ValueError(
                f"CSV row {row_number}: require 0 <= start < end <= video duration ({video_duration:.3f}s)"
            )
        if start < previous_start:
            raise ValueError(f"CSV row {row_number}: intervals must be in chronological start order")
        if start < previous_end and not (allow_annotation_overlap and has_identity):
            raise ValueError(f"CSV row {row_number}: intervals must be non-overlapping")
        previous_start, previous_end = start, max(previous_end, end)
        record["start_seconds"], record["end_seconds"] = start, end
        cleaned.append(record)
    return cleaned


def read_reviewed_csv(path, video_duration, annotation=False):
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = ANNOTATION_FIELDS if annotation else CSV_FIELDS
            if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
                raise ValueError(f"reviewed CSV must contain columns: {', '.join(required)}")
            if not annotation and set(ANNOTATION_ID_FIELDS).issubset(reader.fieldnames):
                raise ValueError("annotation reviewed CSV requires --annotation-folder and --annotation-fps")
            return validate_intervals(reader, video_duration, allow_annotation_overlap=annotation,
                                      require_annotation=annotation)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read reviewed CSV {path}: {exc}") from exc


def probe_video(video):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration,start_time:stream=codec_type,width,height,avg_frame_rate,r_frame_rate,start_time",
         "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(f"FFprobe could not read {video}: {result.stderr.strip() or result.returncode}")
    try:
        info = json.loads(result.stdout)
        video_stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
        duration = float(info["format"]["duration"])
        rate_text = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        rate = Fraction(rate_text)
        avg_rate = Fraction(video_stream.get("avg_frame_rate", rate_text))
        r_rate = Fraction(video_stream.get("r_frame_rate", rate_text))
        start_time = float(video_stream.get("start_time", info["format"].get("start_time", 0)) or 0)
        if rate <= 0:
            raise ValueError("invalid frame rate")
        return {
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
            "duration": duration,
            "rate": rate,
            "avg_rate": avg_rate,
            "r_rate": r_rate,
            "video_start_time": start_time,
            "has_audio": any(s.get("codec_type") == "audio" for s in info["streams"]),
        }
    except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"FFprobe returned incomplete video metadata for {video}: {exc}") from exc


def ffprobe_frame_times(video, start=None, duration=None):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH")
    command = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if start is not None:
        command += ["-read_intervals", f"{start:.6f}%+{duration:.6f}"]
    command += ["-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"FFprobe frame read failed for {video}: {result.stderr.strip() or result.returncode}")
    try:
        frames = json.loads(result.stdout).get("frames", [])
        return [float(frame["best_effort_timestamp_time"]) for frame in frames
                if frame.get("best_effort_timestamp_time") is not None]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"FFprobe returned invalid frame timestamps for {video}: {exc}") from exc


def verify_cfr_timestamps(timestamps, rate, description):
    if len(timestamps) < 2:
        raise RuntimeError(f"cannot verify constant frame rate for {description}: fewer than two frames")
    period = 1.0 / float(rate)
    tolerance = max(0.0001, period * 0.002)
    if any(abs((later - earlier) - period) > tolerance
           for earlier, later in zip(timestamps, timestamps[1:])):
        raise RuntimeError(f"{description} is not constant-frame-rate at {float(rate):g} fps; label export refused")


def source_interval_origin(video, metadata, start, end):
    if metadata["avg_rate"] != metadata["r_rate"]:
        raise RuntimeError("annotation label export requires a constant-frame-rate source; stream rates disagree")
    absolute_start = metadata["video_start_time"] + start
    tolerance = 0.0001
    period = 1.0 / float(metadata["rate"])
    inside = []
    for extra_seconds in (0.2, 5.2, 10.2, 20.2, 40.2):
        query_duration = end - start + extra_seconds
        if query_duration > 60.0:
            continue
        timestamps = ffprobe_frame_times(video, absolute_start, query_duration)
        relative = [timestamp - metadata["video_start_time"] for timestamp in timestamps]
        inside = [timestamp for timestamp in relative if start - tolerance <= timestamp < end - tolerance]
        if inside and inside[-1] >= end - period - tolerance:
            break
    else:
        raise RuntimeError(f"could not probe the full source interval for CFR verification: {start:.3f}-{end:.3f}s")
    if not inside:
        raise RuntimeError(f"FFprobe found no source frames in requested interval {start:.6f}-{end:.6f}s")
    origin = inside[0]
    if origin - start > period + tolerance:
        raise RuntimeError("first source frame is more than one frame after the requested clip start")
    verify_cfr_timestamps(inside, metadata["rate"], f"source interval {start:.3f}-{end:.3f}s")
    return origin, len(inside)


def probe_exported_clip(path, source_metadata):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH")
    command = [ffprobe, "-v", "error", "-count_frames", "-show_entries",
               "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,start_time,duration,nb_read_frames",
               "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"FFprobe could not verify exported clip {path}: {result.stderr.strip() or result.returncode}")
    try:
        info = json.loads(result.stdout)
        video_stream = next(stream for stream in info["streams"] if stream.get("codec_type") == "video")
        audio_streams = [stream for stream in info["streams"] if stream.get("codec_type") == "audio"]
        fps = Fraction(video_stream["avg_frame_rate"])
        r_fps = Fraction(video_stream["r_frame_rate"])
        frames = int(video_stream["nb_read_frames"])
        duration = float(info["format"]["duration"])
        frame_start = float(video_stream.get("start_time", 0) or 0)
        frame_times = ffprobe_frame_times(path)
        if not frame_times or abs(frame_times[0] - frame_start) > 0.001:
            raise RuntimeError(f"first output frame timestamp is inconsistent with stream origin: {path}")
        local_times = [timestamp - frame_times[0] for timestamp in frame_times] if frame_times else []
        verify_cfr_timestamps(local_times, fps, str(path))
        if fps != source_metadata["rate"]:
            raise RuntimeError(f"exported frame rate {fps} differs from source {source_metadata['rate']}")
        if r_fps != fps:
            raise RuntimeError(f"exported stream rates disagree ({fps} avg vs {r_fps} nominal)")
        if (int(video_stream["width"]) != source_metadata["width"] or
                int(video_stream["height"]) != source_metadata["height"]):
            raise RuntimeError(f"exported resolution changed for {path}")
        if video_stream.get("codec_name") != "h264":
            raise RuntimeError(f"export is not H.264: {path}")
        if source_metadata["has_audio"] and not any(stream.get("codec_name") == "aac" for stream in audio_streams):
            raise RuntimeError(f"export is missing AAC audio: {path}")
        if frame_start > 0.001:
            raise RuntimeError(f"exported clip does not start at local time zero: {path}")
        if source_metadata["has_audio"]:
            audio_stream = next(stream for stream in audio_streams if stream.get("codec_name") == "aac")
            audio_start = float(audio_stream.get("start_time", 0) or 0)
            audio_duration = float(audio_stream.get("duration", duration))
            sync_tolerance = max(0.08, 1.0 / float(fps))
            if (abs(audio_start - frame_start) > sync_tolerance or
                    abs(audio_duration - duration) > sync_tolerance):
                raise RuntimeError(f"exported audio is not synchronized with video in {path}")
        if frames != len(frame_times):
            raise RuntimeError(f"FFprobe frame count disagreement for {path}: {frames} vs {len(frame_times)}")
        return {"fps": fps, "frames": frames, "duration": duration,
                "has_audio": bool(audio_streams), "audio_codec": audio_streams[0].get("codec_name") if audio_streams else "none"}
    except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"FFprobe returned incomplete clip metadata for {path}: {exc}") from exc


def validate_bounds(start, end, duration):
    if (start is None) != (end is None):
        raise ValueError("--start and --end must be supplied together")
    if start is None:
        return 0.0, duration
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("--start and --end must be finite seconds")
    if start < 0 or end <= start or end > duration:
        raise ValueError(f"require 0 <= --start < --end <= video duration ({duration:.3f}s)")
    return float(start), float(end)


def validate_component(value, label):
    if not SAFE_COMPONENT.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"{label} must be a safe single path component: {value!r}")
    return value


def parse_coarse_time(value):
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3) or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid whole-second annotation time {value!r}")
    values = [int(part) for part in parts]
    if values[-1] >= 60 or (len(values) == 3 and values[-2] >= 60):
        raise ValueError(f"invalid whole-second annotation time {value!r}")
    if len(values) == 2:
        return values[0] * 60 + values[1]
    return values[0] * 3600 + values[1] * 60 + values[2]


def parse_hit_frame(value, path, row_number):
    try:
        frame = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} row {row_number}: frame_num must be a nonnegative integer") from exc
    if not math.isfinite(frame) or frame < 0 or not frame.is_integer():
        raise ValueError(f"{path} row {row_number}: frame_num must be a nonnegative integer")
    return int(frame)


def annotation_filename(record):
    for field in ANNOTATION_ID_FIELDS[:2]:
        validate_component(str(record[field]), field)
    game, rally_id = str(record["game"]), str(record["original_rally_id"])
    if not game.isdigit() or int(game) < 1 or not rally_id.isdigit() or int(rally_id) < 1:
        raise ValueError("annotation game and rally identity must be positive integers")
    return f"set{int(game)}/rally_{int(rally_id):03d}.mp4"


def read_annotation_folder(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"annotation folder does not exist: {folder}")
    dataset = validate_component(folder.parent.name, "dataset")
    match = validate_component(folder.name, "match")
    files = []
    for path in folder.glob("set*.csv"):
        match_result = re.fullmatch(r"set([1-9]\d*)\.csv", path.name, re.IGNORECASE)
        if match_result:
            files.append((int(match_result.group(1)), path))
    files.sort(key=lambda item: item[0])
    if not files:
        raise ValueError(f"no set<N>.csv annotation files found in {folder}")
    games, groups = set(), {}
    for game, path in files:
        if game in games:
            raise ValueError(f"multiple annotation CSVs identify game {game} in {folder}")
        games.add(game)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"rally", "time", "frame_num"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
                for row_number, row in enumerate(reader, 2):
                    rally_id = str(row.get("rally", "")).strip()
                    if not rally_id.isdigit() or int(rally_id) < 1:
                        raise ValueError(f"{path} row {row_number}: rally must be a positive original rally ID")
                    frame = parse_hit_frame(row.get("frame_num"), path, row_number)
                    try:
                        coarse_seconds = parse_coarse_time(row.get("time", ""))
                    except ValueError as exc:
                        raise ValueError(f"{path} row {row_number}: {exc}") from exc
                    key = (dataset, match, str(game), rally_id)
                    group = groups.setdefault(key, {
                        "dataset": dataset, "match": match, "game": str(game),
                        "original_rally_id": rally_id, "source_csv": path.name,
                        "rows": [], "frames": [], "coarse_seconds": [],
                    })
                    group["rows"].append(dict(row))
                    group["frames"].append(frame)
                    group["coarse_seconds"].append((row_number, coarse_seconds, frame))
        except (OSError, csv.Error) as exc:
            raise ValueError(f"cannot read annotation CSV {path}: {exc}") from exc
    if not groups:
        raise ValueError(f"annotation files contain no rally hits: {folder}")
    return {"folder": folder, "dataset": dataset, "match": match, "groups": groups}


def add_review_flags(record, *flags):
    current = [flag for flag in str(record.get("review_flags", "")).split(";") if flag]
    for flag in flags:
        if flag and flag not in current:
            current.append(flag)
    record["review_flags"] = ";".join(current)


def update_overlap_flags(records, hit_spans=None, preserve_existing=False):
    by_game = {}
    for record in records:
        if not preserve_existing:
            flags = [flag for flag in str(record.get("review_flags", "")).split(";")
                     if flag not in ("padding-overlap", "hit-span-overlap")]
            record["review_flags"] = ";".join(flags)
        key = (record["dataset"], record["match"], str(record["game"]))
        by_game.setdefault(key, []).append(record)
    # ponytail: pairwise checks are quadratic, bounded by the small rally count per game.
    for game_records in by_game.values():
        for index, first in enumerate(game_records):
            for second in game_records[index + 1:]:
                if first["start_seconds"] < second["end_seconds"] and second["start_seconds"] < first["end_seconds"]:
                    add_review_flags(first, "padding-overlap")
                    add_review_flags(second, "padding-overlap")
                if hit_spans is not None:
                    first_span, second_span = hit_spans[first["_identity_key"]], hit_spans[second["_identity_key"]]
                    if first_span[0] <= second_span[1] and second_span[0] <= first_span[1]:
                        add_review_flags(first, "hit-span-overlap")
                        add_review_flags(second, "hit-span-overlap")


def build_annotation_records(context, annotation_fps, time_offset, pre_padding, post_padding, video_duration):
    if not math.isfinite(annotation_fps) or annotation_fps <= 0:
        raise ValueError("--annotation-fps must be finite and greater than zero")
    if not math.isfinite(time_offset):
        raise ValueError("--time-offset must be finite")
    if not (math.isfinite(pre_padding) and math.isfinite(post_padding)) or min(pre_padding, post_padding) < 0:
        raise ValueError("annotation padding must be finite and nonnegative")
    records, hit_spans, discrepancies = [], {}, 0
    for key, group in context["groups"].items():
        hit_times = [frame / annotation_fps + time_offset for frame in group["frames"]]
        if any(not math.isfinite(value) or value < 0 or value > video_duration for value in hit_times):
            raise ValueError(f"annotation hits for {group['source_csv']} rally {group['original_rally_id']} fall outside the video")
        first_hit, last_hit = min(hit_times), max(hit_times)
        start, end = max(0.0, first_hit - pre_padding), min(video_duration, last_hit + post_padding)
        if end <= start:
            raise ValueError(f"annotation rally {group['original_rally_id']} has an empty padded interval")
        record = {
            "rally_id": group["original_rally_id"], "start_seconds": start, "end_seconds": end,
            "filename": "", "dataset": group["dataset"], "match": group["match"],
            "game": group["game"], "original_rally_id": group["original_rally_id"],
            "source_csv": group["source_csv"], "first_hit_frame": min(group["frames"]),
            "last_hit_frame": max(group["frames"]), "annotation_fps": annotation_fps,
            "time_offset": time_offset, "export_start_seconds": start, "export_end_seconds": end,
            "review_flags": "", "_annotation_rows": group["rows"], "_identity_key": key,
        }
        if start == 0.0 and first_hit - pre_padding < 0:
            add_review_flags(record, "start-clamped-to-video")
        if end == video_duration and last_hit + post_padding > video_duration:
            add_review_flags(record, "end-clamped-to-video")
        for row_number, coarse_seconds, hit_frame in group["coarse_seconds"]:
            hit_time = hit_frame
            discrepancy = hit_time / annotation_fps + time_offset - coarse_seconds
            if abs(discrepancy) > 1.0:
                discrepancies += 1
                add_review_flags(record, "coarse-time-discrepancy")
                print(
                    f"Warning: {group['source_csv']} row {row_number} rally {group['original_rally_id']}: "
                    f"frame-derived {hit_time / annotation_fps + time_offset:.3f}s differs from coarse time "
                    f"{coarse_seconds}s by {discrepancy:+.3f}s; no correction applied.", file=sys.stderr,
                )
        record["filename"] = annotation_filename(record)
        records.append(record)
        hit_spans[key] = (first_hit, last_hit)
    records.sort(key=lambda record: (record["start_seconds"], int(record["game"]), int(record["original_rally_id"])))
    update_overlap_flags(records, hit_spans)
    if discrepancies:
        print(f"Reported {discrepancies} coarse-time discrepancy row(s); review before use.", file=sys.stderr)
    else:
        checked = sum(len(group["frames"]) for group in context["groups"].values())
        print(f"Checked {checked} annotation hit times against coarse whole-second values; none differed by more than 1 second.")
    return validate_intervals(records, video_duration, allow_annotation_overlap=True)


def select_annotation_records(records, start, end):
    selected, excluded = [], []
    for record in records:
        if record["start_seconds"] >= start and record["end_seconds"] <= end:
            selected.append(record)
        else:
            excluded.append(record)
    if excluded:
        print(f"Excluded {len(excluded)} rally/rallies whose full padded interval does not fit {start:.3f}-{end:.3f}s:")
        for record in excluded:
            print(f"  game {record['game']} rally {record['original_rally_id']}: "
                  f"{record['start_seconds']:.3f}-{record['end_seconds']:.3f}s")
    return selected, excluded


def restore_reviewed_annotation_records(records, context, annotation_fps, video_duration):
    if not math.isfinite(annotation_fps) or annotation_fps <= 0:
        raise ValueError("--annotation-fps must be finite and greater than zero")
    restored, hit_spans = [], {}
    groups = context["groups"]
    for record in records:
        if record["dataset"] != context["dataset"] or record["match"] != context["match"]:
            raise ValueError("reviewed CSV dataset/match identity does not match --annotation-folder")
        key = (record["dataset"], record["match"], str(record["game"]), str(record["original_rally_id"]))
        group = groups.get(key)
        if group is None:
            raise ValueError(f"reviewed CSV rally identity {key} was not found in the annotation folder")
        if record.get("source_csv") != group["source_csv"]:
            raise ValueError(f"reviewed CSV source_csv does not match rally identity {key}")
        try:
            saved_fps = float(record["annotation_fps"])
            offset = float(record["time_offset"])
            first_frame = int(record["first_hit_frame"])
            last_frame = int(record["last_hit_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"reviewed CSV has invalid annotation metadata for rally identity {key}") from exc
        if not math.isclose(saved_fps, annotation_fps, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"reviewed CSV annotation_fps {saved_fps:g} differs from --annotation-fps {annotation_fps:g}")
        if not math.isfinite(offset) or first_frame != min(group["frames"]) or last_frame != max(group["frames"]):
            raise ValueError(f"reviewed CSV hit-frame identity does not match annotations for rally identity {key}")
        canonical_filename = annotation_filename(record)
        if record.get("filename") != canonical_filename:
            raise ValueError(f"reviewed CSV filename is not canonical for rally identity {key}: expected {canonical_filename}")
        record["annotation_fps"] = saved_fps
        record["time_offset"] = offset
        record["export_start_seconds"] = record["start_seconds"]
        record["export_end_seconds"] = record["end_seconds"]
        record["_annotation_rows"] = group["rows"]
        record["_identity_key"] = key
        restored.append(record)
        hit_spans[key] = (min(group["frames"]) / saved_fps + offset,
                          max(group["frames"]) / saved_fps + offset)
    update_overlap_flags(restored, hit_spans, preserve_existing=True)
    return validate_intervals(restored, video_duration, allow_annotation_overlap=True)


def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SACNN()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device


def decode_predictions(video, metadata, start, end, model, device):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH")
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        command += ["-ss", f"{start:.6f}"]
    command += ["-i", str(video), "-t", f"{end - start:.6f}", "-map", "0:v:0",
                "-an", "-vf", f"fps={SAMPLE_FPS}:start_time=0", "-pix_fmt", "rgb24",
                "-f", "rawvideo", "pipe:1"]

    transform = transforms.Compose([
        transforms.Resize((216, 384)),
        transforms.CenterCrop(216),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    frame_bytes = metadata["width"] * metadata["height"] * 3
    smoother, segmenter = FiveSampleSmoother(), Segmenter()
    batch, timestamps, sample_count = [], [], 0

    def classify_batch():
        with torch.inference_mode():
            logits = model(torch.stack(batch).to(device))
            labels = logits.argmax(dim=1).tolist()
        for label, timestamp in zip(labels, timestamps):
            for smoothed_label, smoothed_time in smoother.push(label, timestamp):
                segmenter.push(smoothed_label, smoothed_time)
        batch.clear()
        timestamps.clear()

    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log)
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError(f"FFmpeg returned a partial RGB frame after sample {sample_count}")
                image = Image.frombytes("RGB", (metadata["width"], metadata["height"]), raw)
                batch.append(transform(image))
                timestamps.append(start + sample_count / SAMPLE_FPS)
                sample_count += 1
                if len(batch) == BATCH_SIZE:
                    classify_batch()
            if batch:
                classify_batch()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            return_code = process.wait()
        error_log.seek(max(0, error_log.tell() - 4000))
        ffmpeg_error = error_log.read().decode("utf-8", "replace").strip()
    if return_code:
        raise RuntimeError(f"FFmpeg decode failed (exit {return_code}): {ffmpeg_error or 'no diagnostic output'}")
    if sample_count == 0:
        raise RuntimeError("FFmpeg decoded no samples for the selected interval")

    for label, timestamp in smoother.flush():
        segmenter.push(label, timestamp)
    return segmenter.finish(end), sample_count


def output_folder_for(video, output_root):
    return Path(output_root).expanduser() / video.stem


def preflight_output(folder, overwrite, csv_input=None):
    csv_output = folder / "rallies.csv"
    same_csv = csv_input is not None and Path(csv_input).resolve() == csv_output.resolve()
    if csv_output.exists() and not same_csv:
        raise FileExistsError(
            f"refusing to overwrite {csv_output}; --overwrite replaces generated numbered clips only"
        )
    existing = sorted(p.name for p in folder.glob("rally_*.mp4") if GENERATED_CLIP.fullmatch(p.name))
    if existing and not overwrite:
        raise FileExistsError(
            f"generated clips already exist in {folder} ({', '.join(existing[:5])}); pass --overwrite to replace matching numbered clips"
        )
    return same_csv


def preflight_annotation_output(folder, records, overwrite, csv_input=None):
    summary = folder / "rallies.csv"
    same_csv = csv_input is not None and Path(csv_input).resolve() == summary.resolve()
    if summary.exists() and not same_csv and not overwrite:
        raise FileExistsError(f"refusing to overwrite {summary}; pass --overwrite")
    targets = []
    for record in records:
        clip = folder / annotation_filename(record)
        targets.extend((clip, clip.with_suffix(".csv")))
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite generated annotation output {existing[0]}; pass --overwrite")
    return same_csv


def interval_bounds(item):
    if isinstance(item, dict):
        return float(item["start_seconds"]), float(item["end_seconds"])
    return float(item[0]), float(item[1])


def nearest_clip_frame(source_seconds, source_origin, output_fps, frame_count):
    frame = math.floor((source_seconds - source_origin) * output_fps + 0.5)
    if frame < 0 or frame >= frame_count:
        raise ValueError(f"annotation hit at {source_seconds:.6f}s maps outside the {frame_count}-frame clip")
    return frame


def write_csv_new(path, intervals, overwrite=False, annotation=False):
    path = Path(path)
    fields = ANNOTATION_FIELDS if annotation else CSV_FIELDS
    temporary = None
    if overwrite:
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(handle)
        target = Path(temporary_name)
        temporary = target
        mode = "w"
    else:
        target, mode = path, "x"
    try:
        with target.open(mode, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for number, item in enumerate(intervals, 1):
                start, end = interval_bounds(item)
                if annotation:
                    row = {field: item.get(field, "") for field in fields}
                    row["start_seconds"] = f"{start:.6f}"
                    row["end_seconds"] = f"{end:.6f}"
                    row["export_start_seconds"] = f"{start:.6f}"
                    row["export_end_seconds"] = f"{end:.6f}"
                    row["annotation_fps"] = f"{float(item['annotation_fps']):.9g}"
                    row["time_offset"] = f"{float(item['time_offset']):.9g}"
                    row["rally_id"] = str(item["original_rally_id"])
                    row["filename"] = annotation_filename(item)
                else:
                    row = {
                        "rally_id": number,
                        "start_seconds": f"{start:.6f}",
                        "end_seconds": f"{end:.6f}",
                        "filename": f"rally_{number:03d}.mp4",
                    }
                writer.writerow(row)
        if temporary is not None:
            os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_annotation_labels(path, record, source_origin, clip_metadata, overwrite=False):
    rows = record["_annotation_rows"]
    if not rows:
        raise ValueError(f"no original annotation rows found for {record['filename']}")
    original_fields = list(rows[0])
    if not set(("frame_num",)).issubset(original_fields):
        raise ValueError(f"source annotation rows lack frame_num for {record['filename']}")
    fields = original_fields + [field for field in LABEL_FIELDS if field not in original_fields]
    path = Path(path)
    temporary = None
    if overwrite:
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(handle)
        target, temporary = Path(temporary_name), Path(temporary_name)
        mode = "w"
    else:
        target, mode = path, "x"
    annotation_fps = float(record["annotation_fps"])
    time_offset = float(record["time_offset"])
    output_fps = float(clip_metadata["fps"])
    try:
        with target.open(mode, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row_number, source_row in enumerate(rows, 2):
                frame = parse_hit_frame(source_row["frame_num"], path, row_number)
                source_seconds = frame / annotation_fps + time_offset
                try:
                    clip_frame = nearest_clip_frame(
                        source_seconds, source_origin, output_fps, clip_metadata["frames"]
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"annotation hit at {source_seconds:.6f}s maps outside {record['filename']} "
                        f"({clip_metadata['frames']} frames)"
                    ) from exc
                labeled = dict(source_row)
                labeled["source_time_seconds"] = f"{source_seconds:.6f}"
                labeled["clip_frame_num"] = clip_frame
                labeled["clip_time_seconds"] = f"{clip_frame / output_fps:.6f}"
                writer.writerow(labeled)
        if temporary is not None:
            os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def export_clips(video, folder, intervals, metadata, overwrite, annotation=False):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH")
    outputs = []
    if not annotation:
        expected = {f"rally_{n:03d}.mp4" for n in range(1, len(intervals) + 1)}
        stale = sorted(p.name for p in folder.glob("rally_*.mp4")
                       if GENERATED_CLIP.fullmatch(p.name) and p.name not in expected)
        if stale:
            print(f"Warning: preserving unmatched numbered clips: {', '.join(stale[:5])}", file=sys.stderr)

    for number, item in enumerate(intervals, 1):
        start, end = interval_bounds(item)
        name = annotation_filename(item) if annotation else f"rally_{number:03d}.mp4"
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
        if annotation:
            source_origin, source_frame_count = source_interval_origin(video, metadata, start, end)
        else:
            source_origin, source_frame_count = None, None
        handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp.mp4", dir=destination.parent)
        os.close(handle)
        temporary = Path(temporary_name)
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-ss", f"{start:.6f}", "-i", str(video), "-t", f"{end - start:.6f}",
                   "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(metadata["rate"]),
                   "-fps_mode", "cfr", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                   str(temporary)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"FFmpeg export failed for {name} ({start:.3f}-{end:.3f}s, exit {result.returncode}): "
                f"{result.stderr.strip() or 'no diagnostic output'}"
            )
        label_temporary = None
        if annotation:
            try:
                clip_metadata = probe_exported_clip(temporary, metadata)
                if clip_metadata["frames"] != source_frame_count:
                    raise RuntimeError(
                        f"exported {clip_metadata['frames']} frames but the verified source interval contains "
                        f"{source_frame_count}; refusing uncertain annotation frame labels"
                    )
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            label_path = destination.with_suffix(".csv")
            handle, label_temporary_name = tempfile.mkstemp(
                prefix=f".{label_path.name}.", suffix=".tmp.csv", dir=destination.parent
            )
            os.close(handle)
            label_temporary = Path(label_temporary_name)
            label_temporary.unlink()
            try:
                write_annotation_labels(label_temporary, item, source_origin, clip_metadata)
            except BaseException:
                temporary.unlink(missing_ok=True)
                label_temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, destination)
        if label_temporary is not None:
            os.replace(label_temporary, destination.with_suffix(".csv"))
        outputs.append(name)
    return outputs


def self_test():
    raw = [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    smoother, outputs = FiveSampleSmoother(), []
    offset = 1000.0
    for index, label in enumerate(raw):
        outputs.extend(smoother.push(label, offset + index / SAMPLE_FPS))
    outputs.extend(smoother.flush())
    labels = [label for label, _ in outputs]
    assert labels == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], labels
    assert math.isclose(outputs[4][1], 1000.4)
    segmenter = Segmenter()
    for label, timestamp in outputs:
        segmenter.push(label, timestamp)
    assert segmenter.finish(1001.1) == [(1000.4, 1000.8)]

    smoother, outputs = FiveSampleSmoother(), []
    for index, label in enumerate([0, 0, 1, 1, 1]):
        outputs.extend(smoother.push(label, 30 + index / SAMPLE_FPS))
    outputs.extend(smoother.flush())
    segmenter = Segmenter()
    for label, timestamp in outputs:
        segmenter.push(label, timestamp)
    assert segmenter.finish(30.5) == [(30.2, 30.5)]  # active candidate closes at EOF

    valid = [{"rally_id": "1", "start_seconds": "2", "end_seconds": "3", "filename": "rally_001.mp4"}]
    assert validate_intervals(valid, 10)[0]["start_seconds"] == 2.0
    for invalid in (
        [{"rally_id": "1", "start_seconds": "3", "end_seconds": "3", "filename": "x"}],
        [{"rally_id": "1", "start_seconds": "-1", "end_seconds": "2", "filename": "x"}],
        [{"rally_id": "1", "start_seconds": "2", "end_seconds": "11", "filename": "x"}],
        [valid[0], {"rally_id": "2", "start_seconds": "2.5", "end_seconds": "4", "filename": "y"}],
        [valid[0], {"rally_id": "1", "start_seconds": "4", "end_seconds": "5", "filename": "y"}],
    ):
        try:
            validate_intervals(invalid, 10)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid intervals were accepted: {invalid}")

    with tempfile.TemporaryDirectory(prefix="trim-rallies-self-test-", dir=HERE) as temporary:
        dataset = Path(temporary) / "set-1"
        annotation_folder = dataset / "Synthetic_Match"
        annotation_folder.mkdir(parents=True)
        headers = ("rally", "time", "frame_num", "hit_x")
        synthetic = {
            1: (("7", "0:00:07", "50.0", "12"), ("7", "0:00:08", "60.0", "13"),
                ("11", "0:00:11", "90.0", "14")),
            2: (("7", "0:00:10", "80.0", "15"),),
        }
        for game, rows in synthetic.items():
            with (annotation_folder / f"set{game}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(headers)
                writer.writerows(rows)
        context = read_annotation_folder(annotation_folder)
        records = build_annotation_records(context, 10.0, 2.0, 1.0, 3.0, 30.0)
        by_key = {(row["game"], row["original_rally_id"]): row for row in records}
        assert set(by_key) == {("1", "7"), ("1", "11"), ("2", "7")}
        assert by_key[("1", "7")]["first_hit_frame"] == 50
        assert math.isclose(by_key[("1", "7")]["start_seconds"], 6.0)
        assert math.isclose(by_key[("1", "7")]["end_seconds"], 11.0)
        assert nearest_clip_frame(50 / 10.0 + 2.0, 6.0, 25.0, 125) == 25
        assert "padding-overlap" in by_key[("1", "7")]["review_flags"]
        assert "padding-overlap" in by_key[("1", "11")]["review_flags"]
        assert "hit-span-overlap" not in by_key[("1", "7")]["review_flags"]
        assert by_key[("2", "7")]["filename"] == "set2/rally_007.mp4"
        assert by_key[("1", "11")]["filename"] == "set1/rally_011.mp4"

        summary = Path(temporary) / "rallies.csv"
        write_csv_new(summary, records, annotation=True)
        reviewed = read_reviewed_csv(summary, 30.0, annotation=True)
        restored = restore_reviewed_annotation_records(reviewed, context, 10.0, 30.0)
        restored_ids = {(row["game"], row["original_rally_id"], row["filename"]) for row in restored}
        assert restored_ids == {
            ("1", "7", "set1/rally_007.mp4"),
            ("1", "11", "set1/rally_011.mp4"),
            ("2", "7", "set2/rally_007.mp4"),
        }
        assert restored[0]["_annotation_rows"]
        try:
            validate_intervals([valid[0], {"rally_id": "2", "start_seconds": "2.5", "end_seconds": "4"}], 10)
        except ValueError:
            pass
        else:
            raise AssertionError("legacy interval validation allowed an overlap")

    print("Self-test passed: smoothing/transient errors, buffered timestamps, EOF, legacy validation, annotation FPS/offset/padding, overlaps, repeated IDs/gaps, and reviewed identity restoration.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export candidate badminton rallies from SA-CNN or match annotations for manual review."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--weights", type=Path, default=HERE / "sacnn.pt")
    parser.add_argument("--output-root", type=Path, default=HERE.parent / "trimmed_rallies")
    parser.add_argument("--start", type=float, help="bounded source start time in seconds")
    parser.add_argument("--end", type=float, help="bounded source end time in seconds")
    parser.add_argument("--reviewed-csv", type=Path, help="export corrected intervals without inference")
    parser.add_argument("--annotation-folder", type=Path, help="one match folder containing set<N>.csv annotations")
    parser.add_argument("--annotation-fps", type=float, help="annotation frame rate; required with --annotation-folder")
    parser.add_argument("--time-offset", type=float, default=0.0, help="seconds added to frame-derived annotation times")
    parser.add_argument("--pre-padding", type=float, default=1.0, help="seconds before the first annotation hit")
    parser.add_argument("--post-padding", type=float, default=3.0, help="seconds after the last annotation hit")
    parser.add_argument("--overwrite", action="store_true", help="replace only generated files for selected rally identities")
    parser.add_argument("--self-test", action="store_true", help="run the small logic self-test and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    video = args.video.expanduser()
    if not video.is_file():
        raise FileNotFoundError(f"input video does not exist or is unreadable: {video}")
    metadata = probe_video(video)

    if args.annotation_folder is not None:
        if args.annotation_fps is None:
            raise ValueError("--annotation-fps is required with --annotation-folder")
        context = read_annotation_folder(args.annotation_folder)
        output_folder = args.output_root.expanduser() / context["dataset"] / context["match"]
        if args.reviewed_csv:
            reviewed = read_reviewed_csv(args.reviewed_csv, metadata["duration"], annotation=True)
            records = restore_reviewed_annotation_records(reviewed, context, args.annotation_fps, metadata["duration"])
        else:
            records = build_annotation_records(
                context, args.annotation_fps, args.time_offset, args.pre_padding,
                args.post_padding, metadata["duration"],
            )
        start, end = validate_bounds(args.start, args.end, metadata["duration"])
        selected, _ = select_annotation_records(records, start, end)
        output_folder.mkdir(parents=True, exist_ok=True)
        same_csv = preflight_annotation_output(output_folder, selected, args.overwrite, args.reviewed_csv)
        filenames = export_clips(video, output_folder, selected, metadata, args.overwrite, annotation=True)
        if not same_csv:
            write_csv_new(output_folder / "rallies.csv", selected, overwrite=args.overwrite, annotation=True)
        print(f"Exported {len(filenames)} annotation-led clips to {output_folder}")
        print("Padding creates reviewable boundaries; it does not guarantee serve-to-finish coverage. Inspect padding overlaps, hit-span overlaps, and timing flags.")
        return 0

    if args.annotation_fps is not None:
        raise ValueError("--annotation-fps requires --annotation-folder")
    if args.reviewed_csv and (args.start is not None or args.end is not None):
        raise ValueError("--start/--end apply only to inference or annotation mode; reviewed CSV timestamps are absolute")
    output_folder = output_folder_for(video, args.output_root.expanduser())

    if args.reviewed_csv:
        intervals = read_reviewed_csv(args.reviewed_csv, metadata["duration"])
        same_csv = preflight_output(output_folder, args.overwrite, args.reviewed_csv)
        output_folder.mkdir(parents=True, exist_ok=True)
        filenames = export_clips(video, output_folder, intervals, metadata, args.overwrite)
        if not same_csv:
            write_csv_new(output_folder / "rallies.csv", intervals)
        print(f"Exported {len(filenames)} reviewed clips to {output_folder}")
        return 0

    start, end = validate_bounds(args.start, args.end, metadata["duration"])
    if output_folder.exists():
        preflight_output(output_folder, args.overwrite)
    model, device = load_model(args.weights.expanduser())
    print(f"Using {device}; classifying {start:.3f}-{end:.3f}s at {SAMPLE_FPS} samples/second.")
    detections, sample_count = decode_predictions(video, metadata, start, end, model, device)
    rows = [
        {"rally_id": str(index), "start_seconds": first, "end_seconds": last, "filename": ""}
        for index, (first, last) in enumerate(detections, 1)
    ]
    intervals = validate_intervals(rows, metadata["duration"])
    output_folder.mkdir(parents=True, exist_ok=True)
    filenames = export_clips(video, output_folder, intervals, metadata, args.overwrite)
    write_csv_new(output_folder / "rallies.csv", intervals)
    print(f"Decoded {sample_count} samples; exported {len(filenames)} candidate clips and rallies.csv to {output_folder}")
    print("Review candidates manually: high-court views can include pauses/replays and can merge or miss rallies; no duration filter was applied.")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
