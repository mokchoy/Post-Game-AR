"""Batch annotation-led trimming for the set-1 roster and verified legacy files."""

import argparse
import contextlib
import csv
import importlib.util
import io
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
V2_SCRIPT = HERE / "trim_rallies_v2.py"
MATCH_CSV = PROJECT / "set-1" / "match.csv"
ANNOTATION_ROOT = PROJECT / "set-1"
DEFAULT_VIDEOS_DIR = Path(r"G:\My Drive\Match_Downloads_2")
DEFAULT_RAW_ROOT = PROJECT / "trimmed_rallies"
DEFAULT_CLEANED_ROOT = PROJECT / "cleaned_rallies"


def load_downloader():
    spec = importlib.util.spec_from_file_location("extract_videos", PROJECT / "extract_videos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_fps(match_id):
    if match_id == 12:
        return 30000 / 1001
    return 25 if match_id < 21 else 30


DOWNLOADER = load_downloader()
MATCHES = DOWNLOADER.read_matches(MATCH_CSV)
EXPECTED_IDS = {row["id"] for row in MATCHES}
VIDEO_MAP = {
    f"{row['video']}.mp4": (row["id"], row["video"], expected_fps(row["id"]))
    for row in MATCHES
}


def load_v2():
    spec = importlib.util.spec_from_file_location("trim_rallies_v2", V2_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_mapping():
    problems = []
    ids = [entry[0] for entry in VIDEO_MAP.values()]
    if len(VIDEO_MAP) != len(MATCHES) or set(ids) != EXPECTED_IDS or len(set(ids)) != len(ids):
        problems.append("mapping must contain every set-1/match.csv identity once")
    for filename, (match_id, folder, fps) in VIDEO_MAP.items():
        if fps != expected_fps(match_id):
            problems.append(f"{filename}: annotation FPS {fps} does not match ID {match_id} timing profile")
        if Path(filename).name != filename or not filename.lower().endswith(".mp4"):
            problems.append(f"invalid mapped video filename: {filename!r}")
        if not folder or Path(folder).name != folder:
            problems.append(f"invalid annotation folder for ID {match_id}: {folder!r}")
    try:
        with MATCH_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = {int(row["id"]): row["video"] for row in csv.DictReader(stream)}
        for filename, (match_id, folder, _) in VIDEO_MAP.items():
            if rows.get(match_id) != folder:
                problems.append(f"ID {match_id} maps to {folder!r}, but set-1/match.csv has {rows.get(match_id)!r}")
    except (OSError, KeyError, ValueError, csv.Error) as exc:
        problems.append(f"could not validate mapping against set-1/match.csv: {exc}")
    return problems


def match_folder(root, annotation_folder):
    return Path(root).expanduser() / "set-1" / annotation_folder


def generated_artifacts(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    artifacts = [folder / name for name in ("rallies.csv", "cleanup_report.csv") if (folder / name).is_file()]
    for pattern in ("set*/rally_*.mp4", "set*/rally_*.csv"):
        artifacts.extend(path for path in folder.glob(pattern) if path.is_file())
    return sorted(set(artifacts))


def build_v2_command(video, annotation_folder, fps, raw_root, cleaned_root):
    return [
        sys.executable, str(V2_SCRIPT),
        "--video", str(Path(video).resolve()),
        "--annotation-folder", str((ANNOTATION_ROOT / annotation_folder).resolve()),
        "--annotation-fps", str(fps),
        "--time-offset", "0",
        "--pre-padding", "1",
        "--post-padding", "3",
        "--raw-output-root", str(Path(raw_root).expanduser()),
        "--cleaned-output-root", str(Path(cleaned_root).expanduser()),
    ]


def scan_video_dir(videos_dir):
    folder = Path(videos_dir).expanduser()
    if not folder.is_dir():
        raise OSError(f"video directory does not exist or is unreadable: {folder}")
    files = list(folder.iterdir())
    videos = {path.name: path for path in files if path.is_file() and path.suffix.lower() == ".mp4"}
    unfinished = [path for path in files if path.is_file() and path.name.lower().endswith((".part", ".ytdl", ".partial", ".tmp"))]
    other_videos = [path for path in files if path.is_file() and path.suffix.lower() in (".webm", ".mkv", ".mov", ".avi")]
    matched, unknown = DOWNLOADER.resolve_videos(MATCHES, videos)
    selected = {path.name for path in matched.values()}
    for name in sorted(set(videos) - set(unknown) - selected, key=str.casefold):
        print(f"ADDITIONAL FILE FOR SAME MATCH (ignored; canonical preferred): {name}")
    resolved = {f"{row['video']}.mp4": matched[row["id"]] for row in MATCHES if row["id"] in matched}
    resolved.update({name: videos[name] for name in unknown})
    return resolved, unfinished, other_videos


def report_inventory(videos, unfinished, other_videos):
    for path in unfinished:
        print(f"UNFINISHED DOWNLOAD (ignored): {path.name}")
    for path in other_videos:
        print(f"NON-MP4 VIDEO (ignored): {path.name}")
    unknown = sorted((name for name in videos if name not in VIDEO_MAP), key=str.casefold)
    for name in unknown:
        print(f"UNKNOWN COMPLETED MP4 (unresolved): {name}")
    missing = sorted((name for name in VIDEO_MAP if name not in videos), key=str.casefold)
    for name in missing:
        match_id = VIDEO_MAP[name][0]
        row = next(row for row in MATCHES if row["id"] == match_id)
        note = "; no download URL in match.csv" if not row["url"] else ""
        print(f"MISSING MAPPED MP4: ID {match_id}: {name}{note}")
    return unknown, missing


def run_batch(args):
    try:
        videos, unfinished, other_videos = scan_video_dir(args.videos_dir)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    unknown, missing = report_inventory(videos, unfinished, other_videos)
    succeeded = skipped = failed = 0
    entries = sorted(VIDEO_MAP.items(), key=lambda item: item[1][0])
    for index, (filename, (match_id, folder, fps)) in enumerate(entries, 1):
        video = videos.get(filename)
        if video is None:
            continue
        cleaned_folder = match_folder(args.cleaned_output_root, folder)
        try:
            artifacts = generated_artifacts(cleaned_folder)
        except OSError as exc:
            print(f"[{index}/{len(entries)}] ID {match_id} FAILED checking cleaned outputs: {exc}", file=sys.stderr)
            failed += 1
            continue
        if artifacts:
            print(f"[{index}/{len(entries)}] ID {match_id} SKIPPED; existing cleaned artifacts found ({len(artifacts)}; may be partial): {cleaned_folder}")
            skipped += 1
            continue
        command = build_v2_command(video, folder, fps, args.raw_output_root, args.cleaned_output_root)
        print(f"[{index}/{len(entries)}] ID {match_id} annotation {fps} fps: {video.name}", flush=True)
        try:
            result = subprocess.run(command, check=False)
        except OSError as exc:
            print(f"  MATCH FAILED: {exc}", file=sys.stderr)
            failed += 1
            continue
        if result.returncode:
            print(f"  MATCH FAILED: v2 exited {result.returncode}", file=sys.stderr)
            failed += 1
        else:
            succeeded += 1
    print(
        f"Totals: mapped completed={len(videos) - len(unknown)}, processed successfully={succeeded}, "
        f"skipped for existing cleaned artifacts={skipped}, failed={failed}, "
        f"unknown completed={len(unknown)}, missing mapped={len(missing)}, "
        f"unfinished downloads={len(unfinished)}"
    )
    return 2 if failed or unknown or missing else 0


def check_batch(args):
    print("Read-only batch check (no outputs or video encoding)")
    mapping_problems = validate_mapping()
    for problem in mapping_problems:
        print(f"MAPPING ERROR: {problem}")
    if not mapping_problems:
        print(f"Mapping OK: {len(VIDEO_MAP)} roster matches; canonical names and {len(DOWNLOADER.LEGACY_FILENAMES)} verified legacy aliases; shared 2019 title maps only to ID 1. Annotation FPS: ID 12=30000/1001, other IDs below 21=25, IDs 21-44=30.")

    try:
        videos, unfinished, other_videos = scan_video_dir(args.videos_dir)
    except OSError as exc:
        print(f"VIDEO DIRECTORY ERROR: {exc}")
        videos, unfinished, other_videos = {}, [], []
        directory_error = True
    else:
        directory_error = False
    unknown, missing = report_inventory(videos, unfinished, other_videos)

    v2 = None
    try:
        v2 = load_v2()
    except Exception as exc:
        print(f"PREREQUISITE ERROR: could not load v2 helpers: {exc}")
    ffprobe_ok = bool(shutil.which("ffprobe"))
    ffmpeg_ok = bool(shutil.which("ffmpeg"))
    prerequisites_ok = ffprobe_ok and ffmpeg_ok and v2 is not None
    if not ffprobe_ok:
        print("PREREQUISITE MISSING: ffprobe is not on PATH; source metadata could not be checked.")
    if not ffmpeg_ok:
        print("PREREQUISITE MISSING: ffmpeg is not on PATH; v2 cannot export clips.")
    if v2 is not None:
        try:
            v2.load_scene_detector()
        except Exception as exc:
            print(f"PREREQUISITE MISSING: {exc} (Python {sys.version.split()[0]}; reported, diagnostics continue).")
            prerequisites_ok = False
    else:
        print("PREREQUISITE MISSING: v2 annotation and metadata helpers are unavailable; diagnostics continue where possible.")

    metadata_ok = annotations_ok = 0
    diagnostics_failed = directory_error or bool(mapping_problems) or not prerequisites_ok
    for filename, (match_id, folder, fps) in sorted(VIDEO_MAP.items(), key=lambda item: item[1][0]):
        video = videos.get(filename)
        source_metadata = None
        if video is not None and ffprobe_ok and v2 is not None:
            try:
                source_metadata = v2.probe_video(video)
                metadata_ok += 1
                print(
                    f"METADATA OK: ID {match_id} {source_metadata['width']}x{source_metadata['height']}, "
                    f"source avg/rate={source_metadata['avg_rate']}/{source_metadata['nominal_rate']} fps, "
                    f"duration={source_metadata['duration']:.3f}s, audio={'yes' if source_metadata['has_audio'] else 'no'}; "
                    f"annotation fps={fps}"
                )
                if source_metadata["avg_rate"] != source_metadata["nominal_rate"]:
                    print(f"SOURCE TIMING ERROR: ID {match_id} average and nominal rates differ; v2 requires CFR.")
                    diagnostics_failed = True
            except Exception as exc:
                print(f"SOURCE METADATA ERROR: ID {match_id} {filename}: {exc}")
                diagnostics_failed = True
        annotation_folder = ANNOTATION_ROOT / folder
        if not annotation_folder.is_dir():
            print(f"ANNOTATION ERROR: ID {match_id} folder missing: {annotation_folder}")
            diagnostics_failed = True
        elif v2 is not None:
            try:
                warning_output = io.StringIO()
                duration = source_metadata["duration"] if source_metadata else math.inf
                with contextlib.redirect_stderr(warning_output):
                    records = v2.read_annotations(annotation_folder, fps, 0.0, 1.0, 3.0, duration)
                annotations_ok += 1
                warning_count = len(warning_output.getvalue().splitlines())
                bound_note = "" if source_metadata else "; source-duration bounds not checked"
                print(f"ANNOTATIONS OK: ID {match_id} {len(records)} rally records, {sum(len(r['_rows']) for r in records)} rows, {len(list(annotation_folder.glob('set*.csv')))} set CSVs{bound_note}; timing offset=0, padding=1s/3s; coarse-time warnings={warning_count}")
            except Exception as exc:
                print(f"ANNOTATION ERROR: ID {match_id} {annotation_folder}: {exc}")
                diagnostics_failed = True

        clean_dir = match_folder(args.cleaned_output_root, folder)
        try:
            artifacts = generated_artifacts(clean_dir)
            if artifacts:
                print(f"CLEANED OUTPUT SKIP: ID {match_id}; existing artifacts ({len(artifacts)}; may be partial): {clean_dir}")
            else:
                print(f"CLEANED OUTPUT CLEAR: ID {match_id}: {clean_dir}")
        except OSError as exc:
            print(f"CLEANED OUTPUT ERROR: ID {match_id}: {exc}")
            diagnostics_failed = True
        raw_dir = match_folder(args.raw_output_root, folder)
        try:
            raw_artifacts = generated_artifacts(raw_dir)
            if raw_artifacts:
                print(f"RAW OUTPUT PRESENT: ID {match_id}; v2 will validate/reuse existing artifacts ({len(raw_artifacts)}): {raw_dir}")
        except OSError as exc:
            print(f"RAW OUTPUT ERROR: ID {match_id}: {exc}")
            diagnostics_failed = True

    print(
        f"Check totals: mapped completed={len(videos) - len(unknown)}, unknown completed={len(unknown)}, "
        f"missing mapped={len(missing)}, unfinished={len(unfinished)}, other video files={len(other_videos)}, "
        f"source metadata checked={metadata_ok}, annotation folders checked={annotations_ok}, "
        f"prerequisites reported; cleaned collisions are skips."
    )
    return 2 if diagnostics_failed or unknown or missing else 0


def self_test():
    ids = [entry[0] for entry in VIDEO_MAP.values()]
    assert len(VIDEO_MAP) == len(MATCHES) and set(ids) == EXPECTED_IDS and len(set(ids)) == len(ids)
    assert not validate_mapping()
    assert 3 in ids and 12 in ids and set(range(39, 45)).issubset(ids)
    assert 3 not in DOWNLOADER.LEGACY_FILENAMES
    assert expected_fps(12) == 30000 / 1001
    assert all(fps == expected_fps(match_id) for match_id, _, fps in VIDEO_MAP.values())
    spaced = next(folder for match_id, folder, _ in VIDEO_MAP.values() if match_id == 14)
    assert " " in spaced

    with tempfile.TemporaryDirectory(prefix=".v3-self-test-", dir=HERE) as temp:
        root = Path(temp)
        videos_dir, raw_root, clean_root = root / "videos", root / "raw", root / "cleaned"
        videos_dir.mkdir()
        args = SimpleNamespace(videos_dir=videos_dir, raw_output_root=raw_root, cleaned_output_root=clean_root)

        complete_name = next(name for name, value in VIDEO_MAP.items() if value[0] == 1)
        partial_name = next(name for name, value in VIDEO_MAP.items() if value[0] == 14)
        (videos_dir / complete_name).touch()
        (videos_dir / partial_name).touch()
        legacy_name = DOWNLOADER.LEGACY_FILENAMES[1]
        (videos_dir / legacy_name).touch()
        inventory, _, _ = scan_video_dir(videos_dir)
        assert inventory[complete_name] == videos_dir / complete_name
        assert len(inventory) == 2
        (videos_dir / complete_name).unlink()
        inventory, _, _ = scan_video_dir(videos_dir)
        assert inventory[complete_name] == videos_dir / legacy_name
        korea_name = next(name for name, value in VIDEO_MAP.items() if value[0] == 3)
        assert korea_name not in inventory
        complete_dir = match_folder(clean_root, VIDEO_MAP[complete_name][1])
        complete_dir.mkdir(parents=True)
        (complete_dir / "rallies.csv").touch()
        (complete_dir / "cleanup_report.csv").touch()
        partial_dir = match_folder(clean_root, VIDEO_MAP[partial_name][1]) / "set1"
        partial_dir.mkdir(parents=True)
        (partial_dir / "rally_001.mp4").touch()
        with patch("subprocess.run") as mocked, contextlib.redirect_stdout(io.StringIO()) as output:
            run_batch(args)
        assert mocked.call_count == 0
        assert "ID 1 SKIPPED" in output.getvalue() and "ID 14 SKIPPED" in output.getvalue()
        assert len(generated_artifacts(complete_dir)) == 2
        assert len(generated_artifacts(partial_dir.parent)) == 1

        clean_root = root / "cleaned-later"
        args.cleaned_output_root = clean_root
        (videos_dir / legacy_name).unlink()
        (videos_dir / partial_name).unlink()
        ordered = [(name, value) for name, value in sorted(VIDEO_MAP.items(), key=lambda item: item[1][0]) if value[0] in (1, 2, 3, 12, 14, 39, 44)]
        for name, _ in ordered:
            (videos_dir / name).touch()
        id4_name = next(name for name, value in VIDEO_MAP.items() if value[0] == 4)
        (videos_dir / f"{id4_name}.part").touch()
        (videos_dir / "Unknown match.mp4").touch()
        mocked_results = [SimpleNamespace(returncode=2)] + [SimpleNamespace(returncode=0)] * (len(ordered) - 1)
        with patch("subprocess.run", side_effect=mocked_results) as mocked, contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            status = run_batch(args)
        assert status == 2 and mocked.call_count == len(ordered)
        assert f"UNFINISHED DOWNLOAD (ignored): {id4_name}.part" in output.getvalue()
        assert "UNKNOWN COMPLETED MP4 (unresolved): Unknown match.mp4" in output.getvalue()
        expected_calls = [
            build_v2_command(videos_dir / name, folder, fps, raw_root, clean_root)
            for name, (match_id, folder, fps) in ordered
        ]
        assert [call.args[0] for call in mocked.call_args_list] == expected_calls
        assert all(call.kwargs == {"check": False} for call in mocked.call_args_list)
        assert all("--overwrite" not in call.args[0] for call in mocked.call_args_list)
        spaced_command = next(command for command in expected_calls if command[command.index("--annotation-folder") + 1].endswith(spaced))
        assert spaced_command[spaced_command.index("--annotation-fps") + 1] == "25"
        assert expected_calls[-1][expected_calls[-1].index("--annotation-fps") + 1] == "30"
        assert "Unknown match.mp4" not in [Path(call.args[0][call.args[0].index("--video") + 1]).name for call in mocked.call_args_list]
        assert not any(Path(call.args[0][call.args[0].index("--video") + 1]).name == f"{id4_name}.part" for call in mocked.call_args_list)

        v2 = load_v2()
        metadata = {"width": 1920, "height": 1080, "avg_rate": 25, "nominal_rate": 25,
                    "duration": 100000, "has_audio": True}
        with (
            patch("subprocess.run") as mocked,
            patch("shutil.which", return_value="tool"),
            patch.object(v2, "probe_video", return_value=metadata),
            patch.object(v2, "load_scene_detector", side_effect=RuntimeError("test missing scene detector")),
            patch.dict(check_batch.__globals__, {"load_v2": lambda: v2}),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = check_batch(args)
        assert status == 2 and not mocked.called
        assert "PREREQUISITE MISSING: test missing scene detector" in output.getvalue()
        assert f"annotation folders checked={len(MATCHES)}" in output.getvalue()

    print("Self-test passed: full roster/FPS, canonical and legacy names, ID 1/3 separation, duplicate-file preference, spaced paths, unfinished/unknown exclusion, complete/partial output skips, exact v2 arguments, continuation after failure, and read-only prerequisite reporting.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--cleaned-output-root", type=Path, default=DEFAULT_CLEANED_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only input, prerequisite, annotation, metadata, and collision diagnostics")
    mode.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.check:
        problems = validate_mapping()
        if problems:
            for problem in problems:
                print(f"MAPPING ERROR: {problem}", file=sys.stderr)
            return 2
    return check_batch(args) if args.check else run_batch(args)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
