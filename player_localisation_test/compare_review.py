"""Matched saved-review comparison; --check never runs model inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

import runner as detector

ROOT = Path(__file__).resolve().parent


def load_keypoint_helpers():
    # Both experiments have a runner.py; load the sibling under a distinct name.
    name = "_comparison_keypoint_review"
    if name not in sys.modules:
        folder = ROOT.parent / "player_keypoint_test"
        spec = importlib.util.spec_from_file_location(name, folder / "runner.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        previous = {key: os.environ.get(key) for key in ("TEMP", "TMP", "YOLO_CONFIG_DIR")}
        previous_path = sys.path[:]
        sys.path.insert(0, str(folder))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = previous_path
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return sys.modules[name]


kp = load_keypoint_helpers()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_selection(selection, info, config, checkpoint, split):
    if (selection["checkpoint_sha256"] != info["last_sha256"]
            or Path(selection["checkpoint"]).resolve() != checkpoint.resolve()):
        raise ValueError("Saved selection checkpoint differs from the completed final run")
    if (selection["dataset_identity"] != split["dataset_identity"]
            or info.get("test_used_for_training") is not False
            or set(selection["held_out_matches"]) != set(split["held_out_matches"])
            or set(info["held_out_matches"]) != set(split["held_out_matches"])
            or set(info["development_matches"]) != set(split["development_matches"])):
        raise ValueError("Selection/final-run/frozen-split provenance mismatch")
    if selection["resolved_settings"] != kp.resolved_settings(config):
        raise ValueError("Selection settings differ from the saved final-run settings")
    keys = [tuple(item["key"]) for item in selection["selected_rallies"]]
    if (len(keys) != 3 or len({key[1] for key in keys}) != 3
            or any(len(key) != 4 or key[0] != "set-1"
                   or key[1] not in split["held_out_matches"] for key in keys)
            or selection.get("selection_saved_before_inference") is not True):
        raise ValueError("Expected three distinct, previously selected held-out rallies")


def validate_smoothing(settings):
    for field, minimum in (("median_window", 1), ("savgol_window", 3)):
        value = settings[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value % 2 != 1:
            raise ValueError(f"Invalid saved smoothing {field}")
    constants = {"polyorder": 2, "court_margin_m": 3.0, "half_margin_m": 1.0,
                 "median_outlier_m": 1.0, "max_short_gap_seconds": 0.2}
    if (set(settings) != {*constants, "median_window", "savgol_window", "gap_policy"}
            or settings["gap_policy"] not in ("continuous", "conservative")
            or any(settings[key] != value for key, value in constants.items())):
        raise ValueError("Saved smoothing settings are incompatible with the current helper")


def read_positions(path, rally):
    rows = kp.read_csv(path)
    seen = set()
    for row in rows:
        frame, player = int(row["clip_frame_num"]), row["player"]
        if kp.stable_key(row) != rally["key"] or (frame, player) in seen:
            raise ValueError(f"Wrong identity or duplicate position in {path}")
        seen.add((frame, player))
        if row["available"] not in ("True", "False"):
            raise ValueError(f"Invalid availability flag in {path}")
        row.update(clip_frame_num=frame, clip_time_seconds=float(row["clip_time_seconds"]),
                   available=row["available"] == "True")
        if "gap_filled" in row:
            if row["gap_filled"] not in ("True", "False"):
                raise ValueError(f"Invalid gap flag in {path}")
            row["gap_filled"] = row["gap_filled"] == "True"
        for field in ("x_native", "y_native", "x_court_m", "y_court_m",
                      "reference_x_native", "reference_y_native", "heatmap_score"):
            if field in row:
                row[field] = float(row[field]) if row[field] else None
        if (player not in ("A", "B") or row["court_half"] != rally["halves"][player]
                or not math.isclose(row["clip_time_seconds"], frame / rally["fps"], abs_tol=1e-6)
                or (row["prediction_status"] == "available") != row["available"]):
            raise ValueError(f"Position timing, status or court-half mismatch in {path}")
        if row["available"]:
            values = [row[key] for key in ("x_native", "y_native", "x_court_m", "y_court_m")]
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise ValueError(f"Nonfinite available position in {path}")
            world = kp.projected_world(rally["geometry"], values[:2], *rally["dimensions"])
            if world is None or not np.allclose(world, values[2:], rtol=1e-7, atol=1e-7):
                raise ValueError(f"Position projection differs from current geometry in {path}")
        if frame in rally["annotations"]:
            reference = rally["annotations"][frame][player]["image"]
            if not np.allclose([row["reference_x_native"], row["reference_y_native"]], reference):
                raise ValueError(f"Saved reference differs from the frozen manifest in {path}")
    expected = {(frame, player) for frame in range(rally["frames"]) for player in ("A", "B")}
    if seen != expected:
        raise ValueError(f"Position rows do not cover every source frame exactly once in {path}")
    return sorted(rows, key=lambda row: (row["clip_frame_num"], row["player"]))


def score_positions(positions, annotations):
    errors = []
    for row in positions:
        frame, player = row["clip_frame_num"], row["player"]
        if frame not in annotations:
            continue
        reference = annotations[frame][player]
        available = row["available"] and reference["world"] is not None
        errors.append({**row, "available": available,
                       "prediction_status": "available" if available else "missing",
                       "error_px": math.dist((row["x_native"], row["y_native"]), reference["image"])
                       if available else None,
                       "error_m": math.dist((row["x_court_m"], row["y_court_m"]), reference["world"])
                       if available else None})
    if len(errors) != 2 * len(annotations):
        raise ValueError("Annotation-time error rows are missing")
    return errors


def validate_saved_errors(path, computed):
    saved = kp.read_csv(path)
    index = {(int(row["clip_frame_num"]), row["player"]): row for row in saved}
    if len(index) != len(saved) or set(index) != {(r["clip_frame_num"], r["player"]) for r in computed}:
        raise ValueError(f"Saved error frame/player keys differ from frozen references: {path}")
    for row in computed:
        other = index[row["clip_frame_num"], row["player"]]
        if (kp.stable_key(other) != kp.stable_key(row)
                or other["available"] != str(row["available"])):
            raise ValueError(f"Saved error identity/availability mismatch: {path}")
        for field in ("error_px", "error_m"):
            value = float(other[field]) if other[field] else None
            if ((value is None) != (row[field] is None)
                    or (value is not None and not math.isclose(value, row[field], rel_tol=1e-7, abs_tol=1e-7))):
                raise ValueError(f"Saved errors do not match current references/projection: {path}")


def smooth_positions(positions, rally, settings):
    filtered = []
    width, height = rally["dimensions"]
    for player in ("A", "B"):
        source = [row for row in positions if row["player"] == player]
        points = [[row["x_court_m"], row["y_court_m"]] if row["available"] else [np.nan, np.nan]
                  for row in source]
        points, statuses, reasons = kp.smooth_court_track(points, rally["halves"][player],
                                                        rally["fps"], settings)
        for raw, world, status, reason in zip(source, points, statuses, reasons, strict=True):
            image = (kp.native_point(rally["geometry"].world_to_annotation_floor(world), width, height)
                     if np.isfinite(world).all() else (np.nan, np.nan))
            available = bool(np.isfinite(image).all() and np.isfinite(world).all())
            filtered.append({**raw, "available": available,
                             "prediction_status": "available" if available else "unavailable_after_smoothing",
                             "x_native": float(image[0]) if available else None,
                             "y_native": float(image[1]) if available else None,
                             "x_court_m": float(world[0]) if available else None,
                             "y_court_m": float(world[1]) if available else None,
                             "filter_status": str(status) if available else "unavailable",
                             "rejection_reason": str(reason),
                             "gap_filled": available and status not in ("smoothed", "unavailable")})
    return sorted(filtered, key=lambda row: (row["clip_frame_num"], row["player"]))


def validate_smoothed(saved, computed):
    for other, row in zip(saved, computed, strict=True):
        if any(other[field] != row[field] for field in
               ("clip_frame_num", "player", "available", "gap_filled", "filter_status", "rejection_reason")):
            raise ValueError("Saved smoothing status differs from recomputed smoothing")
        if row["available"] and not np.allclose(
                [other[key] for key in ("x_native", "y_native", "x_court_m", "y_court_m")],
                [row[key] for key in ("x_native", "y_native", "x_court_m", "y_court_m")],
                rtol=1e-7, atol=1e-7):
            raise ValueError("Saved smoothed coordinates differ from the saved settings/current helper")


def load_context(review_dir):
    review_dir = Path(review_dir).resolve(strict=True)
    selection = read_json(review_dir / "selection.json")
    info, config, checkpoint = kp.final_context(selection["final_run_id"])
    parent = (kp.ROOT / "runs" / selection["final_run_id"] / "selected_rally_reviews").resolve()
    if not review_dir.is_relative_to(parent):
        raise ValueError("Review directory must belong to the selected final run")
    split = read_json(kp.SPLIT_FILE)
    manifest_path = kp.PREP / "frozen_manifest.jsonl"
    rows = kp.read_jsonl(manifest_path)
    kp.validate_manifest(rows, require_images=False)
    if split.get("split_version") != kp.SPLIT_VERSION or kp.dataset_identity(rows) != split["dataset_identity"]:
        raise ValueError("Frozen manifest does not match the saved split")
    validate_selection(selection, info, config, checkpoint, split)
    settings = read_json(review_dir / "review_smoothing.json")
    validate_smoothing(settings)
    groups = defaultdict(list)
    for row in rows:
        groups[kp.stable_key(row)].append(row)
    artifacts = [review_dir / "selection.json", review_dir / "review_smoothing.json", manifest_path,
                 kp.SPLIT_FILE, kp.HOMOGRAPHY, kp.ROOT / "runs" / selection["final_run_id"] / "final.json",
                 Path(kp.__file__), Path(kp.pm.__file__), Path(detector.__file__), Path(__file__),
                 ROOT / "comparison_plots.py"]
    rallies = []
    for item in selection["selected_rallies"]:
        key = tuple(item["key"])
        frozen = sorted(groups[key], key=lambda row: int(row["clip_frame_num"]))
        if not frozen or len({int(row["clip_frame_num"]) for row in frozen}) != len(frozen):
            raise ValueError(f"Missing or duplicate frozen references: {key}")
        template = frozen[0]
        path = Path(item["clip_path"]).resolve(strict=True)
        if (not path.is_relative_to(kp.SOURCE.resolve())
                or any(Path(row["clip_path"]).resolve() != path for row in frozen)):
            raise ValueError(f"Selected clip differs from frozen provenance: {key}")
        metadata = detector.video_metadata(path)
        dimensions = [int(template["width"]), int(template["height"])]
        fps = float(template["fps"])
        halves = {player: template[f"{player}_half"] for player in ("A", "B")}
        if (dimensions != [metadata["width"], metadata["height"]]
                or not math.isclose(fps, metadata["fps"], abs_tol=1e-6)
                or any([int(r["width"]), int(r["height"])] != dimensions
                       or float(r["fps"]) != fps or any(r[f"{p}_half"] != halves[p] for p in halves)
                       or not math.isclose(float(r["clip_time_seconds"]), int(r["clip_frame_num"]) / fps,
                                           abs_tol=1e-6) for r in frozen)):
            raise ValueError(f"Source dimensions/FPS or frozen mapping changed: {key}")
        geometry = kp.import_geometry(key[1])
        annotations = kp.annotations_by_frame(frozen, geometry, *dimensions)
        if (min(annotations) < 0 or max(annotations) >= metadata["frame_count"]
                or any(ref["world"] is None for refs in annotations.values() for ref in refs.values())):
            raise ValueError(f"Invalid annotation frame or court projection: {key}")
        rally = {"key": key, "id": kp.safe_id(key), "clip_path": str(path), "rows": frozen,
                 "dimensions": dimensions, "fps": fps, "frames": metadata["frame_count"],
                 "halves": halves, "geometry": geometry, "annotations": annotations}
        saved_dir = review_dir / rally["id"]
        predictions = {}
        for variant in ("raw", "smoothed"):
            suffix = "" if variant == "raw" else "_smoothed"
            metrics_path = saved_dir / f"metrics{suffix}.json"
            metrics = read_json(metrics_path)
            if (metrics["source_dimensions"] != dimensions or metrics["decoded_frames"] != rally["frames"]
                    or not math.isclose(metrics["source_fps"], fps, abs_tol=1e-6)
                    or metrics["annotated_frames"] != sorted(annotations)
                    or (variant == "smoothed" and metrics["smoothing"] != settings)):
                raise ValueError(f"Saved review metadata/smoothing mismatch: {metrics_path}")
            position_path, error_path = (saved_dir / f"player_positions{suffix}.csv",
                                         saved_dir / f"shot_errors{suffix}.csv")
            predictions[variant] = read_positions(position_path, rally)
            errors = score_positions(predictions[variant], annotations)
            validate_saved_errors(error_path, errors)
            computed = kp.error_summary(errors)
            if any(metrics["combined"][name] != computed[name] for name in
                   ("expected_predictions", "available_predictions", "missing_predictions",
                    "coverage", "within_0_5_m", "within_1_m")):
                raise ValueError(f"Saved coverage/success counts differ: {metrics_path}")
            for name in ("pixel_error", "metre_error"):
                for stat, value in computed[name].items():
                    saved = metrics["combined"][name][stat]
                    # CSV float round-trips can change the last bit of error statistics.
                    if ((saved is None) != (value is None) or (value is not None
                            and not math.isclose(saved, value, rel_tol=1e-7, abs_tol=1e-7))):
                        raise ValueError(f"Saved aggregate errors differ: {metrics_path}")
            artifacts.extend((position_path, error_path, metrics_path))
        validate_smoothed(predictions["smoothed"], smooth_positions(predictions["raw"], rally, settings))
        rally["keypoint"] = predictions
        rallies.append(rally)
        artifacts.append(path)
    digest = detector.validate_checkpoint()
    # Architecture/checkpoint compatibility only: no forward pass and no weight download.
    model, _ = kp.pm.load_checkpoint(checkpoint)
    del model
    return {"review_dir": str(review_dir), "selection": selection, "settings": settings,
            "config": config, "checkpoint": checkpoint, "detector_sha256": digest,
            "rallies": rallies, "artifact_hashes": {str(path): detector.sha256(path) for path in artifacts}}


def detector_prediction(model, frame, device, rally):
    width, height = rally["dimensions"]
    selected = detector.select_candidates(detector.inference(model, frame, device), rally["geometry"],
                                         (width, height), project=lambda point: kp.projected_world(
                                             rally["geometry"], point, width, height))
    return {player: selected[rally["halves"][player]] for player in ("A", "B")}


def keypoint_prediction(model, frame, device, rally, config):
    width, height = rally["dimensions"]
    iw, ih = config["model"]["input_width"], config["model"]["input_height"]
    logits = model(kp.pm.image_tensor(frame, iw, ih).unsqueeze(0).to(device))[0]
    decoded = kp.pm.decode_heatmaps(logits, iw, ih, width, height)
    result = {}
    for player in ("A", "B"):
        point = decoded[kp.pm.CHANNELS[rally["halves"][player]]]
        pixel = (point["x_native"], point["y_native"])
        result[player] = {"bottom": pixel, "court": kp.projected_world(rally["geometry"], pixel, width, height),
                          "heatmap_score": point["heatmap_score"]}
    return result


def position_rows(prediction, frame, rally):
    rows = []
    for player in ("A", "B"):
        candidate = prediction[player]
        available = candidate is not None and candidate["court"] is not None
        point = candidate["bottom"] if candidate else (None, None)
        world = candidate["court"] if available else (None, None)
        reference = rally["annotations"].get(frame, {}).get(player, {}).get("image", (None, None))
        box = candidate.get("box", (None,) * 4) if candidate else (None,) * 4
        rows.append({**dict(zip(kp.KEY_FIELDS, rally["key"])), "clip_frame_num": frame,
                     "clip_time_seconds": frame / rally["fps"], "player": player,
                     "prediction_status": "available" if available else "missing", "available": available,
                     "x_native": point[0], "y_native": point[1], "x_court_m": world[0], "y_court_m": world[1],
                     "court_half": rally["halves"][player], "score": candidate.get("score") if candidate else None,
                     "box_x1_px": box[0], "box_y1_px": box[1], "box_x2_px": box[2], "box_y2_px": box[3],
                     "reference_x_native": reference[0], "reference_y_native": reference[1]})
    return rows


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(predict, rally, device):
    capture = cv2.VideoCapture(rally["clip_path"])
    positions, timings = [], []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {rally['clip_path']}")
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = len(timings)
                if index == 0:
                    for _ in range(10):
                        predict(frame)
                    synchronize(device)
                synchronize(device)
                start = perf_counter()
                prediction = predict(frame)
                synchronize(device)
                elapsed = perf_counter() - start
                timings.append({"rally_id": rally["id"], "clip_frame_num": index,
                                "milliseconds": elapsed * 1000})
                positions.extend(position_rows(prediction, index, rally))
                if index % 250 == 0:
                    print(f"  {rally['id']} frame {index}/{rally['frames']}", flush=True)
    finally:
        capture.release()
    if len(timings) != rally["frames"]:
        raise RuntimeError(f"Source decoded {len(timings)} frames; expected {rally['frames']}")
    return positions, timings


def summarize(errors, positions):
    result = detector.metrics_summary(errors)
    for name, subset in [("combined", positions), *[(p, [r for r in positions if r["player"] == p])
                                                   for p in ("A", "B")]]:
        metrics = result["combined"] if name == "combined" else result["per_player"][name]
        metrics.update(position_rows=len(subset),
                       position_coverage=sum(r["available"] for r in subset) / len(subset) if subset else None,
                       filled_points=sum(r.get("gap_filled", False) for r in subset),
                       filter_counts={status: sum(r.get("filter_status") == status for r in subset)
                                      for status in sorted({r["filter_status"] for r in subset if "filter_status" in r})})
    return result


def timing_summary(rows):
    values = np.asarray([row["milliseconds"] for row in rows])
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Expected positive finite frame timings")
    return {"frames": len(values), "total_seconds": float(values.sum() / 1000),
            "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "fps": float(1000 / values.mean())}


def render_positions(rally, positions, path, smoothed=False):
    width, height = rally["dimensions"]
    capture = cv2.VideoCapture(rally["clip_path"])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), rally["fps"], (width + 600, height))
    by_frame = defaultdict(dict)
    for row in positions:
        by_frame[row["clip_frame_num"]][row["player"]] = row
    previous, index = {}, 0
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError(f"Could not open review source/writer: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= rally["frames"]:
                raise RuntimeError("Source frame count changed during rendering")
            current = {}
            for player, row in by_frame[index].items():
                color = kp.REVIEW_COLORS[player]
                if not row["available"]:
                    cv2.putText(frame, f"{player}: unavailable", (24, 110 if player == "A" else 145),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                    continue
                current[player] = (row["x_court_m"], row["y_court_m"])
                x, y = row["x_native"], row["y_native"]
                if not smoothed and row.get("box_x1_px") is not None:
                    box = [int(round(row[field])) for field in ("box_x1_px", "box_y1_px", "box_x2_px", "box_y2_px")]
                    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
                if 0 <= x < width and 0 <= y < height:
                    point = (int(round(x)), int(round(y)))
                    cv2.circle(frame, point, 8, color, 2 if row.get("gap_filled") else -1)
                    label = f"{player} {row.get('filter_status', 'raw')}"
                    cv2.putText(frame, label, (max(10, point[0]+10), max(24, point[1]-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            refs = {p: ref["world"] for p, ref in rally["annotations"].get(index, {}).items()}
            panel = kp.court_panel(height, current, refs, previous, index, rally["halves"])
            cv2.putText(frame, f"frame {index} | detector {'smoothed' if smoothed else 'raw'}", (24, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
            writer.write(np.hstack((frame, panel)))
            previous, index = current, index + 1
    finally:
        capture.release()
        writer.release()
    return kp.verify_review_video(path, rally["frames"], rally["fps"], (width + 600, height))


def run_comparison(context):
    device = kp.resolve_device(context["config"])
    detector.check_torchvision_ops(device)
    output = detector.fresh_output_dir()
    provenance = {"source_review": context["review_dir"], "selection": context["selection"],
                  "smoothing": context["settings"], "artifact_sha256": context["artifact_hashes"],
                  "checkpoints": {"detector": {"path": str(detector.CHECKPOINT), "sha256": context["detector_sha256"]},
                                  "keypoint": {"path": str(context["checkpoint"]),
                                               "sha256": context["selection"]["checkpoint_sha256"]}},
                  "runtime": {**kp.run_dependencies(), "python": platform.python_version(),
                              "device": str(device), "device_name": torch.cuda.get_device_name(device)
                              if device.type == "cuda" else platform.processor(),
                              "precision": "FP32, no autocast", "batch_size": 1, "warmup_calls_per_clip": 10,
                              "timing_scope": "BGR frame to CPU court positions; excludes decode/load/render/encode/smoothing",
                              "model_order": ["detector", "keypoint"], "torch_threads": torch.get_num_threads()},
                  "detector_settings": {"confidence_threshold": detector.CONFIDENCE_THRESHOLD,
                                        "court_margin_m": detector.MAX_COURT_MARGIN_M,
                                        "transform": "official COCO_V1 internal resize/normalization",
                                        "image_min_size": 800, "image_max_size": 1333}}
    kp.write_json(output / "provenance.json", provenance)
    kp.write_json(output / "references.json", [{"key": list(r["key"]), "halves": r["halves"],
                  "dimensions": r["dimensions"], "fps": r["fps"], "frames": r["frames"], "rows": r["rows"],
                  "homography": {"row_id": r["geometry"].row_id,
                                 "annotation_to_dataset": r["geometry"].homography_annotation_to_dataset.tolist(),
                                 "dataset_to_metric": r["geometry"].dataset_to_metric.tolist()}}
                  for r in context["rallies"]])
    records, timings, smoothing_times = {}, [], []
    for method in ("detector", "keypoint"):
        print(f"Benchmarking {method} on {device}", flush=True)
        model = (detector.load_pretrained_model(device, context["detector_sha256"]) if method == "detector"
                 else kp.pm.load_checkpoint(context["checkpoint"])[0].to(device))
        model.float().eval()
        for rally in context["rallies"]:
            predict = (lambda frame: detector_prediction(model, frame, device, rally)) if method == "detector" else (
                lambda frame: keypoint_prediction(model, frame, device, rally, context["config"]))
            fresh, measured = benchmark(predict, rally, device)
            timings.extend({"method": method, **row} for row in measured)
            raw = fresh if method == "detector" else rally["keypoint"]["raw"]
            start = perf_counter()
            smoothed = smooth_positions(raw, rally, context["settings"])
            seconds = perf_counter() - start
            smoothing_times.append({"method": method, "rally_id": rally["id"], "seconds": seconds,
                                    "frames": rally["frames"]})
            if method == "keypoint":
                validate_smoothed(rally["keypoint"]["smoothed"], smoothed)
                smoothed = rally["keypoint"]["smoothed"]
            for variant, positions in (("raw", raw), ("smoothed", smoothed)):
                errors = score_positions(positions, rally["annotations"])
                records[method, variant, rally["id"]] = {"positions": positions, "errors": errors}
        del model, predict
        if device.type == "cuda":
            torch.cuda.empty_cache()
    kp.write_csv(output / "timings.csv", timings, ["method", "rally_id", "clip_frame_num", "milliseconds"])
    kp.write_csv(output / "smoothing_times.csv", smoothing_times, ["method", "rally_id", "seconds", "frames"])
    results = {"diagnostic_scope": "Three selected held-out rallies; not the full held-out test score",
               "provenance": provenance, "methods": {}}
    for method in ("detector", "keypoint"):
        results["methods"][method] = {}
        for variant in ("raw", "smoothed"):
            entries = [records[method, variant, r["id"]] for r in context["rallies"]]
            summary = summarize([row for e in entries for row in e["errors"]],
                                [row for e in entries for row in e["positions"]])
            summary["per_rally"] = {r["id"]: summarize(records[method, variant, r["id"]]["errors"],
                                                        records[method, variant, r["id"]]["positions"])
                                    for r in context["rallies"]}
            results["methods"][method][variant] = summary
        measured = [r for r in timings if r["method"] == method]
        results["methods"][method]["timing"] = {**timing_summary(measured), "per_rally": {
            rally["id"]: timing_summary([r for r in measured if r["rally_id"] == rally["id"]])
            for rally in context["rallies"]}}
        results["methods"][method]["smoothing_seconds"] = sum(r["seconds"] for r in smoothing_times if r["method"] == method)
    for rally in context["rallies"]:
        folder = output / "detector" / rally["id"]
        folder.mkdir(parents=True)
        for variant in ("raw", "smoothed"):
            entry = records["detector", variant, rally["id"]]
            suffix = "" if variant == "raw" else "_smoothed"
            for name in ("positions", "errors"):
                filename = "player_positions" if name == "positions" else "shot_errors"
                kp.write_csv(folder / f"{filename}{suffix}.csv", entry[name], list(entry[name][0]))
            metrics = results["methods"]["detector"][variant]["per_rally"][rally["id"]]
            metrics = {**metrics, "source_dimensions": rally["dimensions"], "source_fps": rally["fps"],
                       "decoded_frames": rally["frames"], "halves": rally["halves"],
                       **({"smoothing": context["settings"]} if variant == "smoothed" else {})}
            metrics["video_info"] = render_positions(rally, entry["positions"], folder / f"review{suffix}.mp4",
                                                     smoothed=variant == "smoothed")
            kp.write_json(folder / f"metrics{suffix}.json", metrics)
    from comparison_plots import write_graphs_and_report
    write_graphs_and_report(output, context["rallies"], records, timings, results)
    kp.write_json(output / "comparison.json", results)
    print(f"Comparison complete: {output / 'comparison.md'}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate inputs/checkpoints/exports without inference")
    args = parser.parse_args(argv)
    try:
        context = load_context(args.review_dir)
        for name in ("matplotlib", "scipy"):
            if importlib.util.find_spec(name) is None:
                raise RuntimeError(f"Missing installed dependency: {name}")
        device = kp.resolve_device(context["config"])
        detector.check_torchvision_ops(device)
        report = {"ready": True, "review_dir": context["review_dir"],
                  "rallies": [{"key": list(r["key"]), "frames": r["frames"],
                               "references": 2 * len(r["annotations"]), "halves": r["halves"]}
                              for r in context["rallies"]], "smoothing": context["settings"],
                  "device": str(device), "inference_run": False}
        print(json.dumps(report, indent=2), flush=True)
        if not args.check:
            run_comparison(context)
        return 0
    except Exception as error:
        print(f"FAIL comparison: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
