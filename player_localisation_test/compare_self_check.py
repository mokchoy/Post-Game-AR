"""Bounded synthetic comparison checks; no real checkpoint inference."""
from __future__ import annotations

import copy
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import compare_review as comparison
from comparison_plots import eligible_cdf

kp = comparison.kp


class Geometry:
    row_id = "synthetic"
    homography_annotation_to_dataset = np.eye(3)
    dataset_to_metric = np.diag([6.1 / 1280, 13.4 / 720, 1])

    def annotation_floor_to_world(self, point):
        return np.asarray(point) * [6.1 / 1280, 13.4 / 720]

    def world_to_annotation_floor(self, point):
        return np.asarray(point) / [6.1 / 1280, 13.4 / 720]


class HeatmapStub(torch.nn.Module):
    def forward(self, image):
        result = torch.zeros((1, 2, 8, 8), device=image.device)
        result[0, 0, 2, 3] = 10
        result[0, 1, 5, 4] = 10
        return result


def fake_detections(model, frame, device):
    # The middle synthetic clip has no predictions at all.
    if 35 <= int(frame[0, 0, 0]) < 75:
        return {"boxes": np.empty((0, 4)), "scores": np.empty(0), "labels": np.empty(0)}
    return {"boxes": np.array([[100, 20, 160, 80], [170, 180, 230, 270]], dtype=float),
            "scores": np.array([.9, .8]), "labels": np.array([1, 1])}


def expect_failure(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return
    raise AssertionError(f"Expected {function.__name__} to reject invalid input")


def build_fixture(root):
    point_root = root / "keypoint"
    review = point_root / "runs" / "synthetic" / "selected_rally_reviews" / "saved"
    review.mkdir(parents=True)
    config = {"cv": {"folds": 5, "seed": 42}, "model": {"input_width": 32, "input_height": 32},
              "train": {"device": "cpu"}}
    geometry = Geometry()
    settings = {"median_window": 5, "savgol_window": 9, "gap_policy": "continuous",
                "polyorder": 2, "court_margin_m": 3.0, "half_margin_m": 1.0,
                "median_outlier_m": 1.0, "max_short_gap_seconds": .2}
    all_rows, rallies = [], []
    for number in range(3):
        key = ("set-1", f"Synthetic_match_{number}", "1", str(number + 1))
        clip = root / "cleaned" / key[1] / "set1" / f"rally_{number+1:03d}.mp4"
        clip.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 360))
        assert writer.isOpened()
        for frame in range(8):
            writer.write(np.full((360, 320, 3), number * 40 + frame * 2, dtype=np.uint8))
        writer.release()
        halves = {"A": "near", "B": "far"} if number != 2 else {"A": "far", "B": "near"}
        frozen = []
        for frame in (1, 5):
            row = {**dict(zip(kp.KEY_FIELDS, key)), "manifest_version": 2, "clip_frame_num": frame,
                   "frame_num": frame, "clip_time_seconds": frame / 10, "source_time_seconds": frame / 10,
                   "clip_path": str(clip), "fps": 10, "width": 320, "height": 360,
                   "image_path": str(root / "unused.jpg"), "automatic_check_status": "passed"}
            for player in ("A", "B"):
                pixel = (200, 270) if halves[player] == "near" else (130, 80)
                reference = kp.reference_point(pixel, 320, 360)
                row.update({f"{player}_native_x": pixel[0], f"{player}_native_y": pixel[1],
                            f"{player}_ref_x": reference[0], f"{player}_ref_y": reference[1],
                            f"{player}_half": halves[player]})
            frozen.append(row)
        annotations = kp.annotations_by_frame(frozen, geometry, 320, 360)
        rally = {"key": key, "id": kp.safe_id(key), "rows": frozen, "clip_path": str(clip),
                 "dimensions": [320, 360], "fps": 10, "frames": 8, "halves": halves,
                 "geometry": geometry, "annotations": annotations}
        raw = []
        for frame in range(8):
            prediction = {player: None if number == 1 else {
                "bottom": ((200 + frame, 270) if halves[player] == "near" else (130 + frame, 80)),
                "court": kp.projected_world(geometry, (200 + frame, 270) if halves[player] == "near" else (130 + frame, 80), 320, 360)}
                for player in ("A", "B")}
            if number == 0 and frame in (0, 2):
                prediction["A"] = None
            raw.extend(comparison.position_rows(prediction, frame, rally))
        smooth = comparison.smooth_positions(raw, rally, settings)
        folder = review / rally["id"]
        folder.mkdir()
        for suffix, positions in (("", raw), ("_smoothed", smooth)):
            errors = comparison.score_positions(positions, annotations)
            kp.write_csv(folder / f"player_positions{suffix}.csv", positions, list(positions[0]))
            kp.write_csv(folder / f"shot_errors{suffix}.csv", errors, list(errors[0]))
            kp.write_json(folder / f"metrics{suffix}.json", {
                "source_dimensions": [320, 360], "decoded_frames": 8, "source_fps": 10,
                "annotated_frames": sorted(annotations), "combined": kp.error_summary(errors), "smoothing": settings})
        rally["keypoint"] = {"raw": raw, "smoothed": smooth}
        rallies.append(rally)
        all_rows.extend(frozen)
    checkpoint = point_root / "runs" / "synthetic" / "last.pt"
    checkpoint.write_bytes(b"synthetic placeholder; never loaded")
    identity = kp.dataset_identity(all_rows)
    held_out = [r["key"][1] for r in rallies]
    info = {"last_sha256": comparison.detector.sha256(checkpoint), "test_used_for_training": False,
            "held_out_matches": held_out, "development_matches": ["development"]}
    selection = {"checkpoint_sha256": info["last_sha256"], "checkpoint": str(checkpoint),
                 "dataset_identity": identity, "resolved_settings": kp.resolved_settings(config),
                 "held_out_matches": held_out, "final_run_id": "synthetic",
                 "selection_saved_before_inference": True,
                 "selected_rallies": [{"key": list(r["key"]), "clip_path": r["clip_path"]} for r in rallies]}
    kp.write_json(review / "selection.json", selection)
    kp.write_json(review / "review_smoothing.json", settings)
    kp.write_json(checkpoint.parent / "final.json", info)
    prep = point_root / "data"
    prep.mkdir()
    kp.write_jsonl(prep / "frozen_manifest.jsonl", all_rows)
    split = {"split_version": 2, "dataset_identity": identity, "held_out_matches": held_out,
             "development_matches": ["development"]}
    kp.write_json(point_root / "split.json", split)
    (point_root / "homography.csv").write_text("synthetic", encoding="utf-8")
    return point_root, review, config, checkpoint, info, selection, split, rallies


def run():
    outputs = comparison.ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="comparison_fixture_", dir=outputs) as temp, ExitStack() as stack:
        root = Path(temp)
        point_root, review, config, checkpoint, info, selection, split, rallies = build_fixture(root)
        comparison.validate_selection(selection, info, config, checkpoint, split)
        for field, value in (("dataset_identity", "wrong"), ("checkpoint_sha256", "wrong"),
                             ("resolved_settings", {}), ("selection_saved_before_inference", False)):
            expect_failure(comparison.validate_selection, {**selection, field: value}, info, config, checkpoint, split)
        expect_failure(comparison.validate_selection, {**selection, "selected_rallies": [selection["selected_rallies"][0]] * 3},
                       info, config, checkpoint, split)
        settings = comparison.read_json(review / "review_smoothing.json")
        for field, value in (("median_window", 2), ("savgol_window", True), ("gap_policy", "wrong"), ("polyorder", 1)):
            expect_failure(comparison.validate_smoothing, {**settings, field: value})
        matrix = comparison.summarize(
            [{"player": "A", "prediction_status": "available", "error_px": 3, "error_m": .25},
             {"player": "B", "prediction_status": "missing", "error_px": None, "error_m": None}],
            [{"player": "A", "available": True}, {"player": "B", "available": False}])
        assert matrix["combined"]["coverage_percentage"] == 50
        assert matrix["combined"]["within_0_5_m"]["percentage"] == 50
        assert matrix["combined"]["metre_error"]["euclidean_rmse"] == .25
        assert comparison.detector._error_stats([{"error_m": 1}, {"error_m": 3}, {"error_m": 3}], "error_m")["mean"] == 7/3
        x, y = eligible_cdf([{"error_m": .25}, {"error_m": None}], "error_m", 1)
        assert y[-1] == .5 and eligible_cdf([{"error_m": None}], "error_m", 1)[1] == [0, 0]
        assert eligible_cdf([], "error_m", 1)[1] == [0, 0]
        expect_failure(comparison.timing_summary, [{"milliseconds": 0}])
        assert comparison.timing_summary([{"milliseconds": 10}, {"milliseconds": 30}])["fps"] == 50
        for field, value in (("ROOT", point_root), ("SOURCE", root / "cleaned"), ("PREP", point_root / "data"),
                             ("SPLIT_FILE", point_root / "split.json"), ("HOMOGRAPHY", point_root / "homography.csv")):
            stack.enter_context(patch.object(kp, field, value))
        stack.enter_context(patch.object(kp, "final_context", return_value=(info, config, checkpoint)))
        stack.enter_context(patch.object(kp, "import_geometry", return_value=Geometry()))
        stack.enter_context(patch.object(kp.pm, "load_checkpoint", side_effect=lambda path: (HeatmapStub(), {})))
        stack.enter_context(patch.object(comparison.detector, "validate_checkpoint", return_value="synthetic"))
        context = comparison.load_context(review)
        changed = copy.deepcopy(context["rallies"][0]["keypoint"]["smoothed"])
        changed[0]["x_native"] += 1
        expect_failure(comparison.validate_smoothed, changed, context["rallies"][0]["keypoint"]["smoothed"])
        broken = root / "broken.csv"
        raw_path = review / rallies[0]["id"] / "player_positions.csv"
        rows = kp.read_csv(raw_path)
        rows[0]["court_half"] = "far"
        kp.write_csv(broken, rows, list(rows[0]))
        expect_failure(comparison.read_positions, broken, rallies[0])
        errors_path = review / rallies[0]["id"] / "shot_errors.csv"
        rows = kp.read_csv(errors_path)
        rows[0]["error_m"] = 999
        kp.write_csv(broken, rows, list(rows[0]))
        expect_failure(comparison.validate_saved_errors, broken,
                       comparison.score_positions(rallies[0]["keypoint"]["raw"], rallies[0]["annotations"]))
        rows = kp.read_csv(raw_path)
        rows.append(rows[0])
        kp.write_csv(broken, rows, list(rows[0]))
        expect_failure(comparison.read_positions, broken, rallies[0])
        split_path = point_root / "split.json"
        kp.write_json(split_path, {**split, "dataset_identity": "wrong"})
        expect_failure(comparison.load_context, review)
        kp.write_json(split_path, split)
        settings_path = review / "review_smoothing.json"
        kp.write_json(settings_path, {**settings, "gap_policy": "conservative"})
        expect_failure(comparison.load_context, review)
        kp.write_json(settings_path, settings)
        metrics_path = review / rallies[0]["id"] / "metrics.json"
        original = comparison.read_json(metrics_path)
        changed_metrics = copy.deepcopy(original)
        changed_metrics["combined"]["pixel_error"]["count"] += 1
        kp.write_json(metrics_path, changed_metrics)
        expect_failure(comparison.load_context, review)
        kp.write_json(metrics_path, original)
        stack.enter_context(patch.object(kp, "run_dependencies", return_value={"synthetic": True}))
        stack.enter_context(patch.object(comparison.detector, "check_torchvision_ops"))
        stack.enter_context(patch.object(comparison.detector, "load_pretrained_model", return_value=torch.nn.Identity()))
        stack.enter_context(patch.object(comparison.detector, "inference", side_effect=fake_detections))
        original_fresh = comparison.detector.fresh_output_dir
        stack.enter_context(patch.object(comparison.detector, "fresh_output_dir",
                                        side_effect=lambda: original_fresh(stamp="comparison_self_check")))
        with patch.object(comparison, "detector_prediction", wraps=comparison.detector_prediction) as detector_calls, \
                patch.object(comparison, "keypoint_prediction", wraps=comparison.keypoint_prediction) as keypoint_calls:
            output = comparison.run_comparison(context)
            assert detector_calls.call_count == keypoint_calls.call_count == 3 * (10 + 8)
        result = comparison.read_json(output / "comparison.json")
        assert result["methods"]["detector"]["raw"]["combined"]["eligible_references"] == 12
        assert result["methods"]["detector"]["raw"]["combined"]["coverage_percentage"] == 100 * 8 / 12
        assert result["methods"]["detector"]["raw"]["per_rally"][rallies[0]["id"]]["combined"]["metre_error"]["mean"] == 0
        assert result["methods"]["detector"]["raw"]["per_rally"][rallies[1]["id"]]["combined"]["metre_error"]["count"] == 0
        assert result["methods"]["keypoint"]["raw"]["combined"]["coverage_percentage"] == 100 * 8 / 12
        for method in ("detector", "keypoint"):
            assert result["methods"][method]["timing"]["frames"] == 24
        for rally in rallies:
            folder = output / "detector" / rally["id"]
            raw = kp.read_csv(folder / "player_positions.csv")
            assert len(raw) == 16 and raw[0]["court_half"] == rally["halves"]["A"]
            for suffix in ("", "_smoothed"):
                assert len(kp.read_csv(folder / f"shot_errors{suffix}.csv")) == 4
                assert comparison.read_json(folder / f"metrics{suffix}.json")["video_info"]["decoded_frames"] == 8
        assert len(result["graphs"]) == 18
        for relative in result["graphs"]:
            path = output / relative
            assert path.stat().st_size > 1000
            if path.suffix == ".png":
                assert cv2.imread(str(path)) is not None
            else:
                assert path.read_bytes().startswith(b"%PDF")
        report_path = output / "comparison.md"
        report_path.write_text("**SYNTHETIC VALIDATION ONLY — no real model performance.**\n\n" +
                               report_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"PASS: synthetic end-to-end export, graphs, identity/scaling, empty predictions, provenance rejection, "
              f"timing boundaries and 10 warm-ups per clip. Preview: {output}")


if __name__ == "__main__":
    run()
