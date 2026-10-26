from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

import point_model as pm
import runner


class GeometryStub:
    @staticmethod
    def annotation_floor_to_world(point):
        return np.asarray([
            point[0] / runner.REFERENCE_WIDTH * runner.COURT_WIDTH_M,
            point[1] / runner.REFERENCE_HEIGHT * runner.COURT_LENGTH_M,
        ])

    @staticmethod
    def world_to_annotation_floor(point):
        return np.asarray([point[0] / runner.COURT_WIDTH_M * runner.REFERENCE_WIDTH,
                           point[1] / runner.COURT_LENGTH_M * runner.REFERENCE_HEIGHT])


def check_coordinates_targets_and_alignment() -> None:
    original = (639.0, 359.0)
    scaled = runner.native_point(original, 960, 540)
    restored = runner.reference_point(scaled, 960, 540)
    assert np.allclose(restored, original)
    row = {"player": "A", "player_location_x": "300", "player_location_y": "200",
           "opponent_location_x": "900", "opponent_location_y": "600"}
    assert runner.align_references(row) == {"A": (300.0, 200.0), "B": (900.0, 600.0)}
    row["player"] = "B"
    assert runner.align_references(row) == {"A": (900.0, 600.0), "B": (300.0, 200.0)}

    targets = pm.heatmap_target([(41.5, 25.5), (121.5, 101.5)], 32, 40, 2.0)
    assert targets.shape == (2, 32, 40)
    assert torch.allclose(targets.sum(dim=(1, 2)), torch.ones(2), atol=1e-6)
    logits = torch.full((2, 32, 40), -20.0)
    logits[0, 6, 10] = 20
    logits[1, 25, 30] = 20
    points = pm.decode_heatmaps(logits, 160, 128, 160, 128)
    assert len(points) == 2
    assert points[0]["x_native"] == 41.5 and points[0]["y_native"] == 25.5
    assert points[1]["x_native"] == 121.5 and points[1]["y_native"] == 101.5

    with tempfile.TemporaryDirectory(prefix="point_dataset_", dir=runner.ROOT / "checks") as temp:
        image_path = Path(temp) / "frame.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((720, 1280, 3), dtype=np.uint8))
        item = {
            "image_path": str(image_path), "width": 1280, "height": 720,
            "A_native_x": 500, "A_native_y": 100, "B_native_x": 700, "B_native_y": 600,
            "A_half": "near", "B_half": "far",
        }
        image, target = pm.FrameDataset([item], 160, 128, 2.0)[0]
        assert image.shape == (3, 128, 160)
        assert target[1].argmax().item() != target[0].argmax().item()
        assert target[1].argmax().item() == 4 * 40 + 15
        assert target[0].argmax().item() == 26 * 40 + 21


def check_spawned_data_loading() -> None:
    with tempfile.TemporaryDirectory(prefix="point_workers_", dir=runner.ROOT / "checks") as temp:
        image_path = Path(temp) / "frame.png"
        frame = np.arange(64 * 80 * 3, dtype=np.uint8).reshape(64, 80, 3)
        assert cv2.imwrite(str(image_path), frame)
        row = {
            "image_path": str(image_path), "width": 80, "height": 64,
            "A_native_x": 20, "A_native_y": 10, "B_native_x": 60, "B_native_y": 50,
            "A_half": "far", "B_half": "near",
        }
        config = {
            "model": {"input_width": 160, "input_height": 128, "gaussian_sigma": 2.0},
            "train": {"batch": 2, "workers": 4, "device": "cpu", "seed": 42,
                      "horizontal_flip_probability": 1.0},
        }
        for training in (False, True):
            loader = runner.resolved_train_loader([row] * 8, config, training=training)
            loader.multiprocessing_context = "spawn"
            assert loader.num_workers == 4
            expected_image, expected_target = loader.dataset[0]
            assert torch.equal(expected_image, pm.image_tensor(frame, 160, 128).flip(2)
                               if training else pm.image_tensor(frame, 160, 128))
            seen = 0
            for images, targets in loader:
                assert torch.equal(images, expected_image.expand(2, -1, -1, -1))
                assert torch.equal(targets, expected_target.expand(2, -1, -1, -1))
                seen += len(images)
            assert seen == 8


def write_annotations(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_whole_rally_exclusion() -> None:
    with tempfile.TemporaryDirectory(prefix="point_rally_", dir=runner.ROOT / "checks") as temp:
        match_dir = Path(temp) / "synthetic_match"
        clip_dir = match_dir / "set1"
        clip_dir.mkdir(parents=True)
        clip = clip_dir / "rally_001.avi"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720))
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.zeros((720, 1280, 3), dtype=np.uint8))
        writer.release()
        rows = []
        for frame in range(3):
            hitter = "A" if frame != 1 else "B"
            player, opponent = (500, 100), (700, 600)
            player_xy, opponent_xy = (player, opponent) if hitter == "A" else (opponent, player)
            rows.append({
                "player": hitter, "player_location_x": player_xy[0], "player_location_y": player_xy[1],
                "opponent_location_x": opponent_xy[0], "opponent_location_y": opponent_xy[1],
                "flaw": "0", "frame_num": frame, "clip_frame_num": frame,
                "source_time_seconds": f"{frame / 10:.6f}", "clip_time_seconds": f"{frame / 10:.6f}",
            })
        annotation_path = clip.with_suffix(".csv")
        write_annotations(annotation_path, rows)
        inventory = {
            "dataset": "set-1", "match": "synthetic", "game": "1", "original_rally_id": "1",
            "filename": "set1/rally_001.avi", "first_hit_frame": "0", "last_hit_frame": "2",
            "annotation_fps": "10", "time_offset": "0", "start_seconds": "0", "review_flags": "",
        }
        exclusions = []
        clean = runner.inspect_rally(
            match_dir, inventory, [{"outcome": "copied-unchanged", "review_flags": ""}],
            {"synthetic": GeometryStub()}, exclusions,
        )
        assert clean["status"] == "automatic_pass"
        assert clean["A_half"] == "far" and clean["B_half"] == "near"
        rows[1]["flaw"] = "1"
        write_annotations(annotation_path, rows)
        excluded = runner.inspect_rally(
            match_dir, inventory, [{"outcome": "copied-unchanged", "review_flags": ""}],
            {"synthetic": GeometryStub()}, exclusions,
        )
        assert excluded["status"] == "excluded"
        assert "dataset_flaw_flag" in excluded["reasons"]


def check_splits_and_config() -> dict:
    matches = [f"m{i}" for i in range(12)]
    held_a = runner.fixed_test_matches(matches)
    held_b = runner.fixed_test_matches(matches)
    assert held_a == held_b
    folds = runner.grouped_folds(sorted(set(matches) - set(held_a)), 3, 42)
    assert set(folds.values()) == {0, 1, 2}
    for invalid in (1, 13):
        try:
            runner.grouped_folds(matches, invalid, 42)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid fold count {invalid} was accepted")
    split = {"development_matches": sorted(set(matches) - set(held_a)), "held_out_matches": held_a}
    rows = [{"match": f"m{i}"} for i in range(12)]
    train, val = runner.cv_split_rows(rows, split, folds, 0)
    assert {row["match"] for row in train}.isdisjoint({row["match"] for row in val})
    assert not ({row["match"] for row in train} | {row["match"] for row in val}) & set(held_a)

    config = runner.load_config(folds_override=2)
    assert config["cv"]["folds"] == 2 and config["model"]["input_width"] == 960
    invalid = runner.resolved_settings(config)
    invalid["cv"]["folds"] = 1
    with tempfile.TemporaryDirectory(prefix="point_config_", dir=runner.ROOT / "checks") as temp:
        path = Path(temp) / "invalid.yaml"
        path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
        try:
            runner.load_config(str(path))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid YAML fold count was accepted")
    optimizer = runner.build_optimizer(torch.nn.Linear(2, 2), {
        "optimizer": "AdamW", "lr0": 0.004, "weight_decay": 0.002,
    })
    assert optimizer.param_groups[0]["lr"] == 0.004
    assert optimizer.param_groups[0]["weight_decay"] == 0.002
    return config


def check_training_progress() -> None:
    config = {"train": {
        "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0, "device": "cpu",
        "amp": False, "freeze_backbone_epochs": 0, "patience": 1,
    }}
    for final in (False, True):
        for batches in (1, 23):
            model = torch.nn.Sequential()
            model.add_module("backbone", torch.nn.Linear(1, 1))
            loader = [(torch.ones(1, 1), torch.zeros(1, 1))] * batches
            transfer = {"complete_transfer": True, "source_checkpoint_sha256": "synthetic"}
            errors = [{"available": True, "error_px": 1.0, "error_m": 0.5}]
            with (
                tempfile.TemporaryDirectory(prefix="point_progress_", dir=runner.ROOT / "checks") as temp,
                patch.object(runner, "ensure_pretrained_weights", return_value=Path("unused.pt")),
                patch.object(pm, "initialize_from_yolo11n", return_value=(model, transfer)),
                patch.object(runner, "resolved_train_loader", return_value=loader),
                patch.object(pm, "spatial_cross_entropy", side_effect=torch.nn.functional.mse_loss),
                patch.object(pm, "save_checkpoint", return_value="synthetic"),
                patch.object(runner, "score_model_rows", return_value=([], errors)) as score,
                patch("builtins.print") as output,
            ):
                result = runner.run_training([{}] * batches, None if final else [{}],
                                             Path(temp) / "fold_01", config, 42, 3, final=final)
            lines = [call.args[0] for call in output.call_args_list]
            assert all(call.kwargs.get("flush") is True for call in output.call_args_list)
            label = "Final" if final else "CV fold_01"
            assert all(line.startswith(f"[{label}]") for line in lines)
            completed = 3 if final else 2
            assert result["epochs_completed"] == completed
            assert score.call_count == (0 if final else completed)
            assert sum("complete:" in line for line in lines) == completed
            assert any("Early stopping" in line for line in lines) is (not final)
            assert any("val_error_m=0.5000" in line for line in lines) is (not final)
            for epoch in range(1, completed + 1):
                updates = [line for line in lines if f"Epoch {epoch}/3 batch " in line]
                assert 1 <= len(updates) <= 11
                assert f"batch 1/{batches} " in updates[0]
                assert f"batch {batches}/{batches} (100%)" in updates[-1]
                if batches > 1:
                    assert len(updates) < batches


def check_explicit_review_selection() -> None:
    manifest, clips = [], []
    for match in ("heldout_a", "heldout_b"):
        for rally in range(1, 6):
            clip = runner.SOURCE / match / "set1" / f"rally_{rally:03d}.mp4"
            clips.append(clip)
            manifest.append({
                "dataset": "set-1", "match": match, "game": "1", "original_rally_id": str(rally),
                "clip_frame_num": "0", "clip_path": str(clip), "width": "1920", "height": "1080",
                "fps": "25", "A_half": "far", "B_half": "near", "A_native_x": "600",
                "A_native_y": "200", "B_native_x": "800", "B_native_y": "560",
            })
    held_out = {"heldout_a", "heldout_b"}
    selected = runner.review_rallies(manifest, held_out, clips=list(reversed(clips)))
    assert len(selected) == 10
    assert [Path(row["clip_path"]) for row in selected] == list(reversed(clips))
    assert len(runner.review_rallies(manifest, held_out)) == 2
    split = {"held_out_matches": sorted(held_out), "development_matches": [], "dataset_identity": "synthetic"}
    info = {**split, "test_used_for_training": False}
    with patch.object(runner, "final_context", return_value=(info, {}, Path("synthetic.pt"))), \
            patch.object(runner, "sha256_file", return_value="synthetic"), \
            patch.object(runner, "load_frozen_split", return_value=(split, manifest)), \
            patch.object(runner, "import_geometry"), patch.object(Path, "is_file", return_value=True):
        context, blockers = runner.review_test_context("synthetic", clips=clips)
    assert not blockers and len(context["selected"]) == 10, blockers
    for requested, allowed in (([], held_out), ([clips[0], clips[0]], held_out),
                               ([clips[0], clips[0].parent / "rally_999.mp4"], held_out),
                               ([clips[0]], {"heldout_b"})):
        try:
            runner.review_rallies(manifest, allowed, clips=requested)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted empty, duplicate, unknown, or development clips")
    args = runner.parser().parse_args(["review-test", "--run-id", "synthetic", "--smooth",
                                       "--check", "--clips", *map(str, clips)])
    assert args.clips == clips and args.smooth and args.check
    context = {"selected": selected, "checkpoint": Path("synthetic.pt"), "dataset_identity": "synthetic"}
    with patch.object(runner, "review_test_context", return_value=(context, [])) as prepare, \
            patch.object(pm, "load_checkpoint", return_value=(None, {"model_kind": pm.MODEL_KIND})), \
            patch("builtins.print") as output:
        assert runner.review_test_command("synthetic", check_only=True, clips=clips) == 0
    prepare.assert_called_once_with("synthetic", clips=clips)
    report = json.loads(output.call_args.args[0])
    assert report["ready"] and report["selection_mode"] == "explicit_clips"
    assert len(report["selected_rallies"]) == 10 and report["shortfall_from_three"] == 0


def check_deterministic_review_and_render() -> None:
    preferred_match = "Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final"
    matches = [preferred_match, "heldout_b", "heldout_c", "heldout_d"]
    manifest = []
    for match in matches:
        for rally_id in (1, 2):
            clip = (runner.PREFERRED_REVIEW_CLIP if match == preferred_match and rally_id == 2 else
                    runner.SOURCE / match / "set1" / f"rally_{rally_id:03d}.mp4")
            for frame in (0, 2):
                manifest.append({
                    "dataset": "set-1", "match": match, "game": "1", "original_rally_id": str(rally_id),
                    "clip_frame_num": str(frame), "clip_path": str(clip), "width": "1280",
                    "height": "720", "fps": "12.5", "A_half": "far", "B_half": "near",
                    "A_native_x": "600", "A_native_y": "200", "B_native_x": "800", "B_native_y": "560",
                })
    selected = runner.review_rallies(manifest, set(matches), seed=42)
    assert selected == runner.review_rallies(manifest, set(matches), seed=42)
    assert len(selected) == 3 and len({row["key"][1] for row in selected}) == 3
    assert selected[0]["key"] == ("set-1", preferred_match, "1", "2")
    assert len(runner.review_rallies(manifest, {"heldout_b", "heldout_c"}, seed=42)) == 2

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.bad_frames = set()
            logits = torch.full((1, 2, 32, 40), -20.0)
            logits[0, 0, 10, 20] = 20
            logits[0, 1, 25, 24] = 20
            self.register_buffer("logits", logits)

        def forward(self, image):
            self.calls += 1
            logits = self.logits.flip(1) if self.calls in self.bad_frames else self.logits
            return logits.expand(image.shape[0], -1, -1, -1)

    with tempfile.TemporaryDirectory(prefix="point_review_", dir=runner.ROOT / "checks") as temp:
        root = Path(temp)
        clip = root / "source.avi"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 12.5, (1280, 720))
        assert writer.isOpened()
        for index in range(4):
            writer.write(np.full((720, 1280, 3), index * 20, dtype=np.uint8))
        writer.release()
        rally_rows = []
        for frame in (0, 2):
            rally_rows.append({
                "dataset": "set-1", "match": "synthetic", "game": "1", "original_rally_id": "1",
                "clip_frame_num": frame, "clip_path": str(clip), "width": 1280, "height": 720,
                "fps": 12.5, "A_half": "far", "B_half": "near",
                "A_native_x": 600, "A_native_y": 200, "B_native_x": 800, "B_native_y": 560,
            })
        config = runner.load_config()
        config["model"].update(input_width=160, input_height=128)
        config["train"].update(batch=1, workers=0, device="cpu")
        model = FakeModel().eval()
        annotations = runner.annotations_by_frame(rally_rows, GeometryStub(), 1280, 720)
        assert set(annotations) == {0, 2}
        result = runner.render_review_rally(
            {"key": runner.stable_key(rally_rows[0]), "clip_path": str(clip), "rows": rally_rows},
            root / "review", model, GeometryStub(), config, torch.device("cpu"),
        )
        with (root / "review" / "player_positions.csv").open(newline="", encoding="utf-8") as stream:
            positions = list(csv.DictReader(stream))
        with (root / "review" / "shot_errors.csv").open(newline="", encoding="utf-8") as stream:
            errors = list(csv.DictReader(stream))
        assert len(positions) == 8 and len(errors) == 4
        assert {int(row["clip_frame_num"]) for row in errors} == {0, 2}
        assert all(sum(int(row["clip_frame_num"]) == frame for row in positions) == 2 for frame in range(4))
        assert all(row["reference_x_native"] == "" for row in positions if row["clip_frame_num"] == "1")
        assert not any("box" in field.lower() for field in positions[0])
        assert result["video_info"]["decoded_frames"] == 4
        assert result["video_info"]["fps"] == 12.5
        assert (result["video_info"]["width"], result["video_info"]["height"]) == (1880, 720)
        assert not (root / "review" / "review_smoothed.mp4").exists()
        rally = {"key": runner.stable_key(rally_rows[0]), "clip_path": str(clip), "rows": rally_rows}
        model.logits.copy_(model.logits.flip(1))
        missing = runner.render_review_rally(rally, root / "missing", model, GeometryStub(), config,
                                             torch.device("cpu"), runner.load_review_smoothing())
        assert missing["smoothed"]["metrics"]["position_coverage"] == 0.0
        assert missing["smoothed"]["metrics"]["combined"]["missing_predictions"] == 4
        missing_rows = runner.read_csv(root / "missing" / "player_positions_smoothed.csv")
        assert len(missing_rows) == 8 and all(row["x_native"] == "" for row in missing_rows)
        model.logits.copy_(model.logits.flip(1))
        baseline = runner.render_review_rally(rally, root / "raw_reference", model, GeometryStub(),
                                              config, torch.device("cpu"))
        calls = model.calls
        smoothed = runner.render_review_rally(rally, root / "smoothed", model, GeometryStub(),
                                              config, torch.device("cpu"), runner.load_review_smoothing())
        assert model.calls - calls == 4
        assert smoothed["metrics"] == baseline["metrics"]
        for name in ("player_positions.csv", "shot_errors.csv", "metrics.json"):
            assert (root / "smoothed" / name).read_bytes() == (root / "raw_reference" / name).read_bytes()
        assert smoothed["smoothed"]["video_info"] == baseline["video_info"]
        assert smoothed["smoothed"]["metrics"]["position_coverage"] == 1.0
        smooth_rows = runner.read_csv(root / "smoothed" / "player_positions_smoothed.csv")
        assert len(smooth_rows) == 8
        assert {r["player"] for r in smooth_rows} == {"A", "B"}
        smooth_errors = runner.read_csv(root / "smoothed" / "shot_errors_smoothed.csv")
        assert len(smooth_errors) == 4
        assert {int(r["clip_frame_num"]) for r in smooth_errors} == {0, 2}
        for row in smooth_rows:
            image = (float(row["x_native"]), float(row["y_native"]))
            world = (float(row["x_court_m"]), float(row["y_court_m"]))
            assert np.allclose(runner.projected_world(GeometryStub(), image, 1280, 720), world)
            assert row["filter_status"] == "smoothed" and row["gap_filled"] == "False"
        model.bad_frames = {model.calls + 2}
        filled = runner.render_review_rally(rally, root / "filled", model, GeometryStub(), config,
                                            torch.device("cpu"), runner.load_review_smoothing())
        assert filled["smoothed"]["metrics"]["position_coverage"] == 1.0
        assert filled["smoothed"]["metrics"]["filled_points"] == 2
        filled_rows = runner.read_csv(root / "filled" / "player_positions_smoothed.csv")
        assert sum(row["gap_filled"] == "True" for row in filled_rows) == 2
        assert all(row["filter_status"] == "short_interpolation"
                   for row in filled_rows if row["clip_frame_num"] == "1")

    failures = runner.error_summary([
        {"available": True, "error_px": 10.0, "error_m": 0.2},
        {"available": False, "error_px": None, "error_m": None},
        {"available": False, "error_px": None, "error_m": None},
    ])
    assert failures["expected_predictions"] == 3 and failures["available_predictions"] == 1
    assert failures["coverage"] == 1 / 3 and failures["within_1_m"]["denominator"] == 3


def check_review_smoothing() -> None:
    settings = {**runner.load_review_smoothing(), "median_window": 5,
                "savgol_window": 9, "gap_policy": "conservative"}
    with tempfile.TemporaryDirectory(prefix="smoothing_config_", dir=runner.ROOT / "checks") as temp:
        path = Path(temp) / "review.yaml"
        for key, values in (("median_window", [0, 2, True, 3.0]),
                            ("savgol_window", [1, 4, False, 5.0]),
                            ("gap_policy", ["unknown", None])):
            for value in values:
                config = {k: settings[k] for k in ("median_window", "savgol_window", "gap_policy")}
                config[key] = value
                path.write_text(yaml.safe_dump({"smoothing": config}), encoding="utf-8")
                try:
                    runner.load_review_smoothing(path)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Accepted invalid {key}: {value!r}")
        for config in ([], {}, {"smoothing": None}, {"smoothing": {"median_window": 5}}):
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            try:
                runner.load_review_smoothing(path)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted invalid review settings")
        path.write_text(yaml.safe_dump({"smoothing": {
            "median_window": 1, "savgol_window": 3, "gap_policy": "continuous"}}), encoding="utf-8")
        assert runner.load_review_smoothing(path)["median_window"] == 1
    args = runner.parser().parse_args(["review-test", "--run-id", "synthetic", "--smooth", "--check"])
    assert args.smooth and args.check
    for world in ((3.0, 3.0), (3.0, 10.0)):
        native = runner.native_point(GeometryStub().world_to_annotation_floor(world), 1280, 720)
        half = runner.court_half(GeometryStub(), native, 1280, 720)
        accepted, _, _ = runner.smooth_court_track(np.tile(world, (9, 1)), half, 25, settings)
        assert np.allclose(accepted, world)
        other_half = "near" if half == "far" else "far"
        rejected, _, reasons = runner.smooth_court_track(np.tile(world, (9, 1)), other_half, 25, settings)
        assert np.isnan(rejected).all() and set(reasons) == {"court_half"}
    with patch.object(runner, "load_review_smoothing", side_effect=ValueError("bad window")), \
            patch.object(runner, "review_test_context") as context:
        try:
            runner.review_test_command("synthetic", check_only=True, smooth=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid smoothing config did not stop preflight")
        context.assert_not_called()

    rng = np.random.default_rng(42)
    stationary = np.tile([3.0, 3.0], (61, 1)) + rng.normal(0, 0.04, (61, 2))
    smooth, statuses, _ = runner.smooth_court_track(stationary, "far", 25, settings)
    assert np.mean((smooth - 3) ** 2) < np.mean((stationary - 3) ** 2)
    assert set(statuses) == {"smoothed"}
    unfiltered, _, _ = runner.smooth_court_track(
        stationary, "far", 25, {**settings, "median_window": 1, "savgol_window": 3})
    assert np.allclose(unfiltered, stationary)
    assert not np.allclose(unfiltered, smooth)
    moving = np.column_stack((2 + np.arange(61) * 0.02, np.full(61, 3.0)))
    smooth, _, _ = runner.smooth_court_track(moving, "far", 25, settings)
    assert np.allclose(smooth, moving)
    turning = np.column_stack((2 + np.abs(np.arange(61) - 30) * 0.02, np.full(61, 3.0)))
    smooth, _, _ = runner.smooth_court_track(turning, "far", 25, settings)
    assert np.max(np.abs(smooth - turning)) < 0.05
    near, _, _ = runner.smooth_court_track(moving + [0, 7], "near", 25, settings)
    assert np.allclose(near, moving + [0, 7])

    gaps = np.tile([3.0, 3.0], (40, 1))
    gaps[[0, 5, 6, 39]] = np.nan
    gaps[12:18] = np.nan
    gaps[25], gaps[27], gaps[30], gaps[35] = [100, 3], [3, 12], [4.5, 3], [np.inf, 3]
    conservative, statuses, reasons = runner.smooth_court_track(gaps, "far", 25, settings)
    assert np.isnan(conservative[[0, 39]]).all() and np.isnan(conservative[12:18]).all()
    assert list(statuses[5:7]) == ["short_interpolation"] * 2
    assert reasons[25] == "court_bounds" and reasons[27] == "court_half"
    assert reasons[30] == "temporal_outlier" and reasons[35] == "nonfinite_projection"
    continuous, statuses, _ = runner.smooth_court_track(
        gaps, "far", 25, {**settings, "gap_policy": "continuous"})
    assert np.allclose(continuous, 3)
    assert list(statuses[12:18]) == ["long_interpolation"] * 6
    assert list(statuses[[0, 39]]) == ["endpoint_hold"] * 2
    # Gap length is measured in seconds, while filter windows stay in frames.
    faster, statuses, _ = runner.smooth_court_track(gaps, "far", 30, settings)
    assert np.isfinite(faster[12:18]).all() and set(statuses[12:18]) == {"short_interpolation"}
    for length in (1, 2, 3, 4):
        short = moving[:length]
        smooth, _, _ = runner.smooth_court_track(
            short, "far", 25, {**settings, "median_window": 101, "savgol_window": 101})
        assert smooth.shape == short.shape and np.allclose(smooth, short)
    for policy in ("conservative", "continuous"):
        missing, statuses, _ = runner.smooth_court_track(
            np.full((12, 2), np.nan), "far", 25, {**settings, "gap_policy": policy})
        assert np.isnan(missing).all() and set(statuses) == {"unavailable"}


def main() -> None:
    (runner.ROOT / "checks").mkdir(parents=True, exist_ok=True)
    check_coordinates_targets_and_alignment()
    check_spawned_data_loading()
    check_whole_rally_exclusion()
    config = check_splits_and_config()
    check_training_progress()
    check_explicit_review_selection()
    check_deterministic_review_and_render()
    check_review_smoothing()
    model_summary = runner.run_model_checks(config)
    print(json.dumps({"status": "passed", "checks": [
        "coordinate scaling and pixel-centre conversions", "A/B alignment and far/near target channels",
        "normalised Gaussian targets and peak decoding", "whole-rally flaw exclusion",
        "four spawned DataLoader workers match direct loading with and without horizontal flips",
        "match-grouped split disjointness and fold validation", "training config reaches AdamW",
        "sparse flushed batch progress, epoch summaries, and early stopping in CV/final training",
        "deterministic point-only review rendering and failure accounting",
        "review smoothing, YAML windows, gap policies, inverse projection, and raw export compatibility",
        "complete pretrained transfer, gradients, independent initialization, checkpoint reload",
    ], "model": model_summary}, indent=2))


if __name__ == "__main__":
    main()
