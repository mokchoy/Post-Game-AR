from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

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
            logits = torch.full((1, 2, 32, 40), -20.0)
            logits[0, 0, 10, 20] = 20
            logits[0, 1, 25, 24] = 20
            self.register_buffer("logits", logits)

        def forward(self, image):
            return self.logits.expand(image.shape[0], -1, -1, -1)

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

    failures = runner.error_summary([
        {"available": True, "error_px": 10.0, "error_m": 0.2},
        {"available": False, "error_px": None, "error_m": None},
        {"available": False, "error_px": None, "error_m": None},
    ])
    assert failures["expected_predictions"] == 3 and failures["available_predictions"] == 1
    assert failures["coverage"] == 1 / 3 and failures["within_1_m"]["denominator"] == 3


def main() -> None:
    (runner.ROOT / "checks").mkdir(parents=True, exist_ok=True)
    check_coordinates_targets_and_alignment()
    check_whole_rally_exclusion()
    config = check_splits_and_config()
    check_deterministic_review_and_render()
    model_summary = runner.run_model_checks(config)
    print(json.dumps({"status": "passed", "checks": [
        "coordinate scaling and pixel-centre conversions", "A/B alignment and far/near target channels",
        "normalised Gaussian targets and peak decoding", "whole-rally flaw exclusion",
        "match-grouped split disjointness and fold validation", "training config reaches AdamW",
        "deterministic point-only review rendering and failure accounting",
        "complete pretrained transfer, gradients, independent initialization, checkpoint reload",
    ], "model": model_summary}, indent=2))


if __name__ == "__main__":
    main()
