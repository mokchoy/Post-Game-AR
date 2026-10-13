from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

import runner


def check_geometry_and_alignment() -> None:
    assert runner.native_point((640.0, 360.0), 1920, 1080) == (960.0, 540.0)
    row = {
        "player": "A", "player_location_x": "300", "player_location_y": "200",
        "opponent_location_x": "900", "opponent_location_y": "600",
    }
    assert runner.align_references(row) == {"A": (300.0, 200.0), "B": (900.0, 600.0)}
    row["player"] = "B"
    assert runner.align_references(row) == {"A": (900.0, 600.0), "B": (300.0, 200.0)}


class GeometryStub:
    @staticmethod
    def annotation_floor_to_world(point):
        return np.asarray([point[0] / 100.0, point[1] / 50.0])


def check_rally_exclusion_and_boxes() -> None:
    (runner.ROOT / "smoke").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="self_check_", dir=runner.ROOT / "smoke") as temp:
        match_dir = Path(temp) / "synthetic_match"
        clip_dir = match_dir / "set1"
        clip_dir.mkdir(parents=True)
        clip = clip_dir / "rally_001.avi"
        writer = cv2.VideoWriter(
            str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720)
        )
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.zeros((720, 1280, 3), dtype=np.uint8))
        writer.release()
        annotation_rows = []
        for frame in range(3):
            hitter = "A" if frame % 2 == 0 else "B"
            a, b = (500.0, 100.0), (700.0, 600.0)
            player_xy, opponent_xy = (a, b) if hitter == "A" else (b, a)
            annotation_rows.append({
                "player": hitter, "player_location_x": player_xy[0],
                "player_location_y": player_xy[1], "opponent_location_x": opponent_xy[0],
                "opponent_location_y": opponent_xy[1], "flaw": "0",
                "frame_num": str(frame), "clip_frame_num": str(frame),
                "source_time_seconds": f"{frame / 10:.6f}",
                "clip_time_seconds": f"{frame / 10:.6f}",
            })
        annotation_path = clip.with_suffix(".csv")
        with annotation_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(annotation_rows[0]))
            writer.writeheader()
            writer.writerows(annotation_rows)
        inventory = {
            "dataset": "set-1", "match": "synthetic", "game": "1",
            "original_rally_id": "1", "filename": "set1/rally_001.avi",
            "first_hit_frame": "0", "last_hit_frame": "2", "annotation_fps": "10",
            "time_offset": "0", "start_seconds": "0", "review_flags": "",
        }
        exclusions = []
        clean = runner.inspect_rally(
            match_dir, inventory, [{"outcome": "copied-unchanged", "review_flags": ""}],
            {"synthetic": GeometryStub()}, exclusions,
        )
        assert clean["status"] == "pre_box_pass" and clean["A_half"] == "far" and clean["B_half"] == "near"

        annotation_rows[1]["flaw"] = "1"
        with annotation_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(annotation_rows[0]))
            writer.writeheader()
            writer.writerows(annotation_rows)
        excluded = runner.inspect_rally(
            match_dir, inventory, [{"outcome": "copied-unchanged", "review_flags": ""}],
            {"synthetic": GeometryStub()}, exclusions,
        )
        assert excluded["status"] == "excluded"
        assert "dataset_flaw_flag" in excluded["reasons"]

    class TensorLike:
        def __init__(self, value):
            import torch
            self.value = torch.tensor(value, dtype=torch.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value.numpy()

    class Boxes:
        xyxy = TensorLike([[0, 0, 600, 700], [600, 0, 1200, 700]])
        conf = TensorLike([0.8, 0.7])
        cls = TensorLike([0, 0])

        def __len__(self):
            return len(self.xyxy.value)

    class Result:
        boxes = Boxes()

    found, reasons = runner.box_associations(
        Result(), {"A": (500, 100), "B": (700, 600)}, 0.30
    )
    assert not reasons and tuple(found["A"]) == (0.0, 0.0, 600.0, 700.0)
    assert tuple(found["B"]) == (600.0, 0.0, 1200.0, 700.0)


def check_splits_and_config() -> None:
    (runner.ROOT / "smoke").mkdir(exist_ok=True)
    matches = [f"m{i}" for i in range(12)]
    assignments = runner.grouped_folds(matches, 3, 42)
    assert set(assignments.values()) == {0, 1, 2}
    for invalid in (1, 13):
        try:
            runner.grouped_folds(matches, invalid, 42)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid fold count {invalid} was accepted")
    held_out = runner.fixed_test_matches(matches)
    development = sorted(set(matches) - set(held_out))
    split = {"development_matches": development, "held_out_matches": held_out}
    rows = [
        {"match": "m0"}, {"match": "m1"}, {"match": "m2"},
        {"match": held_out[0]},
    ]
    train, val = runner.cv_split_rows(rows, split, {"m0": 0, "m1": 1, "m2": 0}, 0)
    assert {row["match"] for row in train}.isdisjoint({row["match"] for row in val})
    assert not ({row["match"] for row in train} | {row["match"] for row in val}) & set(held_out)

    config = runner.load_config(folds_override=2)
    assert config["cv"]["folds"] == 2
    with tempfile.TemporaryDirectory(prefix="self_check_cfg_", dir=runner.ROOT / "smoke") as temp:
        invalid_config = {key: value for key, value in config.items() if not key.startswith("_")}
        invalid_config["cv"]["folds"] = 1
        path = Path(temp) / "invalid.yaml"
        path.write_text(yaml.safe_dump(invalid_config), encoding="utf-8")
        try:
            runner.load_config(str(path))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid YAML fold count was accepted")
    args = runner.build_training_args(
        {"epochs": 7, "lr0": 0.004, "workers": 0, "mosaic": 0.0, "amp": False},
        runner.ROOT / "data.yaml", runner.ROOT / "runs" / "self-check", "fold",
    )
    assert args["epochs"] == 7 and args["lr0"] == 0.004 and args["amp"] is False
    assert args["data"].endswith("data.yaml") and args["task"] == "pose"
    assert args["name"] == "fold" and args["project"].endswith("self-check")


def check_model_initialization_and_output() -> None:
    runner.ensure_weights()
    first = runner.init_custom_model(seed=91)
    second = runner.init_custom_model(seed=92)
    assert runner.model_contract(first) == runner.model_contract(second) == {
        "classes": 1, "keypoints": 1, "keypoint_dimensions": 3,
    }
    first_report, second_report = runner.transfer_report(first), runner.transfer_report(second)
    assert first_report["verified_transferred_tensors"] > 0
    assert first_report["verified_transferred_key_sha256"] == \
        second_report["verified_transferred_key_sha256"]
    torch, _, _, _ = runner.load_ultralytics()
    first_parameter = next(first.model.parameters())
    second_parameter = next(second.model.parameters())
    assert first_parameter.data_ptr() != second_parameter.data_ptr()
    before = second_parameter.detach().clone()
    with torch.no_grad():
        first_parameter.view(-1)[0].add_(1)
    assert torch.equal(before, second_parameter)
    device = 0 if torch.cuda.is_available() else "cpu"
    prediction = runner.assert_prediction_contract(first, device)
    assert prediction["keypoints_per_detection"] == 1


def main() -> None:
    check_geometry_and_alignment()
    check_rally_exclusion_and_boxes()
    check_splits_and_config()
    check_model_initialization_and_output()
    print(
        "Self-check passed: coordinate/A-B mapping, whole-rally flaw exclusion, "
        "unique box associations, match-group folds, split disjointness, config overrides, "
        "one-keypoint output, and independent checkpoint initialization."
    )


if __name__ == "__main__":
    main()
