from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from runner import align_hit_locations, fresh_output_dir, metrics_summary, select_candidates
from reconstruction import CourtGeometry


def run() -> None:
    matrix = np.eye(3, dtype=np.float64)
    geometry = CourtGeometry(
        "synthetic", "synthetic", matrix,
        np.asarray([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], dtype=np.float64),
        np.zeros((4, 2)), np.zeros((4, 2)), (0.0, 0.0, 1.0, 1.0),
    )
    world = geometry.native_floor_to_world((300.0, 150.0))
    assert np.allclose(world, (2.0, 1.0))
    assert np.allclose(geometry.annotation_floor_to_world((200.0, 100.0)), world)
    assert np.allclose(geometry.world_to_annotation_floor(world), (200.0, 100.0))

    row_a = {"player": "A", "player_location_x": 10, "player_location_y": 20,
             "opponent_location_x": 30, "opponent_location_y": 40}
    row_b = {**row_a, "player": "B"}
    assert align_hit_locations(row_a) == {"A": (10.0, 20.0), "B": (30.0, 40.0)}
    assert align_hit_locations(row_b) == {"B": (10.0, 20.0), "A": (30.0, 40.0)}

    class ProjectionStub:
        @staticmethod
        def native_floor_to_world(point):
            return np.asarray([point[0] - 3.0, point[1]])

    detections = {
        "boxes": np.asarray([
            [5, 4, 7, 5], [5, 6, 7, 7], [5, 3, 7, 4],
            [0.8, 3, 2.8, 4], [17, 6, 19, 7], [5, 6, 7, 7],
            [5, 8, 4, 9], [np.nan, 3, 2, 4],
        ]),
        "scores": np.asarray([0.8, 0.9, 0.29, 0.99, 0.98, 0.999, 0.99, 0.99]),
        "labels": np.asarray([1, 1, 1, 1, 1, 3, 1, 1]),
    }
    selected = select_candidates(detections, ProjectionStub(), (20, 20))
    assert selected["far"]["score"] == 0.8 and selected["far"]["court"] == (3.0, 5.0)
    assert selected["near"]["score"] == 0.9 and selected["near"]["court"] == (3.0, 7.0)
    empty = select_candidates({"boxes": np.empty((0, 4)), "scores": [], "labels": []}, ProjectionStub())
    assert empty == {"far": None, "near": None}

    metric_rows = [
        {"player": "A", "prediction_status": "available", "error_px": 12.0, "error_m": 0.2},
        {"player": "B", "prediction_status": "missing", "error_px": None, "error_m": None},
    ]
    metrics = metrics_summary(metric_rows)
    assert metrics["combined"]["available_predictions"] == 1
    assert metrics["combined"]["coverage_percentage"] == 50.0
    assert metrics["combined"]["metre_error"]["count"] == 1
    assert metrics["combined"]["metre_error"]["euclidean_rmse"] == 0.2
    assert metrics["combined"]["within_0_5_m"] == {"count": 1, "denominator": 2, "percentage": 50.0}
    assert metrics["combined"]["within_1_m"]["percentage"] == 50.0

    with tempfile.TemporaryDirectory(prefix="_self_check_", dir=Path(__file__).resolve().parent) as temp:
        root = Path(temp)
        first = fresh_output_dir(root, "same")
        (first / "keep.txt").write_text("preserved", encoding="utf-8")
        second = fresh_output_dir(root, "same")
        assert first != second and first.is_relative_to(root) and second.is_relative_to(root)
        assert (first / "keep.txt").read_text(encoding="utf-8") == "preserved"
        assert first.name == "run_same" and second.name == "run_same_02"

    print("Self-check passed: scaling/round-trip, A/B reference alignment, candidate filters and halves, "
          "missing-prediction denominators, and contained non-overwriting output directories.")


if __name__ == "__main__":
    run()
