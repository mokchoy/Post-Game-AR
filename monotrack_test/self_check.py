"""Noninteractive checks for the one-rally MonoTrack adaptation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reconstruction import (
    ANNOTATION_TO_VIDEO,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    calibration_world_points,
    fit_camera_projection,
    frame_time_seconds,
    intersect_camera_ray_y,
    join_piecewise,
    load_court_geometry,
    project_point,
    reconstruct_flight,
    simulate_trajectory,
    visible_observations,
)
from run_projection import (
    DEFAULT_HOMOGRAPHY,
    DEFAULT_VIDEO,
    assign_paths,
    draw_calibration_preview,
    fit_hit_flights,
    hold_terminal_marker,
    player_references,
    panel_point,
    render_frame,
    reserve_output_files,
    repo_path,
    video_metadata,
    load_annotations,
)


def synthetic_camera() -> np.ndarray:
    center = np.asarray([3.05, -20.0, 8.0])
    target = np.asarray([3.05, 6.7, 0.0])
    forward = (target - center) / np.linalg.norm(target - center)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.vstack((right, up, forward))
    intrinsic = np.asarray([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
    extrinsic = np.column_stack((rotation, -rotation @ center))
    projection = intrinsic @ extrinsic
    return projection / projection[2, 3]


def test_fit_references(hits: list[dict], geometry, projection: np.ndarray) -> None:
    # A different landing from the next hit verifies that the current row is authoritative.
    test_hits = [dict(row) for row in hits]
    test_hits[2]["landing_x"] = str(float(test_hits[2]["landing_x"]) + 10)
    players = player_references(test_hits, geometry)
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return None, {"phase": kwargs["phase"], "status": "insufficient_data",
                      "start_fixed": kwargs["start_fixed"] is not None,
                      "start_reference_error_m": 1.0, "receive_reference_error_m": 2.0}

    with patch("run_projection.reconstruct_flight", side_effect=spy):
        _, _, _, paths, diagnostics, _ = fit_hit_flights(test_hits, players, [None] * 524, projection, 25)
    assert len(calls) == len(diagnostics) == 17 and not paths
    for index, (call, detail) in enumerate(zip(calls, diagnostics)):
        row = test_hits[index]
        prefix = "player_location" if index == 0 else "hit"
        expected_start = geometry.annotation_floor_to_world([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])])
        expected_end = geometry.annotation_floor_to_world([float(row["landing_x"]), float(row["landing_y"])])
        assert np.allclose(call["start_reference_xy"], expected_start)
        assert np.allclose(call["receive_reference_xy"], expected_end)
        assert call["start_frame"] == row["clip_frame_num"] and call["end_frame"] == test_hits[index + 1]["clip_frame_num"]
        assert call["fps"] == 25 and call["start_fixed"] is None and "end_fixed" not in call
        assert np.allclose(detail["start_reference_xy_m"], expected_start)
        assert np.allclose(detail["receive_reference_xy_m"], expected_end)
        assert detail["start_reference_source"] == ("first_row_player_location" if index == 0 else "current_row_hit_x_y")
        assert detail["receive_reference_source"] == "current_row_landing_x_y"
        assert detail["start_fixed"] is False and detail["start_reference_error_m"] == 1.0 and detail["receive_reference_error_m"] == 2.0

    # Missing, malformed, and nonfinite annotations must fail before any fitter call.
    for index, field, value in ((0, "hit_x", "nan"), (0, "hit_y", "broken"), (0, "landing_x", ""),
                                 (1, "hit_x", ""), (1, "hit_y", None), (2, "landing_y", "broken"),
                                 (16, "landing_x", "inf")):
        invalid = [dict(row) for row in hits]
        invalid[index][field] = value
        invalid_players = player_references(invalid, geometry)
        with patch("run_projection.reconstruct_flight") as fitter:
            try:
                fit_hit_flights(invalid, invalid_players, [None] * 524, projection, 25)
            except ValueError as error:
                assert f"Stroke {index + 1}" in str(error) and field.split("_")[0] in str(error)
            else:
                raise AssertionError(f"Accepted invalid endpoint {field}={value!r}")
            fitter.assert_not_called()
    for point in ([np.nan, 1.0], [1.0, np.inf], [1.0], ["broken", 1.0]):
        invalid_players = [{**row, "endpoints": dict(row["endpoints"])} for row in players]
        invalid_players[1]["endpoints"]["Hit"] = point
        with patch("run_projection.reconstruct_flight") as fitter:
            try:
                fit_hit_flights(test_hits, invalid_players, [None] * 524, projection, 25)
            except ValueError as error:
                assert "Stroke 2" in str(error) and "hit_x/y" in str(error)
            else:
                raise AssertionError(f"Accepted invalid transformed endpoint {point!r}")
            fitter.assert_not_called()
    valid_first_hit = [dict(row) for row in hits]
    valid_first_hit[0].update(hit_x="640.0", hit_y="360.0")
    calls.clear()
    with patch("run_projection.reconstruct_flight", side_effect=spy):
        detail = fit_hit_flights(valid_first_hit, player_references(valid_first_hit, geometry),
                                 [None] * 524, projection, 25)[4][0]
    assert detail["start_reference_source"] == "current_row_hit_x_y"

    # Continuity still carries a fitted 3D start rather than imposing the annotated x/y.
    fixed_point = np.asarray([3.0, 5.5, 1.8])
    observations = [None] * 524
    observations[hits[1]["clip_frame_num"]] = tuple(project_point(projection, fixed_point))
    calls.clear()

    def carry_spy(**kwargs):
        _, detail = spy(**kwargs)
        if len(calls) == 1:
            detail["status"] = "available"
            return np.tile(fixed_point, (kwargs["end_frame"] - kwargs["start_frame"] + 1, 1)), detail
        return None, detail

    with patch("run_projection.reconstruct_flight", side_effect=carry_spy):
        diagnostics = fit_hit_flights(hits[:3], players[:3], observations, projection, 25)[4]
    assert calls[0]["start_fixed"] is None and np.array_equal(calls[1]["start_fixed"], fixed_point)
    assert diagnostics[1]["start_fixed"] is True and diagnostics[1]["start_reference_source"] == "current_row_hit_x_y"


def run_checks(repo_root: Path, checkpoint: Path | None = None) -> None:
    repo_root = repo_root.resolve()
    video = repo_root / DEFAULT_VIDEO.relative_to(DEFAULT_VIDEO.parents[4])
    hits_csv = video.with_suffix(".csv")
    homography_csv = repo_root / DEFAULT_HOMOGRAPHY.relative_to(DEFAULT_HOMOGRAPHY.parents[1])
    metadata = video_metadata(video)
    assert (metadata["width"], metadata["height"], metadata["fps"], metadata["frame_count"]) == (1920, 1080, 25.0, 524)
    hits = load_annotations(hits_csv, metadata["frame_count"], metadata["fps"])
    assert len(hits) == 18
    assert all(abs(frame_time_seconds(hit["clip_frame_num"], 25.0) - hit["clip_time_seconds"]) <= 0.01 for hit in hits)

    geometry = load_court_geometry(homography_csv, video.parent.parent.name)
    assert np.allclose(geometry.mapped_bounds, [25.0, 150.0, 325.0, 810.0], atol=5.0)
    expected_dataset = [[25, 150], [325, 150], [25, 810], [325, 810]]
    expected_metric = [[0, 0], [COURT_WIDTH_M, 0], [0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M]]
    for index, pixel in enumerate(geometry.corners_annotation):
        assert np.allclose(geometry.mapped_corners[index], expected_dataset[index], atol=0.01)
        assert np.allclose(geometry.annotation_floor_to_world(pixel), expected_metric[index], atol=1e-5)
        world = geometry.annotation_floor_to_world(pixel)
        round_trip = geometry.world_to_annotation_floor(world)
        assert np.allclose(round_trip, pixel, atol=0.02)
        assert np.allclose(geometry.native_floor_to_world(pixel * ANNOTATION_TO_VIDEO), world, atol=1e-5)
    center_world = geometry.annotation_floor_to_world([640.0, 360.0])
    assert np.isfinite(center_world).all() and 0 <= center_world[0] <= COURT_WIDTH_M and 0 <= center_world[1] <= COURT_LENGTH_M
    players = player_references(hits, geometry)
    assert players[0]["hitter"] == "A" and players[1]["hitter"] == "B"
    assert np.allclose(players[0]["A"], geometry.native_floor_to_world([655.0 * 1.5, 544.0 * 1.5]))
    assert "Hit" not in players[0]["endpoints"]
    assert np.allclose(players[0]["endpoints"]["Serve fallback"], players[0]["A"])
    for index, (row, references) in enumerate(zip(hits, players)):
        for label, prefix in (("Hit", "hit"), ("Landing", "landing")):
            if index == 0 and label == "Hit":
                continue
            expected = geometry.annotation_floor_to_world([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])])
            assert np.allclose(references["endpoints"][label], expected)

    true_projection = synthetic_camera()
    world_points = calibration_world_points()
    image_points = np.asarray([project_point(true_projection, point) for point in world_points])
    fitted, camera_rmse = fit_camera_projection(image_points[:4], image_points[4:])
    assert camera_rmse < 1e-5
    test_fit_references(hits, geometry, fitted)
    sample = np.asarray([3.0, 6.7, 0.9])
    ray_hit = intersect_camera_ray_y(fitted, project_point(fitted, sample), 6.7)
    assert np.allclose(ray_hit, sample, atol=1e-5)
    truth = simulate_trajectory(np.asarray([3.0, 5.5, 1.8]), np.asarray([0.8, 2.0, 5.0]), 0.12, 20, 25.0)
    terminal_observations = [tuple(project_point(fitted, point)) for point in truth]
    phase, detail = reconstruct_flight(
        start_frame=100, end_frame=120, observations=[None] * 100 + terminal_observations,
        projection=fitted, start_reference_xy=truth[0, :2], receive_reference_xy=truth[-1, :2],
        fps=25.0, start_fixed=truth[0], end_fixed=truth[-1], phase="synthetic_terminal_endpoint",
    )
    assert phase is not None and detail["status"] == "available"
    assert np.allclose(phase[0], truth[0], atol=1e-6) and np.allclose(phase[-1], truth[-1], atol=0.01)
    calibration_preview = draw_calibration_preview(
        np.zeros((1080, 1920, 3), dtype=np.uint8), geometry, "left_post_top", {}, 0, 2.0 / 3.0
    )
    assert calibration_preview.shape == (720, 1280, 3)
    render_probe = render_frame(
        np.zeros((1080, 1920, 3), dtype=np.uint8), 0, [None] * 524,
        np.full((524, 3), np.nan), [""] * 524, [], [], fitted,
    )
    assert render_probe.shape == (720, 1440, 3)
    display_players = [{"frame": frame, "A": np.asarray([1.0, 2.0]), "B": np.asarray([5.0, 11.0]),
                        "endpoints": endpoints} for frame, endpoints in (
        (25, {"Serve fallback": np.asarray([2.0, 4.0]), "Landing": np.asarray([4.0, 9.0])}),
        (37, {"Hit": np.asarray([4.0, 9.0]), "Landing": np.asarray([3.0, 11.0])}),
    )]
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    marker_frame = lambda frame, refs: render_frame(blank, frame, [None] * 524,
                                                   np.full((524, 3), np.nan), [""] * 524, [], refs, fitted)
    no_endpoints = [{**item, "endpoints": {}} for item in display_players]
    rect_width = round(520 * COURT_WIDTH_M / COURT_LENGTH_M)
    rect = (720 + (720 - rect_width) // 2, 100, rect_width, 520)
    for frame, index in ((25, 0), (36, 0), (37, 1), (523, 1)):
        shown = marker_frame(frame, display_players)
        baseline = marker_frame(frame, no_endpoints)
        changed = np.any(shown != baseline, axis=2)
        assert not np.any(changed[:, :720]) and not np.any(changed[:80]) and not np.any(changed[641:])
        for label, xy in display_players[index]["endpoints"].items():
            point = panel_point(xy, rect)
            color = (0, 255, 0) if label == "Landing" else (255, 0, 255)
            assert tuple(shown[point[1], point[0] + 11]) == color
    assert np.array_equal(marker_frame(24, display_players), marker_frame(24, no_endpoints))
    old_hit = panel_point(display_players[0]["endpoints"]["Serve fallback"], rect)
    switched = marker_frame(37, display_players)
    assert not np.any(np.all(switched[old_hit[1] - 12:old_hit[1] + 13,
                                      old_hit[0] - 12:old_hit[0] + 13] == (255, 0, 255), axis=2))
    outside = [{**display_players[0], "endpoints": {"Hit": np.asarray([3.0, -100.0]),
                                                   "Landing": np.asarray([np.nan, 0.0])}}]
    assert np.array_equal(marker_frame(25, outside), marker_frame(25, no_endpoints))

    assert frame_time_seconds(25, 25.0) == 1.0
    assert frame_time_seconds(485, 25.0) == 19.4
    timed_path = simulate_trajectory(np.asarray([0.0, 0.0, 10.0]), np.asarray([1.0, 0.0, 0.0]), 0.0, 25, 25.0)
    assert timed_path.shape == (26, 3) and np.array_equal(timed_path[0], [0.0, 0.0, 10.0])
    assert abs(timed_path[25, 0] - 1.0) < 1e-10
    path_a = np.asarray([[2.0, 5.0, 1.0], [3.0, 6.7, 0.8]])
    path_b = np.asarray([[3.0, 6.7, 0.8], [4.0, 8.0, 0.0]])
    joined = join_piecewise(path_a, path_b)
    assert joined.shape == (3, 3) and np.array_equal(joined[1], path_a[-1])
    piecewise = [
        {"start_frame": 2, "end_frame": 3, "trajectory": path_a, "phase": "last_hit_to_net_contact"},
        {"start_frame": 3, "end_frame": 4, "trajectory": path_b, "phase": "net_contact_to_floor_contact"},
    ]
    output_positions = np.full((6, 3), np.nan)
    output_phases, output_segments = [""] * 6, [-1] * 6
    assign_paths(output_positions, output_phases, output_segments, piecewise, tail=True)
    assert np.array_equal(output_positions[3], path_a[-1])
    assert output_phases[3] == "net_contact_to_floor_contact"
    floor_point = np.asarray([4.0, 9.0, 0.0])
    hold_terminal_marker(output_positions, output_phases, output_segments, 4, floor_point)
    assert np.allclose(output_positions[4:], floor_point)
    missing = [None, (10.0, 20.0), None, (30.0, 40.0)]
    assert [index for index, _ in visible_observations(missing)] == [1, 3]
    unavailable, detail = reconstruct_flight(
        start_frame=0, end_frame=3, observations=missing, projection=fitted,
        start_reference_xy=np.asarray([2.0, 4.0]), receive_reference_xy=np.asarray([4.0, 9.0]),
        fps=25.0,
    )
    assert unavailable is None and detail["status"] == "insufficient_data"

    with tempfile.TemporaryDirectory(prefix="self-check-", dir=Path(__file__).resolve().parent) as temp_name:
        work = Path(temp_name)
        preserved = work / "existing.txt"
        new_file = work / "new.txt"
        preserved.write_text("keep", encoding="utf-8")
        try:
            reserve_output_files([new_file, preserved])
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output did not refuse overwrite")
        assert preserved.read_text(encoding="utf-8") == "keep"
        assert not new_file.exists()

    print("Self-check passed: scaling, dataset-to-metre bounds/round trips, camera/ray calibration, annotated fit references/provenance and serve fallback, invalid endpoint rejection, continuity, frame timing, constrained terminal endpoint, shared contact, rendering layout, current-stroke hit/landing markers, panel clipping, missing detections, player identity, and overwrite refusal. No real-rally fitting or video rendering.")
    if checkpoint is not None:
        tracker_smoke(repo_path(checkpoint))


def tracker_smoke(checkpoint: Path) -> None:
    import torch

    from run_projection import load_tracker

    model, device, tracker, to_tensor = load_tracker(checkpoint)
    sample_frames = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    batch = tracker.preprocess_group(sample_frames, device, to_tensor)
    assert tuple(batch.shape) == (1, 9, 288, 512)
    with torch.inference_mode():
        prediction = model(batch)
    assert tuple(prediction.shape) == (1, 3, 288, 512)
    assert bool(torch.isfinite(prediction).all())
    print(f"TrackNet integration smoke passed: {device}, batch={tuple(batch.shape)}, output={tuple(prediction.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoint", type=Path, help="Also strict-load track.pt and run one synthetic forward pass")
    args = parser.parse_args()
    try:
        run_checks(args.repo_root, args.checkpoint)
    except Exception as error:
        print(f"self_check: failed: {error}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
