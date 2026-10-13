"""Noninteractive checks for the one-rally MonoTrack adaptation."""

from __future__ import annotations

import argparse
import inspect
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
    pixel_residual_cost,
    project_point,
    reconstruct_flight,
    simulate_trajectory,
    validate_pixel_loss,
    visible_observations,
)
from compare_losses import (
    _matplotlib_pyplot,
    _new_output_dir,
    evaluate_trajectories,
    fit_loss_pair,
    load_ground_truth,
    run_comparison,
    write_error_plot,
)
from run_projection import (
    DEFAULT_HOMOGRAPHY,
    DEFAULT_VIDEO,
    assign_paths,
    draw_calibration_preview,
    build_parser as build_projection_parser,
    fit_hit_flights,
    fit_terminal_tail,
    run_projection,
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


def test_fit_references(hits: list[dict], geometry, projection: np.ndarray, frame_count: int) -> None:
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
        _, _, _, paths, diagnostics, _ = fit_hit_flights(
            test_hits, players, [None] * frame_count, projection, 25, "huber", 7.5
        )
    assert len(calls) == len(diagnostics) == len(test_hits) - 1 and not paths
    for index, (call, detail) in enumerate(zip(calls, diagnostics)):
        row = test_hits[index]
        prefix = "player_location" if index == 0 else "hit"
        expected_start = geometry.annotation_floor_to_world([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])])
        expected_end = geometry.annotation_floor_to_world([float(row["landing_x"]), float(row["landing_y"])])
        assert np.allclose(call["start_reference_xy"], expected_start)
        assert np.allclose(call["receive_reference_xy"], expected_end)
        assert call["start_frame"] == row["clip_frame_num"] and call["end_frame"] == test_hits[index + 1]["clip_frame_num"]
        assert call["fps"] == 25 and call["start_fixed"] is None and "end_fixed" not in call
        assert call["pixel_loss"] == detail["pixel_loss"] == "huber"
        assert call["huber_delta_px"] == detail["huber_delta_px"] == 7.5
        assert np.allclose(detail["start_reference_xy_m"], expected_start)
        assert np.allclose(detail["receive_reference_xy_m"], expected_end)
        assert detail["start_reference_source"] == ("first_row_player_location" if index == 0 else "current_row_hit_x_y")
        assert detail["receive_reference_source"] == "current_row_landing_x_y"
        assert detail["start_fixed"] is False and detail["start_reference_error_m"] == 1.0 and detail["receive_reference_error_m"] == 2.0

    # Missing, malformed, and nonfinite annotations must fail before any fitter call.
    for index, field, value in ((0, "hit_x", "nan"), (0, "hit_y", "broken"), (0, "landing_x", ""),
                                 (1, "hit_x", ""), (1, "hit_y", None), (2, "landing_y", "broken"),
                                 (len(hits) - 2, "landing_x", "inf")):
        invalid = [dict(row) for row in hits]
        invalid[index][field] = value
        invalid_players = player_references(invalid, geometry)
        with patch("run_projection.reconstruct_flight") as fitter:
            try:
                fit_hit_flights(invalid, invalid_players, [None] * frame_count, projection, 25)
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
                fit_hit_flights(test_hits, invalid_players, [None] * frame_count, projection, 25)
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
                                 [None] * frame_count, projection, 25)[4][0]
    assert detail["start_reference_source"] == "current_row_hit_x_y"

    # Continuity still carries a fitted 3D start rather than imposing the annotated x/y.
    fixed_point = np.asarray([3.0, 5.5, 1.8])
    observations = [None] * frame_count
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
    assert all(call["pixel_loss"] == "mse" and call["huber_delta_px"] == 10.0 for call in calls)
    assert diagnostics[1]["start_fixed"] is True and diagnostics[1]["start_reference_source"] == "current_row_hit_x_y"


def test_pixel_losses() -> None:
    errors = np.asarray([9.0, 10.0, 11.0])
    assert pixel_residual_cost(errors, "mse", 10.0) == float(np.mean(np.square(errors)))
    assert pixel_residual_cost(errors, "huber", 10.0) == float(np.mean([81.0, 100.0, 120.0]))
    assert validate_pixel_loss("huber", 10.0) == ("huber", 10.0)
    for loss, delta in (("other", 10.0), ("mse", 0.0), ("huber", -1.0), ("huber", np.inf), ("huber", np.nan)):
        try:
            validate_pixel_loss(loss, delta)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid pixel loss configuration: {loss}, {delta}")
    with patch("reconstruction.minimize") as optimizer:
        try:
            reconstruct_flight(
                start_frame=0, end_frame=2, observations=[(0.0, 0.0)] * 3,
                projection=np.eye(3, 4), start_reference_xy=np.zeros(2), receive_reference_xy=np.ones(2),
                fps=25.0, pixel_loss="bad",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("reconstruct_flight accepted an invalid pixel loss")
        optimizer.assert_not_called()
    with patch("run_projection.track_shuttle") as tracker:
        try:
            run_projection(argparse.Namespace(pixel_loss="bad", huber_delta_px=10.0))
        except ValueError:
            pass
        else:
            raise AssertionError("run_projection did not reject an invalid loss before inference")
        tracker.assert_not_called()
    with patch("compare_losses.track_shuttle") as tracker:
        try:
            run_comparison(argparse.Namespace(huber_delta_px=0.0))
        except ValueError:
            pass
        else:
            raise AssertionError("compare_losses did not reject an invalid delta before inference")
        tracker.assert_not_called()
    args = build_projection_parser().parse_args(["run", "--calibration", "calibration.json"])
    assert args.pixel_loss == "mse" and args.huber_delta_px == 10.0
    args = build_projection_parser().parse_args([
        "run", "--calibration", "calibration.json", "--pixel-loss", "huber", "--huber-delta-px", "7.5"
    ])
    assert args.pixel_loss == "huber" and args.huber_delta_px == 7.5


def test_ground_truth_evaluation() -> None:
    with tempfile.TemporaryDirectory(prefix="gt-check-", dir=Path(__file__).resolve().parent) as temp_name:
        gt_csv = Path(temp_name) / "gt.csv"
        gt_csv.write_text(
            "Frame,Visibility,X,Y\n0,1,10,10\n1,0,,\n2,1,12,10\n3,1,14,10\n4,1,16,10\n",
            encoding="utf-8",
        )
        gt = load_ground_truth(gt_csv, 2, 8, 1920, 1080)
        invalid_gt = (
            "Frame,Visibility,X\n0,1,1\n",
            "Frame,Visibility,X,Y\n0,1,1,1\n0,0,,\n",
            "Frame,Visibility,X,Y\n0,2,1,1\n",
            "Frame,Visibility,X,Y\n0,1,nan,1\n",
        )
        for index, content in enumerate(invalid_gt):
            invalid_path = Path(temp_name) / f"invalid_{index}.csv"
            invalid_path.write_text(content, encoding="utf-8")
            try:
                load_ground_truth(invalid_path, 2, 8, 1920, 1080)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid ground-truth CSV: {index}")
        mapped_path = Path(temp_name) / "mapped_outside.csv"
        mapped_path.write_text("Frame,Visibility,X,Y\n0,1,1,1\n", encoding="utf-8")
        try:
            load_ground_truth(mapped_path, -1, 8, 1920, 1080)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted a GT frame mapped outside the cleaned video")
        negative_path = Path(temp_name) / "negative_frame.csv"
        negative_path.write_text("Frame,Visibility,X,Y\n-1,1,1,1\n", encoding="utf-8")
        try:
            load_ground_truth(negative_path, 2, 8, 1920, 1080)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted a negative GT Frame ID")
    assert gt[0]["gt_frame"] == 0 and gt[0]["clean_frame"] == 2
    assert gt[0]["x_clean_px"] == 15.0 and gt[0]["y_clean_px"] == 15.0
    assert gt[1]["visibility"] == 0 and gt[1]["x_clean_px"] is None

    projection = np.asarray([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
    mse_coordinates = np.full((8, 3), np.nan)
    huber_coordinates = np.full((8, 3), np.nan)
    mse_coordinates[2] = [20.0, 15.0, 0.0]
    huber_coordinates[2] = [15.0, 15.0, 0.0]
    huber_coordinates[4] = [18.0, 20.0, 0.0]
    mse_coordinates[5] = huber_coordinates[5] = [20.0, 20.0, 0.0]
    mse_coordinates[6] = huber_coordinates[6] = [20.0, 20.0, 0.0]
    mse_phase, huber_phase = [""] * 8, [""] * 8
    for frame in (2, 4, 5, 6):
        mse_phase[frame] = huber_phase[frame] = "synthetic_flight"
    mse_fit = {"coordinates": mse_coordinates, "phases": mse_phase}
    huber_fit = {"coordinates": huber_coordinates, "phases": huber_phase}
    rows, coverage, metrics = evaluate_trajectories(gt, mse_fit, huber_fit, projection, 0, 4, 5)
    assert rows[0]["mse_error_px"] == 5.0 and rows[0]["huber_error_px"] == 0.0
    assert rows[1]["eligibility_reason"] == "invisible_ground_truth" and rows[1]["mse_error_px"] is None
    assert rows[2]["mse_error_px"] is None and rows[2]["huber_error_px"] == 5.0
    assert rows[3]["eligibility_reason"] == "manual_floor_contact_constraint" and not rows[3]["comparison_eligible"]
    assert rows[4]["eligibility_reason"] == "post_contact" and not rows[4]["comparison_eligible"]
    assert coverage["eligible_visible_gt_frames"] == 2
    assert coverage["mse_available_frames"] == 1 and coverage["huber_available_frames"] == 2
    assert coverage["paired_available_frames"] == 1
    assert metrics["overall"]["mse"]["paired_metrics"]["rmse_px"] == 5.0
    assert metrics["overall"]["huber"]["paired_metrics"]["rmse_px"] == 0.0
    assert metrics["per_flight"]["synthetic_flight"]["huber"]["individual_metrics"]["count"] == 2
    assert metrics["hit_to_hit_baseline"]["coverage"]["paired_available_frames"] == 1
    assert metrics["final_flight_extension"]["huber"]["individual_metrics"]["rmse_px"] == 5.0
    with tempfile.TemporaryDirectory(prefix="plot-check-", dir=Path(__file__).resolve().parent) as temp_name:
        plot_path = Path(temp_name) / "errors.png"
        write_error_plot(plot_path, rows, _matplotlib_pyplot())
        assert plot_path.is_file() and plot_path.stat().st_size > 0

    empty = {"coordinates": np.full((8, 3), np.nan), "phases": [""] * 8}
    _, empty_coverage, empty_metrics = evaluate_trajectories(gt, empty, empty, projection, 0, 4, 5)
    assert empty_coverage["paired_available_frames"] == 0
    assert empty_metrics["overall"]["mse"]["paired_metrics"] == {
        "count": 0, "rmse_px": None, "mean_euclidean_px": None, "median_px": None,
        "p95_px": None, "maximum_px": None,
    }


def test_ground_truth_fit_separation() -> None:
    assert "ground_truth" not in inspect.signature(fit_loss_pair).parameters
    hits, players, observations = [], [], [None] * 3
    projection, calibration, geometry = np.eye(3, 4), {"manual": True}, object()
    fitted = {"coordinates": np.full((3, 3), np.nan), "phases": [""] * 3,
              "hit_diagnostics": [], "terminal_diagnostics": {"phases": []}}
    with patch("compare_losses.fit_chain", return_value=fitted) as fit:
        fit_loss_pair(hits, players, observations, projection, 25.0, calibration, geometry, 10.0)
    assert fit.call_count == 2
    mse_call, huber_call = (call.args for call in fit.call_args_list)
    for left, right in zip(mse_call[:7], huber_call[:7]):
        assert left is right
    assert mse_call[7:] == ("mse", 10.0) and huber_call[7:] == ("huber", 10.0)


def test_comparison_output_protection() -> None:
    with tempfile.TemporaryDirectory(prefix="compare-output-check-", dir=Path(__file__).resolve().parent) as temp_name:
        work = Path(temp_name)
        existing = work / "existing"
        existing.mkdir()
        preserved = existing / "keep.txt"
        preserved.write_text("keep", encoding="utf-8")
        try:
            _new_output_dir(existing)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Comparison output directory did not refuse overwrite")
        assert preserved.read_text(encoding="utf-8") == "keep"
        assert _new_output_dir(work / "new").is_dir()


def run_checks(repo_root: Path, checkpoint: Path | None = None) -> None:
    repo_root = repo_root.resolve()
    video = repo_root / DEFAULT_VIDEO.relative_to(DEFAULT_VIDEO.parents[4])
    hits_csv = video.with_suffix(".csv")
    homography_csv = repo_root / DEFAULT_HOMOGRAPHY.relative_to(DEFAULT_HOMOGRAPHY.parents[1])
    metadata = video_metadata(video)
    assert (metadata["width"], metadata["height"], metadata["fps"]) == (1920, 1080, 25.0)
    assert metadata["frame_count"] > 0
    frame_count = metadata["frame_count"]
    hits = load_annotations(hits_csv, frame_count, metadata["fps"])
    assert hits
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
    test_fit_references(hits, geometry, fitted, frame_count)
    test_pixel_losses()
    test_ground_truth_evaluation()
    test_ground_truth_fit_separation()
    test_comparison_output_protection()
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
    assert detail["pixel_loss"] == "mse" and detail["huber_delta_px"] == 10.0
    assert np.allclose(phase[0], truth[0], atol=1e-6) and np.allclose(phase[-1], truth[-1], atol=0.01)

    direct_start = np.asarray([3.0, 12.0, 1.8])
    direct_end = np.asarray([7.0, 15.0, 0.0])
    direct_truth = simulate_trajectory(direct_start, np.asarray([4.0, 3.0, 3.1]), 0.0, 25, 25.0)
    direct_observations = [None] * 100 + [tuple(project_point(fitted, point)) for point in direct_truth]
    terminal = {"terminal_events": {
        "net_contact": {"status": "unidentified", "frame": None, "pixel_native": None},
        "floor_contact": {"status": "manual", "frame": 125, "pixel_native": [500.0, 500.0]},
    }}
    with patch.object(type(geometry), "native_floor_to_world", return_value=direct_end[:2]):
        direct_paths, direct_detail, floor_world = fit_terminal_tail(
            terminal, geometry, [{"clip_frame_num": 100}], direct_observations, fitted, 25.0,
            np.asarray([direct_start]), "huber", 8.0,
        )
    assert direct_detail["floor_contact_inside_court"] is False
    assert np.allclose(floor_world, direct_end) and direct_paths[0]["phase"] == "last_hit_to_floor_contact"
    assert direct_detail["phases"][-1]["status"] == "available"
    assert direct_detail["phases"][-1]["pixel_loss"] == "huber"
    assert np.allclose(direct_paths[0]["trajectory"][0], direct_start, atol=1e-6)
    assert np.allclose(direct_paths[0]["trajectory"][-1], direct_end, atol=0.01)

    calibration_preview = draw_calibration_preview(
        np.zeros((1080, 1920, 3), dtype=np.uint8), geometry, "left_post_top", {}, 0, frame_count, 2.0 / 3.0
    )
    assert calibration_preview.shape == (720, 1280, 3)
    render_probe = render_frame(
        np.zeros((1080, 1920, 3), dtype=np.uint8), 0, [None] * frame_count,
        np.full((frame_count, 3), np.nan), [""] * frame_count, [], [], fitted,
    )
    assert render_probe.shape == (720, 1440, 3)
    display_players = [{"frame": frame, "A": np.asarray([1.0, 2.0]), "B": np.asarray([5.0, 11.0]),
                        "endpoints": endpoints} for frame, endpoints in (
        (25, {"Serve fallback": np.asarray([2.0, 4.0]), "Landing": np.asarray([4.0, 9.0])}),
        (37, {"Hit": np.asarray([4.0, 9.0]), "Landing": np.asarray([3.0, 11.0])}),
    )]
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    marker_frame = lambda frame, refs: render_frame(blank, frame, [None] * frame_count,
                                                   np.full((frame_count, 3), np.nan), [""] * frame_count, [], refs, fitted)
    no_endpoints = [{**item, "endpoints": {}} for item in display_players]
    rect_width = round(520 * COURT_WIDTH_M / COURT_LENGTH_M)
    rect = (720 + (720 - rect_width) // 2, 100, rect_width, 520)
    for frame, index in ((25, 0), (36, 0), (37, 1), (frame_count - 1, 1)):
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

    print("Self-check passed: dynamic Fuzhou metadata and hit timing, scaling and homography, camera calibration, MSE/Huber behavior and propagation, independent-GT evaluation masks/metrics, GT/fitting separation, direct out-of-court terminal fit, continuity, rendering, and overwrite refusal. No real-rally fitting or video rendering.")
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
