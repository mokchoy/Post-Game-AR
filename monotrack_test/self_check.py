"""Noninteractive checks for the one-rally MonoTrack adaptation."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reconstruction import (
    ANNOTATION_TO_VIDEO,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    extrapolated_landing_distance,
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
    build_parser as build_comparison_parser,
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
    load_player_positions,
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
    test_hits = [dict(row) for row in hits]
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
        receiver = test_hits[index + 1]
        expected_start = geometry.annotation_floor_to_world([row["player_location_x"], row["player_location_y"]])
        expected_end = geometry.annotation_floor_to_world([receiver["player_location_x"], receiver["player_location_y"]])
        assert np.allclose(call["start_reference_xy"], expected_start)
        assert np.allclose(call["receive_reference_xy"], expected_end)
        assert call["start_frame"] == row["clip_frame_num"] and call["end_frame"] == test_hits[index + 1]["clip_frame_num"]
        assert call["fps"] == 25 and call["start_fixed"] is None and "end_fixed" not in call
        assert call["pixel_loss"] == detail["pixel_loss"] == "huber"
        assert call["huber_delta_px"] == detail["huber_delta_px"] == 7.5
        assert np.allclose(detail["start_reference_xy_m"], expected_start)
        assert np.allclose(detail["receive_reference_xy_m"], expected_end)
        assert detail["start_reference_source"] == "current_row_player_location"
        assert detail["receive_reference_source"] == "next_row_player_location"
        assert detail["start_fixed"] is False and detail["start_reference_error_m"] == 1.0 and detail["receive_reference_error_m"] == 2.0

    # Validate every transformed player reference before starting any flight.
    for row_index in (0, 1, len(hits) - 1):
        for point in (None, [np.nan, 1.0], [1.0, np.inf], [1.0], ["broken", 1.0]):
            invalid_players = [dict(row) for row in players]
            invalid_players[row_index][hits[row_index]["player"]] = point
            with patch("run_projection.reconstruct_flight") as fitter:
                try:
                    fit_hit_flights(test_hits, invalid_players, [None] * frame_count, projection, 25)
                except ValueError as error:
                    assert f"Stroke {max(1, row_index)}" in str(error) and "player_location_x/y" in str(error)
                else:
                    raise AssertionError(f"Accepted invalid transformed player position {point!r}")
                fitter.assert_not_called()

    # Continuity still carries a fitted 3D start rather than imposing the player x/y.
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
    assert diagnostics[1]["start_fixed"] is True and diagnostics[1]["start_reference_source"] == "current_row_player_location"

    # Optional shuttle labels and manual contacts cannot affect either loss's trajectories.
    original_hits = hits[:3]
    observations = [None] * frame_count
    baseline_calls = []

    def successful_spy(**kwargs):
        baseline_calls.append(kwargs)
        _, detail = spy(**kwargs)
        detail["status"] = "available"
        return np.tile(fixed_point, (kwargs["end_frame"] - kwargs["start_frame"] + 1, 1)), detail

    with patch("run_projection.reconstruct_flight", side_effect=successful_spy):
        baseline = fit_loss_pair(original_hits, players[:3], observations, projection, 25, {}, geometry, 10.0)
    original_calls = baseline_calls[:]
    for value in (None, "broken", "nan", "inf", "99999"):
        changed_hits = [dict(row) for row in original_hits]
        for row in changed_hits:
            for field in ("hit_x", "hit_y", "landing_x", "landing_y"):
                if value is None:
                    row.pop(field, None)
                else:
                    row[field] = value
        terminal = {"terminal_events": {
            name: {"status": "manual", "frame": frame_count - 1, "pixel_native": [100, 200]}
            for name in ("net_contact", "floor_contact")
        }}
        baseline_calls.clear()
        with patch("run_projection.reconstruct_flight", side_effect=successful_spy):
            changed = fit_loss_pair(changed_hits, player_references(changed_hits, geometry), observations,
                                    projection, 25, terminal, geometry, 10.0)
        assert len(baseline_calls) == len(original_calls) == 4
        for original, current in zip(original_calls, baseline_calls):
            for key in original:
                np.testing.assert_equal(current[key], original[key])
        for method in ("mse", "huber"):
            np.testing.assert_equal(changed[method]["coordinates"], baseline[method]["coordinates"])
            assert changed[method]["phases"] == baseline[method]["phases"]
            assert changed[method]["hit_diagnostics"] == baseline[method]["hit_diagnostics"]
            assert changed[method]["terminal_diagnostics"] == baseline[method]["terminal_diagnostics"]
            final_frame = changed_hits[-1]["clip_frame_num"]
            assert np.array_equal(changed[method]["coordinates"][final_frame], fixed_point)
            assert changed[method]["phases"][final_frame] == "hit_2_to_hit_3"
            assert np.isnan(changed[method]["coordinates"][final_frame + 1:]).all()
    assert np.isnan(fit_hit_flights(hits[:1], players[:1], observations, projection, 25)[0]).all()
    assert np.isnan(fit_hit_flights(hits[:2], players[:2], observations, projection, 25)[0]).all()


def test_saved_player_tracks(geometry, projection: np.ndarray) -> None:
    metadata = {"width": 1920, "height": 1080, "fps": 25.0, "frame_count": 4}
    hits = [{"player": "A", "clip_frame_num": 1, "clip_time_seconds": 0.04},
            {"player": "B", "clip_frame_num": 3, "clip_time_seconds": 0.12}]
    key = ["set-1", geometry.row_video, "1", "15"]
    rows = []
    for frame in range(4):
        for player, half, y in (("A", "far", 2.0), ("B", "near", 10.0)):
            world = np.asarray([1.0 + frame * 0.4 if player == "A" else 4.0, y])
            native = (geometry.world_to_annotation_floor(world) + 0.5) * 1.5 - 0.5
            filled = frame == 3 and player == "B"
            rows.append({**dict(zip(("dataset", "match", "game", "original_rally_id"), key)),
                         "clip_frame_num": frame, "clip_time_seconds": frame / 25,
                         "player": player, "court_half": half, "available": True,
                         "x_native": native[0], "y_native": native[1],
                         "x_court_m": world[0], "y_court_m": world[1],
                         "filter_status": "long_interpolation" if filled else "smoothed", "gap_filled": filled,
                         "reference_x_native": "ignored", "reference_y_native": "ignored"})
    rows[4].update(available=False, filter_status="unavailable", x_native="", y_native="",
                   x_court_m="", y_court_m="")
    metrics = {"source_dimensions": [1920, 1080], "decoded_frames": 4, "source_fps": 25,
               "smoothing": {"median_window": 5, "savgol_window": 9, "gap_policy": "continuous"}}
    selection = {"selected_rallies": [{"key": key, "clip_path": str(DEFAULT_VIDEO)}],
                 "final_run_id": "synthetic_final", "checkpoint": "saved-model.pt", "checkpoint_sha256": "a" * 64}
    with tempfile.TemporaryDirectory(prefix="player-track-check-", dir=repo_path("monotrack_test")) as directory:
        review = Path(directory)
        rally_dir = review / "rally"
        rally_dir.mkdir()
        path = rally_dir / "player_positions_smoothed.csv"

        def write_fixture(current_rows=rows, current_metrics=metrics, current_selection=selection):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(current_rows)
            (rally_dir / "metrics_smoothed.json").write_text(json.dumps(current_metrics), encoding="utf-8")
            (review / "selection.json").write_text(json.dumps(current_selection), encoding="utf-8")

        def load():
            return load_player_positions(path, DEFAULT_VIDEO, metadata, hits, geometry)

        write_fixture()
        tracks, provenance = load()
        assert len(tracks) == 4 and provenance["filled_hitter_positions"] == 1
        assert provenance["smoothing"] == metrics["smoothing"] and len(provenance["csv_sha256"]) == 64
        assert np.array_equal(tracks[1]["A"], [1.4, 2.0]) and np.isnan(tracks[2]["A"]).all()
        players = player_references(hits, geometry, tracks)
        altered_hits = [{**hit, "player_location_x": "broken", "player_location_y": "nan",
                         "opponent_location_x": "inf", "opponent_location_y": "broken"} for hit in hits]
        altered_players = player_references(altered_hits, geometry, tracks)
        for original, altered in zip(players, altered_players):
            for player in ("A", "B"):
                np.testing.assert_equal(original[player], altered[player])
        calls = []

        def spy(**kwargs):
            calls.append(kwargs)
            return None, {"phase": kwargs["phase"], "status": "insufficient_data"}

        with patch("run_projection.reconstruct_flight", side_effect=spy):
            fits = fit_loss_pair(hits, players, [None] * 4, projection, 25, {}, geometry, 10)
        assert len(calls) == 2
        for call in calls:
            assert call["start_frame"] == 1 and call["end_frame"] == 3
            np.testing.assert_equal(call["start_reference_xy"], tracks[1]["A"])
            np.testing.assert_equal(call["receive_reference_xy"], tracks[3]["B"])
        for method in ("mse", "huber"):
            detail = fits[method]["hit_diagnostics"][0]
            assert detail["start_reference_source"] == "current_row_keypoint_smoothed"
            assert detail["receive_reference_source"] == "next_row_keypoint_smoothed"
            assert detail["receive_player_prediction"]["gap_filled"] is True
            assert detail["receive_player_prediction"]["filter_status"] == "long_interpolation"
        altered_rows = [{**row, "reference_x_native": "nan", "reference_y_native": "99999"} for row in rows]
        write_fixture(altered_rows)
        reference_free, _ = load()
        for original, altered in zip(tracks, reference_free):
            for player in ("A", "B"):
                np.testing.assert_equal(original[player], altered[player])

        annotations = review / "hits.csv"
        with annotations.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(hits[0]))
            writer.writeheader()
            writer.writerows(hits)
        assert load_annotations(annotations, 4, 25, require_player_positions=False) == hits
        try:
            load_annotations(annotations, 4, 25)
        except ValueError:
            pass
        else:
            raise AssertionError("Annotation mode accepted missing player coordinates")

        for changes in ({"match": "different_match"}, {"clip_time_seconds": 0.1}, {"clip_frame_num": 10},
                        {"player": "C"}, {"x_court_m": 999}, {"x_court_m": "nan"},
                        {"available": "maybe"}, {"filter_status": "smoothed", "gap_filled": True}):
            invalid = [dict(row) for row in rows]
            invalid[0].update(changes)
            write_fixture(invalid)
            try:
                load()
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid player CSV row: {changes}")
        invalid_variants = [rows + [rows[0]], rows[:-1],
                            [{**row, "court_half": "near"} if i == 2 else row for i, row in enumerate(rows)]]
        for invalid in invalid_variants:
            write_fixture(invalid)
            try:
                load()
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted duplicate/missing player row or unstable identity mapping")
        for changes in ({"source_fps": 30}, {"decoded_frames": 5}, {"source_dimensions": [1280, 720]},
                        {"smoothing": None}):
            write_fixture(current_metrics={**metrics, **changes})
            try:
                load()
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted incompatible player review metadata: {changes}")
        wrong_selection = {**selection, "selected_rallies": [{"key": key, "clip_path": str(review / DEFAULT_VIDEO.name)}]}
        write_fixture(current_selection=wrong_selection)
        try:
            load()
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted a different video with the same rally filename")
        write_fixture()
        (review / "selection.json").unlink()
        try:
            load()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Accepted missing player review metadata")

        invalid = [dict(row) for row in rows]
        invalid[-1].update(available=False, filter_status="unavailable", gap_filled=False)
        write_fixture(invalid)
        args = argparse.Namespace(video=DEFAULT_VIDEO, rally_csv=annotations, homography_csv="unused",
                                  pixel_loss="mse", huber_delta_px=10, player_positions_csv=path)
        for module, command in (("run_projection", run_projection), ("compare_losses", run_comparison)):
            with patch(f"{module}.prepare_inputs", return_value=(DEFAULT_VIDEO, annotations, Path("unused"),
                                                                  metadata, hits, geometry)), \
                    patch(f"{module}.track_shuttle") as tracker, \
                    patch(f"{module}.reserve_output_files") as reserve:
                try:
                    command(args)
                except ValueError as error:
                    assert "frame 3, player B" in str(error)
                else:
                    raise AssertionError("Unavailable required player position did not stop the command")
                tracker.assert_not_called()
                reserve.assert_not_called()

        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        coordinates, phases = np.full((4, 3), np.nan), [""] * 4
        width = round(520 * COURT_WIDTH_M / COURT_LENGTH_M)
        rect = (720 + (720 - width) // 2, 100, width, 520)
        for frame in (1, 3):
            shown = render_frame(blank, frame, [None] * 4, coordinates, phases, [], [], projection, tracks)
            point = panel_point(tracks[frame]["A"], rect)
            assert tuple(shown[point[1], point[0]]) == (30, 80, 255)
        hidden = render_frame(blank, 2, [None] * 4, coordinates, phases, [], [], projection, tracks)
        previous = panel_point(tracks[1]["A"], rect)
        assert tuple(hidden[previous[1], previous[0]]) != (30, 80, 255)
        players[0]["endpoints"] = {"Landing": np.asarray([3.0, 6.0])}
        shown = render_frame(blank, 1, [None] * 4, coordinates, phases, [], players, projection, tracks)
        point = panel_point(players[0]["endpoints"]["Landing"], rect)
        assert tuple(shown[point[1], point[0] + 11]) == (0, 255, 0)
    assert load_player_positions(None, DEFAULT_VIDEO, metadata, hits, geometry) == (None, {"source": "dataset_annotations"})
    for parser, arguments in ((build_projection_parser(), ["run", "--calibration", "calibration.json"]),
                              (build_comparison_parser(), ["--calibration", "calibration.json", "--gt-csv", "gt.csv",
                                                            "--gt-frame-offset", "28"])):
        assert parser.parse_args(arguments).player_positions_csv is None
        assert parser.parse_args(arguments + ["--player-positions-csv", "tracks.csv"]).player_positions_csv == Path("tracks.csv")


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


def test_court_landing_penalty() -> None:
    start, velocity = np.asarray([3.0, 6.0, 1.5]), np.asarray([4.0, 0.5, 4.0])
    start_before, velocity_before = start.copy(), velocity.copy()
    landing_time = (velocity[2] + np.sqrt(velocity[2] ** 2 + 2 * 9.8 * start[2])) / 9.8
    expected_xy = start[:2] + velocity[:2] * landing_time
    flight = simulate_trajectory(start, velocity, 0.0, 75, 25.0, stop_at_ground=True)
    assert flight[-1, 2] == 0.0 and np.all(flight[:-1, 2] > 0.0)
    assert np.allclose(flight[-1, :2], expected_xy, atol=0.01)
    assert np.array_equal(start, start_before) and np.array_equal(velocity, velocity_before)
    expected_distance = expected_xy[0] - COURT_WIDTH_M
    assert abs(extrapolated_landing_distance(start, velocity, 0.0, 25.0) - expected_distance) < 0.01
    assert simulate_trajectory(start, velocity, 0.0, 2, 25.0).shape == (3, 3)
    # The observed interval is inside, but its hypothetical landing is outside.
    observed_path = simulate_trajectory(start, velocity, 0.0, 2, 25.0)
    assert np.all(observed_path[:, 0] < COURT_WIDTH_M) and expected_distance > 0
    for x, y, distance in ((3.0, 6.0, 0.0), (0.0, 0.0, 0.0),
                            (COURT_WIDTH_M, COURT_LENGTH_M, 0.0), (7.1, 6.0, 1.0), (-3.0, -4.0, 5.0)):
        actual = extrapolated_landing_distance(np.asarray([x, y, 1.0]), np.zeros(3), 0.12, 25.0)
        assert np.isclose(actual, distance)
    assert extrapolated_landing_distance(np.asarray([3.0, 6.0, 0.0]), np.zeros(3), 0.0, 25.0) == 0.0
    ascending = simulate_trajectory(np.asarray([3.0, 6.0, 0.0]), velocity, 0.12, 75, 25.0,
                                     stop_at_ground=True)
    assert len(ascending) > 2 and np.max(ascending[:, 2]) > 0 and ascending[-1, 2] == 0
    for point in (np.asarray([3.0, 6.0, -1.0]), np.asarray([np.nan, 6.0, 1.0]),
                  np.asarray([3.0, 6.0, 1e9])):
        assert extrapolated_landing_distance(point, np.zeros(3), 0.0, 25.0) is None

    # Inspect the actual objective for both losses without running an optimizer.
    projection = synthetic_camera()
    observations = [tuple(project_point(projection, point)) for point in observed_path]
    scale = 0.15 * np.linalg.norm(projection[:, :3], ord=2)

    def optimizer(objective, initial, **kwargs):
        return SimpleNamespace(x=initial, success=True, status=0, message="synthetic objective check",
                               fun=objective(initial))

    for loss in ("mse", "huber"):
        inside_cost = None
        for distance in (0.0, 2.0, 5.0, None):
            with patch("reconstruction.minimize", side_effect=optimizer), \
                    patch("reconstruction.extrapolated_landing_distance", return_value=distance):
                path, detail = reconstruct_flight(
                    start_frame=0, end_frame=2, observations=observations, projection=projection,
                    start_reference_xy=start[:2], receive_reference_xy=observed_path[-1, :2],
                    start_fixed=start, fps=25.0, pixel_loss=loss,
                )
            assert detail["landing_outside_distance_m"] == distance
            if distance is None:
                assert path is None and detail["status"] == "failed" and detail["landing_penalty"] is None
                assert detail["reason"] == "hypothetical_ground_contact_unavailable_within_60s"
            else:
                assert path.shape == (3, 3) and detail["status"] == "available"
                assert np.isclose(detail["landing_penalty"], (scale * distance) ** 2)
                if distance == 0:
                    inside_cost = detail["objective"]
                assert np.isclose(detail["objective"] - inside_cost, (scale * distance) ** 2)


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
    assert rows[3]["eligibility_reason"] == "manual_floor_contact_evaluation_boundary" and not rows[3]["comparison_eligible"]
    assert rows[4]["eligibility_reason"] == "post_contact" and not rows[4]["comparison_eligible"]
    assert coverage["eligible_visible_gt_frames"] == 2
    assert coverage["mse_available_frames"] == 1 and coverage["huber_available_frames"] == 2
    assert coverage["paired_available_frames"] == 1
    assert metrics["overall"]["mse"]["paired_metrics"]["rmse_px"] == 5.0
    assert metrics["overall"]["huber"]["paired_metrics"]["rmse_px"] == 0.0
    assert metrics["per_flight"]["synthetic_flight"]["huber"]["individual_metrics"]["count"] == 2
    assert metrics["hit_to_hit_baseline"]["coverage"]["paired_available_frames"] == 1
    assert metrics["hit_to_hit_baseline"]["coverage"]["eligible_visible_gt_frames"] == 2
    assert metrics["hit_to_hit_baseline"]["huber"]["individual_metrics"]["count"] == 2
    assert metrics["final_flight_extension"]["coverage"]["eligible_visible_gt_frames"] == 0
    assert metrics["final_flight_extension"]["huber"]["individual_metrics"]["rmse_px"] is None
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
    test_saved_player_tracks(geometry, fitted)
    test_pixel_losses()
    test_court_landing_penalty()
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

    terminal = {"terminal_events": {
        "net_contact": {"status": "manual", "frame": 110, "pixel_native": [600.0, 400.0]},
        "floor_contact": {"status": "manual", "frame": 125, "pixel_native": [500.0, 500.0]},
    }}
    with patch("run_projection.reconstruct_flight") as fitter:
        direct_paths, direct_detail, floor_world = fit_terminal_tail(
            terminal, geometry, [{"clip_frame_num": 100}], terminal_observations, fitted, 25.0,
            truth, "huber", 8.0,
        )
        fitter.assert_not_called()
    assert not direct_paths and floor_world is None and not direct_detail["phases"]
    assert direct_detail["status"] == "disabled" and direct_detail["reason"] == "manual_contacts_evaluation_only"
    assert direct_detail["pixel_loss"] == "huber" and direct_detail["huber_delta_px"] == 8.0

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

    print("Self-check passed: dynamic Fuzhou metadata and hit timing, scaling and homography, camera calibration, player reference selection/validation, saved smoothed-track validation and preflight rejection, moving/unavailable player markers, MSE/Huber behavior and propagation, extrapolated landing/court penalty, shuttle-label/terminal-contact separation from fitting, final-hit inclusion, independent-GT evaluation masks/metrics, disabled terminal reconstruction, continuity, rendering, and overwrite refusal. No real-rally fitting or video rendering.")
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
