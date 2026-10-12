"""Small assert-based checks for court-plane projection and rally indexing."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True

from court_projection import (
    ANNOTATION_TO_VIDEO,
    DEFAULT_HOMOGRAPHY,
    DEFAULT_RALLY_CSV,
    DEFAULT_VIDEO,
    DEFAULT_TRACKED_CSV,
    DEFAULT_CALIBRATION,
    INTERPOLATED_COLOR,
    MAX_SHORT_GAP_FRAMES,
    COURT_WIDTH_M,
    COURT_LENGTH_M,
    PREVIEW_SIZE,
    calibration_document,
    court_panel_point,
    draw_calibration_preview,
    fill_short_gaps,
    load_calibration,
    load_tracked_detections,
    line_through_points,
    normalize_vanishing_point,
    prepare_strokes,
    project_observations,
    validate_short_gaps,
    print_gap_validation,
    build_parser,
    vertical_vanishing_point,
    intersect_floor_pixel,
    render_frame,
    run,
    trail_edges,
    write_ground_csv,
)
from reconstruction import load_court_geometry, project_point
from run_projection import court_world_lines, player_references, prepare_inputs, reserve_output_files


def synthetic_projection(infinite_vertical: bool) -> np.ndarray:
    return np.asarray([
        [80.0, 10.0, 30.0, 640.0],
        [5.0, 50.0, -40.0, 360.0],
        [0.01, 0.005, 0.0 if infinite_vertical else 0.01, 1.0],
    ])


def test_vanishing_point_and_floor_intersection(infinite_vertical: bool) -> None:
    camera = synthetic_projection(infinite_vertical)
    post_world = {
        "left_post_bottom": [0.0, 6.7, 0.0],
        "left_post_top": [0.0, 6.7, 1.55],
        "right_post_bottom": [6.1, 6.7, 0.0],
        "right_post_top": [6.1, 6.7, 1.55],
    }
    points = {
        name: {"pixel_native_px": project_point(camera, world).tolist(), "frame_index": 0}
        for name, world in post_world.items()
    }
    vp = vertical_vanishing_point(points)
    expected = normalize_vanishing_point(camera[:, 2])
    assert np.allclose(vp, expected, atol=1e-7)
    assert bool(vp[2] == 0.0) == infinite_vertical

    a_world, b_world = np.asarray([0.8, 4.7, 0.0]), np.asarray([5.2, 8.5, 0.0])
    a_pixel, b_pixel = project_point(camera, a_world), project_point(camera, b_world)
    ab_line, status = line_through_points(a_pixel, b_pixel)
    assert status == "available"
    for height in (0.25, 0.8, 1.7, 3.0):
        shuttle = np.asarray([3.0, 6.6, height])
        observed = project_point(camera, shuttle)
        floor_pixel, angle, status = intersect_floor_pixel(observed, ab_line, vp)
        assert status == "available" and angle is not None and angle >= 1.0
        assert np.allclose(floor_pixel, project_point(camera, [3.0, 6.6, 0.0]), atol=1e-6)


def test_projection_validity(geometry, stroke, players) -> None:
    flat_geometry = replace(geometry, homography_annotation_to_dataset=np.diag([0.1, 0.1, 1.0]),
                            dataset_to_metric=np.eye(3))
    floor = np.asarray([45.0, 100.5])
    ab, status = line_through_points(floor - [10, 0], floor + [10, 0])
    one_frame = replace(stroke, start_frame=0, end_frame_exclusive=1, ab_line=ab, line_status=status)
    vp = np.asarray([0.0, 1.0, 0.0])
    for transform in (flat_geometry, replace(flat_geometry,
                                            homography_annotation_to_dataset=-flat_geometry.homography_annotation_to_dataset)):
        row = project_observations([floor - [0, 10]], [one_frame], vp, transform, 25, players)[0]
        assert row["status"] == "available" and np.allclose(row["court_xy"], [3.0, 6.7])
    outside = project_observations([[-7.5, 90.5]], [one_frame], vp, flat_geometry, 25, players)[0]
    assert outside["status"] == "available" and np.allclose(outside["court_xy"], [-0.5, 6.7])
    distant = project_observations([[-45.0, 90.5]], [one_frame], vp, flat_geometry, 25, players)[0]
    assert distant["status"] == "implausible_court_position" and distant["court_xy"] is None

    ab, status = line_through_points(floor - [1, 20], floor + [1, 20])
    sensitive = replace(one_frame, ab_line=ab, line_status=status)
    row = project_observations([floor - [0, 10]], [sensitive], vp, flat_geometry, 25, players)[0]
    assert row["angle_deg"] > 1 and row["status"] == "unstable_floor_projection"
    assert row["court_xy"] is None and row["floor_pixel"] is None

    # Frame 375's old player-based plane produced a finite position about 8 km away.
    ab, status = line_through_points([840.0, 499.5], [900.0, 895.5])
    bad_plane = replace(one_frame, ab_line=ab, line_status=status)
    real_vp = np.asarray([1159.5260115606936, 26609.618497109826, 1.0])
    row = project_observations([(675.0, 581.0)], [bad_plane], real_vp, geometry, 25, players)[0]
    assert row["status"] == "floor_point_at_or_behind_horizon"
    assert row["court_xy"] is None and row["floor_pixel"] is None


def test_short_gaps(geometry, stroke, players) -> None:
    flat = replace(geometry, homography_annotation_to_dataset=np.diag([0.1, 0.1, 1.0]),
                   dataset_to_metric=np.eye(3))
    a, b = np.asarray([30.0, 100.5]), np.asarray([78.0, 100.5])
    ab, status = line_through_points(a, b)
    stroke = replace(stroke, start_frame=0, end_frame_exclusive=24,
                     a_native=a, b_native=b, ab_line=ab, line_status=status)
    vp = np.asarray([0.0, 1.0, 0.0])
    observations = [(30.0 + 2 * frame, 90.5) for frame in range(24)]

    def reconstruct(masked, intervals=None):
        intervals = intervals or [stroke]
        rows = project_observations(masked, intervals, vp, flat, 25, players)
        filled, counts = fill_short_gaps(rows, intervals, vp, flat, 25, players)
        for original, result in zip(rows, filled):
            assert original["observed"] == result["observed"]
            if original["observed"] is not None or original["status"] != "missing_detection":
                assert original["status"] == result["status"] and result["estimated"] is None
        return rows, filled, counts

    masked = list(observations)
    masked[2:5] = [None] * 3
    rows, filled, counts = reconstruct(masked)
    assert counts["accepted_gaps"] == 1 and counts["rejected_gaps"] == 0
    assert all(row["status"] == "missing_detection" and row["estimated"] is None for row in rows[2:5])
    for frame in range(2, 5):
        assert filled[frame]["observed"] is None and filled[frame]["status"] == "available"
        assert filled[frame]["position_source"] == "interpolated"
        assert np.allclose(filled[frame]["estimated"], observations[frame])
    assert observations[2] is not None and masked[2] is None
    edges = trail_edges(filled)
    assert [edge[0] for edge in edges if edge[4]] == [2, 3, 4, 5]

    # A single candidate failing the existing projection gate must reject the whole gap.
    def reject_middle(pixel, *args, **kwargs):
        if np.allclose(pixel, observations[3]):
            return None, 0.5, "intersection_angle_below_1_degree"
        return intersect_floor_pixel(pixel, *args, **kwargs)

    with patch("court_projection.intersect_floor_pixel", side_effect=reject_middle):
        _, rejected, counts = reconstruct(masked)
    assert counts["eligible_gaps"] == counts["rejected_gaps"] == 1 and counts["accepted_gaps"] == 0
    assert all(row["status"] == "missing_detection" and row["position_source"] == "unavailable"
               and row["estimated"] is None and row["court_xy"] is None for row in rejected[2:5])

    for start, end in ((0, 2), (22, 24), (3, 9)):
        masked = list(observations)
        masked[start:end] = [None] * (end - start)
        _, preserved, counts = reconstruct(masked)
        assert counts["eligible_gaps"] == counts["accepted_gaps"] == 0
        assert all(row["position_source"] == "unavailable" for row in preserved[start:end])
    split_strokes = [replace(stroke, end_frame_exclusive=12),
                     replace(stroke, index=1, identifier="stroke_02", start_frame=12)]
    for start, end in ((11, 13), (10, 12), (12, 14)):
        masked = list(observations)
        masked[start:end] = [None] * (end - start)
        _, preserved, counts = reconstruct(masked, split_strokes)
        assert counts["eligible_gaps"] == 0
        assert all(row["position_source"] == "unavailable" for row in preserved[start:end])
    assert not any(edge[0] == 12 for edge in trail_edges(reconstruct(observations, split_strokes)[1]))

    masked = list(observations)
    masked[2:4] = [None] * 2
    masked[4] = (1000.0, 90.5)
    _, preserved, counts = reconstruct(masked)
    assert counts["eligible_gaps"] == 0 and preserved[4]["status"] == "implausible_court_position"
    assert preserved[4]["observed"] == masked[4] and preserved[4]["position_source"] == "unavailable"
    assert all(row["estimated"] is None for row in preserved[2:5])

    # An earlier estimate cannot anchor a later call either.
    repeat = [dict(row) for row in filled]
    repeat[5] = {**repeat[5], "observed": None, "status": "missing_detection",
                 "position_source": "unavailable", "court_xy": None, "floor_pixel": None}
    preserved, counts = fill_short_gaps(repeat, [stroke], vp, flat, 25, players)
    assert counts["eligible_gaps"] == 0 and preserved[5]["position_source"] == "unavailable"

    # Nonuniform timestamps must weight interpolation by time, not frame index.
    timed = project_observations([observations[0], None, observations[4]],
                                [replace(stroke, end_frame_exclusive=3)], vp, flat, 25, players)
    for row, seconds in zip(timed, (0.0, 0.01, 0.04)):
        row["time_seconds"] = seconds
    weighted, _ = fill_short_gaps(timed, [replace(stroke, end_frame_exclusive=3)], vp, flat, 25, players)
    assert np.allclose(weighted[1]["estimated"], observations[1])

    seen = []
    actual_fill = fill_short_gaps

    def inspect_hidden(masked_rows, *args):
        hidden = [row for row in masked_rows if row["observed"] is None]
        assert len(hidden) in (1, 3, 5) and all(row["estimated"] is None for row in hidden)
        seen.append(len(hidden))
        return actual_fill(masked_rows, *args)

    validation_strokes = [replace(stroke, end_frame_exclusive=12), split_strokes[1]]
    with patch("court_projection.fill_short_gaps", side_effect=inspect_hidden):
        validation = validate_short_gaps(observations, validation_strokes, vp, flat, 25, players)
    assert seen == [1, 3, 5] and validation["excluded_stroke_id"] == "stroke_02"
    assert all(trial["start_frame"] == 1 for trial in validation["trials"])
    for length, summary in validation["by_window_frames"].items():
        assert summary["attempted_windows"] == summary["accepted_windows"] == 1
        assert summary["accepted_frames"] == int(length)
        assert summary["native_pixels"]["rmse"] < 1e-10
        assert summary["interpolated_court_m"]["rmse"] < 1e-10
        # Half-stroke timing deliberately makes AB different from the reference motion.
        expected = np.asarray([2 * frame / 15 for frame in range(1, int(length) + 1)])
        assert np.isclose(summary["endpoint_ab_court_m"]["mean_euclidean_error"], expected.mean())
        assert np.isclose(summary["endpoint_ab_court_m"]["rmse"], np.sqrt(np.mean(expected ** 2)))
    def reject_validation(masked_rows, *args):
        result, counts = actual_fill(masked_rows, *args)
        for original, reconstructed in zip(masked_rows, result):
            if original["observed"] is None:
                reconstructed["position_source"] = "unavailable"
        return result, counts

    with patch("court_projection.fill_short_gaps", side_effect=reject_validation):
        rejected_validation = validate_short_gaps(observations, validation_strokes, vp, flat, 25, players)
    for summary in rejected_validation["by_window_frames"].values():
        assert summary["attempted_windows"] == 1 and summary["accepted_windows"] == summary["accepted_frames"] == 0
        assert summary["native_pixels"]["rmse"] is None and summary["endpoint_ab_court_m"]["rmse"] is None
    no_valid = validate_short_gaps([None] * 24, validation_strokes, vp, flat, 25, players)
    assert all(summary["attempted_windows"] == 0 and summary["native_pixels"]["rmse"] is None
               for summary in no_valid["by_window_frames"].values())


def run_checks() -> None:
    test_vanishing_point_and_floor_intersection(False)
    test_vanishing_point_and_floor_intersection(True)
    degenerate, status = line_through_points([4, 8], [4, 8])
    assert degenerate is None and status == "degenerate_ab_line"
    parallel_ab, _ = line_through_points([0, 0], [10, 0])
    floor, angle, status = intersect_floor_pixel([5, 2], parallel_ab, [1, 0, 0])
    assert floor is None and angle is not None and angle < 1.0 and status == "intersection_angle_below_1_degree"
    missing, _, status = intersect_floor_pixel([1, 2], None, [0, 1, 0], "degenerate_ab_line")
    assert missing is None and status == "degenerate_ab_line"

    video, rally_csv, homography_csv, metadata, hits, geometry = prepare_inputs(
        DEFAULT_VIDEO, DEFAULT_RALLY_CSV, DEFAULT_HOMOGRAPHY
    )
    assert metadata == {"width": 1920, "height": 1080, "fps": 25.0, "frame_count": 524}
    assert len(hits) == 18
    original_tracker_bytes = DEFAULT_TRACKED_CSV.read_bytes()
    detections = load_tracked_detections(DEFAULT_TRACKED_CSV, metadata)
    assert len(detections) == 524 and sum(point is not None for point in detections) == 328
    preview_point_native = np.asarray([1800.0, 800.0])
    preview_point = draw_calibration_preview(
        np.zeros((metadata["height"], metadata["width"], 3), dtype=np.uint8),
        geometry, "right_post_top",
        {"right_post_top": {"pixel_native_px": preview_point_native.tolist()}},
        0, "",
    )
    expected_x = round(preview_point_native[0] * PREVIEW_SIZE[0] / metadata["width"])
    expected_y = round(preview_point_native[1] * PREVIEW_SIZE[1] / metadata["height"])
    expected_region = preview_point[max(0, expected_y - 4):expected_y + 5,
                                    max(0, expected_x - 4):expected_x + 5]
    old_x, old_y = round(expected_x * 2 / 3), round(expected_y * 2 / 3)
    old_region = preview_point[max(0, old_y - 4):old_y + 5, max(0, old_x - 4):old_x + 5]
    is_green_marker = lambda region: np.any((region[:, :, 1] > 80) & (region[:, :, 0] < 50) & (region[:, :, 2] < 50))
    assert is_green_marker(expected_region) and not is_green_marker(old_region)
    strokes = prepare_strokes(hits, metadata["frame_count"])
    assert len(strokes) == 18
    assert strokes[0].start_frame == 25 and strokes[0].end_frame_exclusive == 37
    assert strokes[0].a_source == "first_row_player_location"
    assert strokes[0].b_source == "current_row_landing_x_y"
    assert np.allclose(strokes[0].a_native, np.asarray([655.0, 544.0]) * ANNOTATION_TO_VIDEO)
    assert np.allclose(strokes[0].b_annotation, [590.0, 382.0])
    for stroke, hit in zip(strokes[1:], hits[1:]):
        assert stroke.a_source == "current_row_hit_x_y"
        assert np.allclose(stroke.a_annotation, [float(hit["hit_x"]), float(hit["hit_y"])])
    for stroke, hit in zip(strokes, hits):
        assert np.allclose(stroke.b_annotation, [float(hit["landing_x"]), float(hit["landing_y"])])
    for previous, current in zip(strokes, strokes[1:]):
        assert np.allclose(previous.b_annotation, current.a_annotation)
    assert strokes[-1].start_frame == 485 and strokes[-1].end_frame_exclusive == 524
    assert strokes[-1].b_source == "final_row_landing_x_y"
    assert np.allclose(strokes[-1].b_annotation, [680.0, 449.0])
    assert np.allclose(geometry.native_floor_to_world(strokes[0].a_native),
                       geometry.annotation_floor_to_world(strokes[0].a_annotation))
    for index, key, value in ((1, "hit_x", ""), (0, "landing_y", "nan")):
        invalid_hits = [dict(hit) for hit in hits]
        invalid_hits[index][key] = value
        try:
            prepare_strokes(invalid_hits, metadata["frame_count"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid endpoint {key}={value!r}")

    calibration_points = {
        "left_post_bottom": {"pixel_native_px": [400.0, 300.0], "frame_index": 0},
        "left_post_top": {"pixel_native_px": [405.0, 120.0], "frame_index": 0},
        "right_post_bottom": {"pixel_native_px": [1200.0, 310.0], "frame_index": 0},
        "right_post_top": {"pixel_native_px": [1205.0, 130.0], "frame_index": 0},
    }
    vp = vertical_vanishing_point(calibration_points)
    players = player_references(hits, geometry)
    test_projection_validity(geometry, strokes[0], players)
    test_short_gaps(geometry, strokes[0], players)
    assert MAX_SHORT_GAP_FRAMES == 5
    assert not build_parser().parse_args(["run"]).fill_short_gaps
    assert build_parser().parse_args(["run", "--fill-short-gaps"]).fill_short_gaps
    rows = project_observations(detections, strokes, vp, geometry, metadata["fps"], players)
    assert len(rows) == 524 and rows[24]["stroke"] is None
    assert rows[24]["player_locations"] is None
    assert np.allclose(rows[25]["player_locations"]["A"], players[0]["A"])
    assert np.allclose(rows[25]["player_locations"]["B"], players[0]["B"])
    assert rows[36]["stroke"].identifier == "stroke_01"
    assert rows[37]["stroke"].identifier == "stroke_02"
    assert np.allclose(rows[37]["player_locations"]["B"], players[1]["B"])
    assert rows[485]["stroke"].identifier == "stroke_18"
    assert rows[523]["stroke"].identifier == "stroke_18"
    assert np.allclose(rows[523]["player_locations"]["A"], players[-1]["A"])
    assert rows[0]["time_seconds"] == 0.0 and rows[25]["time_seconds"] == 1.0
    missing_frame = next(index for index in range(25, 524) if detections[index] is None)
    assert rows[missing_frame]["status"] == "missing_detection"
    real_calibration = load_calibration(DEFAULT_CALIBRATION, video, metadata, geometry)
    real_vp = np.asarray(real_calibration["homogeneous_vertical_vanishing_point_native"])
    original_rows = project_observations(detections, strokes, real_vp, geometry, metadata["fps"], players)
    filled_rows, gap_counts = fill_short_gaps(original_rows, strokes, real_vp, geometry, metadata["fps"], players)
    assert gap_counts == {"enabled": True, "max_gap_frames": 5, "eligible_gaps": 10,
                          "accepted_gaps": 10, "rejected_gaps": 0, "skipped_gaps": 18}
    assert sum(row["position_source"] == "interpolated" for row in filled_rows) == 21
    assert sum(row["position_source"] == "tracked" for row in filled_rows) == 269
    for original, filled in zip(original_rows, filled_rows):
        assert original["observed"] == filled["observed"]
        if original["status"] != "missing_detection":
            assert original["status"] == filled["status"] and filled["estimated"] is None
            if original["court_xy"] is not None:
                assert np.array_equal(original["court_xy"], filled["court_xy"])
    long_runs = [start for start in range(1, len(detections) - 46)
                 if detections[start - 1] is not None and detections[start + 46] is not None
                 and all(point is None for point in detections[start:start + 46])]
    assert len(long_runs) == 1
    assert all(row["status"] == "missing_detection" and row["estimated"] is None
               and row["position_source"] == "unavailable" for row in filled_rows[long_runs[0]:long_runs[0] + 46])
    validation = validate_short_gaps(detections, strokes, real_vp, geometry, metadata["fps"], players)
    assert validation["excluded_stroke_id"] == strokes[-1].identifier
    for length, attempted in (("1", 16), ("3", 15), ("5", 13)):
        summary = validation["by_window_frames"][length]
        assert summary["attempted_windows"] == summary["accepted_windows"] == attempted
        assert summary["accepted_frames"] == int(length) * attempted
        for key in ("native_pixels", "interpolated_court_m", "endpoint_ab_court_m"):
            assert np.isfinite(list(summary[key].values())).all()
    print_gap_validation(validation)
    print(f"Current rally: {gap_counts['accepted_gaps']} accepted gaps, 21 recovered frames; "
          f"46-frame gap [{long_runs[0]}, {long_runs[0] + 46}) remains unavailable.")
    preview = render_frame(np.zeros((1080, 1920, 3), dtype=np.uint8), rows[25], trail_edges(rows), court_world_lines())
    assert preview.shape == (720, 1440, 3)
    player_row = {**rows[25], "court_xy": None, "player_locations": {
        "A": np.asarray([1.0, 2.0]), "B": np.asarray([5.0, 11.0]),
    }}
    player_preview = render_frame(np.zeros((1080, 1920, 3), dtype=np.uint8), player_row, [], court_world_lines())
    panel_width = round(570 * COURT_WIDTH_M / COURT_LENGTH_M)
    panel_rect = (960 + (480 - panel_width) // 2, 70, panel_width, 570)
    for identity, color in (("A", (0, 70, 255)), ("B", (255, 120, 0))):
        point = court_panel_point(player_row["player_locations"][identity], panel_rect)
        assert point is not None and tuple(player_preview[point[1], point[0]]) == color

    plot_row = {**rows[523], "court_xy": None, "player_locations": None}
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    baseline = render_frame(blank, plot_row, [], court_world_lines())
    overflowing = [(523, 0, np.asarray([3.0, -100.0]), np.asarray([3.0, 100.0]), False)]
    clipped = render_frame(blank, plot_row, overflowing, court_world_lines())
    changed = np.any(clipped != baseline, axis=2)
    assert np.any(changed[52:641, 960:])
    assert not np.any(changed[:52]) and not np.any(changed[641:]) and not np.any(changed[:, :960])
    orange_edges = [(523, 0, np.asarray([3.0, -100.0]), np.asarray([3.0, 100.0]), True)]
    orange_clipped = render_frame(blank, plot_row, orange_edges, court_world_lines())
    changed = np.any(orange_clipped != baseline, axis=2)
    assert np.any(np.all(orange_clipped[52:641, 960:] == INTERPOLATED_COLOR, axis=2))
    assert not np.any(changed[:52]) and not np.any(changed[641:]) and not np.any(changed[:, :960])
    marker_row = {**plot_row, "court_xy": np.asarray([3.0, 6.7]), "floor_pixel": np.asarray([300.0, 300.0]),
                  "observed": None, "estimated": (300.0, 150.0), "position_source": "interpolated"}
    markers = render_frame(blank, marker_row, [], court_world_lines())
    assert tuple(markers[52 + 75, 150]) == INTERPOLATED_COLOR
    assert tuple(markers[52 + 150, 150]) == INTERPOLATED_COLOR
    point = court_panel_point(marker_row["court_xy"], panel_rect)
    assert tuple(markers[point[1], point[0]]) == INTERPOLATED_COLOR
    outside_marker = {**plot_row, "court_xy": np.asarray([3.0, -0.5]), "position_source": "interpolated"}
    outside_preview = render_frame(blank, outside_marker, [], court_world_lines())
    outside_baseline = render_frame(blank, {**outside_marker, "position_source": "tracked"}, [], court_world_lines())
    changed = np.any(outside_preview != outside_baseline, axis=2)
    assert np.any(changed[52:641, 960:])
    assert not np.any(changed[:52]) and not np.any(changed[641:]) and not np.any(changed[:, :960])
    outside_players = {**plot_row, "player_locations": {"A": np.asarray([3.0, -1.0]),
                                                       "B": np.asarray([3.0, 14.0])}}
    assert np.array_equal(render_frame(blank, outside_players, [], court_world_lines()), baseline)

    with tempfile.TemporaryDirectory(prefix="court-projection-check-", dir=Path(__file__).resolve().parent) as temp_name:
        folder = Path(temp_name)
        existing, new_file = folder / "existing.txt", folder / "new.txt"
        trajectory = folder / "trajectory.csv"
        calibration_path = folder / "calibration.json"
        calibration = calibration_document(video, rally_csv, homography_csv, metadata, geometry, calibration_points)
        calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
        loaded = load_calibration(calibration_path, video, metadata, geometry)
        assert loaded["vertical_vanishing_point_kind"] == "at_infinity"
        write_ground_csv(trajectory, rows)
        with trajectory.open("r", newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == 524
        write_ground_csv(trajectory, filled_rows)
        with trajectory.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            exported = list(reader)
            assert reader.fieldnames[-3:] == ["estimated_shuttle_x_native_px", "estimated_shuttle_y_native_px", "position_source"]
        for row, record in zip(filled_rows, exported):
            assert record["position_source"] == row["position_source"] and record["status"] == row["status"]
            if row["observed"] is None:
                assert record["observed_shuttle_x_native_px"] == record["observed_shuttle_y_native_px"] == ""
            else:
                assert record["observed_shuttle_x_native_px"] == f"{row['observed'][0]:.3f}"
                assert record["observed_shuttle_y_native_px"] == f"{row['observed'][1]:.3f}"
            if row["position_source"] == "interpolated":
                assert record["estimated_shuttle_x_native_px"] == f"{row['estimated'][0]:.3f}"
                assert record["estimated_shuttle_y_native_px"] == f"{row['estimated'][1]:.3f}"
                assert record["court_x_m"] and record["court_y_m"] and record["status"] == "available"
            else:
                assert record["estimated_shuttle_x_native_px"] == record["estimated_shuttle_y_native_px"] == ""
        existing.write_text("keep", encoding="utf-8")
        try:
            reserve_output_files([new_file, existing])
        except FileExistsError:
            pass
        else:
            raise AssertionError("Output reservation did not refuse overwrite")
        assert existing.read_text(encoding="utf-8") == "keep" and not new_file.exists()
        # Exercise run's CSV/diagnostic wiring without invoking the video renderer.
        for enabled in (False, True):
            with patch("court_projection.render_h264_with_audio", return_value={"rendered": False}) as renderer, \
                    patch("court_projection.validate_short_gaps", return_value=validation) as validator, \
                    redirect_stdout(io.StringIO()):
                outputs = run(video, rally_csv, homography_csv, DEFAULT_TRACKED_CSV, DEFAULT_CALIBRATION,
                              folder / str(enabled), enabled)
            document = json.loads(outputs[1].read_text(encoding="utf-8"))
            assert document["short_gap_filling"]["enabled"] is enabled
            assert document["short_gap_filling"]["max_gap_frames"] == 5
            assert document["short_gap_filling"]["accepted_gaps"] == (10 if enabled else 0)
            assert document["short_gap_filling"]["rejected_gaps"] == 0
            assert document["availability"]["tracked_projections"] == 269
            assert document["availability"]["interpolated_projections"] == (21 if enabled else 0)
            assert document["availability"]["available"] == (290 if enabled else 269)
            assert sum(item["interpolated_projections"] for item in document["availability"]["strokes"].values()) == (21 if enabled else 0)
            assert document["held_out_validation"] == (validation if enabled else None)
            assert validator.call_count == int(enabled) and renderer.call_count == 1
            assert outputs[2].stat().st_size == 0  # Reserved placeholder only; no video regenerated.
    assert COURT_WIDTH_M == 6.1 and COURT_LENGTH_M == 13.4
    assert DEFAULT_TRACKED_CSV.read_bytes() == original_tracker_bytes
    print("Self-check passed: projection geometry/quality, scaling, endpoints, short-gap filling and atomic rejection, unchanged observations, long/edge/stroke gaps, tracked-only anchors, time weighting, held-out validation and AB scoring, CSV provenance, orange clipped trails/markers, A/B players, and overwrite refusal. No video rendered or inference run.")


if __name__ == "__main__":
    run_checks()
