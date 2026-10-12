"""Small assert-based checks for court-plane projection and rally indexing."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

from court_projection import (
    ANNOTATION_TO_VIDEO,
    DEFAULT_HOMOGRAPHY,
    DEFAULT_RALLY_CSV,
    DEFAULT_VIDEO,
    DEFAULT_TRACKED_CSV,
    COURT_WIDTH_M,
    COURT_LENGTH_M,
    PREVIEW_SIZE,
    calibration_document,
    court_panel_point,
    draw_calibration_preview,
    load_calibration,
    load_tracked_detections,
    line_through_points,
    normalize_vanishing_point,
    prepare_strokes,
    project_observations,
    vertical_vanishing_point,
    intersect_floor_pixel,
    render_frame,
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
    overflowing = [(523, 0, np.asarray([3.0, -100.0]), np.asarray([3.0, 100.0]))]
    clipped = render_frame(blank, plot_row, overflowing, court_world_lines())
    changed = np.any(clipped != baseline, axis=2)
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
        existing.write_text("keep", encoding="utf-8")
        try:
            reserve_output_files([new_file, existing])
        except FileExistsError:
            pass
        else:
            raise AssertionError("Output reservation did not refuse overwrite")
        assert existing.read_text(encoding="utf-8") == "keep" and not new_file.exists()
    assert COURT_WIDTH_M == 6.1 and COURT_LENGTH_M == 13.4
    print("Self-check passed: finite/infinite vanishing points, four shuttle heights, 1.5 scaling, hit/landing endpoints and serve fallback, invalid endpoints, horizon/sensitivity/plausibility rejection, legitimate out shots, clipped trails/markers, A/B player labels, missing/degenerate frames, and overwrite refusal.")


if __name__ == "__main__":
    run_checks()
