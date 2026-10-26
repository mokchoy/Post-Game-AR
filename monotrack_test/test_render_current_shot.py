"""Focused synthetic checks; no model inference or video encoding."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import render_current_shot as renderer


class CurrentShotTests(unittest.TestCase):
    def test_trail_resets_on_hit_and_does_not_cross_missing_samples(self):
        coordinates = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4], [np.nan, np.nan], [5, 5]])
        players = {"A": np.full(2, np.nan), "B": np.full(2, np.nan)}
        for frame, expected in ((0, 0), (2, 1), (3, 0), (5, 0)):
            with self.subTest(frame=frame), patch.object(renderer.cv2, "line") as line:
                renderer.render_frame(np.zeros((720, 1440, 3), dtype=np.uint8),
                                      frame, 25, [1, 3], coordinates, players)
                trails = [call for call in line.call_args_list if call.args[3] == renderer.TRAIL_COLOR]
                self.assertEqual(len(trails), expected)
                if trails:
                    self.assertEqual(trails[0].args[1], renderer.panel_point(coordinates[1], renderer.COURT_RECT))
                    self.assertEqual(trails[0].args[2], renderer.panel_point(coordinates[2], renderer.COURT_RECT))

    def test_players_move_and_unavailable_players_disappear(self):
        coordinates = np.full((2, 2), np.nan)
        for x in (1, 2):
            with patch.object(renderer.cv2, "circle") as circle:
                renderer.render_frame(np.zeros((720, 1440, 3), dtype=np.uint8), 0, 25, [1], coordinates,
                                      {"A": np.asarray([x, 3]), "B": np.full(2, np.nan)})
                self.assertEqual(circle.call_count, 1)
                self.assertEqual(circle.call_args.args[1], renderer.panel_point([x, 3], renderer.COURT_RECT))

    def test_original_rally_panel_is_preserved(self):
        source = np.full((720, 1440, 3), 73, dtype=np.uint8)
        players = {"A": np.full(2, np.nan), "B": np.full(2, np.nan)}
        shown = renderer.render_frame(source, 0, 25, [1], np.full((2, 2), np.nan), players)
        self.assertEqual(shown.shape, source.shape)
        np.testing.assert_array_equal(shown[:, :720], source[:, :720])
        self.assertFalse(np.array_equal(shown[:, 720:], source[:, 720:]))
        self.assertTrue((source == 73).all())

    def test_batch_outputs_stay_beside_original_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            clip = batch / "clip-key"
            clip.mkdir()
            (clip / "rally_002_diagnostics.json").write_text("{}", encoding="utf-8")
            with patch.object(renderer, "BATCH", batch), \
                    patch.object(renderer, "shutil_which", return_value="ffmpeg"), \
                    patch.object(renderer, "load_saved_projection", return_value=()), \
                    patch.object(renderer, "write_video") as write:
                self.assertEqual(renderer.main(), 0)
            self.assertEqual(write.call_args.args[0], clip / "rally_002_projection_current_shot.mp4")

    def test_saved_coordinates_require_complete_ordered_frames_and_finite_points(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.csv"

            def write(rows):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("frame", "time_seconds", "x_m", "y_m"))
                    writer.writerows(rows)

            write([(0, 0, "", ""), (1, 0.04, 1, 2)])
            points = renderer.load_coordinates(path, 2, 25)
            self.assertTrue(np.isnan(points[0]).all())
            np.testing.assert_array_equal(points[1], [1, 2])
            for rows in (
                [(0, 0, 1, 2)],
                [(1, 0, 1, 2), (0, 0.04, 1, 2)],
                [(0, 0, 1, 2), (1, 0.08, 1, 2)],
                [(0, 0, 1, 2), (1, 0.04, "nan", 2)],
                [(0, 0, 1, 2), (1, 0.04, "", 2)],
            ):
                write(rows)
                with self.assertRaises(ValueError):
                    renderer.load_coordinates(path, 2, 25)

    def test_existing_video_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.mp4"
            path.write_bytes(b"keep")
            with self.assertRaises(FileExistsError), patch.object(renderer.subprocess, "Popen") as process:
                renderer.write_video(path, Path("source.mp4"), {}, [], np.empty((0, 2)), [])
            process.assert_not_called()
            self.assertEqual(path.read_bytes(), b"keep")

    def test_failed_render_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "partial.mp4"
            with patch.object(renderer.cv2, "VideoCapture", side_effect=OSError("decoder unavailable")):
                with self.assertRaises(OSError):
                    renderer.write_video(path, Path("source.mp4"), {"fps": 25}, [], np.empty((0, 2)), [])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
