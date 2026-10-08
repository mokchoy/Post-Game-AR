import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import track_video


IMAGE_SIZE = (288, 512)
SCRIPT_DIR = Path(track_video.__file__).resolve().parent


class StubModel(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().cpu().clone())
        return torch.as_tensor(self.predictions[len(self.inputs) - 1], device=inputs.device).unsqueeze(0)


def empty_group():
    return np.zeros((3, *IMAGE_SIZE), dtype=np.float32)


def prediction_group(points):
    prediction = empty_group()
    for index, point in enumerate(points):
        if point is not None:
            x, y, value = point
            prediction[index, y, x] = value
    return prediction


def write_clip(path, frame_count, fps=12.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (80, 40))
    if not writer.isOpened():
        raise RuntimeError(f"Could not initialize synthetic video writer: {path}")
    for index in range(frame_count):
        frame = np.full((40, 80, 3), (20 + 30 * index, 40 + 20 * index, 210 - 25 * index), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


class TrackVideoTests(unittest.TestCase):
    def test_synthetic_clips_1_to_5_keep_layout_counts_order_and_padding(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            for count in range(1, 6):
                source = root / f"clip_{count}.mp4"
                write_clip(source, count)
                source_frames = read_frames(source)
                model = StubModel([empty_group() for _ in range(math.ceil(count / 3))])

                video_path, csv_path, processed = track_video.process_video(
                    source, output_dir, model, torch.device("cpu")
                )

                expected_folder = output_dir / source.stem
                self.assertEqual(video_path, expected_folder / f"{source.stem}_tracked.mp4")
                self.assertEqual(csv_path, expected_folder / f"{source.stem}_ball.csv")
                self.assertEqual(processed, count)
                output_frames = read_frames(video_path)
                self.assertEqual(len(output_frames), count)
                with csv_path.open(newline="", encoding="utf-8") as csv_file:
                    rows = list(csv.DictReader(csv_file))
                self.assertEqual(len(rows), count)
                self.assertEqual([int(row["Frame"]) for row in rows], list(range(count)))
                self.assertEqual(list(rows[0]), ["Frame", "Visibility", "X", "Y"])

                for group_index, batch in enumerate(model.inputs):
                    self.assertEqual(tuple(batch.shape), (1, 9, *IMAGE_SIZE))
                    for slot in range(3):
                        frame_index = min(group_index * 3 + slot, count - 1)
                        rgb = cv2.cvtColor(source_frames[frame_index], cv2.COLOR_BGR2RGB)
                        expected = track_video.torchvision.transforms.functional.resize(
                            track_video.torchvision.transforms.ToTensor()(rgb), IMAGE_SIZE, antialias=True
                        )
                        torch.testing.assert_close(batch[0, slot * 3:(slot + 1) * 3], expected)

                output_capture = cv2.VideoCapture(str(video_path))
                self.assertEqual(output_capture.get(cv2.CAP_PROP_FRAME_WIDTH), 80)
                self.assertEqual(output_capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 40)
                self.assertAlmostEqual(output_capture.get(cv2.CAP_PROP_FPS), 12.0, places=1)
                output_capture.release()

            original_video = (output_dir / "clip_5" / "clip_5_tracked.mp4").read_bytes()
            original_csv = (output_dir / "clip_5" / "clip_5_ball.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                track_video.process_video(
                    root / "clip_5.mp4", output_dir, StubModel([empty_group(), empty_group()]), torch.device("cpu")
                )
            self.assertEqual((output_dir / "clip_5" / "clip_5_tracked.mp4").read_bytes(), original_video)
            self.assertEqual((output_dir / "clip_5" / "clip_5_ball.csv").read_bytes(), original_csv)

    def test_detection_scaling_empty_maps_and_strict_threshold(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as temp_dir:
            root = Path(temp_dir)
            source = root / "detections.mp4"
            write_clip(source, 5)
            predictions = [
                prediction_group([(128, 144, 0.51), (256, 144, 0.5), (256, 72, 0.9)]),
                prediction_group([(384, 216, 0.9), None, (500, 280, 0.9)]),
            ]
            model = StubModel(predictions)

            with patch.object(track_video.cv2, "circle", wraps=cv2.circle) as circle:
                _, csv_path, processed = track_video.process_video(
                    source, root / "outputs", model, torch.device("cpu")
                )

            self.assertEqual(processed, 5)
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                [(int(row["Visibility"]), int(row["X"]), int(row["Y"])) for row in rows],
                [(1, 20, 20), (0, 0, 0), (1, 40, 10), (1, 60, 30), (0, 0, 0)],
            )
            self.assertEqual(circle.call_count, 3)
            self.assertEqual(circle.call_args_list[0].args[2:], (8, (0, 0, 255), -1))

    def test_cli_defaults_are_script_relative(self):
        args = track_video.parse_args(["--source", "input.mp4"])
        self.assertEqual(args.weights, SCRIPT_DIR / "track.pt")
        self.assertEqual(args.output_dir, SCRIPT_DIR / "runs")


class RealCheckpointForwardTest(unittest.TestCase):
    @unittest.skipUnless((SCRIPT_DIR / "track.pt").is_file(), "track.pt is not present")
    def test_checkpoint_loads_and_returns_finite_probability_maps(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = track_video.load_model(SCRIPT_DIR / "track.pt", device)
        with torch.inference_mode():
            output = model(torch.zeros((1, 9, *IMAGE_SIZE), device=device))
        self.assertEqual(tuple(output.shape), (1, 3, 288, 512))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
