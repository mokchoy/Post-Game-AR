import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from general import get_shuttle_position
from tracknet import TrackNet


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (288, 512)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Local input video path.")
    parser.add_argument("--weights", type=Path, default=SCRIPT_DIR / "track.pt", help="TrackNet state dictionary.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs", help="Parent output directory.")
    return parser.parse_args(argv)


def load_model(weights, device):
    model = TrackNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    model.eval()
    return model


def preprocess_group(frames, device, to_tensor):
    tensors = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(rgb).to(device)
        tensors.append(torchvision.transforms.functional.resize(tensor, IMAGE_SIZE, antialias=True))
    return torch.cat(tensors, dim=0).unsqueeze(0)


def process_video(source, output_dir, model, device):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")

    output_dir = Path(output_dir).resolve()
    output_folder = output_dir / source.stem
    video_path = output_folder / f"{source.stem}_tracked.mp4"
    csv_path = output_folder / f"{source.stem}_ball.csv"

    capture = cv2.VideoCapture(str(source))
    writer = None
    csv_file = None
    created_paths = []
    completed = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {source}")

        width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not all(math.isfinite(value) and value > 0 for value in (width_value, height_value, fps)):
            raise ValueError("Input video metadata must have positive finite width, height, and FPS")
        width, height = int(width_value), int(height_value)
        if width == 0 or height == 0:
            raise ValueError("Input video metadata has an invalid resolution")

        output_folder.mkdir(parents=True, exist_ok=True)
        existing = [path for path in (video_path, csv_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite output file(s): {', '.join(map(str, existing))}")

        try:
            with video_path.open("xb"):
                pass
            created_paths.append(video_path)
            csv_file = csv_path.open("x", newline="", encoding="utf-8")
            created_paths.append(csv_path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite existing output: {error.filename}") from error

        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize output video writer: {video_path}")

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(("Frame", "Visibility", "X", "Y"))
        to_tensor = torchvision.transforms.ToTensor()
        frame_index = 0

        while True:
            frames = []
            for _ in range(3):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break

            actual_count = len(frames)
            # Repeated context can affect edge predictions; padded frames are not exported.
            frames.extend([frames[-1]] * (3 - actual_count))
            batch = preprocess_group(frames, device, to_tensor)
            with torch.inference_mode():
                predictions = model(batch)
            if tuple(predictions.shape) != (1, 3, *IMAGE_SIZE):
                raise RuntimeError(f"Unexpected model output shape: {tuple(predictions.shape)}")
            predictions = predictions[0].detach().cpu().numpy()

            for index in range(actual_count):
                mask = (predictions[index] > 0.5).astype(np.uint8) * 255
                visible, x, y = get_shuttle_position(mask)
                if visible:
                    x = int(x * width / IMAGE_SIZE[1])
                    y = int(y * height / IMAGE_SIZE[0])
                    cv2.circle(frames[index], (x, y), 8, (0, 0, 255), -1)
                else:
                    x = y = 0

                writer.write(frames[index])
                csv_writer.writerow((frame_index, int(visible), x, y))
                frame_index += 1

        if frame_index == 0:
            raise RuntimeError(f"No frames could be decoded from input video: {source}")
        completed = True
        return video_path, csv_path, frame_index
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if csv_file is not None:
            csv_file.close()
        if not completed:
            for path in created_paths:
                path.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        if not args.source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {args.source}")
        if not args.weights.is_file():
            raise FileNotFoundError(f"Weights file does not exist: {args.weights}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.weights, device)
        video_path, csv_path, frame_count = process_video(args.source, args.output_dir, model, device)
    except Exception as error:
        print(f"track_video: error: {error}", file=sys.stderr)
        return 1

    print(f"Tracked video: {video_path}")
    print(f"Ball CSV: {csv_path}")
    print(f"Processed frames: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
