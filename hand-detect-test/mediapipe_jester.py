"""Save hand landmark overlays for two selected cached Jester clips.

Run from repo root: python hand-detect-test/mediapipe_jester.py
Outputs: hand-detect-test/output/<clip_id>.mp4 (existing files are protected).
Uses installed MediaPipe/OpenCV and the existing local hand_landmarker.task.
"""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "dynamic-gesture-classification" / "cache" / "frames"
MODEL = ROOT / "demo_hand_gesture_recognition" / "hand_landmarker.task"
OUTPUT = Path(__file__).resolve().parent / "output"
FPS = 12
CLIP_IDS = ("92481", "83921", "50297", "67970")


def render_clip(folder, output, hands, image_mode=False):
    frames = sorted(folder.glob("*.jpg"), key=lambda path: int(path.stem))
    if not frames:
        raise ValueError(f"No JPG frames in {folder}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Cannot read frame: {frames[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    detected_frames = 0
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create MP4: {output}")
        for index, path in enumerate(frames):
            frame = first if index == 0 else cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Unreadable frame or inconsistent dimensions: {path}")
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = hands.detect(image) if image_mode else hands.detect_for_video(image, round(index * 1000 / FPS))
            detected_frames += bool(result.hand_landmarks)
            for landmarks in result.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS
                )
            writer.write(frame)
    finally:
        writer.release()
    print(f"{output}: {len(frames)} frames, hands detected in {detected_frames}/{len(frames)}")


def main():
    OUTPUT.mkdir(exist_ok=True)
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    for clip_id in CLIP_IDS:
        folder = FRAMES / clip_id
        # A fresh detector prevents tracking state from crossing clip boundaries.
        with vision.HandLandmarker.create_from_options(options) as hands:
            render_clip(folder, OUTPUT / f"{folder.name}.mp4", hands)


if __name__ == "__main__":
    main()
