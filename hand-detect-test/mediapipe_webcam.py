"""Live MediaPipe hand keypoints on CPU. Press Q or Esc to quit.

Install: python -m pip install mediapipe opencv-python
Run from repo root: python hand-detect-test/mediapipe_webcam.py
Uses the existing demo_hand_gesture_recognition/hand_landmarker.task model.
"""

from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


def main():
    model = Path(__file__).resolve().parents[1] / "demo_hand_gesture_recognition" / "hand_landmarker.task"
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    with vision.HandLandmarker.create_from_options(options) as hands:
        camera = cv2.VideoCapture(0)
        title = "MediaPipe hands - Q / Esc to quit"
        timestamp_ms = -1
        try:
            if not camera.isOpened():
                raise RuntimeError("Cannot open webcam 0.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Cannot read a frame from webcam 0.")

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                result = hands.detect_for_video(image, timestamp_ms)
                for landmarks in result.hand_landmarks:
                    vision.drawing_utils.draw_landmarks(
                        frame, landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS
                    )

                cv2.imshow(title, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
