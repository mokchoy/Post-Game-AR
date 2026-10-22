"""Real-time hand and exposed-arm segmentation.

Uses MediaPipe's pre-trained Hand Landmarker to find a small hand/forearm
region. Supply a trained binary TensorFlow Lite model for pixel segmentation;
otherwise the module uses a lower-accuracy adaptive skin-colour fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter


DEFAULT_MODEL = (Path(__file__).resolve().parents[1] /
                 "demo_hand_gesture_recognition" / "hand_landmarker.task")
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class TFLiteBinarySegmenter:
    """Runs a hand/arm foreground TFLite model within a landmark-derived ROI."""

    def __init__(self, model_path: str | Path, threshold: float = 0.5,
                 foreground_class: int = 1):
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        _, self.input_height, self.input_width, channels = self.input_detail["shape"]
        if channels != 3:
            raise ValueError("segmentation model must accept RGB images")
        self.threshold = threshold
        self.foreground_class = foreground_class

    def _input_tensor(self, rgb: np.ndarray) -> np.ndarray:
        image = cv.resize(rgb, (self.input_width, self.input_height))
        dtype = self.input_detail["dtype"]
        if np.issubdtype(dtype, np.floating):
            return (image.astype(np.float32) / 255.0)[None, ...]
        scale, zero_point = self.input_detail["quantization"]
        if not scale:
            return image[None, ...].astype(dtype)
        return np.clip(image / 255.0 / scale + zero_point, 0, 255).astype(dtype)[None, ...]

    def _probability(self) -> np.ndarray:
        output = self.interpreter.get_tensor(self.output_detail["index"])[0]
        scale, zero_point = self.output_detail["quantization"]
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale
        output = np.squeeze(output)
        if output.ndim == 3:
            output = output[..., self.foreground_class]
        if output.min() < 0 or output.max() > 1:
            output = 1 / (1 + np.exp(-output))
        return output

    def segment(self, frame: np.ndarray, roi: np.ndarray) -> np.ndarray:
        points = cv.findNonZero(roi)
        if points is None:
            return np.zeros(frame.shape[:2], dtype=bool)
        x, y, width, height = cv.boundingRect(points)
        crop = frame[y:y + height, x:x + width]
        rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        self.interpreter.set_tensor(self.input_detail["index"], self._input_tensor(rgb))
        self.interpreter.invoke()
        probability = cv.resize(self._probability(), (width, height))
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[y:y + height, x:x + width] = probability >= self.threshold
        return mask & roi.astype(bool)


class HandArmSegmenter:
    """Segments exposed skin connected to one or more detected hands."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL,
                 max_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 segmentation_model: str | Path | None = None,
                 arm_length_hand_sizes: float = 4.0):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {model_path}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.hand_landmarks = ()
        self._last_timestamp_ms = 0
        self.arm_length_hand_sizes = arm_length_hand_sizes
        self.pixel_segmenter = (
            TFLiteBinarySegmenter(segmentation_model)
            if segmentation_model else None)

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _landmark_masks(self, shape: tuple[int, int], hand_landmarks):
        height, width = shape
        sample = np.zeros((height, width), dtype=np.uint8)
        region = np.zeros((height, width), dtype=np.uint8)
        for hand in hand_landmarks:
            points = np.array([
                (min(int(landmark.x * width), width - 1),
                 min(int(landmark.y * height), height - 1))
                for landmark in hand
            ], dtype=np.int32)
            palm_size = max(12, int(np.linalg.norm(points[0] - points[9])))
            for point in points:
                cv.circle(sample, tuple(point), max(2, palm_size // 20), 255, -1)
            cv.fillConvexPoly(region, cv.convexHull(points), 255)
            direction = points[0] - points[9]
            length = np.linalg.norm(direction)
            if length:
                end = points[0] + (direction / length *
                                   palm_size * self.arm_length_hand_sizes).astype(int)
                cv.line(region, tuple(points[0]), tuple(end), 255, palm_size)
        kernel_size = max(3, min(height, width) // 50)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return sample, cv.dilate(region, kernel, iterations=1)

    @staticmethod
    def _adaptive_skin_mask(frame: np.ndarray, seed: np.ndarray) -> np.ndarray:
        ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        samples = ycrcb[seed > 0, 1:]
        if not len(samples):
            return np.zeros(seed.shape, dtype=np.uint8)
        centre = np.median(samples, axis=0)
        spread = np.clip(3 * np.median(np.abs(samples - centre), axis=0), 10, 25)
        lower = np.array((0, *(centre - spread).clip(0, 255)), dtype=np.uint8)
        upper = np.array((255, *(centre + spread).clip(0, 255)), dtype=np.uint8)
        mask = cv.inRange(ycrcb, lower, upper)
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

    @staticmethod
    def _connected_to_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
        count, labels = cv.connectedComponents(mask)
        selected = np.zeros_like(mask)
        for label in range(1, count):
            component = labels == label
            if np.any(seed[component]):
                selected[component] = 255
        return selected

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Return a boolean mask for all detected hands and exposed arms."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        timestamp_ms = max(int(monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        self.hand_landmarks = result.hand_landmarks
        if not self.hand_landmarks:
            return np.zeros(frame.shape[:2], dtype=bool)
        sample, region = self._landmark_masks(frame.shape[:2], self.hand_landmarks)
        if self.pixel_segmenter:
            return self.pixel_segmenter.segment(frame, region)
        skin_mask = cv.bitwise_and(self._adaptive_skin_mask(frame, sample), region)
        return self._connected_to_seed(skin_mask, sample).astype(bool)


def draw_segmentation(frame: np.ndarray, mask: np.ndarray,
                      hand_landmarks=()) -> np.ndarray:
    """Overlay segmented pixels in red and detected landmarks in green."""
    display = frame.copy()
    overlay = frame.copy()
    overlay[mask] = (0, 0, 255)
    display = cv.addWeighted(overlay, 0.6, display, 0.4, 0)
    height, width = frame.shape[:2]
    for hand in hand_landmarks:
        points = [
            (min(int(landmark.x * width), width - 1),
             min(int(landmark.y * height), height - 1))
            for landmark in hand
        ]
        for start, end in HAND_CONNECTIONS:
            cv.line(display, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv.circle(display, point, 3, (0, 255, 0), -1)
    return display


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--segmentation-model", type=Path,
                        help="trained binary hand/arm .tflite model")
    parser.add_argument("--arm-length", type=float, default=4.0,
                        help="forearm ROI length in palm sizes")
    args = parser.parse_args(argv)

    camera = cv.VideoCapture(args.device, cv.CAP_DSHOW)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {args.device}")
    with HandArmSegmenter(args.model,
                          segmentation_model=args.segmentation_model,
                          arm_length_hand_sizes=args.arm_length) as segmenter:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            mask = segmenter.segment(frame)
            cv.imshow("Hand and Arm Segmentation", draw_segmentation(
                frame, mask, segmenter.hand_landmarks))
            if cv.waitKey(1) & 0xFF in (27, ord("q")):
                break
    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
