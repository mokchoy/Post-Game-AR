"""Run full-frame and crop IMAGE detection on clean originals of output clips.

Run: py -3.14 hand-detect-test/mediapipe_image_crop.py [--self-check]
Writes merged-overlay MP4s, recovered-frame JPGs and frames.csv in output/image_crop_dual.
Existing outputs are protected. No downloads, webcam, training or interpolation.
"""

import argparse
import copy
import csv
import math

import numpy as np

from mediapipe_jester import FPS, FRAMES, MODEL, OUTPUT, cv2, mp, vision


def crop_box(hands, width, height):
    boxes = [(min(p.x for p in hand), min(p.y for p in hand),
              max(p.x for p in hand), max(p.y for p in hand)) for hand in hands]
    left, top, right, bottom = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    side = min(width, height, math.ceil(2 * max((right - left) * width, (bottom - top) * height, 24)))
    x = max(0, min(width - side, math.floor((left + right) * width / 2 - side / 2)))
    y = max(0, min(height - side, math.floor((top + bottom) * height / 2 - side / 2)))
    return x, y, x + side, y + side


def remap(hands, box, width, height):
    x1, y1, x2, y2 = box
    mapped = copy.deepcopy(hands)
    for hand in mapped:
        for point in hand:
            point.x = (x1 + point.x * (x2 - x1)) / width
            point.y = (y1 + point.y * (y2 - y1)) / height
    return mapped


def merge_hands(full, cropped):
    merged = list(full)
    # shortcut: wrists within 0.1 normalized image units count as one hand; calibrate matching for nearby hands.
    for hand in cropped:
        if all(math.hypot(hand[0].x - other[0].x, hand[0].y - other[0].y) > 0.1 for other in merged):
            merged.append(hand)
    return merged


def process_frame(frame, detect, box):
    baseline = detect(frame)
    height, width = frame.shape[:2]
    attempted = crop_box(baseline, width, height) if baseline else box
    cropped = []
    # shortcut: one hand region, retained until clip end; no motion prediction or pose-quality filter.
    if attempted is not None:
        x1, y1, x2, y2 = attempted
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_CUBIC)
        cropped = remap(detect(crop), attempted, width, height)
    # Full-frame detections take priority; crop detections update the anchor when the full-frame pass misses.
    anchor = baseline or cropped
    box = crop_box(anchor, width, height) if anchor else box
    return baseline, cropped, merge_hands(baseline, cropped), box, attempted


def panel(frame, hands, title, box=None):
    image = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for hand in hands:
        vision.drawing_utils.draw_landmarks(image, hand, vision.HandLandmarksConnections.HAND_CONNECTIONS)
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1 * 3, y1 * 3), (x2 * 3, y2 * 3), (0, 255, 255), 1)
    cv2.putText(image, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run():
    clips = [(video.stem, sorted((FRAMES / video.stem).glob("*.jpg"), key=lambda p: int(p.stem)))
             for video in sorted(OUTPUT.glob("*.mp4"))]
    if not clips or any(not frames for _, frames in clips):
        raise FileNotFoundError("Each output MP4 must have clean originals under the cached frames directory")
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    destination = OUTPUT / "image_crop_dual"
    destination.mkdir()  # Refuse to overwrite any previous experiment.
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    with vision.HandLandmarker.create_from_options(options) as detector, \
            (destination / "frames.csv").open("x", newline="", encoding="utf-8") as stream:
        report = csv.writer(stream)
        report.writerow(("clip_id", "frame", "image_hands", "crop_attempted", "crop_hands", "recovered_hands", "combined_hands", "crop_xyxy"))

        def detect(frame):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return detector.detect(image).hand_landmarks

        for clip_id, frames in clips:
            box, writer, shape = None, None, None
            baseline_count = recovered_count = attempts = 0
            try:
                for path in frames:
                    frame = cv2.imread(str(path))
                    if frame is None or (shape is not None and frame.shape != shape):
                        raise ValueError(f"Unreadable frame or inconsistent dimensions: {path}")
                    shape = frame.shape
                    baseline, cropped, combined, box, attempted = process_frame(frame, detect, box)
                    recovered = bool(combined) and not baseline
                    baseline_count += bool(baseline)
                    recovered_count += recovered
                    attempts += attempted is not None
                    status = "RECOVERED" if recovered else ("FOUND" if combined else "MISSING")
                    comparison = panel(frame, combined, f"IMAGE + crop: {status} {path.stem}", attempted)
                    if writer is None:
                        height, width = comparison.shape[:2]
                        writer = cv2.VideoWriter(str(destination / f"{clip_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create MP4 for {clip_id}")
                    writer.write(comparison)
                    report.writerow((clip_id, int(path.stem), len(baseline), int(attempted is not None), len(cropped),
                                     len(combined) if recovered else 0, len(combined), str(attempted or "")))
                    if recovered:
                        folder = destination / clip_id
                        folder.mkdir(exist_ok=True)
                        if not cv2.imwrite(str(folder / path.name), comparison):
                            raise RuntimeError(f"Cannot save recovered frame: {path.name}")
            finally:
                if writer is not None:
                    writer.release()
            stream.flush()
            print(f"{clip_id}: IMAGE {baseline_count}/{len(frames)}, crop +{recovered_count}, attempts {attempts}", flush=True)
    print(f"Saved merged overlays: {destination}", flush=True)


def self_check():
    from types import SimpleNamespace
    from unittest.mock import Mock

    hand = [[SimpleNamespace(x=0.25, y=0.25), SimpleNamespace(x=0.75, y=0.75)]]
    box = crop_box(hand, 176, 100)
    assert box == (38, 0, 138, 100)
    assert crop_box([[SimpleNamespace(x=0, y=0)]], 176, 100) == (0, 0, 48, 48)
    mapped = remap(hand, (20, 10, 60, 50), 200, 100)
    assert np.allclose([(p.x, p.y) for p in mapped[0]], [(0.15, 0.2), (0.25, 0.4)])
    assert hand[0][0].x == 0.25  # Remapping must not mutate detector results.
    duplicate = [[SimpleNamespace(x=0.26, y=0.25), SimpleNamespace(x=0.75, y=0.75)]]
    other = [[SimpleNamespace(x=0.9, y=0.1), SimpleNamespace(x=0.95, y=0.2)]]
    merged = merge_hands(hand, duplicate + other)
    assert len(merged) == 2 and merged[0] is hand[0] and merged[1] is other[0]
    crop_only = [[SimpleNamespace(x=0.1, y=0.2), SimpleNamespace(x=0.3, y=0.4)]]
    detect = Mock(side_effect=[hand, other, hand, [], [], crop_only, [], [], [], [], [], []])
    frame = np.zeros((100, 176, 3), np.uint8)
    box = None
    results = []
    for _ in range(6):
        previous_box = box
        baseline, cropped, combined, box, attempted = process_frame(frame, detect, box)
        results.append((bool(baseline), bool(combined), attempted is not None))
        if baseline:
            assert box == crop_box(baseline, 176, 100)
        elif cropped:
            assert box == crop_box(cropped, 176, 100) and box != previous_box
        else:
            assert box == previous_box
    assert results == [(True, True, True), (True, True, True), (False, True, True),
                       (False, False, True), (False, False, True), (False, False, True)]
    assert detect.call_count == 12
    assert detect.call_args_list[1].args[0].shape == (256, 256, 3)
    empty = Mock(return_value=[])
    assert process_frame(frame, empty, None) == ([], [], [], None, None)
    empty.assert_called_once()
    print("Self-check passed: crop bounds, remapping, both passes, full-frame priority, crop updates and merged results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Check geometry and retry logic without inference")
    args = parser.parse_args()
    self_check() if args.self_check else run()
