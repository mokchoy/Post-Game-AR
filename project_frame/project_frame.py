"""Click four scene corners (top-left, top-right, bottom-right, bottom-left)."""

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


WINDOW = "AR GIF  |  click TL, TR, BR, BL  |  r reset  |  esc quit"


def gif_frames(path):
    """Return GIF frames as BGRA images, preserving its animation and alpha."""
    image = Image.open(path)
    frames = []
    try:
        while True:
            rgba = np.asarray(image.convert("RGBA"))
            frames.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            image.seek(image.tell() + 1)
    except EOFError:
        return frames


def composite(frame, graphic, corners):
    height, width = graphic.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(source, np.float32(corners))
    warped = cv2.warpPerspective(graphic, transform, (frame.shape[1], frame.shape[0]))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255
    return (frame * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)


class PlaneTracker:
    def __init__(self):
        self.corners = []
        self.reference = self.previous = self.reference_gray = None

    def reset(self):
        self.corners = []
        self.reference = self.previous = self.reference_gray = None

    def initialise(self, gray):
        polygon = np.int32([self.corners])
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, polygon, 255)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01,
                                        minDistance=7, mask=mask)
        if points is None or len(points) < 4:
            print("Need at least four visible textured features inside the selected area; reset and try again.")
            self.reset()
            return
        self.reference = self.previous = points
        self.reference_gray = gray.copy()

    def update(self, gray):
        if self.previous is None:
            return self.corners if len(self.corners) == 4 else None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.reference_gray, gray, self.reference, None)
        valid = status.ravel().astype(bool)
        if valid.sum() < 4:
            print("Tracking lost; press r and select the four corners again.")
            self.reset()
            return None
        homography, _ = cv2.findHomography(self.reference[valid], moved[valid], cv2.RANSAC, 3)
        if homography is None:
            return None
        return cv2.perspectiveTransform(np.float32([self.corners]), homography)[0]


def run(camera_index=0):
    frames = gif_frames(Path(__file__).with_name("placeholder.gif"))
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_index}.")

    tracker = PlaneTracker()

    def on_click(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN and len(tracker.corners) < 4:
            tracker.corners.append([x, y])

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    frame_number = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam frame capture failed.")
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(tracker.corners) == 4 and tracker.previous is None:
                tracker.initialise(gray)
            corners = tracker.update(gray)
            if corners is not None:
                frame = composite(frame, frames[frame_number % len(frames)], corners)
            frame_number += 1

            for i, point in enumerate(corners if corners is not None else tracker.corners):
                center = tuple(map(int, point))
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (center[0] + 8, center[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("r"):
                tracker.reset()
    finally:
        camera.release()
        cv2.destroyAllWindows()


def self_check():
    frame = np.zeros((20, 20, 3), np.uint8)
    graphic = np.zeros((2, 2, 4), np.uint8)
    graphic[:, :, :3], graphic[:, :, 3] = (0, 255, 0), 255
    result = composite(frame, graphic, [[2, 2], [17, 2], [17, 17], [2, 17]])
    assert result[10, 10].tolist() == [0, 255, 0]


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
