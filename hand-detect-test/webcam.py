"""Live hand keypoints on CPU. Press Q or Esc to quit.

Install: python -m pip install rtmlib onnxruntime opencv-python
Run from repo root: python hand-detect-test/webcam.py
RTMLib downloads its default models on first run.
"""

import cv2
from rtmlib import Hand, draw_skeleton


def main():
    hand = Hand(mode="lightweight", backend="onnxruntime", device="cpu")
    camera = cv2.VideoCapture(0)
    title = "RTMLib hands - Q / Esc to quit"
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open webcam 0.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Cannot read a frame from webcam 0.")

            boxes = hand.det_model(frame)
            # RTMPose treats empty boxes as a request to estimate the entire frame.
            if len(boxes):
                keypoints, scores = hand.pose_model(frame, bboxes=boxes)
                frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)

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
