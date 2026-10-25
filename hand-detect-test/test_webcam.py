"""Check the webcam loop without opening hardware or downloading models."""

import unittest
from unittest.mock import Mock, patch

import webcam


class WebcamTest(unittest.TestCase):
    def test_empty_frame_then_hand_and_exit(self):
        frame = object()
        boxes = [[1, 2, 30, 40]]
        camera = Mock()
        camera.read.return_value = (True, frame)
        hand = Mock()
        hand.det_model.side_effect = [[], boxes]
        hand.pose_model.return_value = ("keypoints", "scores")

        with patch.object(webcam, "Hand", return_value=hand) as model, \
                patch.object(webcam, "cv2") as cv, \
                patch.object(webcam, "draw_skeleton", return_value=frame) as draw:
            cv.VideoCapture.return_value = camera
            cv.waitKey.side_effect = [-1, ord("q")]
            cv.getWindowProperty.return_value = 1
            webcam.main()

        model.assert_called_once_with(mode="lightweight", backend="onnxruntime", device="cpu")
        hand.pose_model.assert_called_once_with(frame, bboxes=boxes)
        draw.assert_called_once_with(frame, "keypoints", "scores", kpt_thr=0.5)
        self.assertEqual(cv.imshow.call_count, 2)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()

    def test_camera_failures_release_resources(self):
        for opened in (False, True):
            with self.subTest(opened=opened), \
                    patch.object(webcam, "Hand"), \
                    patch.object(webcam, "cv2") as cv:
                camera = cv.VideoCapture.return_value
                camera.isOpened.return_value = opened
                camera.read.return_value = (False, None)
                with self.assertRaisesRegex(RuntimeError, "webcam 0"):
                    webcam.main()
                camera.release.assert_called_once()
                cv.destroyAllWindows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
