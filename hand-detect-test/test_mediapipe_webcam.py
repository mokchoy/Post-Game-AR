"""Check webcam handling without opening hardware."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mediapipe_webcam as webcam


class MediaPipeWebcamTest(unittest.TestCase):
    def test_empty_frame_then_hand_and_increasing_timestamps(self):
        frame = object()
        landmarks = [object()]
        camera = Mock()
        camera.read.return_value = (True, frame)
        with patch.object(webcam.vision.HandLandmarker, "create_from_options") as create, \
                patch.object(webcam, "cv2") as cv, \
                patch.object(webcam.mp, "Image"), \
                patch.object(webcam.time, "monotonic_ns", return_value=10_000_000), \
                patch.object(webcam.vision.drawing_utils, "draw_landmarks") as draw:
            hands = create.return_value.__enter__.return_value
            hands.detect_for_video.side_effect = [
                SimpleNamespace(hand_landmarks=[]),
                SimpleNamespace(hand_landmarks=[landmarks]),
            ]
            cv.VideoCapture.return_value = camera
            cv.waitKey.side_effect = [-1, ord("q")]
            cv.getWindowProperty.return_value = 1
            webcam.main()

        options = create.call_args.args[0]
        self.assertEqual(options.base_options.delegate, webcam.mp.tasks.BaseOptions.Delegate.CPU)
        self.assertEqual(options.running_mode, webcam.vision.RunningMode.VIDEO)
        self.assertEqual(options.num_hands, 2)
        self.assertEqual([call.args[1] for call in hands.detect_for_video.call_args_list], [10, 11])
        draw.assert_called_once_with(frame, landmarks, webcam.vision.HandLandmarksConnections.HAND_CONNECTIONS)
        self.assertEqual(cv.imshow.call_count, 2)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()
        create.return_value.__exit__.assert_called_once()

    def test_camera_failures_release_resources(self):
        for opened in (False, True):
            with self.subTest(opened=opened), \
                    patch.object(webcam.vision.HandLandmarker, "create_from_options") as create, \
                    patch.object(webcam, "cv2") as cv:
                camera = cv.VideoCapture.return_value
                camera.isOpened.return_value = opened
                camera.read.return_value = (False, None)
                with self.assertRaisesRegex(RuntimeError, "webcam 0"):
                    webcam.main()
                camera.release.assert_called_once()
                cv.destroyAllWindows.assert_called_once()
                create.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
