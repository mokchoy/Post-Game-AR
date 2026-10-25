"""Check IMAGE-only wiring and native frames with a mocked detector."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import mediapipe_image_only as only
import mediapipe_jester as jester


class ImageOnlyTest(unittest.TestCase):
    def test_each_native_frame_uses_image_detection_without_video_tracking(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            folder = Path(directory)
            for name, value in (("10.jpg", 200), ("2.jpg", 100), ("1.jpg", 0)):
                self.assertTrue(jester.cv2.imwrite(str(folder / name), np.full((100, 176, 3), value, np.uint8)))
            hands = Mock()
            hands.detect.return_value = SimpleNamespace(hand_landmarks=[])
            with patch.object(jester.mp, "Image", side_effect=lambda **kwargs: kwargs["data"]), \
                    patch.object(jester.cv2, "VideoWriter") as writer:
                jester.render_clip(folder, folder / "overlay.mp4", hands, image_mode=True)
            hands.detect_for_video.assert_not_called()
            self.assertEqual(hands.detect.call_count, 3)
            for call, value in zip(hands.detect.call_args_list, (0, 100, 200)):
                self.assertEqual(call.args[0].shape, (100, 176, 3))
                self.assertAlmostEqual(float(call.args[0].mean()), value, delta=1)
            self.assertEqual(writer.call_args.args[-1], (176, 100))
            self.assertEqual(writer.return_value.write.call_count, 3)
            writer.return_value.release.assert_called_once()

    def test_all_existing_clips_use_image_mode_and_protect_existing_outputs(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            output = Path(directory)
            names = ("100050", "100503", "50297", "67970", "83921", "92481")
            for name in names:
                (output / f"{name}.mp4").touch()
            model = output / "model.task"
            model.touch()
            with patch.object(only, "OUTPUT", output), patch.object(only, "MODEL", model), \
                    patch.object(only.vision.HandLandmarker, "create_from_options") as create, \
                    patch.object(only, "render_clip") as render:
                only.main()
                options = create.call_args.args[0]
                self.assertEqual(options.running_mode, only.vision.RunningMode.IMAGE)
                self.assertEqual(options.base_options.delegate, only.mp.tasks.BaseOptions.Delegate.CPU)
                self.assertEqual(options.num_hands, 2)
                self.assertEqual([call.args[0].name for call in render.call_args_list], list(names))
                for call in render.call_args_list:
                    self.assertEqual(call.args[0].parent, only.FRAMES)
                    self.assertEqual(call.args[1].parent, output / "image_only")
                    self.assertEqual(call.kwargs, {"image_mode": True})
                create.reset_mock()
                with self.assertRaises(FileExistsError):
                    only.main()
                create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
