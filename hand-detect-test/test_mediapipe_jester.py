"""Check frame order, overlay calls, and saved MP4 contents without inference."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import mediapipe_jester as jester


class JesterOverlayTest(unittest.TestCase):
    def test_render_preserves_frame_order_and_frames_without_hands(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            folder = Path(directory)
            for name, value in (("10.jpg", 200), ("2.jpg", 100), ("1.jpg", 0)):
                self.assertTrue(jester.cv2.imwrite(str(folder / name), np.full((100, 132, 3), value, np.uint8)))
            landmarks = [object()]
            hands = Mock()
            hands.detect_for_video.side_effect = [
                SimpleNamespace(hand_landmarks=[]),
                SimpleNamespace(hand_landmarks=[landmarks]),
                SimpleNamespace(hand_landmarks=[]),
            ]
            output = folder / "overlay.mp4"
            with patch.object(jester.vision.drawing_utils, "draw_landmarks") as draw:
                jester.render_clip(folder, output, hands)
            draw.assert_called_once()
            self.assertIs(draw.call_args.args[1], landmarks)
            self.assertEqual([call.args[1] for call in hands.detect_for_video.call_args_list], [0, 83, 167])
            capture = jester.cv2.VideoCapture(str(output))
            try:
                self.assertAlmostEqual(capture.get(jester.cv2.CAP_PROP_FPS), 12)
                for expected in (0, 100, 200):
                    ok, frame = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape[:2], (100, 132))
                    self.assertLess(abs(float(frame.mean()) - expected), 8)
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                jester.render_clip(folder, output, hands)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
