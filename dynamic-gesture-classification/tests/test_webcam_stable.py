import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import webcam_stable


class FakeCv2:
    def __init__(self, positions):
        self.positions = dict(positions)
        self.created = []
        self.minimums = []
        self.synced = []

    def namedWindow(self, title):
        self.window = title

    def createTrackbar(self, name, title, position, maximum, callback):
        self.created.append((name, title, position, maximum))

    def setTrackbarMin(self, name, title, minimum):
        self.minimums.append((name, title, minimum))

    def getTrackbarPos(self, name, title):
        return self.positions[name]

    def setTrackbarPos(self, name, title, position):
        self.positions[name] = position
        self.synced.append((name, title, position))


class StableWebcamTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.session = SimpleNamespace(current_valid=True, prediction=None)

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def push(self, controller, label, score, timestamp):
        self.session.prediction = (label, score)
        controller.update(self.session, timestamp, True)

    def confirm(self, controller, label="gesture A", start=0.0):
        for index in range(controller.vote_window):
            self.push(controller, label, 0.9, start + index * 0.2)

    def test_fresh_predictions_are_counted_once(self):
        controller = webcam_stable.StableController()
        self.push(controller, "gesture A", 0.9, 0.0)
        for timestamp in (0.01, 0.05, 0.1, 0.19):
            controller.update(self.session, timestamp, False)
        self.assertEqual(list(controller.votes), ["gesture A"])

    def test_confidence_below_equal_and_above_threshold(self):
        for score, expected in ((0.699, None), (0.70, "gesture A"), (0.701, "gesture A")):
            with self.subTest(score=score):
                controller = webcam_stable.StableController(vote_window=1, required_votes=1)
                self.push(controller, "gesture A", score, 0.0)
                self.assertEqual(controller.confirmed_label, expected)

    def test_default_four_of_five_confirms_newest_label_after_isolated_wrong_vote(self):
        controller = webcam_stable.StableController()
        for index, label in enumerate(("gesture A", "gesture A", "gesture B", "gesture A", "gesture A")):
            self.push(controller, label, 0.9, index * 0.2)
        self.assertEqual(controller.confirmed_label, "gesture A")
        self.assertEqual(list(controller.votes), ["gesture A", "gesture A", "gesture B", "gesture A", "gesture A"])

    def test_newest_label_ties_and_configurable_voting(self):
        newest_wrong = webcam_stable.StableController()
        for index, label in enumerate(("gesture A", "gesture A", "gesture A", "gesture A", "gesture B")):
            self.push(newest_wrong, label, 0.9, index * 0.2)
        self.assertIsNone(newest_wrong.confirmed_label)

        tied = webcam_stable.StableController(vote_window=4, required_votes=2)
        for index, label in enumerate(("gesture A", "gesture A", "gesture B", "gesture B")):
            self.push(tied, label, 0.9, index * 0.2)
        self.assertIsNone(tied.confirmed_label)

        configurable = webcam_stable.StableController(vote_window=3, required_votes=2)
        for index, label in enumerate(("gesture A", "gesture B", "gesture B")):
            self.push(configurable, label, 0.9, index * 0.2)
        self.assertEqual(configurable.confirmed_label, "gesture B")

    def test_hold_expiry_is_checked_without_a_new_prediction(self):
        controller = webcam_stable.StableController()
        self.confirm(controller)
        self.session.prediction = controller.last_prediction
        controller.update(self.session, 1.4, False)
        self.assertEqual(controller.status_text, "Holding: gesture A")
        controller.update(self.session, 1.4001, False)
        self.assertIsNone(controller.confirmed_label)
        self.assertEqual(controller.status_text, "Uncertain")

    def test_background_votes_are_inactive_and_do_not_end_a_live_hold(self):
        controller = webcam_stable.StableController()
        for index in range(5):
            self.push(controller, "No gesture", 0.95, index * 0.2)
        self.assertIsNone(controller.confirmed_label)
        self.assertEqual(controller.status_text, "Inactive")

        controller = webcam_stable.StableController()
        self.confirm(controller)
        self.push(controller, "Doing other things", 0.95, 1.0)
        self.assertEqual(controller.status_text, "Holding: gesture A")
        controller.update(self.session, 1.6001, False)
        self.assertEqual(controller.status_text, "Inactive")

    def test_hand_loss_clears_votes_but_allows_the_brief_hold(self):
        controller = webcam_stable.StableController()
        self.confirm(controller)
        self.session.current_valid = False
        self.session.prediction = None
        controller.update(self.session, 0.9, False)
        self.assertEqual(list(controller.votes), [])
        self.assertEqual(controller.status_text, "Holding: gesture A")
        controller.update(self.session, 1.41, False)
        self.assertIsNone(controller.confirmed_label)
        self.assertEqual(controller.status_text, "No hand")

    def test_prediction_gap_over_one_second_clears_votes_and_confirmation(self):
        controller = webcam_stable.StableController()
        self.confirm(controller)
        controller.update(self.session, 1.8001, False)
        self.assertEqual(list(controller.votes), [])
        self.assertIsNone(controller.confirmed_label)
        self.assertIsNone(self.session.prediction)

    def test_model_switch_clears_votes_and_confirmation(self):
        controller = webcam_stable.StableController()
        self.confirm(controller)
        self.session.prediction = None
        controller.model_switched(self.session)
        self.assertEqual(list(controller.votes), [])
        self.assertIsNone(controller.confirmed_label)
        self.assertEqual(controller.status_text, "Uncertain")

    def test_shared_camera_loop_resets_after_a_successful_switch_and_cleans_up(self):
        class FakeSession:
            def __init__(self, experiment, loaded, window_seconds):
                self.model_name = loaded["model_name"]
                self.current_valid = False
                self.current_pose = np.zeros((21, 2), dtype=np.float32)
                self.prediction = None
                self.switch_error = None

            def add_sample(self, timestamp, candidates):
                pass

            def predict_if_due(self, timestamp):
                return False

            def warmed_up(self, timestamp):
                return False

            def switch_model(self, model_name):
                self.model_name = model_name
                return True

        class FakeCapture:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((20, 20, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        capture, detector = FakeCapture(), SimpleNamespace(close=Mock())
        detector.detect = lambda frame, timestamp: {"xy": np.empty((0, 21, 2), dtype=np.float32)}
        controller = webcam_stable.StableController()
        self.confirm(controller)
        args = SimpleNamespace(
            experiment=Path(__file__).resolve().parent,
            window_seconds=3.0,
            camera=0,
            hand_model=Path(__file__).resolve(),
        )
        loaded = {"model_name": "gru"}
        with patch.object(webcam_stable.webcam, "LiveSession", FakeSession), \
                patch.object(webcam_stable.webcam.cv2, "VideoCapture", return_value=capture), \
                patch.object(webcam_stable.webcam.landmarks, "MediaPipeHand", return_value=detector), \
                patch.object(webcam_stable.webcam.cv2, "namedWindow"), \
                patch.object(webcam_stable.webcam.cv2, "createTrackbar"), \
                patch.object(webcam_stable.webcam.cv2, "setTrackbarMin"), \
                patch.object(webcam_stable.webcam.cv2, "getTrackbarPos", side_effect=lambda name, title: 4 if name == webcam_stable.VOTE_TRACKBAR else 3), \
                patch.object(webcam_stable.webcam.cv2, "imshow"), \
                patch.object(webcam_stable.webcam.cv2, "waitKey", side_effect=[ord("2"), ord("q")]), \
                patch.object(webcam_stable.webcam.cv2, "destroyAllWindows"), \
                patch.object(webcam_stable.webcam.time, "monotonic", side_effect=[1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06]):
            webcam_stable.webcam.run_camera(args, loaded, controller=controller)
        self.assertTrue(capture.released)
        detector.close.assert_called_once()
        self.assertIsNone(controller.confirmed_label)
        self.assertEqual(controller.status_text, "No hand")

    def test_setting_changes_reset_state_and_bracket_keys_clamp(self):
        controller = webcam_stable.StableController()
        self.confirm(controller)
        self.assertTrue(controller.set_settings(min_confidence=0.75))
        self.assertIsNone(controller.confirmed_label)
        self.assertEqual(list(controller.votes), [])

        low = webcam_stable.StableController(min_confidence=0.0, vote_window=1, required_votes=1)
        self.push(low, "gesture A", 1.0, 0.0)
        low.handle_key(ord("["))
        self.assertEqual(low.min_confidence, 0.0)
        self.assertEqual(low.confirmed_label, "gesture A")
        low.handle_key(ord("]"))
        self.assertEqual(low.min_confidence, 0.05)
        self.assertIsNone(low.confirmed_label)

        high = webcam_stable.StableController(min_confidence=1.0)
        high.handle_key(ord("]"))
        self.assertEqual(high.min_confidence, 1.0)

    def test_trackbar_bounds_and_required_vote_synchronization(self):
        controller = webcam_stable.StableController()
        fake = FakeCv2({webcam_stable.VOTE_TRACKBAR: 3, webcam_stable.REQUIRED_TRACKBAR: 5})
        controller.start(fake, webcam_stable.WINDOW_TITLE)
        self.assertEqual(fake.created, [
            (webcam_stable.VOTE_TRACKBAR, webcam_stable.WINDOW_TITLE, 5, 25),
            (webcam_stable.REQUIRED_TRACKBAR, webcam_stable.WINDOW_TITLE, 4, 25),
        ])
        self.assertEqual(fake.minimums, [
            (webcam_stable.VOTE_TRACKBAR, webcam_stable.WINDOW_TITLE, 1),
            (webcam_stable.REQUIRED_TRACKBAR, webcam_stable.WINDOW_TITLE, 1),
        ])
        self.confirm(controller)
        controller.poll_controls(fake)
        self.assertEqual((controller.vote_window, controller.required_votes), (3, 3))
        self.assertEqual(fake.synced, [(webcam_stable.REQUIRED_TRACKBAR, webcam_stable.WINDOW_TITLE, 3)])
        self.assertEqual(list(controller.votes), [])
        self.assertIsNone(controller.confirmed_label)

        fake.positions[webcam_stable.VOTE_TRACKBAR] = 100
        fake.positions[webcam_stable.REQUIRED_TRACKBAR] = -4
        controller.poll_controls(fake)
        self.assertEqual((controller.vote_window, controller.required_votes), (25, 1))

    def test_cli_rejects_invalid_settings_before_loading_a_model(self):
        invalid = (
            ["--min-confidence", "nan"],
            ["--min-confidence", "1.01"],
            ["--vote-window", "0"],
            ["--vote-window", "2", "--required-votes", "3"],
        )
        for flags in invalid:
            with self.subTest(flags=flags), patch.object(webcam_stable.webcam, "load_model") as load:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    webcam_stable.main(["--experiment", "unused", *flags])
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
