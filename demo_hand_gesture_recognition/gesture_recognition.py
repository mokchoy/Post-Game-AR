from collections import deque
from time import monotonic

import numpy as np


ZOOM_THRESHOLD_LOWER = 3
ZOOM_THRESHOLD_UPPER = 6
ZOOM_PERIOD_SECONDS = 0.5
COMMAND_COOLDOWN_SECONDS = 1.0
HAND_DISTANCE_THRESHOLD_LOWER = 1.5
HAND_DISTANCE_THRESHOLD_UPPER = 2.0
HAND_DISTANCE_PERIOD_SECONDS = 0.5
DOWNWARD_TAP_DISTANCE_PIXELS = 15
DOUBLE_TAP_INTERVAL_SECONDS = 0.75
PAUSE_FINGER_ADJACENCY_RATIO = 0.6
PAUSE_BENT_FINGER_MAX_EXTENSION = 1.05
'''
Proposed gestures:
Pause/Play
Seek forward
Go back
Zoom in
Zoom out
'''


class GestureRecognition:
    """Keeps the five most recent hand-sign classifications."""

    def __init__(self, zoom_threshold_lower=ZOOM_THRESHOLD_LOWER,
                 zoom_threshold_upper=ZOOM_THRESHOLD_UPPER,
                 zoom_period_seconds=ZOOM_PERIOD_SECONDS,
                 zoom_direction_cooldown_seconds=COMMAND_COOLDOWN_SECONDS,
                 hand_distance_threshold_lower=HAND_DISTANCE_THRESHOLD_LOWER,
                 hand_distance_threshold_upper=HAND_DISTANCE_THRESHOLD_UPPER,
                 hand_distance_period_seconds=HAND_DISTANCE_PERIOD_SECONDS,
                 downward_tap_distance=DOWNWARD_TAP_DISTANCE_PIXELS,
                 double_tap_interval_seconds=DOUBLE_TAP_INTERVAL_SECONDS,
                 pause_finger_adjacency_ratio=PAUSE_FINGER_ADJACENCY_RATIO,
                 pause_bent_finger_max_extension=PAUSE_BENT_FINGER_MAX_EXTENSION):
        if (not 1 < zoom_threshold_lower < zoom_threshold_upper or
                zoom_period_seconds <= 0 or
                zoom_direction_cooldown_seconds < 0 or
                not 1 < hand_distance_threshold_lower <
                hand_distance_threshold_upper or
                hand_distance_period_seconds <= 0 or
                downward_tap_distance <= 0 or
                double_tap_interval_seconds <= 0 or
                pause_finger_adjacency_ratio <= 0 or
                pause_bent_finger_max_extension <= 0):
            raise ValueError("invalid zoom threshold, period, or cooldown")
        self.history = deque(maxlen=5)
        self.current_gesture = None
        self.zoom_threshold_lower = zoom_threshold_lower
        self.zoom_threshold_upper = zoom_threshold_upper
        self.zoom_period_seconds = zoom_period_seconds
        self.zoom_direction_cooldown_seconds = zoom_direction_cooldown_seconds
        self.hand_distance_threshold_lower = hand_distance_threshold_lower
        self.hand_distance_threshold_upper = hand_distance_threshold_upper
        self.hand_distance_period_seconds = hand_distance_period_seconds
        self.downward_tap_distance = downward_tap_distance
        self.double_tap_interval_seconds = double_tap_interval_seconds
        self.pause_finger_adjacency_ratio = pause_finger_adjacency_ratio
        self.pause_bent_finger_max_extension = pause_bent_finger_max_extension
        self.distance_history = deque()
        self.hand_span_history = deque()
        self.last_command = None
        self.last_command_time = None
        self.tap_start_y = None
        self.tap_bottom_y = None
        self.tap_in_progress = False
        self.first_tap_time = None

    def reset_pause_tap(self):
        self.tap_start_y = None
        self.tap_bottom_y = None
        self.tap_in_progress = False
        self.first_tap_time = None

    def is_pause_hand_shape(self, fingertips, landmarks):
        if landmarks is None:
            return False
        thumb, index, middle, ring, pinky = map(np.asarray, fingertips)
        wrist = np.asarray(landmarks[0])
        hand_size = np.linalg.norm(np.asarray(landmarks[9]) - wrist)
        if hand_size == 0 or np.linalg.norm(index - middle) > (
                hand_size * self.pause_finger_adjacency_ratio):
            return False
        return all(np.linalg.norm(tip - wrist) <=
                   np.linalg.norm(np.asarray(landmarks[joint]) - wrist) *
                   self.pause_bent_finger_max_extension
                   for tip, joint in ((thumb, 3), (ring, 14), (pinky, 18)))

    def is_double_down_tap(self, index_y, middle_y, now):
        if (self.first_tap_time is not None and
                now - self.first_tap_time > self.double_tap_interval_seconds):
            self.first_tap_time = None
        if self.tap_start_y is None:
            self.tap_start_y = (index_y, middle_y)
            return False
        if self.tap_in_progress:
            self.tap_bottom_y = (max(self.tap_bottom_y[0], index_y),
                                 max(self.tap_bottom_y[1], middle_y))
            if (index_y <= self.tap_bottom_y[0] - self.downward_tap_distance / 4 and
                    middle_y <= self.tap_bottom_y[1] - self.downward_tap_distance / 4):
                self.tap_in_progress = False
                self.tap_start_y = (index_y, middle_y)
            return False
        self.tap_start_y = (min(self.tap_start_y[0], index_y),
                            min(self.tap_start_y[1], middle_y))
        if (index_y - self.tap_start_y[0] < self.downward_tap_distance or
                middle_y - self.tap_start_y[1] < self.downward_tap_distance):
            return False
        self.tap_in_progress = True
        self.tap_bottom_y = (index_y, middle_y)
        if (self.first_tap_time is not None and
                now - self.first_tap_time <= self.double_tap_interval_seconds):
            self.first_tap_time = None
            return True
        self.first_tap_time = now
        return False

    def update(self, gesture, right_fingertips=None, motion_gesture=None,
               hand_span=None, now=None, right_landmarks=None):
        self.history.append(gesture)
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            # print(f"Current gesture: {gesture} | Last five: {list(self.history)}",
            #       flush=True)
        if right_fingertips is not None:
            thumb, index, middle, ring, pinky = map(list, right_fingertips)
            thumb_index = np.linalg.norm(np.subtract(thumb, index))
            index_middle = np.linalg.norm(np.subtract(index, middle))
            thumb_middle = np.linalg.norm(np.subtract(thumb, middle))
            separation = np.mean((thumb_index, index_middle, thumb_middle))
            now = monotonic() if now is None else now
            self.distance_history.append((now, separation))

            hand_distance = "STATIONARY"
            if hand_span is not None:
                self.hand_span_history.append((now, hand_span))
                hand_cutoff = now - self.hand_distance_period_seconds
                while (len(self.hand_span_history) > 1 and
                       self.hand_span_history[1][0] <= hand_cutoff):
                    self.hand_span_history.popleft()
                hand_baseline_time, hand_baseline = self.hand_span_history[0]
                if hand_baseline_time <= hand_cutoff:
                    closer_factor = hand_span / max(hand_baseline, 1e-6)
                    further_factor = hand_baseline / max(hand_span, 1e-6)
                    if (self.hand_distance_threshold_lower < closer_factor <
                            self.hand_distance_threshold_upper):
                        hand_distance = "CLOSER"
                    elif (self.hand_distance_threshold_lower < further_factor <
                          self.hand_distance_threshold_upper):
                        hand_distance = "FURTHER"

            cutoff = now - self.zoom_period_seconds
            while (len(self.distance_history) > 1 and
                   self.distance_history[1][0] <= cutoff):
                self.distance_history.popleft()

            if self.is_pause_hand_shape(right_fingertips, right_landmarks):
                pause_detected = self.is_double_down_tap(index[1], middle[1], now)
            else:
                self.reset_pause_tap()
                pause_detected = False
            command = "PAUSE" if pause_detected else "NONE"
            pause_pending = self.tap_in_progress or self.first_tap_time is not None
            if pause_pending and self.last_command != "PAUSE":
                self.last_command = None
                self.last_command_time = None
            else:
                command = {
                    "Clockwise": "ANTICLOCKWISE",
                    "Counter Clockwise": "CLOCKWISE",
                }.get(motion_gesture, command)
            baseline_time, baseline = self.distance_history[0]
            if (command == "NONE" and hand_distance == "FURTHER" and
                    baseline_time <= cutoff):
                decrease_factor = baseline / max(separation, 1e-6)
                increase_factor = separation / max(baseline, 1e-6)
                if self.zoom_threshold_lower < decrease_factor < self.zoom_threshold_upper:
                    command = "ZOOM OUT"
                elif self.zoom_threshold_lower < increase_factor < self.zoom_threshold_upper:
                    command = "ZOOM IN"
            if command == "NONE":
                if (self.last_command is not None and
                        now - self.last_command_time <
                        self.zoom_direction_cooldown_seconds):
                    command = self.last_command
                else:
                    self.last_command = None
                    self.last_command_time = None
            elif (self.last_command != command and
                  self.last_command_time is not None and
                  now - self.last_command_time <
                  self.zoom_direction_cooldown_seconds):
                command = self.last_command
            elif command != "NONE":
                self.last_command = command
                self.last_command_time = now
            output = (
                f"\rIndex: ({index[0]:3d},{index[1]:3d}) | distances: "
                f"{thumb_index:6.1f}px, "
                f"{index_middle:6.1f}px, "
                f"{thumb_middle:6.1f}px | "
                f"Hand: {hand_distance:10s} | Command: {command:13s}"
            )

            print(f"{output:<100}", end="", flush=True)


def main(argv=None):
    import app

    app_args = app.get_args(argv)
    gesture_history = GestureRecognition()
    app.main(gesture_history.update, args=app_args)


if __name__ == '__main__':
    main()
