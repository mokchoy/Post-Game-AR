"""Live webcam inference with confidence-gated temporal gesture confirmation."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, deque
from pathlib import Path

import webcam


WINDOW_TITLE = "Dynamic gesture recognition"
VOTE_TRACKBAR = "Window"
REQUIRED_TRACKBAR = "Required"
HOLD_SECONDS = 0.6
PREDICTION_GAP_SECONDS = 1.0
BACKGROUND_LABELS = {"no gesture", "doing other things"}


def _validate_settings(min_confidence: float, vote_window: int, required_votes: int) -> None:
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("--min-confidence must be finite and between 0 and 1")
    if type(vote_window) is not int or not 1 <= vote_window <= 25:
        raise ValueError("--vote-window must be an integer from 1 to 25")
    if type(required_votes) is not int or not 1 <= required_votes <= vote_window:
        raise ValueError("--required-votes must be an integer from 1 to --vote-window")


class StableController:
    def __init__(self, min_confidence: float = 0.70, vote_window: int = 5, required_votes: int = 4):
        _validate_settings(min_confidence, vote_window, required_votes)
        self.min_confidence = min_confidence
        self.vote_window = vote_window
        self.required_votes = required_votes
        self.votes: deque[str | None] = deque(maxlen=vote_window)
        self.confirmed_label: str | None = None
        self.confirmed_at: float | None = None
        self.holding = False
        self.last_fresh_at: float | None = None
        self.last_prediction: tuple[str, float] | None = None
        self.current_valid = False
        self._announced_status: str | None = None

    @property
    def status_text(self) -> str:
        if self.confirmed_label is not None:
            prefix = "Holding" if self.holding else "Confirmed"
            return f"{prefix}: {self.confirmed_label}"
        if not self.current_valid:
            return "No hand"
        if self.last_prediction is None:
            return "Uncertain"
        label, score = self.last_prediction
        if not math.isfinite(score) or score < self.min_confidence:
            return "Uncertain"
        if label.strip().casefold() in BACKGROUND_LABELS:
            return "Inactive"
        return "Uncertain"

    def _announce_status(self) -> None:
        status = self.status_text
        if status != self._announced_status:
            print(status)
            self._announced_status = status

    def _clear_confirmation(self) -> None:
        self.votes.clear()
        self.confirmed_label = None
        self.confirmed_at = None
        self.holding = False
        self.last_fresh_at = None

    def reset(self) -> None:
        self._clear_confirmation()
        self._announce_status()

    def set_settings(
        self,
        *,
        min_confidence: float | None = None,
        vote_window: int | None = None,
        required_votes: int | None = None,
    ) -> bool:
        confidence = self.min_confidence if min_confidence is None else min_confidence
        window = self.vote_window if vote_window is None else vote_window
        required = self.required_votes if required_votes is None else required_votes
        _validate_settings(confidence, window, required)
        if (confidence, window, required) == (self.min_confidence, self.vote_window, self.required_votes):
            return False
        self.min_confidence, self.vote_window, self.required_votes = confidence, window, required
        self.votes = deque(maxlen=window)
        self._clear_confirmation()
        self._announce_status()
        print(
            f"Settings: min-confidence={confidence:.2f}, "
            f"vote-window={window}, required-votes={required}"
        )
        return True

    @staticmethod
    def _eligible_vote(prediction: tuple[str, float] | None, threshold: float) -> str | None:
        if prediction is None:
            return None
        label, score = prediction
        if not math.isfinite(score) or score < threshold or label.strip().casefold() in BACKGROUND_LABELS:
            return None
        return label

    def update(self, session, timestamp: float, fresh_prediction: bool) -> None:
        self.current_valid = session.current_valid
        self.last_prediction = session.prediction

        if self.last_fresh_at is not None and timestamp - self.last_fresh_at > PREDICTION_GAP_SECONDS:
            self._clear_confirmation()
            if not fresh_prediction:
                session.prediction = None
                self.last_prediction = None

        if fresh_prediction:
            self.last_prediction = session.prediction
            self.last_fresh_at = timestamp
            vote = self._eligible_vote(session.prediction, self.min_confidence)
            self.votes.append(vote)
            if vote is not None and len(self.votes) == self.vote_window:
                counts = Counter(label for label in self.votes if label is not None)
                others = max((count for label, count in counts.items() if label != vote), default=0)
                if counts[vote] >= self.required_votes and counts[vote] > others:
                    self.confirmed_label = vote
                    self.confirmed_at = timestamp
                    self.holding = False
                elif self.confirmed_label is not None:
                    self.holding = True
            elif self.confirmed_label is not None:
                self.holding = True
        elif not self.current_valid:
            self.votes.clear()
            if self.confirmed_label is not None:
                self.holding = True
        elif (self.confirmed_label is not None and self.last_fresh_at is not None
                and timestamp - self.last_fresh_at > webcam.PREDICTION_INTERVAL):
            self.holding = True

        if (self.confirmed_at is not None
                and timestamp - self.confirmed_at > HOLD_SECONDS):
            self.confirmed_label = None
            self.confirmed_at = None
            self.holding = False
        self._announce_status()

    def model_switched(self, session) -> None:
        self.current_valid = session.current_valid
        self.last_prediction = session.prediction
        self.reset()

    def start(self, cv2, window_title: str) -> None:
        cv2.namedWindow(window_title)
        cv2.createTrackbar(VOTE_TRACKBAR, window_title, self.vote_window, 25, lambda _: None)
        cv2.setTrackbarMin(VOTE_TRACKBAR, window_title, 1)
        cv2.createTrackbar(REQUIRED_TRACKBAR, window_title, self.required_votes, 25, lambda _: None)
        cv2.setTrackbarMin(REQUIRED_TRACKBAR, window_title, 1)

    def poll_controls(self, cv2) -> None:
        window = max(1, min(25, int(cv2.getTrackbarPos(VOTE_TRACKBAR, WINDOW_TITLE))))
        required = max(1, min(25, int(cv2.getTrackbarPos(REQUIRED_TRACKBAR, WINDOW_TITLE))))
        if required > window:
            required = window
            cv2.setTrackbarPos(REQUIRED_TRACKBAR, WINDOW_TITLE, required)
        self.set_settings(vote_window=window, required_votes=required)

    def handle_key(self, key: int) -> None:
        if key == ord("["):
            self.set_settings(min_confidence=round(max(0.0, self.min_confidence - 0.05), 2))
        elif key == ord("]"):
            self.set_settings(min_confidence=round(min(1.0, self.min_confidence + 0.05), 2))

    def display_lines(self) -> tuple[str, ...]:
        return (
            f"State: {self.status_text}",
            f"Minimum uncalibrated softmax score: {self.min_confidence:.2f}",
            f"Vote window: {self.vote_window}; required votes: {self.required_votes}",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="completed experiment directory")
    parser.add_argument("--model", choices=("gru", "lstm", "ddnet"), default="gru")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hand-model", type=Path, default=webcam.DEFAULT_HAND_MODEL)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--required-votes", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error("--window-seconds must be finite and greater than zero")
    try:
        _validate_settings(args.min_confidence, args.vote_window, args.required_votes)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        loaded = webcam.load_model(args.experiment, args.model)
        webcam.run_camera(
            args,
            loaded,
            controller=StableController(args.min_confidence, args.vote_window, args.required_votes),
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
