"""Step 50: gesture control from a second camera in a daemon thread. The thread owns its capture, MediaPipe hand model, DD-Net
session and StableController (reused from ../gestures, headless: no windows, no waitKey) and puts ONE event per new
confirmation on a queue; the AR loop drains the queue once per frame. Any failure disables gestures; the AR keeps running."""
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path


class EdgeTrigger:
    """Turns the controller's confirmed label into one event per confirmation: fires when a label appears (or changes) and the
    cooldown since the last event has passed. A label that stays confirmed or is still held does not fire again."""
    def __init__(self, cooldown_s):
        self.cooldown_s, self.previous, self.last_at = cooldown_s, None, -1e9

    def update(self, label, now):
        fire = label is not None and label != self.previous and now-self.last_at >= self.cooldown_s
        if fire:
            self.last_at = now
        self.previous = label                   # a confirmation blocked by the cooldown is dropped, not delayed
        return label if fire else None


def action_for(table, label):
    """gestures.actions lookup: label -> (action, value) or None."""
    entry = table.get(label)
    return (entry[0], entry[1]) if entry else None


class GestureWorker(threading.Thread):
    def __init__(self, cfg, events, log_path=None):
        super().__init__(daemon=True, name="gesture-worker")
        self.cfg, self.events, self.log_path = cfg, events, log_path
        self.stop_flag = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.rate_hz = 0.0                      # loop iterations per second, for the gesture-thread rate check
        self.preview = False                    # main thread sets it while the gesture pane is shown; then `frame` is kept current
        self.frame = None                       # latest laptop-camera frame with the webcam_stable overlay (read by the main thread)
        self.pending = None                     # (min_confidence, vote_window, required_votes) requested by the trackbars
        self.settings = (cfg.min_confidence, cfg.vote_window, cfg.required_votes)

    def request(self, min_confidence, vote_window, required_votes):
        """Main thread: ask for new vote settings; the worker applies them itself (the controller is not thread-safe)."""
        if (min_confidence, vote_window, required_votes) != self.settings:
            self.pending = (min_confidence, vote_window, required_votes)

    def stop(self):
        self.stop_flag.set()

    def _log(self, file, row):
        if file is not None:
            file.write(json.dumps(row)+"\n")
            file.flush()

    def run(self):
        capture = detector = log = None
        try:
            import cv2
            gdir = Path(self.cfg.gestures_dir).resolve()
            sys.path.append(str(gdir))          # appended: the AR package's own module names keep priority
            import webcam
            from landmarks import MediaPipeHand
            from webcam_stable import StableController
            loaded = webcam.load_job(gdir/f"{self.cfg.model}_final", self.cfg.model)
            session = webcam.LiveSession(gdir, loaded, self.cfg.window_seconds)
            controller = StableController(self.cfg.min_confidence, self.cfg.vote_window, self.cfg.required_votes)
            detector = MediaPipeHand(gdir/"models"/"hand_landmarker.task")
            capture = cv2.VideoCapture(self.cfg.camera_index)
            if not capture.isOpened():
                raise RuntimeError(f"gesture camera {self.cfg.camera_index} did not open")
            if self.log_path is not None:
                log = open(self.log_path, "w", encoding="utf-8")
            trigger, started, last_sample, last_ms, loops, rate_t = EdgeTrigger(self.cfg.cooldown_s), time.monotonic(), None, -1, 0, time.monotonic()
            self.ready.set()
            while not self.stop_flag.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("gesture camera stopped delivering frames")
                now = time.monotonic()
                if last_sample is None or now-last_sample >= webcam.SAMPLE_INTERVAL:
                    ms = max(int((now-started)*1000), last_ms+1)
                    session.add_sample(now, detector.detect(frame, ms)["xy"])
                    last_sample, last_ms = now, ms
                pending, self.pending = self.pending, None
                if pending is not None:
                    controller.set_settings(min_confidence=pending[0], vote_window=pending[1], required_votes=pending[2])
                    self.settings = pending
                    self._log(log, {"t": now, "settings": {"min_confidence": pending[0], "vote_window": pending[1], "required_votes": pending[2]}})
                fresh = session.predict_if_due(now)
                controller.update(session, now, fresh)
                if self.preview:
                    webcam._draw(frame, session, self.rate_hz, now, prediction_prefix="Raw prediction (uncalibrated softmax score)",
                                 extra_lines=controller.display_lines())
                    self.frame = frame
                if fresh and session.prediction:
                    self._log(log, {"t": now, "prediction": session.prediction[0], "score": session.prediction[1], "confirmed": controller.confirmed_label})
                label = trigger.update(controller.confirmed_label, now)
                if label is not None:
                    self.events.put((now, label))
                    self._log(log, {"t": now, "event": label})
                loops += 1
                if now-rate_t >= 2.0:
                    self.rate_hz, loops, rate_t = loops/(now-rate_t), 0, now
                    self._log(log, {"t": now, "loop_hz": self.rate_hz})
        except Exception as exc:                # camera, model or import failure: report and stop, never crash the AR loop
            self.error = f"{type(exc).__name__}: {exc}"
            logging.error("Gestures disabled: %s", self.error)
        finally:
            self.ready.set()
            for closer in (getattr(capture, "release", None), getattr(detector, "close", None), getattr(log, "close", None)):
                if closer:
                    try:
                        closer()
                    except Exception:
                        pass
