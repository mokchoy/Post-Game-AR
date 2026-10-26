"""Headless DD-Net smoke test: laptop camera -> MediaPipe -> LiveSession -> StableController; prints confirmed labels."""
import sys, time
from pathlib import Path
import cv2, webcam
from landmarks import MediaPipeHand
from webcam_stable import StableController

here = Path(__file__).resolve().parent
cam, seconds = (int(sys.argv[1]) if len(sys.argv) > 1 else 0), (float(sys.argv[2]) if len(sys.argv) > 2 else 30)
loaded = webcam.load_job(here / "ddnet_final", "ddnet")
session, ctl = webcam.LiveSession(here, loaded, 3.0), StableController()
cap, det = cv2.VideoCapture(cam), MediaPipeHand(webcam.DEFAULT_HAND_MODEL)
assert cap.isOpened(), f"camera {cam} will not open"
t0 = last = time.monotonic(); seen = None
while time.monotonic()-t0 < seconds:
    ok, frame = cap.read()
    now = time.monotonic()
    if not ok: break
    if now-last >= webcam.SAMPLE_INTERVAL:
        session.add_sample(now, det.detect(frame, int((now-t0)*1000))["xy"]); last = now
    ctl.update(session, now, session.predict_if_due(now))
    if ctl.confirmed_label != seen:
        seen = ctl.confirmed_label
        if seen: print(f"{now-t0:6.2f}s CONFIRMED {seen}", flush=True)
cap.release(); det.close()
