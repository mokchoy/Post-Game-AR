"""Record a short clip from the laptop camera for the gesture figure.
Run from Integration/: .venv/bin/python runs/fig-gesture/capture_laptop.py [camera_index] [seconds]
Do one slow swipe right after the countdown. Writes runs/fig-gesture/laptop/frame_XXXX.png and laptop.mp4."""
import sys, time
from pathlib import Path
import cv2

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
out = Path(__file__).resolve().parent/"laptop"
out.mkdir(exist_ok=True)
cap = cv2.VideoCapture(index)
if not cap.isOpened():
    sys.exit(f"camera {index} did not open")
start = time.monotonic()
while time.monotonic()-start < 3:                      # countdown with live preview
    ok, frame = cap.read()
    if ok:
        cv2.putText(frame, f"swipe right in {3-int(time.monotonic()-start)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("capture", frame)
    cv2.waitKey(1)
frames, t0 = [], time.monotonic()
while time.monotonic()-t0 < seconds:
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
    cv2.imshow("capture", frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
h, w = frames[0].shape[:2]
fps = len(frames)/seconds
writer = cv2.VideoWriter(str(out/"laptop.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i, f in enumerate(frames):
    cv2.imwrite(str(out/f"frame_{i:04d}.png"), f)
    writer.write(f)
writer.release()
print(f"saved {len(frames)} frames ({w}x{h}, {fps:.1f} fps) to {out}")
