"""Step 48: relative-model depth (vits, input 252) for every 3rd frame of run 092056 that has a logged fit, undistorted.
Writes runs/step48-092056/depth.npz and img/*.jpg. Run via .venv-depth (GPU if available)."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.occlusion import RelativeDepth
from step31_masks import REPO
from step43_relative import REL_CKPT

RUN = Path("runs/20261025T092056-7a6e0212")
OUT = Path("runs/step48-092056")
meta = json.loads((RUN/"metadata.json").read_text())["calibration"]
K, dist = np.array(meta["K"]), np.array(meta["distortion"])
mx, my = cv2.initUndistortRectifyMap(K, dist, None, K, (640, 480), cv2.CV_32FC1)
rows = [json.loads(l) for l in (RUN/"frames.jsonl").read_text().splitlines()]
model = RelativeDepth(REPO, REL_CKPT, 252)
(OUT/"img").mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(RUN/"input.avi"))
out, t0, i = {}, time.time(), -1
while True:
    ok, f = cap.read()
    if not ok:
        break
    i += 1
    if i % 3 or "fit" not in rows[i]["diagnostics"].get("occlusion", {}) or rows[i]["render_T_CW"] is None:
        continue
    bgr = cv2.remap(f, mx, my, cv2.INTER_LINEAR)
    d, _ = model.predict(bgr)
    out[i] = np.asarray(d).astype(np.float16)
    cv2.imwrite(str(OUT/"img"/f"{i:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if len(out) % 100 == 0:
        print(len(out), i, time.time()-t0, flush=True)
np.savez(OUT/"depth.npz", **{str(k): v for k, v in out.items()})
print("done", len(out), time.time()-t0)
