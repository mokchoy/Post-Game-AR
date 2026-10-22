"""Step 22: what the camera sees in the dead stretch (3908-3918 s) and its neighbours. Offline, no tracker change.

Samples a frame every 0.5 s in 3880-3890 (productive), 3904-3922 (dead stretch and edges) and 3925-3935 (productive).
Per sample (tag-reference pose, ORB keypoints, projected landmarks): tag coverage, low-texture cell share (variance of
Laplacian per 4x6 cell below LOW_TEXTURE), blur (whole-frame Laplacian variance), motion (median LK displacement against
the frame 3 earlier), brightness, projected landmarks off-image / on low-texture cells, keypoint count. Contact sheets
(PNG, landmarks green, ORB keypoints red) go to runs/step22-frames/. Writes runs/step22-frames.json.
"""
import json
from dataclasses import replace
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.features import FeatureTracker, feature_mask
from ar_projection.geometry import project
from ar_projection.landmarks import load_map, map_metadata
from ar_projection.pose import PoseSolver
from ar_projection.tags import TagDetector

RANGES = {"productive_a_3880_3890": (3880, 3890), "dead_3904_3922": (3904, 3922), "productive_b_3925_3935": (3925, 3935)}
LOW_TEXTURE = 20.0  # variance of Laplacian per cell; chosen from the data below, not tuned to a result
ROWS, COLS = 4, 6


def cell_texture(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape
    return np.array([[lap[r*h//ROWS:(r+1)*h//ROWS, c*w//COLS:(c+1)*w//COLS].var() for c in range(COLS)] for r in range(ROWS)])


def motion_px(prev, gray):
    pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 8)
    if pts is None:
        return None
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
    ok = ok.ravel() == 1
    return float(np.median(np.linalg.norm((nxt-pts).reshape(-1, 2)[ok], axis=1))) if ok.any() else None


def main():
    cfg = load_config("config-exp-handoff.json")
    capture, tracker, detector = Capture(cfg.camera), FeatureTracker(cfg.features), TagDetector(cfg.tags)
    solver = PoseSolver(replace(cfg, pose=replace(cfg.pose, mode="tag_only"), uncertainty=replace(cfg.uncertainty, enabled=False)))
    packet = capture.read()
    lm = load_map(cfg.mapping.map_path, map_metadata(cfg, packet.camera), cfg)
    out_dir = Path("runs/step22-frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    history, next_t, samples, sheets = [], {}, [], {k: [] for k in RANGES}
    while packet is not None:
        t = packet.timestamp
        for name, (a, b) in RANGES.items():
            if a <= t <= b and t >= next_t.get(name, a):
                next_t[name] = t+0.5
                gray, camera = packet.gray, packet.camera
                tags = detector.detect(gray, camera)
                ref = solver.estimate(tags, [], [], camera)
                h, w = gray.shape
                tag = next((x for x in tags if x.id == cfg.tags.anchor_id), None)
                coverage = float(cv2.contourArea(np.asarray(tag.pixels, np.float32))/(w*h)) if tag is not None else 0.0
                keys = tracker.orb.detect(gray, feature_mask(gray.shape, tags, cfg.features))
                texture = cell_texture(gray)
                row = {"range": name, "t": t, "tag_seen": tag is not None, "tag_reference_valid": bool(ref.valid), "tag_image_fraction": coverage,
                       "low_texture_cell_share": float((texture < LOW_TEXTURE).mean()), "median_cell_laplacian_var": float(np.median(texture)),
                       "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                       "motion_px_per_3_frames": motion_px(history[-3], gray) if len(history) >= 3 else None,
                       "brightness": float(gray.mean()), "orb_keypoints": len(keys)}
                canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for k in keys:
                    cv2.circle(canvas, (int(k.pt[0]), int(k.pt[1])), 2, (0, 0, 255), -1)
                if ref.valid:
                    proj = project(ref.T_CW, lm.points, camera)
                    inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
                    row["landmarks_off_image"] = int((~inside).sum())
                    on_low = sum(texture[min(ROWS-1, int(p[1]*ROWS/h)), min(COLS-1, int(p[0]*COLS/w))] < LOW_TEXTURE for p in proj[inside])
                    row["landmarks_visible"] = int(inside.sum())
                    row["landmarks_on_low_texture_cells"] = int(on_low)
                    for p in proj[inside]:
                        cv2.circle(canvas, (int(p[0]), int(p[1])), 5, (0, 255, 0), 1)
                cv2.putText(canvas, f"{t:.1f} tag={'Y' if tag is not None else 'N'} kp={len(keys)}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                sheets[name].append(canvas)
                samples.append(row)
        history.append(packet.gray)
        history = history[-4:]
        packet = capture.read()
    capture.close()
    for name, images in sheets.items():
        for part in range(0, len(images), 12):
            chunk = [cv2.resize(i, (320, 240)) for i in images[part:part+12]]
            chunk += [np.zeros_like(chunk[0])]*(-len(chunk) % 4)
            sheet = np.vstack([np.hstack(chunk[r:r+4]) for r in range(0, len(chunk), 4)])
            cv2.imwrite(str(out_dir/f"{name}_{part//12}.png"), sheet)
    bins = {}
    for s in samples:
        bins.setdefault((s["range"], int(s["t"]//2*2)), []).append(s)
    table = {}
    for (name, start), rows in sorted(bins.items()):
        mean = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None
        table[f"{start}-{start+2}"] = {"range": name, "samples": len(rows), "tag_seen_share": float(np.mean([r["tag_seen"] for r in rows])),
                                       **{k: mean(k) for k in ("tag_image_fraction", "low_texture_cell_share", "median_cell_laplacian_var", "blur_laplacian_var",
                                                              "motion_px_per_3_frames", "brightness", "orb_keypoints", "landmarks_visible", "landmarks_off_image",
                                                              "landmarks_on_low_texture_cells")}}
    json.dump({"low_texture_threshold": LOW_TEXTURE, "samples": samples, "bins_2s": table}, open("runs/step22-frames.json", "w"), indent=1)
    for k, v in table.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
