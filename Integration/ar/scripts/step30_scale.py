"""Step 30 (first pass): tag-plane scale calibration on V0/V1/V2, via .venv-depth.
For each clip, every frame with the tag visible: run MetricDepth (input 252) on the rectified
frame, compute occlusion.tag_plane_scale(). Reports stability (relative MAD) for V0/V1, and for
V2 a zero-order-hold drift check across each real tag-hidden interval (last tag-derived scale
before the gap vs the fresh one right after -- the simplest possible hold, before adding the
landmark-anchor update logic Step 30 describes). No mask yet. Writes runs/step30-scale.json.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.experiments import read_rows
from ar_projection.occlusion import MetricDepth, tag_plane_scale

REPO = "/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2"
CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
CLIPS = {"V0": "runs/20261024T051838-1eaba384", "V1": "runs/20261024T052053-d73edfab", "V2": "runs/20261024T052253-e8a2ad05"}


def tag_visible(row, anchor_id):
    return any(t.get("valid") and not t.get("ambiguous") and t["id"] == anchor_id for t in row["diagnostics"].get("tags", []))


def rectified_frames(directory):
    cfg = load_config("config-logitech.json")
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(Path(directory)/"input.avi"),
                                       timestamps_path=str(Path(directory)/"frames.jsonl")))
    capture = Capture(cfg.camera)
    frames = []
    while True:
        packet = capture.read()
        if packet is None:
            break
        frames.append(packet.bgr)
    capture.close()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()
    model = MetricDepth(REPO, args.checkpoint, input_size=252)
    checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    out = {"checkpoint_path": args.checkpoint, "checkpoint_name": Path(args.checkpoint).name, "checkpoint_sha256": checkpoint_sha256, "clips": {}}
    for label, directory in CLIPS.items():
        cfg = load_config(Path(directory)/"config.json")
        rows = read_rows(directory)
        frames = rectified_frames(directory)
        camera = None
        series = []
        for row, bgr in zip(rows, frames):
            if camera is None:
                from ar_projection.camera import Preprocessor
                camera = Preprocessor(cfg.camera, (bgr.shape[1], bgr.shape[0])).camera
            if row["source_time_s"] is None or not row["raw_pose"]["valid"] or not tag_visible(row, cfg.tags.anchor_id):
                series.append(None)
                continue
            raw, _ = model.predict(bgr)
            s = tag_plane_scale(np.array(row["raw_pose"]["T_CW"]), raw, cfg.tags.size_m, camera)
            series.append({"t": row["source_time_s"], "scale": s} if s is not None else None)
        valid = [e["scale"] for e in series if e is not None]
        result = {"frames": len(rows), "tag_visible_frames": len(valid),
                   "scale_median": float(np.median(valid)) if valid else None,
                   "scale_relative_mad": float(np.median(np.abs(np.asarray(valid)-np.median(valid)))/np.median(valid)) if valid else None}
        if label == "V2":
            # Zero-order hold drift: last tag-derived scale before each real gap vs the fresh one after.
            times = [e["t"] for e in series if e is not None]
            scales = [e["scale"] for e in series if e is not None]
            drifts = []
            for i in range(1, len(times)):
                if times[i]-times[i-1] > 1.0:  # a real tag-hidden interval occurred between these two tag-visible samples
                    before, after = scales[i-1], scales[i]
                    drifts.append({"gap_s": times[i]-times[i-1], "scale_before": before, "scale_after": after,
                                   "drift_percent": 100*abs(after-before)/before})
            result["zero_order_hold_drift"] = drifts
        out["clips"][label] = result
    Path("runs/step30-scale.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
