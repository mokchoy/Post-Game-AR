"""Step 27b task 6: fit diagnostics on V1/V2 with the new checkpoint, 200-frame sample each
(every frame with >=12 landmark anchors, first 200 such frames). Per clip: anchor count, Huber
affine fit (a,b), relative MAD (same gate as the live code), mean absolute in-sample anchor
error, leave-one-out anchor error, and a scale-only (a fixed at 0) alternative for comparison.
Run via .venv-depth. Writes runs/step27b-fit-diagnostics.json.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.geometry import depths
from ar_projection.occlusion import MetricDepth, fit_affine_huber
from ar_projection.pipeline import Pipeline

REPO = "/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2"
NEW_CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
MAP_PATH = "data/logitech-landmark-map-occ.npz"
CLIPS = {"V1": "runs/20261024T052053-d73edfab", "V2": "runs/20261024T052253-e8a2ad05"}
SAMPLE_SIZE = 200
MIN_ANCHORS = 12


def leave_one_out_error(z, raw):
    """For each anchor, refit on the rest and measure the error predicting the held-out one."""
    errors = []
    for i in range(len(z)):
        mask = np.arange(len(z)) != i
        if mask.sum() < 3:
            continue
        a, b = fit_affine_huber(z[mask], raw[mask])
        if b == 0:
            continue
        pred = (raw[i]-a)/b
        errors.append(abs(pred-z[i]))
    return float(np.mean(errors)) if errors else None


def diagnose(model, directory):
    rec = Path(directory).resolve()
    cfg = load_config("config-logitech.json")
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(MAP_PATH).resolve())),
                  display=replace(cfg.display, enabled=False))
    capture = Capture(cfg.camera)
    packet = capture.read()
    pipeline = Pipeline(cfg, packet.camera, background_mapping=False)

    per_frame = []
    while packet is not None and len(per_frame) < SAMPLE_SIZE:
        result = pipeline.process(packet, withhold_tag=False)
        if result.raw_pose.valid and result.raw_pose.landmark_inliers >= MIN_ANCHORS and pipeline.map is not None:
            raw, _ = model.predict(packet.bgr)
            lookup = {int(id): i for i, id in enumerate(pipeline.map.ids)}
            z_list, r_list = [], []
            for f in pipeline.last_features:
                if f.landmark_id in lookup:
                    x, y = np.rint(f.pixel).astype(int)
                    if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0] and np.isfinite(raw[y, x]) and raw[y, x] > 0:
                        z_true = depths(result.raw_pose.T_CW, pipeline.map.points[lookup[f.landmark_id]][None])[0]
                        if z_true > 0:
                            z_list.append(float(z_true)); r_list.append(float(raw[y, x]))
            if len(z_list) >= MIN_ANCHORS:
                z, r = np.array(z_list), np.array(r_list)
                a, b = fit_affine_huber(z, r)
                pred = (r-a)/b if b != 0 else np.full_like(r, np.nan)
                rel_err = np.abs(pred-z)/z
                # scale-only alternative: raw = b0*z (a fixed at 0)
                b0 = float(np.sum(z*r)/np.sum(z*z))  # least-squares slope through the origin
                pred0 = r/b0 if b0 != 0 else np.full_like(r, np.nan)
                per_frame.append({
                    "anchor_count": len(z), "a": a, "b": b,
                    "relative_mad": float(np.median(np.abs(rel_err-np.median(rel_err)))) if len(rel_err) else None,
                    "mean_abs_anchor_error_m": float(np.mean(np.abs(pred-z))),
                    "leave_one_out_error_m": leave_one_out_error(z, r),
                    "scale_only_b": b0, "scale_only_mean_abs_error_m": float(np.mean(np.abs(pred0-z))),
                })
        packet = capture.read()
    capture.close()

    if not per_frame:
        return {"frames_sampled": 0}
    a_vals = np.array([f["a"] for f in per_frame])
    b_vals = np.array([f["b"] for f in per_frame])
    anchor_err = np.array([f["mean_abs_anchor_error_m"] for f in per_frame])
    loo_err = np.array([f["leave_one_out_error_m"] for f in per_frame if f["leave_one_out_error_m"] is not None])
    scale_only_err = np.array([f["scale_only_mean_abs_error_m"] for f in per_frame])
    return {
        "frames_sampled": len(per_frame),
        "anchor_count_mean": float(np.mean([f["anchor_count"] for f in per_frame])),
        "a_mean": float(a_vals.mean()), "a_std": float(a_vals.std()),
        "b_mean": float(b_vals.mean()), "b_std": float(b_vals.std()),
        "affine_mean_abs_anchor_error_m": float(anchor_err.mean()),
        "affine_leave_one_out_error_m": float(loo_err.mean()) if len(loo_err) else None,
        "scale_only_mean_abs_anchor_error_m": float(scale_only_err.mean()),
        "intercept_still_needed": bool(scale_only_err.mean() > anchor_err.mean()*1.1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=NEW_CHECKPOINT)
    args = parser.parse_args()
    model = MetricDepth(REPO, args.checkpoint, input_size=252)
    report = {"checkpoint": args.checkpoint, "sample_size": SAMPLE_SIZE,
              "old_checkpoint_step31_affine_mean_anchor_error_m": 0.065,  # quoted for comparison, see Instruction.md
              "clips": {label: diagnose(model, directory) for label, directory in CLIPS.items()}}
    Path("runs/step27b-fit-diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
