"""Step 27b: test whether the checkpoint used in Steps 27-31 was the author's relative-depth
file loaded by mistake into the metric model (same key names/shapes as the real metric Hypersim
checkpoint, so it loads without error). Runs BOTH checkpoints on the same 100 frames of
runs/20261019T091439-6ea9d598/input.avi (the original Step 27 recording: joint pose, old map
data/logitech-landmark-map.npz), replayed through the real pipeline so landmark anchors are
available. For each checkpoint, per frame with a valid tag: raw stats, the tag-plane ratio
Z_true/Z_pred (reusing occlusion.tag_plane_scale), and landmark anchors (z_true, raw) pairs
exactly as DepthPrototype.render collects them. Then one pooled regression of raw against true
distance for the tag-plane samples (per-frame median) and for the landmark anchors (every
anchor, every frame). Run via .venv-depth. Writes runs/step27b-checkpoint.json/.md.
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
from ar_projection.geometry import depths, tag_points
from ar_projection.occlusion import MetricDepth, panel_layer, tag_plane_scale
from ar_projection.pipeline import Pipeline

RECORDING = Path("runs/20261019T091439-6ea9d598")
REPO = "/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2"
OLD_CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_vits.pth"
NEW_CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
NUM_FRAMES = 100
MIN_ANCHOR_INLIERS = 12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag_visible(row_tags, anchor_id):
    return any(t.get("valid") and not t.get("ambiguous") and t["id"] == anchor_id for t in row_tags)


def run_checkpoint(model_path, input_size=252):
    # Base off the current local config-logitech.json (same calibration/tag setup) rather than
    # the recording's own config.json, whose paths are absolute Windows paths from the original
    # machine. Point mapping at the OLD map this recording was actually tracked against (the
    # new occlusion map, data/logitech-landmark-map-occ.npz, has different landmark IDs and
    # would not match this clip's features).
    cfg = load_config("config-logitech.json")
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(RECORDING/"input.avi"),
                                       timestamps_path=str(RECORDING/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path="data/logitech-landmark-map.npz", min_ready_landmarks=1),
                  display=replace(cfg.display, enabled=False))
    capture = Capture(cfg.camera)
    packet = capture.read()
    pipeline = Pipeline(cfg, packet.camera, background_mapping=False)
    model = MetricDepth(REPO, model_path, input_size=input_size)

    raw_mins, raw_meds, raw_maxs, saturated_fracs = [], [], [], []
    tag_plane_ratios = []
    tag_plane_points = []  # (true_Z_median_this_frame, raw_median_this_frame)
    anchor_points = []  # (z_true, raw) across all frames, exactly as DepthPrototype.render collects

    count = 0
    while packet is not None and count < NUM_FRAMES:
        result = pipeline.process(packet, withhold_tag=False)
        raw, _ = model.predict(packet.bgr)
        raw = np.asarray(raw, np.float64)
        raw_mins.append(float(raw.min())); raw_meds.append(float(np.median(raw))); raw_maxs.append(float(raw.max()))
        saturated_fracs.append(float(np.mean(np.abs(raw-20) <= 1)))

        if result.raw_pose.valid and tag_visible(result.diagnostics.get("tags", []), cfg.tags.anchor_id):
            ratio = tag_plane_scale(result.raw_pose.T_CW, raw, cfg.tags.size_m, packet.camera)
            if ratio is not None:
                tag_plane_ratios.append(ratio)
            # Pool one (median true Z, median raw) point per frame over the tag's own plane.
            _, valid_mask, z = panel_layer(result.raw_pose.T_CW, tag_points(cfg.tags.size_m), packet.camera)
            if np.any(valid_mask):
                tag_plane_points.append((float(np.median(z[valid_mask])), float(np.median(raw[valid_mask]))))

        if result.raw_pose.valid and result.raw_pose.landmark_inliers >= MIN_ANCHOR_INLIERS and pipeline.map is not None:
            lookup = {int(id): i for i, id in enumerate(pipeline.map.ids)}
            for f in pipeline.last_features:
                if f.landmark_id in lookup:
                    x, y = np.rint(f.pixel).astype(int)
                    if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0] and np.isfinite(raw[y, x]) and raw[y, x] > 0:
                        z_true = depths(result.raw_pose.T_CW, pipeline.map.points[lookup[f.landmark_id]][None])[0]
                        if z_true > 0:
                            anchor_points.append((float(z_true), float(raw[y, x])))
        count += 1
        packet = capture.read()
    capture.close()

    def pooled_regression(points):
        if len(points) < 5:
            return None
        z = np.array([p[0] for p in points]); r = np.array([p[1] for p in points])
        b, a = np.polyfit(z, r, 1)
        ss_res = float(np.sum((r-(a+b*z))**2))
        ss_tot = float(np.sum((r-r.mean())**2))
        return {"n": len(points), "a": float(a), "b": float(b), "r2": 1-ss_res/ss_tot if ss_tot else None,
                "sign": "positive (metric-like)" if b > 0 else "negative (disparity-like)"}

    ratios = np.array(tag_plane_ratios)
    return {
        "checkpoint_path": model_path, "checkpoint_name": Path(model_path).name, "checkpoint_sha256": sha256(model_path),
        "frames": count,
        "raw_whole_frame": {"min": float(np.min(raw_mins)), "median": float(np.median(raw_meds)), "max": float(np.max(raw_maxs))},
        "saturated_near_20_fraction_mean": float(np.mean(saturated_fracs)),
        "tag_plane_ratio_Z_true_over_Z_pred": {
            "n_frames": len(tag_plane_ratios),
            "median": float(np.median(ratios)) if len(ratios) else None,
            "relative_mad": float(np.median(np.abs(ratios-np.median(ratios)))/np.median(ratios)) if len(ratios) else None,
        },
        "tag_plane_regression_raw_vs_true_Z": pooled_regression(tag_plane_points),
        "landmark_anchor_regression_raw_vs_true_Z": pooled_regression(anchor_points),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", default=OLD_CHECKPOINT)
    parser.add_argument("--new-checkpoint", default=NEW_CHECKPOINT)
    args = parser.parse_args()

    old_hash = sha256(args.old_checkpoint)
    if old_hash != "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378":
        print(f"WARNING: old checkpoint SHA-256 {old_hash} does not match Step 27's recorded value")
    new_hash = sha256(args.new_checkpoint)
    if new_hash == old_hash:
        print("STOP: new checkpoint has the same SHA-256 as the old one -- it was not actually replaced. Asking Sahaj.")
        sys.exit(1)

    report = {
        "recording": str(RECORDING), "frames_tested": NUM_FRAMES, "input_size": 252,
        "old": run_checkpoint(args.old_checkpoint),
        "new": run_checkpoint(args.new_checkpoint),
    }
    Path("runs/step27b-checkpoint.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
