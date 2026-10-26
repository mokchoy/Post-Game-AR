"""Step 27: confirm Depth Anything V2 Small (Hypersim metric) runs on the GPU from .venv-depth,
and record its speed and output units. Run with .venv-depth's python. Writes runs/step27-depth-env.json.
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.diagnostics import write_json
from ar_projection.experiments import percentile
from ar_projection.geometry import depths, project
from ar_projection.occlusion import MetricDepth

RECORDING = Path("runs/20261019T091439-6ea9d598")
REPO = Path("/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2")
CHECKPOINT = REPO/"metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
NUM_FRAMES = 100
SIZES = (252, 392, 518)
WARMUP = 5


def rectified_frames(count):
    cfg = load_config("config-logitech.json")
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(RECORDING/"input.avi"),
                                       timestamps_path=str(RECORDING/"frames.jsonl")))
    capture = Capture(cfg.camera)
    frames, camera = [], None
    for _ in range(count):
        packet = capture.read()
        if packet is None:
            break
        frames.append(packet.bgr)
        camera = packet.camera
    capture.close()
    return frames, camera


def tag_pose(frame_index):
    rows = (json.loads(l) for l in open(RECORDING/"frames.jsonl"))
    row = next(r for r in rows if r["frame_index"] == frame_index)
    pose = row["raw_pose"]
    if not (pose["valid"] and pose["source"] == "tag"):
        raise ValueError(f"frame {frame_index} has no valid tag pose")
    return np.array(pose["T_CW"])


def time_size(model, frames):
    import torch
    times_sync, times_raw = [], []
    for i, bgr in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.model.infer_image(bgr, model.input_size)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= WARMUP:
            times_raw.append((t1-t0)*1000)
            times_sync.append((t2-t0)*1000)

    def stats(values):
        return {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)), "p95_ms": percentile(values, 95)}
    return stats(times_raw), stats(times_sync)


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    frames, camera = rectified_frames(NUM_FRAMES)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "repo_path": str(REPO), "checkpoint_path": str(checkpoint), "checkpoint_name": checkpoint.name, "checkpoint_sha256": checkpoint_sha256,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "recording": str(RECORDING), "frames_timed": len(frames)-WARMUP, "warmup_frames": WARMUP,
        "sizes": {},
    }
    for size in SIZES:
        torch.cuda.reset_peak_memory_stats()
        model = MetricDepth(str(REPO), str(checkpoint), input_size=size)
        no_sync, with_sync = time_size(model, frames)
        report["sizes"][size] = {
            "inference_ms_excluding_sync": no_sync, "inference_ms_including_sync": with_sync,
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated()/2**20,
        }
        del model
        torch.cuda.empty_cache()
    # Output units/range check: predict on frame 0 (tag visible), sample the model's raw
    # output at the tag's own projected pixel and compare to the tag's true camera-Z.
    model = MetricDepth(str(REPO), str(checkpoint), input_size=252)
    raw, _ = model.predict(frames[0])
    T_CW = tag_pose(0)
    true_z = float(depths(T_CW, np.zeros((1, 3)))[0])
    px, py = project(T_CW, np.zeros((1, 3)), camera)[0]
    x, y = int(round(px)), int(round(py))
    sampled = float(raw[y, x]) if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0] else None
    report["units_check"] = {
        "frame_index": 0, "tag_pixel": [x, y], "tag_true_camera_z_m": true_z,
        "model_output_at_tag_pixel": sampled, "implied_scale_true_over_model": true_z/sampled if sampled else None,
        "model_output_min": float(np.min(raw)), "model_output_max": float(np.max(raw)), "model_output_median": float(np.median(raw)),
        "note": "model output is unscaled metric depth from the Hypersim-trained head; "
                "implied_scale_true_over_model is the single-frame scale factor a real calibration (Step 30) would need to apply, not a correction applied here",
    }
    write_json("runs/step27-depth-env.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
