"""Step 31 (offline): visibility mask + composite export for the occlusion clips, via .venv-depth.
Calls ar_projection.pipeline.run() directly with occlusion=DepthPrototype -- this bypasses the
tracking-acceptance gate in cli.py, which is a live-only (Step 35) requirement; Steps 30-34 are
explicitly offline and exempt, per Step 35's own gate note in Instruction.md. No tracking-acceptance
report exists for the new map and none is checked here.

Primary pass: all 5 clips (V0-V4) at the default margin (0.02 m), input size 252.
Secondary (V1 only, the clean main depth-eval clip): margin sweep 0.01/0.05 m, and the
invalid-depth-occludes alternative, for comparison against the primary pass.
Writes runs/step31-masks/<label>/ per variant and a summary in runs/step31-masks/summary.json.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ar_projection.config import load_config
from ar_projection.occlusion import DepthPrototype, MetricDepth
from ar_projection.pipeline import run

REPO = "/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2"
CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
MAP_PATH = "data/logitech-landmark-map-occ.npz"
CLIPS = {"V0": "runs/20261024T051838-1eaba384", "V1": "runs/20261024T052053-d73edfab",
         "V2": "runs/20261024T052253-e8a2ad05", "V3": "runs/20261024T054241-016203dd",
         "V4": "runs/20261024T052603-9e9fecca"}
OUT_ROOT = Path("runs/step31-masks")  # overridden by --output-root


def clip_config(directory):
    rec = Path(directory).resolve()
    cfg = load_config("config-logitech.json")
    return replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(MAP_PATH).resolve())),
                  display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))


def export_variant(model, label, directory, margin_m=.02, invalid_occludes=False, slope_sign=1, out_root=None):
    out_root = out_root or OUT_ROOT
    occlusion = DepthPrototype(model, out_root/label, margin_m=margin_m, invalid_occludes=invalid_occludes, slope_sign=slope_sign)
    tracking_run = run(clip_config(directory), occlusion=occlusion)
    stats_files = sorted((out_root/label).glob("*.json"))
    statuses, invalid_fractions, inference_ms, unknown_depth_pixels = [], [], [], []
    for f in stats_files:
        s = json.loads(f.read_text())
        statuses.append(s["status"])
        if s.get("valid_depth_fraction") is not None:
            invalid_fractions.append(1-s["valid_depth_fraction"])
        if s.get("inference_ms") is not None:
            inference_ms.append(s["inference_ms"])
    active = sum("active" in s for s in statuses)
    return {"tracking_run": str(tracking_run), "frames_exported": len(stats_files), "margin_m": margin_m, "invalid_occludes": invalid_occludes,
            "slope_sign": slope_sign,
            "mask_active_fraction": active/len(statuses) if statuses else None,
            "mean_invalid_depth_fraction": float(np.mean(invalid_fractions)) if invalid_fractions else None,
            "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--slope-sign", type=int, default=1, choices=(-1, 1))  # +1 matches the metric checkpoint default above (Step 27b)
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model = MetricDepth(REPO, args.checkpoint, input_size=252)
    checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    summary = {"checkpoint_path": args.checkpoint, "checkpoint_name": Path(args.checkpoint).name,
               "checkpoint_sha256": checkpoint_sha256, "slope_sign": args.slope_sign, "variants": {}}
    for label, directory in CLIPS.items():
        summary["variants"][label] = export_variant(model, label, directory, slope_sign=args.slope_sign, out_root=out_root)
    for margin in (.01, .05):
        label = f"V1-margin{margin}"
        summary["variants"][label] = export_variant(model, label, CLIPS["V1"], margin_m=margin, slope_sign=args.slope_sign, out_root=out_root)
    summary["variants"]["V1-invalid-occludes"] = export_variant(model, "V1-invalid-occludes", CLIPS["V1"], invalid_occludes=True, slope_sign=args.slope_sign, out_root=out_root)
    Path(out_root/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
