"""Evaluate a new map offline on a recording (default: its own build recording, an optimistic upper bound).
Runs: tag_only (tag-valid stretches and windows), then joint hand-off windows with guided matching off and with the
config's guided settings, plus whole-clip landmark_only with the tag withheld. Writes runs/step27-newmap.json."""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.config import load_config
from ar_projection.experiments import summarize
from ar_projection.pipeline import run
from step23_clipwide import make_windows, stretches, summarize_run


def main(recording, map_path, output="runs/step27-newmap.json"):
    cfg = load_config("config-logitech.json")
    rec = Path(recording).resolve()
    base = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                   mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(map_path).resolve())),
                   display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))
    tag_run = run(replace(base, pose=replace(base.pose, mode="tag_only")))
    spans = stretches(tag_run)
    windows = make_windows(spans)
    out = {"recording": str(rec), "map": str(map_path), "tag_only_run": str(tag_run), "tag_only": summarize(tag_run), "stretches": spans, "windows": windows, "hand_off": {}}
    withheld = [(w["start"], w["end"]) for w in windows]
    for label, guided in (("joint_guided_off", replace(base.features.guided_matching, enabled=False)), ("joint_guided_config", base.features.guided_matching)):
        path = run(replace(base, pose=replace(base.pose, mode="joint"), features=replace(base.features, guided_matching=guided)), withhold_tag_windows=withheld)
        out["hand_off"][label] = {"run": str(path), "guided": {k: getattr(guided, k) for k in ("enabled", "geometry_first", "geometry_margin", "max_hamming_distance", "search_radius_px")},
                                  **summarize_run(path, windows)}
    path = run(replace(base, pose=replace(base.pose, mode="landmark_only")), withhold_tag=True)
    out["whole_clip_landmark_only_tag_withheld"] = {"run": str(path), **{k: v for k, v in summarize(path).items() if k in ("frames", "valid_without_tag_fraction", "panel_reference_median_px", "panel_reference_p95_px", "reference_corner_samples", "actual_pose_sources")}}
    json.dump(out, open(output, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
