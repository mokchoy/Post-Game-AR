"""Step 29: pose quality on the recorded occlusion clips (V0-V4), against the new map
`data/logitech-landmark-map-occ.npz`, using the clips exactly as recorded (joint mode,
guided matching on, config-logitech.json defaults -- no --withhold-tag: real occlusion only).
For each clip: overall summarize(), plus the real tag-hidden intervals (from the per-frame
tag-detection diagnostic, not the fusion source label) and whether landmark/joint pose was
supported inside each. Writes runs/step29-clips.json.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.config import load_config
from ar_projection.experiments import read_rows, summarize
from ar_projection.pipeline import run

CLIPS = [
    ("V0", "runs/20261024T051838-1eaba384"),
    ("V1", "runs/20261024T052053-d73edfab"),
    ("V2", "runs/20261024T052253-e8a2ad05"),
    ("V3", "runs/20261024T054241-016203dd"),
    ("V4", "runs/20261024T052603-9e9fecca"),
]


def tag_visible(row, anchor_id):
    return any(t.get("valid") and not t.get("ambiguous") and t["id"] == anchor_id for t in row["diagnostics"].get("tags", []))


def supported(row):
    pose = row["raw_pose"]
    return bool(pose["valid"] and row["actual_pose_source"] in ("tag", "joint", "landmark")
                and pose.get("tag_inliers", 0)+pose.get("landmark_inliers", 0) >= 4)


def hidden_intervals(directory, merge_gap_s=0.5, min_len_s=0.3):
    """Gaps between merged tag-visible stretches: detector flicker (<=merge_gap_s) is absorbed
    into whichever side it sits on, so a real 2-5 s occlusion is not chopped into blips."""
    cfg = load_config(Path(directory)/"config.json")
    rows = [r for r in read_rows(directory) if r["source_time_s"] is not None]
    visible_times = [r["source_time_s"] for r in rows if tag_visible(r, cfg.tags.anchor_id)]
    stretches = []
    for t in visible_times:
        if stretches and t-stretches[-1][1] <= merge_gap_s:
            stretches[-1][1] = t
        else:
            stretches.append([t, t])
    gaps = [(a[1], b[0]) for a, b in zip(stretches, stretches[1:]) if b[0]-a[1] >= min_len_s]
    out = []
    for start, end in gaps:
        frames = [r for r in rows if start <= r["source_time_s"] <= end]
        out.append({"start_s": start, "end_s": end, "duration_s": end-start, "frames": len(frames),
                     "landmark_pose_supported_fraction": sum(supported(r) for r in frames)/len(frames) if frames else None})
    return out


def landmark_only_withheld(directory, map_path):
    """Second pass: landmark_only, --withhold-tag, whole clip -- the tag stays detected as an
    independent reference (not fed to the solver), giving panel_reference_* against it."""
    cfg = load_config("config-logitech.json")
    rec = Path(directory).resolve()
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(map_path).resolve())),
                  pose=replace(cfg.pose, mode="landmark_only"),
                  display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))
    path = run(cfg, withhold_tag=True)
    return summarize(path)


def guided_variant(directory, map_path, guided_enabled):
    """landmark_only, --withhold-tag, whole clip, with guided matching forced on/off
    (everything else as config-logitech.json: Step 24 geometry-first/radius12/hamming128 when on)."""
    cfg = load_config("config-logitech.json")
    rec = Path(directory).resolve()
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(map_path).resolve())),
                  pose=replace(cfg.pose, mode="landmark_only"),
                  features=replace(cfg.features, guided_matching=replace(cfg.features.guided_matching, enabled=guided_enabled)),
                  display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))
    return summarize(run(cfg, withhold_tag=True))


def main():
    out = {}
    for label, directory in CLIPS:
        summary = summarize(directory)
        intervals = hidden_intervals(directory)
        withheld = landmark_only_withheld(directory, "data/logitech-landmark-map-occ.npz")
        out[label] = {"run": directory, "summary": {k: summary[k] for k in (
            "frames", "duration_s", "valid_pose_fraction", "no_tag_frames", "valid_without_tag_fraction",
            "no_tag_duration_s", "longest_no_tag_interval_s")},
            "tag_hidden_intervals": intervals,
            "landmark_only_withheld_tag": {"run": withheld["run"], "valid_without_tag_fraction": withheld["valid_without_tag_fraction"],
                "panel_reference_median_px": withheld["panel_reference_median_px"], "panel_reference_p95_px": withheld["panel_reference_p95_px"],
                "reference_corner_samples": withheld["reference_corner_samples"]}}
    Path("runs/step29-clips.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
