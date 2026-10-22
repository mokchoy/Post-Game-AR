"""Step 24: tail control for geometry-first guided matching, one change per run on the Step 23 windows.
Base = geometry-first, radius 12, Hamming 128 (Step 23 setting c). Writes runs/step24-tail.json."""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.config import load_config
from ar_projection.pipeline import run
from step23_clipwide import summarize_run

SETTINGS = [("margin1.5", {"guided": {"geometry_margin": 1.5}}), ("margin2.0", {"guided": {"geometry_margin": 2.0}}),
            ("margin3.0", {"guided": {"geometry_margin": 3.0}}),
            ("min_inliers20", {"pose": {"min_landmark_inliers": 20}}), ("landmark_rms1.5", {"pose": {"max_landmark_rms_px": 1.5}}),
            ("ransac_2px", {"pose": {"ransac_reprojection_threshold_px": 2.0}}),
            ("speed_1ms_90dps", {"pose": {"max_linear_speed_m_s": 1.0, "max_angular_speed_deg_s": 90.0}})]


COMBINED = [("margin1.5+speed_1ms_90dps", {"guided": {"geometry_margin": 1.5}, "pose": {"max_linear_speed_m_s": 1.0, "max_angular_speed_deg_s": 90.0}}),
            ("margin1.5+min_inliers20", {"guided": {"geometry_margin": 1.5}, "pose": {"min_landmark_inliers": 20}})]


def main(settings=SETTINGS, output="runs/step24-tail.json"):
    cfg = load_config("config-exp-handoff.json")
    windows = json.load(open("runs/step23-windows.json"))["windows"]
    out = {}
    for label, change in settings:
        guided = replace(cfg.features.guided_matching, enabled=True, max_hamming_distance=128, geometry_first=True, **change.get("guided", {}))
        variant = replace(cfg, pose=replace(cfg.pose, mode="joint", **change.get("pose", {})), features=replace(cfg.features, guided_matching=guided))
        path = run(variant, withhold_tag_windows=[(w["start"], w["end"]) for w in windows])
        out[label] = {"run": str(path), "change": change, **summarize_run(path, windows)}
        json.dump(out, open(output, "w"), indent=1)


if __name__ == "__main__":
    main(COMBINED, "runs/step24-combined.json") if "--combined" in sys.argv else main()
