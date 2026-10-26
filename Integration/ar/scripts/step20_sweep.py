"""Step 20: revalidate_existing on the Step 13 protocol, with the Step 17 default gate and with the Step 19 loose gate or geometry-first. Writes runs/step20-sweep.json."""
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.config import load_config
from ar_projection.experiments import read_rows, window_metrics
from ar_projection.pipeline import run
from step16_paired import frame_errors, in_windows, pct

SETTINGS = [("revalidate_default", {"revalidate_existing": True}),
            ("revalidate_hamming128", {"revalidate_existing": True, "max_hamming_distance": 128}),
            ("revalidate_geometry_first", {"revalidate_existing": True, "max_hamming_distance": 128, "geometry_first": True})]


def main():
    cfg = load_config("config-exp-handoff.json")
    windows = [tuple(w) for w in json.load(open("runs/step13-handoff.json"))["windows"]]
    out = {}
    for label, change in SETTINGS:
        guided = replace(cfg.features.guided_matching, enabled=True, **change)
        path = run(replace(cfg, pose=replace(cfg.pose, mode="joint"), features=replace(cfg.features, guided_matching=guided)), withhold_tag_windows=windows)
        total = sum(in_windows(r["source_time_s"], windows) for r in read_rows(path))
        errors = list(frame_errors(path, windows).values())
        per_window = window_metrics(path, windows)
        out[label] = {"run": str(path), "guided": {k: getattr(guided, k) for k in ("search_radius_px", "max_hamming_distance", "ratio_test", "geometry_first", "revalidate_existing")},
                      "window_frames": total, "supported_frames": len(errors), "supported_fraction": len(errors)/total,
                      "median_px": pct(errors, 50), "p95_px": pct(errors, 95), "over_10px": int(np.sum(np.array(errors) > 10)),
                      "per_window_valid_fraction": {k: v["valid_fraction"] for k, v in per_window.items()},
                      "recovery_after_window_s": {k: v["recovery_after_window_s"] for k, v in per_window.items()}}
        out[label]["ids_cleared_total"] = sum((r["diagnostics"].get("map_associations") or {}).get("ids_cleared_by_guided_revalidation", 0) for r in read_rows(path))
        json.dump(out, open("runs/step20-sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
