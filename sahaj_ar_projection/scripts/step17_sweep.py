"""Step 17: one-parameter-at-a-time sweep of guided matching on the Step 13 protocol (joint mode, same windows).
Reports pooled in-window supported frames, per-frame mean-corner reference error (median/p95), frames with error
over 10 px (false-pose proxy; no truth beyond the tag reference) and median recovery. Writes runs/step17-sweep.json."""
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

SETTINGS = [("default", {}), ("radius6", {"search_radius_px": 6.0}), ("radius20", {"search_radius_px": 20.0}),
            ("hamming40", {"max_hamming_distance": 40}), ("hamming50", {"max_hamming_distance": 50}),
            ("ratio0.6", {"ratio_test": 0.6}), ("ratio0.9", {"ratio_test": 0.9})]


def main():
    cfg = load_config("config-exp-handoff.json")
    windows = [tuple(w) for w in json.load(open("runs/step13-handoff.json"))["windows"]]
    out = {}
    for label, change in SETTINGS:
        guided = replace(cfg.features.guided_matching, enabled=True, **change)
        path = run(replace(cfg, pose=replace(cfg.pose, mode="joint"), features=replace(cfg.features, guided_matching=guided)), withhold_tag_windows=windows)
        total = sum(in_windows(r["source_time_s"], windows) for r in read_rows(path))
        errors = list(frame_errors(path, windows).values())
        recovery = [m["recovery_after_window_s"] for m in window_metrics(path, windows).values() if m["recovery_after_window_s"] is not None]
        out[label] = {"run": str(path), "guided": {"search_radius_px": guided.search_radius_px, "max_hamming_distance": guided.max_hamming_distance, "ratio_test": guided.ratio_test},
                      "window_frames": total, "supported_frames": len(errors), "supported_fraction": len(errors)/total,
                      "median_px": pct(errors, 50), "p95_px": pct(errors, 95), "over_10px": int(np.sum(np.array(errors) > 10)),
                      "median_recovery_s": pct(recovery, 50)}
        json.dump(out, open("runs/step17-sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
