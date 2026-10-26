"""Step 23: whole-clip, non-overlapping withhold windows on the tag-valid stretches of the clip, three guided settings.

Windows come from a tag_only run: tag-valid stretches (gaps up to 0.5 s merged); inside a stretch the first 2 s is
tag lead, then windows of 20, 10 or 5 s (largest that fits) separated by 2 s of tag. Joint mode, guided on, one setting
per run. Writes runs/step23-windows.json and runs/step23-clipwide.json.
"""
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

TAG_ONLY_RUN = "runs/20261022T194955-42300b59"  # Step 13 tag_only replay of the same clip
SETTINGS = [("default_h60", {}), ("hamming128", {"max_hamming_distance": 128}),
            ("geometry_first", {"max_hamming_distance": 128, "geometry_first": True})]


def stretches(directory, max_gap_s=0.5, min_len_s=7.0):
    rows = [r for r in read_rows(directory) if r["source_time_s"] is not None]
    good = [r["source_time_s"] for r in rows if r["raw_pose"]["valid"] and r["actual_pose_source"] == "tag"]
    out = []
    for t in good:
        if out and t-out[-1][1] <= max_gap_s:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out if b-a >= min_len_s]


def make_windows(spans, lead_s=2.0):
    windows = []
    for index, (a, b) in enumerate(spans):
        t = a+lead_s
        while b-t >= 5:
            length = 20 if b-t >= 20 else 10 if b-t >= 10 else 5
            windows.append({"stretch": index, "start": t, "end": t+length})
            t += length+lead_s
    return windows


def first_valid_s(rows, start, end):
    for r in rows:
        t = r["source_time_s"]
        if t is not None and start <= t <= end and r["raw_pose"]["valid"] and r["actual_pose_source"] == "landmark":
            return t-start
    return None


def summarize_run(path, windows):
    spans = [(w["start"], w["end"]) for w in windows]
    rows = read_rows(path)
    metrics = window_metrics(path, spans)
    errors = frame_errors(path, spans)
    per_window = []
    for w in windows:
        a, b = w["start"], w["end"]
        m = metrics[f"{a}:{b}"]
        mine = {r["frame_index"] for r in rows if r["source_time_s"] is not None and a <= r["source_time_s"] <= b}
        e = [errors[i] for i in mine if i in errors]
        per_window.append({**w, "length_s": b-a, "valid_fraction": m["valid_fraction"], "time_to_first_valid_s": first_valid_s(rows, a, b),
                           "longest_supported_run_s": m["longest_supported_run_s"], "per_corner_median_px": m["reference_median_px"],
                           "per_corner_p95_px": m["reference_p95_px"], "supported_frames": len(e),
                           "over_10px_share": float(np.mean(np.array(e) > 10)) if e else None, "recovery_after_window_s": m["recovery_after_window_s"]})
    inside = [r for r in rows if in_windows(r["source_time_s"], spans)]
    ref_valid = [r for r in inside if r["diagnostics"].get("tag_reference_valid")]
    supported = lambda r: r["frame_index"] in errors
    per_stretch = {}
    for s in sorted({w["stretch"] for w in windows}):
        ws = [w for w in windows if w["stretch"] == s]
        frames = [r for r in inside if any(w["start"] <= r["source_time_s"] <= w["end"] for w in ws)]
        e = [errors[r["frame_index"]] for r in frames if supported(r)]
        per_stretch[s] = {"windows": len(ws), "seconds": sum(w["end"]-w["start"] for w in ws), "frames": len(frames),
                          "supported_fraction": len(e)/len(frames) if frames else None, "median_px": pct(e, 50), "p95_px": pct(e, 95),
                          "over_10px_share": float(np.mean(np.array(e) > 10)) if e else None}
    allerr = [errors[r["frame_index"]] for r in inside if supported(r)]
    return {"per_window": per_window, "per_stretch": per_stretch,
            "clip": {"in_window_frames": len(inside), "supported_fraction": len(allerr)/len(inside),
                     "supported_fraction_reference_valid_frames": sum(supported(r) for r in ref_valid)/len(ref_valid) if ref_valid else None,
                     "median_px": pct(allerr, 50), "p95_px": pct(allerr, 95),
                     "over_10px_share": float(np.mean(np.array(allerr) > 10)) if allerr else None}}


def main():
    cfg = load_config("config-exp-handoff.json")
    spans = stretches(TAG_ONLY_RUN)
    windows = make_windows(spans)
    json.dump({"tag_only_run": TAG_ONLY_RUN, "stretches": spans, "windows": windows}, open("runs/step23-windows.json", "w"), indent=1)
    out = {}
    for label, change in SETTINGS:
        guided = replace(cfg.features.guided_matching, enabled=True, **change)
        path = run(replace(cfg, pose=replace(cfg.pose, mode="joint"), features=replace(cfg.features, guided_matching=guided)),
                   withhold_tag_windows=[(w["start"], w["end"]) for w in windows])
        out[label] = {"run": str(path), "guided": {k: getattr(guided, k) for k in ("search_radius_px", "max_hamming_distance", "ratio_test", "geometry_first", "revalidate_existing")},
                      **summarize_run(path, windows)}
        json.dump(out, open("runs/step23-clipwide.json", "w"), indent=1)


if __name__ == "__main__":
    main()
