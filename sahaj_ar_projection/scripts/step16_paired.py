"""Step 16: paired reference accuracy (baseline vs guided on common frames), error by association source,
and LK drift of guided-assigned tracks at forward/backward gates 5/2/1 px. Joint mode, Step 13 windows,
association dump on. Offline replays of the existing clip; writes runs/step16-paired.json."""
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.config import load_config
from ar_projection.experiments import read_rows
from ar_projection.geometry import project
from ar_projection.pipeline import run
from score_associations import load_run


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else None


def stats(v):
    return {"frames_or_samples": len(v), "median_px": pct(v, 50), "p95_px": pct(v, 95)}


def in_windows(t, windows):
    return t is not None and any(a <= t <= b for a, b in windows)


def frame_errors(directory, windows):
    """frame_index -> mean corner error vs tag reference, for supported frames inside the windows."""
    out = {}
    for r in read_rows(directory):
        pose, ref = r["raw_pose"], r["diagnostics"].get("reference_panel_pixels")
        if (in_windows(r["source_time_s"], windows) and pose["valid"] and ref is not None and r.get("raw_panel_pixels") is not None
                and r["actual_pose_source"] in ("tag", "joint", "landmark") and pose.get("tag_inliers", 0)+pose.get("landmark_inliers", 0) >= 4):
            out[r["frame_index"]] = float(np.linalg.norm(np.asarray(ref)-r["raw_panel_pixels"], axis=1).mean())
    return out


def paired(base_dir, guided_dir, windows):
    b, g = frame_errors(base_dir, windows), frame_errors(guided_dir, windows)
    common = b.keys() & g.keys()
    return {"common_baseline": stats([b[i] for i in common]), "common_guided": stats([g[i] for i in common]),
            "only_baseline": stats([b[i] for i in b.keys()-g.keys()]), "only_guided": stats([g[i] for i in g.keys()-b.keys()])}


def observations(directory, windows):
    """Yield (frame_index, obs, error_px) for in-window frames with a trusted reference."""
    _, camera, landmark_map, frames = load_run(directory)
    points = {int(i): p for i, p in zip(landmark_map.ids, landmark_map.points)}
    for line in (Path(directory)/"assoc.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        f = frames.get(row["frame_index"])
        d = f["diagnostics"] if f else {}
        if not in_windows(row.get("source_time_s"), windows) or not d.get("tag_reference_valid") or (d.get("tag_reference_rms_px") or 9) > 1.0:
            continue
        T = np.array(d["tag_reference_T_CW"])
        for o in row["observations"]:
            if o["landmark_id"] in points:
                yield row["frame_index"], o, float(np.linalg.norm(project(T, points[o["landmark_id"]][None], camera)[0]-np.array(o["pixel"])))


def by_source(directory, windows):
    allv, inl = defaultdict(list), defaultdict(list)
    for _, o, e in observations(directory, windows):
        allv[o["association_source"]].append(e)
        if o["ransac_inlier"]:
            inl[o["association_source"]].append(e)
    return {"all": {k: {**stats(v), "within_3px": float(np.mean(np.array(v) <= 3))} for k, v in allv.items()},
            "ransac_inliers": {k: stats(v) for k, v in inl.items()}}


def guided_drift(directory, windows):
    """Tracks first seen as source 'guided' within 3 px of the reference: later error vs frames since assignment."""
    first, buckets, started = {}, defaultdict(list), 0
    for frame, o, e in observations(directory, windows):
        key = (o["track_id"], o["landmark_id"])
        if key not in first:
            if o["association_source"] == "guided" and e <= 3.0:
                first[key] = frame
                started += 1
            continue
        age = frame-first[key]
        label = "1-5" if age <= 5 else "6-20" if age <= 20 else "21-50" if age <= 50 else "51+"
        buckets[label].append(e)
    return {"qualifying_tracks": started,
            "by_age_since_assignment": {k: {**stats(v), "within_3px": float(np.mean(np.array(v) <= 3))} for k, v in sorted(buckets.items())}}


def main():
    cfg = load_config("config-exp-handoff.json")
    windows = [tuple(w) for w in json.load(open("runs/step13-handoff.json"))["windows"]]
    base = replace(cfg, pose=replace(cfg.pose, mode="joint"), logging=replace(cfg.logging, association_dump=True))
    runs = {}
    for label, guided, gate in (("baseline_fb5", False, 5.0), ("guided_fb5", True, 5.0), ("guided_fb2", True, 2.0), ("guided_fb1", True, 1.0)):
        variant = replace(base, features=replace(base.features, guided_matching=replace(base.features.guided_matching, enabled=guided),
                                                 lk=replace(base.features.lk, forward_backward_max_error_px=gate)))
        runs[label] = str(run(variant, withhold_tag_windows=windows))
    report = {"runs": runs, "paired_fb5": paired(runs["baseline_fb5"], runs["guided_fb5"], windows),
              "by_source": {k: by_source(v, windows) for k, v in runs.items()},
              "guided_drift": {k: guided_drift(runs[k], windows) for k in ("guided_fb5", "guided_fb2", "guided_fb1")},
              "valid_frames": {k: len(frame_errors(v, windows)) for k, v in runs.items()}}
    json.dump(report, open("runs/step16-paired.json", "w"), indent=1)


if __name__ == "__main__":
    main()
