"""Step 18: per-frame trace around the start of two dead windows and one productive window. Diagnosis only.

For the last 10 tag-valid frames before each window and the first 30 frames inside it (a joint, guided-on,
association-dump run from Step 16 supplies the tracker IDs and RANSAC gates; keypoints are recomputed
offline like oracle_diagnostic.py), reports under the tag-reference pose: landmarks visible, correct keypoints
inside the 12 px guided window, correct-pair Hamming distance and how many fail the gate, features
carrying IDs and their correctness by source, offline guided matches and their correctness, and the RANSAC gate.
The prior used for the offline guided_matches() call is the tag reference, not the tracker seed (not logged).
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.features import FeatureTracker, feature_mask
from ar_projection.geometry import project
from ar_projection.landmarks import guided_matches, load_map, map_metadata
from ar_projection.pose import PoseSolver
from ar_projection.tags import TagDetector
from ar_projection.types import FeatureObservation
from oracle_diagnostic import BORDER_PX, grid_quota_survivors, hamming_distances

WINDOWS = ("3886.062", "3908.531", "3925.234")  # two dead, one productive (start times, as in Step 13)


def trace_frame(index, packet, cfg, tracker, detector, solver, landmark_map, assoc, frame_row):
    camera = packet.camera
    tags = detector.detect(packet.gray, camera)
    ref = solver.estimate(tags, [], [], camera)
    if not (ref.valid and ref.tag_rms is not None and ref.tag_rms <= 1.0):
        return {"frame_index": index, "t": packet.timestamp, "tag_reference_valid": False}
    h, w = packet.gray.shape
    mask = feature_mask(packet.gray.shape, tags, cfg.features)
    keys = tracker.orb.detect(packet.gray, mask)
    keys, desc = tracker.orb.compute(packet.gray, keys)
    keep = sorted(grid_quota_survivors(keys, cfg.features, (w, h))) if keys else []
    pixels = np.array([keys[i].pt for i in keep]).reshape(-1, 2)
    fresh = [FeatureObservation(n, pixels[n], desc[i]) for n, i in enumerate(keep)]
    g = cfg.features.guided_matching
    points = {int(i): p for i, p in zip(landmark_map.ids, landmark_map.points)}
    proj = {i: project(ref.T_CW, p[None], camera)[0] for i, p in points.items()}
    row_of = {int(i): r for r, i in enumerate(landmark_map.ids)}
    visible = [i for i, q in proj.items() if BORDER_PX <= q[0] < w-BORDER_PX and BORDER_PX <= q[1] < h-BORDER_PX and mask[int(q[1]), int(q[0])]]
    in_window = correct = fail_gate = 0
    distances = []
    for i in visible:
        d = np.linalg.norm(pixels-proj[i], axis=1) if len(pixels) else np.array([])
        in_window += bool((d <= g.search_radius_px).any())
        if len(d) and d.min() <= 3.0:
            correct += 1
            dist = float(hamming_distances(fresh[int(d.argmin())].descriptor, landmark_map.descriptors[row_of[i]])[0, 0])
            distances.append(dist)
            fail_gate += dist > g.max_hamming_distance
    matches = guided_matches(fresh, landmark_map, ref.T_CW, camera, g, set())
    guided_correct = sum(np.linalg.norm(fresh[k].pixel-proj[i]) <= 3.0 for k, i, _, _ in matches)
    by_source = {}
    for o in (assoc or {}).get("observations", []):
        if o["landmark_id"] in proj:
            cell = by_source.setdefault(o["association_source"], [0, 0])
            cell[0] += 1
            cell[1] += int(np.linalg.norm(proj[o["landmark_id"]]-np.array(o["pixel"])) <= 3.0)
    pg = frame_row["diagnostics"].get("pose_gates", {}) if frame_row else {}
    return {"frame_index": index, "t": packet.timestamp, "tag_reference_valid": True, "visible_landmarks": len(visible),
            "landmarks_with_keypoint_in_search_window": in_window, "landmarks_with_correct_keypoint_3px": correct,
            "correct_pairs_failing_hamming_gate": int(fail_gate), "correct_pair_hamming_median": float(np.median(distances)) if distances else None,
            "offline_guided_matches": len(matches), "offline_guided_correct": int(guided_correct),
            "tracker_ids_by_source_total_correct": by_source,
            "tracker_ids_total": sum(v[0] for v in by_source.values()), "tracker_ids_correct": sum(v[1] for v in by_source.values()),
            "ransac": {"in_image_input": pg.get("in_image_input"), "ransac_inliers": pg.get("ransac_inliers"),
                       "occupied_cells": pg.get("occupied_cells"), "gate_reason": pg.get("landmark_gate_reason")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--dump-run", required=True, help="joint, guided-on, association_dump run with the Step 13 windows")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    frames = {r["frame_index"]: r for r in map(json.loads, (Path(args.dump_run)/"frames.jsonl").read_text(encoding="utf-8").splitlines())}
    assoc = {r["frame_index"]: r for r in map(json.loads, (Path(args.dump_run)/"assoc.jsonl").read_text(encoding="utf-8").splitlines())}
    all_windows = json.load(open("runs/step13-handoff.json"))["windows"]
    chosen = [w for w in all_windows if any(f"{w[0]:.3f}".startswith(s) for s in WINDOWS)]
    first_inside = {tuple(w): min(i for i, r in frames.items() if r["source_time_s"] is not None and w[0] <= r["source_time_s"] <= w[1]) for w in chosen}
    wanted = {i for f in first_inside.values() for i in range(f-60, f+30)}
    capture, tracker, detector = Capture(cfg.camera), FeatureTracker(cfg.features), TagDetector(cfg.tags)
    solver = PoseSolver(replace(cfg, pose=replace(cfg.pose, mode="tag_only"), uncertainty=replace(cfg.uncertainty, enabled=False)))
    packet = capture.read()
    landmark_map = load_map(cfg.mapping.map_path, map_metadata(cfg, packet.camera), cfg)
    traced, index = {}, 0
    while packet is not None and index <= max(wanted):
        if index in wanted:
            traced[index] = trace_frame(index, packet, cfg, tracker, detector, solver, landmark_map, assoc.get(index), frames.get(index))
        packet = capture.read()
        index += 1
    capture.close()
    report = {"dump_run": args.dump_run, "windows": {}}
    for w, first in first_inside.items():
        before = [traced[i] for i in range(first-60, first) if traced[i]["tag_reference_valid"]][-10:]
        inside = [traced[i] for i in range(first, first+30)]
        report["windows"][f"{w[0]}:{w[1]}"] = {"before_window_last_10_tag_valid": before, "inside_first_30": inside}
    Path(args.output).write_text(json.dumps(report, indent=1))
    keys = ("visible_landmarks", "landmarks_with_keypoint_in_search_window", "landmarks_with_correct_keypoint_3px", "correct_pairs_failing_hamming_gate",
            "offline_guided_matches", "offline_guided_correct", "tracker_ids_total", "tracker_ids_correct")
    lines = ["| window | phase | frames | " + " | ".join(k.replace("_", " ") for k in keys) + " | RANSAC gate reasons |", "|" + "---|"*(len(keys)+4)]
    for name, phases in report["windows"].items():
        for phase, rows in phases.items():
            rows = [r for r in rows if r["tag_reference_valid"]]
            means = [f"{np.mean([r[k] for r in rows]):.1f}" if rows else "-" for k in keys]
            reasons = {}
            for r in rows:
                key = str(r["ransac"]["gate_reason"])
                reasons[key] = reasons.get(key, 0)+1
            lines.append(f"| {name} | {phase} | {len(rows)} | " + " | ".join(means) + f" | {reasons} |")
    Path(args.output).with_suffix(".md").write_text("Per-frame means over tag-reference-valid frames.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
