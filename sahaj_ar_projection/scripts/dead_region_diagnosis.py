"""Step 14: per-10s-bin dead-region diagnosis. Offline; no tracker change.

Per absolute 10 s bin (source_time_s // 10): tag/reference validity, visible map landmarks under the
tag-reference pose, oracle detector hit rate (3/10 px) and median true-pair Hamming distance (reusing
oracle_diagnostic.collect_rows), landmark-only valid fraction and association counts/correct by source
(from an association-dump run), distance of the camera to the nearest camera position seen while building
the map (coverage), image brightness and variance-of-Laplacian sharpness.
"""
import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.geometry import camera_position, project
from ar_projection.pose import PoseSolver
from ar_projection.tags import TagDetector
from oracle_diagnostic import collect_rows
from score_associations import load_run


def reference_pass(cfg, video=None, timestamps=None):
    """Per frame: time, tag-reference validity (RMS<=1px), camera position, brightness, sharpness."""
    if video:
        cfg = replace(cfg, camera=replace(cfg.camera, video_path=str(video), timestamps_path=str(timestamps) if timestamps else None))
    capture, detector = Capture(cfg.camera), TagDetector(cfg.tags)
    solver = PoseSolver(replace(cfg, pose=replace(cfg.pose, mode="tag_only"), uncertainty=replace(cfg.uncertainty, enabled=False)))
    out, packet = [], capture.read()
    while packet is not None:
        ref = solver.estimate(detector.detect(packet.gray, packet.camera), [], [], packet.camera)
        good = bool(ref.valid and ref.tag_rms is not None and ref.tag_rms <= 1.0)
        out.append({"t": packet.timestamp, "ref": good, "pos": camera_position(ref.T_CW) if good else None,
                    "brightness": float(packet.gray.mean()), "sharpness": float(cv2.Laplacian(packet.gray, cv2.CV_64F).var())})
        packet = capture.read()
    capture.close()
    return out


def assoc_by_bin(run_dir):
    config, camera, landmark_map, frames = load_run(run_dir)
    points = {int(i): p for i, p in zip(landmark_map.ids, landmark_map.points)}
    bins = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # bin -> source -> [associations, within 3px]
    valid = defaultdict(lambda: [0, 0])
    for index, f in frames.items():
        t = f["source_time_s"]
        if t is None:
            continue
        valid[int(t//10)][0] += bool(f["raw_pose"]["valid"])
        valid[int(t//10)][1] += 1
    with (Path(run_dir)/"assoc.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            f = frames.get(row["frame_index"])
            d = f["diagnostics"] if f else {}
            if not d.get("tag_reference_valid") or (d.get("tag_reference_rms_px") or 9) > 1.0 or row.get("source_time_s") is None:
                continue
            T = np.array(d["tag_reference_T_CW"])
            for o in row["observations"]:
                p = points.get(o["landmark_id"])
                if p is None:
                    continue
                err = np.linalg.norm(project(T, p[None], camera)[0]-np.array(o["pixel"]))
                cell = bins[int(row["source_time_s"]//10)][o["association_source"]]
                cell[0] += 1
                cell[1] += int(err <= 3.0)
    return bins, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--assoc-run", required=True, help="lk_periodic withhold-tag association-dump run directory")
    ap.add_argument("--build-recording", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows, _, _, _, landmark_map = collect_rows(args.config)
    frames = reference_pass(cfg)
    build = reference_pass(cfg, Path(args.build_recording).resolve()/"input.avi", Path(args.build_recording).resolve()/"frames.jsonl")
    build_pos = np.array([b["pos"] for b in build if b["ref"]])
    assoc, valid = assoc_by_bin(args.assoc_run)
    bins = sorted({int(f["t"]//10) for f in frames if f["t"] is not None})
    report = {}
    for b in bins:
        fs = [f for f in frames if f["t"] is not None and int(f["t"]//10) == b]
        rs = [r for r in rows if int(r["source_time_s"]//10) == b]
        good = [f for f in fs if f["ref"]]
        near = [float(np.linalg.norm(build_pos-f["pos"], axis=1).min()) for f in good]
        per_frame = defaultdict(int)
        for r in rs:
            per_frame[r["frame_index"]] += 1
        td = [r["true_distance"] for r in rs if "true_distance" in r]
        report[str(b)] = {
            "frames": len(fs), "tag_reference_valid_fraction": len(good)/len(fs),
            "visible_landmarks_mean": float(np.mean(list(per_frame.values()))) if per_frame else None,
            "oracle_hit_3px": float(np.mean([r["hit"]["3"] for r in rs])) if rs else None,
            "oracle_hit_10px": float(np.mean([r["hit"]["10"] for r in rs])) if rs else None,
            "true_pair_distance_median": float(np.median(td)) if td else None,
            "landmark_only_valid_fraction": valid[b][0]/valid[b][1] if valid[b][1] else None,
            "associations_by_source": {s: {"count": c, "within_3px": w} for s, (c, w) in assoc[b].items()},
            "camera_to_nearest_build_pose_m_median": float(np.median(near)) if near else None,
            "camera_position_mean": np.mean([f["pos"] for f in good], axis=0).tolist() if good else None,
            "brightness_mean": float(np.mean([f["brightness"] for f in fs])),
            "sharpness_laplacian_var_mean": float(np.mean([f["sharpness"] for f in fs]))}
    Path(args.output).write_text(json.dumps({"bins": report, "map_landmarks": len(landmark_map.ids),
                                             "build_reference_frames": len(build_pos)}, indent=2))
    for b, v in report.items():
        print(b, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k not in ("camera_position_mean",)})


if __name__ == "__main__":
    main()
