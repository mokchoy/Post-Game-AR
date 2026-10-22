"""Offline oracle diagnostic: separates detector repeatability, descriptor distinctiveness/matcher
rules, and map coverage as causes of weak landmark-only tracking.

No behavior change to the tracker: detection is recomputed offline with FeatureTracker.detect's
own ORB instance and config, not by touching the pipeline or adding a new logging path. State:
this script recomputes keypoints/descriptors offline rather than adding logging.keypoint_dump.

For each frame with a trustworthy tag-only reference pose (RMS <= 1px), every map landmark that
projects inside the image (10px border) and outside the tag-exclusion mask is a "kept landmark".
For each kept landmark we find the spatially nearest detected ORB keypoint (before the grid-quota
cap) and report: whether it exists within 3/5/10px (detector hit), its Hamming distance to the
landmark's stored descriptor versus every other landmark (distractor distances), whether the
current match_descriptors rules would accept it, and whether it survived the grid-quota selection.
"""
import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.features import FeatureTracker, feature_mask
from ar_projection.geometry import camera_position, depths, project
from ar_projection.landmarks import load_map, map_metadata
from ar_projection.pose import PoseSolver
from ar_projection.tags import TagDetector

RADII = (3.0, 5.0, 10.0)
BORDER_PX = 10


def hamming_distances(query, train):
    """(Q,32) vs (T,32) uint8 -> (Q,T) bit-level Hamming distances, matching cv2.NORM_HAMMING."""
    return cdist(np.unpackbits(np.atleast_2d(query), axis=1), np.unpackbits(np.atleast_2d(train), axis=1), "hamming")*256


def nearest_keypoint(proj, keypoint_pixels):
    """Spatially nearest keypoint to a projected landmark pixel; (None, inf) if there are none."""
    if len(keypoint_pixels) == 0:
        return None, float("inf")
    d = np.linalg.norm(np.asarray(keypoint_pixels)-proj, axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def grid_quota_survivors(keypoints, cfg, size):
    """Indices of `keypoints` that would survive FeatureTracker.detect's per-cell response quota.

    ponytail: duplicates that quota logic (not exported by features.py) so this read-only
    diagnostic never has to import pipeline internals or touch the tracker; keep in sync if
    FeatureTracker.detect's quota rule changes.
    """
    if not keypoints:
        return set()
    w, h = size
    count = cfg.grid_rows*cfg.grid_cols
    base, remainder = divmod(cfg.max_features, count)
    cells = [[] for _ in range(count)]
    for i, key in enumerate(keypoints):
        col = min(cfg.grid_cols-1, int(key.pt[0]*cfg.grid_cols/w))
        row = min(cfg.grid_rows-1, int(key.pt[1]*cfg.grid_rows/h))
        cells[row*cfg.grid_cols+col].append(i)
    survivors = set()
    for row in range(cfg.grid_rows):
        for col in range(cfg.grid_cols):
            quota = base+(row*cfg.grid_cols+col < remainder)
            survivors.update(sorted(cells[row*cfg.grid_cols+col], key=lambda i: keypoints[i].response, reverse=True)[:quota])
    return survivors


def frame_backward_matches(map_descriptors, frame_descriptors, cfg):
    """Each map landmark's nearest keypoint this frame; computed once per frame, not per candidate.

    ponytail: classify_pair used to redo this whole-map-vs-whole-frame match for every landmark hit
    in the frame (tens of thousands of redundant BFMatcher calls over a full clip) — it only depends
    on the frame, so it's hoisted out here and passed in.
    """
    if not cfg.mutual_matching:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    return {m.queryIdx: m.trainIdx for m in matcher.match(map_descriptors, frame_descriptors)}


def classify_pair(query_descriptor, map_descriptors, true_index, backward, true_query_index, cfg):
    """Apply match_descriptors' exact rules to one query against the whole map.

    Mirrors ar_projection.features.match_descriptors' per-pair logic for a single query so the
    specific rejection reason can be reported (match_descriptors itself only returns aggregate
    counts). `backward` is this frame's `frame_backward_matches(...)` result (or None if
    mutual_matching is off). `true_query_index` is this query's position within the frame's
    descriptors, used for the mutual-match check exactly as match_descriptors computes it.
    """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(query_descriptor[None], map_descriptors, k=2)[0]
    a, b = forward
    true_distance = float(hamming_distances(query_descriptor, map_descriptors[true_index])[0, 0])
    rank = 1 if a.trainIdx == true_index else 2 if b.trainIdx == true_index else ">2"
    accepted_id = None
    if not (a.distance > cfg.max_hamming_distance or a.distance >= cfg.ratio_test*b.distance
            or (backward is not None and backward[a.trainIdx] != true_query_index)):
        accepted_id = a.trainIdx
    reason = ("accepted" if accepted_id == true_index else
             "distractor_closer" if rank != 1 else
             "hamming_rejected" if a.distance > cfg.max_hamming_distance else
             "ratio_rejected" if a.distance >= cfg.ratio_test*b.distance else
             "mutual_rejected" if backward is not None and backward[a.trainIdx] != true_query_index else
             "unknown")
    return {"true_distance": true_distance, "best_distance": float(a.distance), "second_distance": float(b.distance),
           "rank": rank, "accepted": accepted_id == true_index, "reason": reason, "winner_index": int(a.trainIdx)}


def collect_rows(config_path, never_associated_run=None, max_frames=None):
    """Shared per-frame collection: one row per kept landmark occurrence. Reused by descriptor_validity_check.py."""
    cfg = load_config(config_path)
    capture = Capture(cfg.camera)
    packet = capture.read()
    camera = packet.camera
    landmark_map = load_map(cfg.mapping.map_path, map_metadata(cfg, camera), cfg)
    tracker = FeatureTracker(cfg.features)
    tag_detector = TagDetector(cfg.tags)
    reference_solver = PoseSolver(replace(cfg, pose=replace(cfg.pose, mode="tag_only"), uncertainty=replace(cfg.uncertainty, enabled=False)))
    never_associated = set()
    if never_associated_run is not None:
        never_associated = _never_associated_ids(never_associated_run, landmark_map)

    all_rows, scored_frames = [], 0
    index = 0
    while packet is not None and (max_frames is None or index < max_frames):
        tags = tag_detector.detect(packet.gray, camera)
        reference = reference_solver.estimate(tags, [], [], camera)
        if reference.valid and reference.tag_rms is not None and reference.tag_rms <= 1.0:
            camera_pos = camera_position(reference.T_CW).tolist()
            mask = feature_mask(packet.gray.shape, tags, cfg.features)
            keys = tracker.orb.detect(packet.gray, mask)
            keys, descriptors = tracker.orb.compute(packet.gray, keys)
            if descriptors is None:
                keys, descriptors = [], np.empty((0, 32), np.uint8)
            pixels = np.array([k.pt for k in keys]) if keys else np.empty((0, 2))
            h, w = packet.gray.shape
            survivors = grid_quota_survivors(keys, cfg.features, (w, h))
            backward = frame_backward_matches(landmark_map.descriptors, descriptors, cfg.features)
            for row_index, (id, point) in enumerate(zip(landmark_map.ids, landmark_map.points)):
                proj = project(reference.T_CW, point[None], camera)[0]
                if not (BORDER_PX <= proj[0] < w-BORDER_PX and BORDER_PX <= proj[1] < h-BORDER_PX) or not mask[int(proj[1]), int(proj[0])]:
                    continue
                nearest_i, distance_px = nearest_keypoint(proj, pixels)
                row = {"frame_index": index, "source_time_s": packet.timestamp, "landmark_id": int(id),
                      "cell": [min(cfg.features.grid_cols-1, int(proj[0]*cfg.features.grid_cols/w)),
                               min(cfg.features.grid_rows-1, int(proj[1]*cfg.features.grid_rows/h))],
                      "depth_m": float(depths(reference.T_CW, point[None])[0]), "camera_position": camera_pos,
                      "hit": {str(int(r)): distance_px <= r for r in RADII}, "nearest_px": distance_px,
                      "never_associated": int(id) in never_associated}
                if nearest_i is not None and distance_px <= RADII[-1]:
                    row["octave"] = int(keys[nearest_i].octave)
                    row["survived_quota"] = nearest_i in survivors
                    row["pixel_offset"] = (pixels[nearest_i]-proj).tolist()
                    row["landmark_point"] = point.tolist()
                    row.update(classify_pair(descriptors[nearest_i], landmark_map.descriptors, row_index, backward, nearest_i, cfg.features))
                    if row["winner_index"] != row_index:
                        row["distractor_id"] = int(landmark_map.ids[row["winner_index"]])
                    distances = hamming_distances(descriptors[nearest_i], landmark_map.descriptors)[0]
                    row["true_distance"] = float(distances[row_index])
                    row["best_other_distance"] = float(np.delete(distances, row_index).min())
                all_rows.append(row)
            scored_frames += 1
        packet = capture.read()
        index += 1
    capture.close()
    return all_rows, scored_frames, cfg, camera, landmark_map


def run(config_path, output, never_associated_run=None, max_frames=None):
    rows, scored_frames, *_ = collect_rows(config_path, never_associated_run, max_frames)
    report = summarize(rows, scored_frames)
    Path(output).write_text(json.dumps(report, indent=2))
    Path(output).with_suffix(".md").write_text(markdown_summary(report))
    print(json.dumps(report, indent=2)[:4000])
    return report


def _never_associated_ids(run_dir, landmark_map):
    run_dir = Path(run_dir)
    ever = set()
    with (run_dir/"assoc.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ever.update(o["landmark_id"] for o in row["observations"])
    return {int(id) for id in landmark_map.ids}-ever


def _hit_rate(rows, radius="3"):
    return float(np.mean([r["hit"][radius] for r in rows])) if rows else None


def summarize(rows, scored_frames):
    def group(key):
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        return {str(k): {"count": len(v), "hit_rate_3px": _hit_rate(v)} for k, v in groups.items()}

    with_pair = [r for r in rows if "true_distance" in r]
    accepted = [r for r in with_pair if r["accepted"]]
    reasons = defaultdict(int)
    for r in with_pair:
        if not r["accepted"]:
            reasons[r["reason"]] += 1
    never_assoc = [r for r in rows if r["never_associated"]]
    return {
        "scored_frames": scored_frames, "kept_landmark_occurrences": len(rows),
        "detector_hit_rate_overall": {str(int(radius)): _hit_rate(rows, str(int(radius))) for radius in RADII},
        "by_time_bin_10s": group(lambda r: int(r["source_time_s"]//10)),
        "by_image_cell": group(lambda r: tuple(r["cell"])),
        "by_landmark_id": group(lambda r: r["landmark_id"]),
        "by_octave": group(lambda r: r.get("octave")),
        "true_pair_distance_median": float(np.median([r["true_distance"] for r in with_pair])) if with_pair else None,
        "true_pair_distance_p90": float(np.percentile([r["true_distance"] for r in with_pair], 90)) if with_pair else None,
        "best_other_distance_median": float(np.median([r["best_other_distance"] for r in with_pair])) if with_pair else None,
        "matcher_would_accept_fraction": len(accepted)/len(with_pair) if with_pair else None,
        "rejection_reasons": dict(reasons),
        "never_associated_landmarks": {
            "count": len({r["landmark_id"] for r in never_assoc}),
            "hit_rate_3px": _hit_rate(never_assoc),
            "matcher_would_accept_fraction": (lambda p: len([r for r in p if r["accepted"]])/len(p) if p else None)([r for r in never_assoc if "true_distance" in r]),
        },
        "pixel_offset_median_by_time_bin_10s": {
            str(k): np.median([r["pixel_offset"] for r in v], axis=0).tolist()
            for k, v in ((k, [r for r in rows if "pixel_offset" in r and int(r["source_time_s"]//10) == k])
                        for k in sorted({int(r["source_time_s"]//10) for r in rows}))
            if v
        },
        "note_viewing_angle_vs_keyframe": "not computable: the frozen map (ar_projection.types.LandmarkMap) does not persist per-landmark keyframe poses, only points/descriptors/ids/support/rms.",
    }


def markdown_summary(report):
    lines = [f"# Oracle diagnostic ({report['scored_frames']} frames, {report['kept_landmark_occurrences']} kept-landmark occurrences)", ""]
    lines.append(f"- Detector hit rate: 3px={report['detector_hit_rate_overall']['3']:.3f}, "
                f"5px={report['detector_hit_rate_overall']['5']:.3f}, 10px={report['detector_hit_rate_overall']['10']:.3f}")
    lines.append(f"- True-pair Hamming distance: median={report['true_pair_distance_median']}, p90={report['true_pair_distance_p90']} "
                f"(limit 60); best-other median={report['best_other_distance_median']}")
    lines.append(f"- Matcher would accept the true pair: {report['matcher_would_accept_fraction']}")
    lines.append(f"- Rejection reasons: {report['rejection_reasons']}")
    n = report["never_associated_landmarks"]
    lines.append(f"- Never-associated landmarks (n={n['count']}): hit_rate_3px={n['hit_rate_3px']}, matcher_accept={n['matcher_would_accept_fraction']}")
    lines.append(f"- {report['note_viewing_angle_vs_keyframe']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--never-associated-run", help="an existing association_dump run directory; marks its visible-but-never-associated landmarks")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.never_associated_run, args.max_frames)


if __name__ == "__main__":
    main()
