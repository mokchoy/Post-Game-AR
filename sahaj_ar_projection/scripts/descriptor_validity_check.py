"""Step 9b: decide whether Step 9's large true-pair distance is (A) a map-building defect
(wrong stored descriptor), (B) a measurement artifact (wrong nearest-keypoint pairing, or
near-duplicate/clustered landmarks), or (C) a genuine view-coverage problem. Only (C) justifies
a new capture. No tracker behavior change; offline only.

Four tasks, each independently runnable:
  1. tight-radius true pairs (2/3/5px) vs Step 9's 10px pairing
  2. descriptor self-consistency on the build recording (source view vs other supporting views)
  3. true-pair distance vs viewing-angle change from the source keyframe
  4. distractor geometry (3D distance between true and distractor landmarks) and descriptor diversity
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.pipeline import Pipeline
from oracle_diagnostic import collect_rows, hamming_distances  # noqa: E402

TIGHT_RADII = (2.0, 3.0, 5.0, 10.0)
BIT_LIMITS = (30, 40, 60)
NEAREST_KEYPOINT_TOLERANCE_PX = 2.0


# --- Task 1: tight-radius true pairs ------------------------------------------------------------

def tight_radius_report(rows):
    with_pair = [r for r in rows if "true_distance" in r]
    report = {}
    for radius in TIGHT_RADII:
        subset = [r for r in with_pair if r["nearest_px"] <= radius]
        distances = [r["true_distance"] for r in subset]
        accept = [r["accepted"] for r in subset]
        report[str(radius)] = {
            "count": len(subset), "median": float(np.median(distances)) if distances else None,
            "p10": float(np.percentile(distances, 10)) if distances else None,
            "p90": float(np.percentile(distances, 90)) if distances else None,
            "fraction_under": {str(b): float(np.mean(np.array(distances) < b)) if distances else None for b in BIT_LIMITS},
            "matcher_would_accept_fraction": float(np.mean(accept)) if accept else None,
        }
    return report


# --- Task 2/3 shared: replay the build recording and recover provenance ------------------------

def replay_build_with_provenance(config_path, max_frames=5000):
    """Replay a build-mode recording, recording which keyframe/pixel supported each landmark.

    The stored config.json for a build run records its *original* camera (often a live
    "camera" source), not a replayable one — so this always points the capture at that run
    directory's own input.avi/frames.jsonl, the same override Step 4's rebuilds used. Without
    it, Capture opens the live webcam and the loop never reaches end-of-stream.

    Returns (landmark_map, provenance, keyframe_poses) where provenance[landmark_id] is a list of
    (keyframe_index, pixel, keyframe_gray_image) ordered as MapBuilder saw them (index 0 is the
    source view whose descriptor got stored), and keyframe_poses[j] is that keyframe's final
    (post-refinement) T_CW.
    """
    from dataclasses import replace
    cfg = load_config(config_path)
    recording = Path(config_path).resolve().parent
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(recording/"input.avi"),
                                      timestamps_path=str(recording/"frames.jsonl")))
    np.random.seed(cfg.seed)
    cv2.setRNGSeed(cfg.seed)
    capture = Capture(cfg.camera)
    packet = capture.read()
    pipeline = Pipeline(cfg, packet.camera, background_mapping=False)  # drives the real tag/feature/builder path exactly as `run()` does
    builder = pipeline.builder
    keyframe_images = []
    index = 0
    while packet is not None and builder.map is None:
        if index >= max_frames:
            capture.close()
            raise ValueError(f"{config_path}: no map after {max_frames} frames; recording or config is probably wrong")
        before = len(builder.keyframes)
        pipeline.process(packet)
        if len(builder.keyframes) > before:
            keyframe_images.append(packet.gray.copy())
        packet = capture.read()
        index += 1
    capture.close()
    if builder.map is None:
        raise ValueError(f"{config_path}: build did not complete ({builder.reason})")
    provenance = defaultdict(list)
    for j, k in enumerate(builder.keyframes):
        for id, f in k.features.items():
            provenance[id].append((j, f.pixel.copy(), keyframe_images[j]))
    keyframe_poses = [k.T.copy() for k in builder.keyframes]
    return builder.map, provenance, keyframe_poses


# --- Task 2: descriptor self-consistency ---------------------------------------------------

def self_consistency_report(landmark_map, provenance):
    cache = {}

    def detect_near(image_id, image, pixel):
        if image_id not in cache:
            keys, descriptors = cv2.ORB_create().detectAndCompute(image, None)
            cache[image_id] = (np.array([k.pt for k in keys]) if keys else np.empty((0, 2)),
                               descriptors if descriptors is not None else np.empty((0, 32), np.uint8))
        pixels, descriptors = cache[image_id]
        if len(pixels) == 0:
            return None
        d = np.linalg.norm(pixels-pixel, axis=1)
        i = int(np.argmin(d))
        return descriptors[i] if d[i] <= NEAREST_KEYPOINT_TOLERANCE_PX else None

    source_distances, other_distances, missing = [], [], 0
    for row_index, id in enumerate(landmark_map.ids):
        for view_index, (j, pixel, image) in enumerate(provenance.get(int(id), [])):
            fresh = detect_near(j, image, pixel)
            if fresh is None:
                missing += 1
                continue
            distance = float(hamming_distances(fresh, landmark_map.descriptors[row_index])[0, 0])
            (source_distances if view_index == 0 else other_distances).append(distance)

    def stats(values):
        return {"count": len(values), "median": float(np.median(values)) if values else None,
               "p90": float(np.percentile(values, 90)) if values else None}
    return {"source_view": stats(source_distances), "other_supporting_views": stats(other_distances),
           "no_keypoint_within_tolerance": missing}


# --- Task 3: distance vs viewing change ---------------------------------------------------------

def view_change_bin(angle_deg):
    for limit, label in ((5, "0-5"), (15, "5-15"), (30, "15-30")):
        if angle_deg <= limit:
            return label
    return "30+"


def view_change_report(rows, provenance, keyframe_poses, tight_radius=3.0):
    from ar_projection.geometry import camera_position
    source_camera = {id: camera_position(keyframe_poses[views[0][0]]) for id, views in provenance.items()}
    bins = defaultdict(list)
    for r in rows:
        if "true_distance" not in r or r["nearest_px"] > tight_radius:
            continue
        source = source_camera.get(r["landmark_id"])
        if source is None:
            continue
        point = np.array(r["landmark_point"])
        current_direction = point-np.array(r["camera_position"])
        source_direction = point-source
        current_direction = current_direction/np.linalg.norm(current_direction)
        source_direction = source_direction/np.linalg.norm(source_direction)
        angle = float(np.degrees(np.arccos(np.clip(current_direction@source_direction, -1, 1))))
        bins[view_change_bin(angle)].append(r["true_distance"])
    return {k: {"count": len(v), "median_true_distance": float(np.median(v))} for k, v in bins.items() if v}


# --- Task 4: distractor geometry and descriptor diversity --------------------------------------

def distractor_geometry_report(rows, landmark_map):
    points = {int(id): p for id, p in zip(landmark_map.ids, landmark_map.points)}
    metres = []
    for r in rows:
        if r.get("reason") != "distractor_closer" or r.get("distractor_id") not in points:
            continue
        metres.append(float(np.linalg.norm(points[r["landmark_id"]]-points[r["distractor_id"]])))
    within_m = {str(m): float(np.mean(np.array(metres) <= m)) if metres else None for m in (0.02, 0.05, 0.10)}
    pairwise = hamming_distances(landmark_map.descriptors, landmark_map.descriptors)
    np.fill_diagonal(pairwise, 256)
    flat = pairwise[np.triu_indices_from(pairwise, k=1)]
    return {
        "distractor_closer_occurrences_with_geometry": len(metres),
        "fraction_within_metres": within_m, "metres_median": float(np.median(metres)) if metres else None,
        "metres_p10": float(np.percentile(metres, 10)) if metres else None,
        "stored_descriptor_pairwise_hamming": {"median": float(np.median(flat)), "p10": float(np.percentile(flat, 10)),
                                               "fraction_under_40": float(np.mean(flat < 40))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("1", "2", "3", "4", "all"))
    parser.add_argument("--clip-config", default="runs/20261019T133214-ecda4960/config.json")
    parser.add_argument("--build-config", default="runs/20261019T090841-72509636/config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    report = {}
    rows = landmark_map = None
    if args.task in ("1", "3", "4", "all"):
        rows, scored_frames, cfg, camera, landmark_map = collect_rows(args.clip_config, max_frames=args.max_frames)
        report["scored_frames"] = scored_frames
    if args.task in ("1", "all"):
        report["tight_radius"] = tight_radius_report(rows)
    if args.task in ("2", "3", "all"):
        build_map, provenance, keyframe_poses = replay_build_with_provenance(args.build_config)
        if args.task in ("2", "all"):
            report["self_consistency"] = self_consistency_report(build_map, provenance)
        if args.task in ("3", "all"):
            report["view_change"] = view_change_report(rows, provenance, keyframe_poses)
    if args.task in ("4", "all"):
        report["distractor_geometry"] = distractor_geometry_report(rows, landmark_map)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:6000])


if __name__ == "__main__":
    main()
