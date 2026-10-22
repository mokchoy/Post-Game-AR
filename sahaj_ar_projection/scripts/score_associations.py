"""Offline scorer for logging.association_dump output (assoc.jsonl + frames.jsonl).

For each frame with a trustworthy tag reference (tag_reference_valid and
tag_reference_rms_px <= 1px), reprojects every associated landmark under the
tag-reference pose and scores the pixel error against the live association.
Aggregates by association source, 10s time bin, track-age bucket and landmark ID,
plus per-frame visible-landmark recall / associated-landmark precision proxy.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar_projection.config import load_config
from ar_projection.geometry import project
from ar_projection.landmarks import load_map, map_metadata
from ar_projection.types import CameraModel


def age_bucket(age_frames):
    for limit, label in ((1, "1"), (5, "2-5"), (20, "6-20"), (50, "21-50")):
        if age_frames <= limit:
            return label
    return "51+"


def load_run(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir/"metadata.json").read_text())
    config = load_config(run_dir/"config.json")
    camera = CameraModel(np.array(metadata["calibration"]["K"]), np.array(metadata["calibration"]["distortion"]),
                         tuple(metadata["actual_dimensions"]), metadata["calibration"])
    landmark_map = load_map(config.mapping.map_path, map_metadata(config, camera), config)
    frames = {}
    with (run_dir/"frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            frames[row["frame_index"]] = row
    return config, camera, landmark_map, frames


def score(run_dir):
    config, camera, landmark_map, frames = load_run(run_dir)
    points_by_id = {int(id): p for id, p in zip(landmark_map.ids, landmark_map.points)}
    by_source, by_bin, by_age, by_landmark = {}, {}, {}, {}
    per_frame = []
    assoc_path = Path(run_dir)/"assoc.jsonl"
    if not assoc_path.exists():
        raise ValueError(f"{assoc_path}: no association dump; rerun with logging.association_dump=true")
    with assoc_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            frame = frames.get(row["frame_index"])
            if frame is None:
                continue
            diag = frame["diagnostics"]
            if not diag.get("tag_reference_valid") or diag.get("tag_reference_rms_px") is None or diag["tag_reference_rms_px"] > 1.0:
                continue
            T_ref = np.array(diag["tag_reference_T_CW"])
            observations = row["observations"]
            visible = [id for id, p in points_by_id.items()
                      if np.all((proj := project(T_ref, p[None], camera)[0]) >= 0) and np.all(proj < camera.size)]
            recalled = 0
            within_3px = 0
            for obs in observations:
                point = points_by_id.get(obs["landmark_id"])
                if point is None:
                    continue
                error_px = float(np.linalg.norm(project(T_ref, point[None], camera)[0]-np.array(obs["pixel"])))
                within_3px += error_px <= 3.0
                bucket = age_bucket(obs["age_frames"])
                for store, key in ((by_source, obs["association_source"]), (by_bin, int(row["source_time_s"]//10)
                                   if row.get("source_time_s") is not None else None), (by_age, bucket), (by_landmark, obs["landmark_id"])):
                    store.setdefault(key, []).append(error_px)
            for id in visible:
                point = points_by_id[id]
                nearest = min((float(np.linalg.norm(project(T_ref, point[None], camera)[0]-np.array(o["pixel"])))
                              for o in observations if o["landmark_id"] == id), default=None)
                recalled += nearest is not None and nearest <= 3.0
            per_frame.append({"frame_index": row["frame_index"], "visible_landmarks": len(visible),
                              "recalled_within_3px": recalled, "associations": len(observations),
                              "associations_within_3px": within_3px})
    def summarize(groups):
        return {str(key): {"count": len(errors), "median_px": float(np.median(errors)), "p95_px": float(np.percentile(errors, 95))}
                for key, errors in groups.items() if errors}
    report = {"scored_frames": len(per_frame), "by_association_source": summarize(by_source),
              "by_time_bin_10s": summarize(by_bin), "by_track_age_bucket": summarize(by_age),
              "by_landmark_id": summarize(by_landmark),
              "recall_within_3px": float(np.mean([f["recalled_within_3px"]/f["visible_landmarks"] for f in per_frame if f["visible_landmarks"]])) if per_frame else None,
              "precision_within_3px": float(np.mean([f["associations_within_3px"]/f["associations"] for f in per_frame if f["associations"]])) if per_frame else None}
    return report


def markdown_summary(report):
    lines = [f"# Association score ({report['scored_frames']} frames with trustworthy tag reference)", ""]
    if report["recall_within_3px"] is not None:
        lines.append(f"- Mean visible-landmark recall within 3px: {report['recall_within_3px']:.3f}")
    if report["precision_within_3px"] is not None:
        lines.append(f"- Mean association precision within 3px: {report['precision_within_3px']:.3f}")
    lines.append("")
    lines.append("## By association source")
    for source, stats in sorted(report["by_association_source"].items()):
        lines.append(f"- {source}: n={stats['count']}, median={stats['median_px']:.2f}px, p95={stats['p95_px']:.2f}px")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output", required=True, help="JSON report path; a .md sibling is also written")
    args = parser.parse_args()
    report = score(args.run_dir)
    Path(args.output).write_text(json.dumps(report, indent=2))
    Path(args.output).with_suffix(".md").write_text(markdown_summary(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
