"""Reproducible tracking comparisons and explicit physical acceptance gates.

python -m ar_projection.experiments --help
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .diagnostics import code_sha256, write_json


def read_rows(directory):
    return [json.loads(line) for line in (Path(directory)/"frames.jsonl").read_text(encoding="utf-8").splitlines()]


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def summarize(directory):
    rows = read_rows(directory)
    sources, gates, lk_rejections = Counter(), Counter(), Counter()
    lk_input = lk_retained = 0
    intervals, errors, mapped, ages, processing, capture, latency, read_latency, displays = [], [], [], [], [], [], [], [], []
    occlusion_states, inference_times, gpu_memory, depth_sizes = Counter(), [], [], set()
    associations, descriptor_rejections = Counter(), Counter()
    ransac_counts, coverage_counts, final_inliers = [], [], []
    stages = {}
    no_tag_frames = no_tag_valid = 0
    no_tag_duration = longest_no_tag = current_no_tag = 0.0
    failure_start = evidence_return = None
    recoveries, failures = [], []
    acquired, startup_acquisition = False, None  # startup acquisition is reported apart from loss recovery
    cfg = load_config(Path(directory)/"config.json")
    for i, row in enumerate(rows):
        d, pose = row["diagnostics"], row["raw_pose"]
        source = row["actual_pose_source"]
        supported = bool(pose["valid"] and source in ("tag", "joint", "landmark")
                         and pose.get("tag_inliers", 0)+pose.get("landmark_inliers", 0) >= 4)
        sources[source] += 1
        gate = d.get("pose_gates", {}).get("landmark_gate_reason")
        gates[str(gate)] += 1
        pose_gates = d.get("pose_gates", {})
        ransac_counts.append(pose_gates.get("ransac_inliers", 0))
        coverage_counts.append(pose_gates.get("occupied_cells", 0))
        final_inliers.append(pose.get("landmark_inliers", 0))
        match = d.get("map_associations", {})
        associations.update({k: match.get(k, 0) for k in ("inherited_lk_associations", "redetected_orb_associations", "fresh_descriptor_associations")})
        descriptor_rejections.update({k: v for k, v in match.get("descriptor_matching", {}).items() if k.endswith("_rejected")})
        flow = d.get("feature_tracking", {}).get("lk", {})
        lk_input += flow.get("input", 0)
        lk_retained += flow.get("retained", 0)
        lk_rejections.update({k: v for k, v in flow.items() if k.endswith("_rejected") or k == "call_failed"})
        ages.append(d.get("feature_tracking", {}).get("track_age_median_frames", 0))
        mapped.append(d.get("mapped_observations", 0))
        timings = row.get("timings_ms", {})
        for name, value in timings.items():
            if value is not None:
                stages.setdefault(name, []).append(value)
        processing.append(timings.get("total", 0))
        capture.append(timings.get("capture_read", 0))
        if timings.get("acquisition_to_display") is not None:
            latency.append(timings["acquisition_to_display"])
        if timings.get("read_to_display") is not None:
            read_latency.append(timings["read_to_display"])
        occlusion = d.get("occlusion", {})
        if occlusion:
            occlusion_states[occlusion["status"]] += 1
            if occlusion.get("inference_ms") is not None:
                inference_times.append(occlusion["inference_ms"])
            if occlusion.get("gpu_peak_allocated_mb") is not None:
                gpu_memory.append(occlusion["gpu_peak_allocated_mb"])
            if occlusion.get("input_size") is not None:
                depth_sizes.add(occlusion["input_size"])
        if d.get("display_monotonic_s") is not None:
            displays.append(d["display_monotonic_s"])
        now = row["source_time_s"]
        previous = rows[i-1]["source_time_s"] if i else now
        dt = max(0.0, now-previous) if now is not None and previous is not None else 0.0
        intervals.append(dt)
        tag = any(t.get("valid") and not t.get("ambiguous") and t["id"] == cfg.tags.anchor_id for t in d.get("tags", []))
        no_tag = d.get("tag_observations_withheld", False) or not tag
        if no_tag:
            no_tag_frames += 1
            no_tag_valid += supported
            current_no_tag += dt if i and (rows[i-1]["diagnostics"].get("tag_observations_withheld", False)
                                          or not any(t.get("valid") and not t.get("ambiguous") and t["id"] == cfg.tags.anchor_id
                                                     for t in rows[i-1]["diagnostics"].get("tags", []))) else 0.0
            no_tag_duration += dt
            longest_no_tag = max(longest_no_tag, current_no_tag)
        else:
            current_no_tag = 0.0
        if not supported and failure_start is None:
            failure_start = now
        # Evidence return = a valid tag the solver was actually given, or landmark RANSAC passing its
        # inlier/coverage gates (min_landmark_inliers). A withheld tag is never evidence.
        available = (tag and not d.get("tag_observations_withheld", False)) or (d.get("pose_gates", {}).get("ransac_attempted")
                                                                                and d.get("pose_gates", {}).get("landmark_gate_reason") is None)
        if failure_start is not None and available and evidence_return is None:
            evidence_return = now
        if supported:
            if failure_start is not None and now is not None:
                if acquired:
                    failures.append(max(0.0, now-failure_start))
                    if evidence_return is not None:
                        recoveries.append(max(0.0, now-evidence_return))
                else:
                    startup_acquisition = max(0.0, now-failure_start)
            elif not acquired:
                startup_acquisition = 0.0
            acquired = True
            failure_start = evidence_return = None
        reference = d.get("reference_panel_pixels")
        if supported and reference is not None and row.get("raw_panel_pixels") is not None:
            errors.extend(np.linalg.norm(np.asarray(reference)-row["raw_panel_pixels"], axis=1).tolist())
    if failure_start is not None and acquired and rows[-1]["source_time_s"] is not None:
        failures.append(max(0.0, rows[-1]["source_time_s"]-failure_start))
    fps = ((len(displays)-1)/(displays[-1]-displays[0])
           if len(displays) > 1 and displays[-1] > displays[0] else None)
    return {"run": str(Path(directory).resolve()), "frames": len(rows), "duration_s": sum(intervals),
            "actual_pose_sources": dict(sources), "landmark_gate_reasons": dict(gates),
            "valid_pose_fraction": (len(rows)-sources["none"])/len(rows) if rows else 0.0,
            "no_tag_frames": no_tag_frames, "valid_without_tag_fraction": no_tag_valid/no_tag_frames if no_tag_frames else None,
            "no_tag_duration_s": no_tag_duration, "longest_no_tag_interval_s": longest_no_tag,
            "tag_withheld": any(r["diagnostics"].get("tag_observations_withheld", False) for r in rows),
            "lk_survival_fraction": lk_retained/lk_input if lk_input else None,
            "lk_rejections": dict(lk_rejections), "median_track_age_frames": percentile(ages, 50),
            "mapped_median": percentile(mapped, 50), "mapped_p95": percentile(mapped, 95),
            "map_association_observation_totals": dict(associations), "descriptor_rejections": dict(descriptor_rejections),
            "ransac_inliers_median": percentile(ransac_counts, 50), "ransac_inliers_p95": percentile(ransac_counts, 95),
            "ransac_occupied_cells_median": percentile(coverage_counts, 50), "final_landmark_inliers_median": percentile(final_inliers, 50),
            "stage_p95_ms": {k: percentile(v, 95) for k, v in stages.items()},
            "map_quality": next((r["diagnostics"]["map_quality"] for r in rows if r["diagnostics"].get("map_quality")), None),
            "pose_prior_expirations": sum(bool(r["diagnostics"].get("timestamp_gap_reset")) for r in rows),
            "feature_tracker_resets": sum(bool(r["diagnostics"].get("feature_tracker_reset")) for r in rows),
            "panel_reference_median_px": percentile(errors, 50), "panel_reference_p95_px": percentile(errors, 95),
            "reference_corner_samples": len(errors), "reference_note": "tag-reference agreement, not ground truth",
            "processing_p95_ms": percentile(processing, 95),
            "processing_throughput_fps": 1000/float(np.mean(processing)) if processing and np.mean(processing) > 0 else None,
            "capture_read_p95_ms": percentile(capture, 95),
            "acquisition_to_display_p95_ms": percentile(latency, 95), "displayed_fps": fps,
            "read_to_display_p95_ms": percentile(read_latency, 95),
            "occlusion_states": dict(occlusion_states),
            "occlusion_active_fraction": occlusion_states["metric depth active (prototype)"]/len(rows) if rows else 0.0,
            "depth_inference_p95_ms": percentile(inference_times, 95),
            "gpu_peak_allocated_mb": max(gpu_memory, default=None), "depth_input_sizes": sorted(depth_sizes),
            "latency_note": "software acquisition-to-display submission; excludes unknown sensor/exposure buffering",
            "acquired": acquired, "startup_acquisition_s": startup_acquisition,
            "longest_loss_s": max(failures, default=0.0), "recovery_after_available_evidence_max_s": max(recoveries, default=0.0)}


def window_metrics(directory, windows):
    """Metrics restricted to each [(start_s, end_s), ...] window of a run's frames.jsonl.

    Only meaningful for a run made with matching --withhold-tag-windows: valid fraction, time to
    first lost pose, longest continuous supported run, reference corner error (supported frames
    only), and recovery time after the window ends (first supported frame once the tag is no
    longer withheld). Does not count frozen/predicted poses: "supported" is raw_pose.valid, never
    the smoothed render pose, which the pipeline never carries forward once raw is invalid.
    """
    rows = read_rows(directory)

    def supported(row):
        pose = row["raw_pose"]
        return bool(pose["valid"] and row["actual_pose_source"] in ("tag", "joint", "landmark")
                   and pose.get("tag_inliers", 0)+pose.get("landmark_inliers", 0) >= 4)

    results = {}
    for start, end in windows:
        in_window = [r for r in rows if r["source_time_s"] is not None and start <= r["source_time_s"] <= end]
        ok = [supported(r) for r in in_window]
        time_to_first_lost = next((r["source_time_s"]-start for r, o in zip(in_window, ok) if not o), None)
        longest_run, run_start = 0.0, None
        for r, o in zip(in_window, ok):
            if o:
                run_start = r["source_time_s"] if run_start is None else run_start
                longest_run = max(longest_run, r["source_time_s"]-run_start)
            else:
                run_start = None
        errors = []
        for r, o in zip(in_window, ok):
            reference = r["diagnostics"].get("reference_panel_pixels")
            if o and reference is not None and r.get("raw_panel_pixels") is not None:
                errors.extend(np.linalg.norm(np.asarray(reference)-r["raw_panel_pixels"], axis=1).tolist())
        after = [r for r in rows if r["source_time_s"] is not None and r["source_time_s"] > end
                and not r["diagnostics"].get("tag_observations_withheld", False)]
        recovery = next((r["source_time_s"]-end for r in after if supported(r)), None)
        results[f"{start}:{end}"] = {"count": len(in_window), "valid_fraction": (sum(ok)/len(ok)) if ok else None,
                                     "time_to_first_lost_s": time_to_first_lost, "longest_supported_run_s": longest_run,
                                     "reference_median_px": percentile(errors, 50), "reference_p95_px": percentile(errors, 95),
                                     "reference_samples": len(errors), "recovery_after_window_s": recovery}
    return results


def pick_windows(directory, lengths=(5, 10, 20), per_length=3, lead_s=2.0, min_tag_fraction=0.8):
    """Withhold windows spread over a tag_only run, chosen where the tag is valid (so reference error is
    measurable), each preceded by lead_s of valid tag tracking. Same-length windows don't overlap. Not chosen by landmark success."""
    rows = [r for r in read_rows(directory) if r["source_time_s"] is not None]
    t = np.array([r["source_time_s"] for r in rows])
    ok = np.array([bool(r["raw_pose"]["valid"] and r["actual_pose_source"] == "tag") for r in rows])
    frac = lambda a, b: ok[(t >= a) & (t <= b)].mean() if ((t >= a) & (t <= b)).any() else 0.0
    chosen = []
    for length in lengths:
        same = []  # windows of different lengths may overlap (metrics are independent); same-length ones may not
        for k in range(per_length):
            target = t[0]+lead_s+(t[-1]-t[0]-lead_s-length)*(k+.5)/per_length
            for start in t[t >= target]:
                end = start+length
                if end > t[-1]:
                    break
                if (frac(start-lead_s, start) >= .95 and frac(start, end) >= min_tag_fraction
                        and all(end <= a or start >= b for a, b in same)):
                    same.append((float(start), float(end)))
                    break
        chosen += same
    return sorted(chosen)


def hand_off(config_path, windows, output, max_frames=None):
    """Replay the same clip over the same withhold windows: joint mode (tag seeds outside windows)
    versus the landmark_only control, each with guided matching off then on."""
    from .pipeline import run
    cfg = load_config(config_path)
    reports = {}
    for mode in ("joint", "landmark_only"):
        for label, guided in (("baseline", False), ("guided", True)):
            variant = replace(cfg, pose=replace(cfg.pose, mode=mode),
                              features=replace(cfg.features, guided_matching=replace(cfg.features.guided_matching, enabled=guided)))
            path = run(variant, max_frames=max_frames, withhold_tag_windows=windows)
            reports[f"{mode}/{label}"] = {"run": str(path), "pose_mode": mode, "guided": guided,
                                          "sources_inside_windows": window_sources(path, windows),
                                          "windows": window_metrics(path, windows)}
    report = {"windows": [list(w) for w in windows], "reports": reports}
    write_json(output, report)
    return report


def window_sources(directory, windows):
    """Counts of actual_pose_source inside vs outside the windows (inside must never contain tag/joint)."""
    inside, outside = Counter(), Counter()
    for r in read_rows(directory):
        t = r["source_time_s"]
        (inside if t is not None and any(a <= t <= b for a, b in windows) else outside)[r["actual_pose_source"]] += 1
    return {"inside": dict(inside), "outside": dict(outside)}


def fingerprint(config):
    return {"calibration_sha256": hashlib.sha256(Path(config.camera.calibration_path).read_bytes()).hexdigest(),
            "map_sha256": hashlib.sha256(Path(config.mapping.map_path).read_bytes()).hexdigest(),
            "tag_size_m": config.tags.size_m, "anchor_id": config.tags.anchor_id,
            "image_size": [config.camera.width, config.camera.height]}


TRACKING_KEYS = ("tags", "features", "mapping", "pose", "filter", "uncertainty", "seed")


def provenance(directory):
    """Full tracking configuration (not camera source, display or logging) plus the code hash recorded for the run."""
    config = asdict(load_config(Path(directory)/"config.json"))
    metadata = json.loads((Path(directory)/"metadata.json").read_text(encoding="utf-8")) if (Path(directory)/"metadata.json").exists() else {}
    return {"tracking_config": {k: config[k] for k in TRACKING_KEYS}, "code_sha256": metadata.get("code_sha256"), "current_code_sha256": code_sha256()}


def tracking_acceptance(controlled, physical, soak):
    summaries = {name: summarize(path) for name, path in (("controlled", controlled), ("physical", physical), ("soak", soak))}
    c, p, s = (summaries[k] for k in ("controlled", "physical", "soak"))
    fingerprints = [fingerprint(load_config(Path(path)/"config.json")) for path in (controlled, physical, soak)]
    provenances = [provenance(path) for path in (controlled, physical, soak)]
    checks = {"same_camera_tag_map": all(f == fingerprints[0] for f in fingerprints),
              "same_tracking_config": all(x["tracking_config"] == provenances[0]["tracking_config"] for x in provenances),
              "same_code": provenances[0]["code_sha256"] is not None and all(x["code_sha256"] == provenances[0]["code_sha256"] for x in provenances),
              "acquired_before_recovery_claims": all(r["acquired"] for r in summaries.values()),
              "controlled_withholding": c["tag_withheld"],
              "reference_samples": c["reference_corner_samples"] >= 400,
              "reference_median_under_5px": c["panel_reference_median_px"] is not None and c["panel_reference_median_px"] < 5,
              "reference_p95_under_10px": c["panel_reference_p95_px"] is not None and c["panel_reference_p95_px"] < 10,
              "physical_not_simulated": not p["tag_withheld"] and not s["tag_withheld"],
              "physical_tag_hidden_120s": p["longest_no_tag_interval_s"] >= 120,
              "soak_tag_hidden_300s": s["longest_no_tag_interval_s"] >= 300,
              "supported_tracking_95_percent": all((r["valid_without_tag_fraction"] or 0) >= .95 for r in summaries.values()),
              "recovery_within_1s": all(r["recovery_after_available_evidence_max_s"] <= 1 for r in summaries.values()),
              "live_20fps": all((r["displayed_fps"] or 0) >= 20 for r in (p, s)),
              "live_latency_under_100ms": all(r["read_to_display_p95_ms"] is not None and r["read_to_display_p95_ms"] <= 100 for r in (p, s))}
    return {"schema_version": 1, "kind": "tracking_acceptance", "passed": all(checks.values()), "checks": checks,
            "fingerprint": fingerprints[0], "provenance": provenances, "runs": summaries,
            "limitations": "Physical scenarios require the operator to keep the tag fixed, cover it, move the camera, and retain mapped static coverage. This is not absolute pose ground truth."}


def compare(recording, config_path, output, max_frames=None):
    from .pipeline import run
    cfg = load_config(config_path or Path(recording)/"config.json")
    if not cfg.features.exclude_tag_regions:
        raise ValueError("comparison requires features.exclude_tag_regions=true to prevent tag texture leakage")
    base = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(Path(recording).resolve()/"input.avi"),
                                       timestamps_path=str(Path(recording).resolve()/"frames.jsonl")),
                   mapping=replace(cfg.mapping, mode="localize"), display=replace(cfg.display, enabled=False),
                   tags=replace(cfg.tags, enabled=True),
                   logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))
    variants = {"tag_only": replace(base, pose=replace(base.pose, mode="tag_only")),
                "lk_periodic": replace(base, pose=replace(base.pose, mode="landmark_only"), features=replace(base.features, redetect_every_n_frames=15, lk=replace(base.features.lk, enabled=True))),
                "orb_only": replace(base, pose=replace(base.pose, mode="landmark_only"), features=replace(base.features, redetect_every_n_frames=1, lk=replace(base.features.lk, enabled=False))),
                "lk_orb": replace(base, pose=replace(base.pose, mode="landmark_only"), features=replace(base.features, redetect_every_n_frames=1, lk=replace(base.features.lk, enabled=True)))}
    reports = {}
    for name, config in variants.items():
        path = run(config, max_frames=max_frames, withhold_tag=name != "tag_only")
        reports[name] = summarize(path)
    qualifying = [name for name in ("lk_periodic", "orb_only", "lk_orb")
                  if (reports[name]["valid_without_tag_fraction"] or 0) >= .95
                  and reports[name]["reference_corner_samples"] >= 400
                  and reports[name]["panel_reference_median_px"] is not None and reports[name]["panel_reference_median_px"] < 5
                  and reports[name]["panel_reference_p95_px"] is not None and reports[name]["panel_reference_p95_px"] < 10
                  and (reports[name]["processing_throughput_fps"] or 0) >= 20]
    winner = max(qualifying, key=lambda name: (reports[name]["valid_without_tag_fraction"], reports[name]["processing_throughput_fps"])) if qualifying else None
    joint = replace(variants[winner or "lk_periodic"], pose=replace(base.pose, mode="joint"))
    reports["joint"] = summarize(run(joint, max_frames=max_frames))
    report = {"variants": reports, "selected_landmark_variant": winner,
              "selection_note": "headless shortlist only; live performance and physical acceptance must pass separately",
              "comparison_scope": "whole-pipeline: variants differ in redetection cadence and LK, so recovery numbers mix cadence with the recovery policy"}
    write_json(output, report)
    if winner:
        write_json(Path(output).with_suffix(".recommended-config.json"), asdict(replace(cfg, features=joint.features)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize", help="Explain observations, gates, timing and pose availability")
    summary.add_argument("runs", nargs="+")
    summary.add_argument("--output")
    comparison = commands.add_parser("compare", help="Replay four tracking ablations and the resulting joint configuration")
    comparison.add_argument("recording")
    comparison.add_argument("--config")
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--max-frames", type=int)
    assess = commands.add_parser("assess", help="Check controlled, physical 2-minute, and 5-minute soak runs")
    for flag in ("controlled", "physical", "soak", "output"):
        assess.add_argument("--"+flag, required=True)
    handoff = commands.add_parser("handoff", help="Replay a clip with tag-withhold windows, guided matching off then on")
    handoff.add_argument("config")
    handoff.add_argument("--windows", required=True, help="start:end,start:end,... seconds, or picked:<tag_only_run_dir>")
    handoff.add_argument("--output", required=True)
    handoff.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.command == "summarize":
        report = [summarize(path) for path in args.runs]
        if args.output:
            write_json(args.output, report)
    elif args.command == "compare":
        report = compare(args.recording, args.config, args.output, args.max_frames)
    elif args.command == "handoff":
        windows = (pick_windows(args.windows[7:]) if args.windows.startswith("picked:")
                   else [tuple(float(v) for v in part.split(":")) for part in args.windows.split(",")])
        report = hand_off(args.config, windows, args.output, args.max_frames)
    else:
        report = tracking_acceptance(args.controlled, args.physical, args.soak)
        write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
