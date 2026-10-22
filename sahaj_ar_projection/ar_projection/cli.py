"""Legacy arguments are explicit central-configuration overrides."""
import argparse
import logging
from pathlib import Path
from .config import DEFAULT_PATH, load_config
from .pipeline import run


def self_check():
    import numpy as np
    from .geometry import inverse, panel_points, project, transform
    from .types import CameraModel
    panel = panel_points(.2, .12, .15, 90)
    assert np.allclose(panel.mean(0), [0, .15, .06])
    assert np.allclose(panel[:, 2], [0, 0, .12, .12])
    T = transform([.2, .1, .3], [0, 0, 1])
    assert np.allclose(inverse(T)@T, np.eye(4))
    camera = CameraModel(np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]]), np.zeros(5), (640, 480), {})
    assert np.isfinite(project(T, panel, camera)).all()
    print("Projection geometry OK")


def main():
    parser = argparse.ArgumentParser(description="Bounded tag/landmark AR tracker (Python 3.10+)")
    parser.add_argument("--config", default=str(DEFAULT_PATH))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--demo-approx", action="store_true")
    parser.add_argument("--save-map", action="store_true", help="Save once bootstrap meets all criteria")
    parser.add_argument("--replace-map", action="store_true", help="Permit replacement on S or --save-map")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--record-input", action="store_true", help="Record lossless input and authoritative frame timestamps")
    parser.add_argument("--replay-run", help="Replay RUN/input.avi with RUN/frames.jsonl acquisition timestamps")
    parser.add_argument("--timestamps", help="Frame JSONL timestamps for --video")
    parser.add_argument("--withhold-tag", action="store_true", help="Mask detected tags from features and withhold tag observations from the solver")
    parser.add_argument("--withhold-tag-windows", help="Comma-separated start:end second ranges; withhold the tag only inside these windows (e.g. 10:15,30:40)")
    parser.add_argument("--depth-repo", help="Optional DA-V2 author checkout; requires passed tracking acceptance")
    parser.add_argument("--depth-weights", help="DA-V2 Small Hypersim metric checkpoint")
    parser.add_argument("--depth-input-size", type=int, default=252, choices=(252, 392, 518))
    parser.add_argument("--tracking-report", help="Passed experiments assess report for this camera/tag/map")
    parser.add_argument("--occlusion-output", help="New directory for frame-aligned visibility masks and virtual depth")
    arguments = {"camera": (int, "camera.index"), "calibration": (str, "camera.calibration_path"), "video": (str, "camera.video_path"),
                 "backend": (str, "camera.backend"), "tag-size": (float, "tags.size_m"), "width": (float, "panel.width_m"),
                 "height": (float, "panel.height_m"), "elevation": (float, "panel.offset_y_m"), "tilt": (float, "panel.tilt_x_deg"),
                 "pose-mode": (str, "pose.mode"), "mapping-mode": (str, "mapping.mode"), "map": (str, "mapping.map_path")}
    for flag, (kind, _) in arguments.items():
        parser.add_argument("--"+flag, type=kind)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    overrides = {label: getattr(args, flag.replace("-", "_")) for flag, (_, label) in arguments.items() if getattr(args, flag.replace("-", "_")) is not None}
    if args.video is not None:
        overrides["camera.source_type"] = "video"
    elif args.camera is not None:
        overrides["camera.source_type"] = "camera"
    if args.headless:
        overrides["display.enabled"] = False
    if args.demo_approx:
        overrides["camera.demo_approx"] = True
        overrides.setdefault("pose.mode", "tag_only")
        overrides.setdefault("mapping.mode", "off")
        overrides["uncertainty.enabled"] = False
    config_path = args.config
    if args.replay_run:
        recording = Path(args.replay_run).resolve()
        if args.video or args.timestamps:
            parser.error("--replay-run cannot be combined with --video or --timestamps")
        if args.config == str(DEFAULT_PATH):
            config_path = recording/"config.json"
        overrides.update({"camera.source_type": "video", "camera.video_path": str(recording/"input.avi"),
                          "camera.timestamps_path": str(recording/"frames.jsonl"),
                          "logging.save_input_video": False, "logging.save_annotated_video": False})
    if args.timestamps:
        if not args.video:
            parser.error("--timestamps requires --video")
        overrides["camera.timestamps_path"] = args.timestamps
    if args.record_input:
        overrides.update({"logging.enabled": True, "logging.save_input_video": True})
    withhold_tag_windows = None
    if args.withhold_tag_windows:
        try:
            withhold_tag_windows = [tuple(float(v) for v in part.split(":")) for part in args.withhold_tag_windows.split(",")]
            if any(len(w) != 2 or w[0] >= w[1] for w in withhold_tag_windows):
                raise ValueError
        except ValueError:
            parser.error("--withhold-tag-windows must be start:end,start:end,... with start < end")
    if args.withhold_tag or withhold_tag_windows:
        overrides.update({"tags.enabled": True, "features.exclude_tag_regions": True})
    config = load_config(config_path, overrides)
    if (args.withhold_tag or withhold_tag_windows) and (config.mapping.mode != "localize" or config.pose.mode == "tag_only"):
        parser.error("--withhold-tag/--withhold-tag-windows requires loaded-map localization in joint or landmark_only mode")
    if args.save_map and config.mapping.mode != "build":
        parser.error("--save-map requires --mapping-mode build")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Source %s index %s backend %s; calibration %s; pose %s, mapping %s",
                 config.camera.source_type, config.camera.index, config.camera.backend,
                 config.camera.calibration_path, config.pose.mode, config.mapping.mode)
    occlusion = None
    if args.depth_repo or args.depth_weights or args.occlusion_output:
        if not (args.depth_repo and args.depth_weights and args.tracking_report):
            parser.error("depth requires --depth-repo, --depth-weights and --tracking-report")
        if config.mapping.mode != "localize" or not config.camera.undistort:
            parser.error("depth requires loaded-map localization and camera.undistort=true")
        from .occlusion import DepthPrototype, MetricDepth, require_tracking_acceptance
        require_tracking_acceptance(args.tracking_report, config)
        occlusion = DepthPrototype(MetricDepth(args.depth_repo, args.depth_weights, args.depth_input_size), args.occlusion_output)
    run(config, args.save_map, args.replace_map, args.max_frames, withhold_tag=args.withhold_tag,
       withhold_tag_windows=withhold_tag_windows, occlusion=occlusion)
