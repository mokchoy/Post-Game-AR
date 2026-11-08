"""Legacy arguments are explicit central-configuration overrides."""
import argparse
import json
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
    parser.add_argument("--occlusion", action="store_true", help="Depth occlusion from the config 'occlusion' block (Step 45; key o toggles it live)")
    parser.add_argument("--depth-model", choices=("relative", "metric"), help="Override occlusion.model")
    parser.add_argument("--allow-unaccepted-depth", action="store_true", help="Skip the tracking-acceptance report check (logged as bypassed)")
    parser.add_argument("--panel-video", help="Play this video on the panel instead of the checkerboard (needs the occlusion path)")
    parser.add_argument("--array-script", help="Replay aid for config.array: 'frame:action:value,...' e.g. 50:yaw:1,120:scale:1 (actions game yaw scale play stop reset)")
    parser.add_argument("--no-gestures", action="store_true", help="Run the array without the gesture thread even if gestures.enabled")
    parser.add_argument("--gesture-camera", type=int, help="Second (laptop) camera index for the gesture thread; overrides gestures.camera_index and enables gestures")
    parser.add_argument("--save-video", action="store_true", help="Write the annotated frames (projections included) to RUN/annotated.*, also in --replay-run")
    parser.add_argument("--no-smoothing", action="store_true", help="Override filter.enabled=false (pose smoothing off)")
    parser.add_argument("--depth-video", action="store_true", help="With --save-video, also write RUN/depth.avi (the depth the occluder uses: stored background if the prior is on)")
    parser.add_argument("--clean", action="store_true", help="Annotated video/display shows only the camera image and projection (display.clean)")
    parser.add_argument("--no-background-prior", action="store_true", help="Override occlusion.background_prior.enabled=false")
    parser.add_argument("--no-occlusion", action="store_true", help="Step 52: never load the depth model (live recording at full rate; add occlusion later in --replay-run)")
    parser.add_argument("--minimal-hud", action="store_true", help="Step 52: recording overlay (features, tag, ellipses, one status line) instead of the debug text (display.minimal)")
    parser.add_argument("--replay-actions", action="store_true", help="Step 52: with --replay-run, re-apply the array actions (gesture/key/script) logged in that run, at the same frames")
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
        if args.gesture_camera is None:
            overrides["gestures.enabled"] = False     # a replay never opens the laptop camera; logged actions come from --replay-actions
        overrides.update({"camera.source_type": "video", "camera.video_path": str(recording/"input.avi"),
                          "camera.timestamps_path": str(recording/"frames.jsonl"),
                          "logging.save_input_video": False, "logging.save_annotated_video": False})
    if args.timestamps:
        if not args.video:
            parser.error("--timestamps requires --video")
        overrides["camera.timestamps_path"] = args.timestamps
    if args.save_video:
        overrides.update({"logging.enabled": True, "logging.save_annotated_video": True})
    if args.clean:
        overrides["display.clean"] = True
    if args.minimal_hud:
        overrides["display.minimal"] = True
    if args.no_background_prior:
        overrides["occlusion.background_prior.enabled"] = False
    if args.no_smoothing:
        overrides["filter.enabled"] = False
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
    if args.gesture_camera is not None:
        overrides.update({"gestures.camera_index": args.gesture_camera, "gestures.enabled": True})
    if args.no_gestures:
        overrides["gestures.enabled"] = False
    config = load_config(config_path, overrides)
    if config.array.enabled and not config.camera.undistort:
        parser.error("array.enabled requires camera.undistort=true")
    if (args.withhold_tag or withhold_tag_windows) and (config.mapping.mode != "localize" or config.pose.mode == "tag_only"):
        parser.error("--withhold-tag/--withhold-tag-windows requires loaded-map localization in joint or landmark_only mode")
    if args.save_map and config.mapping.mode != "build":
        parser.error("--save-map requires --mapping-mode build")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Source %s index %s backend %s; calibration %s; pose %s, mapping %s",
                 config.camera.source_type, config.camera.index, config.camera.backend,
                 config.camera.calibration_path, config.pose.mode, config.mapping.mode)
    occlusion = None
    if args.no_occlusion and args.occlusion:
        parser.error("--occlusion and --no-occlusion are exclusive")
    if not args.no_occlusion and (args.occlusion or (config.occlusion.enabled and config.mapping.mode == "localize") or args.depth_repo or args.depth_weights or args.occlusion_output):
        block = config.occlusion
        model_name = args.depth_model or block.model
        repo, weights = args.depth_repo or block.repo, args.depth_weights or block.checkpoint
        if args.occlusion and not block.enabled:
            logging.info("--occlusion given; config occlusion.enabled is false, using the block's settings anyway")
        if not (repo and weights):
            parser.error("depth requires a repo and checkpoint (--depth-repo/--depth-weights or the config occlusion block)")
        if not (args.tracking_report or args.allow_unaccepted_depth or not block.require_tracking_acceptance):
            parser.error("depth requires --tracking-report, or --allow-unaccepted-depth to bypass it (logged)")
        if config.mapping.mode != "localize" or not config.camera.undistort:
            parser.error("depth requires loaded-map localization and camera.undistort=true")
        from .occlusion import DepthPrototype, MetricDepth, RelativeDepth, RelativeLineOcclusion, require_tracking_acceptance
        if (args.allow_unaccepted_depth or not block.require_tracking_acceptance) and not args.tracking_report:
            logging.warning("tracking acceptance BYPASSED (--allow-unaccepted-depth): depth masks are untested for this panel, tag limit and map")
            acceptance = "bypassed"
        else:
            require_tracking_acceptance(args.tracking_report, config)
            acceptance = "passed"
        if model_name == "relative":
            occlusion = RelativeLineOcclusion(RelativeDepth(repo, weights, block.input_size or args.depth_input_size), args.occlusion_output, block.margin_floor_m,
                                              block.margin_fraction, block.line_reference, block.line_hold_frames, weights, acceptance,
                                              block.reference, block.plane_extent_m, block.plane_min_span, block.plane_min_px, block.background_prior)
            logging.info("Occlusion: relative model %s sha256 %s", occlusion.run_metadata["checkpoint"], occlusion.run_metadata["checkpoint_sha256"])
        else:
            if config.array.enabled:
                parser.error("array.enabled supports the relative depth model only")
            occlusion = DepthPrototype(MetricDepth(repo, weights, args.depth_input_size), args.occlusion_output)
    if args.depth_video and occlusion is not None and hasattr(occlusion, "keep_view"):
        occlusion.keep_view = True
    script = None
    if args.array_script:
        script = {}
        for item in args.array_script.split(","):
            frame, action, value = item.split(":")
            script.setdefault(int(frame), []).append((action, float(value)))
    if args.replay_actions:
        if not args.replay_run:
            parser.error("--replay-actions requires --replay-run")
        script = script or {}
        n = 0
        for line in (Path(args.replay_run)/"frames.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for act in ((row.get("diagnostics") or {}).get("array") or {}).get("actions") or ():
                script.setdefault(int(row["frame_index"]), []).append((act["action"], float(act["value"])))
                n += 1
        logging.info("--replay-actions: %d logged array actions re-applied", n)
    if args.panel_video and not config.array.enabled:
        if occlusion is None:
            parser.error("--panel-video needs --occlusion")
        from .occlusion import set_panel_video
        set_panel_video(args.panel_video)
    run(config, args.save_map, args.replace_map, args.max_frames, withhold_tag=args.withhold_tag,
       withhold_tag_windows=withhold_tag_windows, occlusion=occlusion,
       left_video=args.panel_video if config.array.enabled else None, array_script=script)
