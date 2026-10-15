"""Legacy arguments are explicit central-configuration overrides."""
import argparse
import logging
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
    config = load_config(args.config, overrides)
    if args.save_map and config.mapping.mode != "build":
        parser.error("--save-map requires --mapping-mode build")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(config, args.save_map, args.replace_map, args.max_frames)
