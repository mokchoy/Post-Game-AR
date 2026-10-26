"""Re-render the 1x4 depth inspection (rectified | real depth | virtual panel Z | mask) using an
affine (scale+offset) fit of raw model output to true distance, instead of the existing
proportional (scale-only) fit -- the Step 30 ablation the numbers argued for. Per frame: replay
the pipeline to that frame to get the same landmark anchors the live scale estimator would see,
fit raw = a + b*Z by least squares, invert to real_m = (raw-a)/b, recompute the mask from that.
Run via .venv-depth.
"""
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.geometry import depths, panel_points
from ar_projection.occlusion import MetricDepth, panel_layer
from ar_projection.pipeline import Pipeline

REPO = "/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2"
CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_vits.pth"
MAP_PATH = "data/logitech-landmark-map-occ.npz"
MARGIN_M = .02

TARGETS = [("V2", "runs/20261024T052253-e8a2ad05", [72, 441]),
           ("V3", "runs/20261024T054241-016203dd", [26]),
           ("V4", "runs/20261024T052603-9e9fecca", [1276])]


def colorize(depth_m, valid, vmin, vmax):
    norm = np.zeros(depth_m.shape, np.uint8)
    norm[valid] = np.clip((vmax-depth_m[valid])/max(vmax-vmin, 1e-6)*255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    heat[~valid] = 40
    return heat


def outline(img, where):
    out = img.copy()
    contours, _ = cv2.findContours(where.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def anchors(result, landmark_map, features, raw):
    lookup = {int(id): i for i, id in enumerate(landmark_map.ids)}
    pairs = []
    for f in features:
        if f.landmark_id in lookup:
            x, y = np.rint(f.pixel).astype(int)
            if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0] and np.isfinite(raw[y, x]) and raw[y, x] > 0:
                z_true = depths(result.raw_pose.T_CW, landmark_map.points[lookup[f.landmark_id]][None])[0]
                if z_true > 0:
                    pairs.append((float(z_true), float(raw[y, x])))
    return pairs


def inspect(model, clip_label, recording, index):
    rec = Path(recording).resolve()
    cfg = load_config("config-logitech.json")
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(MAP_PATH).resolve())),
                  display=replace(cfg.display, enabled=False))
    capture = Capture(cfg.camera)
    packet = capture.read()
    pipeline = Pipeline(cfg, packet.camera, background_mapping=False)
    result = None
    while packet is not None and packet.index <= index:
        result = pipeline.process(packet, withhold_tag=False)
        if packet.index == index:
            break
        packet = capture.read()
    capture.close()

    raw, _ = model.predict(packet.bgr)
    pairs = anchors(result, pipeline.map, pipeline.last_features, raw)
    z_anchor = np.array([p[0] for p in pairs])
    r_anchor = np.array([p[1] for p in pairs])
    b, a = np.polyfit(z_anchor, r_anchor, 1)
    real_affine = (raw-a)/b
    proportional_scale = float(np.median(z_anchor/r_anchor))
    real_proportional = raw*proportional_scale

    panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
    _, visible, z = panel_layer(result.render_pose, panel, packet.camera, real_affine, margin_m=MARGIN_M)
    where = z > 0
    mask_img = np.where(visible[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)

    vmin, vmax = float(real_affine.min()), float(real_affine.max())
    real_heat = label(outline(colorize(real_affine, np.ones_like(real_affine, bool), vmin, vmax), where),
                      f"AFFINE real depth (n={len(pairs)} anchors, raw={b:.2f}*Z+{a:.2f}), range {vmin:.2f}-{vmax:.2f}m")
    panel_heat = label(colorize(z, where, vmin, vmax), f"virtual panel Z, median={float(np.median(z[where])):.2f}m")
    mask_label = label(mask_img, f"mask (white=visible), occluded={float(np.mean(~visible[where])):.1%}" if np.any(where) else "mask")
    row = np.hstack([label(packet.bgr, "rectified"), real_heat, panel_heat, mask_label])
    out = f"/tmp/affine_{clip_label}_{index}.png"
    cv2.imwrite(out, row)

    old_err = abs(float(np.median(real_proportional[where]))-float(np.median(z[where])))
    new_err = abs(float(np.median(real_affine[where]))-float(np.median(z[where])))
    print(f"{clip_label} frame {index}: anchors={len(pairs)}, "
          f"proportional median real={np.median(real_proportional[where]):.3f}m (err {old_err:.3f}m) -> "
          f"affine median real={np.median(real_affine[where]):.3f}m (err {new_err:.3f}m); "
          f"occluded fraction={float(np.mean(~visible[where])):.3f}")
    return out


def main():
    model = MetricDepth(REPO, CHECKPOINT, input_size=252)
    for clip_label, recording, indices in TARGETS:
        for index in indices:
            print(inspect(model, clip_label, recording, index))


if __name__ == "__main__":
    main()
