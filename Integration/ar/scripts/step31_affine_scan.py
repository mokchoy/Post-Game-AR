"""Scan a whole clip with the affine (scale+offset) depth fit, per-frame (no temporal smoothing --
exploratory, not the live hold-phase design), and save composite images for the most-occluded
frames so a human can judge where real occlusion was captured vs missed. Run via .venv-depth.
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
MIN_ANCHORS = 12


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


def main(recording, clip_label, top_n=15):
    rec = Path(recording).resolve()
    cfg = load_config("config-logitech.json")
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(MAP_PATH).resolve())),
                  display=replace(cfg.display, enabled=False))
    capture = Capture(cfg.camera)
    packet = capture.read()
    pipeline = Pipeline(cfg, packet.camera, background_mapping=False)
    model = MetricDepth(REPO, CHECKPOINT, input_size=252)
    panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)

    results = []
    while packet is not None:
        result = pipeline.process(packet, withhold_tag=False)
        if result.raw_pose.valid and result.render_pose is not None:
            raw, _ = model.predict(packet.bgr)
            lookup = {int(id): i for i, id in enumerate(pipeline.map.ids)}
            pairs = []
            for f in pipeline.last_features:
                if f.landmark_id in lookup:
                    x, y = np.rint(f.pixel).astype(int)
                    if 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0] and np.isfinite(raw[y, x]) and raw[y, x] > 0:
                        z_true = depths(result.raw_pose.T_CW, pipeline.map.points[lookup[f.landmark_id]][None])[0]
                        if z_true > 0:
                            pairs.append((float(z_true), float(raw[y, x])))
            if len(pairs) >= MIN_ANCHORS:
                z_anchor = np.array([p[0] for p in pairs])
                r_anchor = np.array([p[1] for p in pairs])
                b, a = np.polyfit(z_anchor, r_anchor, 1)
                if b < 0:  # sanity: raw must decrease with distance, else this frame's fit is garbage
                    real_affine = (raw-a)/b
                    _, visible, z = panel_layer(result.render_pose, panel, packet.camera, real_affine, margin_m=MARGIN_M)
                    where = z > 0
                    if np.any(where):
                        frac = float(np.mean(~visible[where]))
                        results.append((frac, packet.index, packet.timestamp, packet.bgr.copy(), real_affine, z, visible, where))
        packet = capture.read()
    capture.close()

    results.sort(key=lambda r: -r[0])
    print(f"{clip_label}: {len(results)} frames with >= {MIN_ANCHORS} anchors and a valid panel projection")
    print("Top occluded frames:")
    for frac, idx, t, bgr, real_affine, z, visible, where in results[:top_n]:
        print(f"  frame {idx} (t={t:.2f}s): occluded={frac:.1%}")
    for frac, idx, t, bgr, real_affine, z, visible, where in results[:8]:
        vmin, vmax = float(real_affine.min()), float(real_affine.max())
        real_heat = label(outline(colorize(real_affine, np.ones_like(real_affine, bool), vmin, vmax), where), f"affine real depth, t={t:.2f}s")
        panel_heat = label(colorize(z, where, vmin, vmax), f"virtual panel Z, med={float(np.median(z[where])):.2f}m")
        mask_img = np.where(visible[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
        mask_img = label(mask_img, f"occluded={frac:.1%}")
        row = np.hstack([label(bgr, "rectified"), real_heat, panel_heat, mask_img])
        cv2.imwrite(f"/tmp/affine_scan_{clip_label}_{idx}.png", row)
    return results


if __name__ == "__main__":
    main("runs/20261024T052253-e8a2ad05", "V2")
