"""Visualize the actual Depth-Anything-V2 metric depth map behind a specific exported frame:
rectified image | full-frame scaled real depth | panel virtual-Z (same scale) | mask. Re-runs
the model on that one frame (cheap) and reuses the metric_scale already recorded in its JSON,
so this is exactly the comparison the live mask rule makes, made visible. Run via .venv-depth.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.occlusion import MetricDepth

REPO = "/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2"
CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_vits.pth"


def colorize(depth_m, valid, vmin, vmax):
    """Near=red (hot), far=blue (cold) -- inverted from TURBO's native low=blue/high=red."""
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


def inspect(model, label_name, directory, index):
    directory = Path(directory)
    name = f"{index:06d}"
    stats = json.loads((directory/f"{name}.json").read_text())
    image = cv2.imread(str(directory/f"{name}-image.png"))
    z = np.load(directory/f"{name}-virtual-depth.npy")
    mask = cv2.imread(str(directory/f"{name}.png"))
    raw, _ = model.predict(image)
    scale = stats["metric_scale"]
    real_m = raw*scale
    where = z > 0
    vmin, vmax = float(real_m.min()), float(real_m.max())  # shared scale between the two depth panels, this frame only
    real_heat = outline(colorize(real_m, np.ones_like(real_m, bool), vmin, vmax), where)
    real_heat = label(real_heat, f"real depth, scale={scale:.4f} (outline=panel), range {vmin:.2f}-{vmax:.2f} m")
    panel_heat = label(colorize(z, where, vmin, vmax), "virtual panel Z (same colour scale)")
    print(f"{label_name} frame {index}: panel pixels n={int(where.sum())}, "
          f"virtual Z median={float(np.median(z[where])):.3f} m, "
          f"real depth median={float(np.median(real_m[where])):.3f} m, "
          f"margin={stats.get('margin_m', .02)} m, occluded fraction={float(np.mean(mask[where][:, 0] < 128)):.3f}")
    row = np.hstack([label(image, "rectified"), real_heat, panel_heat, label(mask, "mask (white=visible)")])
    out = f"/tmp/depth_inspect_{label_name}_{index}.png"
    cv2.imwrite(out, row)
    return out


def main():
    model = MetricDepth(REPO, CHECKPOINT, input_size=252)
    targets = [("V2", "runs/step31-masks/V2", 72), ("V3", "runs/step31-masks/V3", 26), ("V4", "runs/step31-masks/V4", 1276)]
    for label_name, directory, index in targets:
        print(inspect(model, label_name, directory, index))


if __name__ == "__main__":
    main()
