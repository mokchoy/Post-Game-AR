"""Step 31: contact sheets from scripts/step31_masks.py's export (image | mask | virtual-Z per
frame, the existing export format -- no new rendering added). One sheet per clip/variant,
6 frames evenly spaced across the export. Writes runs/step31-masks/contact-<label>.png."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

OUT_ROOT = Path("runs/step31-masks")
FRAMES_PER_SHEET = 6


def depth_heatmap(z):
    """Near=red (hot), far=blue (cold)."""
    valid = z > 0
    if not np.any(valid):
        return np.zeros((*z.shape, 3), np.uint8)
    lo, hi = z[valid].min(), z[valid].max()
    norm = np.zeros(z.shape, np.uint8)
    norm[valid] = np.clip((hi-z[valid])/max(hi-lo, 1e-6)*255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    return heat


def sheet(label):
    directory = OUT_ROOT/label
    indices = sorted(int(p.stem) for p in directory.glob("[0-9]*.json"))
    if not indices:
        return None
    picks = [indices[i*(len(indices)-1)//(FRAMES_PER_SHEET-1)] for i in range(FRAMES_PER_SHEET)] if len(indices) >= FRAMES_PER_SHEET else indices
    rows = []
    for idx in picks:
        name = f"{idx:06d}"
        image = cv2.imread(str(directory/f"{name}-image.png"))
        mask = cv2.imread(str(directory/f"{name}.png"))
        z = np.load(directory/f"{name}-virtual-depth.npy")
        rows.append(np.hstack([image, mask, depth_heatmap(z)]))
    width = min(r.shape[1] for r in rows)
    grid = np.vstack([r[:, :width] for r in rows])
    out = OUT_ROOT/f"contact-{label}.png"
    cv2.imwrite(str(out), grid)
    return out


def main():
    labels = sorted(p.name for p in OUT_ROOT.iterdir() if p.is_dir())
    for label in labels:
        path = sheet(label)
        print(label, "->", path)


if __name__ == "__main__":
    main()
