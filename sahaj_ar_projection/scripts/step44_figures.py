"""Step 44 images: the labelled F-occ frames with the largest miss rate under Line (A+L); baseline vs line hidden mask
(red) over the photo, labelled FRONT silhouette outline (yellow). System python3 (has matplotlib)."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ar_projection.config import load_config
from ar_projection.geometry import panel_points
from ar_projection.occlusion import panel_layer
R, L, OUT = Path("runs/step44-baseline/F-occ"), Path("runs/step41-labels"), Path("runs/step44-images")
OUT.mkdir(exist_ok=True)
d = json.loads(Path("runs/step44-baseline.json").read_text())["step41_labels"]["line"]["per_frame"]
worst = [p["frame"] for p in sorted((p for p in d if p["miss"] is not None), key=lambda p: -p["miss"])[:8]]
cfg = load_config("config-logitech-far.json")
panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
for n in worst:
    img = cv2.cvtColor(cv2.imread(str(L/f"{n:06d}-image.png")), cv2.COLOR_BGR2RGB)
    s = json.loads((R/f"{n:06d}.json").read_text())
    cam = SimpleNamespace(K=np.array(s["K"]).reshape(3, 3), distortion=np.zeros(5), size=(img.shape[1], img.shape[0]))
    T = np.array(s["render_T_CW"]); lab = cv2.imread(str(L/f"{n:06d}-labels.png"), 0)
    z0 = panel_layer(T, panel, cam)[2]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, tag in zip(ax, ("baseline", "line")):
        Z = np.load(R/f"{n:06d}-scene-{tag}.npy")
        hid = (z0 > 0) & (Z < z0-np.maximum(.1, .1*z0))
        o = img.copy(); o[hid] = (0.5*o[hid]+0.5*np.array([255, 0, 0])).astype(np.uint8)
        cv2.drawContours(o, cv2.findContours((lab == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 0), 1)
        cv2.drawContours(o, cv2.findContours((z0 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 1)
        a.imshow(o); a.axis("off"); a.set_title(f"{tag}: hidden {hid.sum()/max((z0 > 0).sum(), 1):.2f}", fontsize=9)
    fig.suptitle(f"F-occ frame {n}: red = hidden by method, yellow = labelled FRONT occluder, green = panel", fontsize=9)
    fig.savefig(OUT/f"F-occ_{n:06d}.png", dpi=70, bbox_inches="tight"); plt.close(fig)
print(worst)
