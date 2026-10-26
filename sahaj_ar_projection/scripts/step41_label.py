"""Step 41 labelling tool (OpenCV). Sahaj paints the occluder silhouette on ~20 F-occ frames.
  python scripts/step41_label.py prepare   # picks frames, extracts rectified images + panel masks (once)
  python scripts/step41_label.py label     # interactive; resumable; saves after every frame move
Labels -> runs/step41-labels/<frame>-labels.png (0 none, 1 FRONT of the panel plane, 2 BEHIND it).
Ground truth for the panel = panel polygon minus the FRONT silhouette. Frames are picked uniformly over the
clip plus extra at the biggest jumps of the anchor-fit hidden fraction (likely transitions).
Keys: left-drag paint | right-drag erase | f/b class front/behind | [ ] brush size | c clear frame
      s save+next | a save+prev | n/p next/prev without saving | q save+quit"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step40_far import CLIPS, OUT as FAR, far_config
from ar_projection.camera import Capture
from ar_projection.geometry import panel_points
from ar_projection.occlusion import panel_layer

ROOT = Path("runs/step41-labels")
N_UNIFORM, N_TRANSITION, MIN_GAP = 12, 8, 20


def pick():
    st = {int(f.stem): json.loads(f.read_text()) for f in (FAR/"anchor"/"F-occ").glob("[0-9]*.json")}
    idx = sorted(i for i, s in st.items() if "hidden_fraction" in s and s.get("render_T_CW") is not None)
    h = {i: st[i]["hidden_fraction"] for i in idx}
    chosen = []
    ok = lambda i: all(abs(i-j) >= MIN_GAP for j in chosen)
    jumps = sorted((i for i in idx if i-5 in h), key=lambda i: -abs(h[i]-h[i-5]))
    for i in jumps:
        if len(chosen) >= N_TRANSITION: break
        if ok(i): chosen.append(i)
    for i in idx[::max(1, len(idx)//N_UNIFORM)]:
        if len(chosen) >= N_UNIFORM+N_TRANSITION: break
        if ok(i): chosen.append(i)
    return sorted(chosen), st


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    chosen, st = pick()
    cfg = far_config(CLIPS["F-occ"])
    panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
    cap = Capture(cfg.camera)
    p = cap.read()
    while p is not None and chosen and p.index <= max(chosen):
        if p.index in chosen:
            cv2.imwrite(str(ROOT/f"{p.index:06d}-image.png"), p.bgr)
            _, _, z = panel_layer(np.array(st[p.index]["render_T_CW"]), panel, p.camera)
            cv2.imwrite(str(ROOT/f"{p.index:06d}-panel.png"), (z > 0).astype(np.uint8)*255)
        p = cap.read()
    cap.close()
    (ROOT/"frames.json").write_text(json.dumps(chosen))
    print("frames:", chosen)


def label():
    frames = json.loads((ROOT/"frames.json").read_text())
    state = {"i": 0, "cls": 1, "r": 12, "down": 0}
    def load(i):
        n = frames[i]
        img, panel = cv2.imread(str(ROOT/f"{n:06d}-image.png")), cv2.imread(str(ROOT/f"{n:06d}-panel.png"), 0) > 0
        lp = ROOT/f"{n:06d}-labels.png"
        lab = cv2.imread(str(lp), 0) if lp.exists() else np.zeros(panel.shape, np.uint8)
        return img, panel, lab
    img, panel, lab = load(0)
    contours = lambda: cv2.findContours(panel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    def save():
        cv2.imwrite(str(ROOT/f"{frames[state['i']]:06d}-labels.png"), lab)
    def mouse(ev, x, y, flags, _):
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN): state["down"] = 1 if ev == cv2.EVENT_LBUTTONDOWN else 2
        if ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP): state["down"] = 0
        state["xy"] = (x, y)
        if state["down"] and ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            cv2.circle(lab, (x, y), state["r"], state["cls"] if state["down"] == 1 else 0, -1)
    cv2.namedWindow("label"); cv2.setMouseCallback("label", mouse)
    def move(d, do_save):
        nonlocal img, panel, lab
        if do_save: save()
        state["i"] = min(max(state["i"]+d, 0), len(frames)-1)
        img, panel, lab = load(state["i"])
    while True:
        view = img.copy()
        for c, col in ((1, (0, 0, 255)), (2, (255, 0, 0))):
            view[lab == c] = (0.5*view[lab == c]+0.5*np.array(col)).astype(np.uint8)
        cv2.drawContours(view, contours(), -1, (0, 255, 0), 1)
        cv2.circle(view, state.get("xy", (0, 0)), state["r"], (0, 0, 255) if state["cls"] == 1 else (255, 0, 0), 1)
        cv2.putText(view, f"frame {frames[state['i']]} ({state['i']+1}/{len(frames)}) class={'FRONT' if state['cls'] == 1 else 'BEHIND'} r={state['r']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
        cv2.imshow("label", view)
        k = cv2.waitKey(15) & 0xFF
        if k == ord("q"): save(); break
        elif k == ord("f"): state["cls"] = 1
        elif k == ord("b"): state["cls"] = 2
        elif k == ord("["): state["r"] = max(2, state["r"]-2)
        elif k == ord("]"): state["r"] += 2
        elif k == ord("c"): lab[:] = 0
        elif k == ord("s"): move(1, True)
        elif k == ord("a"): move(-1, True)
        elif k == ord("n"): move(1, False)
        elif k == ord("p"): move(-1, False)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    {"prepare": prepare, "label": label}[sys.argv[1]]()
