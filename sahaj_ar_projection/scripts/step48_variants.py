"""Step 48 offline check: replay a recorded run through the production RelativeLineOcclusion with reference line / plane /
plane + background prior, and score hand-on-panel hidden and background false hides. The hand is a SKIN-COLOUR PROXY
(approximate, from step48_analyze_lib.skin) and only every 3rd frame is scored, as in the 26 Oct pre-check.
Usage (from the project root, .venv-depth):  step48_variants.py 092056 | fempty   -> runs/step48-variants-<which>.json"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from ar_projection.config import BackgroundPrior, load_config
from ar_projection.occlusion import RelativeDepth, RelativeLineOcclusion
from ar_projection.pipeline import run
from step31_masks import REPO
from step43_relative import REL_CKPT

SLACKS = (.1, .5, 1e9)


def skin(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = [ycc[..., i].astype(int) for i in range(3)]
    m = ((Cr >= 140) & (Cr <= 180) & (Cb >= 80) & (Cb <= 125) & (Y > 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m, bool)
    for j in range(1, n):
        if st[j, 4] > 2500:
            keep |= lab == j
    return keep


class Score(RelativeLineOcclusion):
    def __init__(self, *a, score=True, **k):
        super().__init__(*a, **k)
        self.keep_view, self.rows, self.score, self.ms = True, [], score, []

    def render(self, packet, result, panel, landmark_map, features):
        before = time.perf_counter()
        out = super().render(packet, result, panel, landmark_map, features)
        self.ms.append((time.perf_counter()-before)*1000)
        v = self.view
        if self.score and v is not None and packet.index % 3 == 0:
            z, scene, vis = v["z"], v["scene"], v["visible"]
            on = z > 0
            hand = skin(packet.bgr)
            handd = cv2.dilate(hand.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
            hp, nh = hand & on, on & ~handd
            margin = np.maximum(.1, .1*z)
            variants = {"": vis}
            if v.get("fg") is not None:
                base = ~(on & (scene+margin < z))
                for sl in SLACKS:
                    variants[f"slack{sl:g}"] = base & ~(v["fg"] & on & (scene < z+sl))
            row = {"i": packet.index, "src": result.diagnostics["occlusion"].get("alignment_source"), "hand_px": int(hp.sum()), "bg_px": int(nh.sum())}
            for name, vv in variants.items():
                row["hand"+name] = float(np.mean(~vv[hp])) if hp.sum() > 1500 else None
                row["bg"+name] = float(np.mean(~vv[nh])) if nh.sum() > 500 else None
                row["all"+name] = float(np.mean(~vv[on])) if on.any() else None
            self.rows.append(row)
        return out


def config(which):
    if which == "092056":
        rd = Path("runs/20261025T092056-7a6e0212")
        cfg = load_config(rd/"config.json")
    else:
        rd = Path("runs/20261025T000913-e058a920")
        cfg = load_config("config-logitech-far.json")
        cfg = replace(cfg, mapping=replace(cfg.mapping, mode="localize", map_path=str(Path("data/logitech-landmark-map-occ.npz").resolve())))
    return replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rd.resolve()/"input.avi"), timestamps_path=str(rd.resolve()/"frames.jsonl")),
                   display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return {"n": len(v), "mean": float(np.mean(v)), "frac_under_half": float(np.mean(np.array(v) < .5)), "frac_over_2pct": float(np.mean(np.array(v) > .02))} if v else None


if __name__ == "__main__":
    which = sys.argv[1]
    n = 400 if which == "fempty" else None
    model = RelativeDepth(REPO, REL_CKPT, 252)
    cfg = config(which)
    out = {}
    for name, ref, prior in (("line", "line", None), ("plane", "plane", None), ("plane+prior", "plane", BackgroundPrior(enabled=True))):
        occ = Score(model, None, .10, .10, True, 30, REL_CKPT, "bypassed", ref, cfg.occlusion.plane_extent_m, cfg.occlusion.plane_min_span, cfg.occlusion.plane_min_px, prior)
        d = run(cfg, max_frames=n, occlusion=occ)
        keys = [""] if prior is None else ["slack0.1", "slack0.5", "slack1e+09"]
        res = {"run": str(d), "frames": len(occ.ms), "ms_per_frame_occlusion": float(np.mean(occ.ms)), "scored_frames": len(occ.rows),
               "sources": {k: sum(r["src"] == k for r in occ.rows) for k in {r["src"] for r in occ.rows}}}
        for k in keys:
            res[k or "default"] = {"hand_hidden": agg(occ.rows, "hand"+k), "bg_false_hide": agg(occ.rows, "bg"+k), "panel_hidden_all": agg(occ.rows, "all"+k)}
        out[name] = res
        print(name, json.dumps(res)[:600], flush=True)
    Path(f"runs/step48-variants-{which}.json").write_text(json.dumps(out, indent=2))
