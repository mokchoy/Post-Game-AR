"""Step 41 scoring of the F-occ labels (runs/step41-labels). `run` replays F-occ once (plane correction on, so one
pass yields the anchor-fit scene and, where its gate passes, the plane scene) and saves scene depth for the labelled
frames; `score` compares the predicted panel visibility with truth = panel polygon minus FRONT silhouette.
Metrics: visibility IoU, boundary F1@2px (occlusion.mask_metrics), false-hide rate (truth-visible pixels predicted
hidden), miss rate (truth-hidden pixels predicted visible); pooled over pixels, and per pose source. Flicker and
transition timing need consecutive labelled frames, which this sparse set does not have. Run via .venv-depth."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.camera import Capture
from ar_projection.geometry import panel_points
from ar_projection.occlusion import DepthPrototype, MetricDepth, mask_metrics, panel_layer
from ar_projection.pipeline import run as pipeline_run
from step31_masks import CHECKPOINT, REPO
from step40_far import CLIPS, MODES, far_config

L = Path("runs/step41-labels")
OUT = Path("runs/step41-run")


def do_run():
    frames = json.loads((L/"frames.json").read_text())
    occ = DepthPrototype(MetricDepth(REPO, CHECKPOINT, 252), OUT, slope_sign=1, plane_correction=MODES["plane"], depth_frames=frames, export_images=False)
    tracking = pipeline_run(far_config(CLIPS["F-occ"]), occlusion=occ)
    (OUT/"tracking_run.txt").write_text(str(tracking))


def score_mode(frames, source, panel, camera, poses, margin=.02, out=None):
    out = out or OUT
    tot = {"truth_vis": 0, "fh": 0, "truth_hid": 0, "miss": 0}
    per, groups = [], {}
    for n in frames:
        sp = out/f"{n:06d}-scene-{source}.npy"
        lab = cv2.imread(str(L/f"{n:06d}-labels.png"), 0)
        T = json.loads((out/f"{n:06d}.json").read_text()).get("render_T_CW")
        if not sp.exists() or T is None:
            continue
        if callable(margin):  # margin as a function of the panel Z map (e.g. max(0.10, 0.10 Z))
            margin_arr = margin(panel_layer(np.array(T), panel, camera)[2])
        else:
            margin_arr = margin
        _, pred_vis, z = panel_layer(np.array(T), panel, camera, np.load(sp), margin_m=margin_arr)
        on = z > 0
        truth_vis = on & (lab != 1)
        fh, miss = (pred_vis[on] == 0) & truth_vis[on], pred_vis[on] & ~truth_vis[on]
        c = {"truth_vis": int(truth_vis.sum()), "fh": int(fh.sum()), "truth_hid": int((on & ~truth_vis).sum()), "miss": int(miss.sum())}
        m = mask_metrics(pred_vis, truth_vis)
        per.append({"frame": n, "pose_source": poses.get(n), **m, "false_hide": c["fh"]/c["truth_vis"] if c["truth_vis"] else None,
                    "miss_rate": c["miss"]/c["truth_hid"] if c["truth_hid"] else None})
        for k, v in c.items():
            tot[k] += v
            g = groups.setdefault("tag/joint" if poses.get(n) in ("tag", "joint") else "landmark", {"truth_vis": 0, "fh": 0, "truth_hid": 0, "miss": 0})
            g[k] += v
    rate = lambda a, b: a/b if b else None
    pooled = lambda t: {"false_hide_rate": rate(t["fh"], t["truth_vis"]), "miss_rate": rate(t["miss"], t["truth_hid"])}
    return {"frames_scored": len(per), "mean_iou": float(np.mean([p["iou"] for p in per])) if per else None,
            "mean_boundary_f1": float(np.mean([p["boundary_f1_at_2px"] for p in per])) if per else None,
            **pooled(tot), "by_pose_source": {k: pooled(v) for k, v in groups.items()}, "per_frame": per}


def do_score():
    frames = json.loads((L/"frames.json").read_text())
    cfg = far_config(CLIPS["F-occ"])
    panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
    cap = Capture(cfg.camera)
    camera = cap.read().camera
    cap.close()
    rows = {json.loads(l)["frame_index"]: json.loads(l) for l in open(Path((OUT/"tracking_run.txt").read_text())/"frames.jsonl")}
    poses = {n: rows[n]["actual_pose_source"] for n in frames if n in rows}
    res = {"labelled_frames": len(frames), "margin_m": .02,
           "note": "anchor = current method; plane = Step 37 noint where its gate passed (few frames at far range)",
           **{s: score_mode(frames, s, panel, camera, poses) for s in ("anchor", "plane")}}
    Path("runs/step41-scores.json").write_text(json.dumps(res, indent=2))
    for s in ("anchor", "plane"):
        r = res[s]
        print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("per_frame", "by_pose_source")}, r["by_pose_source"])


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    {"run": do_run, "score": do_score}[sys.argv[1]]()
