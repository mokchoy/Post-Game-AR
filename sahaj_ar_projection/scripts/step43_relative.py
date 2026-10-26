"""Step 43: relative-depth model with disparity-space alignment (offline). Run via .venv-depth, ONE model per process.
  run --source relative --mode A|B|C [--input-size N] [--clips ...]   (also --source metric --mode A = baseline on the
                                                                       same frames, margins and code path)
  report RUN_DIR...
Alignment (relative): d = s/Z + t (Huber). A: landmark anchors, 5-frame median hold. B: table-plane pixels (Z from the
pose, Step 37 pixel rules, far clips use 0.8-4 m). C: only table pixels within +-25% of the panel's depth. A failed
gate falls back to A, then "visibility unavailable" (panel unoccluded, status logged). Mask: Z_real < Z_panel - margin
for margins 0.15 m, 0.20 m and 10% of panel depth (0.05 m also logged); Z_real = s/(d-t), guarded (d-t<=eps -> far).
Metric baseline: Z=(raw-a)/b from the same anchors as production."""
import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.diagnostics import write_json
from ar_projection.occlusion import MetricDepth, PlaneCorrector, PlaneParams, RelativeDepth, collect_anchors, fit_affine_huber, huber_lstsq, panel_layer
from ar_projection.pipeline import run as pipeline_run
from step31_masks import CLIPS as NEAR, REPO, clip_config
from step37_plane_calibration import STEP36, group_of
from step40_far import CLIPS as FAR_CLIPS, far_config

REL_CKPT = REPO+"/metric_depth/checkpoints/depth_anything_v2_vits.pth"
MET_CKPT = REPO+"/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
OUT = Path("runs/step43-relative")
EPS = 1e-3          # relative-model output units: guard for d - t
GATE_REL = .08      # plane-fit gate: median |Z_hat - Z_oracle| / Z_oracle on calibration pixels
MARGINS = {"0.15": lambda zp: .15, "0.20": lambda zp: .20, "10%": lambda zp: .1*zp, "0.05": lambda zp: .05}


def depth_from(d, s, t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d-t > EPS, s/(d-t), np.inf)


class AlignedPrototype:
    def __init__(self, model, output, source, mode, plane, depth_frames):
        self.model, self.source, self.mode, self.plane = model, source, mode, PlaneCorrector(plane)
        self.hold = {"A": deque(maxlen=5), "P": deque(maxlen=5)}
        self.depth_frames, self.output = set(depth_frames), Path(output)
        self.output.mkdir(parents=True, exist_ok=False)

    def fit_anchors(self, d, result, landmark_map, features):
        Z, D = collect_anchors(d, result, landmark_map, features)
        if len(Z) < 12:
            return None
        Z, D = np.array(Z), np.array(D)
        if self.source == "metric":
            a, b = fit_affine_huber(Z, D)                      # raw = a + b Z
            ok = b > 0 and np.median(np.abs((D-a)/b-Z)/Z) <= .25
            return (a, b) if ok else None
        t, s = fit_affine_huber(1/Z, D)                        # d = t + s / Z
        zh = depth_from(D, s, t)
        return (t, s) if s > 0 and np.all(np.isfinite(zh)) and np.median(np.abs(zh-Z)/Z) <= .25 else None

    def fit_plane(self, d, T, panel, camera, landmark_map, zp):
        mask, oracle = self.plane.calibration_mask(T, camera, panel, d, landmark_map.points if landmark_map is not None else None)
        if self.mode == "C":
            mask &= np.abs(oracle-zp) < .25*zp
        n = int(mask.sum())
        info = {"plane_pixels": n}
        if n < max(self.plane.params.min_pixels//2 if self.mode == "C" else self.plane.params.min_pixels, 500):
            return None, info
        idx = np.flatnonzero(mask.ravel())
        if n > 20000:
            idx = self.plane.rng.choice(idx, 20000, replace=False)
        dd, zo = d.ravel()[idx], oracle.ravel()[idx]
        A = np.stack((np.ones(len(idx)), 1/zo), axis=-1)
        t, s = huber_lstsq(A, dd)
        zh = depth_from(dd, s, t)
        rel = float(np.median(np.abs(zh-zo)/zo)) if np.all(np.isfinite(zh)) else None
        info["plane_rel_residual"] = rel
        return ((t, s) if s > 0 and rel is not None and rel <= GATE_REL else None), info

    def render(self, packet, result, panel, landmark_map, features):
        stats = {"status": "tracking unavailable"}
        result.diagnostics["occlusion"] = stats
        if not result.raw_pose.valid or result.render_pose is None:
            self.hold["A"].clear(); self.hold["P"].clear()
            return None
        d, inf = self.model.predict(packet.bgr)
        d = np.asarray(d, np.float64)
        stats.update(inf)
        T, cam = result.render_pose, packet.camera
        _, _, z = panel_layer(T, panel, cam)
        on = z > 0
        zp = float(np.median(z[on])) if on.any() else None
        fa = self.fit_anchors(d, result, landmark_map, features)
        self.hold["A"].append(fa) if fa else self.hold["A"].clear()
        used, params = None, None
        if self.source == "relative" and self.mode in "BC" and zp:
            fp, info = self.fit_plane(d, T, panel, cam, landmark_map, zp)
            stats.update(info)
            self.hold["P"].append(fp) if fp else self.hold["P"].clear()
            if self.hold["P"]:
                used, params = "plane", np.median(np.array(self.hold["P"]), axis=0)
        if params is None and self.hold["A"]:
            used, params = "anchor", np.median(np.array(self.hold["A"]), axis=0)
        if params is None or not on.any():
            stats["status"] = "visibility unavailable"
            color, visible, z = panel_layer(T, panel, cam)
            Zreal = None
        else:
            if self.source == "metric":
                Zreal = (d-params[0])/params[1]
            else:
                Zreal = depth_from(d, params[1], params[0])
            stats.update(status="aligned depth active", alignment_used=used, fit=[float(x) for x in params], panel_z_median_m=zp)
            stats["hidden_by_margin"] = {k: float(np.mean(np.isfinite(Zreal[on]) & (Zreal[on] > 0) & (Zreal[on] < z[on]-f(z[on])))) for k, f in MARGINS.items()}
            color, visible, _ = panel_layer(T, panel, cam, np.where(np.isfinite(Zreal), Zreal, 1e6), margin_m=.15)
        if packet.index in self.depth_frames:
            n = f"{packet.index:06d}"
            np.save(self.output/f"{n}-raw.npy", d.astype(np.float32))
            if Zreal is not None:
                np.save(self.output/f"{n}-scene-aligned.npy", np.where(np.isfinite(Zreal), Zreal, 99.).astype(np.float32))
        write_json(self.output/f"{packet.index:06d}.json", {**stats, "render_T_CW": T, "K": cam.K})
        return color, visible


def frames_to_save():
    out = {}
    for f in json.loads(STEP36.read_text())["frames"]:
        out.setdefault(f["clip"], set()).add(f["frame_index"])
    out.setdefault("F-occ", set()).update(json.loads(Path("runs/step41-labels/frames.json").read_text()))
    return out


def do_run(a):
    Model, ckpt = (RelativeDepth, REL_CKPT) if a.source == "relative" else (MetricDepth, MET_CKPT)
    model = Model(REPO, ckpt, a.input_size)
    tag = a.tag or f"{a.source}-{a.mode}-s{a.input_size}"
    root, sel, info = OUT/tag, frames_to_save(), {"tag": tag, "source": a.source, "mode": a.mode, "input_size": a.input_size, "checkpoint": Path(ckpt).name,
                                                    "checkpoint_sha256": hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(), "clips": {}}
    clips = {**{k: (clip_config(v), PlaneParams(terms="noint")) for k, v in NEAR.items()},
             **{k: (far_config(v), PlaneParams(terms="noint", z_min_m=.8, z_max_m=4.0)) for k, v in FAR_CLIPS.items()}}
    for label in a.clips or clips:
        cfg, plane = clips[label]
        info["clips"][label] = str(pipeline_run(cfg, occlusion=AlignedPrototype(model, root/label, a.source, a.mode, plane, sel.get(label, ()))))
        print(tag, label, "done", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root/"run.json").write_text(json.dumps(info, indent=2))


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def do_report(dirs):
    from step38_input_size import hand
    S36 = Path("runs/step36-depth-sources")
    sel = json.loads(STEP36.read_text())["frames"]
    res = {}
    for d in map(Path, dirs):
        info = json.loads((d/"run.json").read_text())
        r = {k: info[k] for k in ("source", "mode", "input_size", "checkpoint", "checkpoint_sha256")}
        r["clips"], r["groups"] = {}, {}
        allst = {}
        for label in info["clips"]:
            st = {int(f.stem): json.loads(f.read_text()) for f in (d/label).glob("[0-9]*.json")}
            allst[label] = st
            act = [s for s in st.values() if "hidden_by_margin" in s]
            r["clips"][label] = {"frames": len(st), "active_fraction": len(act)/len(st) if st else None,
                                 "alignment_used": {u: sum(s["alignment_used"] == u for s in act) for u in ("plane", "anchor")},
                                 "mean_hidden": {m: mean(s["hidden_by_margin"][m] for s in act) for m in MARGINS},
                                 "frac_frames_over_5pct@0.15": float(np.mean([s["hidden_by_margin"]["0.15"] > .05 for s in act])) if act else None,
                                 "ms_per_frame": mean(s.get("inference_ms") for s in st.values())}
        for f in sel:
            s = allst.get(f["clip"], {}).get(f["frame_index"])
            if s and "hidden_by_margin" in s:
                r["groups"].setdefault(group_of(f), []).append(s["hidden_by_margin"])
        r["groups"] = {g: {"n": len(v), **{m: mean(x[m] for x in v) for m in MARGINS}} for g, v in r["groups"].items()}
        v0 = allst.get("V0", {})
        w5 = sorted((s["hidden_by_margin"]["0.15"] for s in v0.values() if "hidden_by_margin" in s), reverse=True)[:5]
        r["V0_worst5@0.15"] = [round(x, 4) for x in w5]
        sc = d/"V2"/"000441-scene-aligned.npy"
        if sc.exists():
            r["hand_441"] = hand(S36/"V2_000441", np.load(sc).astype(np.float64), np.load(d/"V2"/"000441-raw.npy"))
        if "F-occ" in info["clips"]:   # Step 41 labels, same scorer
            import step41_score as sc41
            cfg = far_config(FAR_CLIPS["F-occ"])
            from ar_projection.camera import Capture
            from ar_projection.geometry import panel_points
            cap = Capture(cfg.camera); cam = cap.read().camera; cap.close()
            panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
            frames = json.loads(Path("runs/step41-labels/frames.json").read_text())
            rows = {json.loads(l)["frame_index"]: json.loads(l) for l in open(Path(info["clips"]["F-occ"])/"frames.jsonl")}
            poses = {n: rows[n]["actual_pose_source"] for n in frames if n in rows}
            r["step41_labels"] = {}
            for m in (.15, .20):
                sc41.OUT = d/"F-occ"
                x = sc41.score_mode(frames, "aligned", panel, cam, poses, margin=m, out=d/"F-occ")
                x.pop("per_frame"); r["step41_labels"][str(m)] = x
        res[d.name] = r
    Path("runs/step43-relative.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", choices=("relative", "metric"), default="relative")
    r.add_argument("--mode", choices=("A", "B", "C"), default="A")
    r.add_argument("--input-size", type=int, default=252)
    r.add_argument("--tag")
    r.add_argument("--clips", nargs="*")
    q = sub.add_parser("report")
    q.add_argument("dirs", nargs="+")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    do_run(a) if a.cmd == "run" else do_report(a.dirs)
