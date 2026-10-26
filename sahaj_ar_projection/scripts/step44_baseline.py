"""Step 44: base-line (contact-line) depth reference. Relative model, alignment mode A only, margin rule
max(0.10 m, 0.10 Z_panel) only. ONE pass per clip computes both variants from the same depth:
  Baseline (A):   plain mode-A aligned depth (Step 43 mode A, re-scored with the margin rule)
  Line (A+L):     the same, then shift (and, if the line's depth profile spans enough, scale) re-fitted so the
                  model matches the expected depth Z_line along the panel-plane/table-plane intersection line.
Line = through the panel's two base corners, extended +-3 m, projected with the tracker pose, sampled where it is
inside the image. Reference statistic: after a Huber fit, the shift is moved to the farther side (30th percentile of the
residual in relative-disparity units, because occluders only make disparity larger). Gate: >= 40 in-image samples,
>= 30% of them clean (not nearer than Z_line by more than the margin), clean relative depth MAD <= 10%. Failure ->
hold the last good reference up to 30 frames, then plain Baseline depth (logged). Run via .venv-depth.
  run                    all clips        report   -> runs/step44-baseline.json
"""
import json
import sys
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.diagnostics import write_json
from ar_projection.occlusion import PlaneParams, RelativeDepth, huber_lstsq, panel_layer
from ar_projection.pipeline import run as pipeline_run
from step31_masks import CLIPS as NEAR, REPO, clip_config
from step40_far import CLIPS as FAR_CLIPS, far_config
from step43_relative import REL_CKPT, AlignedPrototype, depth_from, frames_to_save

OUT = Path("runs/step44-baseline")
MARGIN = lambda z: np.maximum(.10, .10*z)
MIN_PX, MIN_CLEAN, MAX_REL_MAD, SPAN_MIN, HOLD_FRAMES = 40, .30, .10, .15, 30
BANDS = (("<1.5m", 0, 1.5), ("1.5-2.5m", 1.5, 2.5), (">2.5m", 2.5, 99))


def clean_nan(x):
    """NaN/inf -> None, recursively (an empty row band gives a NaN median; write_json refuses nonfinite values)."""
    if isinstance(x, dict):
        return {k: clean_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean_nan(v) for v in x]
    return None if isinstance(x, (float, np.floating)) and not np.isfinite(x) else x


def line_samples(T, panel, cam):
    """In-image samples of the base line: (u, v, Z_line)."""
    base = panel[np.argsort(panel[:, 2])[:2]]            # the two corners on the table (smallest world Z)
    u = (base[1]-base[0])/np.linalg.norm(base[1]-base[0])
    pts = base[0]+np.arange(-3, 3, .005)[:, None]*u
    pc = pts@T[:3, :3].T+T[:3, 3]
    pc = pc[pc[:, 2] > .1]
    uv = pc@cam.K.T
    uv = uv[:, :2]/uv[:, 2:3]
    w, h = cam.size
    ok = (uv[:, 0] >= 2) & (uv[:, 0] < w-2) & (uv[:, 1] >= 2) & (uv[:, 1] < h-2)
    return np.rint(uv[ok]).astype(int), pc[ok, 2]


class LinePrototype(AlignedPrototype):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.line_hold, self.line_age = None, 0

    def line_fit(self, d, cam_T, panel, cam, t, s):
        """Returns (params or None, info). params = (t', s') in the d = t + s/Z model."""
        uv, Zl = line_samples(cam_T, panel, cam)
        info = {"line_in_px": int(len(Zl))}
        if len(Zl) < MIN_PX:
            return None, {**info, "line_reason": "line outside image"}
        dm = cv2.medianBlur(d.astype(np.float32), 3).astype(np.float64)
        dl = dm[uv[:, 1], uv[:, 0]]
        iz = 1/Zl
        span = float((iz.max()-iz.min())/iz.mean())
        info["line_profile_span"] = span
        if span >= SPAN_MIN:
            t2, s2 = huber_lstsq(np.stack((np.ones(len(iz)), iz), axis=-1), dl)
            info["line_fit"] = "shift+scale"
        else:
            t2, s2 = t, s
            info["line_fit"] = "shift only"
        t2 += float(np.percentile(dl-(t2+s2*iz), 30))   # farther side: occluders only raise disparity
        if s2 <= 0:
            return None, {**info, "line_reason": "scale not positive"}
        Zh = depth_from(dl, s2, t2)
        clean = np.isfinite(Zh) & (Zh >= Zl-MARGIN(Zl))
        info["line_clean_fraction"] = float(clean.mean())
        if clean.mean() < MIN_CLEAN:
            return None, {**info, "line_reason": "line covered"}
        rel = float(np.median(np.abs(Zh[clean]-Zl[clean])/Zl[clean]))
        info["line_rel_mad"] = rel
        info["line_mad_m"] = float(np.median(np.abs(Zh[clean]-Zl[clean])))
        if rel > MAX_REL_MAD:
            return None, {**info, "line_reason": "MAD too large"}
        return (t2, s2), info

    def render(self, packet, result, panel, landmark_map, features):
        stats = {"status": "tracking unavailable"}
        result.diagnostics["occlusion"] = stats
        if not result.raw_pose.valid or result.render_pose is None:
            self.hold["A"].clear(); self.line_hold = None
            return None
        d, inf = self.model.predict(packet.bgr)
        d = np.asarray(d, np.float64)
        stats.update(inf)
        T, cam = result.render_pose, packet.camera
        _, _, z = panel_layer(T, panel, cam)
        on = z > 0
        fa = self.fit_anchors(d, result, landmark_map, features)
        self.hold["A"].append(fa) if fa else self.hold["A"].clear()
        Zb = Zl = None
        if self.hold["A"] and on.any():
            t, s = np.median(np.array(self.hold["A"]), axis=0)
            Zb = depth_from(d, s, t)
            fit, info = self.line_fit(d, T, panel, cam, t, s)
            stats.update(info)
            if fit:
                self.line_hold, self.line_age = fit, 0
                stats["line_source"] = "fresh"
            else:
                self.line_age += 1
                stats["line_source"] = "held" if self.line_hold and self.line_age <= HOLD_FRAMES else "baseline"
                fit = self.line_hold if stats["line_source"] == "held" else None
            Zl = depth_from(d, fit[1], fit[0]) if fit else Zb
            stats.update(status="aligned depth active", panel_z_median_m=float(np.median(z[on])), anchor_fit=[float(t), float(s)])
            hidden = lambda Z: (np.isfinite(Z[on]) & (Z[on] > 0) & (Z[on] < z[on]-MARGIN(z[on])))
            hb, hl = hidden(Zb), hidden(Zl)
            stats["hidden_baseline"], stats["hidden_line"] = float(hb.mean()), float(hl.mean())
            if hl.mean() < .01:                              # presumably empty: bias by row band inside the panel
                ys = np.where(on)[0]
                lo, hi = ys.min(), ys.max()+1
                for name, Z in (("baseline", Zb), ("line", Zl)):
                    stats[f"panel_bias_by_row_{name}"] = {b: float(np.median((Z-z)[(np.arange(Z.shape[0])[:, None] >= lo+k*(hi-lo)/3) & (np.arange(Z.shape[0])[:, None] < lo+(k+1)*(hi-lo)/3) & on & np.isfinite(Z)]))
                                                           for k, b in enumerate(("top", "middle", "bottom"))}
        else:
            stats["status"] = "visibility unavailable"
        if packet.index in self.depth_frames and Zb is not None:
            n = f"{packet.index:06d}"
            for tag, Z in (("baseline", Zb), ("line", Zl)):
                np.save(self.output/f"{n}-scene-{tag}.npy", np.where(np.isfinite(Z), Z, 99.).astype(np.float32))
        stats = clean_nan(stats)
        result.diagnostics["occlusion"] = stats   # the pipeline logs this dict too, and refuses NaN
        write_json(self.output/f"{packet.index:06d}.json", {**stats, "render_T_CW": T, "K": cam.K})
        color, visible, _ = panel_layer(T, panel, cam)
        return color, visible


def do_run():
    model = RelativeDepth(REPO, REL_CKPT, 252)
    sel = frames_to_save()
    clips = {**{k: clip_config(v) for k, v in NEAR.items()}, **{k: far_config(v) for k, v in FAR_CLIPS.items()}}
    info = {}
    for label, cfg in clips.items():
        proto = LinePrototype(model, OUT/label, "relative", "A", PlaneParams(), sel.get(label, ()))
        info[label] = str(pipeline_run(cfg, occlusion=proto))
        print(label, "done", flush=True)
    (OUT/"run.json").write_text(json.dumps(info, indent=2))


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def band(z):
    return next((n for n, lo, hi in BANDS if lo <= z < hi), None)


def do_report():
    import step41_score as sc41
    from ar_projection.camera import Capture
    from ar_projection.geometry import panel_points
    from step37_plane_calibration import STEP36, group_of
    info = json.loads((OUT/"run.json").read_text())
    res = {"clips": {}, "groups": {}, "far_gate": {}}
    allst = {}
    for label in info:
        st = {int(f.stem): json.loads(f.read_text()) for f in (OUT/label).glob("[0-9]*.json")}
        allst[label] = st
        act = [s for s in st.values() if "hidden_line" in s]
        src = [s["line_source"] for s in act]
        reasons = {}
        for s in act:
            if s["line_source"] != "fresh" or s.get("line_reason"):
                reasons[s.get("line_reason", "none")] = reasons.get(s.get("line_reason", "none"), 0)+1
        e = {"frames": len(st), "active_fraction": len(act)/len(st),
             "mean_hidden_baseline": mean(s["hidden_baseline"] for s in act), "mean_hidden_line": mean(s["hidden_line"] for s in act),
             "frac_frames_over_5pct": {"baseline": float(np.mean([s["hidden_baseline"] > .05 for s in act])), "line": float(np.mean([s["hidden_line"] > .05 for s in act]))},
             "line_gate_fresh_fraction": float(np.mean([x == "fresh" for x in src])), "line_held_fraction": float(np.mean([x == "held" for x in src])),
             "line_fell_back_to_baseline_fraction": float(np.mean([x == "baseline" for x in src])), "failure_reasons": reasons,
             "shift_only_fraction": float(np.mean([s.get("line_fit") == "shift only" for s in act if "line_fit" in s])),
             "line_outside_image_fraction": float(np.mean([s.get("line_in_px", 0) < MIN_PX for s in act])),
             "by_panel_distance": {}}
        for n, lo, hi in BANDS:
            sel = [s for s in act if lo <= s["panel_z_median_m"] < hi]
            if sel:
                e["by_panel_distance"][n] = {"n": len(sel), "line_gate_fresh_fraction": float(np.mean([s["line_source"] == "fresh" for s in sel])),
                                              "line_mad_m_median": mean(s.get("line_mad_m") for s in sel),
                                              "hidden_baseline": mean(s["hidden_baseline"] for s in sel), "hidden_line": mean(s["hidden_line"] for s in sel),
                                              "reasons": {r: sum(s.get("line_reason") == r for s in sel) for r in {s.get("line_reason") for s in sel if s.get("line_reason")}}}
        for name in ("baseline", "line"):
            rows = [s[f"panel_bias_by_row_{name}"] for s in act if f"panel_bias_by_row_{name}" in s]
            e[f"empty_frame_panel_bias_by_row_{name}_m"] = {b: mean(r[b] for r in rows) for b in ("top", "middle", "bottom")} if rows else None
        res["clips"][label] = e
    grp = {}
    for f in json.loads(STEP36.read_text())["frames"]:
        s = allst.get(f["clip"], {}).get(f["frame_index"])
        if s and "hidden_line" in s:
            grp.setdefault(group_of(f), []).append(s)
    res["groups"] = {g: {"n": len(v), "hidden_baseline": mean(s["hidden_baseline"] for s in v), "hidden_line": mean(s["hidden_line"] for s in v)} for g, v in grp.items()}
    v0 = sorted((s["hidden_line"] for s in allst["V0"].values() if "hidden_line" in s), reverse=True)[:5]
    v0b = sorted((s["hidden_baseline"] for s in allst["V0"].values() if "hidden_baseline" in s), reverse=True)[:5]
    res["V0_worst5"] = {"baseline": [round(x, 4) for x in v0b], "line": [round(x, 4) for x in v0]}
    # Step 41 labels, same scorer (margin rule as a function of the panel Z map)
    cfg = far_config(FAR_CLIPS["F-occ"])
    cap = Capture(cfg.camera); cam = cap.read().camera; cap.close()
    panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
    frames = json.loads(Path("runs/step41-labels/frames.json").read_text())
    rows = {json.loads(l)["frame_index"]: json.loads(l) for l in open(Path(info["F-occ"])/"frames.jsonl")}
    poses = {n: rows[n]["actual_pose_source"] for n in frames if n in rows}
    res["step41_labels"] = {}
    for name in ("baseline", "line"):
        x = sc41.score_mode(frames, name, panel, cam, poses, margin=MARGIN, out=OUT/"F-occ")
        res["step41_labels"][name] = {k: v for k, v in x.items() if k != "per_frame"}
        res["step41_labels"][name]["per_frame"] = [{"frame": p["frame"], "pose": p["pose_source"], "iou": round(p["iou"], 3), "false_hide": p["false_hide"] and round(p["false_hide"], 3), "miss": p["miss_rate"] and round(p["miss_rate"], 3)} for p in x["per_frame"]]
    Path("runs/step44-baseline.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("groups", "V0_worst5")}, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    do_run() if sys.argv[1:] == ["run"] else do_report()
