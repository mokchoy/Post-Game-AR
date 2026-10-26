"""Step 42: check the instruments (diagnostic only; no default changes).
  select   (any python)  pick contact-sheet frames from the Step 40 anchor-mode per-frame json -> runs/step42-checks/frames.json
  run      (.venv-depth) replay both far clips once with the metric model, anchor fit AND an UNGATED plane fit (stats carry
                         both scenes' hidden fractions at margins .02-.2, panel Z, plane fit terms); saves depth/images for picks
  poses    (.venv-depth) replay with the tag withheld in alternating 45-frame windows: landmark-only pose vs tag pose (task 3)
  report   (.venv-depth) tasks 2-4 -> runs/step42-checks.json
  figures  (system python3, has matplotlib) one PNG per picked frame -> runs/step42-images/
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

CLIPS = {"F-occ": "runs/20261025T000122-210c1bb8", "F-empty": "runs/20261025T000913-e058a920"}
W = Path("runs/step42-checks")
IMG = Path("runs/step42-images")
FAR = Path("runs/step40-far")
BANDS = (("<1.5m", 0, 1.5), ("1.5-2.5m", 1.5, 2.5), (">2.5m", 2.5, 99))
rng = np.random.default_rng(0)


def stats_of(directory):
    return {int(f.stem): json.loads(f.read_text()) for f in Path(directory).glob("[0-9]*.json")}


def select():
    W.mkdir(parents=True, exist_ok=True)
    picks = {}
    for label, (n_top, n_other, other) in (("F-empty", (12, 4, "near-zero")), ("F-occ", (12, 8, "spread"))):
        st = {i: s for i, s in stats_of(FAR/"anchor"/label).items() if "hidden_fraction" in s}
        ranked = sorted(st, key=lambda i: -st[i]["hidden_fraction"])
        top = []
        for i in ranked:  # >=15 frames apart so the sheet is not 12 copies of one moment
            if all(abs(i-j) >= 15 for j in top):
                top.append(i)
            if len(top) == n_top:
                break
        idx = sorted(st)
        if other == "near-zero":
            pool = [i for i in idx if st[i]["hidden_fraction"] < .005 and i not in top]
            extra = sorted(rng.choice(pool, n_other, replace=False).tolist())
        else:
            extra = [idx[k] for k in np.linspace(0, len(idx)-1, n_other).astype(int)]
        stills = [idx[k] for k in np.linspace(0, len(idx)-1, 5).astype(int)[1:4]]
        picks[label] = {"top": top, "other": extra, "stills": stills}
    (W/"frames.json").write_text(json.dumps(picks, indent=2))
    print(picks)


def run():
    from ar_projection.occlusion import DepthPrototype, MetricDepth, PlaneParams
    from ar_projection.pipeline import run as pipeline_run
    from step31_masks import CHECKPOINT, REPO
    from step40_far import far_config
    picks = json.loads((W/"frames.json").read_text())
    model = MetricDepth(REPO, CHECKPOINT, 252)
    # gate disabled (diagnostic only): any finite fit is applied; min_pixels still required
    ungated = PlaneParams(terms="noint", z_min_m=.8, z_max_m=4.0, max_mad_m=1e9, b_range=(-1e9, 1e9))
    info = {}
    for label, d in CLIPS.items():
        frames = sum(picks[label].values(), [])
        occ = DepthPrototype(model, W/"run"/label, slope_sign=1, plane_correction=ungated, depth_frames=frames, export_images=False)
        info[label] = str(pipeline_run(far_config(d), occlusion=occ))
        print(label, "done", flush=True)
    (W/"run.json").write_text(json.dumps(info, indent=2))


def poses():
    from ar_projection.camera import Capture
    from ar_projection.geometry import panel_points
    from ar_projection.pipeline import Pipeline
    from step36_depth_sources import pose_agreement, panel_stats
    from step40_far import far_config
    out = {}
    for label, d in CLIPS.items():
        cfg = far_config(d)
        panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
        cap = Capture(cfg.camera)
        p = cap.read()
        pipe = Pipeline(cfg, p.camera, background_mapping=False)
        rows = {}
        while p is not None:
            # a landmark pose never bootstraps with the tag withheld from frame 0, so the tag is withheld in alternating 45-frame
            # windows: the tracker is seeded by the tag, then runs landmark-only, and the tag reference is compared in those windows
            r = pipe.process(p, withhold_tag=(p.index//45) % 2 == 1)
            dg = r.diagnostics
            C, B = dg.get("tag_reference_T_CW"), r.render_pose
            if C is not None and dg.get("tag_reference_valid") and B is not None and r.raw_pose.valid:
                b, c = panel_stats(B, panel, p.camera), panel_stats(C, panel, p.camera)
                a = pose_agreement(b, c)
                if a:
                    rows[p.index] = {**a, "panel_C_z_median_m": c["z_median_m"], "C_rms_px": dg.get("tag_reference_rms_px"),
                                     "landmark_inliers": r.raw_pose.landmark_inliers}
            p = cap.read()
        cap.close()
        out[label] = rows
        print(label, len(rows), "tag-valid frames with landmark pose", flush=True)
    (W/"poses.json").write_text(json.dumps(out))


def band(z):
    return next((n for n, lo, hi in BANDS if lo <= z < hi), None)


def mstat(v):
    v = np.array([x for x in v if x is not None])
    return {"n": int(len(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "frac_over_5pct": float((v > .05).mean())} if len(v) else None


def report():
    info = json.loads((W/"run.json").read_text())
    poses = json.loads((W/"poses.json").read_text())
    res = {}
    for label in CLIPS:
        st = stats_of(W/"run"/label)
        rows = {json.loads(l)["frame_index"]: json.loads(l) for l in open(Path(info[label])/"frames.jsonl")}
        tag_valid = {i for i, r in rows.items() if any(t["id"] == 0 and t["valid"] for t in r["diagnostics"].get("tags", []))}
        act = {i: s for i, s in st.items() if "anchor_hidden_by_margin" in s or ("hidden_fraction_by_margin" in s and not s.get("plane_gate"))}
        # anchor-scene hidden fractions: anchor_hidden_by_margin when a plane scene replaced it, else the scene used was the anchor one
        anchor = {i: (s.get("anchor_hidden_by_margin") or s["hidden_fraction_by_margin"]) for i, s in st.items() if "hidden_fraction_by_margin" in s and "depth_fit_a" in s}
        e = {"frames_with_anchor_scene": len(anchor), "margins": {}, "margins_by_band": {}}
        for m in ("0.02", "0.15", "0.2"):
            e["margins"][m] = mstat([a[m] for a in anchor.values()])
            e["margins_by_band"][m] = {n: mstat([a[m] for i, a in anchor.items() if band(st[i]["panel_z_median_m"]) == n]) for n, _, _ in BANDS}
        # task 4: gate criteria on every frame where a fit was attempted, ungated run
        fitted = {i: s for i, s in st.items() if "plane_coeffs" in s}
        few = sum("plane_pixels" in s and "plane_coeffs" not in s for s in st.values())
        mad = np.array([s["plane_residual_mad_m"] for s in fitted.values()])
        b = np.array([s["plane_coeffs"][0] for s in fitted.values()])
        e["gate"] = {"frames_with_stats": len(st), "fail_too_few_pixels": few, "fits_attempted": len(fitted),
                     "fail_mad_over_0.05": int((mad > .05).sum()), "fail_coeff_outside_0.5_2": int(((b < .5) | (b > 2)).sum()),
                     "fail_either": int(((mad > .05) | (b < .5) | (b > 2)).sum()), "pass_both": int(((mad <= .05) & (b >= .5) & (b <= 2)).sum()),
                     "mad_m_percentiles_all_fits_5_50_95": np.percentile(mad, [5, 50, 95]).tolist(),
                     "mad_m_median_passing_fits_only": float(np.median(mad[(mad <= .05) & (b >= .5) & (b <= 2)])) if ((mad <= .05) & (b >= .5) & (b <= 2)).any() else None,
                     "coeff_percentiles_5_50_95": np.percentile(b, [5, 50, 95]).tolist()}
        tv = [i for i in fitted if i in tag_valid and i in anchor]
        e["ungated_plane_on_tag_valid_frames"] = {"n": len(tv), **{f"plane_hidden@{m}": mstat([st[i]["hidden_fraction_by_margin"][m] for i in tv]) for m in ("0.02", "0.15")},
                                                  **{f"anchor_hidden@{m}": mstat([anchor[i][m] for i in tv]) for m in ("0.02", "0.15")}}
        pr = poses[label]
        e["pose_check"] = {n: ({k: {"median": float(np.median([r[k] for r in sel])), "p95": float(np.percentile([r[k] for r in sel], 95))}
                               for k in ("camera_position_diff_m", "rotation_diff_deg", "corner_px_median", "corner_px_max", "panel_z_diff_median_m", "panel_z_diff_max_m")} | {"n": len(sel)}
                           if (sel := [r for r in pr.values() if band(r["panel_C_z_median_m"]) == n]) else None) for n, _, _ in BANDS}
        e["pose_check_n_tag_valid_frames_with_landmark_pose"] = len(pr)
        res[label] = e
    Path("runs/step42-checks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


def figures():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = json.loads((W/"frames.json").read_text())
    IMG.mkdir(exist_ok=True)
    for label in CLIPS:
        st = stats_of(W/"run"/label)
        trk = {json.loads(l)["frame_index"]: json.loads(l)["actual_pose_source"] for l in open(Path(json.loads((W/"run.json").read_text())[label])/"frames.jsonl")}
        for kind in ("top", "other"):
            for n in picks[label][kind]:
                d = W/"run"/label
                if not (d/f"{n:06d}-scene-anchor.npy").exists():
                    continue
                img = cv2.cvtColor(cv2.imread(str(d/f"{n:06d}-image.png")), cv2.COLOR_BGR2RGB)
                scene = np.load(d/f"{n:06d}-scene-anchor.npy")
                T = np.array(st[n]["render_T_CW"])
                from ar_projection.config import load_config
                from ar_projection.geometry import panel_points
                from ar_projection.occlusion import panel_layer
                from types import SimpleNamespace
                K = np.array(st[n]["K"]).reshape(3, 3)
                cam = SimpleNamespace(K=K, distortion=np.zeros(5), size=(img.shape[1], img.shape[0]))
                cfg = load_config("config-logitech-far.json")
                panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)
                _, vis, z = panel_layer(T, panel, cam, scene, margin_m=.02)
                hid = (z > 0) & ~vis
                over = img.copy()
                over[hid] = (0.5*over[hid]+0.5*np.array([255, 0, 0])).astype(np.uint8)
                cnt = cv2.findContours((z > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(over, cnt, -1, (0, 255, 0), 1)
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                ax[0].imshow(over); ax[1].imshow(scene, vmin=.5, vmax=4, cmap="turbo"); ax[2].imshow(np.where(z > 0, z, np.nan), vmin=.5, vmax=4, cmap="turbo")
                for a, t in zip(ax, ("photo + panel outline + hidden mask (red)", "model depth, anchor-scaled (m)", "panel virtual Z (m)")):
                    a.set_title(t, fontsize=9); a.axis("off")
                fig.suptitle(f"{label} frame {n} ({kind}) pose={trk.get(n, '?')} margin=0.02 hidden(anchor scene)={float(hid.sum())/max(int((z > 0).sum()), 1):.3f} panelZ={st[n]['panel_z_median_m']:.2f}m", fontsize=10)
                fig.savefig(IMG/f"{label}_{n:06d}.png", dpi=60, bbox_inches="tight"); plt.close(fig)
        for n in picks[label]["stills"]:
            sp = W/"run"/label/f"{n:06d}-image.png"
            if sp.exists():
                cv2.imwrite(str(IMG/f"still_{label}_{n:06d}.png"), cv2.imread(str(sp)))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    {"select": select, "run": run, "poses": poses, "report": report, "figures": figures}[sys.argv[1]]()
