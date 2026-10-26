"""Step 40: far-clip evaluation. Replays F-occ and F-empty with config-logitech-far.json (min_side_px 25,
0.60 x 0.33707 m panel) and the near map, exports per-frame json (no images) in mode (a) anchor fit and
(b) Step 37 plane correction (noint, 0.8-4 m calibration range). `report` prints pose-quality and hidden-
fraction tables. No labels exist yet (Step 41): hidden fraction on F-empty is the false-hide gate; no miss
rate is possible. Run via .venv-depth. Writes runs/step40-far/ and runs/step40-far.json."""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ar_projection.config import load_config
from ar_projection.occlusion import DepthPrototype, MetricDepth, PlaneParams
from ar_projection.pipeline import run
from step31_masks import CHECKPOINT, MAP_PATH, REPO

CLIPS = {"F-occ": "runs/20261025T000122-210c1bb8", "F-empty": "runs/20261025T000913-e058a920"}
OUT = Path("runs/step40-far")
MODES = {"anchor": None, "plane": PlaneParams(terms="noint", z_min_m=.8, z_max_m=4.0)}


def far_config(directory):
    rec = Path(directory).resolve()
    cfg = load_config("config-logitech-far.json")
    # Step 46 changed the far config to the 1.0 m panel with smoothing on; these old replays keep the Step 40-45 settings
    cfg = replace(cfg, panel=replace(cfg.panel, width_m=.60, height_m=.33707), filter=replace(cfg.filter, enabled=False))
    return replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                   mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(MAP_PATH).resolve())),
                   display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))


def run_all():
    model = MetricDepth(REPO, CHECKPOINT, input_size=252)
    info = {}
    for label, d in CLIPS.items():
        for mode, params in MODES.items():
            occ = DepthPrototype(model, OUT/mode/label, slope_sign=1, plane_correction=params, export_images=False)
            info[f"{label}/{mode}"] = str(run(far_config(d), occlusion=occ))
            print(label, mode, "done", flush=True)
    (OUT/"run.json").write_text(json.dumps(info, indent=2))


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def report():
    info = json.loads((OUT/"run.json").read_text())
    res = {}
    for label in CLIPS:
        rows = [json.loads(l) for l in open(Path(info[f"{label}/anchor"])/"frames.jsonl")]
        src = {}
        for r in rows:
            src[r["actual_pose_source"]] = src.get(r["actual_pose_source"], 0)+1
        tags = [t for r in rows for t in r["diagnostics"].get("tags", []) if t["id"] == 0]
        rms = [t["candidate_rms_px"][0] for t in tags if t["valid"] and t["candidate_rms_px"]]
        e = {"frames": len(rows), "pose_sources": src, "tag_valid_fraction": sum(t["valid"] for t in tags)/len(rows),
             "tag_ambiguous_fraction_of_valid": float(np.mean([t["ambiguous"] for t in tags if t["valid"]])) if rms else None,
             "tag_rms_px_median_p95": [pct(rms, 50), pct(rms, 95)]}
        for mode in MODES:
            st = {int(f.stem): json.loads(f.read_text()) for f in (OUT/mode/label).glob("[0-9]*.json")}
            h = {m: [s["hidden_fraction_by_margin"][m] for s in st.values() if "hidden_fraction_by_margin" in s] for m in ("0.02", "0.03", "0.05")}
            act = [s for s in st.values() if s.get("plane_gate")]
            e[mode] = {"exported_frames": len(st), "mask_active_fraction": len(h["0.02"])/len(st),
                       "hidden_mean_by_margin": {m: float(np.mean(v)) for m, v in h.items()},
                       "hidden_frac_over_5pct@0.02": float(np.mean(np.array(h["0.02"]) > .05)), "hidden_p95@0.02": pct(h["0.02"], 95),
                       "anchor_count_median": float(np.median([s.get("scale_anchor_count", 0) for s in st.values()]))}
            if mode == "plane":
                e[mode].update(gate_pass_fraction=len(act)/len(st), residual_mad_m_median=float(np.median([s["plane_residual_mad_m"] for s in act])) if act else None,
                               plane_pixels_median=float(np.median([s["plane_pixels"] for s in act])) if act else None,
                               post_bias_row_band_m={b: float(np.median([s["plane_bias_by_row_band_m"][b] for s in act if s["plane_bias_by_row_band_m"][b] is not None] or [np.nan])) for b in ("top", "middle", "bottom")} if act else None,
                               raw_planarity_r2_median=float(np.median([s["plane_raw_planarity_r2"] for s in act])) if act else None)
        res[label] = e
    Path("runs/step40-far.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report() if sys.argv[1:] == ["report"] else run_all()
