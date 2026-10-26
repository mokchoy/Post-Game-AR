"""Step 37 (and the run half of Step 38): dense table-plane correction of the model depth, offline, GPU.

`run`: replays each clip through ar_projection.pipeline.run() with DepthPrototype(plane_correction=...)
at one input size / one set of fit terms, writing per-frame json only (no image dumps) plus raw and
scene depth arrays for the Step 36 frame selection. The anchor fit is computed in the same run, so
the same frames give both the anchor-scaled and the plane-corrected depth (and the anchor fallback).
`report`: summarises one or more run directories against the baseline (runs/step27b-patch-median,
the Step 36 baseline) into runs/step37-plane.json/.md. Run via .venv-depth.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ar_projection.occlusion import DepthPrototype, MetricDepth, PlaneParams
from ar_projection.pipeline import run
from step31_masks import CHECKPOINT, CLIPS, MAP_PATH, REPO, clip_config

BASELINE = Path("runs/step27b-patch-median")
STEP36 = Path("runs/step36-depth-sources.json")
OUT = Path("runs/step37-plane")
MARGINS = ("0.02", "0.03", "0.05")


def selected_frames():
    out = {}
    for f in json.loads(STEP36.read_text())["frames"]:
        out.setdefault(f["clip"], set()).add(f["frame_index"])
    return out


def run_all(tag, input_size, terms, clips=None, params=None):
    root = OUT/tag
    sel = selected_frames()
    model = MetricDepth(REPO, CHECKPOINT, input_size=input_size)
    params = params or PlaneParams(terms=terms)
    info = {"tag": tag, "input_size": input_size, "plane_params": asdict(params), "clips": {}}
    for label in clips or CLIPS:
        occlusion = DepthPrototype(model, root/label, slope_sign=1, plane_correction=params, depth_frames=sel.get(label, ()), export_images=False)
        info["clips"][label] = str(run(clip_config(CLIPS[label]), occlusion=occlusion))
        print(tag, label, "done", flush=True)
    (root/"run.json").write_text(json.dumps(info, indent=2))


def frame_stats(directory):
    rows = {}
    for f in Path(directory).glob("[0-9]*.json"):
        rows[int(f.stem)] = json.loads(f.read_text())
    return rows


def baseline_hidden(label):
    """Production (anchor-fit) occluded fraction per frame, from the baseline mask exports."""
    cache = OUT/"baseline.json"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if label not in data:
        import cv2
        d, out = BASELINE/label, {}
        for jp in sorted(d.glob("[0-9]*.json")):
            idx = int(jp.stem)
            z = np.load(d/f"{idx:06d}-virtual-depth.npy") > 0
            mask = cv2.imread(str(d/f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if z.any():
                out[idx] = float(np.mean(mask[z] < 128))
        data[label] = {str(k): v for k, v in out.items()}
        cache.write_text(json.dumps(data))
    return {int(k): v for k, v in data[label].items()}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize_clip(label, stats, base):
    idx = sorted(set(stats) & set(base))
    gate = [stats[i].get("plane_gate", False) for i in idx]
    act = [stats[i] for i in idx if stats[i].get("plane_gate")]
    med = lambda k: float(np.median([a[k] for a in act])) if act else None
    coeffs = np.array([a["plane_held_coeffs"] for a in act]) if act else None
    out = {"frames": len(idx), "gate_pass_fraction": float(np.mean(gate)) if gate else None,
           "fallback_status_counts": {s: sum(stats[i]["status"] == s for i in idx) for s in {stats[i]["status"] for i in idx}},
           "plane_pixels_median": med("plane_pixels"), "plane_fraction_median": med("plane_fraction"),
           "residual_mad_m_median": med("plane_residual_mad_m"), "raw_planarity_r2_median": med("plane_raw_planarity_r2"),
           "margin_3sigma_m_median": med("plane_margin_3sigma_m"),
           "coeffs_median": np.median(coeffs, axis=0).tolist() if act else None,
           "post_bias_row_band_m_median": {b: float(np.median([a["plane_bias_by_row_band_m"][b] for a in act if a["plane_bias_by_row_band_m"][b] is not None])) for b in ("top", "middle", "bottom")} if act else None,
           "baseline_mean_hidden": mean(base[i] for i in idx)}
    hidden = lambda m: [stats[i].get("hidden_fraction_by_margin", {}).get(m) for i in idx]
    out["mean_hidden_by_margin"] = {m: mean(hidden(m)) for m in MARGINS}
    inf = [stats[i].get("inference_ms") for i in idx]
    out["mean_inference_ms"] = mean(inf)
    return out, idx


def windows(label, stats, base, idx, lo, hi):
    sel = [i for i in idx if lo <= i <= hi]
    return {"frames": len(sel), "baseline_mean": mean(base[i] for i in sel),
            "after_mean_by_margin": {m: mean(stats[i].get("hidden_fraction_by_margin", {}).get(m) for i in sel) for m in MARGINS},
            "after_max_0.02": max([stats[i].get("hidden_fraction_by_margin", {}).get("0.02") or 0 for i in sel], default=None)}


def group_of(f):
    if f["clip"] == "V3" and 20 <= f["frame_index"] <= 30 or f["clip"] == "V4" and 1270 <= f["frame_index"] <= 1280:
        return "false-hide"
    n = f.get("note") or ""
    if n.startswith("worst V0") or n.startswith("random V0"):
        return "V0"
    if n == "zero-occlusion control":
        return "controls"
    if f["clip"] == "V2":
        return "V2/441 window"
    return "other"


def report(run_dirs, name="step37-plane"):
    sel = {(f["clip"], f["frame_index"]): f for f in json.loads(STEP36.read_text())["frames"]}
    result = {"checkpoint_sha256": hashlib.sha256(Path(CHECKPOINT).read_bytes()).hexdigest(), "checkpoint": Path(CHECKPOINT).name,
              "map_sha256": hashlib.sha256(Path(MAP_PATH).read_bytes()).hexdigest(),
              "calibration_sha256": hashlib.sha256(Path("logitech-camera-calibration.npz").read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path("ar_projection/occlusion.py").read_bytes()).hexdigest(),
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(), "runs": {}}
    for d in run_dirs:
        d = Path(d)
        info = json.loads((d/"run.json").read_text())
        r = {"input_size": info["input_size"], "plane_params": info["plane_params"], "tracking_runs": info["clips"], "clips": {}, "groups": {}}
        grp = {}
        for label in CLIPS:
            stats, base = frame_stats(d/label), baseline_hidden(label)
            r["clips"][label], idx = summarize_clip(label, stats, base)
            for m, k, lo, hi in (("V3 0-60", "V3", 0, 60), ("V4 1240-1300", "V4", 1240, 1300)):
                if label == k:
                    r["clips"][label][m] = windows(label, stats, base, idx, lo, hi)
            if label == "V0":
                worst = sorted(idx, key=lambda i: -(stats[i].get("hidden_fraction_by_margin", {}).get("0.02") or 0))[:5]
                r["clips"][label]["worst5_after_0.02"] = [round(stats[i]["hidden_fraction_by_margin"]["0.02"], 4) for i in worst if "hidden_fraction_by_margin" in stats[i]]
            if label in ("V1", "V2"):  # no labels exist (Step 32 not done): baseline-hidden > 0.3 is a PROXY for "occluder in front"
                occ = [i for i in idx if base[i] > .3]
                r["clips"][label]["occluder_proxy_frames"] = {"n": len(occ), "baseline_mean": mean(base[i] for i in occ),
                    "after_mean_by_margin": {m: mean(stats[i].get("hidden_fraction_by_margin", {}).get(m) for i in occ) for m in MARGINS}}
            for i in idx:
                f = sel.get((label, i))
                if f:
                    g = group_of(f)
                    grp.setdefault(g, []).append((base[i], stats[i], f))
        for g, items in grp.items():
            r["groups"][g] = {"n": len(items), "baseline_mean": mean(b for b, _, _ in items),
                              "after_mean_by_margin": {m: mean(s.get("hidden_fraction_by_margin", {}).get(m) for _, s, _ in items) for m in MARGINS},
                              "oracle_vs_B_0.02_mean (step36)": mean(f["masks"].get("0.02", {}).get("oracle_vs_B") for _, _, f in items)}
        h = frame_stats(d/"V2").get(441)
        r["V2_441"] = {"baseline": baseline_hidden("V2").get(441), "after_by_margin": h.get("hidden_fraction_by_margin") if h else None, "plane_gate": h.get("plane_gate") if h else None}
        result["runs"][d.name] = r
    Path(f"runs/{name}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--input-size", type=int, default=252)
    r.add_argument("--terms", default="full", choices=("full", "affine", "noint", "fixedb"))
    r.add_argument("--tag")
    r.add_argument("--clips", nargs="*")
    q = sub.add_parser("report")
    q.add_argument("run_dirs", nargs="+")
    q.add_argument("--name", default="step37-plane")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        run_all(a.tag or f"s{a.input_size}-{a.terms}", a.input_size, a.terms, a.clips)
    else:
        report(a.run_dirs, a.name)


if __name__ == "__main__":
    main()
