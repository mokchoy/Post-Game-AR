"""Step 38: one-off input-size diagnostic (252 / 392 / 518), on Step 36's 79 frames.

The depth runs come from step37_plane_calibration.py (`run --input-size N --tag ...`), which saves
raw, anchor-scaled and plane-corrected depth for the Step 36 frame selection. This script only
scores those arrays against the Step 36 geometry (oracle table depth, panel Z under pose B and C,
all saved in runs/step36-depth-sources/<clip>_<frame>/). Run via .venv-depth.
Known deviation from Step 36's bias numbers: the tag region is not excluded from the table-pixel
statistics (the tag pose was not saved); the 252 anchor row is printed next to Step 36's own figures
as a sanity check. Writes runs/step38-input-size.json and .md.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from step36_depth_sources import oracle_vs_model
from step37_plane_calibration import STEP36, group_of

S36 = Path("runs/step36-depth-sources")
HAND_POLYGON = ((290, 230), (560, 480))  # step36's rectangle (x0, y0), (x1, y1) for V2/441
MARGINS = (.02, .05)


def hidden(scene, z, margin):
    on = z > 0
    known = np.isfinite(scene[on]) & (scene[on] > 0)
    return float(np.mean(known & (scene[on]+margin < z[on]))) if on.any() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def score_frame(d, scene, f):
    oz, ov = np.load(d/"oracle_z.npy").astype(np.float64), np.load(d/"oracle_valid.npy")
    zb = np.load(d/"panel_z_B.npy") if (d/"panel_z_B.npy").exists() else None
    zc = np.load(d/"panel_z_C.npy") if (d/"panel_z_C.npy").exists() else None
    out = {}
    panel = zb > 0 if zb is not None else np.zeros(oz.shape, bool)
    out["bias"] = oracle_vs_model(scene.astype(np.float64), oz, ov, panel, np.zeros(oz.shape, bool))
    oracle_scene = np.where(ov, oz, 0.)
    for m in MARGINS:
        for pose, z in (("B", zb), ("C", zc)):
            if z is not None:
                out[f"model_vs_{pose}_{m}"] = hidden(scene, z, m)
                out[f"oracle_vs_{pose}_{m}"] = hidden(oracle_scene, z, m)
    return out


def hand(d, scene, raw):
    zc = np.load(d/"panel_z_C.npy")
    oz, ov = np.load(d/"oracle_z.npy"), np.load(d/"oracle_valid.npy")
    (x0, y0), (x1, y1) = HAND_POLYGON
    m = np.zeros(scene.shape, bool)
    m[y0:y1, x0:x1] = True
    under = scene[m]
    return {"scene_under_hand_median_m": float(np.median(under)), "p5": float(np.percentile(under, 5)), "p95": float(np.percentile(under, 95)),
            "panel_C_z_median_m": float(np.median(zc[zc > 0])), "oracle_table_under_hand_median_m": float(np.median(oz[m & ov])),
            "hand_reads_nearer_than_panel": bool(np.median(under) < np.median(zc[zc > 0])), "raw_under_hand_median": float(np.median(raw[m]))}


def agg(rows):
    """rows: list of score_frame outputs -> group means."""
    b = [r["bias"] for r in rows if r["bias"]]
    out = {"n": len(rows), "bias_median_m": mean(x["median_m"] for x in b),
           "bias_row_band_median_m": {k: mean(x["row_band_median_m"][k] for x in b) for k in ("top", "middle", "bottom")},
           "bias_plane_dz_dy_m_per_480px": mean(x["bias_plane_fit"]["dz_dy_per_px"]*480 for x in b if x["bias_plane_fit"]),
           "bias_abs_p25_p75_spread_m": mean(x["p75_m"]-x["p25_m"] for x in b)}
    for k in rows[0]:
        if k != "bias":
            out[k] = mean(r[k] for r in rows)
    return out


def run(run_dirs, name="step38-input-size"):
    sel = json.loads(STEP36.read_text())["frames"]
    result = {"sizes": {}}
    for rd in run_dirs:
        rd = Path(rd)
        info = json.loads((rd/"run.json").read_text())
        res = {"input_size": info["input_size"], "plane_terms": info["plane_params"]["terms"], "inference_ms": None, "groups": {}, "hand_441": {}}
        for source in ("anchor", "plane"):
            groups = {}
            for f in sel:
                d = S36/f"{f['clip']}_{f['frame_index']:06d}"
                a = rd/f["clip"]/f"{f['frame_index']:06d}-scene-{source}.npy"
                if not a.exists() or not (d/"oracle_z.npy").exists():
                    continue
                groups.setdefault(group_of(f), []).append(score_frame(d, np.load(a), f))
                if f["clip"] == "V2" and f["frame_index"] == 441:
                    res["hand_441"][source] = hand(d, np.load(a), np.load(rd/f["clip"]/f"{f['frame_index']:06d}-raw.npy"))
            res["groups"][source] = {g: agg(rows) for g, rows in groups.items()}
        result["sizes"][f"{info['input_size']}-{rd.name}"] = res
    Path(f"runs/{name}.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    run(p.parse_args().run_dirs)
