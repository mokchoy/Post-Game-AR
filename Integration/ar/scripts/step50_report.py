"""Step 50: per-run timing and array summary from frames.jsonl. Usage: step50_report.py RUN [RUN ...]"""
import json, sys
import numpy as np

for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{run}/frames.jsonl")]
    t = lambda k: np.array([r["timings_ms"].get(k) for r in rows if r["timings_ms"].get(k) is not None])
    valid = [r for r in rows if r["actual_pose_source"] != "none"]
    out = {"run": run.rstrip("/").split("/")[-1], "frames": len(rows), "valid_pose": len(valid)}
    for k in ("total", "occlusion", "array_render", "array_content", "gesture_queue", "rendering"):
        v = t(k)
        if len(v):
            out[k+"_ms"] = {"median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2), "n": len(v)}
    arr = [r["diagnostics"]["array"] for r in rows if "array" in r["diagnostics"]]
    if arr:
        out["array_final"] = {k: arr[-1][k] for k in ("yaw_deg", "scale", "game", "playing")}
        out["actions"] = [a for x in arr for a in x["actions"]]
        for n in arr[-1]["panel_pixels"]:
            f = [x["panel_visible_fraction"][n] for x in arr if x["panel_visible_fraction"].get(n) is not None]
            out.setdefault("panel_visible_fraction_mean", {})[n] = round(float(np.mean(f)), 3) if f else None
            px = [x["panel_pixels"].get(n, 0) for x in arr]
            out.setdefault("panel_pixels_median", {})[n] = int(np.median(px))
    print(json.dumps(out, indent=1))
