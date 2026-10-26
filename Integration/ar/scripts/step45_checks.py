"""Step 45 offline checks for RelativeLineOcclusion (GPU, no camera). Run via .venv-depth.
  a  F-occ 400 frames vs runs/step44-baseline/F-occ   b  F-empty 400 frames, mean hidden
  c  occlusion disabled == no occlusion (pose/panel)   d  ms per frame, display-draw on/off
Writes runs/step45-checks.json."""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ar_projection.occlusion import RelativeDepth, RelativeLineOcclusion
from ar_projection.pipeline import run
from step31_masks import REPO
from step40_far import CLIPS, far_config
from step43_relative import REL_CKPT

N = 400
model = RelativeDepth(REPO, REL_CKPT, 252)


def replay(label, n=N, occ=True, enabled=True, draw=False):
    cfg = far_config(CLIPS[label])
    if draw:
        cfg = replace(cfg, logging=replace(cfg.logging, save_annotated_video=True))
    o = None
    if occ:
        o = RelativeLineOcclusion(model, None, .10, .10, True, 30, REL_CKPT)
        o.enabled = enabled
    d = run(cfg, max_frames=n, occlusion=o)
    return [json.loads(l) for l in (Path(d)/"frames.jsonl").read_text().splitlines()], str(d)


def a(rows):
    ref = {int(p.stem): json.loads(p.read_text()) for p in Path("runs/step44-baseline/F-occ").glob("[0-9]*.json")}
    diffs, src, srcref, n = [], {}, {}, 0
    for i, r in enumerate(rows):
        o, s = r["diagnostics"].get("occlusion", {}), ref.get(r.get("frame_index", i))
        if s is None or "hidden_line" not in s:
            continue
        n += 1
        srcref[s["line_source"]] = srcref.get(s["line_source"], 0)+1
        k = o.get("alignment_source", "unavailable"); src[k] = src.get(k, 0)+1
        diffs.append(abs(o.get("hidden_fraction", np.nan)-s["hidden_line"]))
    d = np.array(diffs)
    return {"active_frames_ref": n, "frames_within_0.5pct": float(np.mean(d <= .005)), "nan_frames": int(np.isnan(d).sum()), "max_abs_diff": float(np.nanmax(d)),
            "source_counts_prod": src, "source_counts_step44": srcref}


def ms(rows):
    t = lambda k: float(np.mean([r["timings_ms"][k] for r in rows if k in r["timings_ms"]]))
    return {k: t(k) for k in ("total", "occlusion") if any(k in r["timings_ms"] for r in rows)} | {"all_keys": sorted(rows[0]["timings_ms"])}


if __name__ == "__main__":
    out = {}
    rows, d = replay("F-occ")
    out["a"] = a(rows); out["d_occ_draw_off"] = ms(rows); out["a_run"] = d
    rows, d = replay("F-empty")
    h = [r["diagnostics"]["occlusion"].get("hidden_fraction") for r in rows]
    h = [x for x in h if x is not None]
    out["b"] = {"frames_with_hidden": len(h), "mean_hidden": float(np.mean(h)), "run": d}
    base, _ = replay("F-occ", 150, occ=False)
    off, _ = replay("F-occ", 150, enabled=False)
    key = lambda r: (r["state"], r["actual_pose_source"], json.dumps(r["render_T_CW"]))
    out["c"] = {"frames": len(base), "identical_pose_state": all(key(x) == key(y) for x, y in zip(base, off)), "n_off": len(off)}
    rows, d = replay("F-occ", 150, draw=True)
    out["d_occ_draw_on"] = ms(rows)
    out["d_no_occ"] = ms(base)
    out["d_occ_disabled"] = ms(off)
    Path("runs/step45-checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
