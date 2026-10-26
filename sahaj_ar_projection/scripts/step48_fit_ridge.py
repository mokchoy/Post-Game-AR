"""Step 48 pre-check (26 Oct): from frames.jsonl only (no depth model), test whether the per-frame relative-depth fit
d = t + s/Z is identifiable. If all (t, s) lie on one line s = Z0*(d0 - t), only one depth (Z0) is constrained and the
fit's slope is free. Usage: python3 scripts/step48_fit_ridge.py runs/<run> [min_camera_distance_m]"""
import json
import sys
import numpy as np

run = sys.argv[1]
min_d = float(sys.argv[2]) if len(sys.argv) > 2 else 1.3
t, s, span, hid = [], [], [], []
for line in open(f"{run}/frames.jsonl"):
    r = json.loads(line)
    o = r["diagnostics"]["occlusion"]
    p = r["raw_camera_position_W"]
    if "fit" in o and p and np.linalg.norm(p) > min_d:
        t.append(o["fit"][0]); s.append(o["fit"][1]); span.append(o.get("line_profile_span", np.nan)); hid.append(o["hidden_fraction"])
t, s, span, hid = map(np.array, (t, s, span, hid))
a, b = np.linalg.lstsq(np.stack([np.ones_like(t), -t], 1), s, rcond=None)[0]
print(f"frames {len(t)}  corr(t,s) {np.corrcoef(t, s)[0, 1]:.3f}  ridge Z0 {b:.2f} m  d0 {a/b:.2f}")
print("line inverse-depth span p10/50/90", np.round(np.nanpercentile(span, [10, 50, 90]), 3))
for d in (2, 3, 5, 8, 12):
    z = np.where(d-t > 0, s/(d-t), np.inf)
    print(f"model d={d:>2}: implied Z p10/50/90 {np.round(np.percentile(z, [10, 50, 90]), 2)}")
big = t < -6
print(f"hidden fraction mean: t<-6 {hid[big].mean():.3f} ({big.sum()} frames), others {hid[~big].mean():.3f}")
