"""Pose-source jitter table from one recorded run: tag only, landmark only, joint, joint + filter.

usage: ../.venv/bin/python scripts/jitter_report.py ../runs/<recording> [--reuse tag=<run> landmark=<run>]
The recording's own run is the joint (+ filter) run: frames.jsonl holds the raw pose and the filtered render pose.
Tag-only is a tag_only replay; landmark-only is a joint replay with the tag withheld after LEAD_S seconds. Both are replayed from RUN/input.avi. Each column is the std over frames of the distance from the
mean (and the RMS of the same distance); rotation = angle of the frame pose from the mean rotation, translation = camera
position, marker = world origin (tag centre) projected through the pose, corners = the panel's four projected corners.
Only frames valid for all six sources are used, so the columns are comparable."""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent.parent
LEAD_S = 2.0   # landmark-only needs the tag to seed the pose; it is withheld after this many seconds (pure landmark_only never bootstraps)


def replay(rec, mode, *extra):
    out = subprocess.run([sys.executable, "april-tag-projection.py", "--replay-run", str(rec), "--headless", "--no-gestures",
                          "--pose-mode", mode, *extra], cwd=HERE, capture_output=True, text=True)
    found = re.search(r"Run saved to (.+?);", out.stderr+out.stdout)
    if not found:
        raise SystemExit(f"{mode} replay failed:\n{out.stderr[-800:]}")
    return Path(found.group(1))


def load(run):
    return [json.loads(line) for line in open(run/"frames.jsonl")]


def pose_rows(rows, filtered):
    out = {}
    for r in rows:
        T = r["render_T_CW"] if filtered else (r["raw_pose"]["T_CW"] if r["raw_pose"]["valid"] else None)
        panel = r["render_panel_pixels" if filtered else "raw_panel_pixels"]
        if T is not None and panel is not None and len(panel) == 4:
            out[r["frame_index"]] = (np.array(T, float), np.array(panel, float))
    return out


def spread(x):
    d = np.linalg.norm(x-x.mean(0), axis=1)
    return float(d.std()), float(np.sqrt((d**2).mean()))


def main():
    rec = Path(sys.argv[1]).resolve()
    reuse = dict(a.split("=") for a in sys.argv[3:]) if len(sys.argv) > 2 and sys.argv[2] == "--reuse" else {}
    K = np.array(json.load(open(rec/"metadata.json"))["calibration"]["K"], float)
    base = load(rec)
    tag = load(Path(reuse["tag"]) if "tag" in reuse else replay(rec, "tag_only"))
    lm = load(Path(reuse["landmark"]) if "landmark" in reuse else replay(rec, "joint", "--withhold-tag-windows", f"{base[0]['source_time_s']+LEAD_S}:{base[-1]['source_time_s']+1}"))
    sources = {}                                    # each replay logs the raw pose and the filtered render pose
    for name, rows in (("tag only", tag), ("landmark only", lm), ("joint", base)):
        sources[name], sources[name+" + filter"] = pose_rows(rows, False), pose_rows(rows, True)
    common = sorted(set.intersection(*(set(v) for v in sources.values())))
    table = {}
    for name, rows in sources.items():
        T = np.array([rows[i][0] for i in common])
        Rm = Rotation.from_matrix(T[:, :3, :3])
        angle = np.degrees((Rm.mean().inv()*Rm).magnitude())
        centre = -np.einsum("nji,nj->ni", T[:, :3, :3], T[:, :3, 3])*1000.        # camera position in the tag frame, mm
        uv = T[:, :3, 3]@K.T
        marker = uv[:, :2]/uv[:, 2:]                                              # tag centre = world origin
        corners = np.array([rows[i][1] for i in common])                         # (n, 4, 2)
        per_corner = [spread(corners[:, k]) for k in range(4)]
        table[name] = {"frames": len(common), "valid_frames": len(rows),
                       "rotation_deg": [float(angle.std()), float(np.sqrt((angle**2).mean()))],
                       "translation_mm": list(spread(centre)), "marker_px": list(spread(marker)),
                       "corner_px": [float(np.mean([c[0] for c in per_corner])), float(np.mean([c[1] for c in per_corner]))]}
    (rec.parent/f"jitter_{rec.name}.json").write_text(json.dumps(table, indent=1))
    lines = [f"Recording {rec.name}: {len(common)} frames valid for all six sources (of {len(base)}). std (RMS) of distance from the mean.", "",
             "| source | valid frames | rotation (deg) | translation (mm) | marker centre (px) | panel corners (px) |", "|---|---|---|---|---|---|"]
    for name, t in table.items():
        f = lambda k: f"{t[k][0]:.3f} ({t[k][1]:.3f})"
        lines.append(f"| {name} | {t['valid_frames']} | {f('rotation_deg')} | {f('translation_mm')} | {f('marker_px')} | {f('corner_px')} |")
    (rec.parent/f"jitter_{rec.name}.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
