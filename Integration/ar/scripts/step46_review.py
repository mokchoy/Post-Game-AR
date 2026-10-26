"""Step 46 task 4: review of the five Step 45 live runs from their frames.jsonl (no model needed), plus a replay of the
last run to get tag_log. Usage: python3 scripts/step46_review.py review | replay   (replay needs .venv-depth: no, plain .venv)
Writes runs/step46-live-review.json."""
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

RUNS = ["20261025T081445-adcde7c8", "20261025T081546-aa877612", "20261025T081836-22c82ea7", "20261025T083944-75475811", "20261025T084746-303f35ec"]
BANDS = (("<1.5m", 0, 1.5), ("1.5-2.5m", 1.5, 2.5), (">2.5m", 2.5, 99))


def load(name):
    return [json.loads(l) for l in (Path("runs")/name/"frames.jsonl").read_text().splitlines()]


def dist(r):
    p = r["raw_camera_position_W"]
    return float(np.linalg.norm(p)) if p is not None else None


def band(d):
    return next((n for n, lo, hi in BANDS if d is not None and lo <= d < hi), "no pose")


def corners(r):
    return np.array(r["raw_panel_pixels"]) if r["raw_panel_pixels"] is not None and r["actual_pose_source"] != "none" else None


def still_windows(rows, span=1.0):
    """Windows of >= span s with valid pose, one pose source, camera position range < 1.5 cm. Returns (source, jitter_px)."""
    out, i = [], 0
    t = [r["source_time_s"] for r in rows]
    while i < len(rows):
        j = i
        while j+1 < len(rows) and rows[j+1]["actual_pose_source"] == rows[i]["actual_pose_source"] != "none" and corners(rows[j+1]) is not None:
            j += 1
        if rows[i]["actual_pose_source"] != "none" and t[j]-t[i] >= span:
            a, b = i, i
            while a < j:                               # greedy: grow a window while the camera barely moves
                b = a
                while b+1 <= j and np.ptp([rows[k]["raw_camera_position_W"] for k in range(a, b+2)], axis=0).max() < .015:
                    b += 1
                if t[b]-t[a] >= span:
                    c = np.array([corners(rows[k]) for k in range(a, b+1)])
                    out.append((rows[a]["actual_pose_source"], float(np.sqrt((c.std(axis=0)**2).mean())), dist(rows[a])))
                a = b+1
        i = j+1
    return out


def review(name):
    rows = load(name)
    n = len(rows)
    res = {"frames": n, "duration_s": rows[-1]["source_time_s"]-rows[0]["source_time_s"] if rows[0]["source_time_s"] is not None else None}
    src = [r["actual_pose_source"] for r in rows]
    res["pose_source_counts"] = dict(Counter(src))
    res["by_distance_band"] = {}
    for b, lo, hi in BANDS:
        sel = [r for r in rows if r["actual_pose_source"] != "none" and lo <= dist(r) < hi]
        res["by_distance_band"][b] = {"frames": len(sel), "sources": dict(Counter(r["actual_pose_source"] for r in sel)),
                                       "landmark_inliers_median": float(np.median([r["raw_pose"]["landmark_inliers"] for r in sel])) if sel else None}
    det = [r for r in rows if any(t["id"] == 0 for t in r["diagnostics"]["tags"])]
    res["tag_detected_fraction"] = len(det)/n
    res["when_tag_detected"] = dict(Counter(("valid" if t["valid"] else "invalid:"+str(t["reason"])) + (" ambiguous" if t["ambiguous"] else "") + " -> " + r["actual_pose_source"]
                                              for r in det for t in r["diagnostics"]["tags"] if t["id"] == 0))
    res["tag_valid_not_used_reason"] = dict(Counter(str(r["raw_pose"]["reason"]) for r in det if r["actual_pose_source"] in ("landmark", "none")
                                                     and any(t["id"] == 0 and t["valid"] for t in r["diagnostics"]["tags"])))
    res["tag_rms_px_of_joint_median"] = float(np.median([r["raw_pose"]["tag_rms"] for r in rows if r["actual_pose_source"] == "joint"] or [np.nan]))
    # loss
    gaps, cur = [], None
    for r in rows:
        if r["actual_pose_source"] == "none":
            cur = cur or [r["source_time_s"], r["source_time_s"]]; cur[1] = r["source_time_s"]
        elif cur:
            gaps.append(cur[1]-cur[0]); cur = None
    if cur:
        gaps.append(cur[1]-cur[0])
    res["loss"] = {"frames_fraction": src.count("none")/n, "gaps": len(gaps), "gap_s_median": float(np.median(gaps)) if gaps else 0, "gap_s_max": float(max(gaps, default=0))}
    res["loss_reasons"] = dict(Counter((r["diagnostics"].get("pose_gates", {}).get("landmark_gate_reason") or r["raw_pose"]["reason"]) for r in rows if r["actual_pose_source"] == "none").most_common(6))
    # still-hold jitter
    w = still_windows(rows)
    res["still_hold"] = {s: {"windows": sum(x[0] == s for x in w), "corner_std_px_median": float(np.median([x[1] for x in w if x[0] == s])),
                             "corner_std_px_p90": float(np.percentile([x[1] for x in w if x[0] == s], 90))} for s in {x[0] for x in w}}
    # switches
    sw = {}
    prev = None
    for r in rows:
        c = corners(r)
        if c is not None and prev is not None and prev[0] != r["actual_pose_source"] and r["frame_index"] == prev[2]+1:
            sw.setdefault(f"{prev[0]}->{r['actual_pose_source']}", []).append(float(np.median(np.linalg.norm(c-prev[1], axis=1))))
        prev = (r["actual_pose_source"], c, r["frame_index"]) if c is not None else None
    res["switch_jump_px"] = {k: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))} for k, v in sw.items() if len(v) >= 3}
    res["occlusion_status"] = dict(Counter(r["diagnostics"].get("occlusion", {}).get("status", "absent") for r in rows))
    res["max_distance_by_source_m"] = {s: float(max(dist(r) for r in rows if r["actual_pose_source"] == s)) for s in set(src) - {"none"}}
    res["ms_per_frame_median"] = float(np.median([r["timings_ms"]["total"] for r in rows]))
    return res


def replay(name):
    """Replay a live run headless with the current code (tag_log on); record every tag observation to compare the tag with the pose."""
    from dataclasses import replace
    from ar_projection.config import load_config
    from ar_projection.geometry import camera_position, project
    from ar_projection.pipeline import run
    from ar_projection.tags import TagDetector
    seen = []
    orig = TagDetector.detect
    def detect(self, gray, camera):
        obs = orig(self, gray, camera)
        seen.append([(o.pixels.copy(), o.world, [c.copy() for c in o.candidates], o.valid, o.ambiguous) for o in obs if o.id == self.cfg.anchor_id])
        return obs
    TagDetector.detect = detect
    rd = Path("runs")/name
    cfg = load_config(rd/"config.json")
    cfg = replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rd.resolve()/"input.avi"), timestamps_path=str(rd.resolve()/"frames.jsonl")),
                  display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))
    out = run(cfg)
    rows = [json.loads(l) for l in (Path(out)/"frames.jsonl").read_text().splitlines()]
    fx = cfg.camera and json.loads((rd/"metadata.json").read_text())["calibration"]["K"][0][0]
    K = np.array(json.loads((rd/"metadata.json").read_text())["calibration"]["K"])
    class Cam: pass
    cam = Cam(); cam.K, cam.distortion = K, np.zeros(5)
    res = {"replay_run": str(out), "frames": len(rows), "tag_log_status": dict(Counter(r["diagnostics"]["tag_log"]["anchor_status"] for r in rows)),
           "ids_rejected_by_reason": dict(Counter(r["diagnostics"]["tag_log"]["anchor_status"] for r in rows if r["actual_pose_source"] in ("landmark", "none")))}
    side, impl, trk, rms_lm, dcm, band_rows = [], [], [], [], [], []
    for r, obs in zip(rows, seen):
        if not obs:
            continue
        px, world, cands, valid, amb = obs[0]
        sides = np.linalg.norm(px-np.roll(px, 1, axis=0), axis=1)
        if r["actual_pose_source"] in ("tag", "joint") and r["raw_pose"]["T_CW"] is not None:
            side.append(sides.mean()); impl.append(fx*0.17/sides.mean()); trk.append(float(np.linalg.norm(camera_position(np.array(r["raw_pose"]["T_CW"])))))
        if valid and r["actual_pose_source"] == "landmark" and r["raw_pose"]["T_CW"] is not None:
            T = np.array(r["raw_pose"]["T_CW"])
            rms_lm.append(float(np.sqrt(np.mean(np.sum((project(T, world, cam)-px)**2, axis=1)))))
            dcm.append(float(np.linalg.norm(camera_position(T)-camera_position(cands[0])))*100)
    arr = lambda x: {"n": len(x), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))} if x else None
    res["scale_check"] = {"n_tag_or_joint_frames": len(side), "implied_distance_from_tag_size_m": arr(impl), "tracker_distance_m": arr(trk),
                          "ratio_tracker_over_implied_median": float(np.median(np.array(trk)/np.array(impl))) if impl else None,
                          "max_tracker_distance_tag_or_joint_m": float(max(trk, default=0)), "max_implied_distance_m": float(max(impl, default=0))}
    res["landmark_pose_vs_valid_tag"] = {"tag_corner_rms_under_landmark_pose_px": arr(rms_lm), "camera_position_gap_to_tag_pose_cm": arr(dcm)}
    return res


if __name__ == "__main__":
    out = Path("runs/step46-live-review.json")
    data = json.loads(out.read_text()) if out.exists() else {}
    if sys.argv[1] == "review":
        data["runs"] = {n: review(n) for n in RUNS}
    elif sys.argv[1] == "replay":
        data.setdefault("replays", {})[sys.argv[2]] = replay(sys.argv[2])
    out.write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=1)[:6000])
