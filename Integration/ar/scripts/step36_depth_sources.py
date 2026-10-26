"""Step 36: depth-source comparison on the false-hide and miss frames (diagnosis only, no
behaviour change to occlusion.py's defaults, fit, anchors or margin).

Reuses, rather than recomputes, what's already on disk from the current production export
(runs/step27b-patch-median/, the metric checkpoint + slope_sign=1 + patch-median/edge-rejection
anchors, the live defaults as of 2026-10-24): its tracking_run frames.jsonl gives pose B
(render_T_CW), tag quality, and pose_source for free; its <label>/<idx>-image.png gives the
rectified frame for free; its <label>/<idx>.json gives the already-fitted (a, b) for free. Only
three things are computed fresh here: C (a tag-only pose refit per frame) and the landmark-only
pose B_landmark, both via ONE --withhold-tag replay per clip (pipeline.py already computes both
as part of its own withhold-tag diagnostics, reused as-is); and the oracle table-plane depth via
the new occlusion.plane_depth() helper. Run via .venv-depth. Writes runs/step36-depth-sources/.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.camera import Capture
from ar_projection.config import load_config
from ar_projection.experiments import percentile
from ar_projection.geometry import panel_points, project, tag_points
from ar_projection.occlusion import MetricDepth, panel_layer, plane_depth
from ar_projection.pipeline import Pipeline

REPO = "/home/sahaj/Desktop/Studies/compvis/Depth-Anything-V2"
CHECKPOINT = REPO+"/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
MAP_PATH = "data/logitech-landmark-map-occ.npz"
MASK_EXPORT = Path("runs/step27b-patch-median")  # the live/current production export: B, tag quality, A's (a,b)
OUT_ROOT = Path("runs/step36-depth-sources")
CLIPS = {"V0": "runs/20261024T051838-1eaba384", "V1": "runs/20261024T052053-d73edfab",
         "V2": "runs/20261024T052253-e8a2ad05", "V3": "runs/20261024T054241-016203dd",
         "V4": "runs/20261024T052603-9e9fecca"}
MARGINS = (.02, .05)
RNG = np.random.default_rng(0)


# ---------- frame selection (task 1): reads MASK_EXPORT's own exports, no replay ----------

def occluded_fraction(label, idx):
    d = MASK_EXPORT/label
    jp = d/f"{idx:06d}.json"
    if not jp.exists():
        return None
    mask = cv2.imread(str(d/f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
    z = np.load(d/f"{idx:06d}-virtual-depth.npy")
    panel = z > 0
    if not np.any(panel):
        return None
    return float(np.mean(mask[panel] < 128))


def tag_quality(tracking_rows, idx, anchor_id=0):
    row = tracking_rows.get(idx)
    if row is None:
        return None
    tags = [t for t in row["diagnostics"].get("tags", []) if t["id"] == anchor_id]
    if not tags or not tags[0]["valid"]:
        return {"ok": False, "reason": "tag not valid"}
    t = tags[0]
    rms = t["candidate_rms_px"][0] if t["candidate_rms_px"] else None
    ok = (not t["ambiguous"]) and rms is not None and rms <= 1.0
    return {"ok": ok, "rms_px": rms, "ambiguous": t["ambiguous"]}


def load_tracking_rows(tracking_run):
    rows = [json.loads(l) for l in open(Path(tracking_run)/"frames.jsonl")]
    return {r["frame_index"]: r for r in rows}


def pick_with_substitution(label, idx, tracking_rows, behaviour_test):
    """If idx's tag quality is poor, search +-30 frames for one that's good AND shows the same
    behaviour (per behaviour_test(occluded_fraction) -> bool). Returns (chosen_idx, note)."""
    q = tag_quality(tracking_rows, idx)
    if q and q["ok"]:
        return idx, None
    for d in sorted(range(1, 31), key=lambda d: d):
        for cand in (idx-d, idx+d):
            qc = tag_quality(tracking_rows, cand)
            frac = occluded_fraction(label, cand)
            if qc and qc["ok"] and frac is not None and behaviour_test(frac):
                return cand, f"substituted for {idx} (tag quality {q}), distance {d} frames"
    return idx, f"no good-quality neighbour found within 30 frames of {idx} (tag quality {q}); using as-is"


def select_frames():
    selection = {}  # label -> {idx: note_or_None}
    tracking = {label: load_tracking_rows(json.loads((MASK_EXPORT/"summary.json").read_text())["variants"][label]["tracking_run"])
                for label in CLIPS}

    def add(label, idx, note=None):
        selection.setdefault(label, {})[idx] = note

    # V3: frame 26 plus the 20-30 window, substituting 26 if needed for a nearby false-hide frame
    v3_idx, note = pick_with_substitution("V3", 26, tracking["V3"], lambda f: f > .3)
    add("V3", v3_idx, note or "primary false-hide frame")
    for i in range(20, 31):
        add("V3", i)

    # V4: frame 1276 plus the 1270-1280 window
    v4_idx, note = pick_with_substitution("V4", 1276, tracking["V4"], lambda f: f > .3)
    add("V4", v4_idx, note or "primary false-hide frame")
    for i in range(1270, 1281):
        add("V4", i)

    # V2: the hand-miss frame and the hand-clear frame. The tag is, by construction, covered at
    # 441 (that's the whole point of the test) so C is unavailable there; also pull in a small
    # window so task 6 can report a nearby tag-valid frame's C/oracle as a labelled reference
    # for "where the panel/table should be", not substituting for 441 itself.
    add("V2", 441, "hand on panel (known miss)")
    add("V2", 72, "hand clear of panel")
    for i in range(431, 452):
        add("V2", i, "window around the hand-miss frame, for a nearby C reference")

    # V0: worst 5 and 5 random (should all read ~0%)
    fracs = [(occluded_fraction("V0", i), i) for i in sorted(tracking["V0"])]
    fracs = [(f, i) for f, i in fracs if f is not None]
    fracs.sort(reverse=True)
    for f, i in fracs[:5]:
        add("V0", i, f"worst V0 (occluded={f:.1%})")
    rest = [i for f, i in fracs[5:]]
    for i in RNG.choice(rest, size=min(5, len(rest)), replace=False):
        add("V0", int(i), "random V0 control")

    # 5 zero-occlusion control frames per clip
    for label in CLIPS:
        zeros = [i for i in sorted(tracking[label]) if occluded_fraction(label, i) == 0.0]
        for i in RNG.choice(zeros, size=min(5, len(zeros)), replace=False) if zeros else []:
            add(label, int(i), "zero-occlusion control")

    return selection, tracking


# ---------- per-clip withhold-tag replay: gives B_landmark and C together (task: pose agreement) ----------

def clip_config(directory, withhold):
    rec = Path(directory).resolve()
    cfg = load_config("config-logitech.json")
    return replace(cfg, camera=replace(cfg.camera, source_type="video", video_path=str(rec/"input.avi"), timestamps_path=str(rec/"frames.jsonl")),
                  mapping=replace(cfg.mapping, mode="localize", map_path=str(Path(MAP_PATH).resolve())),
                  display=replace(cfg.display, enabled=False)), withhold


def withhold_replay(directory, target_indices):
    cfg, _ = clip_config(directory, True)
    capture = Capture(cfg.camera)
    packet = capture.read()
    pipeline = Pipeline(cfg, packet.camera, background_mapping=False)
    camera = packet.camera
    out = {}
    max_idx = max(target_indices)
    while packet is not None and packet.index <= max_idx:
        result = pipeline.process(packet, withhold_tag=True)
        if packet.index in target_indices:
            d = result.diagnostics
            out[packet.index] = {
                "B_landmark_T_CW": result.render_pose.tolist() if result.render_pose is not None else None,
                "B_landmark_valid": bool(result.raw_pose.valid),
                "B_landmark_landmark_inliers": result.raw_pose.landmark_inliers,
                "B_landmark_rms_px": result.raw_pose.landmark_rms,
                "ransac_inliers": d.get("pose_gates", {}).get("ransac_inliers"),
                "mapped_observations": d.get("mapped_observations"),
                "C_T_CW": d.get("tag_reference_T_CW"),
                "C_valid": bool(d.get("tag_reference_valid", False)),
                "C_rms_px": d.get("tag_reference_rms_px"),
                "bgr": packet.bgr.copy(),
            }
        packet = capture.read()
    capture.close()
    return out, camera


# ---------- per-frame numeric computation (tasks 3-6) ----------

def panel_stats(T, panel, camera):
    if T is None:
        return None
    T = np.array(T)
    _, visible, z = panel_layer(T, panel, camera)
    pixels = project(T, panel, camera)
    return {"T_CW": T.tolist(), "corners_px": pixels.tolist(), "z_median_m": float(np.median(z[z > 0])) if np.any(z > 0) else None,
            "z_min_m": float(z[z > 0].min()) if np.any(z > 0) else None, "z_max_m": float(z[z > 0].max()) if np.any(z > 0) else None,
            "visible": visible, "z": z}


def pose_agreement(b_stats, c_stats):
    if b_stats is None or c_stats is None:
        return None
    Tb, Tc = np.array(b_stats["T_CW"]), np.array(c_stats["T_CW"])
    cam_b, cam_c = -Tb[:3, :3].T@Tb[:3, 3], -Tc[:3, :3].T@Tc[:3, 3]
    r_diff = Tb[:3, :3]@Tc[:3, :3].T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r_diff)-1)/2, -1, 1))))
    corners_b, corners_c = np.array(b_stats["corners_px"]), np.array(c_stats["corners_px"])
    corner_diff = np.linalg.norm(corners_b-corners_c, axis=1)
    both_valid = (b_stats["z"] > 0) & (c_stats["z"] > 0)
    z_diff = np.abs(b_stats["z"]-c_stats["z"])[both_valid]
    return {"camera_position_diff_m": float(np.linalg.norm(cam_b-cam_c)), "rotation_diff_deg": angle,
            "corner_px_median": float(np.median(corner_diff)), "corner_px_max": float(np.max(corner_diff)),
            "panel_z_diff_median_m": float(np.median(z_diff)) if np.any(both_valid) else None,
            "panel_z_diff_max_m": float(np.max(z_diff)) if np.any(both_valid) else None}


def oracle_vs_model(model_scaled, oracle_z, oracle_valid, panel_mask, tag_mask):
    exclude = panel_mask | tag_mask
    region = oracle_valid & ~exclude & np.isfinite(model_scaled) & (model_scaled > 0)
    if not np.any(region):
        return None
    diff_full = model_scaled-oracle_z
    diff = diff_full[region]
    h, w = oracle_z.shape
    bands = {}
    for name, sl in (("top", slice(0, h//3)), ("middle", slice(h//3, 2*h//3)), ("bottom", slice(2*h//3, h))):
        d = diff_full[sl, :][region[sl, :]]
        bands[name] = float(np.median(d)) if d.size else None
    col_bands = {}
    for name, sl in (("left", slice(0, w//3)), ("center", slice(w//3, 2*w//3)), ("right", slice(2*w//3, w))):
        d = diff_full[:, sl][region[:, sl]]
        col_bands[name] = float(np.median(d)) if d.size else None
    ys, xs = np.where(region)
    plane_fit = None
    if len(ys) > 10:
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        coeffs, *_ = np.linalg.lstsq(A, diff, rcond=None)
        plane_fit = {"dz_dx_per_px": float(coeffs[0]), "dz_dy_per_px": float(coeffs[1]), "offset_m": float(coeffs[2])}
    return {"n_pixels": int(region.sum()), "median_m": float(np.median(diff)), "p25_m": float(np.percentile(diff, 25)),
            "p75_m": float(np.percentile(diff, 75)), "p5_m": float(np.percentile(diff, 5)), "p95_m": float(np.percentile(diff, 95)),
            "row_band_median_m": bands, "col_band_median_m": col_bands, "bias_plane_fit": plane_fit}


def masks(panel, camera, model_scaled, oracle_z, oracle_valid, Tb, Tc, margin):
    out = {}
    oracle_for_panel_layer = np.where(oracle_valid, oracle_z, 0.0)
    for pose_name, T in (("B", Tb), ("C", Tc)):
        if T is None:
            continue
        T = np.array(T)
        for depth_name, scene in (("model", model_scaled), ("oracle", oracle_for_panel_layer)):
            _, visible, z = panel_layer(T, panel, camera, scene, margin_m=margin)
            if np.any(z > 0):
                out[f"{depth_name}_vs_{pose_name}"] = float(np.mean(~visible[z > 0]))
    return out


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    selection, tracking = select_frames()
    model = MetricDepth(REPO, args.checkpoint, input_size=252)
    cfg = load_config("config-logitech.json")
    panel = panel_points(cfg.panel.width_m, cfg.panel.height_m, cfg.panel.offset_y_m, cfg.panel.tilt_x_deg)

    report = {"checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "map_sha256": hashlib.sha256(Path(MAP_PATH).read_bytes()).hexdigest(),
              "calibration_sha256": hashlib.sha256(Path("logitech-camera-calibration.npz").read_bytes()).hexdigest(),
              "frames": []}

    for label, idx_notes in selection.items():
        target_indices = set(idx_notes)
        withheld, camera = withhold_replay(CLIPS[label], target_indices)
        rows = tracking[label]
        for idx, note in sorted(idx_notes.items()):
            row = rows.get(idx)
            w = withheld.get(idx)
            if row is None or w is None:
                report["frames"].append({"clip": label, "frame_index": idx, "note": note, "error": "frame not reached by replay"})
                continue
            image_path = MASK_EXPORT/label/f"{idx:06d}-image.png"
            image = cv2.imread(str(image_path)) if image_path.exists() else w["bgr"]
            prod_json_path = MASK_EXPORT/label/f"{idx:06d}.json"
            prod = json.loads(prod_json_path.read_text()) if prod_json_path.exists() else {}

            raw, inference = model.predict(image)
            raw = np.asarray(raw, np.float64)
            a, b = prod.get("depth_fit_a"), prod.get("depth_fit_b")
            model_scaled = (raw-a)/b if a is not None and b is not None else None

            Tb = row.get("render_T_CW")
            Tc = w.get("C_T_CW")
            c_source_frame = idx
            if Tc is None:
                # Tag not detected this frame (e.g. covered by the hand, which is the point of
                # the test) -- borrow the nearest other selected frame in this clip with a valid
                # tag fit, purely as a reference for "where the panel/table should be". Never
                # used as this frame's own pose, only for oracle/C context; labelled below.
                candidates = [(abs(j-idx), j) for j, wj in withheld.items() if wj.get("C_valid") and j != idx]
                if candidates:
                    _, c_source_frame = min(candidates)
                    Tc = withheld[c_source_frame]["C_T_CW"]
            b_stats = panel_stats(Tb, panel, camera)
            c_stats = panel_stats(Tc, panel, camera)
            agreement = pose_agreement(b_stats, c_stats) if c_source_frame == idx else None

            oracle_z, oracle_valid = plane_depth(np.array(Tc), np.zeros(3), np.array([0., 0, 1]), camera) if Tc is not None else (None, None)
            panel_mask = (b_stats["z"] > 0) if b_stats else np.zeros(raw.shape, bool)
            tag_mask = np.zeros(raw.shape, bool)
            if Tc is not None:
                _, tag_visible_mask, _ = panel_layer(np.array(Tc), tag_points(cfg.tags.size_m), camera)
                tag_mask = tag_visible_mask

            bias = oracle_vs_model(model_scaled, oracle_z, oracle_valid, panel_mask, tag_mask) if model_scaled is not None and oracle_z is not None else None

            mask_results = {}
            if model_scaled is not None and oracle_z is not None:
                for m in MARGINS:
                    mask_results[str(m)] = masks(panel, camera, model_scaled, oracle_z, oracle_valid, Tb, Tc, m)

            entry = {
                "clip": label, "frame_index": idx, "note": note,
                "pose_source": row.get("actual_pose_source"), "tag_quality": tag_quality(rows, idx),
                "production_fit": {"a": a, "b": b, "anchor_count": prod.get("scale_anchor_count"), "relative_mad": prod.get("fit_relative_mad")},
                "production_occluded_fraction": occluded_fraction(label, idx),
                "B_landmark_inliers": w["B_landmark_landmark_inliers"], "B_landmark_rms_px": w["B_landmark_rms_px"],
                "ransac_inliers": w["ransac_inliers"], "mapped_observations": w["mapped_observations"],
                "C_valid": w["C_valid"], "C_rms_px": w["C_rms_px"],
                "c_source_frame": c_source_frame if c_source_frame != idx else None,  # set only when C was borrowed from a nearby frame
                "panel_B_z_median_m": b_stats["z_median_m"] if b_stats else None,
                "panel_C_z_median_m": c_stats["z_median_m"] if c_stats else None,
                "pose_agreement_B_vs_C": agreement,
                "model_vs_oracle_bias": bias,
                "masks": mask_results,
            }
            if label == "V2" and idx == 441:
                # task 6, hand frame: no hand-drawn mask available (Sahaj's to draw); using a
                # simple rectangular polygon instead, as the step explicitly allows. Drawn by eye
                # from the rectified image at this frame: the hand occupies roughly this region.
                hand_polygon = np.array([[290, 230], [560, 230], [560, 480], [290, 480]])
                hand_mask = np.zeros(raw.shape, np.uint8)
                cv2.fillConvexPoly(hand_mask, hand_polygon, 1)
                hand_mask = hand_mask.astype(bool)
                hand_model = model_scaled[hand_mask] if model_scaled is not None else raw[hand_mask]
                entry["hand_frame_task6"] = {
                    "hand_polygon_px": hand_polygon.tolist(),
                    "model_depth_under_hand_median_m": float(np.median(hand_model)),
                    "model_depth_under_hand_p5_m": float(np.percentile(hand_model, 5)),
                    "model_depth_under_hand_p95_m": float(np.percentile(hand_model, 95)),
                    "panel_C_z_median_m_from_nearby_reference": c_stats["z_median_m"] if c_stats else None,
                    "oracle_table_depth_under_hand_median_m": float(np.median(oracle_z[hand_mask & oracle_valid])) if oracle_z is not None and np.any(hand_mask & oracle_valid) else None,
                    "hand_reads_nearer_than_panel": bool(np.median(hand_model) < c_stats["z_median_m"]) if c_stats and c_stats["z_median_m"] else None,
                    "raw_model_output_under_hand_median": float(np.median(raw[hand_mask])),
                }
            report["frames"].append(entry)
            frame_dir = OUT_ROOT/f"{label}_{idx:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_dir/"image.png"), image)
            np.save(frame_dir/"raw.npy", raw.astype(np.float32))
            if model_scaled is not None:
                np.save(frame_dir/"model_scaled.npy", model_scaled.astype(np.float32))
            if oracle_z is not None:
                np.save(frame_dir/"oracle_z.npy", oracle_z.astype(np.float32))
                np.save(frame_dir/"oracle_valid.npy", oracle_valid)
            if b_stats:
                np.save(frame_dir/"panel_z_B.npy", b_stats["z"].astype(np.float32))
            if c_stats:
                np.save(frame_dir/"panel_z_C.npy", c_stats["z"].astype(np.float32))
            print(f"{label} {idx}: pose_source={row.get('actual_pose_source')} prod_occluded={entry['production_occluded_fraction']}")

    Path(OUT_ROOT/"step36-depth-sources.json").write_text(json.dumps(report, indent=2))
    Path("runs/step36-depth-sources.json").write_text(json.dumps(report, indent=2))
    print(f"done: {len(report['frames'])} frames")


if __name__ == "__main__":
    main()
