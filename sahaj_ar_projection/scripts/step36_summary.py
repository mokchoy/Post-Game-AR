"""Step 36 task 8: grouped summary tables and decision-table assessment from
runs/step36-depth-sources.json. Pure numpy/json, no GPU, no ar_projection import needed.
Writes runs/step36-depth-sources.md (and the grouped numbers are appended into the json)."""
import json
from pathlib import Path
import numpy as np

REPORT = Path("runs/step36-depth-sources.json")


def g(f, *path):
    v = f
    for p in path:
        if v is None:
            return None
        v = v.get(p) if isinstance(v, dict) else None
    return v


def pose_agree_stats(group):
    agrees = [g(f, "pose_agreement_B_vs_C") for f in group]
    agrees = [a for a in agrees if a]
    if not agrees:
        return None
    zd = [a["panel_z_diff_median_m"] for a in agrees if a["panel_z_diff_median_m"] is not None]
    return {"n": len(agrees), "camera_pos_diff_median_m": float(np.median([a["camera_position_diff_m"] for a in agrees])),
            "rotation_diff_median_deg": float(np.median([a["rotation_diff_deg"] for a in agrees])),
            "corner_px_median": float(np.median([a["corner_px_median"] for a in agrees])),
            "panel_z_diff_median_m": float(np.median(zd)) if zd else None, "panel_z_diff_max_m": float(np.max(zd)) if zd else None,
            "frac_frames_above_2cm": float(np.mean([(z or 0) > 0.02 for z in zd])) if zd else None}


def bias_stats(group):
    biases = [g(f, "model_vs_oracle_bias") for f in group]
    biases = [b for b in biases if b]
    if not biases:
        return None
    top = [b["row_band_median_m"]["top"] for b in biases if b["row_band_median_m"]["top"] is not None]
    bottom = [b["row_band_median_m"]["bottom"] for b in biases if b["row_band_median_m"]["bottom"] is not None]
    return {"n": len(biases), "median_bias_m": float(np.median([b["median_m"] for b in biases])),
            "p25_of_medians_m": float(np.percentile([b["median_m"] for b in biases], 25)),
            "p75_of_medians_m": float(np.percentile([b["median_m"] for b in biases], 75)),
            "row_top_median_m": float(np.median(top)) if top else None, "row_bottom_median_m": float(np.median(bottom)) if bottom else None,
            "top_minus_bottom_m": float(np.median(top)-np.median(bottom)) if top and bottom else None}


def mask_stats(group, margin="0.02"):
    out = {}
    for key in ("model_vs_B", "model_vs_C", "oracle_vs_B", "oracle_vs_C"):
        vals = [g(f, "masks", margin, key) for f in group]
        vals = [v for v in vals if v is not None]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "n": len(vals)}
    return out


def decide(pose_agree, bias, masks02):
    rows = []
    if pose_agree and pose_agree["frac_frames_above_2cm"] and pose_agree["frac_frames_above_2cm"] > .5:
        rows.append("pose: B and C disagree on most frames (>2cm panel Z, or >3px corners) -- pose selection is the problem here")
    if masks02:
        ob, oc = masks02.get("oracle_vs_B", {}).get("mean"), masks02.get("oracle_vs_C", {}).get("mean")
        mb, mc = masks02.get("model_vs_B", {}).get("mean"), masks02.get("model_vs_C", {}).get("mean")
        if (ob or 0) < .01 and (oc or 0) < .01 and ((mb or 0) > .05 or (mc or 0) > .05):
            rows.append("model: oracle gives ~no false hides with either pose, but model depth does -- model/fit problem confirmed, not pose")
        if ob is not None and oc is not None and abs(ob-oc) > .05:
            rows.append(f"pose-sensitive oracle result: oracle_vs_B={ob:.1%} vs oracle_vs_C={oc:.1%} -- the two poses disagree enough to matter even for the oracle")
    if bias and bias.get("top_minus_bottom_m") is not None and abs(bias["top_minus_bottom_m"]) > .05:
        rows.append(f"structured row-dependent bias: top-bottom median difference {bias['top_minus_bottom_m']:.3f} m -- plane-like/row bias, not a flat offset a simple rescale could remove")
    return rows or ["no decision-table row clearly applies from the aggregate numbers; see per-frame detail"]


def main():
    report = json.loads(REPORT.read_text())
    frames = report["frames"]
    groups = {
        "false_hide_V3_V4": [f for f in frames if f["clip"] in ("V3", "V4") and (f.get("production_occluded_fraction") or 0) > .1],
        "V0_control_clip": [f for f in frames if f["clip"] == "V0"],
        "zero_occlusion_controls_all_clips": [f for f in frames if (f.get("note") or "") == "zero-occlusion control"],
        "V2_hand_window": [f for f in frames if f["clip"] == "V2" and 431 <= f["frame_index"] <= 451],
    }
    summary = {}
    for name, group in groups.items():
        pa, bs, m02, m05 = pose_agree_stats(group), bias_stats(group), mask_stats(group, "0.02"), mask_stats(group, "0.05")
        summary[name] = {"n_frames": len(group), "pose_agreement_B_vs_C": pa, "model_vs_oracle_bias": bs,
                          "masks_margin_0.02": m02, "masks_margin_0.05": m05, "decision_rows": decide(pa, bs, m02)}

    hand = next((f for f in frames if f["clip"] == "V2" and f["frame_index"] == 441), None)
    summary["hand_frame_441"] = hand.get("hand_frame_task6") if hand else None

    report["grouped_summary"] = summary
    REPORT.write_text(json.dumps(report, indent=2))

    lines = ["# Step 36: depth-source comparison -- grouped summary\n",
             "Full per-frame detail in `step36-depth-sources.json`. 79 frames across V0-V4.\n"]
    for name, s in summary.items():
        if name == "hand_frame_441":
            continue
        lines.append(f"## {name} (n={s['n_frames']})\n")
        if s["pose_agreement_B_vs_C"]:
            pa = s["pose_agreement_B_vs_C"]
            lines.append(f"- B vs C pose agreement (n={pa['n']}): camera position diff median {pa['camera_pos_diff_median_m']*100:.2f} cm, "
                         f"rotation diff median {pa['rotation_diff_median_deg']:.2f} deg, panel Z diff median {pa['panel_z_diff_median_m']*100:.2f} cm "
                         f"(max {pa['panel_z_diff_max_m']*100:.2f} cm), {pa['frac_frames_above_2cm']:.0%} of frames exceed 2 cm\n")
        if s["model_vs_oracle_bias"]:
            b = s["model_vs_oracle_bias"]
            lines.append(f"- model vs oracle bias (n={b['n']}): median {b['median_bias_m']*100:.1f} cm (IQR {b['p25_of_medians_m']*100:.1f} to {b['p75_of_medians_m']*100:.1f} cm), "
                         f"row-band top {b['row_top_median_m']*100:.1f} cm vs bottom {b['row_bottom_median_m']*100:.1f} cm (top-bottom {b['top_minus_bottom_m']*100:.1f} cm)\n")
        if s["masks_margin_0.02"]:
            m = s["masks_margin_0.02"]
            lines.append("- occluded fraction @ margin 0.02 m: " + ", ".join(f"{k}={v['mean']:.1%}" for k, v in m.items()) + "\n")
        lines.append("- decision: " + "; ".join(s["decision_rows"]) + "\n")
    if summary.get("hand_frame_441"):
        h = summary["hand_frame_441"]
        lines.append(f"## hand frame V2/441\n- model depth under hand: median {h['model_depth_under_hand_median_m']:.3f} m "
                     f"(p5-p95 {h['model_depth_under_hand_p5_m']:.3f}-{h['model_depth_under_hand_p95_m']:.3f} m)\n"
                     f"- panel/oracle depth there (nearby reference): panel {h['panel_C_z_median_m_from_nearby_reference']:.3f} m, oracle table {h['oracle_table_depth_under_hand_median_m']:.3f} m\n"
                     f"- hand reads nearer than panel: **{h['hand_reads_nearer_than_panel']}** -- "
                     + ("model limit at close range (decision table)\n" if h["hand_reads_nearer_than_panel"] is False else "fit/scaling hides a real near reading\n"))
    Path("runs/step36-depth-sources.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
