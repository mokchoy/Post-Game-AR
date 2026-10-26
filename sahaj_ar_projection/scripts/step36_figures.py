"""Step 36 task 7: one PNG per frame from the data scripts/step36_depth_sources.py already saved
(runs/step36-depth-sources/<clip>_<idx>/*.npy, runs/step36-depth-sources.json for the per-frame
numbers). No GPU needed -- run with the system python3 (has matplotlib, unlike .venv/.venv-depth).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

DATA_ROOT = Path("runs/step36-depth-sources")
REPORT = Path("runs/step36-depth-sources.json")


def figure_for(entry):
    label, idx = entry["clip"], entry["frame_index"]
    d = DATA_ROOT/f"{label}_{idx:06d}"
    if not d.exists():
        return None
    image = cv2.cvtColor(cv2.imread(str(d/"image.png")), cv2.COLOR_BGR2RGB)
    raw = np.load(d/"raw.npy")
    model_scaled = np.load(d/"model_scaled.npy") if (d/"model_scaled.npy").exists() else None
    oracle_z = np.load(d/"oracle_z.npy") if (d/"oracle_z.npy").exists() else None
    oracle_valid = np.load(d/"oracle_valid.npy") if (d/"oracle_valid.npy").exists() else None
    panel_z_b = np.load(d/"panel_z_B.npy") if (d/"panel_z_B.npy").exists() else None
    panel_z_c = np.load(d/"panel_z_C.npy") if (d/"panel_z_C.npy").exists() else None

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 2, 2, 2, 2])
    title = (f"{label} frame {idx}  |  {entry.get('note') or ''}  |  pose_source={entry.get('pose_source')}  |  "
             f"tag_rms={entry.get('tag_quality', {}).get('rms_px')}  |  anchors={entry.get('production_fit', {}).get('anchor_count')}  "
             f"a={entry.get('production_fit', {}).get('a')}  b={entry.get('production_fit', {}).get('b')}")
    fig.suptitle(title, fontsize=9)

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(image)
    for arr, color, name in ((panel_z_b, "lime", "B"), (panel_z_c, "magenta", "C")):
        if arr is None or not np.any(arr > 0):
            continue
        contours, _ = cv2.findContours((arr > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c.reshape(-1, 2)
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color=color, linewidth=2, label=f"panel from {name}")
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")
    ax.set_title("rectified photo; panel outline from B (green) and C (magenta) where available", fontsize=8)
    ax.axis("off")

    def heat(ax, data, valid_mask, title, cmap="turbo", vmin=None, vmax=None):
        shown = np.where(valid_mask, data, np.nan) if valid_mask is not None else data
        im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    model_valid = np.isfinite(raw) & (raw > 0) if model_scaled is None else np.isfinite(model_scaled)
    vmin = np.nanpercentile(model_scaled[model_valid], 2) if model_scaled is not None and np.any(model_valid) else None
    vmax = np.nanpercentile(model_scaled[model_valid], 98) if model_scaled is not None and np.any(model_valid) else None
    if model_scaled is not None:
        heat(fig.add_subplot(gs[1, 0]), model_scaled, model_valid, "model depth, scaled (m)", vmin=vmin, vmax=vmax)
    if oracle_z is not None:
        heat(fig.add_subplot(gs[1, 1]), oracle_z, oracle_valid, "oracle table depth D (m)", vmin=vmin, vmax=vmax)
    if model_scaled is not None and oracle_z is not None:
        diff = model_scaled-oracle_z
        both = model_valid & oracle_valid
        dv = np.nanpercentile(np.abs(diff[both]), 95) if np.any(both) else 1
        heat(fig.add_subplot(gs[1, 2]), diff, both, "model - oracle (m)", cmap="coolwarm", vmin=-dv, vmax=dv)

    if panel_z_b is not None:
        pv = panel_z_b > 0
        heat(fig.add_subplot(gs[2, 0]), panel_z_b, pv, "virtual panel Z from B (m)")
    if panel_z_c is not None:
        pv = panel_z_c > 0
        heat(fig.add_subplot(gs[2, 1]), panel_z_c, pv, "virtual panel Z from C (m)")
    if panel_z_b is not None and panel_z_c is not None:
        both = (panel_z_b > 0) & (panel_z_c > 0)
        d_bc = panel_z_b-panel_z_c
        dv = np.nanpercentile(np.abs(d_bc[both]), 95) if np.any(both) else .01
        heat(fig.add_subplot(gs[2, 2]), d_bc, both, "B - C panel Z (m)", cmap="coolwarm", vmin=-dv, vmax=dv)

    margins = entry.get("masks", {})
    ax_masks = fig.add_subplot(gs[3, :2])
    ax_masks.axis("off")
    thumbs = []
    labels = []
    m02 = margins.get("0.02", {})
    for key in ("model_vs_B", "model_vs_C", "oracle_vs_B", "oracle_vs_C"):
        thumbs.append(m02.get(key))
        labels.append(key)
    ax_masks.set_title("occluded fraction @ margin 0.02: " + ", ".join(f"{l}={v:.1%}" if v is not None else f"{l}=n/a" for l, v in zip(labels, thumbs)), fontsize=8)

    ax_scatter = fig.add_subplot(gs[3, 2])
    if model_scaled is not None and oracle_z is not None:
        both = model_valid & oracle_valid
        if np.any(both):
            ms, os_ = model_scaled[both], oracle_z[both]
            sample = np.random.default_rng(0).choice(len(ms), size=min(3000, len(ms)), replace=False)
            ax_scatter.scatter(os_[sample], ms[sample], s=1, alpha=.3)
            lo, hi = np.nanmin(os_[sample]), np.nanmax(os_[sample])
            ax_scatter.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="identity")
            ax_scatter.set_xlabel("oracle depth (m)"); ax_scatter.set_ylabel("model depth (m)")
            ax_scatter.legend(fontsize=7)
    ax_scatter.set_title("model vs oracle, table pixels", fontsize=8)

    ax_profile = fig.add_subplot(gs[4, :])
    if model_scaled is not None:
        h, w = model_scaled.shape
        col = w//2
        y = np.arange(h)
        ax_profile.plot(y, model_scaled[:, col], label="model (scaled)", linewidth=1)
        if oracle_z is not None:
            oz = oracle_z[:, col].copy(); oz[~oracle_valid[:, col]] = np.nan
            ax_profile.plot(y, oz, label="oracle", linewidth=1)
        if panel_z_b is not None:
            pb = panel_z_b[:, col].copy(); pb[panel_z_b[:, col] <= 0] = np.nan
            ax_profile.plot(y, pb, label="panel Z (B)", linewidth=1)
        if panel_z_c is not None:
            pc = panel_z_c[:, col].copy(); pc[panel_z_c[:, col] <= 0] = np.nan
            ax_profile.plot(y, pc, label="panel Z (C)", linewidth=1)
        ax_profile.set_xlabel("image row (vertical line through image centre column)"); ax_profile.set_ylabel("depth (m)")
        ax_profile.legend(fontsize=7)
    ax_profile.set_title("depth profile along a vertical line through the image centre", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, .96])
    out = DATA_ROOT/f"{label}_{idx:06d}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    report = json.loads(REPORT.read_text())
    made = 0
    for entry in report["frames"]:
        if "error" in entry:
            continue
        out = figure_for(entry)
        if out:
            made += 1
            print(out)
    print(f"done: {made} figures")


if __name__ == "__main__":
    main()
