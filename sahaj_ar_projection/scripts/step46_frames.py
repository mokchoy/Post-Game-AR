"""Step 46 task 5: side-by-side images + margin/background numbers for frames where Sahaj's hand / black object covers the panel.
Replays two live runs with the current far config (1.0 m panel, smoothing on) and RelativeLineOcclusion. Run via .venv-depth.
Writes runs/step46-images/<run>-<frame>.png and runs/step46-frames.json."""
import json
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ar_projection.config import load_config
from ar_projection.occlusion import RelativeDepth, RelativeLineOcclusion
from ar_projection.pipeline import run
from ar_projection.rendering import side_by_side
from step31_masks import REPO
from step43_relative import REL_CKPT

TARGETS = {"20261025T081546-aa877612": [10, 17, 20, 25, 26.5, 28, 42], "20261025T081836-22c82ea7": [3, 5, 19]}
OUT = Path("runs/step46-images")


class Grab(RelativeLineOcclusion):
    def __init__(self, *a, frames, tag, **k):
        super().__init__(*a, **k)
        self.keep_view, self.frames, self.tag, self.rows = True, frames, tag, []

    def render(self, packet, result, panel, landmark_map, features):
        layer = super().render(packet, result, panel, landmark_map, features)
        if packet.index in self.frames:
            row = {"run": self.tag, "frame": packet.index, "target_s": self.frames[packet.index], "status": result.diagnostics["occlusion"].get("status")}
            if self.view is not None and layer is not None:
                v = self.view
                z, s = v["z"], v["scene"]
                on = z > 0
                margin = np.maximum(self.floor, self.fraction*z)
                diff = (z-s)[on]
                row.update(alignment_source=result.diagnostics["occlusion"]["alignment_source"], panel_z_median_m=float(np.median(z[on])),
                           hidden_fraction=float(np.mean(~v["visible"][on])),
                           nearer_but_inside_margin_fraction=float(np.mean((diff > 0) & (diff <= margin[on]))),
                           at_or_beyond_panel_fraction=float(np.mean(diff <= 0)),
                           scene_min_in_panel_m=float(np.nanmin(s[on])), scene_p5_in_panel_m=float(np.nanpercentile(s[on], 5)))
                left = packet.bgr.copy()
                color, vis = layer
                left[vis] = color[vis]
                OUT.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(OUT/f"{self.tag[9:15]}-{packet.index:06d}.png"), side_by_side(left, v))
            self.rows.append(row)
        return layer


if __name__ == "__main__":
    model = RelativeDepth(REPO, REL_CKPT, 252)
    out = []
    for name, times in TARGETS.items():
        rd = Path("runs")/name
        rows = [json.loads(l) for l in (rd/"frames.jsonl").read_text().splitlines()]
        t0 = rows[0]["source_time_s"]
        frames = {}
        for t in times:
            ok = [r for r in rows if r["actual_pose_source"] != "none" and abs(r["source_time_s"]-t0-t) <= 1.0]
            if ok:
                frames[min(ok, key=lambda r: abs(r["source_time_s"]-t0-t))["frame_index"]] = t
        cfg = load_config("config-logitech-far.json")
        cfg = replace(cfg, mapping=replace(cfg.mapping, mode="localize", map_path=str(Path("data/logitech-landmark-map-occ.npz").resolve())),
                      camera=replace(cfg.camera, source_type="video", video_path=str(rd.resolve()/"input.avi"), timestamps_path=str(rd.resolve()/"frames.jsonl")),
                      display=replace(cfg.display, enabled=False), logging=replace(cfg.logging, enabled=True, save_input_video=False, save_annotated_video=False))
        occ = Grab(model, None, .10, .10, True, 30, REL_CKPT, "bypassed", frames=frames, tag=name)
        run(cfg, occlusion=occ)
        out += occ.rows
    Path("runs/step46-frames.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "run"})
