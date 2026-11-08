"""Step 52: build the demo video from recorded runs (offline, no camera, no depth model).

Subcommands (run from Integration/ar with ../.venv/bin/python):
  probe RUN [--window S]          what a run contains: streams, duration, pose validity per second, array actions and gesture
                                  events (run-relative seconds), the best windows of S seconds, and the darkest frames (for
                                  syncing an external phone video by covering the lens).
  segment PLAN NAME [NAME ...]    render the named segments of a plan to <output_dir>/<name>.mp4
  stitch PLAN                     join the rendered segments with transitions into <output_dir>/<final>
  all PLAN                        segment (every segment) + stitch

Time: every pane has a `start` in run-relative seconds (0 = the run's first frame, from frames.jsonl source_time_s). A
--replay-run of a recording keeps the original source times, so a live run and its replay share one timeline. Frames are
resampled to the output rate by sample-and-hold on the logged times (the live loop runs at 10-30 fps with uneven gaps; the
.avi files' nominal fps is not used).

Pane kinds (see recording/plan.example.json):
  run      {"run", "stream": input|rgb|annotated|depth}      a video written by the AR run, aligned by frames.jsonl
  gesture  {"run"}                                           RUN/gesture.avi (laptop camera + classifier overlay), aligned by
                                                             RUN/gesture_frames.jsonl (same monotonic clock as the AR frames)
  panel    {"run", "source": video|topdown|full}             the source video the array was showing, at the frame index logged
                                                             in diagnostics.array.video_frames (full = FullOriginal side-by-side)
  posemap  {"run", "exaggerate": "auto"|number, "trail_s"}   top-down map: landmarks, tag, array, camera trajectory coloured
                                                             by pose source, camera frustum and the 2-sigma position ellipse
  file     {"path"}                                          any video (phone footage); start = seconds into that file
  text     {"text"}                                          a caption card
Common keys: "start", "label", "fit": contain|cover.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FONT = cv2.FONT_HERSHEY_SIMPLEX
SOURCE_COLOURS = {"tag": (0, 165, 255), "landmark": (255, 220, 0), "joint": (80, 220, 80)}


# ----------------------------------------------------------------------------- run data
def read_rows(run):
    rows = [json.loads(line) for line in (Path(run)/"frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{run}: empty frames.jsonl")
    return rows


class Run:
    _cache = {}

    def __new__(cls, path):
        path = Path(path).resolve()
        if path not in cls._cache:
            self = super().__new__(cls)
            self.path = path
            self.rows = read_rows(path)
            self.t0 = float(self.rows[0]["source_time_s"])
            self.t = np.array([float(r["source_time_s"]) for r in self.rows])-self.t0
            cls._cache[path] = self
        return cls._cache[path]

    def index_at(self, t_rel):
        """Last AR frame at or before t_rel (sample and hold); 0 before the first frame."""
        return max(0, int(np.searchsorted(self.t, t_rel, side="right"))-1)

    def config(self):
        return json.loads((self.path/"config.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- readers
class SeqReader:
    """Frame-indexed reader for intra-coded AVI/MP4: reads forward, seeks only when going back or far ahead."""
    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.pos, self.frame = -1, None

    def get(self, k):
        if k is None or k < 0:
            return None
        if k == self.pos:
            return self.frame
        if k < self.pos or k > self.pos+30:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            self.pos = k-1
        while self.pos < k:
            ok, frame = self.cap.read()
            if not ok:
                return self.frame
            self.pos, self.frame = self.pos+1, frame
        return self.frame


def fit(image, size, mode="contain"):
    w, h = size
    if image is None:
        return np.zeros((h, w, 3), np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ih, iw = image.shape[:2]
    k = min(w/iw, h/ih) if mode == "contain" else max(w/iw, h/ih)
    scaled = cv2.resize(image, (max(1, round(iw*k)), max(1, round(ih*k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3), np.uint8)
    sh, sw = scaled.shape[:2]
    if mode == "contain":
        x, y = (w-sw)//2, (h-sh)//2
        out[y:y+sh, x:x+sw] = scaled
    else:
        x, y = (sw-w)//2, (sh-h)//2
        out[:] = scaled[y:y+h, x:x+w]
    return out


def label(image, text, scale=.7, top=False):
    """Pane caption, bottom left (the AR HUD uses the top left); top left for the posemap (its sigma strip is at the bottom)."""
    if not text:
        return
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 2)
    x, y = 10, 10 if top else image.shape[0]-th-base-22
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x+tw+16, y+th+base+12), (0, 0, 0), -1)
    cv2.addWeighted(overlay, .6, image, .4, 0, image)
    cv2.putText(image, text, (x+8, y+th+6), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------- panes
class RunPane:
    def __init__(self, spec, base, size):
        self.run = Run(base/spec["run"])
        name = spec.get("stream", "annotated")
        path = self.run.path/f"{name}.avi"
        if not path.exists():
            raise ValueError(f"{path} missing (input needs --record-input; rgb/annotated need --save-video; depth needs --depth-video)")
        self.reader, self.start, self.size, self.mode = SeqReader(path), float(spec.get("start", 0)), size, spec.get("fit", "contain")

    def render(self, tau):
        return fit(self.reader.get(self.run.index_at(self.start+tau)), self.size, self.mode)


class GesturePane:
    def __init__(self, spec, base, size):
        self.run = Run(base/spec["run"])
        stamps = self.run.path/"gesture_frames.jsonl"
        if not stamps.exists():
            raise ValueError(f"{stamps} missing: record the live run with --record-input or --save-video and gestures on")
        self.t = np.array([json.loads(l)["t"] for l in stamps.read_text(encoding="utf-8").splitlines() if l.strip()])-self.run.t0
        self.reader, self.start, self.size, self.mode = SeqReader(self.run.path/"gesture.avi"), float(spec.get("start", 0)), size, spec.get("fit", "contain")

    def render(self, tau):
        k = int(np.searchsorted(self.t, self.start+tau, side="right"))-1
        return fit(self.reader.get(k) if k >= 0 else None, self.size, self.mode)


class PanelSourcePane:
    """The source video the array showed: diagnostics.array.video_frames gives [game, frame index] per AR frame."""
    def __init__(self, spec, base, size):
        self.run = Run(base/spec["run"])
        cfg = self.run.config()["array"]
        manifest = Path(cfg["games_manifest"])
        if not manifest.is_absolute():
            manifest = (self.run.path/manifest).resolve()
        if not manifest.exists():
            manifest = (ROOT/"../DataToProject/games.json").resolve()
        self.games, self.root = json.loads(manifest.read_text(encoding="utf-8")), manifest.parent
        self.source, self.readers = spec.get("source", "video"), {}
        self.start, self.size, self.mode = float(spec.get("start", 0)), size, spec.get("fit", "contain")

    def _path(self, game):
        g = self.games[game]
        if self.source == "full":
            return self.root/"FullOriginal"/f"{g['rally']}_projection_current_shot.mp4"
        return self.root/g[self.source]

    def render(self, tau):
        row = self.run.rows[self.run.index_at(self.start+tau)]
        frames = ((row.get("diagnostics") or {}).get("array") or {}).get("video_frames") or {}
        entry = frames.get("left" if self.source == "topdown" else "game") or frames.get("game")
        if not entry or entry[1] is None:
            return fit(None, self.size)
        game, k = int(entry[0]), int(entry[1])
        if game not in self.readers:
            self.readers[game] = SeqReader(self._path(game))
        return fit(self.readers[game].get(k), self.size, self.mode)


class FilePane:
    def __init__(self, spec, base, size):
        path = Path(spec["path"])
        self.reader = SeqReader(path if path.is_absolute() else base/path)
        self.start, self.size, self.mode = float(spec.get("start", 0)), size, spec.get("fit", "contain")

    def render(self, tau):
        return fit(self.reader.get(int(round((self.start+tau)*self.reader.fps))), self.size, self.mode)


class TextPane:
    def __init__(self, spec, base, size):
        self.image = np.full((size[1], size[0], 3), 20, np.uint8)
        lines = str(spec.get("text", "")).split("\n")
        for i, line in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(line, FONT, 1.0, 2)
            y = size[1]//2+(i-(len(lines)-1)/2)*(th+18)
            cv2.putText(self.image, line, ((size[0]-tw)//2, int(y+th/2)), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    def render(self, tau):
        return self.image.copy()


class PoseMapPane:
    """Top-down (world X-Y; the tag plane is the floor) map of the run: landmarks coloured by height, tag, array footprint,
    camera trajectory coloured by pose source, camera frustum, and the 2-sigma ellipse of the raw pose's camera-position
    covariance (conditional local covariance, see CONTEXT.md section 11), drawn magnified by a stated factor. A strip at the
    bottom plots sigma_x/y/z (mm) over the segment."""
    STRIP = 110

    def __init__(self, spec, base, size, duration):
        from ar_projection.filters import covariance_wc
        self.run, self.size = Run(base/spec["run"]), size
        self.start, self.dur, self.trail = float(spec.get("start", 0)), duration, float(spec.get("trail_s", 30))
        cfg = self.run.config()
        n = len(self.run.rows)
        self.valid = np.zeros(n, bool)
        self.c = np.full((n, 3), np.nan)
        self.fwd = np.full((n, 3), np.nan)
        self.cov = np.full((n, 3, 3), np.nan)
        self.src = ["none"]*n
        self.array_state = [None]*n
        for i, r in enumerate(self.run.rows):
            pose = r.get("raw_pose") or {}
            arr = (r.get("diagnostics") or {}).get("array")
            if arr:
                self.array_state[i] = (arr.get("yaw_deg", 0.), arr.get("scale", 1.))
            if not pose.get("valid") or pose.get("T_CW") is None:
                continue
            T = np.asarray(pose["T_CW"], float)
            R = T[:3, :3]
            self.valid[i], self.src[i] = True, pose.get("source", "none")
            self.c[i] = -R.T@T[:3, 3]
            self.fwd[i] = R.T@np.array([0., 0., 1.])
            if pose.get("covariance") is not None:
                self.cov[i] = covariance_wc(T, np.asarray(pose["covariance"], float))[:3, :3]
        # map, tag, array
        self.points = np.zeros((0, 3))
        raw = Path(cfg["mapping"]["map_path"])
        candidates = [raw, self.run.path/raw, ROOT/raw, ROOT/"data"/raw.name]   # absolute paths differ between machines
        map_path = next((m for m in candidates if m.exists()), None)
        if map_path is None:
            print(f"posemap: map {raw} not found; landmarks not drawn")
        else:
            with np.load(map_path, allow_pickle=False) as data:
                self.points = np.asarray(data["points"], float)
        s = cfg["tags"]["size_m"]/2
        self.tag = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])
        self.array_cfg = self.offset_y = None
        if cfg.get("array", {}).get("enabled"):
            try:
                from ar_projection.config import load_config
                full = load_config(self.run.path/"config.json")
                self.array_cfg, self.offset_y = full.array, full.panel.offset_y_m
            except Exception as exc:                       # the map still renders without the array footprint
                print(f"posemap: array footprint skipped ({type(exc).__name__}: {exc})")
        K = np.asarray(json.loads((self.run.path/"metadata.json").read_text(encoding="utf-8"))["calibration"]["K"], float)
        w = cfg["camera"]["width"]
        self.half_fov = math.atan(w/2/K[0, 0])
        self._layout(spec.get("exaggerate", "auto"))

    def _layout(self, exaggerate):
        lo, hi = self.run.index_at(self.start), self.run.index_at(self.start+self.dur)
        seg = np.arange(lo, hi+1)
        cams = self.c[seg][self.valid[seg]]
        if not len(cams):
            cams = self.c[self.valid]
        if not len(cams):
            raise ValueError("posemap: no valid pose in the run")
        d = np.median(cams[:, :2], axis=0)                 # rotate so the cameras sit at the bottom of the map
        self.theta = round((-math.pi/2-math.atan2(d[1], d[0]))/(math.pi/2))*(math.pi/2)   # snapped to 90 deg: grid stays square
        pts = [self._rot(cams[:, :2]), self._rot(self.tag[:, :2])]
        if len(self.points):
            p = self._rot(self.points[:, :2])
            pts.append(np.percentile(p, [3, 97], axis=0))
        allp = np.vstack(pts)
        mn, mx = allp.min(0)-.4, allp.max(0)+.4
        W, H = self.size[0], self.size[1]-self.STRIP
        self.scale = min(W/(mx[0]-mn[0]), H/(mx[1]-mn[1]))
        self.origin = (mn+mx)/2
        major = []
        for i in seg:
            if self.valid[i] and np.isfinite(self.cov[i]).all():
                major.append(2*math.sqrt(max(np.linalg.eigvalsh(self.cov[i][:2, :2]).max(), 0)))
        if exaggerate == "auto":
            px = (np.median(major) if major else .002)*self.scale
            self.k = next((k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if px*k >= 14), 1000)
        else:
            self.k = float(exaggerate)
        sig = np.sqrt(np.maximum(np.einsum("nii->ni", self.cov[seg]), 0))*1000
        self.sig = sig
        finite = sig[np.isfinite(sig)]
        self.sig_max = float(np.percentile(finite, 95))*1.2 if finite.size else 10.

    def _rot(self, xy):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.asarray(xy)@np.array([[c, s], [-s, c]])

    def _px(self, xy):
        r = self._rot(np.atleast_2d(xy))
        W, H = self.size[0], self.size[1]-self.STRIP
        return np.c_[W/2+(r[:, 0]-self.origin[0])*self.scale, H/2-(r[:, 1]-self.origin[1])*self.scale]

    def render(self, tau):
        W, H = self.size[0], self.size[1]-self.STRIP
        img = np.full((self.size[1], W, 3), 22, np.uint8)
        # grid every 0.5 m
        for gx in np.arange(-10, 10.01, .5):
            a, b = self._px([[gx, -10]]).ravel(), self._px([[gx, 10]]).ravel()
            cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (40, 40, 40), 1)
            a, b = self._px([[-10, gx]]).ravel(), self._px([[10, gx]]).ravel()
            cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), (40, 40, 40), 1)
        img[H:] = 12
        # landmarks coloured by height
        if len(self.points):
            z = self.points[:, 2]
            zn = np.clip((z-np.percentile(z, 2))/max(1e-6, np.percentile(z, 98)-np.percentile(z, 2)), 0, 1)
            colours = cv2.applyColorMap((zn*255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
            for p, col in zip(self._px(self.points[:, :2]), colours):
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, tuple(int(c) for c in col), -1, cv2.LINE_AA)
        # tag
        cv2.polylines(img, [np.int32(self._px(self.tag[:, :2]))], True, (255, 255, 255), 2, cv2.LINE_AA)
        i = self.run.index_at(self.start+tau)
        # array footprint (panels are upright: their top-down footprint is the bottom edge)
        if self.array_cfg is not None and self.array_state[i] is not None:
            from ar_projection.array import ArrayState, world_corners
            yaw, scale = self.array_state[i]
            for name, c in world_corners(self.array_cfg, ArrayState(yaw_deg=yaw, scale=scale), self.offset_y).items():
                q = np.int32(self._px(c[:2, :2]))
                cv2.line(img, tuple(q[0]), tuple(q[1]), (200, 120, 255), 4, cv2.LINE_AA)
        # trajectory
        lo = self.run.index_at(max(0., self.start+tau-self.trail))
        for a in range(lo, i):
            b = a+1
            if self.valid[a] and self.valid[b] and self.run.t[b]-self.run.t[a] < .5:
                p, q = self._px(self.c[[a, b], :2])
                cv2.line(img, tuple(np.int32(p)), tuple(np.int32(q)), SOURCE_COLOURS.get(self.src[b], (160, 160, 160)), 2, cv2.LINE_AA)
        # current camera
        lines = []
        if self.valid[i]:
            c = self.c[i]
            f = self.fwd[i][:2]
            if np.linalg.norm(f) > 1e-6:
                f = f/np.linalg.norm(f)
                L = .35
                rot = lambda v, a: np.array([v[0]*math.cos(a)-v[1]*math.sin(a), v[0]*math.sin(a)+v[1]*math.cos(a)])
                tri = np.array([c[:2], c[:2]+L*rot(f, self.half_fov), c[:2]+L*rot(f, -self.half_fov)])
                cv2.fillPoly(img, [np.int32(self._px(tri))], (90, 90, 90), cv2.LINE_AA)
                cv2.polylines(img, [np.int32(self._px(tri))], True, (255, 255, 255), 1, cv2.LINE_AA)
            cov = self.cov[i]
            centre = self._px(c[:2]).ravel()
            if np.isfinite(cov).all():
                # ellipse axes in the map frame: rotate the XY covariance by theta (screen y points down)
                R2 = np.array([[math.cos(self.theta), math.sin(self.theta)], [-math.sin(self.theta), math.cos(self.theta)]])
                C = R2.T@cov[:2, :2]@R2
                vals, vecs = np.linalg.eigh(C)
                axes = (max(1, int(round(2*math.sqrt(max(vals[1], 0))*self.k*self.scale))), max(1, int(round(2*math.sqrt(max(vals[0], 0))*self.k*self.scale))))
                angle = -math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
                cv2.ellipse(img, tuple(np.int32(centre)), axes, angle, 0, 360, (0, 0, 255), 2, cv2.LINE_AA)
                s = np.sqrt(np.maximum(np.diag(cov), 0))*1000
                lines.append(f"sigma x {s[0]:.1f}  y {s[1]:.1f}  z {s[2]:.1f} mm")
            cv2.circle(img, tuple(np.int32(centre)), 4, SOURCE_COLOURS.get(self.src[i], (200, 200, 200)), -1, cv2.LINE_AA)
            lines.insert(0, f"pose: {self.src[i]}")
        else:
            lines.append("pose: LOST (panel hidden)")
        y = 26                                             # top right: the pane label sits top left
        for t in lines:
            tw = cv2.getTextSize(t, FONT, .55, 1)[0][0]
            cv2.putText(img, t, (W-tw-10, y), FONT, .55, (255, 255, 255) if self.valid[i] else (0, 0, 255), 1, cv2.LINE_AA)
            y += 22
        legend = [("tag", SOURCE_COLOURS["tag"]), ("landmark", SOURCE_COLOURS["landmark"]), ("joint", SOURCE_COLOURS["joint"])]
        x = 10
        for name, col in legend:
            cv2.line(img, (x, H-14), (x+18, H-14), col, 3)
            cv2.putText(img, name, (x+22, H-9), FONT, .45, (220, 220, 220), 1, cv2.LINE_AA)
            x += 30+cv2.getTextSize(name, FONT, .45, 1)[0][0]+10
        k = f"{self.k:g}"
        cv2.putText(img, f"red: 2-sigma position ellipse x{k}", (x+6, H-9), FONT, .45, (0, 0, 255), 1, cv2.LINE_AA)
        # sigma strip
        lo, hi = self.run.index_at(self.start), self.run.index_at(self.start+self.dur)
        top, bottom = H+18, self.size[1]-10
        cv2.putText(img, f"position sigma (mm), 0-{self.sig_max:.0f}", (10, H+14), FONT, .42, (200, 200, 200), 1, cv2.LINE_AA)
        span = max(1e-6, self.dur)
        for axis, col in zip(range(3), ((80, 80, 255), (80, 255, 80), (255, 160, 80))):
            prev = None
            for j in range(lo, hi+1):
                v = self.sig[j-lo, axis] if j-lo < len(self.sig) else np.nan
                if not np.isfinite(v):
                    prev = None
                    continue
                p = (int(10+(self.run.t[j]-self.start)/span*(W-20)), int(bottom-min(v/self.sig_max, 1)*(bottom-top)))
                if prev is not None:
                    cv2.line(img, prev, p, col, 1, cv2.LINE_AA)
                prev = p
        cx = int(10+tau/span*(W-20))
        cv2.line(img, (cx, top), (cx, bottom), (255, 255, 255), 1)
        return img


def make_pane(spec, base, size, duration):
    kind = spec.get("kind", "run")
    if kind == "posemap":
        return PoseMapPane(spec, base, size, duration)
    return {"run": RunPane, "gesture": GesturePane, "panel": PanelSourcePane, "file": FilePane, "text": TextPane}[kind](spec, base, size)


# ----------------------------------------------------------------------------- segments and stitching
def ffmpeg_writer(path, size, fps):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
                             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
                            stdin=subprocess.PIPE)


def render_segment(plan, base, seg, out_dir):
    fps = plan.get("fps", 30)
    cell = tuple(seg.get("cell", plan.get("cell", [640, 480])))
    rows, cols = seg["grid"]
    header = 48 if seg.get("title") else 0
    duration = float(seg["duration"])
    panes = [make_pane(p, base, cell, duration) for p in seg["panes"]]
    if len(panes) != rows*cols:
        raise ValueError(f"segment {seg['name']}: grid {rows}x{cols} needs {rows*cols} panes, got {len(panes)}")
    size = (cols*cell[0], header+rows*cell[1])
    size = (size[0]+size[0] % 2, size[1]+size[1] % 2)
    out = out_dir/f"{seg['name']}.mp4"
    proc = ffmpeg_writer(out, size, fps)
    n = int(round(duration*fps))
    for f in range(n):
        tau = f/fps
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        if header:
            cv2.putText(canvas, seg["title"], (16, 34), FONT, .95, (255, 255, 255), 2, cv2.LINE_AA)
        for k, (pane, spec) in enumerate(zip(panes, seg["panes"])):
            r, c = divmod(k, cols)
            img = pane.render(tau)
            label(img, spec.get("label"), top=spec.get("kind") == "posemap")
            canvas[header+r*cell[1]:header+(r+1)*cell[1], c*cell[0]:(c+1)*cell[0]] = img
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed for {out}")
    print(f"{out}: {n} frames, {size[0]}x{size[1]}, {n/fps:.2f} s")
    return out


def title_card(text, path, size, fps, duration):
    proc = ffmpeg_writer(path, size, fps)
    img = TextPane({"text": text}, None, size).render(0)
    for _ in range(int(round(duration*fps))):
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return path


def stitch(plan, base, out_dir):
    fps = plan.get("fps", 30)
    W, H = plan.get("final_size", [1920, 1080])
    T = float(plan.get("transition_s", .6))
    parts, kinds = [], []
    if plan.get("intro"):
        parts.append(title_card(plan["intro"]["text"], out_dir/"00_intro.mp4", (W, H), fps, plan["intro"].get("duration", 2.5)))
    for seg in plan["segments"]:
        path = out_dir/f"{seg['name']}.mp4"
        if not path.exists():
            raise ValueError(f"{path} missing: run `segment` first")
        parts.append(path)
        kinds.append(seg.get("transition", plan.get("transition", "fade")))
    durations = []
    for p in parts:
        d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
        durations.append(float(d))
    inputs = sum((["-i", str(p)] for p in parts), [])
    norm = [f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[n{i}]"
            for i in range(len(parts))]
    chain, last, offset = [], "n0", 0.
    kinds = ([plan.get("transition", "fade")] if plan.get("intro") else [])+kinds
    for i in range(1, len(parts)):
        offset += durations[i-1]-T
        chain.append(f"[{last}][n{i}]xfade=transition={kinds[i-1] if i-1 < len(kinds) else 'fade'}:duration={T}:offset={offset:.3f}[x{i}]")
        last = f"x{i}"
    graph = ";".join(norm+chain)
    out = out_dir/plan.get("final", "demo.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    print(f"{out}: {sum(durations)-T*(len(parts)-1):.2f} s")


# ----------------------------------------------------------------------------- probe
def probe(run_path, window):
    run = Run(run_path)
    n, dur = len(run.rows), float(run.t[-1])
    print(f"{run.path.name}: {n} frames, {dur:.1f} s, {n/max(dur, 1e-6):.1f} fps mean, median dt {np.median(np.diff(run.t))*1000:.0f} ms")
    print("streams:", ", ".join(p.name for p in sorted(run.path.glob("*.avi"))) or "none")
    cfg = run.config()
    print(f"mapping {cfg['mapping']['mode']}, array {cfg.get('array', {}).get('enabled')}, occlusion block {cfg.get('occlusion', {}).get('enabled')}")
    valid = np.array([bool((r.get("raw_pose") or {}).get("valid")) for r in run.rows])
    src = [(r.get("raw_pose") or {}).get("source", "none") for r in run.rows]
    print(f"valid pose {100*valid.mean():.1f}%; sources " + ", ".join(f"{s} {src.count(s)}" for s in sorted(set(src))))
    print("per second (valid %, main source):")
    line = []
    for s in range(int(math.ceil(dur))):
        idx = np.where((run.t >= s) & (run.t < s+1))[0]
        if not len(idx):
            line.append(f"{s:>4}: -")
            continue
        ss = [src[i] for i in idx]
        line.append(f"{s:>4}: {100*valid[idx].mean():3.0f}% {max(set(ss), key=ss.count)[:4]}")
    for i in range(0, len(line), 6):
        print("   " + " | ".join(line[i:i+6]))
    acts = [(run.t[i], a) for i, r in enumerate(run.rows) for a in (((r.get("diagnostics") or {}).get("array") or {}).get("actions") or ())]
    if acts:
        print("array actions (t s, source, label, action, value, changed):")
        for t, a in acts:
            print(f"   {t:7.2f}  {a['source']:8} {str(a.get('label') or ''):32} {a['action']:6} {a['value']:+g} {a['changed']}")
    if (run.path/"gestures.jsonl").exists():
        ev = [json.loads(l) for l in (run.path/"gestures.jsonl").read_text(encoding="utf-8").splitlines() if '"event"' in l]
        if ev:
            print("gesture events (t s, label): " + "; ".join(f"{e['t']-run.t0:.2f} {e['event']}" for e in ev))
    if window:
        w = float(window)
        best = []
        for s in np.arange(0, max(0., dur-w)+1e-9, .5):
            idx = np.where((run.t >= s) & (run.t < s+w))[0]
            if len(idx):
                best.append((valid[idx].mean(), s))
        best.sort(reverse=True)
        print(f"best {w:g} s windows by valid pose: " + ", ".join(f"start {s:.1f} s ({100*v:.0f}%)" for v, s in best[:5]))
    for name in ("rgb", "input"):
        path = run.path/f"{name}.avi"
        if path.exists():
            cap, means = cv2.VideoCapture(str(path)), []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                means.append(float(cv2.resize(frame, (64, 48)).mean()))
            means = np.array(means[:n])
            dark = np.argsort(means)[:5]
            print(f"darkest frames in {name}.avi (sync marks: cover the lens): " + ", ".join(f"{run.t[i]:.2f} s (mean {means[i]:.0f})" for i in sorted(dark)))
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("run")
    p.add_argument("--window", type=float, default=5)
    s = sub.add_parser("segment")
    s.add_argument("plan")
    s.add_argument("names", nargs="+")
    t = sub.add_parser("stitch")
    t.add_argument("plan")
    a = sub.add_parser("all")
    a.add_argument("plan")
    args = ap.parse_args()
    if args.cmd == "probe":
        probe(args.run, args.window)
        return
    plan_path = Path(args.plan).resolve()
    plan, base = json.loads(plan_path.read_text(encoding="utf-8")), plan_path.parent
    out_dir = (base/plan.get("output_dir", "../../runs/step52-video")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd in ("segment", "all"):
        names = None if args.cmd == "all" else set(args.names)
        for seg in plan["segments"]:
            if names is None or seg["name"] in names:
                render_segment(plan, base, seg, out_dir)
    if args.cmd in ("stitch", "all"):
        stitch(plan, base, out_dir)


if __name__ == "__main__":
    main()
