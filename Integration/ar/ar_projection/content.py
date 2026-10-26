"""Step 50 panel content: textures for the array panels, rendered at the panel's aspect ratio (hard-coded content for now)."""
import time
import cv2
import numpy as np


def texture_size(panel, long_px):
    w, h = panel["width_m"], panel["height_m"]
    return (long_px, max(8, round(long_px*h/w))) if w >= h else (max(8, round(long_px*w/h)), long_px)


def fit_frame(frame, size, mode):
    """Put a video frame on a texture of `size` (w, h): contain = letterbox, cover = centre crop, stretch."""
    w, h = size
    if mode == "stretch":
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    fh, fw = frame.shape[:2]
    k = min(w/fw, h/fh) if mode == "contain" else max(w/fw, h/fh)
    scaled = cv2.resize(frame, (max(1, round(fw*k)), max(1, round(fh*k))), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    sh, sw = scaled.shape[:2]
    x, y = (w-sw)//2, (h-sh)//2
    if mode == "contain":
        out[y:y+sh, x:x+sw] = scaled
    else:
        out[:] = scaled[-y:-y+h, -x:-x+w]
    return out


def text_fit(image, text, box, color=(255, 255, 255), thickness=3):
    """Largest Hershey text that fits `box` = (x, y, w, h), centred."""
    x, y, w, h = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, thickness)
    k = min(w/tw, h/th)
    (tw, th), base = cv2.getTextSize(text, font, k, max(1, round(thickness*k)))
    cv2.putText(image, text, (x+(w-tw)//2, y+(h+th)//2), font, k, color, max(1, round(thickness*k)), cv2.LINE_AA)


def checkerboard(size, cell=16):
    w, h = size
    y, x = np.indices((h, w))
    out = np.empty((h, w, 3), np.uint8)
    checks = ((x//cell+y//cell) % 2).astype(bool)
    out[checks], out[~checks] = (220, 180, 40), (245, 245, 245)
    return out


class VideoLoop:
    """Looping video on the wall clock; frozen (and resumed without a jump) while not playing."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.
        self.frame, self.n, self.t0, self.paused_at = None, 0, time.monotonic(), None

    def update(self, playing, now=None):
        now = time.monotonic() if now is None else now
        if not playing:
            self.paused_at = self.paused_at if self.paused_at is not None else now
            return self.frame
        if self.paused_at is not None:
            self.t0 += now-self.paused_at
            self.paused_at = None
        due = int((now-self.t0)*self.fps)
        while self.n <= due or self.frame is None:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.t0, self.n, due = now, 0, 0
                continue
            self.frame, self.n = frame, self.n+1
        return self.frame

    def close(self):
        self.cap.release()


class ArrayContent:
    """Textures per panel name for the current ArrayState. Static textures are rendered once."""
    def __init__(self, cfg, left_video=None):
        self.cfg = cfg
        self.panels = {p["name"]: p for p in cfg.panels}
        self.sizes = {n: texture_size(p, cfg.texture_long_px) for n, p in self.panels.items()}
        path = left_video or cfg.left_video
        self.left = VideoLoop(path) if path else None
        self.game_videos = {i: VideoLoop(g["video"]) for i, g in enumerate(cfg.games) if g.get("video")}
        self.static = {n: self._static(p, self.sizes[n]) for n, p in self.panels.items() if p["role"] in ("score", "stats")}
        self.cache = {}

    def _static(self, p, size):
        img = np.full((size[1], size[0], 3), 25, np.uint8)
        if p["role"] == "score":
            text_fit(img, self.cfg.score, (int(size[0]*.05), int(size[1]*.1), int(size[0]*.9), int(size[1]*.8)), thickness=4)
            return img
        w, h = size
        for i, person in enumerate(self.cfg.players[:2]):
            x0 = i*w//2+6
            cv2.putText(img, person["title"], (x0, int(h*.2)), cv2.FONT_HERSHEY_SIMPLEX, h/200, (0, 255, 255), 2, cv2.LINE_AA)
            for j, line in enumerate(person["lines"]):
                cv2.putText(img, line, (x0, int(h*(.45+.22*j))), cv2.FONT_HERSHEY_SIMPLEX, h/300, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _placeholder(self, key, size, label):
        if key not in self.cache:
            img = checkerboard(size)
            text_fit(img, label, (int(size[0]*.05), int(size[1]*.3), int(size[0]*.9), int(size[1]*.4)), (0, 0, 0), 4)
            self.cache[key] = img
        return self.cache[key]

    def textures(self, state):
        out = dict(self.static)
        for name, p in self.panels.items():
            size = self.sizes[name]
            if p["role"] == "game":
                game = self.cfg.games[state.game % len(self.cfg.games)]
                loop = self.game_videos.get(state.game % len(self.cfg.games))
                frame = loop.update(state.playing) if loop else None
                out[name] = fit_frame(frame, size, self.cfg.video_fit) if frame is not None else self._placeholder(("game", game["name"]), size, game["name"])
            elif p["role"] == "video":
                frame = self.left.update(state.playing) if self.left else None
                out[name] = fit_frame(frame, size, self.cfg.video_fit) if frame is not None else self._placeholder("video", size, "VIDEO")
        return out

    def close(self):
        for loop in ([self.left] if self.left else [])+list(self.game_videos.values()):
            loop.close()


# Keyboard fallback (also for replay tests); none of these clash with the tracker keys q d f p o v r s.
KEYS = {ord(","): ("game", -1), ord("."): ("game", 1), ord("j"): ("yaw", 1), ord("k"): ("yaw", -1),
        ord("="): ("scale", 1), ord("-"): ("scale", -1), ord("t"): ("play", 1), ord("b"): ("reset", 0), ord("x"): ("stop", 0)}


class ArrayRuntime:
    """ArrayState + content + the per-frame PanelSet; applies gesture events, keys and scripted actions."""
    def __init__(self, config, left_video=None):
        from .array import ArrayState
        self.cfg, self.offset_y, self.gestures = config.array, config.panel.offset_y_m, config.gestures
        self.state, self.content = ArrayState(), ArrayContent(config.array, left_video)
        self.centre = next(p["name"] for p in self.cfg.panels if p["side"] == "center")
        self.log = []                         # actions applied since the last logged frame

    def do(self, action, value, source, label=None, t=None):
        changed = self.state.apply(self.cfg, action, value)
        self.log.append({"source": source, "label": label, "t": t, "action": action, "value": value, "changed": changed})
        return changed

    def drain(self, events):
        """Apply every queued gesture event (non-blocking)."""
        import queue
        from .gestures import action_for
        while events is not None:
            try:
                t, label = events.get_nowait()
            except queue.Empty:
                return
            act = action_for(self.gestures.actions, label)
            if act:
                self.do(*act, "gesture", label, t)

    def key(self, key):
        if key in KEYS:
            self.do(*KEYS[key], "key")

    def panelset(self):
        from .array import PanelSet, world_corners
        return PanelSet(world_corners(self.cfg, self.state, self.offset_y), self.content.textures(self.state), self.centre, self.cfg.near_plane_m)

    def take_log(self):
        out, self.log = self.log, []
        return out

    def close(self):
        self.content.close()
