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
    """Looping video on a caller-supplied clock (the frame's source time, Step 52; wall clock if none is given); frozen (and
    resumed without a jump) while not playing. Driving it from the source time makes a --replay-run show the same panel
    frames as the live run, and `index` (the frame on screen) is logged per AR frame for the recording compositor."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.
        self.frame, self.n, self.t0, self.paused_at = None, 0, None, None

    @property
    def index(self):
        return self.n-1 if self.frame is not None else None

    def update(self, playing, now=None):
        now = time.monotonic() if now is None else now
        if self.t0 is None:
            self.t0 = now
        if not playing:
            self.paused_at = self.paused_at if self.paused_at is not None else now
            if self.frame is None:                       # stopped right after a swipe: still show the first frame
                ok, self.frame = self.cap.read()
                self.n = 1 if ok else 0
                self.frame = self.frame if ok else None
            return self.frame
        if self.paused_at is not None:
            self.t0 += now-self.paused_at
            self.paused_at = None
        due = int((now-self.t0)*self.fps+1e-6)          # +1e-6: 0.4 s at 25 fps is frame 10, not 9.999...
        while self.n <= due or self.frame is None:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.t0, self.n, due = now, 0, 0
                continue
            self.frame, self.n = frame, self.n+1
        return self.frame

    def restart(self, now=None):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame, self.n, self.t0, self.paused_at = None, 0, time.monotonic() if now is None else now, None

    def close(self):
        self.cap.release()


def score_texture(size, players):
    """Two columns, each a flag above that player's score (score going into the rally), a colon between."""
    w, h = size
    img = np.full((h, w, 3), 25, np.uint8)
    fw = int(w*.34)
    fh = int(fw*2/3)
    for i, p in enumerate(players[:2]):
        x0 = i*w//2
        flag = cv2.imread(p["flag"]) if p.get("flag") else None
        if flag is not None:
            x = x0+(w//2-fw)//2
            img[8:8+fh, x:x+fw] = cv2.resize(flag, (fw, fh), interpolation=cv2.INTER_AREA)
        text_fit(img, str(p["score_before"]), (x0+6, fh+20, w//2-12, int((h-fh-28)*.8)), thickness=4)
    text_fit(img, ":", (int(w*.46), fh+20, int(w*.08), int((h-fh-28)*.8)), thickness=4)
    return img


def stats_texture(size, players):
    """Two columns: the player's name, then Height / Weight / Rank."""
    w, h = size
    img = np.full((h, w, 3), 25, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    kname = min([h/220]+[(w//2-14)/cv2.getTextSize(p["name"], font, 1.0, 2)[0][0] for p in players[:2]])   # one size for both names
    for i, p in enumerate(players[:2]):
        x0 = i*w//2+8
        cv2.putText(img, p["name"], (x0, int(h*.16)), font, kname, (0, 255, 255), 2, cv2.LINE_AA)
        lines = [f"Height: {p['height']}", f"Weight: {p['weight']}", f"Rank: {p['rank']}"]
        k = min(h/300, (w//2-12)/max(cv2.getTextSize(t, font, 1.0, 1)[0][0] for t in lines))
        for j, t in enumerate(lines):
            cv2.putText(img, t, (x0, int(h*(.45+.22*j))), font, k, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class ArrayContent:
    """Textures per panel name for the current ArrayState. Each game entry may carry its own game video, top-down video and
    players (flags, scores, stats); switching game restarts both videos together so they stay in sync."""
    def __init__(self, cfg, left_video=None):
        self.cfg = cfg
        self.panels = {p["name"]: p for p in cfg.panels}
        self.sizes = {n: texture_size(p, cfg.texture_long_px) for n, p in self.panels.items()}
        path = left_video or cfg.left_video
        self.override_left = VideoLoop(path) if path else None
        self.loops = {}                       # (game index, "video" | "topdown") -> VideoLoop, opened on first use
        self.cache = {}
        self.current = None
        self.frames = {}                      # role -> [game, frame index on screen] for the current AR frame (Step 52 log)

    def _loop(self, game, key):
        path = self.cfg.games[game].get(key)
        if not path:
            return None
        if (game, key) not in self.loops:
            self.loops[(game, key)] = VideoLoop(path)
        return self.loops[(game, key)]

    def _static(self, name, game):
        if (name, game) not in self.cache:
            role, players = self.panels[name]["role"], self.cfg.games[game].get("players")
            size = self.sizes[name]
            if players:
                self.cache[(name, game)] = score_texture(size, players) if role == "score" else stats_texture(size, players)
            else:                                             # no per-game data: the config-wide placeholder
                img = np.full((size[1], size[0], 3), 25, np.uint8)
                if role == "score":
                    text_fit(img, self.cfg.score, (int(size[0]*.05), int(size[1]*.1), int(size[0]*.9), int(size[1]*.8)), thickness=4)
                else:
                    stats = [{"name": p["title"], "height": p["lines"][0][8:], "weight": p["lines"][1][8:], "rank": p["lines"][2][6:]} for p in self.cfg.players]
                    img = stats_texture(size, stats)
                self.cache[(name, game)] = img
        return self.cache[(name, game)]

    def _placeholder(self, key, size, label):
        if key not in self.cache:
            img = checkerboard(size)
            text_fit(img, label, (int(size[0]*.05), int(size[1]*.3), int(size[0]*.9), int(size[1]*.4)), (0, 0, 0), 4)
            self.cache[key] = img
        return self.cache[key]

    def textures(self, state, now=None):
        now = time.monotonic() if now is None else now
        game = state.game % len(self.cfg.games)
        self.frames = {}
        if game != self.current:                              # swipe: restart this game's videos together
            for key in ("video", "topdown"):
                loop = self._loop(game, key)
                if loop:
                    loop.restart(now)
            if self.override_left:
                self.override_left.restart(now)
            self.current = game
        out = {}
        for name, p in self.panels.items():
            size = self.sizes[name]
            if p["role"] in ("score", "stats"):
                out[name] = self._static(name, game)
            elif p["role"] == "game":
                loop = self._loop(game, "video")
                frame = loop.update(state.playing, now) if loop else None
                self.frames["game"] = [game, loop.index if loop else None]
                label = self.cfg.games[game]["name"]
                out[name] = fit_frame(frame, size, self.cfg.video_fit) if frame is not None else self._placeholder(("game", label), size, label)
            else:
                loop = self.override_left or self._loop(game, "topdown")
                frame = loop.update(state.playing, now) if loop else None
                self.frames["left"] = [game, loop.index if loop else None, "override" if self.override_left else "topdown"]
                out[name] = fit_frame(frame, size, self.cfg.video_fit) if frame is not None else self._placeholder("video", size, "VIDEO")
        return out

    def close(self):
        for loop in ([self.override_left] if self.override_left else [])+list(self.loops.values()):
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

    def panelset(self, now=None):
        """now: the frame's source time (Step 52), so panel videos follow the recording's clock in --replay-run."""
        from .array import PanelSet, world_corners
        return PanelSet(world_corners(self.cfg, self.state, self.offset_y), self.content.textures(self.state, now), self.centre, self.cfg.near_plane_m)

    def take_log(self):
        out, self.log = self.log, []
        return out

    def close(self):
        self.content.close()
