"""Step 50: array of upright panels anchored on the tag plane (world Z = 0, Z up). Pure geometry and state, no drawing.

Layout is viewer-centric (x right, y away from the viewer, z up, origin = pivot = bottom centre of the centre panel on the
floor). It maps to the world by yaw `facing_yaw_deg` about Z, then the array state: yaw about Z through the pivot and a
uniform scale about the pivot. Positive state yaw turns the array so the viewer's left panel comes nearer the viewer."""
from dataclasses import dataclass
import numpy as np


def _rz(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@dataclass
class ArrayState:
    yaw_deg: float = 0.0
    scale: float = 1.0
    game: int = 0
    playing: bool = True

    def apply(self, cfg, action, value=0):
        """Apply one action from the gesture or keyboard table; returns True when the state changed."""
        before = (self.yaw_deg, self.scale, self.game, self.playing)
        if action == "yaw":
            self.yaw_deg = float(np.clip(self.yaw_deg+value*cfg.yaw_step_deg, -cfg.yaw_limit_deg, cfg.yaw_limit_deg))
        elif action == "scale":
            self.scale = float(np.clip(self.scale*cfg.scale_step**value, cfg.scale_min, cfg.scale_max))
        elif action == "game":
            self.game = (self.game+int(value)) % len(cfg.games)
        elif action == "play":
            self.playing = True
        elif action == "stop":
            self.playing = False
        elif action == "reset":
            self.yaw_deg, self.scale = 0.0, 1.0
        else:
            raise ValueError(f"unknown array action {action}")
        return before != (self.yaw_deg, self.scale, self.game, self.playing)


def local_corners(cfg):
    """name -> (4, 3) corners [bottom-left, bottom-right, top-right, top-left] as the viewer sees them, scale 1, yaw 0."""
    panels = {p["name"]: p for p in cfg.panels}
    half = next(p for p in cfg.panels if p["side"] == "center")["width_m"]/2
    def base(p):
        return 0.0 if not p.get("above") else base(panels[p["above"]])+panels[p["above"]]["height_m"]+cfg.gap_m
    out = {}
    for p in cfg.panels:
        w, h, z0 = p["width_m"], p["height_m"], base(p)
        t = np.deg2rad(p.get("toe_in_deg", 0.0))
        if p["side"] == "center":
            bl, br = (-w/2, 0.0), (w/2, 0.0)
        elif p["side"] == "left":                       # hinged at the inner edge; outer end moves towards the viewer (-y)
            br = (-(half+cfg.gap_m), 0.0)
            bl = (br[0]-w*np.cos(t), -w*np.sin(t))
        else:
            bl = (half+cfg.gap_m, 0.0)
            br = (bl[0]+w*np.cos(t), -w*np.sin(t))
        out[p["name"]] = np.array([[*bl, z0], [*br, z0], [*br, z0+h], [*bl, z0+h]], np.float64)
    return out


def world_corners(cfg, state, offset_y_m):
    """name -> (4, 3) world corners for the current state. The pivot (0, offset_y_m, 0) never leaves the floor."""
    R = _rz(cfg.facing_yaw_deg+state.yaw_deg)
    pivot = np.array([0.0, offset_y_m, 0.0])
    return {n: state.scale*(c @ R.T)+pivot for n, c in local_corners(cfg).items()}


def clip_near(T, corners, znear=0.05):
    """Sutherland-Hodgman against the plane camera Z = znear. Returns the clipped polygon in camera coordinates (n, 3) or None."""
    pts = np.asarray(corners, np.float64) @ T[:3, :3].T+T[:3, 3]
    out = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        ina, inb = a[2] >= znear, b[2] >= znear
        if ina:
            out.append(a)
        if ina != inb:
            out.append(a+(b-a)*((znear-a[2])/(b[2]-a[2])))
    return np.array(out) if len(out) >= 3 else None


class PanelSet:
    """What panel_layer draws instead of one quad: world corners and a BGR texture per panel. `ids` (set by the renderer) is the
    per-pixel index into `names` of the nearest panel (-1 = none)."""
    def __init__(self, corners, textures, centre, near=0.05):
        self.corners, self.textures, self.centre, self.near = corners, textures, centre, near
        self.names = list(corners)
        self.ids = None

    @property
    def centre_corners(self):
        return self.corners[self.centre]
