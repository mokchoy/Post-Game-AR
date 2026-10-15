"""Fixed world panel, pose EMA in T_WC, and display-only rounded pixels."""
import cv2
import numpy as np
from .geometry import depths, inverse, panel_points, project, projection_jacobian


class PoseSmoother:
    def __init__(self, config):
        self.config = config
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, T):
        if not self.config.enabled:
            return T.copy()
        current = inverse(T)
        if self.previous is not None:
            weight = self.config.measurement_weight
            old = self.previous
            delta = cv2.Rodrigues(current[:3, :3] @ old[:3, :3].T)[0]
            current[:3, :3] = cv2.Rodrigues(weight*delta)[0] @ old[:3, :3]
            current[:3, 3] = (1-weight)*old[:3, 3] + weight*current[:3, 3]
        self.previous = current.copy()
        return inverse(current)


def panel_geometry(cfg):
    return panel_points(cfg.width_m, cfg.height_m, cfg.offset_y_m, cfg.tilt_x_deg)


def visible_panel(T, panel, camera):
    if T is None or np.any(depths(T, panel) <= 0):
        return None
    pixels = project(T, panel, camera)
    if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels >= camera.size):
        return None  # first pass hides the whole polygon if any corner is outside full extent
    return pixels


def panel_covariance(T, panel, camera, cov, uncertainty):
    if cov is None or not uncertainty.enabled or not uncertainty.project_to_panel_corners or visible_panel(T, panel, camera) is None:
        return None
    G = projection_jacobian(T, panel, camera, uncertainty).reshape(-1, 2, 6)
    return np.array([g@cov@g.T for g in G])


def annotate(packet, result, tags, features, display, approximate=False):
    image = packet.bgr.copy()
    if result.render_panel is not None:
        cv2.fillConvexPoly(image, np.rint(result.render_panel).astype(np.int32), (0, 200, 0))
    if display.debug:
        for tag in tags:
            cv2.polylines(image, [np.rint(tag.pixels).astype(np.int32)], True, (255, 180, 0), 2)
        if result.render_pose is not None:
            axes = project(result.render_pose, np.array([[0, 0, 0], [.05, 0, 0], [0, .05, 0], [0, 0, .05]]), packet.camera)
            if np.isfinite(axes).all() and np.all(np.abs(axes) < 2**30):
                pts = np.rint(axes).astype(int)
                for end, color in zip(pts[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    cv2.line(image, tuple(pts[0]), tuple(end), color, 2)
    if display.draw_features:
        for f in features:
            cv2.circle(image, tuple(np.rint(f.pixel).astype(int)), 2, (255, 0, 255) if f.landmark_id is not None else (255, 200, 0), -1)
    if display.draw_uncertainty and result.pixel_covariance is not None:
        for p, cov in zip(result.raw_panel, result.pixel_covariance):
            values, vectors = np.linalg.eigh(cov)
            axes = tuple(np.maximum(1, np.rint(2*np.sqrt(np.maximum(values[::-1], 0)))).clip(1, 10000).astype(int))
            angle = np.rad2deg(np.arctan2(vectors[1, -1], vectors[0, -1]))
            cv2.ellipse(image, tuple(np.rint(p).astype(int)), axes, float(angle), 0, 360, (0, 180, 255), 1)
        cv2.putText(image, "Raw conditional ~2 sigma (assumed noise)", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
    text = f"{result.state} | {result.raw_pose.source} | ready map {result.diagnostics['map_size']}"
    cv2.putText(image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0) if result.raw_pose.valid else (0, 0, 255), 2)
    if result.diagnostics["mapping_mode"] == "build":
        progress = (f"Stable landmarks: {result.diagnostics['stable_landmarks']}/{result.diagnostics['landmark_target']}"
                    f" | keyframes: {result.diagnostics['keyframes']}/{result.diagnostics['keyframe_limit']}")
    elif result.diagnostics["map_ready"]:
        progress = f"Loaded map: {result.diagnostics['stable_landmarks']} landmarks"
    else:
        progress = "Landmark mapping: off"
    cv2.putText(image, progress, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 2)
    if result.diagnostics["bootstrap_capacity_reached"]:
        cv2.putText(image, "Keyframe limit reached; press R to reset", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 165, 255), 2)
    if approximate:
        cv2.putText(image, "DEMO: APPROXIMATE CALIBRATION", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 165, 255), 2)
    return image
