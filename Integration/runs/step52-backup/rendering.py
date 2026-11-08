"""Fixed world panel, pose EMA in T_WC, and display-only rounded pixels."""
import cv2
import numpy as np
from .array import clip_near
from .geometry import depths, inverse, panel_points, project, projection_jacobian


class PoseSmoother:
    def __init__(self, config):
        self.config = config
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, T, t=None, covariance=None):   # t, covariance: unused, the interface the other filters share
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


def visible_panel(T, panel, camera, znear=0.05):
    """Projected polygon of the panel clipped to camera Z >= znear (a panel partly behind the camera keeps its visible part;
    more than 4 vertices are possible), or None when it is fully behind or off the image."""
    if T is None:
        return None
    clipped = clip_near(T, panel, znear)
    if clipped is None:
        return None
    pixels = project(np.eye(4), clipped, camera)
    if not np.isfinite(pixels).all() or np.any(np.abs(pixels) >= 2**28):
        return None
    w, h = camera.size
    area, _ = cv2.intersectConvexConvex(pixels.astype(np.float32), np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32))
    if area <= 0:
        return None
    return pixels


def panel_covariance(T, panel, camera, cov, uncertainty):
    if cov is None or not uncertainty.enabled or not uncertainty.project_to_panel_corners or np.any(depths(T, panel) <= 0) or visible_panel(T, panel, camera) is None:
        return None
    G = projection_jacobian(T, panel, camera, uncertainty).reshape(-1, 2, 6)
    return np.array([g@cov@g.T for g in G])


def annotate(packet, result, tags, features, display, approximate=False, panel_layer=None):
    image = packet.bgr.copy()
    if panel_layer is not None:
        color, visible = panel_layer
        image[visible] = color[visible]
    elif result.render_panel is not None:
        cv2.fillConvexPoly(image, np.rint(result.render_panel).astype(np.int32), (0, 200, 0))
    if display.clean:
        return image
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
    if display.draw_uncertainty and result.pixel_covariance is not None and len(result.raw_panel) == len(result.pixel_covariance):
        for p, cov in zip(result.raw_panel, result.pixel_covariance):
            values, vectors = np.linalg.eigh(cov)
            axes = tuple(np.maximum(1, np.rint(2*np.sqrt(np.maximum(values[::-1], 0)))).clip(1, 10000).astype(int))
            angle = np.rad2deg(np.arctan2(vectors[1, -1], vectors[0, -1]))
            cv2.ellipse(image, tuple(np.rint(p).astype(int)), axes, float(angle), 0, 360, (0, 180, 255), 1)
        cv2.putText(image, "Raw conditional ~2 sigma (assumed noise)", (12, 216), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
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
    tracking = result.diagnostics.get("feature_tracking", {})
    if tracking.get("enabled") and "lk" in tracking:
        lk, orb = tracking["lk"], tracking["orb"]
        feature_line = f"LK {lk['retained']}/{lk['input']}"
        if orb["ran"]:
            feature_line += (f" | ORB[{orb['reason']}] {orb['keypoints']}kp "
                             f"{orb['descriptor_matches']}match {orb['reused_tracks']}reuse +{orb['new_tracks']}new")
        else:
            feature_line += " | ORB waiting"
        cv2.putText(image, feature_line, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    if result.diagnostics["bootstrap_capacity_reached"]:
        cv2.putText(image, "Keyframe limit reached; press R to reset", (12, 190), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 165, 255), 2)
    elif result.diagnostics["mapping_mode"] == "build" and not result.diagnostics["map_ready"]:
        cv2.putText(image, "Map: "+str(result.diagnostics.get("bootstrap_reason", ""))[:95],
                    (12, 190), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 210, 255), 1)
    associations = result.diagnostics.get("map_associations", {})
    gates = result.diagnostics.get("pose_gates", {})
    counts = (f"Map: LK {associations.get('inherited_lk_associations', 0)} + ORB {associations.get('fresh_descriptor_associations', 0)+associations.get('redetected_orb_associations', 0)}"
              f" | RANSAC {gates.get('ransac_inliers', 0)} | cells {gates.get('occupied_cells', 0)}")
    cv2.putText(image, counts, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    reason = gates.get("landmark_gate_reason") or "landmarks accepted"
    if associations.get("reason"):
        reason = associations["reason"]
    cv2.putText(image, "Landmarks: "+reason, (12, 134), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 210, 255), 1)
    status = result.diagnostics.get("panel_visibility", "")
    if result.raw_pose.reason:
        status += " | "+result.raw_pose.reason
    cv2.putText(image, "Panel: "+status, (12, 161), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    if approximate:
        cv2.putText(image, "DEMO: APPROXIMATE CALIBRATION", (12, 242), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 165, 255), 2)
    if "occlusion" in result.diagnostics:
        occ = result.diagnostics["occlusion"]
        cv2.putText(image, "Occlusion: "+occ["status"], (12, 268), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        if display.debug and "alignment_source" in occ:
            lines = [f"pose {result.raw_pose.source} | align {occ['alignment_source']}" + (f" | line: {occ['line_reason']}" if occ.get("line_reason") else ""),
                     f"panel Z {occ.get('panel_z_median_m', float('nan')):.2f} m | margin {occ.get('margin_m', float('nan')):.2f} m | hidden {occ.get('hidden_fraction', float('nan')):.2f}",
                     f"depth {occ.get('inference_ms', float('nan')):.0f} ms | occlusion {result.timings.get('occlusion', float('nan')):.0f} ms | total so far {result.timings['total']:.0f} ms"]
            for i, line in enumerate(lines):
                cv2.putText(image, line, (12, 292+22*i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    return image


def side_by_side(left, view, z_range=(.3, 5.), diff_range=.5, pane=(480, 360)):
    """Debug window: masked camera image | aligned depth Z_real (fixed colour scale, panel outline, base line) |
    decision map Z_panel - Z_real inside the panel (red = scene nearer; white contour = where the mask flips)."""
    scene, z, visible = view["scene"], view["z"], view["visible"]
    on = z > 0
    lo, hi = z_range
    depth = cv2.applyColorMap((np.clip((np.where(np.isfinite(scene), scene, hi)-lo)/(hi-lo), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    outline = cv2.findContours(on.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(depth, outline, -1, (255, 255, 255), 1)
    if view.get("line_uv") is not None:
        for u, v in view["line_uv"][::4]:
            cv2.circle(depth, (int(u), int(v)), 1, (0, 0, 0), -1)
    for i, label in enumerate((f"Z_real {lo:g}-{hi:g} m (turbo, near=blue)", )):
        cv2.putText(depth, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    diff = np.where(on & np.isfinite(scene), z-scene, 0.)
    decision = np.full(depth.shape, 90, np.uint8)
    gray = cv2.applyColorMap((np.clip(diff/(2*diff_range)+.5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_COOL)
    decision[on] = gray[on]
    cv2.drawContours(decision, cv2.findContours((on & ~visible).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 255), 1)
    cv2.putText(decision, f"Z_panel - Z_real, +-{diff_range:g} m; white = hidden edge", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    panes = [cv2.resize(x, pane, interpolation=cv2.INTER_AREA) for x in (left, depth, decision)]
    if view.get("fg") is not None:                       # Step 48b: foreground mask and stored background map
        fg = cv2.cvtColor((view["fg"]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        stored = cv2.applyColorMap((np.clip((np.where(view["prior"] > 0, view["prior"], hi)-lo)/(hi-lo), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        stored[view["prior"] <= 0] = 60
        cv2.putText(fg, "foreground (nearer than stored background)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        cv2.putText(stored, "stored background depth (grey = none)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        second = [cv2.resize(x, pane, interpolation=cv2.INTER_AREA) for x in (fg, stored)]+[np.zeros_like(panes[0])]
        return np.vstack([np.hstack(panes), np.hstack(second)])
    return np.hstack(panes)


def depth_pane(view, shape, z_range=(.3, 5.)):
    """Text-free turbo depth frame: the stored temporal background (grey = none) when the prior is on, else the one-shot depth."""
    lo, hi = z_range
    if view is None:
        return np.zeros(shape, np.uint8)
    prior = view.get("prior")
    depth = prior if prior is not None else np.where(np.isfinite(view["scene"]), view["scene"], hi)
    out = cv2.applyColorMap((np.clip((np.where(depth > 0, depth, hi)-lo)/(hi-lo), 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if prior is not None:
        out[prior <= 0] = 60
    return out


def open_window(name):
    """Resizable window (the whole image, text included, scales with it); key f toggles full screen via toggle_fullscreen."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)


def toggle_fullscreen(name):
    full = cv2.getWindowProperty(name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
