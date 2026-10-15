"""AprilTag 36h11 observations with IPPE candidates and explicit planar ambiguity."""
import cv2
import numpy as np
from .geometry import tag_points, transform, project, depths, rotation_gap
from .types import TagObservation


class TagDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def observation(self, marker_id, pixels, camera):
        c = self.cfg
        world = tag_points(c.size_m)
        obs = TagObservation(int(marker_id), np.asarray(pixels, np.float64).reshape(4, 2).copy(), world)
        if obs.id != c.anchor_id:
            obs.reason = "not configured anchor"
            return obs
        w, h = camera.size
        if not np.isfinite(obs.pixels).all() or np.any(obs.pixels < 0) or np.any(obs.pixels >= [w, h]):
            obs.reason = "invalid/out-of-image tag corners"
            return obs
        if np.min(np.linalg.norm(obs.pixels-np.roll(obs.pixels, 1, axis=0), axis=1)) < c.min_side_px:
            obs.reason = "tag too small"
            return obs
        solved, rvecs, tvecs, _ = cv2.solvePnPGeneric(world, obs.pixels, camera.K, camera.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if solved:
            for rvec, tvec in zip(rvecs, tvecs):
                if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                    continue
                T = transform(rvec, tvec)
                error = float(np.sqrt(np.mean(np.sum((project(T, world, camera)-obs.pixels)**2, axis=1))))
                if np.all(depths(T, world) > 0) and error <= c.max_reprojection_rms_px:
                    obs.candidates.append(T)
                    obs.errors.append(error)
        order = np.argsort(obs.errors)
        obs.candidates = [obs.candidates[i] for i in order]
        obs.errors = [obs.errors[i] for i in order]
        obs.valid = bool(obs.candidates)
        if not obs.valid:
            obs.reason = "no positive-depth low-error IPPE candidate"
        elif len(obs.candidates) > 1:
            obs.ambiguous = (obs.errors[1]-obs.errors[0] < c.ambiguity_error_gap_px and
                             np.rad2deg(rotation_gap(*obs.candidates[:2])) > c.ambiguity_rotation_gap_deg)
            if obs.ambiguous:
                obs.reason = "planar pose ambiguity: additional viewpoint or map evidence required"
        return obs

    def detect(self, gray, camera):
        if not self.cfg.enabled:
            return []
        corners, ids, _ = self.detector.detectMarkers(gray)
        observations = []
        if ids is not None:
            for pixels, marker_id in zip(corners, ids.ravel()):
                w = self.cfg.corner_refinement_window_px
                p = pixels.reshape(4, 2)
                if np.all(p >= w+1) and np.all(p < np.array(camera.size)-w-1):
                    p = cv2.cornerSubPix(gray, p.astype(np.float32).reshape(4, 1, 2), (w, w), (-1, -1),
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, .01)).reshape(4, 2)
                observations.append(self.observation(marker_id, p, camera))
        return observations
