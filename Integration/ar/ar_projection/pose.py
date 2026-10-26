"""PnP-RANSAC and one noise-weighted robust reprojection solve, never pose averaging."""
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from .geometry import camera_position, depths, project, projection_jacobian, retract, rotation_gap, transform, valid_transform
from .types import PoseEstimate


def coverage(pixels, size, features):
    w, h = size
    inside = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)]
    cells = np.floor(inside / [w/features.grid_cols, h/features.grid_rows]).astype(int)
    return len(np.unique(cells, axis=0))


def rms(error):
    return float(np.sqrt(np.mean(np.sum(error**2, axis=1)))) if len(error) else None


def robust_weights(residual, loss, scale):
    z = (residual/scale)**2
    if loss == "linear":
        return np.ones_like(z)
    if loss == "huber":
        return 1/np.sqrt(np.maximum(1, z))
    if loss == "soft_l1":
        return 1/np.sqrt(1+z)
    if loss == "cauchy":
        return 1/(1+z)
    return 1/(1+z*z)  # arctan


def covariance(T, world, pixels, sigma, camera, config):
    """Conditional robust-weighted local inverse information, LEFT [dt,dr]."""
    u = config.uncertainty
    if not u.enabled:
        return None, "disabled"
    J = projection_jacobian(T, world, camera, u) / np.repeat(sigma, 2)[:, None]
    residual = ((project(T, world, camera)-pixels)/sigma[:, None]).ravel()
    weights = robust_weights(residual, config.pose.robust_loss, config.pose.robust_f_scale)
    scales = np.array([u.conditioning_translation_scale_m]*3+[u.conditioning_rotation_scale_rad]*3)
    A = J*np.sqrt(weights)[:, None]*scales
    if not np.isfinite(A).all():
        return None, "nonfinite Jacobian"
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    if len(s) < 6 or s[0] <= 0 or s[-1]/s[0] < u.min_scaled_singular_value_ratio:
        return None, "rank deficient or ill-conditioned scaled Jacobian"
    cov = scales[:, None]*((vt.T/(s*s))@vt)*scales[None, :]
    cov = (cov+cov.T)/2
    if not np.isfinite(cov).all() or np.min(np.linalg.eigvalsh(cov)) < -1e-12:
        return None, "nonfinite or non-PSD covariance"
    return cov, None


class PoseSolver:
    def __init__(self, config):
        self.config = config

    def estimate(self, tags, landmark_world, landmark_pixels, camera, previous=None, dt=None):
        self.last_diagnostics = {"mapped_input": len(landmark_world), "ransac_attempted": False,
                                 "ransac_inliers": 0, "occupied_cells": 0,
                                 "landmark_gate_reason": "insufficient mapped observations",
                                 "policy": "landmarks must pass standalone PnP before fusion"}
        estimate = self._estimate(tags, landmark_world, landmark_pixels, camera, previous, dt)
        estimate.diagnostics.update(self.last_diagnostics)
        estimate.diagnostics["final_landmark_inliers"] = estimate.landmark_inliers
        if self.last_diagnostics["landmark_gate_reason"] is None:
            if not estimate.valid:
                estimate.diagnostics["landmark_gate_reason"] = "pose rejected: "+str(estimate.reason)
            elif not estimate.landmark_inliers:
                estimate.diagnostics["landmark_gate_reason"] = "landmarks rejected during refinement"
        return estimate

    def _estimate(self, tags, landmark_world, landmark_pixels, camera, previous=None, dt=None):
        c = self.config
        p = c.pose
        X = np.asarray(landmark_world, np.float64).reshape(-1, 3)
        U = np.asarray(landmark_pixels, np.float64).reshape(-1, 2)
        if len(X) != len(U) or not np.isfinite(X).all() or not np.isfinite(U).all():
            raise ValueError("pose: invalid landmark observation arrays")
        inside = np.all((U >= 0) & (U < camera.size), axis=1)
        X, U = X[inside], U[inside]
        self.last_diagnostics["in_image_input"] = len(X)
        if p.mode == "tag_only" or not c.features.enabled:
            X, U = X[:0], U[:0]
            self.last_diagnostics["landmark_gate_reason"] = "disabled by pose mode or feature configuration"
        tag = next((t for t in tags if t.id == c.tags.anchor_id and t.valid), None) if p.mode != "landmark_only" and c.tags.enabled else None
        seeds = []
        if len(X) >= p.min_landmark_inliers:
            self.last_diagnostics.update(ransac_attempted=True, landmark_gate_reason="PnP RANSAC failed")
            ok, rv, tv, inliers = cv2.solvePnPRansac(X, U, camera.K, camera.distortion,
                                                  iterationsCount=p.ransac_iterations, reprojectionError=p.ransac_reprojection_threshold_px,
                                                  confidence=p.ransac_confidence, flags=cv2.SOLVEPNP_EPNP)
            if ok and inliers is not None:
                selected = inliers.ravel()
                X, U = X[selected], U[selected]
                cells = coverage(U, camera.size, c.features)
                self.last_diagnostics.update(ransac_inliers=len(X), occupied_cells=cells,
                                             landmark_gate_reason="insufficient RANSAC inliers" if len(X) < p.min_landmark_inliers else
                                             "insufficient image coverage" if cells < p.min_occupied_grid_cells else "invalid landmark pose/depth")
                if len(X) >= p.min_landmark_inliers and cells >= p.min_occupied_grid_cells:
                    candidate = transform(rv, tv)
                    if valid_transform(candidate) and np.all(depths(candidate, X) > 0):
                        seeds.append(candidate)
                        self.last_diagnostics["landmark_gate_reason"] = None
        if not seeds:
            X, U = X[:0], U[:0]
        if tag is not None and tag.ambiguous and not len(X):
            tag = None
            tag_reason = "unresolved planar tag ambiguity"
        else:
            tag_reason = "no supported current observations"
        # Independently mapped evidence can reject a bad tag observation as a group.
        rejected_tag = False
        if tag is not None and seeds and rms(project(seeds[0], tag.world, camera)-tag.pixels) > p.max_tag_rms_px:
            tag = None
            rejected_tag = True
        if tag is not None:
            seeds += tag.candidates if len(X) else tag.candidates[:1]
        if previous is not None and dt is not None and 0 < dt <= max(p.max_prediction_gap_s, p.prior_grace_s) and seeds:
            seeds.append(previous)
        if not seeds:
            return PoseEstimate(reason=tag_reason)
        attempts = [self._refine(seed, tag, X, U, camera) for seed in seeds]
        accepted = [a for a in attempts if a[0].valid]
        if not accepted:
            return attempts[0][0]
        estimate, world, pixels, sigma, cost = min(accepted, key=lambda a: a[4])
        # Similar minima in distinct modes are not represented by local covariance.
        if tag is not None and tag.ambiguous:
            for other, _, _, _, score in accepted:
                if rotation_gap(other.T_CW, estimate.T_CW) > np.deg2rad(c.tags.ambiguity_rotation_gap_deg) and abs(score-cost) < c.tags.ambiguity_error_gap_px:
                    return PoseEstimate(reason="mapped evidence did not resolve planar ambiguity")
        if previous is not None and dt is not None and 0 < dt <= max(p.max_prediction_gap_s, p.prior_grace_s):
            linear = np.linalg.norm(camera_position(estimate.T_CW)-camera_position(previous))/dt
            angular = np.rad2deg(rotation_gap(estimate.T_CW, previous))/dt
            if linear > p.max_linear_speed_m_s or angular > p.max_angular_speed_deg_s:
                return PoseEstimate(reason="timestamp-based pose speed gate rejected jump")
        if rejected_tag:
            estimate.reason = "tag rejected: inconsistent with landmark geometry"
        start = time.perf_counter()
        estimate.covariance, estimate.covariance_reason = covariance(estimate.T_CW, world, pixels, sigma, camera, c)
        estimate.covariance_ms = (time.perf_counter()-start)*1000
        return estimate

    def _refine(self, seed, tag, X, U, camera):
        c, p = self.config, self.config.pose
        input_landmarks = len(X)
        Tw = tag.world if tag is not None else np.empty((0, 3))
        Tu = tag.pixels if tag is not None else np.empty((0, 2))
        status = None
        T = seed
        tag_rejected = False
        for iteration in range(2):
            world = np.vstack((Tw, X))
            pixels = np.vstack((Tu, U))
            sigma = np.array([c.tags.corner_sigma_px]*len(Tw)+[c.features.observation_sigma_px]*len(X))
            if len(world) < 4:
                return PoseEstimate(reason="not enough observations after rejection"), world, pixels, sigma, np.inf
            def residual(delta):
                return ((project(retract(T, delta), world, camera)-pixels)/sigma[:, None]).ravel()
            fit = least_squares(residual, np.zeros(6), method="trf", jac="3-point", x_scale=1.0, tr_solver="exact",
                                loss=p.robust_loss, f_scale=p.robust_f_scale, max_nfev=p.max_nfev)
            status = int(fit.status)
            if not fit.success or not np.isfinite(fit.x).all():
                return PoseEstimate(optimization_status=status, reason="pose refinement did not converge"), world, pixels, sigma, np.inf
            T = retract(T, fit.x)
            errors = project(T, world, camera)-pixels
            tag_error = errors[:len(Tw)]
            feature_error = errors[len(Tw):]
            keep = np.linalg.norm(feature_error, axis=1) <= p.ransac_reprojection_threshold_px
            bad_tag = len(Tw) and rms(tag_error) > p.max_tag_rms_px
            if iteration == 0 and (not keep.all() or bad_tag):
                X, U = X[keep], U[keep]
                if bad_tag:
                    Tw, Tu = Tw[:0], Tu[:0]
                    tag_rejected = True
                if len(X) < p.min_landmark_inliers or coverage(U, camera.size, c.features) < p.min_occupied_grid_cells:
                    X, U = X[:0], U[:0]
                continue
            break
        feature_rms, tag_rms = rms(feature_error), rms(tag_error)
        enough = len(X) >= p.min_landmark_inliers and coverage(U, camera.size, c.features) >= p.min_occupied_grid_cells
        reason = None
        if not valid_transform(T) or not np.isfinite(errors).all() or np.any(depths(T, world) <= 0):
            reason = "invalid transform or nonpositive observation depth"
        elif len(X) and (not enough or not keep.all() or feature_rms > p.max_landmark_rms_px):
            reason = "landmark count/coverage/residual gate"
        elif len(Tw) and tag_rms > p.max_tag_rms_px:
            reason = "tag residual gate"
        elif not len(Tw) and not enough:
            reason = "no sufficient current evidence"
        source = "joint" if len(Tw) and enough else "tag" if len(Tw) else "landmark"
        estimate = PoseEstimate(T_CW=T if reason is None else None, valid=reason is None, source=source if reason is None else "none",
                                landmark_inliers=len(X), tag_inliers=len(Tw), landmark_rms=feature_rms, tag_rms=tag_rms,
                                optimization_status=status, reason=reason or ("tag rejected by joint residuals" if tag_rejected else None),
                                diagnostics={"refinement_occupied_cells": coverage(U, camera.size, c.features),
                                             "refinement_landmarks_rejected": input_landmarks-len(X),
                                             "refinement_landmark_rms_px": feature_rms,
                                             "refinement_reason": reason})
        cost = float(np.mean(((errors/sigma[:, None])**2)))
        return estimate, world, pixels, sigma, cost
