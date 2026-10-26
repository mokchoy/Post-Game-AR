"""Metres, radians, T_CW maps W into C. Left local retraction is [dt,dr]."""
import cv2
import numpy as np
from .types import CameraModel


def transform(rvec, tvec):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def valid_transform(T):
    return (T.shape == (4, 4) and np.isfinite(T).all() and np.allclose(T[3], [0, 0, 0, 1])
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-7) and abs(np.linalg.det(T[:3, :3])-1) < 1e-7)


def inverse(T):
    if not valid_transform(T):
        raise ValueError("invalid rigid T_CW")
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def camera_position(T):
    return inverse(T)[:3, 3]


def retract(T, delta):
    return transform(delta[3:], delta[:3]) @ T


def depths(T, points):
    return (np.asarray(points, np.float64) @ T[:3, :3].T + T[:3, 3])[:, 2]


def project(T, points, camera: CameraModel):
    return cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 3), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], camera.K, camera.distortion)[0].reshape(-1, 2)


def rotation_gap(A, B):
    return float(np.linalg.norm(cv2.Rodrigues(A[:3, :3] @ B[:3, :3].T)[0]))


def tag_points(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def panel_points(width, height, elevation, tilt_deg):
    a, b = width / 2, height / 2
    corners = np.array([[-a, -b, 0], [a, -b, 0], [a, b, 0], [-a, b, 0]], np.float64)
    R = cv2.Rodrigues(np.array([np.deg2rad(tilt_deg), 0., 0.]))[0]
    return corners @ R.T + [0, elevation, height / 2]


def projection_jacobian(T, points, camera, uncertainty):
    steps = [uncertainty.translation_difference_step_m]*3 + [uncertainty.rotation_difference_step_rad]*3
    J = np.empty((len(points)*2, 6))
    for i, step in enumerate(steps):
        delta = np.zeros(6)
        delta[i] = step
        J[:, i] = ((project(retract(T, delta), points, camera) - project(retract(T, -delta), points, camera)) / (2*step)).ravel()
    return J


def triangulate(T1, T2, u1, u2, camera, cfg):
    """Two normalized rays; reject geometry before returning a 3D point."""
    if np.linalg.norm(camera_position(T1)-camera_position(T2)) < cfg.min_baseline_m:
        return None
    rays = [cv2.undistortPoints(np.asarray(u, np.float64).reshape(1, 1, 2), camera.K, camera.distortion).reshape(2) for u in (u1, u2)]
    X = cv2.triangulatePoints(T1[:3], T2[:3], rays[0].reshape(2, 1), rays[1].reshape(2, 1)).ravel()
    if not np.isfinite(X).all() or abs(X[3]) < 1e-12:
        return None
    X = X[:3]/X[3]
    ds = np.array([depths(T, X[None])[0] for T in (T1, T2)])
    if np.any(ds < cfg.min_depth_m) or np.any(ds > cfg.max_depth_m):
        return None
    vectors = [X-camera_position(T) for T in (T1, T2)]
    cos = np.dot(*vectors) / (np.linalg.norm(vectors[0])*np.linalg.norm(vectors[1]))
    if np.rad2deg(np.arccos(np.clip(cos, -1, 1))) < cfg.min_parallax_deg:
        return None
    error = np.array([project(T, X[None], camera)[0]-u for T, u in ((T1, u1), (T2, u2))])
    if np.sqrt(np.mean(np.sum(error**2, axis=1))) > cfg.max_reprojection_rms_px:
        return None
    return X
