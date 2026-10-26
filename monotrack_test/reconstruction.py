"""Small, annotation-assisted MonoTrack reconstruction core for one rally."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
NET_Y_M = 6.7
POST_TOP_M = 1.55
ANNOTATION_TO_VIDEO = 1.5
SOURCE_COMMIT = "39e9227bd44fa2a579731b1cc97c7f529f391cd6"
MIN_VISIBLE_PER_FLIGHT = 3


def transform_point(matrix: np.ndarray, point: tuple[float, float] | np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(1, 1, 2)
    return cv2.perspectiveTransform(point, np.asarray(matrix, dtype=np.float64))[0, 0]


@dataclass(frozen=True)
class CourtGeometry:
    row_id: str
    row_video: str
    homography_annotation_to_dataset: np.ndarray
    dataset_to_metric: np.ndarray
    corners_annotation: np.ndarray
    mapped_corners: np.ndarray
    mapped_bounds: tuple[float, float, float, float]

    @property
    def annotation_to_metric(self) -> np.ndarray:
        return self.dataset_to_metric @ self.homography_annotation_to_dataset

    @property
    def metric_to_annotation(self) -> np.ndarray:
        return np.linalg.inv(self.annotation_to_metric)

    def annotation_floor_to_world(self, pixel: tuple[float, float] | np.ndarray) -> np.ndarray:
        dataset_xy = transform_point(self.homography_annotation_to_dataset, pixel)
        return transform_point(self.dataset_to_metric, dataset_xy)

    def native_floor_to_world(self, pixel: tuple[float, float] | np.ndarray) -> np.ndarray:
        return self.annotation_floor_to_world(np.asarray(pixel, dtype=np.float64) / ANNOTATION_TO_VIDEO)

    def world_to_annotation_floor(self, xy_m: tuple[float, float] | np.ndarray) -> np.ndarray:
        return transform_point(self.metric_to_annotation, xy_m)


def load_court_geometry(homography_csv: str | Path, match_name: str) -> CourtGeometry:
    with Path(homography_csv).open("r", newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("video") == match_name]
    if len(rows) != 1:
        raise ValueError(f"Expected one homography row for {match_name!r}; found {len(rows)}")
    row = rows[0]
    homography = np.asarray(json.loads(row["homography_matrix"]), dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("The stored homography must be a finite 3x3 matrix")
    corner_keys = ("upleft", "upright", "downleft", "downright")
    corners = np.asarray(
        [[float(row[f"{name}_x"]), float(row[f"{name}_y"])] for name in corner_keys],
        dtype=np.float64,
    )
    mapped = np.asarray([transform_point(homography, corner) for corner in corners])
    bounds = (
        float(mapped[:, 0].min()),
        float(mapped[:, 1].min()),
        float(mapped[:, 0].max()),
        float(mapped[:, 1].max()),
    )
    expected = np.asarray([25.0, 150.0, 325.0, 810.0])
    if not np.allclose(bounds, expected, atol=5.0):
        raise ValueError(
            f"Mapped court bounds {bounds} do not match the expected dataset range "
            "x=25..325, y=150..810"
        )

    metric_corners = np.asarray(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [0.0, COURT_LENGTH_M],
         [COURT_WIDTH_M, COURT_LENGTH_M]],
        dtype=np.float32,
    )
    dataset_to_metric = cv2.getPerspectiveTransform(mapped.astype(np.float32), metric_corners)
    if not np.isfinite(dataset_to_metric).all() or abs(np.linalg.det(dataset_to_metric)) < 1e-12:
        raise ValueError("Could not derive an invertible dataset-to-metre transform")
    return CourtGeometry(
        row_id=row["id"],
        row_video=row["video"],
        homography_annotation_to_dataset=homography,
        dataset_to_metric=dataset_to_metric.astype(np.float64),
        corners_annotation=corners,
        mapped_corners=mapped,
        mapped_bounds=bounds,
    )


def calibration_world_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [COURT_WIDTH_M, 0.0, 0.0],
            [0.0, COURT_LENGTH_M, 0.0],
            [COURT_WIDTH_M, COURT_LENGTH_M, 0.0],
            [0.0, NET_Y_M, POST_TOP_M],
            [COURT_WIDTH_M, NET_Y_M, POST_TOP_M],
        ],
        dtype=np.float64,
    )


def fit_camera_projection(corners_native: np.ndarray, post_tips_native: np.ndarray) -> tuple[np.ndarray, float]:
    """Adapt MonoTrack's six-point, fixed-scale linear camera calibration."""
    image_points = np.vstack((corners_native, post_tips_native)).astype(np.float64)
    world_points = calibration_world_points()
    if image_points.shape != (6, 2) or not np.isfinite(image_points).all():
        raise ValueError("Camera calibration requires four court corners and two post tips")
    design = np.zeros((12, 11), dtype=np.float64)
    target = np.zeros((12, 1), dtype=np.float64)
    for i, (world, image) in enumerate(zip(world_points, image_points, strict=True)):
        for axis in range(2):
            row = 2 * i + axis
            offset = 4 if axis else 0
            design[row, offset:offset + 3] = world
            design[row, offset + 3] = 1.0
            design[row, -3:] = -image[axis] * world
            target[row, 0] = image[axis]
    if np.linalg.matrix_rank(design) < 11:
        raise ValueError("The six calibration correspondences do not determine a camera matrix")
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    projection = np.zeros((3, 4), dtype=np.float64)
    for index in range(11):
        projection[index // 4, index % 4] = coefficients[index, 0]
    projection[2, 3] = 1.0
    projected = np.asarray([project_point(projection, p) for p in world_points])
    rmse = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
    if not np.isfinite(projection).all() or not np.isfinite(rmse):
        raise ValueError("Camera calibration produced non-finite values")
    return projection, rmse


def project_point(projection: np.ndarray, point_m: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    q = np.asarray(projection, dtype=np.float64) @ np.append(np.asarray(point_m, dtype=np.float64), 1.0)
    if abs(q[2]) < 1e-10:
        return np.asarray([np.nan, np.nan])
    return q[:2] / q[2]


def intersect_camera_ray_y(
    projection: np.ndarray, pixel_native: tuple[float, float] | np.ndarray, y_m: float = NET_Y_M
) -> np.ndarray:
    """Intersect a camera ray with a vertical court plane; never use the floor homography here."""
    matrix = np.asarray(projection, dtype=np.float64)
    left = matrix[:, :3]
    center = -np.linalg.solve(left, matrix[:, 3])
    direction = np.linalg.solve(left, np.asarray([pixel_native[0], pixel_native[1], 1.0]))
    if abs(direction[1]) < 1e-10:
        raise ValueError("Camera ray is parallel to the requested net plane")
    scale = (float(y_m) - center[1]) / direction[1]
    point = center + scale * direction
    if scale <= 0 or not np.isfinite(point).all():
        raise ValueError("Camera ray does not meet the net plane in front of the camera")
    return point


def frame_time_seconds(frame: int, fps: float) -> float:
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("FPS must be positive and finite")
    return int(frame) / float(fps)


def visible_observations(observations: list[tuple[float, float] | None]) -> list[tuple[int, np.ndarray]]:
    return [
        (index, np.asarray(point, dtype=np.float64))
        for index, point in enumerate(observations)
        if point is not None and np.isfinite(point).all()
    ]


def validate_pixel_loss(pixel_loss: str, huber_delta_px: float) -> tuple[str, float]:
    if pixel_loss not in ("mse", "huber"):
        raise ValueError("pixel_loss must be 'mse' or 'huber'")
    try:
        delta = float(huber_delta_px)
    except (TypeError, ValueError) as error:
        raise ValueError("huber_delta_px must be positive and finite") from error
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("huber_delta_px must be positive and finite")
    return pixel_loss, delta


def pixel_residual_cost(errors: list[float] | np.ndarray, pixel_loss: str, huber_delta_px: float) -> float:
    if pixel_loss == "mse":
        return float(np.mean(np.square(errors)))
    errors = np.asarray(errors, dtype=np.float64)
    return float(np.mean(np.where(
        errors <= huber_delta_px,
        np.square(errors),
        2.0 * huber_delta_px * errors - huber_delta_px ** 2,
    )))


def simulate_trajectory(
    start_m: np.ndarray,
    velocity_mps: np.ndarray,
    drag: float,
    frame_count: int,
    fps: float,
    substeps: int = 30,
    stop_at_ground: bool = False,
) -> np.ndarray:
    """Return frame positions, optionally stopping at an interpolated ground crossing."""
    if frame_count < 0 or fps <= 0 or substeps < 1:
        raise ValueError("Invalid trajectory timing")
    position = np.asarray(start_m, dtype=np.float64).copy()
    velocity = np.asarray(velocity_mps, dtype=np.float64).copy()
    result = [position.copy()]
    if stop_at_ground and position[2] <= 0 and velocity[2] <= 0:
        return np.asarray(result)
    dt = 1.0 / (float(fps) * substeps)
    gravity = np.asarray([0.0, 0.0, -9.8])
    for _ in range(frame_count):
        for _ in range(substeps):
            previous = position.copy() if stop_at_ground else None
            velocity += dt * (gravity - float(drag) * np.linalg.norm(velocity) * velocity)
            position += dt * velocity
            if stop_at_ground:
                if not np.isfinite(position).all():
                    raise FloatingPointError("Non-finite hypothetical flight")
                if previous[2] >= 0 and position[2] <= 0:
                    fraction = previous[2] / (previous[2] - position[2])
                    landing = previous + fraction * (position - previous)
                    landing[2] = 0.0
                    return np.asarray(result + [landing])
        result.append(position.copy())
    return np.asarray(result)


def extrapolated_landing_distance(start_m: np.ndarray, velocity_mps: np.ndarray, drag: float, fps: float) -> float | None:
    """Distance outside the full court if this shot continues without another hit."""
    if (np.asarray(start_m).shape != (3,) or np.asarray(velocity_mps).shape != (3,)
            or not np.isfinite(start_m).all() or not np.isfinite(velocity_mps).all()
            or not np.isfinite(drag) or drag < 0 or not np.isfinite(fps) or fps <= 0 or start_m[2] < 0):
        return None
    # shortcut: cap hypothetical flights at 60 s; revisit if velocity/height bounds grow.
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            landing = simulate_trajectory(start_m, velocity_mps, drag, int(np.ceil(60 * fps)), fps,
                                          stop_at_ground=True)[-1]
        except FloatingPointError:
            return None
    if landing[2] != 0:
        return None
    nearest = np.clip(landing[:2], [0.0, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M])
    return float(np.linalg.norm(landing[:2] - nearest))


def reconstruct_flight(
    *,
    start_frame: int,
    end_frame: int,
    observations: list[tuple[float, float] | None],
    projection: np.ndarray,
    start_reference_xy: np.ndarray,
    receive_reference_xy: np.ndarray,
    fps: float,
    start_fixed: np.ndarray | None = None,
    end_fixed: np.ndarray | None = None,
    phase: str = "hit_to_hit",
    pixel_loss: str = "mse",
    huber_delta_px: float = 10.0,
) -> tuple[np.ndarray | None, dict]:
    """Fit one gravity-plus-quadratic-drag flight with frame-aligned reprojection loss."""
    pixel_loss, huber_delta_px = validate_pixel_loss(pixel_loss, huber_delta_px)
    configuration = {"pixel_loss": pixel_loss, "huber_delta_px": huber_delta_px}
    duration_frames = int(end_frame) - int(start_frame)
    if duration_frames < 2:
        return None, {"phase": phase, **configuration, "status": "insufficient_data", "reason": "fewer_than_3_frames"}
    if start_frame < 0 or end_frame >= len(observations):
        return None, {"phase": phase, **configuration, "status": "insufficient_data", "reason": "frame_range_outside_tracks"}
    local = observations[start_frame:end_frame + 1]
    visible = visible_observations(local)
    base = {
        "phase": phase,
        **configuration,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "duration_seconds": frame_time_seconds(duration_frames, fps),
        "visible_detections": len(visible),
        "missing_detections": len(local) - len(visible),
        "start_fixed": start_fixed is not None,
        "end_fixed": end_fixed is not None,
    }
    if len(visible) < MIN_VISIBLE_PER_FLIGHT:
        return None, {**base, "status": "insufficient_data", "reason": "fewer_than_3_visible_detections"}
    if fps <= 0 or not np.isfinite(projection).all():
        return None, {**base, "status": "unidentifiable", "reason": "invalid_fps_or_camera"}

    start_reference_xy = np.asarray(start_reference_xy, dtype=np.float64)
    receive_reference_xy = np.asarray(receive_reference_xy, dtype=np.float64)
    fixed_start = None if start_fixed is None else np.asarray(start_fixed, dtype=np.float64)
    fixed_end = None if end_fixed is None else np.asarray(end_fixed, dtype=np.float64)
    manual_floor_endpoint = fixed_end is not None and fixed_end[2] == 0.0
    landing_weight = 0.0 if manual_floor_endpoint else 0.15
    duration = duration_frames / float(fps)
    if fixed_start is None:
        y0 = float(np.clip(start_reference_xy[1], 0.0, COURT_LENGTH_M))
        x0 = float(np.clip(start_reference_xy[0], 0.0, COURT_WIDTH_M))
        initial_start = np.asarray([x0, y0, 0.45 * 1.75])
    else:
        initial_start = fixed_start.copy()
    target_xy = fixed_end[:2] if fixed_end is not None else receive_reference_xy
    initial_velocity = (target_xy - initial_start[:2]) / duration
    initial_vz = 9.8 * duration / 2.0 - 1.0 / duration
    if fixed_end is not None:
        initial_vz = (fixed_end[2] - initial_start[2]) / duration + 4.9 * duration
    initial_vz = float(np.clip(initial_vz, -150.0, 150.0))
    vy_direction = float(target_xy[1] - initial_start[1])
    if abs(vy_direction) < 0.05:
        vy_bounds = (-150.0, 150.0)
    elif vy_direction > 0:
        vy_bounds = (0.0, 150.0)
    else:
        vy_bounds = (-150.0, 0.0)
    initial_velocity[1] = np.clip(initial_velocity[1], *vy_bounds)
    initial_velocity = np.append(initial_velocity, initial_vz)

    if fixed_start is None:
        half_bounds = (0.0, NET_Y_M) if start_reference_xy[1] <= NET_Y_M else (NET_Y_M, COURT_LENGTH_M)
        initial = np.concatenate((initial_start, initial_velocity, [0.2]))
        bounds = [(0.0, COURT_WIDTH_M), half_bounds, (0.1, 6.0)]
        bounds += [(-150.0, 150.0), vy_bounds, (-150.0, 150.0), (0.0, 0.4)]
    else:
        initial = np.concatenate((initial_velocity, [0.2]))
        bounds = [(-150.0, 150.0), vy_bounds, (-150.0, 150.0), (0.0, 0.4)]

    camera_scale = float(np.linalg.norm(np.asarray(projection)[:, :3], ord=2))

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if fixed_start is None:
            start = params[:3]
            velocity = params[3:6]
            drag = float(params[6])
        else:
            start = fixed_start
            velocity = params[:3]
            drag = float(params[3])
        return start, velocity, drag

    def trajectory_for(params: np.ndarray) -> np.ndarray:
        start, velocity, drag = unpack(params)
        return simulate_trajectory(start, velocity, drag, duration_frames, fps)

    def objective(params: np.ndarray) -> float:
        path = trajectory_for(params)
        errors = []
        for index, observed in visible:
            projected = project_point(projection, path[index])
            if not np.isfinite(projected).all():
                return 1e12
            errors.append(float(np.linalg.norm(projected - observed)))
        cost = pixel_residual_cost(errors, pixel_loss, huber_delta_px)
        if fixed_start is None:
            start_error_m = float(np.linalg.norm(path[0, :2] - start_reference_xy))
            cost += (0.15 * camera_scale * start_error_m) ** 2
        if fixed_end is None:
            endpoint_error_m = float(np.linalg.norm(path[-1, :2] - receive_reference_xy))
            cost += (0.15 * camera_scale * endpoint_error_m) ** 2
        start, velocity, drag = unpack(params)
        landing_distance_m = extrapolated_landing_distance(start, velocity, drag, fps)
        if landing_distance_m is None:
            return 1e12
        # A measured landing may be out; the court prior must not fight that endpoint.
        cost += (landing_weight * camera_scale * landing_distance_m) ** 2
        return cost if np.isfinite(cost) else 1e12

    constraints = []
    if fixed_end is not None:
        constraints.append({"type": "eq", "fun": lambda params: trajectory_for(params)[-1] - fixed_end})
    if manual_floor_endpoint:
        constraints.append({"type": "ineq", "fun": lambda params: trajectory_for(params)[:-1, 2]})
    try:
        result = minimize(
            objective,
            initial,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 160, "ftol": 1e-6, "disp": False},
        )
        path = trajectory_for(result.x)
        start, velocity, drag = unpack(result.x)
        landing_distance_m = extrapolated_landing_distance(start, velocity, drag, fps)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return None, {**base, "status": "failed", "reason": f"optimizer_error: {error}"}

    reprojection_errors = np.asarray(
        [np.linalg.norm(project_point(projection, path[index]) - point) for index, point in visible],
        dtype=np.float64,
    )
    endpoint_error = 0.0 if fixed_end is None else float(np.linalg.norm(path[-1] - fixed_end))
    available = (
        bool(result.success)
        and np.isfinite(path).all()
        and np.isfinite(reprojection_errors).all()
        and endpoint_error <= 0.01
        and landing_distance_m is not None
        and (not manual_floor_endpoint or np.all(path[:-1, 2] >= -1e-6))
    )
    diagnostics = {
        **base,
        "status": "available" if available else "failed",
        "optimizer": "SLSQP",
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "objective": float(result.fun),
        "landing_outside_distance_m": landing_distance_m,
        "landing_penalty": None if landing_distance_m is None else (landing_weight * camera_scale * landing_distance_m) ** 2,
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(reprojection_errors)))),
        "reprojection_median_px": float(np.median(reprojection_errors)),
        "reprojection_max_px": float(np.max(reprojection_errors)),
        "endpoint_constraint_error_m": endpoint_error,
        "start_reference_error_m": float(np.linalg.norm(path[0, :2] - start_reference_xy)),
        "receive_reference_error_m": float(np.linalg.norm(path[-1, :2] - receive_reference_xy)),
    }
    if landing_distance_m is None:
        diagnostics["reason"] = "hypothetical_ground_contact_unavailable_within_60s"
    return (path if available else None), diagnostics


def join_piecewise(first: np.ndarray, second: np.ndarray, tolerance_m: float = 0.01) -> np.ndarray:
    """Join two phases only when their shared contact position agrees."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (3,) or second.shape[1:] != (3,):
        raise ValueError("Piecewise paths must have shape (frames, 3)")
    if not len(first) or not len(second) or np.linalg.norm(first[-1] - second[0]) > tolerance_m:
        raise ValueError("Piecewise flight phases do not share a contact position")
    return np.vstack((first, second[1:]))
