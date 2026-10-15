"""Small tag-assisted bootstrap, sparse one-time refinement, frozen-map persistence."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .features import match_descriptors
from .geometry import camera_position, depths, project, retract, triangulate, valid_transform
from .types import FeatureObservation, LandmarkMap, TagObservation


def map_metadata(config, camera):
    return {"schema_version": 1, "units": "metres", "world_frame": "anchor: X right, Y up, Z printed front", "transform": "T_CW",
            "anchor_id": config.tags.anchor_id, "tag_family": config.tags.family, "tag_size_m": config.tags.size_m,
            "feature_backend": config.features.backend, "camera": camera.identity}


def validate_map(m, expected, config):
    n = len(m.ids)
    if m.metadata != expected:
        raise ValueError("mapping.map_path: incompatible schema/anchor/calibration/preprocessing metadata")
    if (m.points.shape != (n, 3) or m.points.dtype != np.float64 or m.descriptors.shape != (n, 32) or m.descriptors.dtype != np.uint8
            or m.ids.shape != (n,) or m.ids.dtype != np.int64 or len(np.unique(m.ids)) != n or np.any(m.ids < 0)
            or m.support.shape != (n,) or m.support.dtype != np.int64 or m.rms.shape != (n,) or m.rms.dtype != np.float64):
        raise ValueError("map: invalid dimensions, dtypes or IDs")
    if (not np.isfinite(m.points).all() or not np.isfinite(m.rms).all() or np.any(m.rms < 0)
            or np.any(m.rms > config.mapping.max_reprojection_rms_px) or np.any(m.support < config.mapping.min_confirming_views)
            or np.any(m.support > config.mapping.max_keyframes) or not config.mapping.min_ready_landmarks <= n <= config.mapping.max_landmarks):
        raise ValueError("map: invalid finite points, support, quality or capacity")


def save_map(filename, m, replace=False):
    """Atomic complete write; hard-link publication protects existing destinations."""
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".map-", suffix=".npz", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, points=m.points, descriptors=m.descriptors, ids=m.ids, support=m.support, rms=m.rms,
                                metadata=np.array(json.dumps(m.metadata, allow_nan=False)))
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(name, target)
        else:
            os.link(name, target)  # fails if destination exists, including concurrent writers
    finally:
        Path(name).unlink(missing_ok=True)


def load_map(filename, expected, config):
    try:
        with np.load(filename, allow_pickle=False) as data:
            if set(data.files) != {"points", "descriptors", "ids", "support", "rms", "metadata"}:
                raise ValueError("map schema keys differ")
            meta = data["metadata"]
            if meta.shape != () or meta.dtype.kind != "U":
                raise ValueError("metadata must be a Unicode JSON scalar")
            metadata = json.loads(str(meta), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            m = LandmarkMap(*(data[k].copy() for k in ("points", "descriptors", "ids", "support", "rms")), metadata)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"mapping.map_path: cannot load {filename}: {exc}") from exc
    validate_map(m, expected, config)
    return m


def associate(features, m, cfg):
    """Preserve LK landmark IDs; use descriptors for new observations/recovery."""
    index = {int(id): i for i, id in enumerate(m.ids)}
    used = set()
    for f in features:
        if f.landmark_id not in index or f.landmark_id in used:
            f.landmark_id = None
        else:
            used.add(f.landmark_id)
    fresh = [f for f in features if f.landmark_id is None and f.descriptor is not None]
    if fresh:
        for q, t in match_descriptors(np.array([f.descriptor for f in fresh]), m.descriptors, cfg):
            id = int(m.ids[t])
            if id not in used:
                fresh[q].landmark_id = id
                used.add(id)
    selected = [f for f in features if f.landmark_id is not None]
    return (np.array([m.points[index[f.landmark_id]] for f in selected], np.float64).reshape(-1, 3),
            np.array([f.pixel for f in selected], np.float64).reshape(-1, 2))


@dataclass
class Keyframe:
    T: np.ndarray
    tag: TagObservation
    features: dict[int, FeatureObservation]


class MapBuilder:
    def __init__(self, config, camera):
        self.config, self.camera = config, camera
        self.keyframes = []
        self.map = None
        self.reason = "waiting for unambiguous anchor and translated views"
        self.candidate_count = 0

    def add(self, T, tag, features):
        c = self.config.mapping
        if self.map is not None:
            return
        if not tag.valid or tag.ambiguous or not valid_transform(T):
            self.reason = "anchor absent/ambiguous; bootstrap keyframe rejected"
            return
        if (tag.id != self.config.tags.anchor_id or not np.isfinite(tag.pixels).all()
                or np.any(depths(T, tag.world) <= 0)
                or np.sqrt(np.mean(np.sum((project(T, tag.world, self.camera)-tag.pixels)**2, axis=1))) > self.config.tags.max_reprojection_rms_px):
            self.reason = "anchor geometry failed bootstrap pose validation"
            return
        if len(self.keyframes) >= c.max_keyframes:
            self.reason = "keyframe capacity reached before map became ready; press R to reset"
            return
        if any(np.linalg.norm(camera_position(T)-camera_position(k.T)) < c.min_baseline_m for k in self.keyframes):
            self.reason = "insufficient new translational baseline"
            return
        if len(features) > self.config.features.max_features or len({f.track_id for f in features}) != len(features):
            raise ValueError("bootstrap features exceed capacity or have duplicate track IDs")
        current = {f.track_id: FeatureObservation(f.track_id, f.pixel.copy(), f.descriptor.copy()) for f in features if f.valid and f.descriptor is not None}
        # Bounded descriptor matching recovers identities after redetection/temporary track loss.
        canonical = {}
        for k in self.keyframes:
            canonical.update(k.features)
        if current and canonical:
            old = list(canonical.values())
            new = list(current.values())
            identities = {q: old[t].track_id for q, t in match_descriptors(np.array([f.descriptor for f in new]), np.array([f.descriptor for f in old]), self.config.features)}
            merged = {}
            for q, f in enumerate(new):
                id = f.track_id if f.track_id in canonical else identities.get(q, f.track_id)
                if id not in merged:
                    merged[id] = FeatureObservation(id, f.pixel, f.descriptor)
            current = merged
        self.keyframes.append(Keyframe(T.copy(), tag, current))
        self._build()

    def _build(self):
        c = self.config.mapping
        observations = {}
        for j, k in enumerate(self.keyframes):
            for id, f in k.features.items():
                observations.setdefault(id, []).append((j, f))
        points, ids, views = [], [], []
        for id, obs in observations.items():
            if len(obs) < c.min_confirming_views:
                continue
            found = None
            for a in range(len(obs)-1):
                for b in range(a+1, len(obs)):
                    ja, fa = obs[a]
                    jb, fb = obs[b]
                    found = triangulate(self.keyframes[ja].T, self.keyframes[jb].T, fa.pixel, fb.pixel, self.camera, c)
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is None:
                continue
            good = self._quality(found, obs)
            if good is None:
                continue
            points.append(found)
            ids.append(id)
            views.append(obs)
            if len(points) == c.max_landmarks:
                break
        self.candidate_count = len(points)
        self.reason = f"bootstrap: {len(self.keyframes)}/{c.max_keyframes} keyframes; {len(points)}/{c.min_ready_landmarks} stable landmarks"
        if len(points) < c.min_ready_landmarks:
            if len(self.keyframes) == c.max_keyframes:
                self.reason += "; capacity exhausted, reset required"
            return
        points = np.array(points)
        if c.refine_bootstrap_map:
            refined = self._refine(points, views)
            if refined is None:
                self.reason = "bounded map refinement failed; will retry after next keyframe"
                return
            points = refined
        keep = [(i, self._quality(p, obs)) for i, (p, obs) in enumerate(zip(points, views))]
        keep = [(i, rms) for i, rms in keep if rms is not None]
        self.candidate_count = len(keep)
        if len(keep) < c.min_ready_landmarks:
            self.reason = "map refinement failed final support/depth/reprojection checks; will retry after next keyframe"
            return
        self.map = LandmarkMap(np.array([points[i] for i, _ in keep], np.float64),
                               np.array([views[i][0][1].descriptor for i, _ in keep], np.uint8),
                               np.array([ids[i] for i, _ in keep], np.int64),
                               np.array([len(views[i]) for i, _ in keep], np.int64),
                               np.array([rms for _, rms in keep], np.float64), map_metadata(self.config, self.camera))
        validate_map(self.map, map_metadata(self.config, self.camera), self.config)
        self.reason = "map ready and frozen; explicit save required"

    def _quality(self, point, obs):
        c = self.config.mapping
        errors = []
        directions = []
        for j, f in obs:
            T = self.keyframes[j].T
            d = depths(T, point[None])[0]
            if not c.min_depth_m <= d <= c.max_depth_m:
                return None
            errors.append(project(T, point[None], self.camera)[0]-f.pixel)
            direction = point-camera_position(T)
            directions.append(direction/np.linalg.norm(direction))
        angles = np.rad2deg(np.arccos(np.clip(np.array(directions)@np.array(directions).T, -1, 1)))
        if np.max(angles) < c.min_parallax_deg:
            return None
        rms = float(np.sqrt(np.mean(np.sum(np.array(errors)**2, axis=1))))
        return rms if np.isfinite(rms) and rms <= c.max_reprojection_rms_px else None

    def _refine(self, points, views):
        """Joint BA of bounded keyframe poses/points; tag fixed at known W removes gauge."""
        K, N = len(self.keyframes), len(points)
        observations = [(i, j, f.pixel) for i, obs in enumerate(views) for j, f in obs]
        rows = 8*K + 2*len(observations)
        sparsity = lil_matrix((rows, 6*K+3*N), dtype=np.int8)
        for j in range(K):
            sparsity[8*j:8*j+8, 6*j:6*j+6] = 1
        for n, (i, j, _) in enumerate(observations):
            r = 8*K+2*n
            sparsity[r:r+2, 6*j:6*j+6] = 1
            sparsity[r:r+2, 6*K+3*i:6*K+3*i+3] = 1
        def decode(x):
            Ts = [retract(k.T, x[6*j:6*j+6]) for j, k in enumerate(self.keyframes)]
            return Ts, points + x[6*K:].reshape(N, 3)
        def residual(x):
            Ts, Ps = decode(x)
            tag_errors = [(project(T, k.tag.world, self.camera)-k.tag.pixels).ravel()/self.config.tags.corner_sigma_px for T, k in zip(Ts, self.keyframes)]
            feature_errors = [(project(Ts[j], Ps[i:i+1], self.camera)[0]-u)/self.config.features.observation_sigma_px for i, j, u in observations]
            return np.concatenate(tag_errors+feature_errors)
        fit = least_squares(residual, np.zeros(6*K+3*N), method="trf", jac="3-point", jac_sparsity=sparsity.tocsr(),
                            tr_solver="lsmr", x_scale=1.0, loss=self.config.pose.robust_loss, f_scale=self.config.pose.robust_f_scale,
                            max_nfev=self.config.mapping.refinement_max_nfev)
        if not fit.success or not np.isfinite(fit.x).all():
            return None
        Ts, Ps = decode(fit.x)
        for T, k in zip(Ts, self.keyframes):
            rms = np.sqrt(np.mean(np.sum((project(T, k.tag.world, self.camera)-k.tag.pixels)**2, axis=1)))
            if np.any(depths(T, k.tag.world) <= 0) or rms > self.config.tags.max_reprojection_rms_px:
                return None
        for T, k in zip(Ts, self.keyframes):
            k.T = T
        return Ps
