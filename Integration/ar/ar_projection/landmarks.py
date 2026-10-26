"""Small tag-assisted bootstrap, sparse one-time refinement, frozen-map persistence."""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import logging
import os
import tempfile
import threading
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .features import group_match_descriptors, match_descriptors
from .geometry import camera_position, depths, project, retract, triangulate, valid_transform
from .types import FeatureObservation, LandmarkMap, TagObservation


def map_metadata(config, camera):
    return {"schema_version": 1, "units": "metres", "world_frame": "anchor: X right, Y up, Z printed front", "transform": "T_CW",
            "anchor_id": config.tags.anchor_id, "tag_family": config.tags.family, "tag_size_m": config.tags.size_m,
            "feature_backend": config.features.backend, "camera": camera.identity}


def validate_map(m, expected, config):
    n = len(m.ids)
    if m.metadata != expected:
        differing = [key for key in set(m.metadata) | set(expected) if m.metadata.get(key) != expected.get(key)]
        raise ValueError("mapping.map_path: incompatible schema/anchor/calibration/preprocessing metadata: "
                         + ", ".join(sorted(differing)) + "; build a separate map for this camera and measured tag size")
    if (m.points.shape != (n, 3) or m.points.dtype != np.float64 or m.descriptors.shape != (n, 32) or m.descriptors.dtype != np.uint8
            or m.ids.shape != (n,) or m.ids.dtype != np.int64 or len(np.unique(m.ids)) != n or np.any(m.ids < 0)
            or m.support.shape != (n,) or m.support.dtype != np.int64 or m.rms.shape != (n,) or m.rms.dtype != np.float64):
        raise ValueError("map: invalid dimensions, dtypes or IDs")
    if (not np.isfinite(m.points).all() or not np.isfinite(m.rms).all() or np.any(m.rms < 0)
            or np.any(m.rms > config.mapping.max_reprojection_rms_px) or np.any(m.support < config.mapping.min_confirming_views)
            or np.any(m.support > config.mapping.max_keyframes) or not config.mapping.min_ready_landmarks <= n <= config.mapping.max_landmarks):
        raise ValueError("map: invalid finite points, support, quality or capacity")
    if (m.extra_descriptors is None) != (m.extra_descriptor_counts is None):
        raise ValueError("map: extra_descriptors and extra_descriptor_counts must be both present or both absent")
    if m.extra_descriptors is not None:
        if (m.extra_descriptors.ndim != 3 or m.extra_descriptors.shape[0] != n or m.extra_descriptors.shape[2] != 32
                or m.extra_descriptors.dtype != np.uint8 or m.extra_descriptor_counts.shape != (n,)
                or m.extra_descriptor_counts.dtype != np.int64 or np.any(m.extra_descriptor_counts < 0)
                or np.any(m.extra_descriptor_counts > m.extra_descriptors.shape[1])):
            raise ValueError("map: invalid multi-view descriptor bank dimensions/dtypes")


def save_map(filename, m, replace=False):
    """Atomic complete write; hard-link publication protects existing destinations."""
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".map-", suffix=".npz", dir=target.parent)
    try:
        arrays = dict(points=m.points, descriptors=m.descriptors, ids=m.ids, support=m.support, rms=m.rms,
                      metadata=np.array(json.dumps(m.metadata, allow_nan=False)))
        if m.extra_descriptors is not None:
            arrays.update(extra_descriptors=m.extra_descriptors, extra_descriptor_counts=m.extra_descriptor_counts)
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(name, target)
        else:
            os.link(name, target)  # fails if destination exists, including concurrent writers
    finally:
        Path(name).unlink(missing_ok=True)


def load_map(filename, expected, config):
    base_keys = {"points", "descriptors", "ids", "support", "rms", "metadata"}
    extra_keys = base_keys | {"extra_descriptors", "extra_descriptor_counts"}
    try:
        with np.load(filename, allow_pickle=False) as data:
            if set(data.files) not in (base_keys, extra_keys):
                raise ValueError("map schema keys differ")
            meta = data["metadata"]
            if meta.shape != () or meta.dtype.kind != "U":
                raise ValueError("metadata must be a Unicode JSON scalar")
            metadata = json.loads(str(meta), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            m = LandmarkMap(*(data[k].copy() for k in ("points", "descriptors", "ids", "support", "rms")), metadata,
                            extra_descriptors=data["extra_descriptors"].copy() if "extra_descriptors" in data.files else None,
                            extra_descriptor_counts=data["extra_descriptor_counts"].copy() if "extra_descriptor_counts" in data.files else None)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"mapping.map_path: cannot load {filename}: {exc}") from exc
    validate_map(m, expected, config)
    return m


def revalidate_ids(features, m, cfg, lost):
    """Clear suspect inherited landmark_id values so associate() can re-match them fresh this frame.

    Only touches ids carried forward by LK/redetection (never a freshly-matched orb_map id).
    Returns the number cleared; cfg.mode == "off" clears nothing (default, zero behavior change).
    """
    cleared = 0
    inherited_sources = ("lk", "orb_redetection")
    if cfg.mode == "off":
        pass
    elif cfg.mode == "loss_clear":
        if lost:
            for f in features:
                if f.landmark_id is not None and f.association_source in inherited_sources:
                    f.landmark_id = None
                    cleared += 1
    elif cfg.mode == "age_expire":
        for f in features:
            if f.landmark_id is not None and f.association_source in inherited_sources and f.frames_since_confirmed > cfg.max_age_frames:
                f.landmark_id = None
                cleared += 1
    elif cfg.mode == "descriptor_check":
        index = {int(id): i for i, id in enumerate(m.ids)} if m is not None else {}
        for f in features:
            if (f.landmark_id is not None and f.association_source in inherited_sources
                    and f.landmark_id in index and f.descriptor is not None):
                distance = int(np.unpackbits(np.bitwise_xor(f.descriptor, m.descriptors[index[f.landmark_id]])).sum())
                if distance > cfg.max_descriptor_distance:
                    f.landmark_id = None
                    cleared += 1
    else:
        raise ValueError(f"features.id_revalidation.mode: unsupported {cfg.mode}")
    return cleared


def guided_matches(fresh, m, prior_T_CW, camera, cfg, used):
    """Match landmark_id=None features to map landmarks near their prior-pose projection.

    For each unclaimed map landmark, keep fresh features within cfg.search_radius_px of its
    projection under the prior pose; accept the closest by Hamming distance if it clears the
    distance limit and (when more than one candidate falls in the window) the ratio test against
    the window's second-best. One-to-one per feature and per landmark. Returns a list of
    (fresh_index, landmark_id, distance, ratio) like match_descriptors/group_match_descriptors.
    """
    if not fresh or prior_T_CW is None:
        return []
    pixels = np.array([f.pixel for f in fresh])
    w, h = camera.size
    used_features = set()
    matches = []
    for i, (id, point) in enumerate(zip(m.ids, m.points)):
        id = int(id)
        if id in used:
            continue
        proj = project(prior_T_CW, point[None], camera)[0]
        if not (0 <= proj[0] < w and 0 <= proj[1] < h):
            continue
        d = np.linalg.norm(pixels-proj, axis=1)
        within = [k for k in np.flatnonzero(d <= cfg.search_radius_px) if k not in used_features]
        if not within:
            continue
        distances = [float(cv2.norm(fresh[k].descriptor, m.descriptors[i], cv2.NORM_HAMMING)) for k in within]
        if cfg.geometry_first:
            # ponytail: ties = within 1 px of the nearest; outlier = 32 bits worse than the window median (needs 3+ candidates). Constants, not config, until a sweep says otherwise.
            pixel = d[within]
            tied = [j for j in range(len(within)) if pixel[j] <= pixel.min()+1.0]
            j = min(tied, key=lambda t: distances[t])
            if distances[j] > cfg.max_hamming_distance or (len(within) >= 3 and distances[j] > np.median(distances)+32):
                continue
            others = np.delete(pixel, j)
            if len(others) and others.min() < cfg.geometry_margin*pixel[j]:  # ambiguous: a rival is nearly as close (ponytail: pixel margin only, no descriptor tie-break)
                continue
            matches.append((within[j], id, distances[j], None))
            used_features.add(within[j])
            continue
        order = np.argsort(distances)
        best_distance = distances[order[0]]
        if best_distance > cfg.max_hamming_distance:
            continue
        ratio = None
        if len(order) > 1:
            second_distance = distances[order[1]]
            ratio = best_distance/second_distance if second_distance > 0 else float("inf")
            if best_distance >= cfg.ratio_test*second_distance:
                continue
        best_k = within[order[0]]
        matches.append((best_k, id, best_distance, ratio))
        used_features.add(best_k)
    return matches


def associate(features, m, cfg, stats=None, prior_T_CW=None, camera=None):
    """Preserve LK landmark IDs; guided matching near a prior pose when available, else global descriptors."""
    index = {int(id): i for i, id in enumerate(m.ids)}
    used = set()
    g = cfg.guided_matching
    cleared = 0
    if g.enabled and g.revalidate_existing and prior_T_CW is not None and camera is not None:
        for f in features:  # only the tracker's own prior pose, never the tag reference
            if f.landmark_id in index and f.association_source in ("lk", "orb_redetection") and f.descriptor is not None                     and np.linalg.norm(project(prior_T_CW, m.points[index[f.landmark_id]][None], camera)[0]-f.pixel) > g.search_radius_px:
                f.landmark_id = None
                cleared += 1
    for f in features:
        if f.landmark_id not in index or f.landmark_id in used:
            f.landmark_id = None
        else:
            used.add(f.landmark_id)
    fresh = [f for f in features if f.landmark_id is None and f.descriptor is not None]
    inherited = len(used)
    guided_count = 0
    if cfg.guided_matching.enabled and prior_T_CW is not None and camera is not None:
        for k, id, distance, ratio in guided_matches(fresh, m, prior_T_CW, camera, cfg.guided_matching, used):
            fresh[k].landmark_id = id
            fresh[k].association_source = "guided"
            fresh[k].match_distance = distance
            fresh[k].match_ratio = ratio
            used.add(id)
            guided_count += 1
        fresh = [f for f in fresh if f.landmark_id is None]
    matching = {}
    if fresh and m.extra_descriptors is not None:
        rows = [m.extra_descriptors[i, k] for i, count in enumerate(m.extra_descriptor_counts) for k in range(int(count))]
        ids = [m.ids[i] for i, count in enumerate(m.extra_descriptor_counts) for k in range(int(count))]
        train_descriptors = np.concatenate([m.descriptors, np.array(rows, np.uint8)]) if rows else m.descriptors
        train_ids = np.concatenate([m.ids, np.array(ids, np.int64)]) if rows else m.ids
        matches = group_match_descriptors(np.array([f.descriptor for f in fresh]), train_descriptors, train_ids, cfg, matching)
    elif fresh:
        matches = [(q, int(m.ids[t]), distance, ratio)
                  for q, t, distance, ratio in match_descriptors(np.array([f.descriptor for f in fresh]), m.descriptors, cfg, matching)]
    else:
        matches = []
    for q, id, distance, ratio in matches:
        if id not in used:
            fresh[q].landmark_id = id
            fresh[q].association_source = "orb_map"
            fresh[q].match_distance = distance
            fresh[q].match_ratio = ratio
            used.add(id)
    selected = [f for f in features if f.landmark_id is not None]
    if stats is not None:
        stats.update(inherited_lk_associations=sum(f.association_source == "lk" for f in selected),
                     redetected_orb_associations=sum(f.association_source == "orb_redetection" for f in selected),
                     guided_associations=guided_count, ids_cleared_by_guided_revalidation=cleared,
                     carried_associations=inherited, fresh_descriptor_associations=len(selected)-inherited,
                     mapped_observations=len(selected), descriptor_matching=matching)
    return (np.array([m.points[index[f.landmark_id]] for f in selected], np.float64).reshape(-1, 3),
            np.array([f.pixel for f in selected], np.float64).reshape(-1, 2))


@dataclass
class Keyframe:
    T: np.ndarray
    tag: TagObservation
    features: dict[int, FeatureObservation]


class _RefinementStopped(RuntimeError):
    pass


class MapBuilder:
    def __init__(self, config, camera, background=False):
        self.config, self.camera = config, camera
        self.keyframes = []
        self.map = None
        self.reason = "waiting for unambiguous anchor and translated views"
        self.candidate_count = 0
        self.refinement_stats = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-bootstrap") if background else None
        self._future = self._job = None
        self._submitted_keyframes = 0
        self._retry_at = 0.
        self._cancel = threading.Event()
        self._closed = False

    @property
    def pending(self):
        return not self._closed and self.map is None and self._executor is not None and (
            self._future is not None or len(self.keyframes) > self._submitted_keyframes)

    def poll(self):
        """Publish only completed, validated snapshots; never wait on the preview thread."""
        if self._closed or self._executor is None or self.map is not None:
            return
        if self._future is not None:
            self.candidate_count = self._job.candidate_count
            self.refinement_stats = dict(self._job.refinement_stats)
            if not self._future.done():
                self.reason = self._job.reason + " (background)"
                return
            try:
                self._future.result()
                self.reason = self._job.reason
                self.candidate_count = self._job.candidate_count
                self.refinement_stats = dict(self._job.refinement_stats)
                if self._job.map is not None:
                    self.map = self._job.map
                    for original, refined in zip(self.keyframes, self._job.keyframes):
                        original.T = refined.T
                if self.refinement_stats.get("outcome") not in (None, "accepted"):
                    self._retry_at = time.monotonic()+self.config.mapping.refinement_retry_interval_s
            except Exception as exc:
                self.reason = f"map worker failed: {type(exc).__name__}: {exc}"
                self.refinement_stats = {"outcome": "worker_error", "message": self.reason}
                self._retry_at = time.monotonic()+self.config.mapping.refinement_retry_interval_s
                logging.warning(self.reason)
            finally:
                self._future = self._job = None
        if self.map is None and len(self.keyframes) > self._submitted_keyframes and time.monotonic() >= self._retry_at:
            job = MapBuilder(self.config, self.camera)
            # Observations are private copies made by add() and are never modified.
            # Each worker owns its pose matrices and keyframe dictionaries.
            job.keyframes = [Keyframe(k.T.copy(), k.tag, dict(k.features)) for k in self.keyframes]
            job._cancel = self._cancel
            self._submitted_keyframes = len(job.keyframes)
            self._job = job
            self._future = self._executor.submit(job._build)
            self.reason = f"processing {self._submitted_keyframes} keyframes in background"

    def worker_stats(self):
        return {"background": self._executor is not None, "running": self._future is not None,
                "pending": self.pending, "snapshot_keyframes": self._submitted_keyframes,
                "queued_keyframes": max(0, len(self.keyframes)-self._submitted_keyframes) if self._executor else 0,
                "retry_in_s": max(0., self._retry_at-time.monotonic()) if self.pending else 0.}

    def close(self):
        self._closed = True
        self._cancel.set()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _check_stop(self, evaluations=None):
        if self._cancel.is_set():
            raise _RefinementStopped("cancelled")
        if evaluations is not None and evaluations >= self.config.mapping.refinement_max_evaluations:
            raise _RefinementStopped("evaluation_budget_exceeded")

    def add(self, T, tag, features):
        c = self.config.mapping
        if self.map is not None or self._closed:
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
            identities = {q: old[t].track_id for q, t, _, _ in match_descriptors(np.array([f.descriptor for f in new]), np.array([f.descriptor for f in old]), self.config.features)}
            merged = {}
            for q, f in enumerate(new):
                id = f.track_id if f.track_id in canonical else identities.get(q, f.track_id)
                if id not in merged:
                    merged[id] = FeatureObservation(id, f.pixel, f.descriptor)
            current = merged
        self.keyframes.append(Keyframe(T.copy(), tag, current))
        if self._executor is None:
            self._build()
        else:
            self.poll()

    def _build(self):
        c = self.config.mapping
        observations = {}
        for j, k in enumerate(self.keyframes):
            for id, f in k.features.items():
                observations.setdefault(id, []).append((j, f))
        points, ids, views = [], [], []
        for id, obs in observations.items():
            self._check_stop()
            if len(obs) < c.min_confirming_views:
                continue
            found = None
            for a in range(len(obs)-1):
                for b in range(a+1, len(obs)):
                    self._check_stop()
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
                outcome = self.refinement_stats.get("outcome", "failed")
                self.reason = f"map refinement {outcome}; will retry after next keyframe"
                return
            points = refined
        keep = [(i, self._quality(p, obs)) for i, (p, obs) in enumerate(zip(points, views))]
        keep = [(i, rms) for i, rms in keep if rms is not None]
        self.candidate_count = len(keep)
        if len(keep) < c.min_ready_landmarks:
            self.refinement_stats = {**self.refinement_stats, "outcome": "final_quality_failed", "final_landmarks": len(keep)}
            self.reason = "map refinement failed final support/depth/reprojection checks; will retry after next keyframe"
            return
        extra_descriptors = extra_counts = None
        bank = c.multiview_descriptor_bank
        if bank:
            extra_descriptors = np.zeros((len(keep), bank, 32), np.uint8)
            extra_counts = np.zeros(len(keep), np.int64)
            for row, (i, _) in enumerate(keep):
                chosen, count = self._descriptor_bank(points[i], views[i], bank)
                extra_descriptors[row, :count] = chosen
                extra_counts[row] = count
        self.map = LandmarkMap(np.array([points[i] for i, _ in keep], np.float64),
                               np.array([views[i][0][1].descriptor for i, _ in keep], np.uint8),
                               np.array([ids[i] for i, _ in keep], np.int64),
                               np.array([len(views[i]) for i, _ in keep], np.int64),
                               np.array([rms for _, rms in keep], np.float64), map_metadata(self.config, self.camera),
                               extra_descriptors, extra_counts)
        validate_map(self.map, map_metadata(self.config, self.camera), self.config)
        self.reason = "map ready and frozen; explicit save required"

    def _descriptor_bank(self, point, obs, bank):
        """Up to `bank` extra descriptors from the views angularly farthest from the primary (first-confirming) view."""
        primary_T = self.keyframes[obs[0][0]].T
        primary_dir = point-camera_position(primary_T)
        primary_dir = primary_dir/np.linalg.norm(primary_dir)
        others = []
        for j, f in obs[1:]:
            direction = point-camera_position(self.keyframes[j].T)
            direction = direction/np.linalg.norm(direction)
            angle = float(np.arccos(np.clip(primary_dir@direction, -1, 1)))
            others.append((angle, f.descriptor))
        others.sort(key=lambda x: -x[0])
        chosen = [descriptor for _, descriptor in others[:bank]]
        return (np.array(chosen, np.uint8) if chosen else np.empty((0, 32), np.uint8)), len(chosen)

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
        started = time.perf_counter()
        K, N = len(self.keyframes), len(points)
        observations = [(i, j, f.pixel) for i, obs in enumerate(views) for j, f in obs]
        point_index = np.array([i for i, _, _ in observations], dtype=np.intp)
        camera_index = np.array([j for _, j, _ in observations], dtype=np.intp)
        pixels = np.array([u for _, _, u in observations])
        groups = [np.flatnonzero(camera_index == j) for j in range(K)]
        self.refinement_stats = {"outcome": "running", "keyframes": K, "landmarks": N,
                                 "feature_observations": len(observations), "max_evaluations": self.config.mapping.refinement_max_evaluations}
        self.reason = f"refining {N} landmarks across {K} keyframes"
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
        evaluations = 0
        def residual(x):
            nonlocal evaluations
            self._check_stop(evaluations)
            evaluations += 1
            Ts, Ps = decode(x)
            errors = np.empty(rows)
            feature_errors = errors[8*K:].reshape(-1, 2)
            for j, (T, k, indices) in enumerate(zip(Ts, self.keyframes, groups)):
                errors[8*j:8*j+8] = ((project(T, k.tag.world, self.camera)-k.tag.pixels)/self.config.tags.corner_sigma_px).ravel()
                if len(indices):
                    feature_errors[indices] = ((project(T, Ps[point_index[indices]], self.camera)-pixels[indices])
                                                /self.config.features.observation_sigma_px)
            return errors
        try:
            fit = least_squares(residual, np.zeros(6*K+3*N), method="trf", jac="3-point", jac_sparsity=sparsity.tocsr(),
                                tr_solver="lsmr", x_scale="jac", loss=self.config.pose.robust_loss, f_scale=self.config.pose.robust_f_scale,
                                max_nfev=self.config.mapping.refinement_max_nfev)
        except _RefinementStopped as exc:
            self.refinement_stats = {**self.refinement_stats, "outcome": str(exc),
                                     "residual_evaluations": evaluations, "elapsed_ms": (time.perf_counter()-started)*1000}
            return None
        self.refinement_stats = {**self.refinement_stats, "outcome": "solver_failed", "optimizer_status": int(fit.status),
                                 "message": str(fit.message), "nfev": int(fit.nfev), "njev": int(fit.njev),
                                 "residual_evaluations": evaluations, "elapsed_ms": (time.perf_counter()-started)*1000,
                                 "cost": float(fit.cost) if np.isfinite(fit.cost) else None,
                                 "optimality": float(fit.optimality) if np.isfinite(fit.optimality) else None}
        if not fit.success or not np.isfinite(fit.x).all():
            return None
        Ts, Ps = decode(fit.x)
        for T, k in zip(Ts, self.keyframes):
            rms = np.sqrt(np.mean(np.sum((project(T, k.tag.world, self.camera)-k.tag.pixels)**2, axis=1)))
            if np.any(depths(T, k.tag.world) <= 0) or rms > self.config.tags.max_reprojection_rms_px:
                self.refinement_stats = {**self.refinement_stats, "outcome": "tag_geometry_failed"}
                return None
        for T, k in zip(Ts, self.keyframes):
            k.T = T
        self.refinement_stats = {**self.refinement_stats, "outcome": "accepted"}
        return Ps
