"""Grid ORB, reciprocal descriptor associations and bounded forward/backward LK."""
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from .types import FeatureObservation


def match_descriptors(query, train, cfg, stats=None):
    """Return one-to-one (query index, train index, hamming distance, nearest-neighbor ratio), never descriptor IDs."""
    if stats is not None:
        stats.update(query_count=len(query) if query is not None else 0,
                     train_count=len(train) if train is not None else 0,
                     hamming_rejected=0, ratio_rejected=0, mutual_rejected=0, matches=0)
    if query is None or train is None or len(query) == 0 or len(train) < 2:
        return []
    query, train = np.asarray(query), np.asarray(train)
    if query.dtype != np.uint8 or train.dtype != np.uint8 or query.shape[1:] != (32,) or train.shape[1:] != (32,):
        raise ValueError("ORB descriptors must be uint8 (N,32)")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(query, train, k=2)
    backward = {m.queryIdx: m.trainIdx for m in matcher.match(train, query)} if cfg.mutual_matching else None
    candidates = []
    for pair in forward:
        if len(pair) != 2:
            continue
        a, b = pair
        ratio = a.distance/b.distance if b.distance > 0 else float("inf")
        reason = ("hamming_rejected" if a.distance > cfg.max_hamming_distance else
                  "ratio_rejected" if a.distance >= cfg.ratio_test*b.distance else
                  "mutual_rejected" if backward is not None and backward[a.trainIdx] != a.queryIdx else None)
        if reason is None:
            candidates.append((a, ratio))
        elif stats is not None:
            stats[reason] += 1
    used, result = set(), []
    for m, ratio in sorted(candidates, key=lambda x: x[0].distance):
        if m.trainIdx not in used:
            used.add(m.trainIdx)
            result.append((m.queryIdx, m.trainIdx, float(m.distance), float(ratio)))
    if stats is not None:
        stats["matches"] = len(result)
    return result


def group_match_descriptors(query, train, train_ids, cfg, stats=None):
    """Like match_descriptors, but train rows sharing a train_id are one candidate (its best row).

    Needed because a flat ratio test over `match_descriptors` breaks once a landmark
    contributes more than one descriptor row: the landmark's own second-best row then
    often wins the "second nearest neighbour" slot, shrinking the ratio-test gap and
    rejecting matches that would have passed with one descriptor per landmark.
    Reciprocity (`cfg.mutual_matching`) is checked landmark-vs-query, not row-vs-query.
    """
    if stats is not None:
        stats.update(query_count=len(query) if query is not None else 0,
                     train_count=len(train) if train is not None else 0,
                     hamming_rejected=0, ratio_rejected=0, mutual_rejected=0, matches=0)
    if query is None or train is None or len(query) == 0 or len(train) < 2:
        return []
    query, train, train_ids = np.asarray(query), np.asarray(train), np.asarray(train_ids)
    if query.dtype != np.uint8 or train.dtype != np.uint8 or query.shape[1:] != (32,) or train.shape[1:] != (32,):
        raise ValueError("ORB descriptors must be uint8 (N,32)")
    distances = cdist(np.unpackbits(query, axis=1), np.unpackbits(train, axis=1), "hamming")*256
    order = np.argsort(train_ids, kind="stable")
    ids, starts = np.unique(train_ids[order], return_index=True)
    if len(ids) < 2:
        return []
    group_min = np.minimum.reduceat(distances[:, order], starts, axis=1)  # (Q, G); best row per landmark
    ranked = np.argsort(group_min, axis=1)
    best, second = ranked[:, 0], ranked[:, 1]
    rows = np.arange(len(query))
    best_distance, second_distance = group_min[rows, best], group_min[rows, second]
    ratio = np.where(second_distance > 0, best_distance/np.maximum(second_distance, 1e-9), np.inf)
    # Reciprocity at the landmark level: this query must also be that landmark's closest query.
    backward_best_query = np.argmin(group_min, axis=0) if cfg.mutual_matching else None
    candidates = []
    for q, (d, r, id) in enumerate(zip(best_distance, ratio, ids[best])):
        reason = ("hamming_rejected" if d > cfg.max_hamming_distance else
                  "ratio_rejected" if d >= cfg.ratio_test*second_distance[q] else
                  "mutual_rejected" if backward_best_query is not None and backward_best_query[best[q]] != q else None)
        if reason is None:
            candidates.append((q, int(id), float(d), float(r)))
        elif stats is not None:
            stats[reason] += 1
    used, result = set(), []
    for q, id, d, r in sorted(candidates, key=lambda x: x[2]):
        if id not in used:
            used.add(id)
            result.append((q, id, d, r))
    if stats is not None:
        stats["matches"] = len(result)
    return result


def feature_mask(shape, tags, cfg):
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    for x, y, rw, rh in cfg.exclude_rectangles:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x+rw), min(h, y+rh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    if cfg.exclude_tag_regions:
        tagmask = np.zeros_like(mask)
        for tag in tags:
            if np.isfinite(tag.pixels).all():
                cv2.fillConvexPoly(tagmask, np.rint(tag.pixels).astype(np.int32), 255)
        margin = cfg.tag_mask_margin_px
        if margin:
            tagmask = cv2.dilate(tagmask, np.ones((2*margin+1, 2*margin+1), np.uint8))
        mask[tagmask != 0] = 0
    return mask


def empty_tracking_stats(enabled=True):
    return {
        "enabled": bool(enabled),
        "lk": {"input": 0, "retained": 0, "dropped": 0, "status_rejected": 0,
               "invalid_rejected": 0, "forward_backward_rejected": 0,
               "out_of_bounds_rejected": 0, "mask_rejected": 0, "call_failed": 0},
        "orb": {"ran": False, "reason": "not_due", "keypoints": 0,
                "descriptor_matches": 0, "reused_tracks": 0, "new_tracks": 0},
        "tracks_after_update": 0,
    }


class FeatureTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.orb = cv2.ORB_create(nfeatures=cfg.max_features*3, scaleFactor=cfg.scale_factor, nlevels=cfg.nlevels,
                                   edgeThreshold=cfg.edge_threshold, patchSize=cfg.patch_size, fastThreshold=cfg.fast_threshold)
        self.previous_gray = None
        self.tracks = []
        self.next_id = 0
        self.detection_ms = self.tracking_ms = 0.0
        self.last_lk_stats = empty_tracking_stats()["lk"]
        self.frame_stats = empty_tracking_stats(cfg.enabled)

    def reset(self):
        self.previous_gray = None
        self.tracks = []

    def detect(self, gray, mask):
        c = self.cfg
        h, w = gray.shape
        keys = []
        count = c.grid_rows*c.grid_cols
        base, remainder = divmod(c.max_features, count)
        # Detect once, then enforce cell quotas. Running a full ORB pyramid for
        # each grid cell made redetection frames exceed the pose-prior timeout.
        cells = [[] for _ in range(count)]
        for key in self.orb.detect(gray, mask):
            col = min(c.grid_cols-1, int(key.pt[0]*c.grid_cols/w))
            row = min(c.grid_rows-1, int(key.pt[1]*c.grid_rows/h))
            cells[row*c.grid_cols+col].append(key)
        for row in range(c.grid_rows):
            for col in range(c.grid_cols):
                quota = base + (row*c.grid_cols+col < remainder)
                keys.extend(sorted(cells[row*c.grid_cols+col], key=lambda k: k.response, reverse=True)[:quota])
        # compute returns the surviving keypoints; descriptors index THAT list.
        keys, desc = self.orb.compute(gray, keys)
        if desc is None:
            return []
        return [FeatureObservation(-1, np.array(k.pt, np.float64), d.copy()) for k, d in zip(keys, desc)]

    def flow(self, gray, mask):
        count = len(self.tracks)
        stats = {"input": count, "retained": 0, "dropped": 0, "status_rejected": 0,
                 "invalid_rejected": 0, "forward_backward_rejected": 0,
                 "out_of_bounds_rejected": 0, "mask_rejected": 0, "call_failed": 0}
        self.last_lk_stats = stats
        if self.previous_gray is None or not self.tracks:
            if count:
                stats["call_failed"] = count
                stats["dropped"] = count
            return []
        lk = self.cfg.lk
        old = np.array([f.pixel for f in self.tracks], np.float32).reshape(-1, 1, 2)
        args = dict(winSize=(lk.window_size_px, lk.window_size_px), maxLevel=lk.max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, lk.max_iterations, lk.epsilon))
        new, success, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, old, None, **args)
        if new is None or success is None:
            stats["call_failed"] = count
            stats["dropped"] = count
            return []
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, new, None, **args)
        if back is None or reverse is None:
            stats["call_failed"] = count
            stats["dropped"] = count
            return []
        errors = np.linalg.norm(back.reshape(-1, 2)-old.reshape(-1, 2), axis=1)
        h, w = gray.shape
        result = []
        for f, p, s, r, error in zip(self.tracks, new.reshape(-1, 2), success.ravel(), reverse.ravel(), errors):
            if not (s and r):
                stats["status_rejected"] += 1
            elif not np.isfinite(p).all() or not np.isfinite(error):
                stats["invalid_rejected"] += 1
            elif not (0 <= p[0] < w and 0 <= p[1] < h):
                stats["out_of_bounds_rejected"] += 1
            elif error > lk.forward_backward_max_error_px:
                stats["forward_backward_rejected"] += 1
            elif not mask[int(p[1]), int(p[0])]:
                stats["mask_rejected"] += 1
            else:
                result.append(FeatureObservation(f.track_id, p.astype(np.float64), f.descriptor, f.landmark_id,
                                                 float(error), age_frames=f.age_frames+1,
                                                 association_source="lk" if f.landmark_id is not None else None,
                                                 frames_since_confirmed=f.frames_since_confirmed+1))
        stats["retained"] = len(result)
        stats["dropped"] = count-len(result)
        return result

    def update(self, gray, tags, index, force_detect=False):
        import time
        c = self.cfg
        self.detection_ms = self.tracking_ms = 0.0
        self.frame_stats = empty_tracking_stats(c.enabled)
        if not c.enabled:
            self.reset()
            self.frame_stats["orb"]["reason"] = "features_disabled"
            return []
        mask = feature_mask(gray.shape, tags, c)
        started = time.perf_counter()
        tracks = self.flow(gray, mask) if c.lk.enabled else []
        self.tracking_ms = (time.perf_counter()-started)*1000
        self.frame_stats["lk"] = dict(self.last_lk_stats) if c.lk.enabled else empty_tracking_stats()["lk"]
        reasons = []
        if force_detect:
            reasons.append("forced")
        if not c.lk.enabled:
            reasons.append("lk_disabled")
        if index % c.redetect_every_n_frames == 0:
            reasons.append("interval")
        if len(tracks) < c.redetect_below_tracks:
            reasons.append("low_tracks")
        if reasons:
            orb_stats = self.frame_stats["orb"]
            orb_stats.update(ran=True, reason="+".join(reasons))
            started = time.perf_counter()
            detected = self.detect(gray, mask)
            orb_stats["detect_compute_ms"] = (time.perf_counter()-started)*1000
            orb_stats["keypoints"] = len(detected)
            previous = tracks if c.lk.enabled else self.tracks
            matching_started = time.perf_counter()
            if previous and detected:
                pairs = match_descriptors(np.array([f.descriptor for f in detected]), np.array([f.descriptor for f in previous]), c)
            else:
                pairs = []
            orb_stats["descriptor_matches"] = len(pairs)
            orb_stats["descriptor_matching_ms"] = (time.perf_counter()-matching_started)*1000
            associations = {q: t for q, t, _, _ in pairs}
            reused = set()
            new_ids = set()
            for i, f in enumerate(detected):
                if i in associations:
                    old = previous[associations[i]]
                    # Descriptor identity plus spatial agreement for LK; descriptors alone when LK off.
                    if not c.lk.enabled or np.linalg.norm(old.pixel-f.pixel) <= c.redetection_match_radius_px:
                        f.track_id, f.landmark_id = old.track_id, old.landmark_id
                        f.age_frames = old.age_frames if c.lk.enabled else old.age_frames+1
                        f.frames_since_confirmed = old.frames_since_confirmed
                        f.association_source = "orb_redetection" if old.landmark_id is not None else None
                        reused.add(old.track_id)
                if f.track_id < 0:
                    f.track_id = self.next_id
                    self.next_id += 1
                    new_ids.add(f.track_id)
            orb_stats["reused_tracks"] = len(reused)
            # Keep surviving mapped tracks preferentially; suppress redetection duplicates.
            retained = [f for f in tracks if f.track_id not in reused]
            duplicate_started = time.perf_counter()
            if retained and detected:
                nearest_sq = cdist(np.array([f.pixel for f in detected]),
                                   np.array([f.pixel for f in retained]), "sqeuclidean").min(axis=1)
                detected = [f for f, distance_sq in zip(detected, nearest_sq)
                            if f.track_id in reused or distance_sq > c.redetection_match_radius_px**2]
            orb_stats["duplicate_filter_ms"] = (time.perf_counter()-duplicate_started)*1000
            tracks = sorted(retained + detected, key=lambda f: f.landmark_id is None)[:c.max_features]
            orb_stats["new_tracks"] = sum(f.track_id in new_ids for f in tracks)
            self.detection_ms = (time.perf_counter()-started)*1000
        self.tracks = tracks
        self.previous_gray = gray.copy()
        self.frame_stats["tracks_after_update"] = len(tracks)
        self.frame_stats["track_age_median_frames"] = float(np.median([f.age_frames for f in tracks])) if tracks else 0.0
        self.frame_stats["track_age_max_frames"] = max((f.age_frames for f in tracks), default=0)
        return tracks
