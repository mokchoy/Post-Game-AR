"""Grid ORB, reciprocal descriptor associations and bounded forward/backward LK."""
import cv2
import numpy as np
from .types import FeatureObservation


def match_descriptors(query, train, cfg):
    """Return one-to-one (query index, train index) pairs, never descriptor IDs."""
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
        if a.distance <= cfg.max_hamming_distance and a.distance < cfg.ratio_test*b.distance and (backward is None or backward[a.trainIdx] == a.queryIdx):
            candidates.append(a)
    used, result = set(), []
    for m in sorted(candidates, key=lambda x: x.distance):
        if m.trainIdx not in used:
            used.add(m.trainIdx)
            result.append((m.queryIdx, m.trainIdx))
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


class FeatureTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.orb = cv2.ORB_create(nfeatures=cfg.max_features, scaleFactor=cfg.scale_factor, nlevels=cfg.nlevels,
                                   edgeThreshold=cfg.edge_threshold, patchSize=cfg.patch_size, fastThreshold=cfg.fast_threshold)
        self.previous_gray = None
        self.tracks = []
        self.next_id = 0
        self.detection_ms = self.tracking_ms = 0.0

    def reset(self):
        self.previous_gray = None
        self.tracks = []

    def detect(self, gray, mask):
        c = self.cfg
        h, w = gray.shape
        keys = []
        count = c.grid_rows*c.grid_cols
        base, remainder = divmod(c.max_features, count)
        for row in range(c.grid_rows):
            for col in range(c.grid_cols):
                cell = np.zeros_like(mask)
                y0, y1 = row*h//c.grid_rows, (row+1)*h//c.grid_rows
                x0, x1 = col*w//c.grid_cols, (col+1)*w//c.grid_cols
                cell[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
                quota = base + (row*c.grid_cols+col < remainder)
                keys.extend(sorted(self.orb.detect(gray, cell), key=lambda k: k.response, reverse=True)[:quota])
        # compute returns the surviving keypoints; descriptors index THAT list.
        keys, desc = self.orb.compute(gray, keys)
        if desc is None:
            return []
        return [FeatureObservation(-1, np.array(k.pt, np.float64), d.copy()) for k, d in zip(keys, desc)]

    def flow(self, gray, mask):
        if self.previous_gray is None or not self.tracks:
            return []
        lk = self.cfg.lk
        old = np.array([f.pixel for f in self.tracks], np.float32).reshape(-1, 1, 2)
        args = dict(winSize=(lk.window_size_px, lk.window_size_px), maxLevel=lk.max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, lk.max_iterations, lk.epsilon))
        new, success, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, old, None, **args)
        if new is None:
            return []
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, new, None, **args)
        if back is None:
            return []
        errors = np.linalg.norm(back.reshape(-1, 2)-old.reshape(-1, 2), axis=1)
        h, w = gray.shape
        result = []
        for f, p, s, r, error in zip(self.tracks, new.reshape(-1, 2), success.ravel(), reverse.ravel(), errors):
            if s and r and np.isfinite(p).all() and np.isfinite(error) and 0 <= p[0] < w and 0 <= p[1] < h and error <= lk.forward_backward_max_error_px and mask[int(p[1]), int(p[0])]:
                result.append(FeatureObservation(f.track_id, p.astype(np.float64), f.descriptor, f.landmark_id, float(error)))
        return result

    def update(self, gray, tags, index, force_detect=False):
        import time
        c = self.cfg
        self.detection_ms = self.tracking_ms = 0.0
        if not c.enabled:
            self.reset()
            return []
        mask = feature_mask(gray.shape, tags, c)
        started = time.perf_counter()
        tracks = self.flow(gray, mask) if c.lk.enabled else []
        self.tracking_ms = (time.perf_counter()-started)*1000
        if force_detect or not c.lk.enabled or index % c.redetect_every_n_frames == 0 or len(tracks) < c.redetect_below_tracks:
            started = time.perf_counter()
            detected = self.detect(gray, mask)
            previous = tracks if c.lk.enabled else self.tracks
            if previous and detected:
                pairs = match_descriptors(np.array([f.descriptor for f in detected]), np.array([f.descriptor for f in previous]), c)
            else:
                pairs = []
            associations = dict(pairs)
            reused = set()
            for i, f in enumerate(detected):
                if i in associations:
                    old = previous[associations[i]]
                    # Descriptor identity plus spatial agreement for LK; descriptors alone when LK off.
                    if not c.lk.enabled or np.linalg.norm(old.pixel-f.pixel) <= c.lk.forward_backward_max_error_px:
                        f.track_id, f.landmark_id = old.track_id, old.landmark_id
                        reused.add(old.track_id)
                if f.track_id < 0:
                    f.track_id = self.next_id
                    self.next_id += 1
            # Keep surviving mapped tracks preferentially; suppress redetection duplicates.
            retained = [f for f in tracks if f.track_id not in reused]
            if retained:
                detected = [f for f in detected if f.track_id in reused or min(np.linalg.norm(f.pixel-g.pixel) for g in retained) > c.lk.forward_backward_max_error_px]
            tracks = (retained + detected)[:c.max_features]
            self.detection_ms = (time.perf_counter()-started)*1000
        self.tracks = tracks
        self.previous_gray = gray.copy()
        return tracks
