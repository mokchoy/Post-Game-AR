"""Step 51: EMA vs 1 euro vs constant-velocity Kalman on logged raw poses (offline, no replay, no depth model).

usage (from Integration/ar): ../.venv/bin/python scripts/step51_filters.py [--out ../runs/step51-filters]
Reads the static and movement runs named below; writes results.json, report.md and four png plots. See Instruction.md Step 51."""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ar_projection.array import ArrayState, local_corners, world_corners
from ar_projection.config import load_config
from ar_projection.filters import KalmanPose, OneEuroPose, _join, _split, alpha, exp, log
from ar_projection.rendering import PoseSmoother

RUNS = Path(__file__).resolve().parent.parent.parent/"runs"
STATIC, MOVE = "20261027T033608-9db092df", "20261027T042028-839c66ed"
SG = {5: np.array([-3, 12, 17, 12, -3])/35., 7: np.array([-2, 3, 6, 7, 6, 3, -2])/21.}
TAUS = np.arange(-100, 401, 5)/1000.
EMA_W = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FMIN = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5]
BETA = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
QS = list(np.logspace(-4, 2, 13))
BETA_EXT, QS_EXT = [20, 50], [1e-8, 1e-7, 1e-6, 1e-5]       # beyond the specified grids: the best settings sat on their edges
ext = lambda p: p.get("q", 1) < 1e-4 or p.get("beta", 0) > 10
SR = [0.25, 0.5, 1, 2, 4]


class Cfg:                                          # filter parameters, same field names as config.Filter
    def __init__(self, **kw):
        self.__dict__.update(dict(measurement_weight=0.4, enabled=True, min_cutoff_hz=1.0, beta=0.0, d_cutoff_hz=1.0, scene_distance_m=2.0, q=1.0,
                                  r_scale=1.0, gate_chi2=22.46, measurement_sigma=None), **kw)


class EmaDt:                                        # dt-aware EMA, w = 1 - exp(-dt / tau)
    def __init__(self, tau):
        self.tau, self.c = tau, None

    def reset(self):
        self.c = None

    def update(self, T, t=None, covariance=None):
        cm, Rm = _split(T)
        if self.c is None:
            self.c, self.R, self.t = cm, Rm, t
            return T.copy()
        w = 1-np.exp(-(t-self.t)/self.tau) if self.tau > 0 else 1.0
        e = log(Rm @ self.R.T)
        self.c, self.R, self.t = self.c+w*(cm-self.c), exp(w*e) @ self.R, t
        return _join(self.c, self.R)


class Run:
    def __init__(self, name):
        self.name, self.cfg = name, load_config(RUNS/name/"config.json")
        rows = [json.loads(l) for l in open(RUNS/name/"frames.jsonl")]
        meta = json.load(open(RUNS/name/"metadata.json"))
        self.K = np.array(meta["calibration"]["K"], float)
        self.w, self.h = meta["calibration"]["size"]
        self.n = len(rows)
        self.t = np.array([r["source_time_s"] for r in rows])
        self.valid = np.array([r["raw_pose"]["valid"] for r in rows])
        self.src = [r["actual_pose_source"] for r in rows]
        self.T = np.full((self.n, 4, 4), np.nan)
        self.cov = [None]*self.n
        for k, r in enumerate(rows):
            if self.valid[k]:
                self.T[k] = r["raw_pose"]["T_CW"]
                self.cov[k] = r["raw_pose"].get("covariance")
        arr = [r["diagnostics"].get("array") for r in rows]
        self.array_static = all(a is None or (a["yaw_deg"] == 0 and a["scale"] == 1) for a in arr)
        prev_valid, prev_t = False, None
        self.reset = np.zeros(self.n, bool)         # the pipeline's reset rule, applied before the first update of a segment
        for k in range(self.n):
            gap = prev_t is not None and self.t[k]-prev_t > self.cfg.pose.max_prediction_gap_s
            self.reset[k] = self.valid[k] and (not prev_valid or gap)
            prev_valid, prev_t = self.valid[k], self.t[k]
        self.seg = np.cumsum(self.reset)*self.valid-(~self.valid)          # segment id per valid frame, -1 invalid
        W = world_corners(self.cfg.array, ArrayState(), self.cfg.panel.offset_y_m)
        self.names = list(W)
        self.X = np.concatenate(list(W.values()))
        loc = np.concatenate(list(local_corners(self.cfg.array).values()))
        self.centre = np.array([i//4 == self.names.index("game") for i in range(16)])
        self.outer = np.abs(loc[:, 0]) >= 0.9*np.abs(loc[:, 0]).max()
        c, _ = zip(*[_split(T) for T in self.T[self.valid]])
        self.d = float(np.median(np.linalg.norm(c, axis=1)))                # camera-to-tag distance
        self.raw = self.poses(self.T)
        self.uv_raw, z = self.project(*self.raw)
        ok = self.valid[:, None] & (z > 0.05) & (self.uv_raw[..., 0] > -200) & (self.uv_raw[..., 0] < self.w+200) \
            & (self.uv_raw[..., 1] > -200) & (self.uv_raw[..., 1] < self.h+200)
        self.M = ok
        self.rel = self.t-self.t[0]

    def poses(self, T):
        """(c, R) of T_WC for every frame, NaN where the pose is NaN."""
        c, R = np.full((self.n, 3), np.nan), np.full((self.n, 3, 3), np.nan)
        for k in np.flatnonzero(np.isfinite(T[:, 0, 0])):
            c[k], R[k] = _split(T[k])
        return c, R

    def project(self, c, R):
        Xc = np.einsum("nki,nij->nkj", self.X[None]-c[:, None, :], R)       # R_WC^T (X - c)
        z = Xc[..., 2]
        uv = Xc @ self.K.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return uv[..., :2]/uv[..., 2:3], z

    def filtered(self, factory):
        """Filtered poses for every frame (NaN when the raw pose is invalid), the gate flags and the NIS values."""
        sm, out, gated, nis = factory(), np.full((self.n, 4, 4), np.nan), np.zeros(self.n, bool), []
        for k in np.flatnonzero(self.valid):
            if self.reset[k]:
                sm.reset()
            out[k] = sm.update(self.T[k], self.t[k], self.cov[k] if self.cov[k] is None else np.array(self.cov[k]))
            gated[k] = getattr(sm, "gated", False)
            if getattr(sm, "nis", None) is not None:
                nis.append(sm.nis)
        return out, gated, nis


def windows(run, half_s):
    """Index slices [lo, hi) of the +-half_s window around every valid frame, inside its segment (time-based)."""
    lo, hi = np.zeros(run.n, int), np.zeros(run.n, int)
    for s in np.unique(run.seg[run.seg >= 0]):
        idx = np.flatnonzero(run.seg == s)
        tt = run.t[idx]
        lo[idx] = idx[0]+np.searchsorted(tt, tt-half_s, "left")
        hi[idx] = idx[0]+np.searchsorted(tt, tt+half_s, "right")
    return lo, hi


def median_reference(run, half_s=1.0):
    """Drift-removed reference: centred moving median over 2 s of the raw projected points, of c and of the rotation vectors
    Log(R_j R_k^T) about frame k. Returns (p_ref, c_ref, R_ref)."""
    c, R = run.raw
    lo, hi = windows(run, half_s)
    p = np.full((run.n, 16, 2), np.nan)
    cr, Rr = np.full((run.n, 3), np.nan), np.full((run.n, 3, 3), np.nan)
    for k in np.flatnonzero(run.valid):
        sl = slice(lo[k], hi[k])
        p[k] = np.median(run.uv_raw[sl], axis=0)
        cr[k] = np.median(c[sl], axis=0)
        Rr[k] = exp(np.median([log(R[j] @ R[k].T) for j in range(lo[k], hi[k])], axis=0)) @ R[k]
    return p, cr, Rr


def sg_reference(run, W):
    """Zero-phase Savitzky-Golay (order 2, W frames) pose reference inside each segment; frames within W//2 of a segment end are NaN."""
    c, R = run.raw
    h, w = W//2, SG[W]
    cr, Rr = np.full((run.n, 3), np.nan), np.full((run.n, 3, 3), np.nan)
    for k in np.flatnonzero(run.valid):
        if k-h < 0 or k+h >= run.n or not (run.seg[k-h:k+h+1] == run.seg[k]).all():
            continue
        cr[k] = w @ c[k-h:k+h+1]
        Rr[k] = exp(sum(wj*log(R[k+j-h] @ R[k].T) for j, wj in enumerate(w))) @ R[k]
    p, z = run.project(cr, Rr)
    p[~(z > 0.05)] = np.nan
    return p, cr, Rr


def speeds(run, p):
    """Median-over-points reference speed (px/s), central difference inside a segment; NaN elsewhere."""
    s = np.full(run.n, np.nan)
    for k in range(1, run.n-1):
        if run.seg[k-1] == run.seg[k] == run.seg[k+1] >= 0 and np.isfinite(p[k-1:k+2]).all():
            s[k] = np.median(np.linalg.norm(p[k+1]-p[k-1], axis=1)/(run.t[k+1]-run.t[k-1]))
    return s


class Motion:
    """Everything about the movement run that does not depend on the filter, for one Savitzky-Golay window."""

    def __init__(self, run, W):
        self.run, self.W = run, W
        self.p_ref, self.c_ref, self.R_ref = sg_reference(run, W)
        self.speed = speeds(run, self.p_ref)
        self.ok = np.isfinite(self.p_ref[:, 0, 0]) & run.M.any(axis=1)
        self.pm = run.M & self.ok[:, None]
        self.moving = self.ok & (self.speed > 30)
        self.still = self.ok & (self.speed <= 30)
        half = run.rel < run.rel[-1]/2
        self.half = {"tune": half, "test": ~half}
        self.ref_tau = self._ref_tau()
        self.stops = self._stops()
        self.switches = [k for k in range(1, run.n) if run.seg[k] == run.seg[k-1] >= 0 and run.src[k] != run.src[k-1]]

    def _ref_tau(self):
        """p_ref(t_k - tau) linearly interpolated inside the reference-valid part of the segment, for tau in TAUS;
        frames whose whole tau range is not covered are dropped (self.lagframes)."""
        run = self.run
        out = np.full((len(TAUS), run.n, 16, 2), np.nan)
        for s in np.unique(run.seg[run.seg >= 0]):
            idx = np.flatnonzero((run.seg == s) & self.ok)
            if len(idx) < 2:
                continue
            ts, P = run.t[idx], self.p_ref[idx]
            for i, tau in enumerate(TAUS):
                q = ts-tau
                good = (q >= ts[0]) & (q <= ts[-1])
                j = np.clip(np.searchsorted(ts, q)-1, 0, len(ts)-2)
                f = ((q-ts[j])/(ts[j+1]-ts[j]))[:, None, None]
                val = P[j]*(1-f)+P[j+1]*f
                out[i, idx[good]] = val[good]
        self.lagframes = np.isfinite(out[:, :, 0, 0]).all(axis=0)
        return out

    def _stops(self):
        """(frame, unit direction per point) for every drop of the reference speed from >60 to <15 px/s."""
        run, ev = self.run, []
        for k in range(3, run.n):
            if np.isfinite(self.speed[k]) and self.speed[k] < 15 and np.isfinite(self.speed[k-1]) and self.speed[k-1] >= 15 \
                    and np.nanmax(self.speed[k-3:k]) > 60:
                d = self.p_ref[k]-self.p_ref[k-3]
                n = np.linalg.norm(d, axis=1, keepdims=True)
                if np.isfinite(d).all() and (n > 1e-6).all():
                    ev.append((k, d/n))
        return ev

    def metrics(self, p_f):
        """Movement-run metrics of one filtered track, per half (tune/test)."""
        run, res = self.run, {}
        err = np.linalg.norm(p_f-self.p_ref, axis=2)
        sm = np.full_like(p_f, np.nan)                     # the track's own SG-5 smoothing
        for k in range(2, run.n-2):
            if (run.seg[k-2:k+3] == run.seg[k]).all() and run.seg[k] >= 0:
                sm[k] = np.tensordot(SG[5], p_f[k-2:k+3], axes=1)
        noise = np.linalg.norm(p_f-sm, axis=2)
        sq = ((p_f[None]-self.ref_tau)**2).sum(-1)                                  # (tau, n, 16)
        for name, half in self.half.items():
            mv = (self.moving & half)[:, None] & self.pm
            st = (self.still & half)[:, None] & self.pm
            lf = (self.moving & half & self.lagframes)[:, None] & self.pm
            e = {"E_dyn": float(np.sqrt(np.mean(err[mv]**2))) if mv.any() else np.nan,
                 "E_still": float(np.sqrt(np.mean(err[st]**2))) if st.any() else np.nan,
                 "noise": float(np.sqrt(np.nanmean(noise[mv]**2))) if mv.any() else np.nan}
            tot = np.array([np.nansum(sq[i][lf]) for i in range(len(TAUS))])
            i = int(np.argmin(tot))
            tau = TAUS[i]
            if 0 < i < len(TAUS)-1 and tot[i-1]-2*tot[i]+tot[i+1] > 0:
                tau += 0.5*(tot[i-1]-tot[i+1])/(tot[i-1]-2*tot[i]+tot[i+1])*(TAUS[1]-TAUS[0])
            e["lag_ms"] = float(tau*1000) if lf.any() else np.nan
            e["lag_at_edge"] = bool(i in (0, len(TAUS)-1))
            over = []
            for k, d in self.stops:
                js = [j for j in range(k, min(run.n, k+12)) if run.seg[j] == run.seg[k] and np.isfinite(p_f[j]).all()]
                if half[k] and js:
                    along = np.array([np.sum((p_f[j]-self.p_ref[j])*d, axis=1) for j in js])      # (frames, points)
                    over.append(float(np.mean(np.maximum(0, along.max(axis=0)))))
            e["overshoot_median"] = float(np.median(over)) if over else np.nan
            e["overshoot_max"] = float(np.max(over)) if over else np.nan
            e["n_stops"] = len(over)
            steps = []
            for k in self.switches:
                if not half[k]:
                    continue
                s = [np.max(np.linalg.norm(p_f[j+1]-p_f[j], axis=1)) for j in range(max(1, k-3), min(run.n-1, k+3))
                     if run.seg[j] == run.seg[j+1] >= 0]
                if s:
                    steps.append(max(s))
            e["switch_step"] = float(np.mean(steps)) if steps else np.nan
            e["n_switches"] = len(steps)
            res[name] = e
        return res


def static_metrics(run, ref, out):
    """Jitter about the drift-removed reference for filtered poses `out` (T_CW per frame)."""
    p_ref, c_ref, R_ref = ref
    c, R = run.poses(out)
    p, _ = run.project(c, R)
    v = run.M & np.isfinite(p[..., 0])
    err = np.linalg.norm(p-p_ref, axis=2)
    mean = np.array([np.nanmean(np.where(v[:, i, None], p[:, i], np.nan), axis=0) for i in range(16)])
    err0 = np.linalg.norm(p-mean[None], axis=2)
    ok = np.flatnonzero(run.valid)
    rot = np.degrees([np.linalg.norm(log(R[k] @ R_ref[k].T)) for k in ok])
    pos = np.linalg.norm(c[ok]-c_ref[ok], axis=1)*1000
    rms = lambda e, sub=slice(None): float(np.sqrt(np.mean(e[:, sub][v[:, sub]]**2)))
    return {"J": rms(err), "J_centre": rms(err, run.centre), "J_outer": rms(err, run.outer), "J_no_drift_removal": rms(err0),
            "rot_std_deg": float(rot.std()), "rot_rms_deg": float(np.sqrt(np.mean(rot**2))),
            "pos_std_mm": float(pos.std()), "pos_rms_mm": float(np.sqrt(np.mean(pos**2)))}


def front(points):
    """Indices of the non-dominated (lower-left) points of [(x, y)], sorted by x."""
    order = sorted((i for i, p in enumerate(points) if np.isfinite(p).all()), key=lambda i: points[i])
    out, best = [], np.inf
    for i in order:
        if points[i][1] < best:
            out.append(i)
            best = points[i][1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS/"step51-filters"))
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)
    S, M = Run(STATIC), Run(MOVE)
    print("array static in both runs:", S.array_static, M.array_static, "| scene distance d:", round(S.d, 3), round(M.d, 3))
    d_scene = float(np.mean([S.d, M.d]))
    ref_s = median_reference(S)
    mot = {5: Motion(M, 5), 7: Motion(M, 7)}

    # offline EMA reproduces the logged render pose (the acceptance check)
    logged = [json.loads(l) for l in open(RUNS/STATIC/"frames.jsonl")][:400]
    ema, _, _ = S.filtered(lambda: PoseSmoother(S.cfg.filter))
    diffs = []
    for k, r in enumerate(logged):
        if r["render_T_CW"] is not None and np.isfinite(ema[k, 0, 0]):
            a, b = _split(ema[k]), _split(np.array(r["render_T_CW"]))
            diffs.append(max(np.abs(log(a[1] @ b[1].T)).max(), np.abs(a[0]-b[0]).max()))
    repro = float(max(diffs))
    print("EMA 0.4 offline vs logged render_T_CW, max abs diff over", len(diffs), "frames:", repro)
    if repro > 1e-6:
        raise SystemExit("offline EMA does not reproduce the logged render pose; stop and report")

    # measured measurement noise: static raw pose minus its 2 s moving median
    c, R = S.raw
    _, c_ref, R_ref = ref_s
    ok = np.flatnonzero(S.valid)
    dev = np.array([np.r_[c[k]-c_ref[k], log(R[k] @ R_ref[k].T)] for k in ok])
    sigma = dev.std(axis=0)
    print("measured sigma (c m, rot rad):", sigma.round(6))

    def make(kind, **kw):
        if kind == "ema":
            return lambda: PoseSmoother(Cfg(measurement_weight=kw["w"]))
        if kind == "ema_dt":
            return lambda: EmaDt(kw["tau"])
        if kind == "one_euro":
            return lambda: OneEuroPose(Cfg(min_cutoff_hz=kw["fmin"], beta=kw["beta"], scene_distance_m=kw.get("d", d_scene)))
        return lambda: KalmanPose(Cfg(q=kw["q"], r_scale=kw.get("sr", 1.0), measurement_sigma=list(sigma), scene_distance_m=kw.get("d", d_scene)),
                                  use_covariance=kind == "kalman_cov")

    def evaluate(kind, **kw):
        f = make(kind, **kw)
        so, gs, nis = S.filtered(f)
        mo, gm, _ = M.filtered(f)
        p_f, z = M.project(*M.poses(mo))
        p_f[~(z > 0.05)] = np.nan
        row = {"kind": kind, "params": {k: float(v) for k, v in kw.items()}, "static": static_metrics(S, ref_s, so),
               "motion": {W: mot[W].metrics(p_f) for W in (5, 7)}, "gate_events_static": int(gs.sum()), "gate_events_move": int(gm.sum()),
               "mean_nis_static": float(np.mean(nis)) if nis else None}
        row["motion"] = {str(W): v for W, v in row["motion"].items()}
        return row, p_f, (so, mo)

    rows = {}
    def sweep(label, kind, grid):
        rows[label] = []
        for kw in grid:
            rows[label].append(evaluate(kind, **kw)[0])
        print(label, len(rows[label]), "settings done")

    sweep("EMA", "ema", [{"w": w} for w in EMA_W])
    dt_med = float(np.median(np.diff(M.t)))
    sweep("EMA-dt", "ema_dt", [{"tau": (-dt_med/np.log(1-w) if w < 1 else 0.0)} for w in EMA_W])
    sweep("1euro", "one_euro", [{"fmin": f, "beta": b} for f in FMIN for b in BETA+BETA_EXT])
    sweep("KF-fixed", "kalman", [{"q": q} for q in QS_EXT+QS])
    key = lambda r: (r["static"]["J"], r["motion"]["5"]["tune"]["E_dyn"])
    # equal-jitter q of KF-fixed decides where the NIS test for s_R is done
    ema04 = next(r for r in rows["EMA"] if r["params"]["w"] == 0.4)
    target_J = ema04["static"]["J"]
    f_fixed = front([key(r) for r in rows["KF-fixed"]])
    q_nis = next((rows["KF-fixed"][i]["params"]["q"] for i in reversed(f_fixed) if rows["KF-fixed"][i]["static"]["J"] <= target_J),
                 rows["KF-fixed"][f_fixed[0]]["params"]["q"])
    nis = {sr: evaluate("kalman_cov", q=q_nis, sr=sr)[0]["mean_nis_static"] for sr in [0.03, 0.06, 0.12]+SR}     # first three: outside the grid, for information
    sr_best = min(SR, key=lambda s: abs(nis[s]-6))                                # grid choice as specified
    print("NIS at q =", q_nis, nis, "-> s_R", sr_best)
    sweep("KF-cov", "kalman_cov", [{"q": q, "sr": sr_best} for q in QS_EXT+QS])

    # operating points (selection on the tuning half, reporting on the test half)
    scale = (ema04["static"]["J"], ema04["motion"]["5"]["tune"]["E_dyn"])
    lag_target = ema04["motion"]["5"]["tune"]["lag_ms"]
    chosen = {}
    for label in ("EMA", "1euro", "KF-fixed", "KF-cov"):
        r = rows[label]
        fr = front([key(x) for x in r])
        ej = [i for i in fr if r[i]["static"]["J"] <= target_J*(1+1e-9)]
        knee = min(fr, key=lambda i: np.hypot(key(r[i])[0]/scale[0], key(r[i])[1]/scale[1]))
        el = [i for i, x in enumerate(r) if x["motion"]["5"]["tune"]["lag_ms"] <= lag_target+1e-9]
        chosen[label] = {"front": fr, "equal_jitter": min(ej, key=lambda i: key(r[i])[1]) if ej else min(range(len(r)), key=lambda i: key(r[i])[0]),
                         "equal_jitter_reached": bool(ej), "knee": knee,
                         "equal_lag": min(el, key=lambda i: r[i]["static"]["J"]) if el else None}
    raw_row = next(r for r in rows["EMA"] if r["params"]["w"] == 1.0)

    # scene-distance sensitivity of the chosen equal-jitter settings
    sens = {}
    for label, kind in (("1euro", "one_euro"), ("KF-fixed", "kalman"), ("KF-cov", "kalman_cov")):
        i = chosen[label]["equal_jitter"]
        if i is None:
            continue
        for mult in (0.5, 1.0, 2.0):
            kw = dict(rows[label][i]["params"], d=d_scene*mult)
            r = evaluate(kind, **kw)[0]
            sens[f"{label} d x{mult}"] = {"J": r["static"]["J"], "E_dyn_test": r["motion"]["5"]["test"]["E_dyn"], "lag_test_ms": r["motion"]["5"]["test"]["lag_ms"]}

    def pick(label, which):
        i = chosen[label][which]
        return None if i is None else rows[label][i]
    results = {"scene_distance_m": d_scene, "measured_sigma": sigma.tolist(), "ema_reproduction_max_abs_diff": repro, "dt_median_s": dt_med,
               "split": "tune = first 50% of the movement run by time, test = second 50%; static run used whole", "target_J_px": target_J,
               "lag_target_ms": lag_target, "nis_by_s_R": {str(k): v for k, v in nis.items()}, "s_R": sr_best, "q_for_nis": q_nis,
               "rows": rows, "chosen": chosen, "sensitivity_d": sens}
    (out_dir/"results.json").write_text(json.dumps(results, indent=1, default=float))
    plots(out_dir, S, M, mot[5], ref_s, rows, chosen, make, d_scene, raw_row)
    write_report(out_dir, results, rows, chosen, raw_row, ema04, mot, M, S.d)
    print("wrote", out_dir)


def plots(out_dir, S, M, mo, ref_s, rows, chosen, make, d_scene, raw_row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"EMA": "tab:blue", "1euro": "tab:orange", "KF-fixed": "tab:green", "KF-cov": "tab:red"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, r in rows.items():
        if label not in colours:
            continue
        pts = np.array([[x["static"]["J"], x["motion"]["5"]["tune"]["E_dyn"]] for x in r])
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.35, color=colours[label])
        fr = chosen[label]["front"]
        ax.plot(pts[fr, 0], pts[fr, 1], "-o", ms=3, color=colours[label], label=label)
    e = next(x for x in rows["EMA"] if x["params"]["w"] == 0.4)
    ax.plot(e["static"]["J"], e["motion"]["5"]["tune"]["E_dyn"], "k*", ms=14, label="EMA w=0.4 (current)")
    ax.set_xlabel("static jitter J (px RMS)")
    ax.set_ylabel("movement error E_dyn, tuning half (px RMS)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir/"pareto.png", dpi=150)
    plt.close(fig)

    sel = {"raw": ("ema", {"w": 1.0}), "EMA 0.4": ("ema", {"w": 0.4})}
    for label, kind in (("1euro", "one_euro"), ("KF-fixed", "kalman"), ("KF-cov", "kalman_cov")):
        i = chosen[label]["equal_jitter"]
        if i is not None:
            sel[label] = (kind, rows[label][i]["params"])
    paths = {}
    for name, (kind, kw) in sel.items():
        f = make(kind, **kw)
        paths[name] = (S.filtered(f)[0], M.filtered(f)[0])
    pt = 2                                           # centre ("game") panel, top-right corner
    def track(run, out, sl):
        c, R = run.poses(out)
        p, _ = run.project(c, R)
        return p[sl, pt]
    # (2) movement segment with a stop; (3) static window
    stop = next((k for k, _ in mo.stops if mo.run.rel[k] > mo.run.rel[-1]/2), mo.stops[0][0] if mo.stops else mo.run.n//2)
    lo = int(np.searchsorted(M.rel, M.rel[stop]-3.0)); hi = int(np.searchsorted(M.rel, M.rel[stop]+2.0))
    s0 = int(np.searchsorted(S.rel, 60.0)); s1 = int(np.searchsorted(S.rel, 65.0))
    for fname, run, i, sl, ref in (("motion_segment.png", M, 1, slice(lo, hi), mo.p_ref), ("static_segment.png", S, 0, slice(s0, s1), ref_s[0])):
        fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
        tt = run.rel[sl]
        for a, comp in zip(axs, (0, 1)):
            a.plot(tt, track(run, paths["raw"][i], sl)[:, comp], "k.-", lw=.8, ms=3, label="raw")
            a.plot(tt, ref[sl, pt, comp], "--", color="gray", label="reference")
            for name in sel:
                if name != "raw":
                    a.plot(tt, track(run, paths[name][i], sl)[:, comp], lw=1.3, label=name)
            a.set_ylabel("xy"[comp]+" (px)")
            a.grid(alpha=.3)
        axs[1].set_xlabel("time (s)")
        axs[0].legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir/fname, dpi=150)
        plt.close(fig)
    # (4) bars at equal jitter
    labels = [n for n in sel if n != "raw"]
    stats = {"E_dyn": [], "lag_ms": [], "overshoot_median": [], "switch_step": []}
    for n in labels:
        row = evaluate_row(rows, chosen, n)
        for k in stats:
            stats[k].append(row["motion"]["5"]["test"][k])
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, (k, v) in zip(axs, stats.items()):
        a.bar(range(len(labels)), np.nan_to_num(v), color=[colours.get(l.replace("EMA 0.4", "EMA"), "tab:blue") for l in labels])
        a.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        a.set_title(k + (" (px)" if k != "lag_ms" else ""))
    fig.tight_layout()
    fig.savefig(out_dir/"equal_jitter_bars.png", dpi=150)
    plt.close(fig)


def evaluate_row(rows, chosen, name):
    if name == "EMA 0.4":
        return next(r for r in rows["EMA"] if r["params"]["w"] == 0.4)
    return rows[name][chosen[name]["equal_jitter"]]


def fmt(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x+0.0:.{nd}f}".replace("-0", "0") if abs(x) < 0.5*10**-nd else f"{x:.{nd}f}"


def line(name, r):
    s, m = r["static"], r["motion"]["5"]["test"]
    p = ", ".join(f"{k}={v:.4g}" for k, v in r["params"].items())+("*" if ext(r["params"]) else "")
    return (f"| {name} | {p} | {fmt(s['J'], 3)} | {fmt(s['rot_std_deg'], 3)} | {fmt(s['pos_std_mm'], 2)} | {fmt(m['E_dyn'])} | "
            f"{fmt(m['lag_ms'], 0)} | {fmt(m['noise'], 2)} | {fmt(m['overshoot_median'])} / {fmt(m['overshoot_max'])} | {fmt(m['switch_step'])} |")


def write_report(out_dir, res, rows, chosen, raw_row, ema04, mot, M, S_D=0.0):
    M_D = M.d
    head = ["| filter | settings | static J px | rot std deg | pos std mm | E_dyn px | lag ms | motion noise px | overshoot px med/max | switch step px |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    ej = head+[line("raw", raw_row), line("EMA 0.4", ema04)]
    el = list(head)+[line("raw", raw_row), line("EMA 0.4", ema04)]
    knee = list(head)
    for label in ("EMA", "1euro", "KF-fixed", "KF-cov"):
        i = chosen[label]["equal_jitter"]
        if i is not None:
            ej.append(line(label+(" (best at equal J)" if label == "EMA" else "" if chosen[label]["equal_jitter_reached"] else " (lowest J; target not reached)"), rows[label][i]))
        i = chosen[label]["equal_lag"]
        if i is not None:
            el.append(line(label, rows[label][i]))
        knee.append(line(label, rows[label][chosen[label]["knee"]]))
    w7 = ["| filter | settings | E_dyn px (W=5) | E_dyn px (W=7) | lag ms (W=5) | lag ms (W=7) |", "|---|---|---|---|---|---|"]
    for name in ("EMA 0.4", "EMA", "1euro", "KF-fixed", "KF-cov"):
        r = ema04 if name == "EMA 0.4" else (rows[name][chosen[name]["equal_jitter"]] if chosen[name]["equal_jitter"] is not None else None)
        if r:
            a, b = r["motion"]["5"]["test"], r["motion"]["7"]["test"]
            w7.append(f"| {name} | {', '.join(f'{k}={v:.4g}' for k, v in r['params'].items())} | {fmt(a['E_dyn'])} | {fmt(b['E_dyn'])} | {fmt(a['lag_ms'], 0)} | {fmt(b['lag_ms'], 0)} |")
    sens = ["| setting | static J px | E_dyn test px | lag test ms |", "|---|---|---|---|"]+[
        f"| {k} | {fmt(v['J'], 3)} | {fmt(v['E_dyn_test'])} | {fmt(v['lag_test_ms'], 0)} |" for k, v in res["sensitivity_d"].items()]
    dt_rows = [(a, b) for a, b in zip(rows["EMA"], rows["EMA-dt"])]
    worst = 0.0
    for a, b in dt_rows:
        for x, y in ((a["static"]["J"], b["static"]["J"]), (a["motion"]["5"]["test"]["E_dyn"], b["motion"]["5"]["test"]["E_dyn"]),
                     (a["motion"]["5"]["test"]["lag_ms"], b["motion"]["5"]["test"]["lag_ms"])):
            if np.isfinite(x) and np.isfinite(y) and abs(x) > 1e-9 and a["params"]["w"] < 1:
                worst = max(worst, abs(y-x)/abs(x))
    dtt = ["| w | tau s | J EMA | J EMA-dt | E_dyn EMA | E_dyn EMA-dt | lag EMA ms | lag EMA-dt ms |", "|---|---|---|---|---|---|---|---|"]
    for a, b in dt_rows:
        if a["params"]["w"] in (0.2, 0.4, 0.6):
            ta, tb = a["motion"]["5"]["test"], b["motion"]["5"]["test"]
            dtt.append(f"| {a['params']['w']} | {b['params']['tau']:.3f} | {fmt(a['static']['J'], 3)} | {fmt(b['static']['J'], 3)} | {fmt(ta['E_dyn'])} | {fmt(tb['E_dyn'])} | {fmt(ta['lag_ms'], 0)} | {fmt(tb['lag_ms'], 0)} |")
    ms = f"{M.n} frames, {int(M.valid.sum())} valid, {len(mot[5].switches)} source switches inside segments, {len(mot[5].stops)} stops, {int(mot[5].moving.sum())} moving frames, {int(mot[5].still.sum())} still"
    md = f"""# Step 51: pose-filter comparison (offline, logged raw poses)

Static run `{STATIC}`, movement run `{MOVE}` ({ms}). Array state yaw 0 / scale 1 in both runs (checked from `diagnostics.array`). Evaluation points: the 16 panel corners, projected with the run's K; scene distance d = {res['scene_distance_m']:.3f} m (mean of the two runs' median camera-to-tag distances, {S_D:.3f} and {M_D:.3f}).

- Acceptance check: offline `PoseSmoother` (EMA 0.4, the reset rule below) reproduces the logged `render_T_CW` with max abs difference {res['ema_reproduction_max_abs_diff']:.2e} (rad / m).
- Reset rule: first valid pose after an invalid frame, and dt > 0.2 s. Frames with no raw pose produce no output.
- Split: selection on the first 50% of the movement run (by time) plus the whole static run; movement numbers below are on the second 50%.
- Measured KF-fixed measurement noise (static run, raw minus 2 s moving median), sigma = {np.array(res['measured_sigma']).round(5).tolist()} (c in m, rotation vector in rad).
- KF-cov scale: mean NIS by s_R at q = {res['q_for_nis']:.3g}: {{{', '.join(f'{k}: {v:.2f}' for k, v in res['nis_by_s_R'].items())}}}; s_R = {res['s_R']} (closest to 6).
- dt-aware EMA (w = 1 - exp(-dt/tau)) vs fixed-weight EMA: largest relative difference on J, E_dyn or lag over the weight grid {worst*100:.1f}% ({'reported separately in results.json' if worst > 0.05 else 'below 5%, so not reported further'}). Median dt {res['dt_median_s']*1000:.0f} ms.
- Lag: delay tau minimising sum ||p_f(t) - p_ref(t - tau)||^2 over moving frames (reference speed > 30 px/s), tau in -100..400 ms, 5 ms steps, parabola refined; frames whose whole tau range lies outside the segment are dropped. Positive = the filtered point is late.

## dt-aware EMA against the fixed-weight EMA (test half)

""" + "\n".join(dtt)+f"""

Settings marked * lie outside the specified grids (KF q < 1e-4, 1 euro beta > 10); they were added because the best settings sat on the grid edges.
Test half of the movement run: {int(sum(1 for k in mot[5].switches if mot[5].half['test'][k]))} source switches and {sum(1 for k, _ in mot[5].stops if mot[5].half['test'][k])} stops, so the overshoot column rests on that many events.

## Equal jitter (static J <= EMA 0.4's {res['target_J_px']:.3f} px; best E_dyn on the tuning half)

""" + "\n".join(ej)+f"""

## Equal lag (lag <= EMA 0.4's tuning-half lag {res['lag_target_ms']:.0f} ms; lowest static J)

""" + "\n".join(el)+"""

## Knee of each front (both axes scaled by EMA 0.4's values)

""" + "\n".join(knee)+"""

## Savitzky-Golay window sensitivity (test half, equal-jitter settings)

""" + "\n".join(w7)+"""

## Scene-distance sensitivity (equal-jitter settings re-run with d x 0.5 / 1 / 2)

""" + "\n".join(sens)+"""

Plots: `pareto.png`, `motion_segment.png`, `static_segment.png`, `equal_jitter_bars.png`. Full numbers: `results.json`.
"""
    (out_dir/"report.md").write_text(md)


if __name__ == "__main__":
    main()
